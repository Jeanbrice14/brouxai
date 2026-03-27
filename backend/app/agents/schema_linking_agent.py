from __future__ import annotations

import difflib
from pathlib import Path
from urllib.parse import urlparse

import pandas as pd
import structlog

from app.agents.base_agent import BaseAgent
from app.config import settings
from app.pipeline.state import PipelineState
from app.services.llm import call_llm_json
from app.services.storage import read_dataframe

logger = structlog.get_logger(__name__)

_FK_NAME_SIM_THRESHOLD = 0.50
_FK_COVERAGE_THRESHOLD = 0.70
_COMPOSITE_SCORE_THRESHOLD = 0.60
_TOP_K_CANDIDATES = 5
_SAMPLE_SIZE = 5


# ── Helpers ────────────────────────────────────────────────────────────────────


def _table_name(ref: str) -> str:
    """Extrait le nom lisible de la table depuis une référence S3.

    Ex: "s3://bucket/uploads/ventes.csv" → "ventes"
    """
    return Path(urlparse(ref).path).stem


def _is_datetime(dtype) -> bool:
    return "datetime" in str(dtype)


def _are_dtypes_compatible(dtype_a, dtype_b) -> bool:
    """Filtre rapide : exclut les paires clairement incompatibles (datetime vs non-datetime)."""
    return _is_datetime(dtype_a) == _is_datetime(dtype_b)


# ── Step 1 — classify_table ────────────────────────────────────────────────────


def classify_table(df: pd.DataFrame, metadata: dict) -> str:
    """Classify a table as 'fact', 'dimension', 'calendar', or 'unknown'.

    Uses a Power BI-style star schema heuristic based on row count,
    column count, numeric metrics, key candidates, and date columns.
    """
    n_rows = len(df)
    n_cols = len(df.columns)

    # Count metrics (numeric non-key columns)
    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    # A column is "key_candidate" if unique_ratio > 0.85
    key_candidates = [c for c in df.columns if df[c].nunique() / max(len(df[c].dropna()), 1) > 0.85]
    metrics = [c for c in numeric_cols if c not in key_candidates]

    # Detect date columns
    date_cols = [c for c in df.columns if "datetime" in str(df[c].dtype)]

    if date_cols and n_cols <= 4:
        return "calendar"
    if n_rows > 100 and len(metrics) >= 1 and len(key_candidates) >= 2:
        return "fact"
    # fallback fact: large table with metrics
    if n_rows > 100 and len(metrics) >= 1:
        return "fact"
    # dimension: petite table de référence sans métriques numériques
    # (la règle "exactement 1 key_candidate" est trop stricte car les attributs
    #  textuels uniques comme "nom" gonflent aussi key_candidates)
    if n_rows <= 100 and len(metrics) == 0:
        return "dimension"
    return "unknown"


# ── Step 2 — name_similarity ───────────────────────────────────────────────────


def name_similarity(col_a: str, col_b: str) -> float:
    """Similarity after stripping id/ref/code suffixes and underscores."""

    def normalize(name: str) -> str:
        n = name.lower()
        # remove suffixes
        for suffix in ("_id", "_ref", "_code", "_key", "_fk"):
            if n.endswith(suffix):
                n = n[: -len(suffix)]
                break
        # remove prefixes
        for prefix in ("id_", "ref_", "code_", "fk_"):
            if n.startswith(prefix):
                n = n[len(prefix) :]
                break
        # remove underscores
        n = n.replace("_", "")
        return n

    a = normalize(col_a)
    b = normalize(col_b)
    if not a or not b:
        return 0.0
    return difflib.SequenceMatcher(None, a, b).ratio()


# ── Step 3 — is_sequential_pk ─────────────────────────────────────────────────


def is_sequential_pk(series: pd.Series) -> bool:
    """True if series looks like an auto-increment PK.

    Criteria: integers, starts at 1, no gaps, unique_ratio > 0.95.
    """
    s = series.dropna()
    if len(s) == 0:
        return False
    unique_ratio = s.nunique() / len(s)
    if unique_ratio <= 0.95:
        return False
    try:
        nums = sorted(int(v) for v in s.unique())
    except (ValueError, TypeError):
        return False
    return len(nums) > 1 and nums[0] == 1 and nums[-1] == len(nums)


# ── Step 4 — is_valid_fk_cardinality ──────────────────────────────────────────


def is_valid_fk_cardinality(fact_series: pd.Series, dim_series: pd.Series) -> bool:
    """FK must have unique_ratio < 0.90; PK target must have unique_ratio > 0.85."""
    fact_ur = fact_series.nunique() / max(len(fact_series.dropna()), 1)
    dim_ur = dim_series.nunique() / max(len(dim_series.dropna()), 1)
    return fact_ur < 0.90 and dim_ur > 0.85


# ── Step 5 — _auto_generate_date_table ────────────────────────────────────────


def _auto_generate_date_table(df: pd.DataFrame, date_col_name: str) -> pd.DataFrame:
    """Generate a calendar table from a datetime column in the fact table."""
    dates = pd.to_datetime(df[date_col_name].dropna().unique())
    date_range = pd.date_range(dates.min(), dates.max(), freq="D")
    cal = pd.DataFrame({"date": date_range})
    cal["annee"] = cal["date"].dt.year
    cal["trimestre"] = cal["date"].dt.quarter
    cal["mois"] = cal["date"].dt.month
    cal["nom_mois"] = cal["date"].dt.strftime("%B")
    cal["num_semaine"] = cal["date"].dt.isocalendar().week.astype(int)
    cal["jour_semaine"] = cal["date"].dt.dayofweek
    cal["est_weekend"] = cal["jour_semaine"] >= 5
    return cal


# ── Step 6 — _detect_cycles ───────────────────────────────────────────────────


def _detect_cycles(relations: list[dict]) -> list[str]:
    """Returns list of warning strings for any cycle detected (manual DFS, no networkx)."""
    # Build adjacency: table_a -> table_b
    graph: dict[str, set[str]] = {}
    for rel in relations:
        src = rel["table_a"]
        dst = rel["table_b"]
        graph.setdefault(src, set()).add(dst)

    visited: set[str] = set()
    rec_stack: set[str] = set()
    warnings: list[str] = []

    def dfs(node: str, path: list[str]) -> None:
        visited.add(node)
        rec_stack.add(node)
        for neighbor in graph.get(node, set()):
            if neighbor not in visited:
                dfs(neighbor, path + [neighbor])
            elif neighbor in rec_stack:
                cycle = " → ".join(path + [neighbor])
                warnings.append(f"Relation circulaire détectée : {cycle} — relations ignorées")
        rec_stack.discard(node)

    for node in list(graph.keys()):
        if node not in visited:
            dfs(node, [node])

    return warnings


# ── Step 8 — _detect_star_relations ───────────────────────────────────────────


def _detect_star_relations(
    fact_ref: str,
    df_fact: pd.DataFrame,
    dim_ref: str,
    df_dim: pd.DataFrame,
) -> list[dict]:
    """Detect FK relations from fact table to dimension table.

    Applies 4 filters in order, logging each rejection.
    """
    candidates: list[dict] = []

    for fact_col in df_fact.columns:
        for dim_col in df_dim.columns:
            reject_reason: str | None = None

            # Filtre 1: dtype compatibility
            if not _are_dtypes_compatible(df_fact[fact_col].dtype, df_dim[dim_col].dtype):
                continue  # silent skip (too many pairs)

            # Filtre 2: sequential PK detection
            fact_is_seq = is_sequential_pk(df_fact[fact_col])
            dim_is_seq = is_sequential_pk(df_dim[dim_col])

            if fact_is_seq and dim_is_seq:
                reject_reason = "PK sequentielle des deux cotes (PK->PK non valide)"
            elif fact_is_seq:
                reject_reason = "fact_col est une PK sequentielle (ne peut pas etre FK)"

            if reject_reason:
                logger.debug(
                    "relation_rejected",
                    **{
                        "from": f"{_table_name(fact_ref)}.{fact_col}",
                        "to": f"{_table_name(dim_ref)}.{dim_col}",
                        "reason": reject_reason,
                    },
                )
                continue

            # Filtre 4: name similarity (calculé en premier pour permettre le bypass cardinalité)
            n_sim = name_similarity(fact_col, dim_col)
            if n_sim < _FK_NAME_SIM_THRESHOLD:
                reject_reason = (
                    f"Similarité nom insuffisante ({n_sim:.2f} < {_FK_NAME_SIM_THRESHOLD})"
                )
                logger.info(
                    "relation_rejected",
                    **{
                        "from": f"{_table_name(fact_ref)}.{fact_col}",
                        "to": f"{_table_name(dim_ref)}.{dim_col}",
                        "reason": reject_reason,
                    },
                )
                continue

            # Filtre 3: cardinalité — bypass pour attributs de dimension partagés (ex: region→region)
            # Les colonnes catégorielles partagées (même nom, haute couverture) ne sont pas des FK/PK
            is_shared_dim_attr = n_sim >= 0.95
            if not is_shared_dim_attr and not is_valid_fk_cardinality(df_fact[fact_col], df_dim[dim_col]):
                reject_reason = "Cardinalité invalide (fact_col quasi-unique ou dim_col non-unique)"
                logger.debug(
                    "relation_rejected",
                    **{
                        "from": f"{_table_name(fact_ref)}.{fact_col}",
                        "to": f"{_table_name(dim_ref)}.{dim_col}",
                        "reason": reject_reason,
                    },
                )
                continue

            # Coverage
            vals_fact = {str(v) for v in df_fact[fact_col].dropna().unique()}
            vals_dim = {str(v) for v in df_dim[dim_col].dropna().unique()}
            if len(vals_fact) < 2 or len(vals_dim) < 2:
                continue

            intersection = vals_fact & vals_dim
            coverage = len(intersection) / len(vals_fact)
            if coverage < _FK_COVERAGE_THRESHOLD:
                logger.debug(
                    "relation_rejected",
                    **{
                        "from": f"{_table_name(fact_ref)}.{fact_col}",
                        "to": f"{_table_name(dim_ref)}.{dim_col}",
                        "reason": f"Coverage insuffisant ({coverage:.2f} < {_FK_COVERAGE_THRESHOLD})",
                    },
                )
                continue

            orphan_rate = round(1.0 - coverage, 4)
            cardinality = "N:1" if len(vals_dim) >= len(vals_fact.intersection(vals_dim)) else "1:1"
            cardinality_score = 1.0 if cardinality == "N:1" else 0.5
            unique_ratio_fact = round(len(vals_fact) / max(len(df_fact[fact_col].dropna()), 1), 4)

            composite_score = round(coverage * 0.40 + n_sim * 0.40 + cardinality_score * 0.20, 4)
            if composite_score < _COMPOSITE_SCORE_THRESHOLD:
                logger.info(
                    "relation_rejected",
                    **{
                        "from": f"{_table_name(fact_ref)}.{fact_col}",
                        "to": f"{_table_name(dim_ref)}.{dim_col}",
                        "reason": f"Score composite insuffisant ({composite_score:.2f} < {_COMPOSITE_SCORE_THRESHOLD})",
                    },
                )
                continue

            candidates.append(
                {
                    "ref_a": fact_ref,
                    "table_a": _table_name(fact_ref),
                    "col_a": fact_col,
                    "ref_b": dim_ref,
                    "table_b": _table_name(dim_ref),
                    "col_b": dim_col,
                    "coverage": round(coverage, 4),
                    "orphan_rate": orphan_rate,
                    "cardinality": cardinality,
                    "name_sim": round(n_sim, 4),
                    "unique_ratio": unique_ratio_fact,
                    "composite_score": composite_score,
                    "score": composite_score,
                    "high_orphan_rate": orphan_rate > settings.hitl_orphan_rate_threshold,
                    "description": "",
                    "sample_a": sorted(vals_fact)[:_SAMPLE_SIZE],
                    "sample_b": sorted(vals_dim)[:_SAMPLE_SIZE],
                }
            )

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:_TOP_K_CANDIDATES]


# ── Step 9 — _check_temporal_granularity ──────────────────────────────────────


def _check_temporal_granularity(rel: dict, dfs: dict[str, pd.DataFrame]) -> None:
    """Detect mismatched temporal granularity and annotate the relation in-place."""
    fact_df = dfs.get(rel["ref_a"])
    dim_df = dfs.get(rel["ref_b"])
    if fact_df is None or dim_df is None:
        return

    fact_col = rel["col_a"]
    dim_col = rel["col_b"]

    if fact_col not in fact_df.columns or dim_col not in dim_df.columns:
        return

    try:
        fact_dates = pd.to_datetime(fact_df[fact_col], errors="coerce", format="mixed").dropna()
        dim_dates = pd.to_datetime(dim_df[dim_col], errors="coerce", format="mixed").dropna()
    except Exception:
        return

    if len(fact_dates) == 0 or len(dim_dates) == 0:
        return

    # Check if fact has day-level granularity and dim has month-level
    fact_has_days = (fact_dates.dt.day != 1).any()
    dim_day1_ratio = (dim_dates.dt.day == 1).mean() if len(dim_dates) > 0 else 0

    if fact_has_days and dim_day1_ratio > 0.9:
        rel["requires_aggregation"] = True
        rel["aggregation_hint"] = (
            f"Agréger {rel['table_a']} par mois avant jointure avec {rel['table_b']}"
        )


# ── Helper — _build_relation_prompt ───────────────────────────────────────────


def _build_relation_prompt(cand: dict) -> str:
    return (
        "Décris en une phrase la relation métier entre ces deux colonnes de données.\n\n"
        f'Table source : "{cand["table_a"]}" — colonne "{cand["col_a"]}" '
        f"(échantillon : {cand['sample_a']})\n"
        f'Table cible  : "{cand["table_b"]}" — colonne "{cand["col_b"]}" '
        f"(échantillon : {cand['sample_b']})\n"
        f"Couverture : {cand['coverage']:.0%} | Cardinalité : {cand['cardinality']} | "
        f"Score composite : {cand['composite_score']:.2f}\n\n"
        'Retourne : {"description": "..."}'
    )


# ── Agent ──────────────────────────────────────────────────────────────────────


class SchemaLinkingAgent(BaseAgent):
    """Agent 2 — Détecte les relations implicites entre plusieurs fichiers uploadés.

    Implémente un algorithme de détection de star schema style Power BI.

    Étapes :
        A. Cas fichier unique → schema vide, pas de HITL.
        B. Chargement des DataFrames.
        C. Classification fact / dimension / calendar / unknown.
        D. Identification des tables de faits et dimensions.
        E. Détection star schema (fact → dimension uniquement).
           - Auto-génération de DateTable pour les colonnes datetime.
        F. Détection de cycles (DFS manuel).
        G. Vérification granularité temporelle.
        H. Déduction du schema_type (star / unknown).
        I. Construction de state["schema"].
        J. HITL toujours déclenché sur multi-fichiers.

    Input  : state["raw_data_refs"] + state["metadata"]
    Output : state["schema"]
    Modèle : settings.litellm_cheap_model
    HITL CP2 : toujours sur multi-fichiers + flag high_orphan_rate si > 5%
    """

    name = "schema_linking_agent"

    async def run(self, state: PipelineState) -> PipelineState:
        refs = state["raw_data_refs"]
        files_meta = state.get("metadata", {}).get("files", {})
        log = logger.bind(report_id=state.get("report_id"), n_files=len(refs))

        # ── A : fichier unique ────────────────────────────────────────────────
        if len(refs) < 2:
            log.info("schema_single_file")
            state["schema"] = {
                "tables": {ref: files_meta.get(ref, {}) for ref in refs},
                "table_types": {},
                "relations": [],
                "auto_generated_tables": [],
                "warnings": [],
                "multi_table": False,
                "schema_type": "unknown",
            }
            return state

        # ── B : chargement DataFrames ─────────────────────────────────────────
        log.info("schema_multi_file_start")
        dfs: dict[str, pd.DataFrame] = {}
        for ref in refs:
            dfs[ref] = await read_dataframe(ref)

        # ── C : classifier les tables ─────────────────────────────────────────
        table_types: dict[str, str] = {}
        for ref in refs:
            tname = _table_name(ref)
            ttype = classify_table(dfs[ref], files_meta.get(ref, {}))
            table_types[tname] = ttype
            log.info("table_classified", table=tname, type=ttype)

        # ── D : identifier fact vs dimension ──────────────────────────────────
        fact_refs = [r for r in refs if table_types[_table_name(r)] == "fact"]
        dim_refs = [
            r for r in refs if table_types[_table_name(r)] in ("dimension", "calendar", "unknown")
        ]

        warnings: list[str] = []
        if not fact_refs:
            # Fallback : plus grande table = fact
            fact_refs = [max(refs, key=lambda r: len(dfs[r]))]
            dim_refs = [r for r in refs if r not in fact_refs]
            fallback_name = _table_name(fact_refs[0])
            log.warning("no_fact_table_fallback", fact=fallback_name)
            warnings.append(
                f"Aucune table de faits claire détectée — fallback sur table la plus grande : {fallback_name}"
            )

        # ── E : détection star schema (fact → dimension only) ─────────────────
        all_relations: list[dict] = []
        auto_generated_tables: list[str] = []

        for fact_ref in fact_refs:
            df_fact = dfs[fact_ref]

            # Auto-generate DateTable for date columns
            fact_meta_cols = files_meta.get(fact_ref, {}).get("columns", {})
            for col_name, _col_meta in fact_meta_cols.items():
                if col_name in df_fact.columns:
                    col_dtype = str(df_fact[col_name].dtype)
                    if "datetime" in col_dtype and "__date_table__" not in auto_generated_tables:
                        date_df = _auto_generate_date_table(df_fact, col_name)
                        date_ref = "__date_table__"
                        dfs[date_ref] = date_df
                        dim_refs.append(date_ref)
                        table_types["__date_table__"] = "calendar"
                        auto_generated_tables.append("__date_table__")
                        all_relations.append(
                            {
                                "ref_a": fact_ref,
                                "table_a": _table_name(fact_ref),
                                "col_a": col_name,
                                "ref_b": date_ref,
                                "table_b": "__date_table__",
                                "col_b": "date",
                                "coverage": 1.0,
                                "orphan_rate": 0.0,
                                "cardinality": "N:1",
                                "name_sim": 1.0,
                                "unique_ratio": round(
                                    df_fact[col_name].nunique()
                                    / max(len(df_fact[col_name].dropna()), 1),
                                    4,
                                ),
                                "composite_score": 1.0,
                                "score": 1.0,
                                "high_orphan_rate": False,
                                "description": "Relation vers table calendrier générée automatiquement",
                                "semantic": "Table calendrier générée automatiquement",
                                "confidence": 1.0,
                                "sample_a": [],
                                "sample_b": [],
                            }
                        )
                        log.info(
                            "date_table_generated",
                            fact=_table_name(fact_ref),
                            date_col=col_name,
                        )
                        break

            # Detect FK relations: fact → each dimension
            for dim_ref in dim_refs:
                if dim_ref == "__date_table__":
                    continue  # already handled
                df_dim = dfs[dim_ref]
                log.info(
                    "schema_checking_pair",
                    fact=_table_name(fact_ref),
                    dim=_table_name(dim_ref),
                )

                candidates = _detect_star_relations(fact_ref, df_fact, dim_ref, df_dim)

                for cand in candidates:
                    prompt = _build_relation_prompt(cand)
                    llm_result = await call_llm_json(
                        prompt=prompt,
                        system="Retourne UNIQUEMENT du JSON valide avec une clé 'description'.",
                        model=settings.litellm_cheap_model,
                    )
                    cand["description"] = llm_result.get("description", "")

                log.info(
                    "schema_pair_done",
                    fact=_table_name(fact_ref),
                    dim=_table_name(dim_ref),
                    n=len(candidates),
                )
                all_relations.extend(candidates)

        # ── E2 : attributs de dimension partagés (dim→dim, ex: clients.region ↔ budget.region) ──
        # Le scan fact→dim ne couvre pas les relations entre deux tables non-fact.
        # On détecte ici les colonnes de même nom avec haute couverture de valeurs.
        actual_dim_refs = [r for r in dim_refs if r != "__date_table__"]
        for i in range(len(actual_dim_refs)):
            for j in range(i + 1, len(actual_dim_refs)):
                ref_a, ref_b = actual_dim_refs[i], actual_dim_refs[j]
                df_a, df_b = dfs[ref_a], dfs[ref_b]
                shared_cols = set(df_a.columns) & set(df_b.columns)
                for col in shared_cols:
                    if is_sequential_pk(df_a[col]) or is_sequential_pk(df_b[col]):
                        continue
                    vals_a = {str(v) for v in df_a[col].dropna().unique()}
                    vals_b = {str(v) for v in df_b[col].dropna().unique()}
                    if len(vals_a) < 2 or len(vals_b) < 2:
                        continue
                    intersection = vals_a & vals_b
                    cov_a = len(intersection) / len(vals_a)
                    cov_b = len(intersection) / len(vals_b)
                    coverage = max(cov_a, cov_b)
                    if coverage < _FK_COVERAGE_THRESHOLD:
                        continue
                    orphan_rate = round(1.0 - min(cov_a, cov_b), 4)
                    composite_score = round(coverage * 0.60 + 1.0 * 0.40, 4)  # n_sim=1.0 (même col)
                    if composite_score < _COMPOSITE_SCORE_THRESHOLD:
                        continue
                    # Direction : table la plus grande → table la plus petite
                    if len(df_a) >= len(df_b):
                        src_ref, src_tbl = ref_a, _table_name(ref_a)
                        tgt_ref, tgt_tbl = ref_b, _table_name(ref_b)
                        unique_ratio = round(len(vals_a) / max(len(df_a[col].dropna()), 1), 4)
                    else:
                        src_ref, src_tbl = ref_b, _table_name(ref_b)
                        tgt_ref, tgt_tbl = ref_a, _table_name(ref_a)
                        unique_ratio = round(len(vals_b) / max(len(df_b[col].dropna()), 1), 4)
                    rel: dict = {
                        "ref_a": src_ref,
                        "table_a": src_tbl,
                        "col_a": col,
                        "ref_b": tgt_ref,
                        "table_b": tgt_tbl,
                        "col_b": col,
                        "coverage": round(coverage, 4),
                        "orphan_rate": orphan_rate,
                        "cardinality": "N:M",
                        "name_sim": 1.0,
                        "unique_ratio": unique_ratio,
                        "composite_score": composite_score,
                        "score": composite_score,
                        "high_orphan_rate": orphan_rate > settings.hitl_orphan_rate_threshold,
                        "description": "",
                        "sample_a": sorted(vals_a)[:_SAMPLE_SIZE],
                        "sample_b": sorted(vals_b)[:_SAMPLE_SIZE],
                    }
                    prompt = _build_relation_prompt(rel)
                    llm_result = await call_llm_json(
                        prompt=prompt,
                        system="Retourne UNIQUEMENT du JSON valide avec une clé 'description'.",
                        model=settings.litellm_cheap_model,
                    )
                    rel["description"] = llm_result.get("description", "")
                    all_relations.append(rel)
                    log.info(
                        "shared_dim_relation_found",
                        table_a=src_tbl,
                        col=col,
                        table_b=tgt_tbl,
                        coverage=round(coverage, 4),
                    )

        # ── F : détection cycles ──────────────────────────────────────────────
        cycle_warnings = _detect_cycles(all_relations)
        if cycle_warnings:
            warnings.extend(cycle_warnings)
            log.warning("cycles_detected", count=len(cycle_warnings))

        # ── G : check temporal granularity ────────────────────────────────────
        for rel in all_relations:
            _check_temporal_granularity(rel, dfs)

        # ── H : schema_type ───────────────────────────────────────────────────
        has_fact = bool(fact_refs)
        has_dim = bool(
            [r for r in refs if table_types.get(_table_name(r)) in ("dimension", "calendar")]
        )
        schema_type = "star" if (has_fact and has_dim and all_relations) else "unknown"

        # ── I : build state["schema"] ─────────────────────────────────────────
        tables = {ref: files_meta.get(ref, {}) for ref in refs}
        state["schema"] = {
            "tables": tables,
            "table_types": table_types,
            "relations": all_relations,
            "auto_generated_tables": auto_generated_tables,
            "warnings": warnings,
            "multi_table": True,
            "schema_type": schema_type,
        }

        # ── J : HITL ──────────────────────────────────────────────────────────
        state["hitl_pending"] = True
        state["hitl_checkpoint"] = "cp2_schema"

        high_orphan_count = sum(1 for r in all_relations if r.get("high_orphan_rate"))
        log.info(
            "schema_complete",
            n_relations=len(all_relations),
            schema_type=schema_type,
            high_orphan=high_orphan_count,
        )

        return state
