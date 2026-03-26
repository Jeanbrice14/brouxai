from __future__ import annotations

import difflib
from itertools import combinations
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

_FK_COVERAGE_THRESHOLD = 0.70
_FK_NAME_SIM_THRESHOLD = 0.55          # relevé de 0.50 → bloque "client" vs "vente" (0.545)
_FK_SAME_SUFFIX_ENTITY_MIN = 0.75      # défense en profondeur : entités _id/_ref doivent matcher
_FK_UNIQUE_RATIO_MAX = 0.90
_COMPOSITE_SCORE_THRESHOLD = 0.60
_TOP_K_CANDIDATES = 5
_SAMPLE_SIZE = 5


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


def _normalize_col_name(name: str) -> str:
    """Retire les préfixes/suffixes courants avant comparaison de similarité."""
    n = name.lower()
    for suffix in ("_id", "_ref", "_fk"):
        if n.endswith(suffix):
            return n[: -len(suffix)]
    for prefix in ("id_", "ref_", "fk_"):
        if n.startswith(prefix):
            return n[len(prefix):]
    return n


def _has_id_suffix(name: str) -> bool:
    """Retourne True si le nom a un suffixe/préfixe d'identifiant (_id, _ref, _fk…)."""
    n = name.lower()
    return (
        any(n.endswith(s) for s in ("_id", "_ref", "_fk"))
        or any(n.startswith(s) for s in ("id_", "ref_", "fk_"))
    )


def _name_similarity(col_a: str, col_b: str) -> float:
    """Similarité SequenceMatcher entre deux noms de colonnes normalisés."""
    a = _normalize_col_name(col_a)
    b = _normalize_col_name(col_b)
    return difflib.SequenceMatcher(None, a, b).ratio()


def _passes_name_filter(col_a: str, col_b: str, name_sim: float) -> bool:
    """Filtre 1 étendu — deux niveaux de contrôle :

    - Seuil général : name_sim ≥ _FK_NAME_SIM_THRESHOLD (0.55)
    - Règle même-suffixe : si les deux colonnes portent un suffixe ID/REF,
      les entités extraites doivent être similaires à ≥ _FK_SAME_SUFFIX_ENTITY_MIN (0.75).
      Bloque les faux positifs du type client_id → vente_id (entités différentes,
      même suffixe _id, sim ≈ 0.545 < 0.75).
    """
    if name_sim < _FK_NAME_SIM_THRESHOLD:
        return False
    if _has_id_suffix(col_a) and _has_id_suffix(col_b):
        if name_sim < _FK_SAME_SUFFIX_ENTITY_MIN:
            return False
    return True


def _is_sequential(vals: set[str]) -> bool:
    """Retourne True si les valeurs forment une séquence entière [N, N+1, …, N+k].

    Utilisé pour détecter les PKs auto-incrément et les exclure comme cibles de FK.
    """
    try:
        nums = sorted(int(v) for v in vals)
    except (ValueError, TypeError):
        return False
    return len(nums) > 1 and nums == list(range(nums[0], nums[0] + len(nums)))


def _detect_candidates(
    ref_a: str,
    df_a: pd.DataFrame,
    ref_b: str,
    df_b: pd.DataFrame,
) -> list[dict]:
    """Détecte les relations candidates entre deux DataFrames par analyse statistique.

    Filtres appliqués dans l'ordre :
    1. Compatibilité dtype (datetime vs non-datetime)
    2. Filtre nom  : similarité normalisée > _FK_NAME_SIM_THRESHOLD (0.50)
    3. Filtre PK séquentielle : col_b = [1,2,3..N] → exclue comme CIBLE de FK
    4. Filtre cardinalité : unique_ratio(col_a) < _FK_UNIQUE_RATIO_MAX (0.90)
    5. Filtre coverage  : coverage > _FK_COVERAGE_THRESHOLD (0.70)
    6. Score composite  : coverage×0.40 + name_sim×0.40 + cardinality×0.20 ≥ 0.60

    Les DataFrames bruts ne sont JAMAIS transmis au LLM — seulement des agrégats.
    """
    candidates: list[dict] = []

    for col_a in df_a.columns:
        for col_b in df_b.columns:
            if not _are_dtypes_compatible(df_a[col_a].dtype, df_b[col_b].dtype):
                continue

            vals_a = {str(v) for v in df_a[col_a].dropna().unique()}
            vals_b = {str(v) for v in df_b[col_b].dropna().unique()}

            if len(vals_a) < 2 or len(vals_b) < 2:
                continue

            # ── Filtre 1 : similarité nom (seuil général + règle même-suffixe) ─
            name_sim = _name_similarity(col_a, col_b)
            if not _passes_name_filter(col_a, col_b, name_sim):
                continue

            # ── Filtre 2 : PK séquentielle — exclure comme CIBLE ─────────────
            if _is_sequential(vals_b):
                continue

            # ── Filtre 3 : unique_ratio de la colonne FK (col_a) ─────────────
            total_a = len(df_a[col_a].dropna())
            unique_ratio = len(vals_a) / total_a if total_a > 0 else 1.0
            if unique_ratio >= _FK_UNIQUE_RATIO_MAX:
                continue

            # ── Filtre 4 (pré) : coverage ────────────────────────────────────
            intersection = vals_a & vals_b
            coverage = len(intersection) / len(vals_a)
            if coverage < _FK_COVERAGE_THRESHOLD:
                continue

            orphan_rate = round(1.0 - coverage, 4)
            # N:1 si la table cible (b) a autant ou plus de valeurs uniques
            cardinality = "N:1" if len(vals_b) >= len(vals_a) else "1:1"
            cardinality_score = 1.0 if cardinality == "N:1" else 0.5

            # ── Filtre 4 : score composite ────────────────────────────────────
            composite_score = round(
                coverage * 0.40 + name_sim * 0.40 + cardinality_score * 0.20, 4
            )
            if composite_score < _COMPOSITE_SCORE_THRESHOLD:
                continue

            candidates.append(
                {
                    "ref_a": ref_a,
                    "table_a": _table_name(ref_a),
                    "col_a": col_a,
                    "ref_b": ref_b,
                    "table_b": _table_name(ref_b),
                    "col_b": col_b,
                    "coverage": round(coverage, 4),
                    "orphan_rate": orphan_rate,
                    "cardinality": cardinality,
                    "name_sim": round(name_sim, 4),
                    "unique_ratio": round(unique_ratio, 4),
                    "composite_score": composite_score,
                    "score": composite_score,
                    "high_orphan_rate": orphan_rate > settings.hitl_orphan_rate_threshold,
                    "description": "",
                    # Petits échantillons uniquement — jamais les données brutes
                    "sample_a": sorted(vals_a)[:_SAMPLE_SIZE],
                    "sample_b": sorted(vals_b)[:_SAMPLE_SIZE],
                }
            )

    candidates.sort(key=lambda x: x["score"], reverse=True)
    return candidates[:_TOP_K_CANDIDATES]


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


class SchemaLinkingAgent(BaseAgent):
    """Agent 2 — Détecte les relations implicites entre plusieurs fichiers uploadés.

    Étapes :
        A. Cas fichier unique → schema vide, pas de HITL.
        B. Cas multi-fichiers → chargement des DataFrames.
        C. Détection statistique des candidats FK (4 filtres + score composite).
        D. Enrichissement sémantique via LLM (description métier).
        E. Construction de state["schema"].
        F. HITL toujours déclenché sur multi-fichiers.

    Filtres de détection (dans l'ordre) :
        1. Similarité nom normalisé > 0.50  (SequenceMatcher, sans _id/_ref)
        2. Cible non séquentielle           (exclut PKs auto-incrément)
        3. unique_ratio(FK) < 0.90          (exclut PKs côté source)
        4. Score composite ≥ 0.60           (coverage×0.40 + name_sim×0.40 + card×0.20)

    Input  : state["raw_data_refs"] + state["metadata"]
    Output : state["schema"]
    Modèle : settings.litellm_cheap_model
    HITL CP2 : toujours sur multi-fichiers + flag high_orphan_rate si > 5%
    """

    name = "schema_linking_agent"

    async def run(self, state: PipelineState) -> PipelineState:
        refs = state["raw_data_refs"]
        files_meta = state.get("metadata", {}).get("files", {})
        tables = {ref: files_meta.get(ref, {}) for ref in refs}
        log = logger.bind(report_id=state.get("report_id"), n_files=len(refs))

        # ── Étape A : fichier unique ─────────────────────────────────────────
        if len(refs) < 2:
            log.info("schema_single_file", ref=refs[0] if refs else "none")
            state["schema"] = {
                "tables": tables,
                "relations": [],
                "multi_table": False,
            }
            return state

        # ── Étape B : chargement des DataFrames ──────────────────────────────
        log.info("schema_multi_file_start")
        dfs: dict[str, pd.DataFrame] = {}
        for ref in refs:
            dfs[ref] = await read_dataframe(ref)

        # ── Étapes C + D : détection + enrichissement LLM ───────────────────
        all_relations: list[dict] = []

        for ref_a, ref_b in combinations(refs, 2):
            log.info("schema_checking_pair", table_a=_table_name(ref_a), table_b=_table_name(ref_b))
            candidates = _detect_candidates(ref_a, dfs[ref_a], ref_b, dfs[ref_b])

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
                table_a=_table_name(ref_a),
                table_b=_table_name(ref_b),
                n_relations=len(candidates),
            )
            all_relations.extend(candidates)

        # ── Étape E : construction du schema ─────────────────────────────────
        state["schema"] = {
            "tables": tables,
            "relations": all_relations,
            "multi_table": True,
        }

        # ── Étape F : HITL toujours déclenché sur multi-fichiers ─────────────
        state["hitl_pending"] = True
        state["hitl_checkpoint"] = "cp2_schema"

        high_orphan_count = sum(1 for r in all_relations if r.get("high_orphan_rate"))
        log.info(
            "schema_complete",
            n_relations=len(all_relations),
            high_orphan=high_orphan_count,
            hitl=True,
        )

        return state
