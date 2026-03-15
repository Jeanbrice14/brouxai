from __future__ import annotations

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


def _detect_candidates(
    ref_a: str,
    df_a: pd.DataFrame,
    ref_b: str,
    df_b: pd.DataFrame,
) -> list[dict]:
    """Détecte les relations candidates entre deux DataFrames par analyse statistique.

    Pour chaque paire (col_a, col_b) :
    - Vérifie la compatibilité des dtypes
    - Calcule le taux de couverture = |set_a ∩ set_b| / |set_a|
    - Garde les paires avec coverage > _FK_COVERAGE_THRESHOLD
    - Retourne les _TOP_K_CANDIDATES meilleurs par score (coverage)

    Les DataFrames bruts ne sont JAMAIS transmis au LLM — seulement des agrégats.
    """
    candidates: list[dict] = []

    for col_a in df_a.columns:
        for col_b in df_b.columns:
            if not _are_dtypes_compatible(df_a[col_a].dtype, df_b[col_b].dtype):
                continue

            # Conversion en str pour la robustesse face aux types mixtes (int vs object)
            vals_a = {str(v) for v in df_a[col_a].dropna().unique()}
            vals_b = {str(v) for v in df_b[col_b].dropna().unique()}

            if len(vals_a) < 2 or len(vals_b) < 2:
                continue

            intersection = vals_a & vals_b
            coverage = len(intersection) / len(vals_a)

            if coverage < _FK_COVERAGE_THRESHOLD:
                continue

            orphan_rate = round(1.0 - coverage, 4)
            # N:1 si col_b (côté PK) a autant ou plus de valeurs uniques que col_a (côté FK)
            cardinality = "N:1" if len(vals_b) >= len(vals_a) else "1:1"

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
                    "score": round(coverage, 4),
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
        f"Couverture : {cand['coverage']:.0%} | Cardinalité : {cand['cardinality']}\n\n"
        'Retourne : {"description": "..."}'
    )


class SchemaLinkingAgent(BaseAgent):
    """Agent 2 — Détecte les relations implicites entre plusieurs fichiers uploadés.

    Étapes :
        A. Cas fichier unique → schema vide, pas de HITL.
        B. Cas multi-fichiers → chargement des DataFrames.
        C. Détection statistique des candidats FK par paire de fichiers.
        D. Enrichissement sémantique via LLM (description métier).
        E. Construction de state["schema"].
        F. HITL toujours déclenché sur multi-fichiers.

    Input  : state["raw_data_refs"] + state["metadata"]
    Output : state["schema"]
    Modèle : settings.litellm_cheap_model (gpt-4o-mini)
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
