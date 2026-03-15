from __future__ import annotations

import math

import pandas as pd
import structlog

from app.agents.base_agent import BaseAgent
from app.config import settings
from app.pipeline.state import PipelineState
from app.services.llm import call_llm_json
from app.services.storage import read_dataframe

logger = structlog.get_logger(__name__)

# Nombre de valeurs d'échantillon envoyées au LLM
_SAMPLE_SIZE = 5

_COLUMN_SYSTEM_PROMPT = (
    "Tu es un expert en data engineering. "
    "Analyse le profil d'une colonne et retourne UNIQUEMENT un objet JSON valide."
)

_GRAIN_SYSTEM_PROMPT = (
    "Tu es un expert en data engineering. "
    "Retourne UNIQUEMENT un objet JSON valide avec une seule clé 'grain'."
)


def _build_column_profile(series: pd.Series) -> dict:
    """Calcule le profil statistique d'une Series pandas.

    Ne transmet au LLM que des agrégats — jamais les données brutes.
    """
    n = len(series)
    null_count = int(series.isna().sum())
    null_pct = round(null_count / n, 4) if n > 0 else 0.0
    n_unique = int(series.nunique(dropna=True))

    # Échantillon de valeurs non-nulles (dédoublonnées)
    sample_vals = series.dropna().unique().tolist()[:_SAMPLE_SIZE]
    # Sérialiser les types non-JSON-natifs (Timestamp, numpy, ...)
    sample = [_safe_scalar(v) for v in sample_vals]

    profile: dict = {
        "dtype": str(series.dtype),
        "n_unique": n_unique,
        "null_pct": null_pct,
        "sample": sample,
        "min": None,
        "max": None,
        "mean": None,
    }

    # Statistiques numériques uniquement
    if pd.api.types.is_numeric_dtype(series):
        numeric = series.dropna()
        if len(numeric) > 0:
            profile["min"] = _safe_scalar(numeric.min())
            profile["max"] = _safe_scalar(numeric.max())
            profile["mean"] = round(float(numeric.mean()), 4)

    return profile


def _safe_scalar(v) -> float | int | str | bool | None:
    """Convertit les types numpy/pandas en scalaires Python natifs JSON-sérialisables."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if hasattr(v, "item"):  # numpy scalar
        return v.item()
    if hasattr(v, "isoformat"):  # datetime / Timestamp
        return v.isoformat()
    return v


def _build_column_prompt(col_name: str, profile: dict) -> str:
    lines = [
        "Analyse cette colonne de données :",
        f'- Nom technique : "{col_name}"',
        f"- Type Python   : {profile['dtype']}",
        f"- Valeurs uniques : {profile['n_unique']}",
        f"- % nulles : {profile['null_pct']:.1%}",
    ]
    if profile["min"] is not None:
        lines.append(
            f"- Min : {profile['min']},  Max : {profile['max']},  Moyenne : {profile['mean']}"
        )
    lines.append(f"- Échantillon de valeurs : {profile['sample']}")
    lines.append("")
    lines.append(
        "Retourne un JSON avec exactement ces champs :\n"
        "{\n"
        '  "semantic_name": "nom lisible en français",\n'
        '  "description": "description courte du contenu",\n'
        '  "type": "categorical|numeric|date|text|identifier",\n'
        '  "unit": "unité ou chaîne vide si sans unité",\n'
        '  "confidence": 0.0 à 1.0,\n'
        '  "is_key_candidate": true|false\n'
        "}"
    )
    return "\n".join(lines)


def _build_grain_prompt(col_summaries: list[str]) -> str:
    cols_str = "\n".join(f"  - {s}" for s in col_summaries)
    return (
        f"Voici les colonnes d'un fichier de données :\n{cols_str}\n\n"
        "En une phrase concise, qu'est-ce que représente UNE ligne dans ce fichier ?\n\n"
        'Retourne : {"grain": "..."}'
    )


class MetadataAgent(BaseAgent):
    """Agent 1 — Analyse les fichiers uploadés et construit le Data Dictionary.

    Étapes :
        A. Profil statistique de chaque colonne (sans données brutes).
        B. Inférence sémantique via LLM (semantic_name, type, unit, confidence…).
        C. Calcul avg_confidence par fichier.
        D. Inférence du grain du fichier via LLM.
        E. Déclenchement HITL si confidence < seuil.

    Input  : state["raw_data_refs"]
    Output : state["metadata"]
    Modèle : settings.litellm_cheap_model (gpt-4o-mini)
    HITL CP1 : confidence colonne < settings.hitl_metadata_confidence_threshold (0.85)
    """

    name = "metadata_agent"

    async def run(self, state: PipelineState) -> PipelineState:
        log = logger.bind(report_id=state.get("report_id"))
        files_meta: dict = {}
        trigger_hitl = False

        for ref in state["raw_data_refs"]:
            log.info("metadata_processing_file", ref=ref)
            df = await read_dataframe(ref)
            file_meta = await self._process_file(df, ref, log)
            files_meta[ref] = file_meta

            # Vérifier si le HITL doit être déclenché pour ce fichier
            for col_info in file_meta["columns"].values():
                if col_info["confidence"] < settings.hitl_metadata_confidence_threshold:
                    trigger_hitl = True
                    log.warning(
                        "metadata_low_confidence",
                        ref=ref,
                        col=col_info.get("semantic_name"),
                        confidence=col_info["confidence"],
                    )

        state["metadata"] = {"files": files_meta}

        if trigger_hitl:
            state["hitl_pending"] = True
            state["hitl_checkpoint"] = "cp1_metadata"
            log.info("metadata_hitl_triggered")

        return state

    async def _process_file(self, df: pd.DataFrame, ref: str, log) -> dict:
        """Traite un fichier et retourne ses métadonnées complètes."""
        columns_meta: dict = {}

        for col_name in df.columns:
            profile = _build_column_profile(df[col_name])
            prompt = _build_column_prompt(col_name, profile)

            llm_result = await call_llm_json(
                prompt=prompt,
                system=_COLUMN_SYSTEM_PROMPT,
                model=settings.litellm_cheap_model,
            )

            # Fusionner profil statistique + inférence LLM
            columns_meta[col_name] = {
                **profile,
                "semantic_name": llm_result.get("semantic_name", col_name),
                "description": llm_result.get("description", ""),
                "type": llm_result.get("type", "unknown"),
                "unit": llm_result.get("unit", ""),
                "confidence": float(llm_result.get("confidence", 0.0)),
                "is_key_candidate": bool(llm_result.get("is_key_candidate", False)),
            }

        # Calcul de la confiance moyenne du fichier
        confidences = [c["confidence"] for c in columns_meta.values()]
        avg_confidence = round(sum(confidences) / len(confidences), 4) if confidences else 0.0

        # Inférence du grain
        col_summaries = [
            f"{col}: {info['semantic_name']} ({info['type']})" for col, info in columns_meta.items()
        ]
        grain_result = await call_llm_json(
            prompt=_build_grain_prompt(col_summaries),
            system=_GRAIN_SYSTEM_PROMPT,
            model=settings.litellm_cheap_model,
        )
        grain = grain_result.get("grain", "")

        log.info(
            "metadata_file_processed",
            ref=ref,
            rows=len(df),
            cols=len(df.columns),
            avg_confidence=avg_confidence,
            grain=grain,
        )

        return {
            "row_count": len(df),
            "col_count": len(df.columns),
            "columns": columns_meta,
            "grain": grain,
            "avg_confidence": avg_confidence,
        }
