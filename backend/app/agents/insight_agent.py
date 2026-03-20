from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import urlparse

import structlog

from app.agents.base_agent import BaseAgent
from app.config import settings
from app.pipeline.state import PipelineState
from app.services.llm import call_llm_json

logger = structlog.get_logger(__name__)

_INSIGHT_SYSTEM_PROMPT = (
    "Tu es un data analyst senior expert en business intelligence. "
    "Tu analyses des données de manière factuelle et conservatrice. "
    "Tu ne poses jamais d'hypothèses non étayées par les données. "
    "Si une donnée est absente, tu le signales explicitement plutôt que d'inventer. "
    "Retourne UNIQUEMENT du JSON valide."
)

# Champs requis pour chaque insight
_REQUIRED_FIELDS = ("title", "description", "type", "confidence", "supporting_data", "impact")
_VALID_TYPES = ("trend", "anomaly", "comparison", "highlight")
_VALID_IMPACTS = ("high", "medium", "low")

# Seuil de filtrage des insights trop incertains
_MIN_CONFIDENCE = 0.30


def _table_name(ref: str) -> str:
    return Path(urlparse(ref).path).stem


def _build_aggregates_summary(aggregates: dict) -> str:
    """Résumé des agrégats limité à 3000 chars — données déjà agrégées, pas brutes."""
    lines: list[str] = []
    for key, rows in aggregates.items():
        if not isinstance(rows, list) or not rows:
            continue
        cols = list(rows[0].keys())
        sample = rows[:10]
        lines.append(f"Agrégat '{key}' : {len(rows)} lignes | colonnes : {cols}")
        for row in sample:
            lines.append(f"  {row}")
    summary = "\n".join(lines)
    return summary[:3000]


def _build_schema_context(state: PipelineState) -> str:
    """Résumé sémantique du schéma (noms + types) pour contextualiser l'analyse."""
    lines: list[str] = []
    files_meta = state.get("metadata", {}).get("files", {})
    for ref, meta in files_meta.items():
        table = _table_name(ref)
        lines.append(f"Table: {table}")
        for col, info in meta.get("columns", {}).items():
            semantic = info.get("semantic_name", col)
            col_type = info.get("type", "unknown")
            unit = info.get("unit", "")
            unit_str = f" ({unit})" if unit else ""
            lines.append(f"  - {col}: {semantic} [{col_type}]{unit_str}")
        grain = meta.get("grain", "")
        if grain:
            lines.append(f"  Grain: {grain}")
    return "\n".join(lines)


def _build_insights_prompt(user_prompt: str, schema_ctx: str, agg_summary: str) -> str:
    return (
        f"Demande utilisateur : {user_prompt}\n\n"
        f"Contexte du schéma :\n{schema_ctx}\n\n"
        f"Données agrégées :\n{agg_summary}\n\n"
        "Génère entre 3 et 5 insights analytiques basés UNIQUEMENT sur les données ci-dessus.\n"
        "Règles strictes :\n"
        "  - Ne jamais inventer de chiffres\n"
        "  - Si une donnée manque, l'indiquer dans 'supporting_data'\n"
        "  - 'confidence' reflète la certitude basée sur les données disponibles\n\n"
        'Retourne : {"insights": [{"title": "...", "description": "...", '
        '"type": "trend|anomaly|comparison|highlight", "confidence": 0.0-1.0, '
        '"supporting_data": "...", "impact": "high|medium|low"}]}'
    )


def _fill_defaults(insight: dict) -> dict:
    """Complète les champs manquants d'un insight avec des valeurs par défaut."""
    defaults = {
        "title": "Insight sans titre",
        "description": "",
        "type": "highlight",
        "confidence": 0.0,
        "supporting_data": "",
        "impact": "medium",
    }
    for field, default in defaults.items():
        if field not in insight or insight[field] is None:
            insight[field] = default
    # Normaliser les valeurs
    if insight["type"] not in _VALID_TYPES:
        insight["type"] = "highlight"
    if insight["impact"] not in _VALID_IMPACTS:
        insight["impact"] = "medium"
    insight["confidence"] = max(0.0, min(1.0, float(insight["confidence"])))
    return insight


def _has_large_variation(supporting_data: str) -> bool:
    """Détecte une variation > 20% dans le champ supporting_data."""
    matches = re.findall(r"(\d+(?:\.\d+)?)\s*%", str(supporting_data))
    return any(float(m) > 20 for m in matches)


class InsightAgent(BaseAgent):
    """Agent 4 — Analyse les agrégats et détecte tendances, anomalies, corrélations.

    Étapes :
        A. Construction du contexte analytique (agrégats + schéma, max ~4000 tokens).
        B. Appel LLM gpt-4o pour générer 3-5 insights structurés.
        C. Validation et complétion des champs + filtrage confidence < 0.30.
        D. HITL si confidence < seuil, anomalie détectée, ou variation > 20%.
        E. Écriture dans state["insights"].

    Input  : state["aggregates"] + state["metadata"]
    Output : state["insights"]
    Modèle : settings.litellm_default_model (gpt-4o)
    HITL CP3 : confidence < 0.80 | type "anomaly" | variation > 20%
    """

    name = "insight_agent"

    async def run(self, state: PipelineState) -> PipelineState:
        log = logger.bind(report_id=state.get("report_id"))

        # ── Étape A : construction du contexte ──────────────────────────────
        agg_summary = _build_aggregates_summary(state.get("aggregates", {}))
        schema_ctx = _build_schema_context(state)
        prompt = _build_insights_prompt(state["prompt"], schema_ctx, agg_summary)

        # ── Étape B : appel LLM gpt-4o ──────────────────────────────────────
        llm_result = await call_llm_json(
            prompt=prompt,
            system=_INSIGHT_SYSTEM_PROMPT,
            model=settings.litellm_default_model,
        )

        raw_insights: list[dict] = llm_result.get("insights", [])
        if not isinstance(raw_insights, list):
            raw_insights = []

        # ── Étape C : validation + filtrage ─────────────────────────────────
        validated: list[dict] = []
        for item in raw_insights:
            if not isinstance(item, dict):
                continue
            item = _fill_defaults(item)
            if item["confidence"] < _MIN_CONFIDENCE:
                log.info(
                    "insight_filtered_low_confidence",
                    title=item["title"],
                    confidence=item["confidence"],
                )
                continue
            validated.append(item)

        log.info("insights_generated", total=len(raw_insights), kept=len(validated))

        # ── Étape D : HITL ───────────────────────────────────────────────────
        trigger_hitl = False
        for insight in validated:
            if insight["confidence"] < settings.hitl_confidence_threshold:
                trigger_hitl = True
                log.warning("insight_hitl_low_confidence", confidence=insight["confidence"])
            if insight["type"] == "anomaly":
                trigger_hitl = True
                log.warning("insight_hitl_anomaly", title=insight["title"])
            if _has_large_variation(insight.get("supporting_data", "")):
                trigger_hitl = True
                log.warning("insight_hitl_large_variation", title=insight["title"])

        if trigger_hitl:
            state["hitl_pending"] = True
            state["hitl_checkpoint"] = "cp3_insights"

        # ── Étape E ──────────────────────────────────────────────────────────
        state["insights"] = validated
        return state
