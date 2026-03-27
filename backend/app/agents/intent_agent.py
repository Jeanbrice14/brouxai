from __future__ import annotations

import structlog

from app.agents.base_agent import BaseAgent
from app.config import settings
from app.pipeline.state import PipelineState
from app.services.llm import call_llm_json

logger = structlog.get_logger(__name__)

SIMPLE_QUERY_KEYWORDS = [
    "top",
    "flop",
    "liste",
    "donne moi",
    "quels sont",
    "quel est",
    "combien",
    "meilleur",
    "pire",
    "classement",
    "ranking",
    "nombre",
    "total",
    "somme",
    "moyenne",
    "max",
    "min",
    "qui a",
    "lequel",
]

CHART_REQUEST_KEYWORDS = [
    "évolution",
    "tendance",
    "progression",
    "compare",
    "comparaison",
    "variation",
    "par mois",
    "par région",
    "par semaine",
    "courbe",
    "graphique",
    "chart",
    "visualise",
    "montre",
    "affiche",
    "répartition",
    "distribution",
    "sur l'année",
    "mensuel",
    "trimestriel",
    "historique",
]

FULL_REPORT_KEYWORDS = [
    "analyse complète",
    "rapport",
    "storytelling",
    "analyse approfondie",
    "explique",
    "contexte",
    "recommandations",
    "insights",
    "synthèse",
    "bilan",
    "performance globale",
    "vue d'ensemble",
    "diagnostic",
]

_INTENT_TO_RESPONSE_TYPE = {
    "simple_query": "table",
    "chart_request": "chart",
    "full_report": "report",
}


def _count_keyword_matches(text: str, keywords: list[str]) -> int:
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw in text_lower)


class IntentAgent(BaseAgent):
    """Agent 0 — Détecte l'intention du prompt utilisateur.

    Étape A : Détection par mots-clés (0 coût LLM).
    Étape B : Fallback LLM si aucun keyword matché.
    Étape C : Mapping intent → response_type.

    Input  : state["prompt"]
    Output : state["intent"], state["intent_confidence"], state["response_type"]
    """

    name = "intent_agent"

    async def run(self, state: PipelineState) -> PipelineState:
        log = logger.bind(report_id=state.get("report_id"))
        prompt = state.get("prompt", "")

        # ── Étape A : détection par mots-clés ───────────────────────────────
        scores = {
            "simple_query": _count_keyword_matches(prompt, SIMPLE_QUERY_KEYWORDS),
            "chart_request": _count_keyword_matches(prompt, CHART_REQUEST_KEYWORDS),
            "full_report": _count_keyword_matches(prompt, FULL_REPORT_KEYWORDS),
        }

        total_matches = sum(scores.values())

        if total_matches > 0:
            # La catégorie avec le plus de matches gagne
            # En cas d'égalité → simple_query (plus rapide)
            max_score = max(scores.values())
            if scores["simple_query"] == max_score:
                intent = "simple_query"
            elif scores["chart_request"] == max_score:
                intent = "chart_request"
            else:
                intent = "full_report"
            confidence = min(1.0, max_score / max(total_matches, 1) + 0.5)
            log.info("intent_detected_keywords", intent=intent, scores=scores)

        else:
            # ── Étape B : fallback LLM ──────────────────────────────────────
            log.info("intent_fallback_llm", prompt_len=len(prompt))
            try:
                result = await call_llm_json(
                    prompt=(
                        f"Classifie ce prompt data : '{prompt}'\n"
                        "Réponds UNIQUEMENT avec ce JSON :\n"
                        '{{"intent": "simple_query|chart_request|full_report", '
                        '"confidence": 0.0-1.0, '
                        '"response_type": "table|chart|report"}}'
                    ),
                    system="Retourne UNIQUEMENT du JSON valide.",
                    model=settings.litellm_cheap_model,
                )
                intent = result.get("intent", "full_report")
                confidence = float(result.get("confidence", 0.7))
                if intent not in _INTENT_TO_RESPONSE_TYPE:
                    intent = "full_report"
            except Exception:
                # Fallback silencieux : full_report sans bloquer le pipeline
                intent = "full_report"
                confidence = 0.5
                log.warning("intent_llm_fallback_failed", fallback="full_report")

        # ── Étape C : mapping intent → response_type ────────────────────────
        response_type = _INTENT_TO_RESPONSE_TYPE[intent]

        log.info(
            "intent_complete",
            intent=intent,
            response_type=response_type,
            confidence=round(confidence, 2),
        )

        state["intent"] = intent
        state["intent_confidence"] = round(confidence, 2)
        state["response_type"] = response_type
        return state
