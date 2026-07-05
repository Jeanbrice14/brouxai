from __future__ import annotations

import re
import unicodedata

import structlog

from app.agents.base_agent import BaseAgent
from app.config import settings
from app.pipeline.state import PipelineState
from app.services.conversation_memory import format_history_for_prompt
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

# Répartition par une dimension quelconque ("par catégorie", "par produit", "par client"...).
# Les mots-clés CHART_REQUEST_KEYWORDS ci-dessus ne couvrent que "par mois"/"par région"/
# "par semaine" en dur — toute autre dimension (ex: "ventes par catégorie") ratait le match
# et tombait en simple_query/table via le fallback LLM. Un "par X" dans une question data
# signifie quasi toujours une agrégation groupée, qui se prête mieux à un graphique qu'à
# une table brute.
_BREAKDOWN_BY_DIMENSION_PATTERN = re.compile(r"\bpar\s+\w+", re.IGNORECASE)

# Exception : "moyenne ... par X" est presque toujours un ratio par unité ("moyenne du
# panier par commande" = CA total / nb commandes, un seul chiffre), pas une répartition par
# dimension — contrairement à "ventes par catégorie". Trouvé via la matrice de tests de
# prompts (cf. test_intent_agent.py) : sans cette exception, "quelle est la moyenne du
# panier par commande ?" matchait le pattern ci-dessus et basculait à tort en chart_request.
_RATE_KEYWORDS = ("moyenne",)

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


def _fold_accents(text: str) -> str:
    """Retire les accents (é→e, à→a...) pour un matching de mots-clés insensible aux
    accents. Régression réelle : "quelle est l'evolution des commandes en 2022" (sans
    accent sur "evolution") ne matchait aucun mot-clé — "évolution" dans
    CHART_REQUEST_KEYWORDS est accentué — et tombait sur le fallback LLM, non déterministe,
    qui a classé la question en simple_query/table. Beaucoup d'utilisateurs tapent sans
    accents (clavier, habitude, mobile) ; plusieurs mots-clés dans les 3 listes le sont
    ("évolution", "par région", "répartition", "synthèse"...)."""
    nfkd = unicodedata.normalize("NFKD", text)
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def _count_keyword_matches(text: str, keywords: list[str]) -> int:
    text_folded = _fold_accents(text.lower())
    return sum(1 for kw in keywords if _fold_accents(kw) in text_folded)


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
        prompt_folded = _fold_accents(prompt.lower())
        is_rate_question = any(_fold_accents(kw) in prompt_folded for kw in _RATE_KEYWORDS)
        breakdown_bonus = (
            1 if not is_rate_question and _BREAKDOWN_BY_DIMENSION_PATTERN.search(prompt) else 0
        )
        scores = {
            "simple_query": _count_keyword_matches(prompt, SIMPLE_QUERY_KEYWORDS),
            "chart_request": _count_keyword_matches(prompt, CHART_REQUEST_KEYWORDS) + breakdown_bonus,
            "full_report": _count_keyword_matches(prompt, FULL_REPORT_KEYWORDS),
        }

        total_matches = sum(scores.values())

        if total_matches > 0:
            # La catégorie avec le plus de matches gagne. En cas d'égalité → chart_request
            # d'abord : les mots-clés simple_query ("nombre", "total", "combien"...) décrivent
            # surtout QUELLE mesure est demandée et apparaissent aussi couramment dans des
            # questions d'évolution/répartition (ex: "l'évolution du nombre de commandes" →
            # "nombre" + "évolution" à égalité) — alors qu'un mot-clé chart_request ("évolution",
            # "tendance", "répartition"...) signale sans ambiguïté un besoin de visualisation.
            # Régression réelle : cette question tombait en simple_query/table avant ce correctif.
            max_score = max(scores.values())
            if scores["chart_request"] == max_score:
                intent = "chart_request"
            elif scores["simple_query"] == max_score:
                intent = "simple_query"
            else:
                intent = "full_report"
            confidence = min(1.0, max_score / max(total_matches, 1) + 0.5)
            log.info("intent_detected_keywords", intent=intent, scores=scores)

        else:
            # ── Étape B : fallback LLM ──────────────────────────────────────
            log.info("intent_fallback_llm", prompt_len=len(prompt))
            history_block = format_history_for_prompt(state.get("chat_history") or [])
            history_section = (
                f"Questions précédentes de cette conversation (pour comprendre une question "
                f"de suivi courte comme \"et pour le mois dernier ?\") :\n{history_block}\n\n"
                if history_block
                else ""
            )
            try:
                result = await call_llm_json(
                    prompt=(
                        f"{history_section}"
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
