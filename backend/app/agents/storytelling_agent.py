from __future__ import annotations

import re

import structlog

from app.agents.base_agent import BaseAgent
from app.config import settings
from app.pipeline.state import PipelineState
from app.services.llm import call_llm, call_llm_json

logger = structlog.get_logger(__name__)

_TONE_INSTRUCTIONS = {
    "formel": "Utilise un registre formel et professionnel, adapté à un rapport de direction.",
    "neutre": "Utilise un registre neutre et factuel, accessible à tout public.",
    "synthétique": "Utilise un registre concis et synthétique, va à l'essentiel.",
}

# ── System prompts ─────────────────────────────────────────────────────────────

_NARRATIVE_JSON_SYSTEM = (
    "Tu es un expert en communication data-driven et storytelling analytique. "
    "Tu produis UNIQUEMENT du JSON valide, sans markdown ni commentaires. "
    "Tu te bases UNIQUEMENT sur les insights fournis. "
    "Tu n'inventes JAMAIS de chiffres ni d'informations non présentes dans les insights."
)


def _format_insights_for_prompt(insights: list[dict]) -> str:
    lines: list[str] = []
    for i, ins in enumerate(insights, 1):
        lines.append(
            f"{i}. {ins.get('title', 'Insight')} [confiance: {ins.get('confidence', 0):.0%}]"
        )
        lines.append(f"   {ins.get('description', '')}")
        lines.append(f"   Données sources: {ins.get('supporting_data', '')}")
        lines.append(f"   Impact: {ins.get('impact', 'medium')}")
    return "\n".join(lines)


def _build_report_json_prompt(
    user_prompt: str,
    insights: list[dict],
    tone: str,
    language: str,
    template: str | None = None,
) -> str:
    tone_instruction = _TONE_INSTRUCTIONS.get(tone, _TONE_INSTRUCTIONS["formel"])
    formatted_insights = _format_insights_for_prompt(insights)

    base = (
        f"Demande originale : {user_prompt}\n\n"
        f"Langue : {language} | Ton : {tone}\n"
        f"{tone_instruction}\n\n"
        f"Insights disponibles :\n{formatted_insights}\n\n"
        "Produis un JSON avec ce format EXACT :\n"
        "{\n"
        '  "executive_summary": "2 phrases MAX, texte pur sans titre ni label. '
        "Commence directement par les chiffres clés. "
        "Exemple : 736 107 € de CA total sur 198 ventes. "
        'Le segment PME domine à 279 256 € devant Grand Compte à 175 396 €.",\n'
        '  "recommendations": [\n'
        '    "Action concrète et actionnable 1 (verbe + qui + quoi)",\n'
        '    "Action concrète et actionnable 2 (verbe + qui + quoi)"\n'
        "  ]\n"
        "}\n\n"
        "Règles strictes :\n"
        "  - executive_summary : 2 phrases MAX, KPIs chiffrés obligatoires\n"
        "  - Ne JAMAIS commencer par un titre, un label ou 'RÉSUMÉ EXÉCUTIF'\n"
        "  - Ne JAMAIS inclure les mots 'RÉSUMÉ', 'INSIGHTS', 'RECOMMANDATIONS'\n"
        "  - Commencer directement par le chiffre ou le fait le plus important\n"
        "  - Arrondir les nombres à l'entier (pas de décimales dans le résumé)\n"
        "  - recommendations : 1 à 2 items MAXIMUM, concrets et actionnables, jamais vagues\n"
        "  - Basé UNIQUEMENT sur les insights fournis\n"
        "  - Ne pas inventer de chiffres\n"
    )
    if template:
        base += f"\nTemplate narratif à respecter :\n{template}\n"
    return base


def _strip_markdown(text: str) -> str:
    text = re.sub(r"\*{3}(.+?)\*{3}", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"\*{2}(.+?)\*{2}", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"\*(.+?)\*", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"`(.+?)`", r"\1", text)
    return text.strip()


def _word_count(text: str) -> int:
    return len(text.split())


class StorytellingAgent(BaseAgent):
    """Agent 5 — Construit la narration data-driven structurée.

    3 modes selon state["response_type"] :
    - "table"  : UNE seule phrase de synthèse (max 20 mots) — call_llm
    - "chart"  : 2-3 phrases (tendance principale + point notable) — call_llm
    - "report" : JSON {executive_summary, recommendations} — call_llm_json

    Input  : state["insights"] + state["brand_kit"] + state["response_type"]
    Output : state["narrative"] (résumé exécutif) + state["recommendations"]
    """

    name = "storytelling_agent"

    async def run(self, state: PipelineState) -> PipelineState:
        log = logger.bind(report_id=state.get("report_id"))
        brand_kit = state.get("brand_kit", {})
        response_type = state.get("response_type", "report")

        tone = brand_kit.get("tone", "formel")
        language = brand_kit.get("language", "fr")
        template = brand_kit.get("narrative_template")
        insights = state.get("insights", [])

        # ── Mode "table" : 1 seule phrase ────────────────────────────────────
        if response_type == "table":
            prompt_text = (
                f"Données : {state.get('prompt', '')}\n"
                f"Insights : {_format_insights_for_prompt(insights[:2])}\n\n"
                "Rédige UNE SEULE phrase de synthèse (maximum 20 mots). "
                "Commence directement par la phrase, sans titre ni liste."
            )
            narrative = await call_llm(
                prompt=prompt_text,
                system="Tu rédiges UNE seule phrase factuelle et concise. Texte pur, sans Markdown.",
                model=settings.litellm_cheap_model,
                temperature=0.2,
            )
            narrative = _strip_markdown(narrative)
            sentences = [s.strip() for s in narrative.replace("\n", " ").split(".") if s.strip()]
            narrative = (sentences[0] + ".") if sentences else narrative[:120]
            log.info("storytelling_complete", mode="table", words=_word_count(narrative))
            state["narrative"] = narrative
            return state

        # ── Mode "chart" : 2-3 phrases ───────────────────────────────────────
        if response_type == "chart":
            prompt_text = (
                f"Données : {state.get('prompt', '')}\n"
                f"Insights : {_format_insights_for_prompt(insights[:3])}\n\n"
                "Rédige 2 à 3 phrases maximum décrivant la tendance principale "
                "et un point notable. Commence directement, sans titre ni liste."
            )
            narrative = await call_llm(
                prompt=prompt_text,
                system="Tu rédiges 2-3 phrases factuelles et concises. Texte pur, sans Markdown.",
                model=settings.litellm_cheap_model,
                temperature=0.2,
            )
            narrative = _strip_markdown(narrative)
            sentences = [s.strip() for s in narrative.replace("\n", " ").split(".") if s.strip()]
            narrative = ". ".join(sentences[:3]) + ("." if sentences[:3] else "")
            log.info("storytelling_complete", mode="chart", words=_word_count(narrative))
            state["narrative"] = narrative
            return state

        # ── Mode "report" : résumé exécutif + recommandations (JSON) ────────
        prompt_json = _build_report_json_prompt(
            state["prompt"], insights, tone, language, template
        )
        result = await call_llm_json(
            prompt=prompt_json,
            system=_NARRATIVE_JSON_SYSTEM,
            model=settings.litellm_default_model,
        )

        executive_summary = _strip_markdown(str(result.get("executive_summary", "")))
        raw_recommendations = result.get("recommendations", [])
        recommendations = [
            _strip_markdown(str(r)) for r in raw_recommendations if r
        ][:2]  # max 2

        log.info(
            "storytelling_complete",
            mode="report",
            summary_words=_word_count(executive_summary),
            recommendations=len(recommendations),
        )
        state["narrative"] = executive_summary
        state["recommendations"] = recommendations
        return state
