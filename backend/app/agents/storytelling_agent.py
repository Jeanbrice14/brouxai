from __future__ import annotations

import re

import structlog

from app.agents.base_agent import BaseAgent
from app.config import settings
from app.pipeline.state import PipelineState
from app.services.llm import call_llm_json

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
        '  "title": "titre court (5-8 mots) du rapport, reformulé en intitulé — JAMAIS la '
        "demande originale recopiée telle quelle, jamais de point d'interrogation. Exemple : "
        "demande 'Quelle est la performance commerciale ?' → titre 'Performance commerciale "
        "2025'.\",\n"
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


_QUESTION_PREFIXES = re.compile(
    r"^(quelle?s?\s+(est|sont|était|étaient)|montre[- ]moi|affiche|donne[- ]moi|peux[- ]tu\s+"
    r"(me\s+)?(montrer|afficher|donner)|combien|comment|est[- ]ce\s+que)\b\s*",
    re.IGNORECASE,
)


def _fallback_title(prompt: str) -> str:
    """Reformulation minimale (règles, pas de LLM) si le titre LLM est vide — filet de
    sécurité, jamais le mode normal. Retire les tournures interrogatives et la ponctuation
    de question pour éviter d'afficher la question brute comme titre."""
    title = _QUESTION_PREFIXES.sub("", prompt.strip()).strip()
    title = title.rstrip("?!.").strip()
    if not title:
        return "Résultats"
    return title[0].upper() + title[1:] if len(title) > 1 else title.upper()


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

    3 modes selon state["response_type"], tous en call_llm_json (titre + narration dans le
    même appel — jamais le prompt brut affiché comme titre) :
    - "table"  : {title, summary} — 1 seule phrase de synthèse (max 20 mots)
    - "chart"  : {title, narrative} — 2-3 phrases (tendance principale + point notable)
    - "report" : {executive_summary, recommendations} (titre géré séparément, cf. _build_report_json_prompt)

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

        # ── Mode "table" : titre reformulé + 1 seule phrase ──────────────────
        if response_type == "table":
            prompt_text = (
                f"Question de l'utilisateur : {state.get('prompt', '')}\n"
                f"Insights : {_format_insights_for_prompt(insights[:2])}\n\n"
                'Retourne UNIQUEMENT ce JSON : {"title": "...", "summary": "..."}\n'
                "  - title : titre court (5-8 mots), reformulé en intitulé de résultat — "
                "JAMAIS la question recopiée telle quelle, jamais de point d'interrogation. "
                "Exemple : question 'Quel est le total des ventes par région ?' → titre "
                "'Total des ventes par région'.\n"
                "  - summary : UNE SEULE phrase de synthèse (maximum 20 mots), texte pur "
                "sans titre ni liste."
            )
            result = await call_llm_json(
                prompt=prompt_text,
                system=(
                    "Tu rédiges un titre court et UNE phrase factuelle et concise. "
                    "Texte pur, sans Markdown. Réponds UNIQUEMENT en JSON valide."
                ),
                model=settings.litellm_cheap_model,
            )
            narrative = _strip_markdown(str(result.get("summary", "")))
            sentences = [s.strip() for s in narrative.replace("\n", " ").split(".") if s.strip()]
            narrative = (sentences[0] + ".") if sentences else narrative[:120]
            title = _strip_markdown(str(result.get("title", ""))).strip() or _fallback_title(state.get("prompt", ""))
            log.info("storytelling_complete", mode="table", words=_word_count(narrative), title=title)
            state["narrative"] = narrative
            state["narrative_title"] = title
            return state

        # ── Mode "chart" : titre reformulé + 2-3 phrases ─────────────────────
        if response_type == "chart":
            prompt_text = (
                f"Question de l'utilisateur : {state.get('prompt', '')}\n"
                f"Insights : {_format_insights_for_prompt(insights[:3])}\n\n"
                'Retourne UNIQUEMENT ce JSON : {"title": "...", "narrative": "..."}\n'
                "  - title : titre court (5-8 mots) du graphique, reformulé en intitulé — "
                "JAMAIS la question recopiée telle quelle, jamais de point d'interrogation. "
                "Exemple : question 'Montre-moi l'évolution du CA' → titre "
                "'Évolution du chiffre d'affaires'.\n"
                "  - narrative : 2 à 3 phrases maximum. Commence directement par le chiffre ou "
                "la variation la plus importante — jamais par une phrase du type 'un graphique "
                "est possible' ou 'on pourrait montrer'. Base-toi UNIQUEMENT sur les chiffres "
                "déjà présents dans les insights ci-dessus (ne recalcule rien, ne suppose rien "
                "qui n'y figure pas).\n\n"
                "Style attendu pour narrative (à adapter aux chiffres réels des insights, ne "
                "jamais copier ces valeurs) :\n"
                '  "Le chiffre d\'affaires a progressé de +43 % entre la première et la '
                "dernière période, porté par une nette accélération récente (+25 % vs période "
                'précédente). L\'Île-de-France reste le premier contributeur avec 38 % du total."\n\n'
                "Si les insights fournis ne contiennent aucun chiffre exploitable (ex: seulement "
                "un comptage de lignes, aucune valeur agrégée pertinente), dis-le explicitement "
                "dans narrative plutôt que de décrire ce qu'un graphique pourrait montrer en théorie."
            )
            result = await call_llm_json(
                prompt=prompt_text,
                system=(
                    "Tu rédiges un titre court et 2-3 phrases factuelles et concises, chiffrées "
                    "quand les données le permettent. Texte pur, sans Markdown. Réponds "
                    "UNIQUEMENT en JSON valide."
                ),
                model=settings.litellm_cheap_model,
            )
            narrative = _strip_markdown(str(result.get("narrative", "")))
            sentences = [s.strip() for s in narrative.replace("\n", " ").split(".") if s.strip()]
            narrative = ". ".join(sentences[:3]) + ("." if sentences[:3] else "")
            title = _strip_markdown(str(result.get("title", ""))).strip() or _fallback_title(state.get("prompt", ""))
            log.info("storytelling_complete", mode="chart", words=_word_count(narrative), title=title)
            state["narrative"] = narrative
            state["narrative_title"] = title
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
        title = _strip_markdown(str(result.get("title", ""))).strip() or _fallback_title(state.get("prompt", ""))

        log.info(
            "storytelling_complete",
            mode="report",
            summary_words=_word_count(executive_summary),
            recommendations=len(recommendations),
            title=title,
        )
        state["narrative"] = executive_summary
        state["narrative_title"] = title
        state["recommendations"] = recommendations
        return state
