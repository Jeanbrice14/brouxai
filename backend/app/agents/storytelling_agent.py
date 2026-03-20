from __future__ import annotations

import re

import structlog

from app.agents.base_agent import BaseAgent
from app.config import settings
from app.pipeline.state import PipelineState
from app.services.llm import call_llm

logger = structlog.get_logger(__name__)

_NARRATIVE_SYSTEM_PROMPT = (
    "Tu es un expert en communication data-driven et storytelling analytique. "
    "Tu rédiges des narrations claires, professionnelles et factuelles. "
    "Tu te bases UNIQUEMENT sur les insights fournis. "
    "Tu n'inventes JAMAIS de chiffres ni d'informations non présentes dans les insights. "
    "Ta réponse est du texte pur, sans aucun formatage Markdown (pas de **, ##, *, ``)."
)

_MIN_WORDS = 200
_MAX_WORDS = 2000
_NARRATIVE_TEMPERATURE = 0.3

_TONE_INSTRUCTIONS = {
    "formel": "Utilise un registre formel et professionnel, adapté à un rapport de direction.",
    "neutre": "Utilise un registre neutre et factuel, accessible à tout public.",
    "synthétique": "Utilise un registre concis et synthétique, va à l'essentiel.",
}


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


def _build_narrative_prompt(
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
        f"Insights à narrer :\n{formatted_insights}\n\n"
        "Rédige la narration avec ces 4 sections :\n"
        "1. RÉSUMÉ EXÉCUTIF (2-3 phrases maximum)\n"
        "2. ANALYSE PRINCIPALE (un paragraphe par insight majeur)\n"
        "3. POINTS D'ATTENTION (uniquement si anomalies présentes, sinon omettre)\n"
        "4. RECOMMANDATIONS (2-3 actions concrètes et actionnables)\n\n"
        "Instructions strictes :\n"
        "  - Basé UNIQUEMENT sur les insights fournis\n"
        "  - Ne pas inventer de chiffres\n"
        "  - Texte pur, sans Markdown\n"
    )
    if template:
        base += f"\nTemplate narratif à respecter :\n{template}\n"
    return base


def _build_enriched_prompt(original_prompt: str, short_narrative: str) -> str:
    return (
        f"{original_prompt}\n\n"
        f"La narration précédente était trop courte ({len(short_narrative.split())} mots). "
        f"Développe davantage chaque section. "
        f"L'analyse principale doit être plus détaillée avec des exemples tirés des données."
    )


def _strip_markdown(text: str) -> str:
    """Supprime les balises Markdown du texte pour obtenir du texte pur."""
    # Bold+italic : ***text*** → text
    text = re.sub(r"\*{3}(.+?)\*{3}", r"\1", text, flags=re.DOTALL)
    # Bold : **text** → text
    text = re.sub(r"\*{2}(.+?)\*{2}", r"\1", text, flags=re.DOTALL)
    # Italic : *text* → text
    text = re.sub(r"\*(.+?)\*", r"\1", text, flags=re.DOTALL)
    # Headers : ## Titre → Titre
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Inline code : `code` → code
    text = re.sub(r"`(.+?)`", r"\1", text)
    return text.strip()


def _word_count(text: str) -> int:
    return len(text.split())


class StorytellingAgent(BaseAgent):
    """Agent 5 — Construit la narration data-driven structurée.

    Étapes :
        A. Récupération du contexte narratif depuis brand_kit (ton, langue, template).
        B. Construction du prompt narratif avec les insights.
        C. Post-traitement : nettoyage Markdown + vérification longueur (200-2000 mots).
           Re-génération unique si < 200 mots.
        D. Écriture dans state["narrative"].

    Input  : state["insights"] + state["brand_kit"]
    Output : state["narrative"]
    Modèle : settings.litellm_default_model (gpt-4o), temperature=0.3
    Règle  : ne jamais inventer de chiffres — texte pur sans Markdown
    """

    name = "storytelling_agent"

    async def run(self, state: PipelineState) -> PipelineState:
        log = logger.bind(report_id=state.get("report_id"))
        brand_kit = state.get("brand_kit", {})

        # ── Étape A : contexte narratif ──────────────────────────────────────
        tone = brand_kit.get("tone", "formel")
        language = brand_kit.get("language", "fr")
        template = brand_kit.get("narrative_template")

        # ── Étape B : prompt narratif ────────────────────────────────────────
        insights = state.get("insights", [])
        narrative_prompt = _build_narrative_prompt(
            state["prompt"], insights, tone, language, template
        )

        # ── Étape C : génération + post-traitement ───────────────────────────
        narrative = await call_llm(
            prompt=narrative_prompt,
            system=_NARRATIVE_SYSTEM_PROMPT,
            model=settings.litellm_default_model,
            temperature=_NARRATIVE_TEMPERATURE,
        )
        narrative = _strip_markdown(narrative)

        # Re-générer une fois si trop court
        if _word_count(narrative) < _MIN_WORDS:
            log.warning(
                "storytelling_narrative_too_short",
                words=_word_count(narrative),
                threshold=_MIN_WORDS,
            )
            enriched = _build_enriched_prompt(narrative_prompt, narrative)
            narrative = await call_llm(
                prompt=enriched,
                system=_NARRATIVE_SYSTEM_PROMPT,
                model=settings.litellm_default_model,
                temperature=_NARRATIVE_TEMPERATURE,
            )
            narrative = _strip_markdown(narrative)

        words = _word_count(narrative)
        log.info("storytelling_complete", words=words)

        # ── Étape D ──────────────────────────────────────────────────────────
        state["narrative"] = narrative
        return state
