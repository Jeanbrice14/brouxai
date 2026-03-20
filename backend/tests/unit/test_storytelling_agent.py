"""Tests unitaires Sprint 5 — StorytellingAgent complet.

Le LLM est systématiquement mocké (aucun appel réseau réel).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.agents.storytelling_agent import StorytellingAgent, _word_count
from app.pipeline.state import initial_state

# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_INSIGHTS = [
    {
        "title": "Domination IDF",
        "description": "L'Île-de-France génère 42% du CA total.",
        "type": "highlight",
        "confidence": 0.92,
        "supporting_data": "ca_ht IDF: 42100",
        "impact": "high",
    },
    {
        "title": "Croissance Nord",
        "description": "Le Nord affiche une croissance de 15% sur la période.",
        "type": "trend",
        "confidence": 0.85,
        "supporting_data": "ca_ht Nord: 12500, +15%",
        "impact": "medium",
    },
    {
        "title": "Faiblesse Ouest",
        "description": "L'Ouest reste en retrait avec un CA de 5400.",
        "type": "comparison",
        "confidence": 0.80,
        "supporting_data": "ca_ht Ouest: 5400",
        "impact": "low",
    },
]

# Narration longue (> 200 mots) — sans Markdown
_LONG_NARRATIVE = (
    "L'analyse des ventes régionales révèle des disparités marquées. "
    "L'Île-de-France représente la part la plus importante du chiffre d'affaires total. "
    "Cette concentration géographique mérite une attention particulière dans la stratégie. "
    "Le Nord affiche une progression encourageante sur la période analysée. "
    "Cette croissance est portée par une base clientèle solide et diversifiée. "
    "Les équipes commerciales du Nord ont su capitaliser sur les opportunités du marché. "
    "En revanche, l'Ouest présente des résultats en deçà des attentes fixées. "
    "Un plan d'action spécifique devra être déployé pour améliorer la performance. "
    "La redistribution des ressources commerciales constitue une piste à explorer. "
    "En termes de recommandations, trois axes prioritaires se dégagent de cette analyse. "
    "Premièrement, renforcer la présence commerciale dans les régions sous-performantes. "
    "Deuxièmement, capitaliser sur le succès de l'IDF pour dupliquer les bonnes pratiques. "
    "Troisièmement, mettre en place un suivi mensuel des indicateurs par région. "
) * 2  # ~200+ mots

# Narration courte (< 200 mots)
_SHORT_NARRATIVE = "Les ventes sont bonnes. L'IDF domine. Croissance au Nord."


def _make_state(brand_kit: dict | None = None, insights: list | None = None) -> dict:
    state = initial_state(
        tenant_id="tenant-test",
        user_id="user-test",
        report_id="report-test",
        prompt="Analyse les ventes par région",
        raw_data_refs=["s3://narr8-dev/uploads/ventes.csv"],
        brand_kit=brand_kit or {"tone": "formel", "language": "fr"},
    )
    state["insights"] = insights if insights is not None else SAMPLE_INSIGHTS
    return state


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_nominal_generates_narrative():
    """3 insights valides → narration non vide et > 200 mots."""
    state = _make_state()
    agent = StorytellingAgent()

    with patch("app.agents.storytelling_agent.call_llm", AsyncMock(return_value=_LONG_NARRATIVE)):
        result = await agent(state)

    assert result["status"] != "error", f"Erreurs: {result['errors']}"
    narrative = result["narrative"]
    assert narrative, "state['narrative'] ne doit pas être vide"
    assert _word_count(narrative) >= 200, (
        f"Narration trop courte: {_word_count(narrative)} mots (attendu ≥ 200)"
    )


@pytest.mark.asyncio
async def test_respects_brand_kit_tone():
    """Le ton du brand_kit apparaît dans le prompt envoyé au LLM."""
    state = _make_state(brand_kit={"tone": "synthétique", "language": "fr"})
    agent = StorytellingAgent()

    mock_llm = AsyncMock(return_value=_LONG_NARRATIVE)

    with patch("app.agents.storytelling_agent.call_llm", mock_llm):
        await agent(state)

    assert mock_llm.called
    call_kwargs = mock_llm.call_args
    prompt_sent: str = call_kwargs.kwargs.get("prompt") or call_kwargs.args[0]

    assert "synthétique" in prompt_sent, (
        f"Le ton 'synthétique' devrait apparaître dans le prompt. "
        f"Prompt (500 premiers chars): {prompt_sent[:500]}"
    )


@pytest.mark.asyncio
async def test_cleans_markdown_from_narrative():
    """La narration finale ne contient pas de balises Markdown."""
    # Construit un texte avec markdown qui dépasse 200 mots
    paragraph = (
        "La **performance** des ventes est très *bonne* cette année. "
        "L'équipe commerciale a réalisé des `résultats` exceptionnels. "
    )
    md_narrative = f"## Résumé exécutif\n{paragraph * 15}\n## Analyse\n{paragraph * 15}"

    state = _make_state()
    agent = StorytellingAgent()

    with patch("app.agents.storytelling_agent.call_llm", AsyncMock(return_value=md_narrative)):
        result = await agent(state)

    narrative = result["narrative"]
    assert "**" not in narrative, "Les balises '**' ne doivent pas être dans la narration finale"
    assert "##" not in narrative, "Les balises '##' ne doivent pas être dans la narration finale"
    assert "*" not in narrative, "Les balises '*' ne doivent pas être dans la narration finale"
    # Le contenu textuel doit toujours être présent
    assert "performance" in narrative
    assert "ventes" in narrative


@pytest.mark.asyncio
async def test_regenerates_if_too_short():
    """Narration trop courte (< 200 mots) → second appel LLM avec prompt enrichi."""
    state = _make_state()
    agent = StorytellingAgent()

    # Premier appel : texte court ; deuxième appel : texte long
    mock_llm = AsyncMock(side_effect=[_SHORT_NARRATIVE, _LONG_NARRATIVE])

    with patch("app.agents.storytelling_agent.call_llm", mock_llm):
        result = await agent(state)

    assert mock_llm.call_count == 2, (
        f"call_llm devrait être appelé 2 fois (court → long). Appelé {mock_llm.call_count} fois."
    )
    # La narration finale est le résultat du deuxième appel
    narrative = result["narrative"]
    assert _word_count(narrative) >= 200, (
        f"La narration finale doit être longue ({_word_count(narrative)} mots < 200)"
    )
