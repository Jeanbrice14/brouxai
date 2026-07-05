"""Tests unitaires StorytellingAgent — mode report utilise call_llm_json.

Le LLM est systématiquement mocké (aucun appel réseau réel).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.agents.storytelling_agent import StorytellingAgent
from app.pipeline.state import initial_state
from app.services.llm import call_llm_json

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

_MOCK_REPORT_RESPONSE = {
    "executive_summary": (
        "CA total : 99k€ en progression de 196% sur la période. "
        "L'Île-de-France domine avec 42% du chiffre d'affaires, "
        "tandis que le Nord affiche une croissance de 15%."
    ),
    "recommendations": [
        "Investiguer les freins commerciaux en région Ouest pour améliorer la performance.",
        "Capitaliser sur le modèle Île-de-France pour dupliquer les bonnes pratiques.",
    ],
}


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
async def test_report_mode_generates_summary_and_reco():
    """Mode report → narrative (résumé) + recommendations (liste) non vides."""
    state = _make_state()
    agent = StorytellingAgent()

    with patch("app.agents.storytelling_agent.call_llm_json", AsyncMock(return_value=_MOCK_REPORT_RESPONSE)):
        result = await agent(state)

    assert result["status"] != "error", f"Erreurs : {result['errors']}"
    assert result["narrative"], "narrative ne doit pas être vide"
    assert isinstance(result["recommendations"], list), "recommendations doit être une liste"
    assert len(result["recommendations"]) > 0, "recommendations ne doit pas être vide"


@pytest.mark.asyncio
async def test_respects_brand_kit_tone():
    """Le ton du brand_kit apparaît dans le prompt envoyé au LLM."""
    state = _make_state(brand_kit={"tone": "synthétique", "language": "fr"})
    agent = StorytellingAgent()

    mock_llm = AsyncMock(return_value=_MOCK_REPORT_RESPONSE)
    with patch("app.agents.storytelling_agent.call_llm_json", mock_llm):
        await agent(state)

    assert mock_llm.called
    call_kwargs = mock_llm.call_args
    prompt_sent: str = call_kwargs.kwargs.get("prompt") or call_kwargs.args[0]
    assert "synthétique" in prompt_sent, (
        f"Le ton 'synthétique' devrait apparaître dans le prompt. "
        f"Prompt (500 premiers chars) : {prompt_sent[:500]}"
    )


@pytest.mark.asyncio
async def test_strips_markdown_from_executive_summary():
    """Le résumé exécutif ne contient pas de balises Markdown."""
    md_response = {
        "executive_summary": "**Performance** exceptionnelle : CA *99k€*. `Croissance` de 196%.",
        "recommendations": ["Action 1"],
    }
    state = _make_state()
    agent = StorytellingAgent()

    with patch("app.agents.storytelling_agent.call_llm_json", AsyncMock(return_value=md_response)):
        result = await agent(state)

    narrative = result["narrative"]
    assert "**" not in narrative, "Les balises '**' ne doivent pas être dans la narration finale"
    assert "*" not in narrative, "Les balises '*' ne doivent pas être dans la narration finale"
    assert "`" not in narrative, "Les balises '`' ne doivent pas être dans la narration finale"
    assert "Performance" in narrative
    assert "99k" in narrative


@pytest.mark.asyncio
async def test_recommendations_capped_at_two():
    """Les recommandations sont limitées à 2 maximum."""
    many_reco_response = {
        "executive_summary": "Résumé court.",
        "recommendations": ["Reco 1", "Reco 2", "Reco 3", "Reco 4"],
    }
    state = _make_state()
    agent = StorytellingAgent()

    with patch("app.agents.storytelling_agent.call_llm_json", AsyncMock(return_value=many_reco_response)):
        result = await agent(state)

    assert len(result["recommendations"]) <= 2, (
        f"Max 2 recommandations attendues, obtenu {len(result['recommendations'])}"
    )


@pytest.mark.asyncio
async def test_table_mode_single_sentence():
    """Mode table → titre reformulé + une seule phrase via call_llm_json."""
    state = _make_state()
    state["response_type"] = "table"
    agent = StorytellingAgent()

    with patch(
        "app.agents.storytelling_agent.call_llm_json",
        AsyncMock(return_value={
            "title": "Ventes par région Nord",
            "summary": "Les ventes ont progressé de 23% en région Nord.",
        }),
    ):
        result = await agent(state)

    assert result["status"] != "error"
    narrative = result["narrative"]
    assert narrative, "narrative ne doit pas être vide"
    sentences = [s for s in narrative.split(".") if s.strip()]
    assert len(sentences) <= 1, f"Mode table : 1 seule phrase attendue, obtenu : {narrative!r}"
    assert result["narrative_title"] == "Ventes par région Nord"


@pytest.mark.asyncio
async def test_chart_mode_short_narrative():
    """Mode chart → titre reformulé + 2-3 phrases via call_llm_json."""
    state = _make_state()
    state["response_type"] = "chart"
    agent = StorytellingAgent()

    with patch(
        "app.agents.storytelling_agent.call_llm_json",
        AsyncMock(return_value={
            "title": "Évolution du CA par région",
            "narrative": "IDF domine le CA. Le Nord est en croissance. L'Ouest est en retrait.",
        }),
    ):
        result = await agent(state)

    assert result["status"] != "error"
    narrative = result["narrative"]
    assert narrative, "narrative ne doit pas être vide"
    sentences = [s.strip() for s in narrative.split(".") if s.strip()]
    assert len(sentences) <= 3, f"Mode chart : max 3 phrases attendues. Obtenu : {narrative!r}"
    assert result["narrative_title"] == "Évolution du CA par région"


# ── Régression signature : call_llm_json n'accepte pas `temperature` ──────────
# (contrairement à call_llm) — un AsyncMock nu ne l'aurait jamais détecté, d'où
# spec=call_llm_json ici pour faire échouer l'appel si un kwarg invalide est passé.


@pytest.mark.asyncio
async def test_table_mode_call_matches_real_call_llm_json_signature():
    """Régression réelle : storytelling_agent appelait call_llm_json(..., temperature=0.2),
    or cette fonction n'accepte pas ce kwarg (contrairement à call_llm) — TypeError silencieux
    côté agent (catché par BaseAgent, narrative/narrative_title restaient vides sans jamais
    faire échouer le test, un AsyncMock nu acceptant n'importe quel kwarg)."""
    state = _make_state()
    state["response_type"] = "table"
    agent = StorytellingAgent()

    spec_mock = AsyncMock(spec=call_llm_json)
    spec_mock.return_value = {"title": "Titre test", "summary": "Résumé test."}

    with patch("app.agents.storytelling_agent.call_llm_json", spec_mock):
        result = await agent(state)

    assert result["status"] != "error", f"Erreurs: {result['errors']}"
    assert result["narrative"] == "Résumé test."
    assert result["narrative_title"] == "Titre test"


@pytest.mark.asyncio
async def test_chart_mode_call_matches_real_call_llm_json_signature():
    """Même régression que ci-dessus, pour le mode chart."""
    state = _make_state()
    state["response_type"] = "chart"
    agent = StorytellingAgent()

    spec_mock = AsyncMock(spec=call_llm_json)
    spec_mock.return_value = {"title": "Titre test", "narrative": "Narration test."}

    with patch("app.agents.storytelling_agent.call_llm_json", spec_mock):
        result = await agent(state)

    assert result["status"] != "error", f"Erreurs: {result['errors']}"
    assert result["narrative"] == "Narration test."
    assert result["narrative_title"] == "Titre test"
