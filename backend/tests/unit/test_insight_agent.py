"""Tests unitaires Sprint 5 — InsightAgent complet.

Le LLM est systématiquement mocké (aucun appel réseau réel).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.agents.insight_agent import InsightAgent
from app.config import settings
from app.pipeline.state import initial_state

# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_AGGREGATES = {
    "by_region": [
        {"region": "Nord", "ca_ht": 12500.0},
        {"region": "Sud", "ca_ht": 8750.5},
        {"region": "Est", "ca_ht": 21300.0},
        {"region": "Ouest", "ca_ht": 5400.0},
        {"region": "Ile-de-France", "ca_ht": 42100.0},
    ]
}

_INSIGHT_HIGH = {
    "title": "Domination IDF",
    "description": "L'Île-de-France génère 42% du CA total.",
    "type": "highlight",
    "confidence": 0.92,
    "supporting_data": "ca_ht IDF: 42100 soit 42% du total",
    "impact": "high",
}
_INSIGHT_MED = {
    "title": "Croissance Nord",
    "description": "Le Nord affiche une croissance solide.",
    "type": "trend",
    "confidence": 0.85,
    "supporting_data": "ca_ht Nord: 12500",
    "impact": "medium",
}
_INSIGHT_LOW = {
    "title": "Faiblesse Ouest",
    "description": "L'Ouest reste en retrait.",
    "type": "comparison",
    "confidence": 0.80,
    "supporting_data": "ca_ht Ouest: 5400 soit le plus bas",
    "impact": "low",
}


def _make_state(aggregates: dict | None = None) -> dict:
    state = initial_state(
        tenant_id="tenant-test",
        user_id="user-test",
        report_id="report-test",
        prompt="Analyse les ventes par région",
        raw_data_refs=["s3://narr8-dev/uploads/ventes.csv"],
    )
    state["aggregates"] = aggregates if aggregates is not None else SAMPLE_AGGREGATES
    return state


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_nominal_generates_insights():
    """3 insights valides : structure complète et présence dans state."""
    state = _make_state()
    agent = InsightAgent()

    mock_response = {"insights": [_INSIGHT_HIGH, _INSIGHT_MED, _INSIGHT_LOW]}

    with patch("app.agents.insight_agent.call_llm_json", AsyncMock(return_value=mock_response)):
        result = await agent(state)

    assert result["status"] != "error", f"Erreurs: {result['errors']}"
    insights = result["insights"]
    assert len(insights) == 3, f"Attendu 3 insights, obtenu {len(insights)}"

    required = ("title", "description", "type", "confidence")
    for ins in insights:
        for field in required:
            assert field in ins, f"Champ '{field}' manquant dans insight: {ins}"
        assert isinstance(ins["confidence"], float)
        assert 0.0 <= ins["confidence"] <= 1.0


@pytest.mark.asyncio
async def test_triggers_hitl_on_low_confidence():
    """Confidence 0.50 < seuil 0.80 → HITL déclenché."""
    state = _make_state()
    agent = InsightAgent()

    low_conf_insight = {**_INSIGHT_HIGH, "confidence": 0.50}
    mock_response = {"insights": [low_conf_insight]}

    with patch("app.agents.insight_agent.call_llm_json", AsyncMock(return_value=mock_response)):
        result = await agent(state)

    assert result["hitl_pending"] is True
    assert result["hitl_checkpoint"] == "cp3_insights"


@pytest.mark.asyncio
async def test_triggers_hitl_on_anomaly():
    """type='anomaly' → HITL même si confidence élevée."""
    state = _make_state()
    agent = InsightAgent()

    anomaly_insight = {**_INSIGHT_HIGH, "type": "anomaly", "confidence": 0.90}
    mock_response = {"insights": [anomaly_insight]}

    with patch("app.agents.insight_agent.call_llm_json", AsyncMock(return_value=mock_response)):
        result = await agent(state)

    assert result["hitl_pending"] is True
    assert result["hitl_checkpoint"] == "cp3_insights"


@pytest.mark.asyncio
async def test_filters_very_low_confidence_insights():
    """Insight avec confidence=0.25 < 0.30 doit être filtré."""
    state = _make_state()
    agent = InsightAgent()

    filtered_insight = {**_INSIGHT_HIGH, "confidence": 0.25, "title": "A filtrer"}
    mock_response = {"insights": [_INSIGHT_HIGH, _INSIGHT_MED, filtered_insight]}

    with patch("app.agents.insight_agent.call_llm_json", AsyncMock(return_value=mock_response)):
        result = await agent(state)

    assert result["status"] != "error"
    insights = result["insights"]
    assert len(insights) == 2, f"Attendu 2 insights après filtrage, obtenu {len(insights)}"
    titles = [i["title"] for i in insights]
    assert "A filtrer" not in titles, "L'insight avec confidence=0.25 aurait dû être filtré"


@pytest.mark.asyncio
async def test_uses_default_model_not_cheap():
    """InsightAgent utilise litellm_default_model (gpt-4o), pas le modèle économique."""
    state = _make_state()
    agent = InsightAgent()

    mock_llm = AsyncMock(return_value={"insights": [_INSIGHT_HIGH]})

    with patch("app.agents.insight_agent.call_llm_json", mock_llm):
        await agent(state)

    assert mock_llm.called, "call_llm_json aurait dû être appelé"
    call_kwargs = mock_llm.call_args
    model_used = call_kwargs.kwargs.get("model") or (
        call_kwargs.args[2] if len(call_kwargs.args) > 2 else None
    )

    assert model_used == settings.litellm_default_model, (
        f"Modèle utilisé: '{model_used}', attendu: '{settings.litellm_default_model}'. "
        "InsightAgent doit utiliser gpt-4o, pas gpt-4o-mini."
    )
    assert model_used != settings.litellm_cheap_model, (
        "InsightAgent ne doit PAS utiliser litellm_cheap_model"
    )
