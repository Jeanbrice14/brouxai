from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.agents.intent_agent import IntentAgent
from app.pipeline.state import initial_state


def _make_state(prompt: str) -> dict:
    return initial_state(
        tenant_id="test",
        user_id="test",
        report_id="test-id",
        prompt=prompt,
        raw_data_refs=[],
    )


@pytest.mark.asyncio
async def test_detects_simple_query_from_keywords():
    agent = IntentAgent()
    state = _make_state("Donne moi le top 5 des clients")
    result = await agent.run(state)
    assert result["intent"] == "simple_query"
    assert result["response_type"] == "table"
    assert result["intent_confidence"] > 0


@pytest.mark.asyncio
async def test_detects_chart_request_from_keywords():
    agent = IntentAgent()
    state = _make_state("Montre l'évolution mensuelle des ventes")
    result = await agent.run(state)
    assert result["intent"] == "chart_request"
    assert result["response_type"] == "chart"


@pytest.mark.asyncio
async def test_detects_full_report_from_keywords():
    agent = IntentAgent()
    state = _make_state("Génère une analyse complète des performances Q3")
    result = await agent.run(state)
    assert result["intent"] == "full_report"
    assert result["response_type"] == "report"


@pytest.mark.asyncio
async def test_fallback_to_llm_on_no_keywords():
    """Prompt sans aucun keyword → fallback LLM."""
    agent = IntentAgent()
    state = _make_state("Que se passe-t-il avec les données ?")

    llm_response = {"intent": "simple_query", "confidence": 0.7, "response_type": "table"}

    with patch(
        "app.agents.intent_agent.call_llm_json", AsyncMock(return_value=llm_response)
    ) as mock_llm:
        result = await agent.run(state)

    mock_llm.assert_called_once()
    assert result["intent"] == "simple_query"
    assert result["response_type"] == "table"


@pytest.mark.asyncio
async def test_simple_query_preferred_on_tie():
    """En cas d'égalité → simple_query (plus rapide)."""
    agent = IntentAgent()
    # "top" = simple_query keyword, "évolution" = chart keyword → égalité → simple_query
    state = _make_state("top évolution des ventes")
    result = await agent.run(state)
    assert result["intent"] == "simple_query"


@pytest.mark.asyncio
async def test_intent_never_blocks_pipeline_on_llm_error():
    """Si le LLM échoue → fallback full_report sans exception."""
    agent = IntentAgent()
    state = _make_state("données incompréhensibles xyz abc")

    with patch(
        "app.agents.intent_agent.call_llm_json",
        AsyncMock(side_effect=Exception("LLM error")),
    ):
        result = await agent.run(state)

    assert result["intent"] == "full_report"
    assert result["response_type"] == "report"
    assert "errors" not in result or len(result.get("errors", [])) == 0
