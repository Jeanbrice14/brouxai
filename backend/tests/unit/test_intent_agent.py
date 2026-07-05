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
async def test_detects_chart_request_for_breakdown_by_arbitrary_dimension():
    """Régression réelle : "ventes par catégorie" tombait en simple_query/table car seuls
    "par mois"/"par région"/"par semaine" étaient reconnus en dur — toute autre dimension
    ("catégorie", "produit", "client"...) ratait le match et finissait sur le fallback LLM,
    peu fiable pour ce cas. Un "par X" générique doit être reconnu comme chart_request."""
    agent = IntentAgent()
    state = _make_state("Quelles sont les ventes par category en 2022")
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
async def test_fallback_llm_includes_chat_history_when_present():
    """La mémoire de conversation doit apparaître dans le prompt du fallback LLM — utile
    pour classifier une question de suivi courte ("et pour le mois dernier ?")."""
    agent = IntentAgent()
    state = _make_state("Que se passe-t-il avec les données ?")
    state["chat_history"] = [
        {"question": "Quelle catégorie a le plus de ventes ?", "answer_summary": "Bikes domine avec 23.6M€."},
    ]

    llm_response = {"intent": "simple_query", "confidence": 0.7, "response_type": "table"}

    with patch(
        "app.agents.intent_agent.call_llm_json", AsyncMock(return_value=llm_response)
    ) as mock_llm:
        await agent.run(state)

    prompt_sent = mock_llm.call_args.kwargs.get("prompt") or mock_llm.call_args.args[0]
    assert "Quelle catégorie a le plus de ventes ?" in prompt_sent
    assert "Bikes domine avec 23.6M€." in prompt_sent


@pytest.mark.asyncio
async def test_chart_request_preferred_on_tie():
    """En cas d'égalité → chart_request l'emporte, pas simple_query.

    Régression réelle : "quelle est l'évolution du nombre de commande en 2022" tombait en
    table car "nombre" (simple_query) et "évolution" (chart_request) faisaient jeu égal, et
    l'ancien tie-break favorisait simple_query. Un mot-clé chart_request est un signal
    explicite de visualisation — il doit primer sur un mot-clé simple_query générique qui
    décrit juste la mesure demandée."""
    agent = IntentAgent()
    # "top" = simple_query keyword, "évolution" = chart keyword → égalité → chart_request
    state = _make_state("top évolution des ventes")
    result = await agent.run(state)
    assert result["intent"] == "chart_request"


@pytest.mark.asyncio
async def test_detects_chart_request_for_evolution_tied_with_generic_metric_keyword():
    """Régression réelle exacte : "quelle est l'évolution du nombre de commande en 2022"."""
    agent = IntentAgent()
    state = _make_state("quelle est l'évolution du nombre de commande en 2022")
    result = await agent.run(state)
    assert result["intent"] == "chart_request"
    assert result["response_type"] == "chart"


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


@pytest.mark.asyncio
async def test_moyenne_par_x_not_treated_as_breakdown():
    """"moyenne ... par X" est un ratio par unité (ex: panier moyen / commande, un seul
    chiffre), pas une répartition par dimension — contrairement à "ventes par catégorie".
    Sans cette exception, le bonus de répartition générique faisait basculer ce genre de
    question à tort en chart_request (trouvé en testant le fix de "ventes par catégorie")."""
    agent = IntentAgent()
    state = _make_state("Quelle est la moyenne du panier par commande ?")
    result = await agent.run(state)
    assert result["intent"] == "simple_query"
    assert result["response_type"] == "table"


@pytest.mark.asyncio
async def test_detects_chart_request_without_accent_on_keyword():
    """Régression réelle : "quelle est l'evolution des commandes en 2022" (sans accent sur
    "evolution") ne matchait aucun mot-clé de aucune des 3 catégories — "évolution" dans
    CHART_REQUEST_KEYWORDS est accentué — et tombait sur le fallback LLM, non déterministe,
    qui a classé la question en simple_query/table. De nombreux utilisateurs tapent sans
    accents (clavier, habitude, mobile)."""
    agent = IntentAgent()
    state = _make_state("quelle est l'evolution des commandes en 2022")
    result = await agent.run(state)
    assert result["intent"] == "chart_request"
    assert result["response_type"] == "chart"
