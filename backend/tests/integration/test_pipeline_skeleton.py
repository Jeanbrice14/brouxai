"""Tests d'intégration Sprint 1 — pipeline squelette LangGraph.

Vérifie que :
1. Le pipeline tourne de bout en bout sans erreur
2. Chaque agent a bien écrit dans sa clé de state
3. Le status final n'est pas "error"

Notes :
- Sprint 2 : MetadataAgent réel → appels storage + LLM mockés.
- Sprint 3 : SchemaLinkingAgent réel → appels storage + LLM mockés (single-file → pas d'appels réels).
- Sprint 4 : DataAgent réel → cache mocké (hit), LLM + storage non appelés via cache hit.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from app.pipeline.graph import build_pipeline
from app.pipeline.state import initial_state

# ── DataFrame minimal utilisé par MetadataAgent dans le pipeline ──────────────
_MOCK_DF = pd.DataFrame(
    {
        "date": pd.to_datetime(["2024-01-05", "2024-02-03"]),
        "ca_ht": [12500.0, 8750.5],
        "region": ["Nord", "Sud"],
    }
)

# Agrégats pré-calculés retournés par le mock cache DataAgent (cache hit)
_CACHED_AGGREGATES = {"by_region": [{"region": "Nord", "ca_ht": 12500.0}]}

# Réponse LLM universelle : satisfait aussi bien les appels "colonne" que "grain"
_LLM_ANY_RESPONSE = {
    "semantic_name": "Valeur test",
    "description": "Description test",
    "type": "numeric",
    "unit": "",
    "confidence": 0.90,
    "is_key_candidate": False,
    "grain": "Une ligne représente une vente.",
}


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def sample_state():
    return initial_state(
        tenant_id="tenant-001",
        user_id="user-001",
        report_id="report-001",
        prompt="Analyse les ventes par région pour Q3 2024",
        raw_data_refs=["s3://narr8-dev/uploads/sales_q3.csv"],
        brand_kit={"colors": {"primary": "#1E3A8A"}, "tone": "formel"},
    )


@pytest.fixture
def pipeline():
    return build_pipeline()


@pytest.fixture(autouse=True)
def mock_external_deps():
    """Mocke les appels externes de MetadataAgent, SchemaLinkingAgent et DataAgent.

    Stratégie par agent :
    - MetadataAgent    : read_dataframe + call_llm_json mockés
    - SchemaLinkingAgent : read_dataframe + call_llm_json mockés (single-file → non appelés)
    - DataAgent        : cache hit mocké → LLM + storage non nécessaires
    """
    with (
        patch(
            "app.agents.metadata_agent.read_dataframe",
            AsyncMock(return_value=_MOCK_DF),
        ),
        patch(
            "app.agents.metadata_agent.call_llm_json",
            AsyncMock(return_value=_LLM_ANY_RESPONSE),
        ),
        patch(
            "app.agents.schema_linking_agent.read_dataframe",
            AsyncMock(return_value=_MOCK_DF),
        ),
        patch(
            "app.agents.schema_linking_agent.call_llm_json",
            AsyncMock(return_value={"description": "Relation de test."}),
        ),
        patch(
            "app.agents.data_agent.get_cache",
            AsyncMock(return_value=_CACHED_AGGREGATES),
        ),
        patch(
            "app.agents.data_agent.set_cache",
            AsyncMock(),
        ),
    ):
        yield


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_pipeline_runs_end_to_end(pipeline, sample_state):
    """Le pipeline tourne de bout en bout sans lever d'exception."""
    result = await pipeline.ainvoke(sample_state)
    assert result is not None


@pytest.mark.asyncio
async def test_pipeline_status_not_error(pipeline, sample_state):
    """Le status final du pipeline n'est pas 'error'."""
    result = await pipeline.ainvoke(sample_state)
    assert result["status"] != "error", f"Pipeline errors: {result.get('errors')}"


@pytest.mark.asyncio
async def test_pipeline_no_errors_list(pipeline, sample_state):
    """Aucune erreur enregistrée dans state['errors']."""
    result = await pipeline.ainvoke(sample_state)
    assert result["errors"] == [], f"Unexpected errors: {result['errors']}"


@pytest.mark.asyncio
async def test_metadata_agent_wrote_state(pipeline, sample_state):
    """MetadataAgent a écrit dans state['metadata']."""
    result = await pipeline.ainvoke(sample_state)
    assert result["metadata"] != {}, "metadata est vide — MetadataAgent n'a pas écrit"


@pytest.mark.asyncio
async def test_schema_agent_wrote_state(pipeline, sample_state):
    """SchemaLinkingAgent a écrit dans state['schema']."""
    result = await pipeline.ainvoke(sample_state)
    assert result["schema"] != {}, "schema est vide — SchemaLinkingAgent n'a pas écrit"


@pytest.mark.asyncio
async def test_data_agent_wrote_state(pipeline, sample_state):
    """DataAgent a écrit dans state['aggregates']."""
    result = await pipeline.ainvoke(sample_state)
    assert result["aggregates"] != {}, "aggregates est vide — DataAgent n'a pas écrit"


@pytest.mark.asyncio
async def test_insight_agent_wrote_state(pipeline, sample_state):
    """InsightAgent a écrit dans state['insights']."""
    result = await pipeline.ainvoke(sample_state)
    assert result["insights"] != [], "insights est vide — InsightAgent n'a pas écrit"


@pytest.mark.asyncio
async def test_storytelling_agent_wrote_state(pipeline, sample_state):
    """StorytellingAgent a écrit dans state['narrative']."""
    result = await pipeline.ainvoke(sample_state)
    assert result["narrative"] != "", "narrative est vide — StorytellingAgent n'a pas écrit"


@pytest.mark.asyncio
async def test_viz_agent_wrote_state(pipeline, sample_state):
    """VizAgent a écrit dans state['viz_specs']."""
    result = await pipeline.ainvoke(sample_state)
    assert result["viz_specs"] != [], "viz_specs est vide — VizAgent n'a pas écrit"


@pytest.mark.asyncio
async def test_qa_agent_wrote_state(pipeline, sample_state):
    """QAAgent a écrit dans state['qa_report']."""
    result = await pipeline.ainvoke(sample_state)
    assert result["qa_report"] != {}, "qa_report est vide — QAAgent n'a pas écrit"


@pytest.mark.asyncio
async def test_layout_agent_wrote_state(pipeline, sample_state):
    """LayoutAgent a écrit dans state['report_urls']."""
    result = await pipeline.ainvoke(sample_state)
    assert result["report_urls"] != {}, "report_urls est vide — LayoutAgent n'a pas écrit"
    assert "html_url" in result["report_urls"], "html_url manquant dans report_urls"


@pytest.mark.asyncio
async def test_pipeline_status_complete(pipeline, sample_state):
    """Le status final est 'complete' (défini par LayoutAgent)."""
    result = await pipeline.ainvoke(sample_state)
    assert result["status"] == "complete", f"Status inattendu: {result['status']}"


@pytest.mark.asyncio
async def test_hitl_not_triggered_in_stub(pipeline, sample_state):
    """Confidence 0.90 → pas de HITL déclenché."""
    result = await pipeline.ainvoke(sample_state)
    assert result["hitl_pending"] is False
    assert result["hitl_checkpoint"] is None
