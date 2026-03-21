"""Test d'intégration bout en bout — pipeline complet Sprint 7.

Tous les agents réels sont invoqués mais leurs appels LLM, storage et cache
sont mockés. Vérifie que le pipeline produit un rapport HTML complet.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from app.pipeline.graph import build_pipeline
from app.pipeline.state import initial_state

# ── Mocks ─────────────────────────────────────────────────────────────────────

_MOCK_DF = pd.DataFrame(
    {
        "date": pd.to_datetime(["2024-01-05", "2024-02-03", "2024-03-01"]),
        "region": ["Nord", "Sud", "IDF"],
        "ca_ht": [12500.0, 8750.5, 42100.0],
    }
)

_CACHED_AGGREGATES = {
    "by_region": [
        {"region": "Nord", "ca_ht": 12500.0},
        {"region": "Sud", "ca_ht": 8750.5},
        {"region": "IDF", "ca_ht": 42100.0},
    ]
}

_LLM_METADATA_RESPONSE = {
    "semantic_name": "Chiffre d'affaires HT",
    "description": "Valeur de vente hors taxe",
    "type": "numeric",
    "unit": "EUR",
    "confidence": 0.92,
    "is_key_candidate": False,
    "grain": "Une ligne représente une vente par région.",
}

_LLM_INSIGHTS_RESPONSE = {
    "insights": [
        {
            "title": "Domination IDF",
            "description": "L'IDF génère 42% du CA total.",
            "type": "highlight",
            "confidence": 0.92,
            "supporting_data": "ca_ht IDF: 42100",
            "impact": "high",
        }
    ]
}

_LONG_NARRATIVE = (
    "L'analyse des ventes régionales révèle des disparités marquées.\n\n"
    "L'Île-de-France domine avec la plus grande part du chiffre d'affaires total. "
    "Cette concentration géographique mérite une attention particulière.\n\n"
    "Des actions correctives sont recommandées pour rééquilibrer la performance "
    "régionale et optimiser les ressources commerciales sur l'ensemble du territoire."
) * 3

_VIZ_SPEC = {
    "chart_type": "bar",
    "title": "CA par région",
    "data_key": "by_region",
    "x": "region",
    "y": "ca_ht",
    "color_by": None,
    "colors": {"primary": "#1E3A8A", "positive": "#16A34A", "negative": "#DC2626"},
    "annotations": [],
    "insight_ref": "Domination IDF",
}

_HTML_URL = "http://localhost:9000/narr8-dev/tenant-001/reports/report-001/report.html"


# ── State initial ─────────────────────────────────────────────────────────────


@pytest.fixture
def full_state():
    return initial_state(
        tenant_id="tenant-001",
        user_id="user-001",
        report_id="report-001",
        prompt="Analyse les ventes par région pour Q1 2024",
        raw_data_refs=["s3://narr8-dev/uploads/ventes_q1.csv"],
        brand_kit={
            "colors": {"primary": "#1E3A8A", "positive": "#16A34A", "negative": "#DC2626"},
            "tone": "formel",
            "language": "fr",
            "company_name": "ACME Corp",
        },
    )


@pytest.fixture
def pipeline():
    return build_pipeline()


@pytest.fixture(autouse=True)
def mock_all_external():
    """Mocke tous les appels LLM, storage et cache pour le pipeline complet."""
    with (
        # MetadataAgent
        patch("app.agents.metadata_agent.read_dataframe", AsyncMock(return_value=_MOCK_DF)),
        patch(
            "app.agents.metadata_agent.call_llm_json",
            AsyncMock(return_value=_LLM_METADATA_RESPONSE),
        ),
        # SchemaLinkingAgent (single-file → non appelé réellement)
        patch("app.agents.schema_linking_agent.read_dataframe", AsyncMock(return_value=_MOCK_DF)),
        patch(
            "app.agents.schema_linking_agent.call_llm_json",
            AsyncMock(return_value={"description": "Test"}),
        ),
        # DataAgent — cache hit
        patch("app.agents.data_agent.get_cache", AsyncMock(return_value=_CACHED_AGGREGATES)),
        patch("app.agents.data_agent.set_cache", AsyncMock()),
        # InsightAgent
        patch(
            "app.agents.insight_agent.call_llm_json", AsyncMock(return_value=_LLM_INSIGHTS_RESPONSE)
        ),
        # StorytellingAgent
        patch("app.agents.storytelling_agent.call_llm", AsyncMock(return_value=_LONG_NARRATIVE)),
        # VizAgent
        patch("app.agents.viz_agent.call_llm_json", AsyncMock(return_value=_VIZ_SPEC)),
        # QAAgent
        patch("app.agents.qa_agent.call_llm_json", AsyncMock(return_value={"issues": []})),
        # LayoutAgent
        patch("app.agents.layout_agent.upload_file", AsyncMock()),
    ):
        yield


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_full_pipeline_status_complete(pipeline, full_state):
    """Le pipeline complet se termine avec status == 'complete'."""
    result = await pipeline.ainvoke(full_state)
    assert result["status"] == "complete", (
        f"Status attendu 'complete', obtenu '{result['status']}'. Erreurs: {result.get('errors')}"
    )


@pytest.mark.asyncio
async def test_full_pipeline_no_errors(pipeline, full_state):
    """Aucune erreur dans state['errors'] tout au long du pipeline."""
    result = await pipeline.ainvoke(full_state)
    assert result["errors"] == [], f"Erreurs inattendues: {result['errors']}"


@pytest.mark.asyncio
async def test_full_pipeline_report_url_set(pipeline, full_state):
    """state['report_urls']['html_url'] est renseigné après le pipeline."""
    result = await pipeline.ainvoke(full_state)
    assert "html_url" in result["report_urls"], (
        f"html_url absent de report_urls: {result['report_urls']}"
    )
    assert result["report_urls"]["html_url"], "html_url ne doit pas être vide"


@pytest.mark.asyncio
async def test_full_pipeline_all_agents_wrote_state(pipeline, full_state):
    """Chaque agent a écrit dans sa clé de state."""
    result = await pipeline.ainvoke(full_state)

    assert result["metadata"] != {}, "MetadataAgent n'a pas écrit state['metadata']"
    assert result["schema"] != {}, "SchemaLinkingAgent n'a pas écrit state['schema']"
    assert result["aggregates"] != {}, "DataAgent n'a pas écrit state['aggregates']"
    assert result["insights"] != [], "InsightAgent n'a pas écrit state['insights']"
    assert result["narrative"] != "", "StorytellingAgent n'a pas écrit state['narrative']"
    assert result["viz_specs"] != [], "VizAgent n'a pas écrit state['viz_specs']"
    assert result["qa_report"] != {}, "QAAgent n'a pas écrit state['qa_report']"
    assert result["report_urls"] != {}, "LayoutAgent n'a pas écrit state['report_urls']"


@pytest.mark.asyncio
async def test_full_pipeline_no_hitl(pipeline, full_state):
    """Avec des mocks à haute confiance, pas de HITL déclenché."""
    result = await pipeline.ainvoke(full_state)
    assert result["hitl_pending"] is False, (
        f"HITL ne devrait pas être déclenché. checkpoint: {result.get('hitl_checkpoint')}"
    )
    assert result["hitl_checkpoint"] is None
