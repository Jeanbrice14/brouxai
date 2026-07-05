"""Test d'intégration Sprint 9 — workflow HITL complet.

Vérifie que :
1. Le pipeline s'arrête sur un checkpoint HITL (cp3_insights simulé)
2. resume_pipeline() construit un sous-pipeline fonctionnel
3. Le sous-pipeline reprend à partir du bon agent et va jusqu'à END

LLM, storage et cache sont mockés.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from app.agents.insight_agent import InsightAgent
from app.pipeline.checkpoints import resume_pipeline
from app.pipeline.graph import build_pipeline
from app.pipeline.state import initial_state

# ── Mocks ─────────────────────────────────────────────────────────────────────

_MOCK_DF = pd.DataFrame(
    {
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
    "description": "Valeur de vente HT",
    "type": "numeric",
    "unit": "EUR",
    "confidence": 0.92,
    "is_key_candidate": False,
    "grain": "Une ligne = une vente.",
}

_LLM_INSIGHTS_LOW_CONFIDENCE = {
    "insights": [
        {
            "title": "Tendance incertaine",
            "description": "Evolution incertaine des ventes.",
            "type": "trend",
            "confidence": 0.65,  # < 0.80 → déclenche HITL cp3_insights
            "supporting_data": "ca_ht by_region",
            "impact": "medium",
        }
    ]
}

_LLM_INSIGHTS_HIGH_CONFIDENCE = {
    "insights": [
        {
            "title": "IDF domine",
            "description": "L'IDF génère 42% du CA.",
            "type": "highlight",
            "confidence": 0.92,
            "supporting_data": "ca_ht IDF: 42100",
            "impact": "high",
        }
    ]
}

_STORYTELLING_REPORT_RESPONSE = {
    "title": "Analyse des ventes régionales",
    "executive_summary": (
        "L'Île-de-France domine avec la plus grande part du chiffre d'affaires total. "
        "Cette concentration géographique appelle une attention particulière."
    ),
    "recommendations": [
        "Diversifier les sources de revenus hors Île-de-France.",
        "Renforcer la présence commerciale en région Nord et Sud.",
    ],
}

_MOCK_VIZ_SPEC = {
    "chart_type": "bar",
    "title": "CA par région",
    "data_key": "by_region",
    "x": "region",
    "y": "ca_ht",
    "colors": {"primary": "#1E3A8A"},
    "annotations": [],
}

_MOCK_QA_LLM_RESPONSE = {
    "issues": [],
    "confidence_score": 0.92,
}

_BASE_PATCHES = {
    "app.agents.metadata_agent.read_dataframe": AsyncMock(return_value=_MOCK_DF),
    "app.agents.metadata_agent.call_llm_json": AsyncMock(return_value=_LLM_METADATA_RESPONSE),
    "app.agents.schema_linking_agent.read_dataframe": AsyncMock(return_value=_MOCK_DF),
    "app.agents.schema_linking_agent.call_llm_json": AsyncMock(
        return_value={"relations": [], "alerts": []}
    ),
    "app.agents.data_agent.read_dataframe": AsyncMock(return_value=_MOCK_DF),
    "app.agents.data_agent.call_llm_json": AsyncMock(
        return_value={
            "code": "result = df.groupby('region')['ca_ht'].sum().reset_index().to_dict('records')"
        }
    ),
    "app.agents.data_agent.get_cache": AsyncMock(return_value=_CACHED_AGGREGATES),
    "app.agents.data_agent.set_cache": AsyncMock(),
    "app.agents.storytelling_agent.call_llm_json": AsyncMock(return_value=_STORYTELLING_REPORT_RESPONSE),
    "app.agents.viz_agent.call_llm_json": AsyncMock(return_value={"viz_specs": [_MOCK_VIZ_SPEC]}),
    "app.agents.qa_agent.call_llm_json": AsyncMock(return_value=_MOCK_QA_LLM_RESPONSE),
    "app.agents.layout_agent.upload_file": AsyncMock(),
    "app.agents.base_agent.save_report_state": AsyncMock(),
    "app.agents.base_agent.notify_hitl_required": AsyncMock(),
}


def _make_state(**overrides):
    state = initial_state(
        tenant_id="test-tenant",
        user_id="test-user",
        report_id="test-hitl-001",
        prompt="Génère un rapport complet d'analyse pour valider le HITL",
        raw_data_refs=["s3://narr8-dev/test-tenant/datasets/test-hitl-001/ventes.csv"],
        brand_kit={"colors": {"primary": "#1E3A8A"}},
    )
    state.update(overrides)
    return state


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_pipeline_stops_at_hitl_checkpoint():
    """Le pipeline s'arrête quand un agent déclenche hitl_pending=True."""
    state = _make_state()

    patches = {**_BASE_PATCHES}
    # Insight avec faible confiance → HITL cp3_insights déclenché
    patches["app.agents.insight_agent.call_llm_json"] = AsyncMock(
        return_value=_LLM_INSIGHTS_LOW_CONFIDENCE
    )

    with (
        patch(
            "app.agents.metadata_agent.read_dataframe",
            patches["app.agents.metadata_agent.read_dataframe"],
        ),
        patch(
            "app.agents.metadata_agent.call_llm_json",
            patches["app.agents.metadata_agent.call_llm_json"],
        ),
        patch(
            "app.agents.schema_linking_agent.read_dataframe",
            patches["app.agents.schema_linking_agent.read_dataframe"],
        ),
        patch(
            "app.agents.schema_linking_agent.call_llm_json",
            patches["app.agents.schema_linking_agent.call_llm_json"],
        ),
        patch(
            "app.agents.data_agent.read_dataframe", patches["app.agents.data_agent.read_dataframe"]
        ),
        patch(
            "app.agents.data_agent.call_llm_json", patches["app.agents.data_agent.call_llm_json"]
        ),
        patch("app.agents.data_agent.get_cache", patches["app.agents.data_agent.get_cache"]),
        patch("app.agents.data_agent.set_cache", patches["app.agents.data_agent.set_cache"]),
        patch(
            "app.agents.insight_agent.call_llm_json",
            patches["app.agents.insight_agent.call_llm_json"],
        ),
        patch(
            "app.agents.base_agent.save_report_state",
            patches["app.agents.base_agent.save_report_state"],
        ),
        patch(
            "app.agents.base_agent.notify_hitl_required",
            patches["app.agents.base_agent.notify_hitl_required"],
        ),
    ):
        pipeline = build_pipeline()
        result = await pipeline.ainvoke(state)

    assert result["hitl_pending"] is True, f"HITL non déclenché: {result}"
    assert result["hitl_checkpoint"] == "cp3_insights", (
        f"Checkpoint attendu cp3_insights, obtenu: {result.get('hitl_checkpoint')}"
    )
    assert result["status"] == "hitl_required", f"Status: {result['status']}"


@pytest.mark.asyncio
async def test_resume_pipeline_cp3_completes_report():
    """resume_pipeline depuis cp3_insights → pipeline reprend et termine le rapport."""
    # State simulant un rapport arrêté en cp3_insights avec corrections
    state = _make_state(
        status="running",
        hitl_pending=False,
        hitl_checkpoint=None,
        metadata={"files": {"ventes.csv": {"columns": {}}}},
        schema={"relations": [], "alerts": []},
        aggregates=_CACHED_AGGREGATES,
        insights=_LLM_INSIGHTS_HIGH_CONFIDENCE["insights"],  # insights corrigés
        narrative="",
        viz_specs=[],
        qa_report={},
        report_urls={},
        errors=[],
    )

    with (
        patch(
            "app.agents.storytelling_agent.call_llm_json",
            AsyncMock(return_value=_STORYTELLING_REPORT_RESPONSE),
        ),
        patch(
            "app.agents.viz_agent.call_llm_json",
            AsyncMock(return_value={"viz_specs": [_MOCK_VIZ_SPEC]}),
        ),
        patch("app.agents.qa_agent.call_llm_json", AsyncMock(return_value=_MOCK_QA_LLM_RESPONSE)),
        patch("app.agents.layout_agent.upload_file", AsyncMock()),
        patch("app.agents.base_agent.save_report_state", AsyncMock()),
        patch("app.agents.base_agent.notify_hitl_required", AsyncMock()),
    ):
        pipeline = resume_pipeline("test-hitl-001", {**state, "hitl_checkpoint": "cp3_insights"})
        result = await pipeline.ainvoke(state)

    assert result["status"] == "complete", (
        f"Status attendu 'complete', obtenu '{result['status']}'. Errors: {result.get('errors')}"
    )
    assert result["report_urls"].get("html_url"), "html_url absent après reprise"
    assert result["narrative"], "Narrative vide après reprise"
    assert result["hitl_pending"] is False, "hitl_pending devrait être False après reprise"


@pytest.mark.asyncio
async def test_resume_pipeline_cp2_schema_always_invokes_insight_agent():
    """resume_pipeline depuis cp2_schema (entrée data_agent), quel que soit l'intent, doit
    invoquer InsightAgent — le insight/narrative doit apparaître sur tout prompt, y compris
    simple_query (cf. graph.py::build_pipeline, data_agent → insight_agent inconditionnel).

    Avant ce changement, intent=simple_query court-circuitait insight/storytelling/viz/qa
    pour aller direct au layout ; la demande utilisateur explicite est que l'insight
    apparaisse pour tous les prompts, ce test valide donc l'inverse de l'ancien comportement.
    """
    state = _make_state(
        intent="simple_query",
        response_type="table",
        status="running",
        hitl_pending=False,
        hitl_checkpoint=None,
        metadata={"files": {"ventes.csv": {"columns": {}}}},
        schema={"relations": [], "alerts": []},
        aggregates={},
        insights=[],
        narrative="",
        viz_specs=[],
        qa_report={},
        report_urls={},
        errors=[],
    )

    original_insight_run = InsightAgent.run
    insight_calls = {"count": 0}

    async def _spy_insight_run(self, s):
        insight_calls["count"] += 1
        return await original_insight_run(self, s)

    with (
        patch("app.agents.data_agent.read_dataframe", AsyncMock(return_value=_MOCK_DF)),
        patch(
            "app.agents.data_agent.call_llm_json",
            AsyncMock(
                return_value={
                    "queries": [
                        {
                            "key": "by_region",
                            "sql": "SELECT region, SUM(ca_ht) AS ca_ht FROM ventes GROUP BY region",
                        }
                    ]
                }
            ),
        ),
        patch("app.agents.data_agent.get_cache", AsyncMock(return_value=None)),
        patch("app.agents.data_agent.set_cache", AsyncMock()),
        patch("app.agents.insight_agent.call_llm_json", AsyncMock(return_value=_LLM_INSIGHTS_HIGH_CONFIDENCE)),
        patch(
            # response_type="table" pour ce test → StorytellingAgent prend la branche
            # {title, summary} (pas {title, executive_summary} du mode "report").
            "app.agents.storytelling_agent.call_llm_json",
            AsyncMock(return_value={"title": "CA par région", "summary": "L'IDF domine le CA régional."}),
        ),
        patch("app.agents.viz_agent.call_llm_json", AsyncMock(return_value={"viz_specs": [_MOCK_VIZ_SPEC]})),
        patch("app.agents.qa_agent.call_llm_json", AsyncMock(return_value=_MOCK_QA_LLM_RESPONSE)),
        patch("app.agents.layout_agent.upload_file", AsyncMock()),
        patch("app.agents.base_agent.save_report_state", AsyncMock()),
        patch("app.agents.base_agent.notify_hitl_required", AsyncMock()),
        patch.object(InsightAgent, "run", _spy_insight_run),
    ):
        pipeline = resume_pipeline("test-hitl-001", {**state, "hitl_checkpoint": "cp2_schema"})
        result = await pipeline.ainvoke(state)

    assert insight_calls["count"] == 1, (
        f"InsightAgent aurait dû être invoqué exactement une fois, appelé {insight_calls['count']} fois"
    )
    assert result["status"] == "complete", f"Errors: {result.get('errors')}"
    assert result["narrative"], "narrative vide — le insight/narrative doit apparaître même pour simple_query"
    assert result["response_type"] == "table"
    assert result["response"], "state['response'] non écrit par LayoutAgent (mode table)"
    assert result["report_urls"] == {}, "Pas d'upload HTML attendu en mode table"


@pytest.mark.asyncio
async def test_resume_pipeline_invalid_checkpoint_raises():
    """resume_pipeline avec checkpoint inconnu → ValueError."""
    state = _make_state(hitl_checkpoint="cp99_unknown")

    with pytest.raises(ValueError, match="checkpoint inconnu"):
        resume_pipeline("test-hitl-001", state)


@pytest.mark.asyncio
async def test_resume_pipeline_missing_checkpoint_raises():
    """resume_pipeline sans hitl_checkpoint → ValueError."""
    state = _make_state()
    state["hitl_checkpoint"] = None

    with pytest.raises(ValueError, match="hitl_checkpoint manquant"):
        resume_pipeline("test-hitl-001", state)


# ── setup_only : la reprise HITL pendant la connexion initiale doit s'arrêter avant
# data_agent/insight_agent, pas enchaîner l'analyse complète sur le prompt générique de
# connexion (régression réelle — CP1/CP2 approuvés déclenchaient un CP3 avant la première
# vraie question de l'utilisateur) ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_resume_pipeline_cp1_csv_setup_only_stops_at_setup_complete():
    """Reprise cp1_metadata (mode csv, setup_only=True) : schema_linking_agent tourne
    (relations nécessaires), mais le pipeline s'arrête ensuite — data_agent n'est jamais
    invoqué, state['response'] == {"type": "setup_complete"}."""
    state = _make_state(
        setup_only=True,
        status="running",
        hitl_pending=False,
        hitl_checkpoint=None,
        metadata={"files": {"ventes.csv": {"columns": {}}}},
    )

    with (
        patch("app.agents.schema_linking_agent.read_dataframe", _BASE_PATCHES["app.agents.schema_linking_agent.read_dataframe"]),
        patch("app.agents.schema_linking_agent.call_llm_json", _BASE_PATCHES["app.agents.schema_linking_agent.call_llm_json"]),
        patch("app.agents.base_agent.save_report_state", AsyncMock()),
        patch("app.agents.base_agent.notify_hitl_required", AsyncMock()),
    ):
        pipeline = resume_pipeline("test-hitl-001", {**state, "hitl_checkpoint": "cp1_metadata"})
        result = await pipeline.ainvoke(state)

    assert result["response"] == {"type": "setup_complete"}
    assert result["status"] == "complete"
    assert result["aggregates"] == {}, "data_agent n'a jamais dû tourner — aggregates doit rester vide"


@pytest.mark.asyncio
async def test_resume_pipeline_cp2_csv_setup_only_stops_immediately():
    """Reprise cp2_schema (mode csv, setup_only=True) : entry_node doit être redirigé
    directement vers setup_complete_node, sans jamais toucher data_agent."""
    state = _make_state(
        setup_only=True,
        status="running",
        hitl_pending=False,
        hitl_checkpoint=None,
        metadata={"files": {"ventes.csv": {"columns": {}}}},
        schema={"relations": [], "alerts": []},
    )

    with (
        patch("app.agents.base_agent.save_report_state", AsyncMock()),
        patch("app.agents.base_agent.notify_hitl_required", AsyncMock()),
    ):
        pipeline = resume_pipeline("test-hitl-001", {**state, "hitl_checkpoint": "cp2_schema"})
        result = await pipeline.ainvoke(state)

    assert result["response"] == {"type": "setup_complete"}
    assert result["status"] == "complete"
    assert result["aggregates"] == {}


@pytest.mark.asyncio
async def test_resume_pipeline_cp1_powerbi_setup_only_stops_immediately():
    """Reprise cp1_metadata (mode powerbi_local, setup_only=True) : entry_node doit être
    redirigé directement vers setup_complete_node, sans jamais toucher data_agent (qui
    referait une connexion MCP / génération DAX inutile sur le prompt de connexion)."""
    state = _make_state(
        data_source="powerbi_local",
        pbix_file_name="AdventureWorks",
        setup_only=True,
        status="running",
        hitl_pending=False,
        hitl_checkpoint=None,
        metadata={"files": {"powerbi://Sales": {"columns": {}}}},
        semantic_model_info={"tables": {}, "measures": {}, "relations": []},
    )

    with (
        patch("app.agents.base_agent.save_report_state", AsyncMock()),
        patch("app.agents.base_agent.notify_hitl_required", AsyncMock()),
    ):
        pipeline = resume_pipeline("test-hitl-001", {**state, "hitl_checkpoint": "cp1_metadata"})
        result = await pipeline.ainvoke(state)

    assert result["response"] == {"type": "setup_complete"}
    assert result["status"] == "complete"
    assert result["aggregates"] == {}
