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

_LONG_NARRATIVE = (
    "L'analyse des ventes régionales révèle des disparités marquées.\n\n"
    "L'Île-de-France domine avec la plus grande part du chiffre d'affaires total. "
    "Cette concentration géographique appelle une attention particulière pour "
    "diversifier les sources de revenus et renforcer la présence dans d'autres régions. "
    "Les équipes commerciales devraient prioriser le développement en région Nord et Sud."
    "\n\nEn conclusion, le portefeuille régional présente un fort potentiel de croissance."
)

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
        return_value={"code": "result = df.groupby('region')['ca_ht'].sum().reset_index().to_dict('records')"}
    ),
    "app.agents.data_agent.get_cache": AsyncMock(return_value=_CACHED_AGGREGATES),
    "app.agents.data_agent.set_cache": AsyncMock(),
    "app.agents.storytelling_agent.call_llm": AsyncMock(return_value=_LONG_NARRATIVE),
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
        prompt="Analyse les ventes par région pour valider le HITL",
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
        patch("app.agents.metadata_agent.read_dataframe", patches["app.agents.metadata_agent.read_dataframe"]),
        patch("app.agents.metadata_agent.call_llm_json", patches["app.agents.metadata_agent.call_llm_json"]),
        patch("app.agents.schema_linking_agent.read_dataframe", patches["app.agents.schema_linking_agent.read_dataframe"]),
        patch("app.agents.schema_linking_agent.call_llm_json", patches["app.agents.schema_linking_agent.call_llm_json"]),
        patch("app.agents.data_agent.read_dataframe", patches["app.agents.data_agent.read_dataframe"]),
        patch("app.agents.data_agent.call_llm_json", patches["app.agents.data_agent.call_llm_json"]),
        patch("app.agents.data_agent.get_cache", patches["app.agents.data_agent.get_cache"]),
        patch("app.agents.data_agent.set_cache", patches["app.agents.data_agent.set_cache"]),
        patch("app.agents.insight_agent.call_llm_json", patches["app.agents.insight_agent.call_llm_json"]),
        patch("app.agents.base_agent.save_report_state", patches["app.agents.base_agent.save_report_state"]),
        patch("app.agents.base_agent.notify_hitl_required", patches["app.agents.base_agent.notify_hitl_required"]),
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
        metadata={"files": [{"filename": "ventes.csv", "columns": []}]},
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
        patch("app.agents.storytelling_agent.call_llm", AsyncMock(return_value=_LONG_NARRATIVE)),
        patch("app.agents.viz_agent.call_llm_json", AsyncMock(return_value={"viz_specs": [_MOCK_VIZ_SPEC]})),
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
