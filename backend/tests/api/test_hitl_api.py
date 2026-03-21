"""Tests API Sprint 9 — endpoints HITL /api/v1/reports/{id}/review.

Pipeline, storage et Redis sont mockés.
"""

from __future__ import annotations

import copy
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

# ── Fixtures de state ─────────────────────────────────────────────────────────

_MOCK_STATE_HITL = {
    "report_id": "test-report-001",
    "tenant_id": "demo-tenant",
    "user_id": "demo-user",
    "prompt": "Analyse les ventes par région pour Q1 2024",
    "status": "hitl_required",
    "current_agent": "metadata_agent",
    "hitl_pending": True,
    "hitl_checkpoint": "cp3_insights",
    "hitl_corrections": [],
    "raw_data_refs": ["s3://narr8-dev/demo-tenant/datasets/test-report-001/ventes.csv"],
    "brand_kit": {},
    "metadata": {"columns": [{"name": "ca_ht", "type": "float"}]},
    "schema": {},
    "aggregates": {"by_region": [{"region": "Nord", "ca_ht": 12500}]},
    "insights": [
        {"title": "Croissance Nord", "description": "CA Nord +23%", "confidence": 0.90}
    ],
    "narrative": "Les ventes ont progressé de 23% en région Nord.",
    "viz_specs": [],
    "qa_report": {},
    "report_urls": {},
    "errors": [],
}

_MOCK_STATE_NO_HITL = {
    **_MOCK_STATE_HITL,
    "hitl_pending": False,
    "hitl_checkpoint": None,
    "status": "running",
}


# ── Tests GET /{report_id}/review ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_review_returns_checkpoint_data():
    """GET /review → 200 avec checkpoint et data contextuels."""
    from app.main import app

    with patch("app.api.v1.hitl.get_report_state", AsyncMock(return_value=copy.deepcopy(_MOCK_STATE_HITL))):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(f"/api/v1/reports/{_MOCK_STATE_HITL['report_id']}/review")

    assert response.status_code == 200, f"Réponse: {response.text}"
    body = response.json()

    assert body["report_id"] == "test-report-001"
    assert body["checkpoint"] == "cp3_insights"
    assert "data" in body
    # cp3_insights expose insights + aggregates
    assert "insights" in body["data"]
    assert "aggregates" in body["data"]


@pytest.mark.asyncio
async def test_get_review_returns_404_for_unknown_report():
    """GET /review → 404 si rapport inexistant."""
    from app.main import app

    with patch("app.api.v1.hitl.get_report_state", AsyncMock(return_value=None)):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/v1/reports/id-inexistant/review")

    assert response.status_code == 404, f"Réponse: {response.text}"


@pytest.mark.asyncio
async def test_get_review_returns_409_when_no_hitl_pending():
    """GET /review → 409 si aucun HITL en attente."""
    from app.main import app

    with patch("app.api.v1.hitl.get_report_state", AsyncMock(return_value=copy.deepcopy(_MOCK_STATE_NO_HITL))):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(f"/api/v1/reports/{_MOCK_STATE_NO_HITL['report_id']}/review")

    assert response.status_code == 409, f"Réponse: {response.text}"


# ── Tests POST /{report_id}/review ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_post_review_approved_resumes_pipeline():
    """POST /review action=approved → hitl_pending=False, status='resumed'."""
    from app.main import app

    saved_states: list[dict] = []

    def _fake_create_task(coro):
        coro.close()
        return MagicMock()

    async def _mock_save(report_id: str, state: dict) -> None:
        saved_states.append(dict(state))

    mock_pipeline = MagicMock()
    mock_pipeline.ainvoke = AsyncMock(return_value={})

    with (
        patch("app.api.v1.hitl.get_report_state", AsyncMock(return_value=copy.deepcopy(_MOCK_STATE_HITL))),
        patch("app.api.v1.hitl.save_report_state", side_effect=_mock_save),
        patch("app.api.v1.hitl.resume_pipeline", return_value=mock_pipeline),
        patch("app.api.v1.hitl.asyncio.create_task", side_effect=_fake_create_task),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                f"/api/v1/reports/{_MOCK_STATE_HITL['report_id']}/review",
                content=json.dumps(
                    {"checkpoint": "cp3_insights", "action": "approved", "corrections": {}}
                ),
                headers={"Content-Type": "application/json"},
            )

    assert response.status_code == 200, f"Réponse: {response.text}"
    body = response.json()
    assert body["status"] == "resumed"
    assert body["report_id"] == "test-report-001"

    assert saved_states, "save_report_state n'a pas été appelé"
    final_state = saved_states[-1]
    assert final_state["hitl_pending"] is False
    assert final_state["status"] == "running"


@pytest.mark.asyncio
async def test_post_review_corrected_applies_corrections():
    """POST /review action=corrected → corrections appliquées au state."""
    from app.main import app

    saved_states: list[dict] = []

    def _fake_create_task(coro):
        coro.close()
        return MagicMock()

    async def _mock_save(report_id: str, state: dict) -> None:
        saved_states.append(dict(state))

    mock_pipeline = MagicMock()
    mock_pipeline.ainvoke = AsyncMock(return_value={})

    new_insights = [{"title": "Nouveau insight", "description": "Corrigé", "confidence": 0.95}]

    with (
        patch("app.api.v1.hitl.get_report_state", AsyncMock(return_value=copy.deepcopy(_MOCK_STATE_HITL))),
        patch("app.api.v1.hitl.save_report_state", side_effect=_mock_save),
        patch("app.api.v1.hitl.resume_pipeline", return_value=mock_pipeline),
        patch("app.api.v1.hitl.asyncio.create_task", side_effect=_fake_create_task),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                f"/api/v1/reports/{_MOCK_STATE_HITL['report_id']}/review",
                content=json.dumps(
                    {
                        "checkpoint": "cp3_insights",
                        "action": "corrected",
                        "corrections": {"insights": new_insights},
                    }
                ),
                headers={"Content-Type": "application/json"},
            )

    assert response.status_code == 200, f"Réponse: {response.text}"

    assert saved_states, "save_report_state n'a pas été appelé"
    final_state = saved_states[-1]
    assert final_state["insights"] == new_insights, (
        f"Corrections non appliquées: {final_state['insights']}"
    )


# ── Tests DELETE /{report_id}/review ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_delete_review_rejects_report():
    """DELETE /review → status='rejected', rapport en erreur."""
    from app.main import app

    saved_states: list[dict] = []

    async def _mock_save(report_id: str, state: dict) -> None:
        saved_states.append(dict(state))

    with (
        patch("app.api.v1.hitl.get_report_state", AsyncMock(return_value=copy.deepcopy(_MOCK_STATE_HITL))),
        patch("app.api.v1.hitl.save_report_state", side_effect=_mock_save),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.delete(
                f"/api/v1/reports/{_MOCK_STATE_HITL['report_id']}/review"
            )

    assert response.status_code == 200, f"Réponse: {response.text}"
    body = response.json()
    assert body["status"] == "rejected"

    assert saved_states, "save_report_state n'a pas été appelé"
    final_state = saved_states[-1]
    assert final_state["status"] == "error"
    assert final_state["hitl_pending"] is False
    assert any("rejeté" in e for e in final_state.get("errors", []))
