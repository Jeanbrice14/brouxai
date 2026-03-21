"""Tests API — endpoints /api/v1/reports.

Pipeline, storage et Redis sont mockés.
POST /{id}/review est testé dans test_hitl_api.py (Sprint 9).
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

# ── Helpers ───────────────────────────────────────────────────────────────────

_VALID_PROMPT = "Analyse les ventes par région pour Q1 2024"

_MOCK_STATE_RUNNING = {
    "report_id": "test-report-001",
    "tenant_id": "demo-tenant",
    "user_id": "demo-user",
    "prompt": _VALID_PROMPT,
    "status": "running",
    "current_agent": "metadata_agent",
    "hitl_pending": False,
    "hitl_checkpoint": None,
    "hitl_corrections": [],
    "raw_data_refs": ["s3://narr8-dev/demo-tenant/datasets/test-report-001/ventes.csv"],
    "brand_kit": {},
    "metadata": {},
    "schema": {},
    "aggregates": {},
    "insights": [],
    "narrative": "",
    "viz_specs": [],
    "qa_report": {},
    "report_urls": {},
    "errors": [],
    "created_at": datetime.now(tz=UTC).isoformat(),
}

_MOCK_STATE_HITL = {
    **_MOCK_STATE_RUNNING,
    "hitl_pending": True,
    "hitl_checkpoint": "cp3_insights",
    "status": "hitl_required",
}

_MOCK_STATE_COMPLETE = {
    **_MOCK_STATE_RUNNING,
    "status": "complete",
    "report_urls": {
        "html_url": "http://localhost:9000/narr8-dev/demo-tenant/reports/test-report-001/report.html"
    },
}


def _make_csv_bytes() -> bytes:
    return b"date,region,ca_ht\n2024-01-01,Nord,12500\n2024-01-02,IDF,42100\n"


# ── Fixture client ─────────────────────────────────────────────────────────────


@pytest.fixture
def mock_pipeline():
    """Pipeline mocké qui ne fait rien (background task)."""
    pipeline = MagicMock()
    pipeline.ainvoke = AsyncMock(return_value={})
    return pipeline


@pytest.fixture
def app_client(mock_pipeline):
    """Client HTTP asynchrone avec pipeline + storage + Redis mockés."""
    from app.main import app

    with (
        patch("app.main.get_pipeline", return_value=mock_pipeline),
        patch("app.api.v1.reports._get_pipeline", return_value=mock_pipeline),
        patch("app.api.v1.reports.upload_file", AsyncMock()),
        patch("app.api.v1.reports.save_report_state", AsyncMock()),
        patch("app.api.v1.reports.get_report_state", AsyncMock(return_value=None)),
    ):
        yield app


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_generate_report_returns_report_id():
    """POST /generate → 200 avec report_id (UUID valide) et status 'running'."""
    from app.main import app

    def _fake_create_task(coro):
        """Consomme la coroutine sans la planifier pour éviter les warnings."""
        coro.close()
        return MagicMock()

    with (
        patch("app.api.v1.reports._get_pipeline", return_value=MagicMock(ainvoke=AsyncMock())),
        patch("app.api.v1.reports.upload_file", AsyncMock()),
        patch("app.api.v1.reports.save_report_state", AsyncMock()),
        patch("app.api.v1.reports.asyncio.create_task", side_effect=_fake_create_task),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/v1/reports/generate",
                data={"prompt": _VALID_PROMPT, "brand_kit": "{}"},
                files={"files": ("ventes.csv", _make_csv_bytes(), "text/csv")},
            )

    assert response.status_code == 200, f"Réponse: {response.text}"
    body = response.json()
    assert "report_id" in body, f"report_id absent: {body}"
    assert body["status"] == "running"

    # Vérifier que report_id est un UUID valide
    try:
        uuid.UUID(body["report_id"])
    except ValueError:
        pytest.fail(f"report_id n'est pas un UUID valide: {body['report_id']}")


@pytest.mark.asyncio
async def test_get_report_returns_state():
    """GET /{report_id} → 200 avec les champs ReportResponse."""
    from app.main import app

    with patch(
        "app.api.v1.reports.get_report_state",
        AsyncMock(return_value=_MOCK_STATE_RUNNING),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/v1/reports/test-report-001")

    assert response.status_code == 200, f"Réponse: {response.text}"
    body = response.json()

    assert body["report_id"] == "test-report-001"
    assert body["status"] == "running"
    assert body["prompt"] == _VALID_PROMPT
    assert "created_at" in body


@pytest.mark.asyncio
async def test_get_unknown_report_returns_404():
    """GET /id-inexistant → 404."""
    from app.main import app

    with patch(
        "app.api.v1.reports.get_report_state",
        AsyncMock(return_value=None),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/v1/reports/id-inexistant")

    assert response.status_code == 404, f"Réponse: {response.text}"


@pytest.mark.asyncio
async def test_generate_rejects_invalid_prompt():
    """POST /generate avec prompt < 10 chars → 422."""
    from app.main import app

    with (
        patch("app.api.v1.reports.upload_file", AsyncMock()),
        patch("app.api.v1.reports.save_report_state", AsyncMock()),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/v1/reports/generate",
                data={"prompt": "abc", "brand_kit": "{}"},
                files={"files": ("ventes.csv", _make_csv_bytes(), "text/csv")},
            )

    assert response.status_code == 422, (
        f"Attendu 422 pour prompt trop court, obtenu {response.status_code}: {response.text}"
    )
