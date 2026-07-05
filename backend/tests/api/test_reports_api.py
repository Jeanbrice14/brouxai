"""Tests API — endpoints /api/v1/reports.

Pipeline, storage et Redis sont mockés.
POST /{id}/review est testé dans test_hitl_api.py (Sprint 9).
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

UTC = timezone.utc

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


# ── generate-powerbi : réutilisation de metadata via base_report_id ────────────
# (évite de redéclencher CP1 à chaque message — cf. graph.py::_route_after_intent)


@pytest.mark.asyncio
async def test_generate_powerbi_reuses_metadata_from_base_report_id():
    """base_report_id fourni + metadata/semantic_model_info présents dans le rapport de
    base → l'état initial du nouveau rapport doit les reprendre tels quels (CP1 ne doit
    pas se redéclencher, cf. _route_after_intent qui vérifie state['metadata'])."""
    from app.main import app

    base_metadata = {"files": {"powerbi://Sales": {"columns": {"Region": {"confidence": 0.6}}}}}
    base_semantic_model_info = {"tables": {"Sales": {}}, "measures": {}, "relations": []}
    base_state = {
        "report_id": "base-report-001",
        "metadata": base_metadata,
        "semantic_model_info": base_semantic_model_info,
        "schema": {},
    }

    captured_state: dict = {}

    async def _fake_save_report_state(report_id, state):
        captured_state.update(state)

    def _fake_create_task(coro):
        coro.close()
        return MagicMock()

    with (
        patch("app.api.v1.reports.get_report_state", AsyncMock(return_value=base_state)),
        patch("app.api.v1.reports.save_report_state", AsyncMock(side_effect=_fake_save_report_state)),
        patch("app.api.v1.reports._get_pipeline", return_value=MagicMock(ainvoke=AsyncMock())),
        patch("app.api.v1.reports.asyncio.create_task", side_effect=_fake_create_task),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/v1/reports/generate-powerbi",
                data={
                    "prompt": "Quelles sont les ventes totales par catégorie ?",
                    "pbix_file_name": "AdventureWorks",
                    "brand_kit": "{}",
                    "session_id": "session-reuse-test",
                    "base_report_id": "base-report-001",
                },
            )

    assert response.status_code == 200, f"Réponse: {response.text}"
    assert captured_state.get("metadata") == base_metadata
    assert captured_state.get("semantic_model_info") == base_semantic_model_info


@pytest.mark.asyncio
async def test_generate_powerbi_without_base_report_id_starts_with_empty_metadata():
    """Sans base_report_id (premier appel de la session) : metadata reste vide — c'est
    le comportement attendu, metadata_agent doit tourner pour construire le Data Dictionary."""
    from app.main import app

    captured_state: dict = {}

    async def _fake_save_report_state(report_id, state):
        captured_state.update(state)

    def _fake_create_task(coro):
        coro.close()
        return MagicMock()

    with (
        patch("app.api.v1.reports.save_report_state", AsyncMock(side_effect=_fake_save_report_state)),
        patch("app.api.v1.reports._get_pipeline", return_value=MagicMock(ainvoke=AsyncMock())),
        patch("app.api.v1.reports.asyncio.create_task", side_effect=_fake_create_task),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/v1/reports/generate-powerbi",
                data={
                    "prompt": "Connecte-toi au modèle et analyse le schéma",
                    "pbix_file_name": "AdventureWorks",
                    "brand_kit": "{}",
                },
            )

    assert response.status_code == 200, f"Réponse: {response.text}"
    assert captured_state.get("metadata") == {}
    assert captured_state.get("semantic_model_info") == {}


# ── setup_only : connexion initiale sans vraie question ────────────────────────
# (régression : le champ était envoyé par le frontend mais jamais lu côté backend)


@pytest.mark.asyncio
async def test_generate_powerbi_parses_setup_only_true():
    """setup_only="true" (form-data, toujours une string) doit devenir state["setup_only"] = True."""
    from app.main import app

    captured_state: dict = {}

    async def _fake_save_report_state(report_id, state):
        captured_state.update(state)

    def _fake_create_task(coro):
        coro.close()
        return MagicMock()

    with (
        patch("app.api.v1.reports.save_report_state", AsyncMock(side_effect=_fake_save_report_state)),
        patch("app.api.v1.reports._get_pipeline", return_value=MagicMock(ainvoke=AsyncMock())),
        patch("app.api.v1.reports.asyncio.create_task", side_effect=_fake_create_task),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/v1/reports/generate-powerbi",
                data={
                    "prompt": "Connecte-toi au modèle et analyse le schéma",
                    "pbix_file_name": "AdventureWorks",
                    "brand_kit": "{}",
                    "setup_only": "true",
                },
            )

    assert response.status_code == 200, f"Réponse: {response.text}"
    assert captured_state.get("setup_only") is True


@pytest.mark.asyncio
async def test_generate_powerbi_without_setup_only_defaults_false():
    """Sans le champ setup_only (messages normaux) : state["setup_only"] doit rester False."""
    from app.main import app

    captured_state: dict = {}

    async def _fake_save_report_state(report_id, state):
        captured_state.update(state)

    def _fake_create_task(coro):
        coro.close()
        return MagicMock()

    with (
        patch("app.api.v1.reports.save_report_state", AsyncMock(side_effect=_fake_save_report_state)),
        patch("app.api.v1.reports._get_pipeline", return_value=MagicMock(ainvoke=AsyncMock())),
        patch("app.api.v1.reports.asyncio.create_task", side_effect=_fake_create_task),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/v1/reports/generate-powerbi",
                data={
                    "prompt": "Quel est le total des ventes par région ?",
                    "pbix_file_name": "AdventureWorks",
                    "brand_kit": "{}",
                },
            )

    assert response.status_code == 200, f"Réponse: {response.text}"
    assert captured_state.get("setup_only") is False


@pytest.mark.asyncio
async def test_generate_report_csv_parses_setup_only_true():
    """Idem pour /reports/generate (mode CSV) — le frontend envoie déjà ce champ pour
    l'upload initial (app/page.tsx), il doit maintenant être effectivement lu."""
    from app.main import app

    captured_state: dict = {}

    async def _fake_save_report_state(report_id, state):
        captured_state.update(state)

    def _fake_create_task(coro):
        coro.close()
        return MagicMock()

    with (
        patch("app.api.v1.reports.upload_file", AsyncMock()),
        patch("app.api.v1.reports.save_report_state", AsyncMock(side_effect=_fake_save_report_state)),
        patch("app.api.v1.reports._get_pipeline", return_value=MagicMock(ainvoke=AsyncMock())),
        patch("app.api.v1.reports.asyncio.create_task", side_effect=_fake_create_task),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/v1/reports/generate",
                data={"prompt": "Analyse et modélise ces données", "brand_kit": "{}", "setup_only": "true"},
                files={"files": ("ventes.csv", _make_csv_bytes(), "text/csv")},
            )

    assert response.status_code == 200, f"Réponse: {response.text}"
    assert captured_state.get("setup_only") is True


# ── Mémoire de conversation (k derniers tours) ─────────────────────────────────


@pytest.mark.asyncio
async def test_generate_powerbi_populates_chat_history_from_conversation_memory():
    """state["chat_history"] doit être peuplé depuis get_conversation_history() à la
    création du state — c'est ce qui permet à DataAgent de résoudre des références comme
    "cette catégorie" d'un message au suivant."""
    from app.main import app

    captured_state: dict = {}
    fake_history = [{"question": "Quelle catégorie a le plus de ventes ?", "answer_summary": "Bikes domine."}]

    async def _fake_save_report_state(report_id, state):
        captured_state.update(state)

    def _fake_create_task(coro):
        coro.close()
        return MagicMock()

    with (
        patch("app.api.v1.reports.save_report_state", AsyncMock(side_effect=_fake_save_report_state)),
        patch("app.api.v1.reports.get_conversation_history", AsyncMock(return_value=fake_history)),
        patch("app.api.v1.reports._get_pipeline", return_value=MagicMock(ainvoke=AsyncMock())),
        patch("app.api.v1.reports.asyncio.create_task", side_effect=_fake_create_task),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/v1/reports/generate-powerbi",
                data={
                    "prompt": "Quel est le taux de retour pour cette catégorie ?",
                    "pbix_file_name": "AdventureWorks",
                    "brand_kit": "{}",
                    "session_id": "session-memory-test",
                },
            )

    assert response.status_code == 200, f"Réponse: {response.text}"
    assert captured_state.get("chat_history") == fake_history


@pytest.mark.asyncio
async def test_run_pipeline_records_turn_after_completion():
    """_run_pipeline doit appeler maybe_record_turn() une fois le pipeline terminé, pour
    que le tour courant devienne disponible aux messages suivants de la session."""
    from app.api.v1.reports import _run_pipeline

    final_state = {
        "status": "complete",
        "setup_only": False,
        "narrative": "Bikes domine avec 23.6M€.",
        "tenant_id": "tenant-test",
        "session_id": "session-test",
        "prompt": "Quelle catégorie a le plus de ventes ?",
    }
    pipeline = MagicMock(ainvoke=AsyncMock(return_value=final_state))
    mock_record = AsyncMock()

    with (
        patch("app.api.v1.reports.save_report_state", AsyncMock()),
        patch("app.api.v1.reports.maybe_record_turn", mock_record),
    ):
        await _run_pipeline(pipeline, {"report_id": "report-test"}, "report-test")

    mock_record.assert_awaited_once_with(final_state)
