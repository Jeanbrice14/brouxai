"""Tests API Sprint 8 — endpoint GET /health."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient


@pytest.mark.asyncio
async def test_health_returns_all_components():
    """GET /health → 200 avec les 3 composants : pipeline, redis, storage."""
    from app.main import app

    with (
        patch("app.main._check_redis", AsyncMock(return_value="ok")),
        patch("app.main._check_storage", AsyncMock(return_value="ok")),
        patch("app.main._pipeline", new=object()),  # pipeline non-None
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/health")

    assert response.status_code == 200, f"Réponse: {response.text}"
    body = response.json()

    assert "status" in body, "Champ 'status' manquant"
    assert "pipeline" in body, "Champ 'pipeline' manquant"
    assert "redis" in body, "Champ 'redis' manquant"
    assert "storage" in body, "Champ 'storage' manquant"

    assert body["status"] == "ok"


@pytest.mark.asyncio
async def test_health_reports_redis_error():
    """GET /health → redis: 'error' quand Redis indisponible."""
    from app.main import app

    with (
        patch("app.main._check_redis", AsyncMock(return_value="error")),
        patch("app.main._check_storage", AsyncMock(return_value="ok")),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["redis"] == "error", f"redis devrait être 'error': {body}"
    assert body["status"] == "ok", "status global reste 'ok' même si Redis down"


@pytest.mark.asyncio
async def test_health_reports_storage_error():
    """GET /health → storage: 'error' quand MinIO indisponible."""
    from app.main import app

    with (
        patch("app.main._check_redis", AsyncMock(return_value="ok")),
        patch("app.main._check_storage", AsyncMock(return_value="error")),
    ):
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["storage"] == "error", f"storage devrait être 'error': {body}"
