"""Tests de l'indexation de sessions (sidebar historique) dans report_store.py."""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.services.report_store import list_sessions, save_report_state


class _FakeRedis:
    """Redis async minimal en mémoire — couvre les commandes utilisées par report_store."""

    def __init__(self) -> None:
        self.strings: dict[str, str] = {}
        self.hashes: dict[str, dict[str, str]] = {}
        self.sets: dict[str, set[str]] = {}
        self.zsets: dict[str, dict[str, float]] = {}

    async def ping(self) -> bool:
        return True

    async def set(self, key: str, value: str, ex: int | None = None) -> None:
        self.strings[key] = value

    async def get(self, key: str) -> str | None:
        return self.strings.get(key)

    async def hsetnx(self, key: str, field: str, value: str) -> None:
        self.hashes.setdefault(key, {})
        if field not in self.hashes[key]:
            self.hashes[key][field] = value

    async def hset(self, key: str, mapping: dict) -> None:
        self.hashes.setdefault(key, {}).update(mapping)

    async def hgetall(self, key: str) -> dict:
        return dict(self.hashes.get(key, {}))

    async def sadd(self, key: str, member: str) -> None:
        self.sets.setdefault(key, set()).add(member)

    async def scard(self, key: str) -> int:
        return len(self.sets.get(key, set()))

    async def zadd(self, key: str, mapping: dict) -> None:
        self.zsets.setdefault(key, {}).update(mapping)

    async def zrevrange(self, key: str, start: int, stop: int) -> list[str]:
        items = sorted(self.zsets.get(key, {}).items(), key=lambda kv: kv[1], reverse=True)
        end = None if stop == -1 else stop + 1
        return [member for member, _ in items[start:end]]

    async def expire(self, key: str, ttl: int) -> None:
        pass

    async def aclose(self) -> None:
        pass


@pytest.mark.asyncio
async def test_save_report_state_indexes_session_for_list_sessions():
    """Sauver un état avec tenant_id/session_id doit le rendre visible via list_sessions."""
    fake = _FakeRedis()
    with patch("app.services.report_store._get_client", AsyncMock(return_value=fake)):
        await save_report_state(
            "report-1",
            {
                "tenant_id": "tenant-a",
                "session_id": "session-1",
                "prompt": "Analyse et modélise ces données",
                "status": "running",
            },
        )

        sessions = await list_sessions("tenant-a")

    assert len(sessions) == 1
    assert sessions[0]["session_id"] == "session-1"
    assert sessions[0]["title"] == "Analyse et modélise ces données"
    assert sessions[0]["last_prompt"] == "Analyse et modélise ces données"
    assert sessions[0]["last_status"] == "running"
    assert sessions[0]["report_count"] == 1


@pytest.mark.asyncio
async def test_save_report_state_preserves_first_prompt_as_title():
    """Le titre (first_prompt) ne doit pas changer sur les messages suivants de la session,
    même si last_prompt et report_count sont mis à jour."""
    fake = _FakeRedis()
    with patch("app.services.report_store._get_client", AsyncMock(return_value=fake)):
        await save_report_state(
            "report-1",
            {"tenant_id": "tenant-a", "session_id": "session-1", "prompt": "Question initiale", "status": "complete"},
        )
        await save_report_state(
            "report-2",
            {"tenant_id": "tenant-a", "session_id": "session-1", "prompt": "Question suivante", "status": "running"},
        )

        sessions = await list_sessions("tenant-a")

    assert len(sessions) == 1
    assert sessions[0]["title"] == "Question initiale"
    assert sessions[0]["last_prompt"] == "Question suivante"
    assert sessions[0]["last_status"] == "running"
    assert sessions[0]["report_count"] == 2


@pytest.mark.asyncio
async def test_save_report_state_resaving_same_report_id_does_not_inflate_count():
    """Un même report_id sauvé plusieurs fois (création + reprise HITL + erreur) ne doit
    compter que pour un seul rapport — SADD est idempotent."""
    fake = _FakeRedis()
    with patch("app.services.report_store._get_client", AsyncMock(return_value=fake)):
        state = {"tenant_id": "tenant-a", "session_id": "session-1", "prompt": "Q", "status": "running"}
        await save_report_state("report-1", state)
        state["status"] = "hitl_required"
        await save_report_state("report-1", state)
        state["status"] = "complete"
        await save_report_state("report-1", state)

        sessions = await list_sessions("tenant-a")

    assert sessions[0]["report_count"] == 1
    assert sessions[0]["last_status"] == "complete"


@pytest.mark.asyncio
async def test_list_sessions_sorted_by_recent_activity():
    """Les sessions doivent être triées par activité la plus récente en premier.

    Horodatage figé (pas datetime.now() réel) : deux sauvegardes back-to-back peuvent
    tomber sur le même timestamp au tick d'horloge près et rendre le tri non déterministe.
    """
    fake = _FakeRedis()
    from datetime import datetime as real_datetime
    from datetime import timezone as real_timezone

    t1 = real_datetime(2026, 1, 1, 10, 0, 0, tzinfo=real_timezone.utc)
    t2 = real_datetime(2026, 1, 1, 10, 5, 0, tzinfo=real_timezone.utc)

    with (
        patch("app.services.report_store._get_client", AsyncMock(return_value=fake)),
        patch("app.services.report_store.datetime") as mock_dt,
    ):
        mock_dt.now.side_effect = [t1, t2]
        await save_report_state(
            "report-1", {"tenant_id": "tenant-a", "session_id": "session-old", "prompt": "Ancienne", "status": "complete"}
        )
        await save_report_state(
            "report-2", {"tenant_id": "tenant-a", "session_id": "session-new", "prompt": "Récente", "status": "complete"}
        )

        sessions = await list_sessions("tenant-a")

    assert [s["session_id"] for s in sessions] == ["session-new", "session-old"]


@pytest.mark.asyncio
async def test_list_sessions_scoped_to_tenant():
    """Les sessions d'un autre tenant ne doivent jamais apparaître."""
    fake = _FakeRedis()
    with patch("app.services.report_store._get_client", AsyncMock(return_value=fake)):
        await save_report_state(
            "report-1", {"tenant_id": "tenant-a", "session_id": "session-1", "prompt": "Q", "status": "complete"}
        )
        await save_report_state(
            "report-2", {"tenant_id": "tenant-b", "session_id": "session-2", "prompt": "Q", "status": "complete"}
        )

        sessions_a = await list_sessions("tenant-a")
        sessions_b = await list_sessions("tenant-b")

    assert [s["session_id"] for s in sessions_a] == ["session-1"]
    assert [s["session_id"] for s in sessions_b] == ["session-2"]


@pytest.mark.asyncio
async def test_save_report_state_without_session_id_does_not_crash():
    """Un état sans session_id/tenant_id (état transitoire) ne doit jamais faire planter
    la sauvegarde — l'indexation est simplement ignorée."""
    fake = _FakeRedis()
    with patch("app.services.report_store._get_client", AsyncMock(return_value=fake)):
        await save_report_state("report-1", {"prompt": "Q", "status": "running"})

    assert fake.zsets == {}


@pytest.mark.asyncio
async def test_list_sessions_redis_unavailable_returns_empty():
    """Si Redis est indisponible, list_sessions retourne [] plutôt que de planter."""
    with patch("app.services.report_store._get_client", AsyncMock(return_value=None)):
        sessions = await list_sessions("tenant-a")

    assert sessions == []
