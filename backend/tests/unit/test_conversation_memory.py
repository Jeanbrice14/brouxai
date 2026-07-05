"""Tests unitaires — app/services/conversation_memory.py.

Mémoire de conversation à k tours (question, réponse résumée) par session, pour résoudre
les références d'un tour à l'autre ("cette catégorie", "et le mois dernier ?"). Redis
mocké — aucun appel réseau réel.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.services.conversation_memory import (
    append_conversation_turn,
    format_history_for_prompt,
    get_conversation_history,
    maybe_record_turn,
    should_record_turn,
)


class _FakeRedis:
    """Redis async minimal en mémoire — couvre lpush/ltrim/lrange/expire."""

    def __init__(self) -> None:
        self.lists: dict[str, list[str]] = {}

    async def ping(self) -> bool:
        return True

    async def lpush(self, key: str, value: str) -> None:
        self.lists.setdefault(key, []).insert(0, value)

    async def ltrim(self, key: str, start: int, end: int) -> None:
        items = self.lists.get(key, [])
        self.lists[key] = items[start : end + 1] if end != -1 else items[start:]

    async def lrange(self, key: str, start: int, end: int) -> list[str]:
        items = self.lists.get(key, [])
        return items[start : end + 1] if end != -1 else items[start:]

    async def expire(self, key: str, ttl: int) -> None:
        pass

    async def aclose(self) -> None:
        pass


@pytest.mark.asyncio
async def test_append_then_get_returns_chronological_order():
    """Les tours doivent être relus du plus ancien au plus récent, pas l'inverse."""
    fake = _FakeRedis()
    with patch("app.services.conversation_memory._get_client", AsyncMock(return_value=fake)):
        await append_conversation_turn("tenant-a", "session-1", "Q1", "R1")
        await append_conversation_turn("tenant-a", "session-1", "Q2", "R2")
        await append_conversation_turn("tenant-a", "session-1", "Q3", "R3")

        history = await get_conversation_history("tenant-a", "session-1")

    assert [t["question"] for t in history] == ["Q1", "Q2", "Q3"]
    assert [t["answer_summary"] for t in history] == ["R1", "R2", "R3"]


@pytest.mark.asyncio
async def test_history_bounded_to_k():
    """Un 4e tour avec k=3 doit faire disparaître le plus ancien (Q1)."""
    fake = _FakeRedis()
    with patch("app.services.conversation_memory._get_client", AsyncMock(return_value=fake)):
        await append_conversation_turn("tenant-a", "session-1", "Q1", "R1", k=3)
        await append_conversation_turn("tenant-a", "session-1", "Q2", "R2", k=3)
        await append_conversation_turn("tenant-a", "session-1", "Q3", "R3", k=3)
        await append_conversation_turn("tenant-a", "session-1", "Q4", "R4", k=3)

        history = await get_conversation_history("tenant-a", "session-1", k=3)

    assert [t["question"] for t in history] == ["Q2", "Q3", "Q4"], (
        "Q1 aurait dû être évincé — historique borné à k=3"
    )


@pytest.mark.asyncio
async def test_history_scoped_per_session():
    """L'historique d'une session ne doit jamais fuiter vers une autre."""
    fake = _FakeRedis()
    with patch("app.services.conversation_memory._get_client", AsyncMock(return_value=fake)):
        await append_conversation_turn("tenant-a", "session-1", "Q1", "R1")
        await append_conversation_turn("tenant-a", "session-2", "Q2", "R2")

        history_1 = await get_conversation_history("tenant-a", "session-1")
        history_2 = await get_conversation_history("tenant-a", "session-2")

    assert [t["question"] for t in history_1] == ["Q1"]
    assert [t["question"] for t in history_2] == ["Q2"]


@pytest.mark.asyncio
async def test_get_conversation_history_redis_unavailable_returns_empty():
    with patch("app.services.conversation_memory._get_client", AsyncMock(return_value=None)):
        history = await get_conversation_history("tenant-a", "session-1")
    assert history == []


@pytest.mark.asyncio
async def test_get_conversation_history_missing_ids_returns_empty_without_redis_call():
    """Pas de tenant_id/session_id -> pas d'appel Redis du tout (pas juste une liste vide)."""
    mock_get_client = AsyncMock()
    with patch("app.services.conversation_memory._get_client", mock_get_client):
        assert await get_conversation_history("", "session-1") == []
        assert await get_conversation_history("tenant-a", "") == []
    mock_get_client.assert_not_called()


def test_format_history_for_prompt_empty_returns_empty_string():
    assert format_history_for_prompt([]) == ""


def test_format_history_for_prompt_includes_question_and_answer():
    history = [
        {"question": "Quelle catégorie a le plus de ventes ?", "answer_summary": "Bikes domine avec 23.6M€."},
    ]
    text = format_history_for_prompt(history)
    assert "Quelle catégorie a le plus de ventes ?" in text
    assert "Bikes domine avec 23.6M€." in text
    assert "Q1" in text and "R1" in text


def test_should_record_turn_true_for_complete_with_narrative():
    state = {"status": "complete", "setup_only": False, "narrative": "Bikes domine."}
    assert should_record_turn(state) is True


def test_should_record_turn_false_for_setup_only():
    """Le prompt générique de connexion ne doit jamais polluer la mémoire de conversation."""
    state = {"status": "complete", "setup_only": True, "narrative": "peu importe"}
    assert should_record_turn(state) is False


def test_should_record_turn_false_when_not_complete():
    state = {"status": "hitl_required", "setup_only": False, "narrative": ""}
    assert should_record_turn(state) is False


def test_should_record_turn_false_when_no_narrative():
    state = {"status": "complete", "setup_only": False, "narrative": ""}
    assert should_record_turn(state) is False


@pytest.mark.asyncio
async def test_maybe_record_turn_appends_when_relevant():
    fake = _FakeRedis()
    state = {
        "status": "complete",
        "setup_only": False,
        "narrative": "Bikes domine avec 23.6M€.",
        "tenant_id": "tenant-a",
        "session_id": "session-1",
        "prompt": "Quelle catégorie a le plus de ventes ?",
    }
    with patch("app.services.conversation_memory._get_client", AsyncMock(return_value=fake)):
        await maybe_record_turn(state)
        history = await get_conversation_history("tenant-a", "session-1")

    assert len(history) == 1
    assert history[0]["question"] == "Quelle catégorie a le plus de ventes ?"
    assert history[0]["answer_summary"] == "Bikes domine avec 23.6M€."


@pytest.mark.asyncio
async def test_maybe_record_turn_skips_setup_only():
    fake = _FakeRedis()
    state = {
        "status": "complete",
        "setup_only": True,
        "narrative": "peu importe",
        "tenant_id": "tenant-a",
        "session_id": "session-1",
        "prompt": "Connecte-toi au modèle et analyse le schéma",
    }
    with patch("app.services.conversation_memory._get_client", AsyncMock(return_value=fake)):
        await maybe_record_turn(state)
        history = await get_conversation_history("tenant-a", "session-1")

    assert history == []
