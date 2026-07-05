from __future__ import annotations

import json
from datetime import datetime, timezone

import structlog

from app.config import settings

logger = structlog.get_logger(__name__)

_REPORT_TTL = 86_400  # 24 heures
_KEY_PREFIX = "report:"

# Historique des conversations (sidebar) — TTL plus long que le report TTL : perdre l'état
# détaillé d'un pipeline après 24h est acceptable, perdre la liste des conversations dans la
# sidebar après seulement 24h ne le serait pas.
_SESSION_TTL = 30 * 24 * 3600  # 30 jours


def _report_key(report_id: str) -> str:
    return f"{_KEY_PREFIX}{report_id}"


def _session_key(tenant_id: str, session_id: str) -> str:
    return f"session:{tenant_id}:{session_id}"


def _session_reports_key(tenant_id: str, session_id: str) -> str:
    return f"session_reports:{tenant_id}:{session_id}"


def _sessions_index_key(tenant_id: str) -> str:
    return f"sessions_index:{tenant_id}"


async def _get_client():
    """Crée un client Redis async. Retourne None si Redis non disponible."""
    try:
        import redis.asyncio as aioredis

        client = aioredis.from_url(settings.redis_url, decode_responses=True)
        await client.ping()
        return client
    except Exception as exc:
        logger.warning("report_store_redis_unavailable", error=str(exc))
        return None


async def save_report_state(report_id: str, state: dict) -> None:
    """Sérialise et stocke le PipelineState dans Redis.

    Clé : report:{report_id} — TTL 24h. Met aussi à jour l'index de sessions (sidebar
    historique) — voir _index_session(). Ne plante jamais : Redis down = log warning + continuer.
    """
    client = await _get_client()
    if client is None:
        return
    try:
        await client.set(_report_key(report_id), json.dumps(state, default=str), ex=_REPORT_TTL)
        logger.debug("report_state_saved", report_id=report_id)
        await _index_session(client, report_id, state)
    except Exception as exc:
        logger.warning("report_state_save_error", report_id=report_id, error=str(exc))
    finally:
        await client.aclose()


async def _index_session(client, report_id: str, state: dict) -> None:
    """Met à jour l'index des sessions/conversations pour la sidebar historique.

    Un seul point d'entrée (appelé depuis save_report_state, donc à chaque création ET
    reprise de pipeline) — évite de dupliquer cette logique aux différents endpoints
    (generate, generate-powerbi, sessions/messages, hitl review). Jamais bloquant.
    """
    session_id = state.get("session_id")
    tenant_id = state.get("tenant_id")
    if not session_id or not tenant_id:
        return
    try:
        now = datetime.now(tz=timezone.utc)
        session_key = _session_key(tenant_id, session_id)
        reports_key = _session_reports_key(tenant_id, session_id)
        index_key = _sessions_index_key(tenant_id)

        await client.hsetnx(session_key, "first_prompt", state.get("prompt", ""))
        await client.hsetnx(session_key, "created_at", now.isoformat())
        await client.hset(
            session_key,
            mapping={
                "session_id": session_id,
                "last_prompt": state.get("prompt", ""),
                "last_status": state.get("status", ""),
                "last_activity_at": now.isoformat(),
                "data_source": state.get("data_source", "csv"),
            },
        )
        await client.expire(session_key, _SESSION_TTL)

        # SADD idempotent : save_report_state est appelé plusieurs fois pour le même
        # report_id (création, reprise HITL, erreur) — SCARD reste donc un compte correct
        # de rapports distincts, pas un compteur d'appels.
        await client.sadd(reports_key, report_id)
        await client.expire(reports_key, _SESSION_TTL)

        await client.zadd(index_key, {session_id: now.timestamp()})
        await client.expire(index_key, _SESSION_TTL)
    except Exception as exc:
        logger.warning("session_index_update_error", session_id=session_id, error=str(exc))


async def get_report_state(report_id: str) -> dict | None:
    """Lit et désérialise le PipelineState depuis Redis.

    Retourne None si clé absente ou Redis indisponible.
    """
    client = await _get_client()
    if client is None:
        return None
    try:
        raw = await client.get(_report_key(report_id))
        if raw is None:
            return None
        return json.loads(raw)
    except Exception as exc:
        logger.warning("report_state_get_error", report_id=report_id, error=str(exc))
        return None
    finally:
        await client.aclose()


async def update_report_status(report_id: str, status: str) -> None:
    """Met à jour uniquement le champ status d'un rapport existant."""
    state = await get_report_state(report_id)
    if state is None:
        logger.warning("report_state_not_found_for_update", report_id=report_id)
        return
    state["status"] = status
    await save_report_state(report_id, state)


async def list_sessions(tenant_id: str, limit: int = 50) -> list[dict]:
    """Liste les sessions (conversations) d'un tenant, triées par activité récente.

    Alimente la sidebar historique du frontend. Retourne [] si Redis indisponible ou si
    aucune session — ne bloque jamais l'appelant.
    """
    client = await _get_client()
    if client is None:
        return []
    try:
        index_key = _sessions_index_key(tenant_id)
        session_ids: list[str] = await client.zrevrange(index_key, 0, limit - 1)

        sessions: list[dict] = []
        for session_id in session_ids:
            session_key = _session_key(tenant_id, session_id)
            reports_key = _session_reports_key(tenant_id, session_id)
            data = await client.hgetall(session_key)
            if not data:
                continue
            report_count = await client.scard(reports_key)
            sessions.append(
                {
                    "session_id": session_id,
                    "title": data.get("first_prompt", ""),
                    "last_prompt": data.get("last_prompt", ""),
                    "last_status": data.get("last_status", ""),
                    "last_activity_at": data.get("last_activity_at", ""),
                    "created_at": data.get("created_at", ""),
                    "data_source": data.get("data_source", "csv"),
                    "report_count": report_count,
                }
            )
        return sessions
    except Exception as exc:
        logger.warning("list_sessions_error", tenant_id=tenant_id, error=str(exc))
        return []
    finally:
        await client.aclose()
