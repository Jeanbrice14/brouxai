from __future__ import annotations

import json

import structlog

from app.config import settings

logger = structlog.get_logger(__name__)

_REPORT_TTL = 86_400  # 24 heures
_KEY_PREFIX = "report:"


def _report_key(report_id: str) -> str:
    return f"{_KEY_PREFIX}{report_id}"


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

    Clé : report:{report_id} — TTL 24h.
    Ne plante jamais : Redis down = log warning + continuer.
    """
    client = await _get_client()
    if client is None:
        return
    try:
        await client.set(_report_key(report_id), json.dumps(state, default=str), ex=_REPORT_TTL)
        logger.debug("report_state_saved", report_id=report_id)
    except Exception as exc:
        logger.warning("report_state_save_error", report_id=report_id, error=str(exc))
    finally:
        await client.aclose()


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
