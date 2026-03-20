from __future__ import annotations

import hashlib
import json

import structlog

from app.config import settings

logger = structlog.get_logger(__name__)


async def _get_client():
    """Crée un client Redis async. Retourne None si Redis non disponible."""
    try:
        import redis.asyncio as aioredis

        client = aioredis.from_url(settings.redis_url, decode_responses=True)
        await client.ping()
        return client
    except Exception as exc:
        logger.warning("cache_redis_unavailable", error=str(exc))
        return None


async def get_cache(key: str) -> dict | None:
    """Lit une valeur depuis le cache Redis.

    Returns:
        dict si la clé existe, None si miss ou Redis indisponible.
    """
    client = await _get_client()
    if client is None:
        return None
    try:
        raw = await client.get(key)
        if raw:
            logger.info("cache_hit", key=key)
            return json.loads(raw)
        logger.info("cache_miss", key=key)
        return None
    except Exception as exc:
        logger.warning("cache_get_error", key=key, error=str(exc))
        return None
    finally:
        await client.aclose()


async def set_cache(key: str, value: dict, ttl: int = 3600) -> None:
    """Écrit une valeur dans le cache Redis avec TTL.

    Ne plante jamais — Redis down = log warning + continuer.
    """
    client = await _get_client()
    if client is None:
        return
    try:
        await client.set(key, json.dumps(value), ex=ttl)
        logger.info("cache_set", key=key, ttl=ttl)
    except Exception as exc:
        logger.warning("cache_set_error", key=key, error=str(exc))
    finally:
        await client.aclose()


async def make_cache_key(tenant_id: str, dataset_refs: list[str], prompt: str) -> str:
    """Génère une clé de cache MD5 déterministe depuis les paramètres.

    Format : data_agent:<md5(tenant_id|sorted_refs|prompt)>
    """
    raw = f"{tenant_id}|{sorted(dataset_refs)}|{prompt}"
    digest = hashlib.md5(raw.encode()).hexdigest()  # noqa: S324
    return f"data_agent:{digest}"
