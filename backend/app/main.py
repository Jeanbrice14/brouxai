from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from app.api.v1.hitl import router as hitl_router
from app.api.v1.reports import router as reports_router
from app.pipeline.graph import build_pipeline

logger = structlog.get_logger(__name__)

# Pipeline compilé au démarrage de l'application
_pipeline = None

# Intervalle de polling WebSocket (secondes)
_WS_POLL_INTERVAL = 0.5


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    logger.info("pipeline_init_start")
    _pipeline = build_pipeline()
    logger.info("pipeline_init_complete")
    yield
    logger.info("pipeline_shutdown")


app = FastAPI(
    title="BrouxAI API",
    description="Plateforme SaaS no-code de data storytelling multi-agent",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(reports_router)
app.include_router(hitl_router)


# ── Health ────────────────────────────────────────────────────────────────────


async def _check_redis() -> str:
    """Vérifie la disponibilité de Redis."""
    try:
        import redis.asyncio as aioredis

        from app.config import settings

        client = aioredis.from_url(settings.redis_url, decode_responses=True)
        await client.ping()
        await client.aclose()
        return "ok"
    except Exception:
        return "error"


async def _check_storage() -> str:
    """Vérifie la disponibilité du Blob Storage (MinIO)."""
    try:
        import boto3

        from app.config import settings

        def _ping():
            client = boto3.client(
                "s3",
                endpoint_url=settings.storage_endpoint,
                aws_access_key_id=settings.storage_key,
                aws_secret_access_key=settings.storage_secret,
                region_name="us-east-1",
            )
            client.list_buckets()

        await asyncio.to_thread(_ping)
        return "ok"
    except Exception:
        return "error"


@app.get("/health")
async def health():
    redis_status, storage_status = await asyncio.gather(
        _check_redis(),
        _check_storage(),
        return_exceptions=True,
    )
    # gather with return_exceptions=True may return exceptions as values
    if isinstance(redis_status, Exception):
        redis_status = "error"
    if isinstance(storage_status, Exception):
        storage_status = "error"

    return {
        "status": "ok",
        "pipeline": "initialized" if _pipeline is not None else "not_initialized",
        "redis": redis_status,
        "storage": storage_status,
    }


def get_pipeline():
    """Retourne le pipeline compilé (utilisé comme dépendance FastAPI)."""
    return _pipeline


# ── WebSocket /ws/reports/{report_id} ────────────────────────────────────────


@app.websocket("/ws/reports/{report_id}")
async def ws_report_status(websocket: WebSocket, report_id: str):
    """WebSocket de suivi d'avancement du pipeline.

    Envoie un message JSON à chaque changement de current_agent.
    Ferme la connexion quand status == "complete" ou "error".
    """
    from app.services.report_store import get_report_state

    await websocket.accept()
    log = logger.bind(report_id=report_id)
    log.info("ws_connected")

    last_agent: str | None = None

    try:
        while True:
            state = await get_report_state(report_id)

            if state is None:
                await websocket.send_json({"report_id": report_id, "error": "Rapport introuvable."})
                break

            current_agent = state.get("current_agent", "")
            status = state.get("status", "pending")

            # Envoyer uniquement si changement d'agent ou fin de pipeline
            if current_agent != last_agent or status in ("complete", "error", "hitl_required"):
                await websocket.send_json(
                    {
                        "report_id": report_id,
                        "status": status,
                        "current_agent": current_agent,
                        "hitl_pending": state.get("hitl_pending", False),
                        "hitl_checkpoint": state.get("hitl_checkpoint"),
                    }
                )
                last_agent = current_agent

            if status in ("complete", "error"):
                break

            await asyncio.sleep(_WS_POLL_INTERVAL)

    except WebSocketDisconnect:
        log.info("ws_disconnected")
    except Exception as exc:
        log.error("ws_error", error=str(exc))
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
        log.info("ws_closed")
