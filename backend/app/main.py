from __future__ import annotations

from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI

from app.pipeline.graph import build_pipeline

logger = structlog.get_logger(__name__)

# Pipeline compilé au démarrage de l'application
_pipeline = None


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


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "pipeline": "initialized" if _pipeline is not None else "not_initialized",
    }


def get_pipeline():
    """Retourne le pipeline compilé (utilisé comme dépendance FastAPI)."""
    return _pipeline
