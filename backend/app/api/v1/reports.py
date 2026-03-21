from __future__ import annotations

import asyncio
import json
import uuid
from datetime import UTC, datetime

import structlog
from fastapi import APIRouter, Form, HTTPException, UploadFile
from fastapi.responses import RedirectResponse, Response

from app.models.report import ReportResponse, ReportStatus
from app.pipeline.state import initial_state
from app.services.report_store import get_report_state, save_report_state
from app.services.storage import upload_file

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1/reports", tags=["reports"])

# tenant_id fixe pour la v0 (Auth multi-tenant → Sprint 11)
_DEMO_TENANT = "demo-tenant"


def _get_pipeline():
    """Importe get_pipeline depuis main pour éviter l'import circulaire."""
    from app.main import get_pipeline

    return get_pipeline()


# ── POST /generate ────────────────────────────────────────────────────────────


@router.post("/generate", status_code=200)
async def generate_report(
    prompt: str = Form(...),
    files: list[UploadFile] = Form(...),
    brand_kit: str = Form(default="{}"),
) -> dict:
    """Lance la génération d'un rapport.

    - Accepte multipart/form-data : prompt + files + brand_kit (JSON stringifié)
    - Upload les fichiers vers MinIO
    - Lance le pipeline en background
    - Retourne immédiatement {"report_id": ..., "status": "running"}
    """
    # Validation prompt longueur
    if len(prompt.strip()) < 10:
        raise HTTPException(status_code=422, detail="Le prompt doit faire au moins 10 caractères.")
    if len(prompt.strip()) > 500:
        raise HTTPException(
            status_code=422, detail="Le prompt ne doit pas dépasser 500 caractères."
        )

    report_id = str(uuid.uuid4())
    tenant_id = _DEMO_TENANT

    # Désérialisation brand_kit
    try:
        brand_kit_dict: dict = json.loads(brand_kit) if brand_kit else {}
    except json.JSONDecodeError:
        brand_kit_dict = {}

    # Upload des fichiers vers MinIO
    raw_data_refs: list[str] = []
    for upload in files:
        filename = upload.filename or f"file_{uuid.uuid4().hex}"
        ref = f"s3://{_storage_bucket()}/{tenant_id}/datasets/{report_id}/{filename}"
        data = await upload.read()
        await upload_file(ref, data, content_type=upload.content_type or "application/octet-stream")
        raw_data_refs.append(ref)

    if not raw_data_refs:
        raise HTTPException(status_code=422, detail="Au moins un fichier est requis.")

    # Construction du PipelineState initial
    state = initial_state(
        tenant_id=tenant_id,
        user_id="demo-user",
        report_id=report_id,
        prompt=prompt.strip(),
        raw_data_refs=raw_data_refs,
        brand_kit=brand_kit_dict,
    )
    state["status"] = "running"

    # Persister l'état initial
    await save_report_state(report_id, dict(state))

    # Lancer le pipeline en background
    pipeline = _get_pipeline()
    asyncio.create_task(_run_pipeline(pipeline, state, report_id))

    logger.info("report_generation_started", report_id=report_id, tenant_id=tenant_id)
    return {"report_id": report_id, "status": "running"}


async def _run_pipeline(pipeline, state: dict, report_id: str) -> None:
    """Tâche background : exécute le pipeline et met à jour Redis."""
    try:
        await pipeline.ainvoke(state)
    except Exception as exc:
        logger.error("pipeline_background_error", report_id=report_id, error=str(exc))
        await _mark_error(report_id, str(exc))


async def _mark_error(report_id: str, error: str) -> None:
    current = await get_report_state(report_id) or {}
    current["status"] = "error"
    current["errors"] = current.get("errors", []) + [error]
    await save_report_state(report_id, current)


def _storage_bucket() -> str:
    from app.config import settings

    return settings.storage_bucket


# ── GET /{report_id} ──────────────────────────────────────────────────────────


@router.get("/{report_id}", response_model=ReportResponse)
async def get_report(report_id: str) -> ReportResponse:
    """Retourne l'état courant d'un rapport."""
    state = await get_report_state(report_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Rapport '{report_id}' introuvable.")

    errors = state.get("errors", [])
    error_str = errors[-1] if errors else None

    return ReportResponse(
        report_id=report_id,
        status=ReportStatus(state.get("status", "pending")),
        prompt=state.get("prompt", ""),
        created_at=datetime.now(tz=UTC),
        report_urls=state.get("report_urls", {}),
        qa_report=state.get("qa_report", {}),
        error=error_str,
    )


# ── GET /{report_id}/html ─────────────────────────────────────────────────────


@router.get("/{report_id}/html")
async def get_report_html(report_id: str):
    """Redirige vers le rapport HTML stocké dans MinIO.

    - 302 si rapport prêt
    - 202 si rapport en cours de génération
    """
    state = await get_report_state(report_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Rapport '{report_id}' introuvable.")

    html_url = state.get("report_urls", {}).get("html_url")
    if not html_url:
        return Response(
            content='{"status": "pending", "message": "Rapport en cours de génération."}',
            status_code=202,
            media_type="application/json",
        )

    return RedirectResponse(url=html_url, status_code=302)
