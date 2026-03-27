from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone

import structlog
from fastapi import APIRouter, Form, HTTPException, UploadFile
from fastapi.responses import RedirectResponse, Response

from app.pipeline.state import initial_state
from app.services.report_store import get_report_state, save_report_state
from app.services.storage import upload_file

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1", tags=["reports"])

# tenant_id fixe pour la v0 (Auth multi-tenant → Sprint 11)
_DEMO_TENANT = "demo-tenant"


def _get_pipeline():
    """Importe get_pipeline depuis main pour éviter l'import circulaire."""
    from app.main import get_pipeline

    return get_pipeline()


# ── POST /reports/generate ────────────────────────────────────────────────────


@router.post("/reports/generate", status_code=200)
async def generate_report(
    prompt: str = Form(...),
    files: list[UploadFile] | None = None,
    brand_kit: str = Form(default="{}"),
    session_id: str = Form(default=""),
    tenant_id: str = Form(default=""),
    user_id: str = Form(default=""),
    dataset_refs: str = Form(default="[]"),
    base_report_id: str = Form(default=""),
) -> dict:
    """Lance la génération d'un rapport.

    - Accepte multipart/form-data : prompt + files + brand_kit (JSON stringifié)
    - Upload les fichiers vers MinIO
    - Lance le pipeline en background
    - Retourne immédiatement {"report_id": ..., "status": "running", "session_id": ...}
    """
    # Validation prompt longueur
    if len(prompt.strip()) < 10:
        raise HTTPException(status_code=422, detail="Le prompt doit faire au moins 10 caractères.")
    if len(prompt.strip()) > 500:
        raise HTTPException(
            status_code=422, detail="Le prompt ne doit pas dépasser 500 caractères."
        )

    report_id = str(uuid.uuid4())
    _tenant_id = tenant_id or _DEMO_TENANT
    _user_id = user_id or "demo-user"
    _session_id = session_id or str(uuid.uuid4())

    # Désérialisation brand_kit
    try:
        brand_kit_dict: dict = json.loads(brand_kit) if brand_kit else {}
    except json.JSONDecodeError:
        brand_kit_dict = {}

    # Upload des fichiers vers MinIO (ou réutilisation des refs existantes)
    raw_data_refs: list[str] = []
    if files:
        for upload in files:
            filename = upload.filename or f"file_{uuid.uuid4().hex}"
            ref = f"s3://{_storage_bucket()}/{_tenant_id}/datasets/{report_id}/{filename}"
            data = await upload.read()
            await upload_file(ref, data, content_type=upload.content_type or "application/octet-stream")
            raw_data_refs.append(ref)

    # Fallback : refs passées explicitement (session sans re-upload)
    if not raw_data_refs:
        try:
            raw_data_refs = json.loads(dataset_refs) if dataset_refs else []
        except json.JSONDecodeError:
            raw_data_refs = []

    if not raw_data_refs:
        raise HTTPException(status_code=422, detail="Au moins un fichier est requis.")

    # Construction du PipelineState initial
    state = initial_state(
        tenant_id=_tenant_id,
        user_id=_user_id,
        report_id=report_id,
        prompt=prompt.strip(),
        raw_data_refs=raw_data_refs,
        brand_kit=brand_kit_dict,
    )
    state["status"] = "running"
    state["session_id"] = _session_id

    # Réutiliser metadata+schema d'un rapport précédent (court-circuite CP1+CP2)
    if base_report_id:
        base_state = await get_report_state(base_report_id)
        if base_state:
            if base_state.get("metadata"):
                state["metadata"] = base_state["metadata"]
            if base_state.get("schema"):
                state["schema"] = base_state["schema"]
            # Conserver les refs du rapport de base si pas de nouveaux fichiers
            if not raw_data_refs and base_state.get("raw_data_refs"):
                state["raw_data_refs"] = base_state["raw_data_refs"]
                raw_data_refs = base_state["raw_data_refs"]

    # Persister l'état initial
    await save_report_state(report_id, dict(state))

    # Lancer le pipeline en background
    pipeline = _get_pipeline()
    asyncio.create_task(_run_pipeline(pipeline, state, report_id))

    logger.info(
        "report_generation_started",
        report_id=report_id,
        tenant_id=_tenant_id,
        session_id=_session_id,
    )
    return {
        "report_id": report_id,
        "status": "running",
        "session_id": _session_id,
        "raw_data_refs": raw_data_refs,
    }


async def _run_pipeline(pipeline, state: dict, report_id: str) -> None:
    """Tâche background : exécute le pipeline et met à jour Redis."""
    try:
        final_state = await pipeline.ainvoke(state)
        # Sauvegarder l'état final (hitl_required ou complete non capturés par les agents)
        if isinstance(final_state, dict):
            await save_report_state(report_id, final_state)
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


# ── GET /reports/{report_id} ──────────────────────────────────────────────────


@router.get("/reports/{report_id}")
async def get_report(report_id: str) -> dict:
    """Retourne l'état courant d'un rapport."""
    state = await get_report_state(report_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Rapport '{report_id}' introuvable.")

    errors = state.get("errors", [])
    error_str = errors[-1] if errors else None

    return {
        "report_id": report_id,
        "status": state.get("status", "pending"),
        "intent": state.get("intent", ""),
        "response_type": state.get("response_type", "report"),
        "response": state.get("response", {}),
        "prompt": state.get("prompt", ""),
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
        "report_urls": state.get("report_urls", {}),
        "qa_report": state.get("qa_report", {}),
        "schema": state.get("schema", {}),
        "hitl_pending": state.get("hitl_pending", False),
        "hitl_checkpoint": state.get("hitl_checkpoint"),
        "current_agent": state.get("current_agent", ""),
        "session_id": state.get("session_id", ""),
        "raw_data_refs": state.get("raw_data_refs", []),
        "error": error_str,
    }


# ── GET /reports/{report_id}/html ─────────────────────────────────────────────


@router.get("/reports/{report_id}/html")
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


# ── Sessions ──────────────────────────────────────────────────────────────────


@router.post("/sessions/{session_id}/messages", status_code=200)
async def send_session_message(
    session_id: str,
    prompt: str = Form(...),
    dataset_refs: str = Form(default="[]"),
) -> dict:
    """Envoie un message dans une session existante.

    Réutilise les metadata + schema déjà calculés.
    Le pipeline repart depuis data_agent.
    """
    if len(prompt.strip()) < 3:
        raise HTTPException(status_code=422, detail="Le prompt est trop court.")

    # En v0 : on crée un nouveau rapport qui court-circuite metadata+schema
    report_id = str(uuid.uuid4())

    try:
        refs: list[str] = json.loads(dataset_refs)
    except json.JSONDecodeError:
        refs = []

    state = initial_state(
        tenant_id=_DEMO_TENANT,
        user_id="demo-user",
        report_id=report_id,
        prompt=prompt.strip(),
        raw_data_refs=refs,
    )
    state["status"] = "running"
    state["session_id"] = session_id

    await save_report_state(report_id, dict(state))

    pipeline = _get_pipeline()
    asyncio.create_task(_run_pipeline(pipeline, state, report_id))

    return {"report_id": report_id, "status": "running", "session_id": session_id}


@router.get("/sessions/{session_id}")
async def get_session(session_id: str) -> dict:
    """Retourne les informations d'une session."""
    # En v0: retourner une session vide (stockage sessions → Sprint 11)
    return {
        "session_id": session_id,
        "chat_history": [],
        "created_at": datetime.now(tz=timezone.utc).isoformat(),
    }
