from __future__ import annotations

import asyncio

import structlog
from fastapi import APIRouter, HTTPException

from app.models.report import HITLReviewRequest
from app.pipeline.checkpoints import resume_pipeline
from app.services.report_store import get_report_state, save_report_state

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/api/v1/reports", tags=["hitl"])

# Données exposées à l'humain selon le checkpoint
_CHECKPOINT_DATA_KEYS: dict[str, list[str]] = {
    "cp1_metadata": ["metadata", "raw_data_refs"],
    "cp2_schema": ["schema", "metadata"],
    "cp3_insights": ["insights", "aggregates"],
    "cp4_narrative": ["narrative", "insights"],
}


# ── GET /{report_id}/review ────────────────────────────────────────────────────


@router.get("/{report_id}/review")
async def get_review(report_id: str) -> dict:
    """Retourne les données à valider pour le checkpoint HITL courant.

    - 200 + {checkpoint, data} si HITL en attente
    - 404 si rapport introuvable
    - 409 si aucun HITL en attente
    """
    state = await get_report_state(report_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Rapport '{report_id}' introuvable.")

    if not state.get("hitl_pending"):
        raise HTTPException(
            status_code=409,
            detail="Aucune validation HITL en attente pour ce rapport.",
        )

    checkpoint = state.get("hitl_checkpoint", "")
    keys = _CHECKPOINT_DATA_KEYS.get(checkpoint, [])
    data = {k: state.get(k) for k in keys}

    return {
        "report_id": report_id,
        "checkpoint": checkpoint,
        "data": data,
        "prompt": state.get("prompt", ""),
    }


# ── POST /{report_id}/review ───────────────────────────────────────────────────


@router.post("/{report_id}/review")
async def post_review(report_id: str, body: HITLReviewRequest) -> dict:
    """Soumet la validation humaine et relance le pipeline.

    - Applique les corrections au state si action == 'corrected'
    - Réinitialise hitl_pending
    - Relance le pipeline en background depuis le bon agent
    - 404 si rapport introuvable
    - 409 si aucun HITL en attente
    """
    state = await get_report_state(report_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Rapport '{report_id}' introuvable.")

    if not state.get("hitl_pending"):
        raise HTTPException(
            status_code=409,
            detail="Aucune validation HITL en attente pour ce rapport.",
        )

    # Appliquer les corrections selon le checkpoint
    if body.action == "corrected" and body.corrections:
        _apply_corrections(state, body.checkpoint, body.corrections)

    # Réinitialiser le flag HITL
    state["hitl_pending"] = False
    state["hitl_checkpoint"] = None
    state["hitl_corrections"] = state.get("hitl_corrections", []) + [
        {
            "checkpoint": body.checkpoint,
            "action": body.action,
            "corrections": body.corrections,
        }
    ]
    state["status"] = "running"

    await save_report_state(report_id, state)

    # Construire le pipeline de reprise et le lancer en background
    try:
        pipeline = resume_pipeline(report_id, state)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    asyncio.create_task(_run_resume(pipeline, state, report_id))

    logger.info("hitl_review_submitted", report_id=report_id, action=body.action)
    return {"status": "resumed", "report_id": report_id}


# ── DELETE /{report_id}/review ─────────────────────────────────────────────────


@router.delete("/{report_id}/review")
async def delete_review(report_id: str) -> dict:
    """Rejette le rapport — arrête le pipeline définitivement.

    - Met status = 'error', hitl_pending = False
    - 404 si rapport introuvable
    - 409 si aucun HITL en attente
    """
    state = await get_report_state(report_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Rapport '{report_id}' introuvable.")

    if not state.get("hitl_pending"):
        raise HTTPException(
            status_code=409,
            detail="Aucune validation HITL en attente pour ce rapport.",
        )

    state["hitl_pending"] = False
    state["hitl_checkpoint"] = None
    state["status"] = "error"
    state["errors"] = state.get("errors", []) + ["Rapport rejeté par l'utilisateur."]

    await save_report_state(report_id, state)

    logger.info("hitl_report_rejected", report_id=report_id)
    return {"status": "rejected", "report_id": report_id}


# ── Helpers ───────────────────────────────────────────────────────────────────


async def _run_resume(pipeline, state: dict, report_id: str) -> None:
    """Tâche background : exécute la reprise du pipeline."""
    try:
        await pipeline.ainvoke(state)
    except Exception as exc:
        logger.error("resume_pipeline_error", report_id=report_id, error=str(exc))
        from app.services.report_store import get_report_state, save_report_state

        current = await get_report_state(report_id) or {}
        current["status"] = "error"
        current["errors"] = current.get("errors", []) + [str(exc)]
        await save_report_state(report_id, current)


def _apply_corrections(state: dict, checkpoint: str, corrections: dict) -> None:
    """Applique les corrections humaines au state selon le checkpoint."""
    if checkpoint == "cp1_metadata":
        existing = state.get("metadata", {})
        existing.update(corrections)
        state["metadata"] = existing
    elif checkpoint == "cp2_schema":
        existing = state.get("schema", {})
        existing.update(corrections)
        state["schema"] = existing
    elif checkpoint == "cp3_insights":
        if "insights" in corrections:
            state["insights"] = corrections["insights"]
    elif checkpoint == "cp4_narrative":
        if "narrative" in corrections:
            state["narrative"] = corrections["narrative"]
