from __future__ import annotations

import asyncio

import structlog
from fastapi import APIRouter, HTTPException

from app.models.report import HITLReviewRequest
from app.pipeline.checkpoints import resume_pipeline
from app.services.conversation_memory import maybe_record_turn
from app.services.report_store import get_report_state, save_report_state
from app.services.schema_rag import update_field_from_hitl

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

    # Construire le pipeline de reprise AVANT de réinitialiser le checkpoint
    try:
        pipeline = resume_pipeline(report_id, state)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    # Appliquer les corrections selon le checkpoint
    if body.action == "corrected" and body.corrections:
        _apply_corrections(state, body.checkpoint, body.corrections)

    # CP1 en mode powerbi_local : synchroniser les descriptions validées par l'humain
    # vers le RAG schéma (pgvector) — un champ à la fois, jamais une réindexation complète.
    if body.checkpoint == "cp1_metadata" and state.get("data_source") == "powerbi_local":
        await _sync_cp1_fields_to_schema_rag(state, body.corrections)

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
        final_state = await pipeline.ainvoke(state)
        if isinstance(final_state, dict):
            await save_report_state(report_id, final_state)
            await maybe_record_turn(final_state)
    except Exception as exc:
        logger.error("resume_pipeline_error", report_id=report_id, error=str(exc))
        from app.services.report_store import get_report_state

        current = await get_report_state(report_id) or {}
        current["status"] = "error"
        current["errors"] = current.get("errors", []) + [str(exc)]
        await save_report_state(report_id, current)


async def _sync_cp1_fields_to_schema_rag(state: dict, corrections: dict) -> None:
    """Répercute les descriptions validées humainement (CP1) vers le RAG schéma.

    Ne traite QUE les champs explicitement présents dans `corrections["fields"]`
    (format attendu : {"fields": {"Table[Colonne]": "description validée", ...}}) — pas
    une boucle sur tout le schéma. Un CP1 sur un modèle Power BI non documenté peut
    couvrir 100+ champs (cf. AdventureWorks) ; ré-embedder tout le lot sur une simple
    approbation sans corrections ciblées serait coûteux (1 appel d'embedding par champ)
    et n'indique pas vraiment quels champs l'humain a réellement validé un par un.
    """
    tenant_id = state.get("tenant_id", "")
    model_id = state.get("pbix_file_name", "")
    if not tenant_id or not model_id:
        return

    fields = corrections.get("fields") if isinstance(corrections, dict) else None
    if not isinstance(fields, dict) or not fields:
        return

    for qualified_name, description in fields.items():
        await update_field_from_hitl(tenant_id, model_id, qualified_name, str(description))


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
