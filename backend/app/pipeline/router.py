from __future__ import annotations

from langgraph.graph import END

from app.pipeline.state import PipelineState

# Valeurs retournées par le routeur
HITL_WAIT = "hitl_wait"
CONTINUE = "continue"


def should_trigger_hitl(state: PipelineState) -> str:
    """Routeur conditionnel utilisé après les agents susceptibles de déclencher le HITL.

    Retourne :
    - "hitl_wait"  → le pipeline s'arrête, attend validation humaine
    - END          → le pipeline est terminé (erreur ou complet)
    - "continue"   → le pipeline passe à l'agent suivant
    """
    if state.get("status") == "error":
        return END

    if state.get("hitl_pending"):
        return HITL_WAIT

    if state.get("status") == "complete":
        return END

    return CONTINUE
