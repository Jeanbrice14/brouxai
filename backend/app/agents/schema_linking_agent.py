from __future__ import annotations

from app.agents.base_agent import BaseAgent
from app.pipeline.state import PipelineState


class SchemaLinkingAgent(BaseAgent):
    """Agent 2 — Détecte les relations implicites entre plusieurs fichiers.

    Input  : state["raw_data_refs"] + state["metadata"]
    Output : state["schema"]
    Modèle : gpt-4o-mini
    HITL CP2 : toujours sur multi-fichiers + si taux orphelins > 5%
    """

    name = "schema_linking_agent"

    async def run(self, state: PipelineState) -> PipelineState:
        # TODO Sprint 3 : implémenter la logique métier complète
        state["schema"] = {"status": "stub"}
        return state
