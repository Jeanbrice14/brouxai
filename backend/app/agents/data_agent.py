from __future__ import annotations

from app.agents.base_agent import BaseAgent
from app.pipeline.state import PipelineState


class DataAgent(BaseAgent):
    """Agent 3 — Interprète le prompt, génère du code pandas et exécute les agrégations.

    Input  : state["prompt"] + state["schema"] + state["metadata"]
    Output : state["aggregates"]
    Modèle : gpt-4o-mini
    Note   : ne passe JAMAIS les données brutes au LLM
    """

    name = "data_agent"

    async def run(self, state: PipelineState) -> PipelineState:
        # TODO Sprint 4 : implémenter la logique métier complète
        state["aggregates"] = {"status": "stub"}
        return state
