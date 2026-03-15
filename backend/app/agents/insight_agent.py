from __future__ import annotations

from app.agents.base_agent import BaseAgent
from app.pipeline.state import PipelineState


class InsightAgent(BaseAgent):
    """Agent 4 — Analyse les agrégats et détecte tendances, anomalies, corrélations.

    Input  : state["aggregates"] + state["metadata"]
    Output : state["insights"]
    Modèle : gpt-4o
    HITL CP3 : si confiance insight < 0.80 OU anomalie OU variation > ±20%
    """

    name = "insight_agent"

    async def run(self, state: PipelineState) -> PipelineState:
        # TODO Sprint 5 : implémenter la logique métier complète
        state["insights"] = [{"status": "stub"}]
        return state
