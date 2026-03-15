from __future__ import annotations

from app.agents.base_agent import BaseAgent
from app.pipeline.state import PipelineState


class VizAgent(BaseAgent):
    """Agent 6 — Sélectionne le type de graphique et génère les viz_spec JSON.

    Input  : state["insights"] + state["aggregates"] + state["brand_kit"]
    Output : state["viz_specs"]
    Modèle : gpt-4o-mini
    Règles : bar→comparaison, line→temporel, pie→proportion (max 5 cat.), scatter→corrélation
    V0     : viz_spec JSON pour Recharts uniquement — export PNG/PDF reporté v1
    """

    name = "viz_agent"

    async def run(self, state: PipelineState) -> PipelineState:
        # TODO Sprint 6 : implémenter la logique métier complète
        state["viz_specs"] = [{"status": "stub"}]
        return state
