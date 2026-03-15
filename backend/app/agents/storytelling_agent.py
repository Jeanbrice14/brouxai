from __future__ import annotations

from app.agents.base_agent import BaseAgent
from app.pipeline.state import PipelineState


class StorytellingAgent(BaseAgent):
    """Agent 5 — Construit la narration data-driven structurée.

    Input  : state["insights"] + state["brand_kit"]
    Output : state["narrative"]
    Modèle : gpt-4o
    HITL CP4 : optionnel selon plan tarifaire (obligatoire Business+)
    Règle   : ne doit JAMAIS inventer de chiffres
    """

    name = "storytelling_agent"

    async def run(self, state: PipelineState) -> PipelineState:
        # TODO Sprint 5 : implémenter la logique métier complète
        state["narrative"] = "stub"
        return state
