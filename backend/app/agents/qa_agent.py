from __future__ import annotations

from app.agents.base_agent import BaseAgent
from app.pipeline.state import PipelineState


class QAAgent(BaseAgent):
    """Agent 7 — Vérifie la cohérence et détecte les hallucinations LLM.

    Input  : rapport complet + state["aggregates"]
    Output : state["qa_report"]
    Modèle : gpt-4o-mini
    HITL   : si confidence_score < 0.80 → HITL obligatoire avant publication
    Pattern : Constitutional AI
    """

    name = "qa_agent"

    async def run(self, state: PipelineState) -> PipelineState:
        # TODO Sprint 6 : implémenter la logique métier complète
        state["qa_report"] = {"status": "stub", "confidence_score": 1.0}
        return state
