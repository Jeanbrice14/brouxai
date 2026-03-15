from __future__ import annotations

from app.agents.base_agent import BaseAgent
from app.pipeline.state import PipelineState


class LayoutAgent(BaseAgent):
    """Agent 8 — Assemble le rendu HTML interactif final.

    Input  : state["narrative"] + state["viz_specs"] + state["aggregates"] + state["brand_kit"]
    Output : state["report_urls"]  — {html_url} stocké dans MinIO/R2
    Outils : Jinja2 (templates HTML), Recharts (rendu frontend interactif)
    V0     : HTML uniquement — export PDF reporté v1
    """

    name = "layout_agent"

    async def run(self, state: PipelineState) -> PipelineState:
        # TODO Sprint 7 : implémenter la logique métier complète
        state["report_urls"] = {"html_url": "stub://report.html"}
        state["status"] = "complete"
        return state
