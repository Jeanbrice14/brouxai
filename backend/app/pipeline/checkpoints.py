from __future__ import annotations

import structlog
from langgraph.graph import END, StateGraph

from app.agents.data_agent import DataAgent
from app.agents.insight_agent import InsightAgent
from app.agents.layout_agent import LayoutAgent
from app.agents.qa_agent import QAAgent
from app.agents.schema_linking_agent import SchemaLinkingAgent
from app.agents.storytelling_agent import StorytellingAgent
from app.agents.viz_agent import VizAgent
from app.pipeline.router import CONTINUE, HITL_WAIT, should_trigger_hitl
from app.pipeline.state import PipelineState

logger = structlog.get_logger(__name__)

# Checkpoint → premier agent à relancer
_CHECKPOINT_RESUME_MAP: dict[str, str] = {
    "cp1_metadata": "schema_linking_agent",
    "cp2_schema": "data_agent",
    "cp3_insights": "storytelling_agent",
    "cp4_narrative": "viz_agent",
}


async def _hitl_wait_node(state: PipelineState) -> PipelineState:
    state["status"] = "hitl_required"
    return state


def _build_resume_pipeline(entry_node: str):
    """Construit un sous-pipeline LangGraph démarrant à `entry_node`.

    Topologie complète (agents disponibles à la reprise) :
        schema → data → insight → (HITL?) → storytelling → viz → qa → (HITL?) → layout → END
    """
    schema_agent = SchemaLinkingAgent()
    data_agent = DataAgent()
    insight_agent = InsightAgent()
    storytelling_agent = StorytellingAgent()
    viz_agent = VizAgent()
    qa_agent = QAAgent()
    layout_agent = LayoutAgent()

    graph = StateGraph(PipelineState)

    # Nœuds disponibles pour la reprise
    graph.add_node("schema_linking_agent", schema_agent)
    graph.add_node("data_agent", data_agent)
    graph.add_node("insight_agent", insight_agent)
    graph.add_node("storytelling_agent", storytelling_agent)
    graph.add_node("viz_agent", viz_agent)
    graph.add_node("qa_agent", qa_agent)
    graph.add_node("layout_agent", layout_agent)
    graph.add_node("hitl_wait", _hitl_wait_node)

    graph.set_entry_point(entry_node)

    # schema → data (HITL possible après schema)
    graph.add_conditional_edges(
        "schema_linking_agent",
        should_trigger_hitl,
        {HITL_WAIT: "hitl_wait", CONTINUE: "data_agent", END: END},
    )

    # data → insight (pas de HITL sur data)
    graph.add_edge("data_agent", "insight_agent")

    # insight → storytelling (HITL possible)
    graph.add_conditional_edges(
        "insight_agent",
        should_trigger_hitl,
        {HITL_WAIT: "hitl_wait", CONTINUE: "storytelling_agent", END: END},
    )

    # storytelling → viz → qa (pas de HITL direct sur storytelling/viz)
    graph.add_edge("storytelling_agent", "viz_agent")
    graph.add_edge("viz_agent", "qa_agent")

    # qa → layout (HITL possible)
    graph.add_conditional_edges(
        "qa_agent",
        should_trigger_hitl,
        {HITL_WAIT: "hitl_wait", CONTINUE: "layout_agent", END: END},
    )

    # layout → END
    graph.add_conditional_edges(
        "layout_agent",
        should_trigger_hitl,
        {HITL_WAIT: "hitl_wait", CONTINUE: END, END: END},
    )

    graph.add_edge("hitl_wait", END)

    return graph.compile()


def resume_pipeline(report_id: str, state: dict):
    """Identifie l'agent de reprise et retourne le sous-pipeline compilé.

    Args:
        report_id: Identifiant du rapport (pour le logging).
        state: PipelineState courant (doit contenir hitl_checkpoint).

    Returns:
        Pipeline LangGraph compilé démarrant au bon agent.

    Raises:
        ValueError: Si hitl_checkpoint est invalide ou absent.
    """
    checkpoint = state.get("hitl_checkpoint")
    if not checkpoint:
        raise ValueError(f"report {report_id}: hitl_checkpoint manquant dans le state")

    entry_node = _CHECKPOINT_RESUME_MAP.get(checkpoint)
    if not entry_node:
        raise ValueError(
            f"report {report_id}: checkpoint inconnu '{checkpoint}'. "
            f"Valeurs valides : {list(_CHECKPOINT_RESUME_MAP)}"
        )

    logger.info(
        "resume_pipeline_built",
        report_id=report_id,
        checkpoint=checkpoint,
        entry_node=entry_node,
    )
    return _build_resume_pipeline(entry_node)
