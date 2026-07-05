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
from app.pipeline.graph import (
    NODE_SETUP_COMPLETE,
    SETUP_COMPLETE,
    _route_after_schema,
    _setup_complete_node,
)
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
    from app.services.report_store import save_report_state

    state["status"] = "hitl_required"
    if state.get("report_id"):
        await save_report_state(state["report_id"], dict(state))
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
    graph.add_node(NODE_SETUP_COMPLETE, _setup_complete_node)

    graph.set_entry_point(entry_node)

    # schema → data, ou arrêt si setup_only (cf. graph.py::_route_after_schema, réutilisée
    # ici pour rester cohérente avec le graphe principal — CP1 (mode csv) peut être approuvé
    # pendant le setup initial, avant que l'utilisateur ait posé une vraie question).
    graph.add_conditional_edges(
        "schema_linking_agent",
        _route_after_schema,
        {HITL_WAIT: "hitl_wait", CONTINUE: "data_agent", END: END, SETUP_COMPLETE: NODE_SETUP_COMPLETE},
    )
    graph.add_edge(NODE_SETUP_COMPLETE, END)

    # data → insight, pour tout intent — cohérent avec le graphe principal (graph.py),
    # qui envoie désormais systématiquement data_agent vers insight_agent quel que soit
    # l'intent (le insight/narrative doit apparaître sur tout prompt, y compris simple_query).
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

    # cp1_metadata reprend normalement vers schema_linking_agent (mode csv), mais ce
    # dernier est sauté entièrement en mode powerbi_local (cf. graph.py::_route_after_metadata)
    # — la reprise doit donc aller directement à data_agent, SAUF si setup_only (connexion
    # initiale sans vraie question) : dans ce cas le setup s'arrête ici, comme dans le graphe
    # principal (_route_after_metadata renvoie SETUP_COMPLETE pour powerbi_local+setup_only).
    if checkpoint == "cp1_metadata" and state.get("data_source") == "powerbi_local":
        entry_node = NODE_SETUP_COMPLETE if state.get("setup_only") else "data_agent"

    # cp2_schema (mode csv uniquement) reprend normalement vers data_agent, mais si la
    # session est encore en setup_only (CP2 déclenché pendant la connexion initiale, avant
    # toute vraie question), le setup doit s'arrêter ici plutôt que de lancer data_agent sur
    # le prompt générique de connexion — cohérent avec _route_after_schema dans le graphe principal.
    if checkpoint == "cp2_schema" and state.get("setup_only"):
        entry_node = NODE_SETUP_COMPLETE

    logger.info(
        "resume_pipeline_built",
        report_id=report_id,
        checkpoint=checkpoint,
        entry_node=entry_node,
    )
    return _build_resume_pipeline(entry_node)
