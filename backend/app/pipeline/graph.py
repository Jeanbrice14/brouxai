from __future__ import annotations

from langgraph.graph import END, StateGraph

from app.agents.data_agent import DataAgent
from app.agents.insight_agent import InsightAgent
from app.agents.intent_agent import IntentAgent
from app.agents.layout_agent import LayoutAgent
from app.agents.metadata_agent import MetadataAgent
from app.agents.qa_agent import QAAgent
from app.agents.schema_linking_agent import SchemaLinkingAgent
from app.agents.storytelling_agent import StorytellingAgent
from app.agents.viz_agent import VizAgent
from app.pipeline.router import CONTINUE, HITL_WAIT, should_trigger_hitl
from app.pipeline.state import PipelineState

NODE_INTENT = "intent_agent"
NODE_METADATA = "metadata_agent"
NODE_SCHEMA = "schema_linking_agent"
NODE_DATA = "data_agent"
NODE_INSIGHT = "insight_agent"
NODE_STORYTELLING = "storytelling_agent"
NODE_VIZ = "viz_agent"
NODE_QA = "qa_agent"
NODE_LAYOUT = "layout_agent"
NODE_HITL_WAIT = "hitl_wait"


async def _hitl_wait_node(state: PipelineState) -> PipelineState:
    """Nœud terminal HITL — le pipeline s'arrête ici jusqu'à reprise humaine."""
    from app.services.report_store import save_report_state

    state["status"] = "hitl_required"
    if state.get("report_id"):
        await save_report_state(state["report_id"], dict(state))
    return state


def _route_after_intent(state: PipelineState) -> str:
    """Court-circuiter metadata+schema si déjà calculés (session existante)."""
    if state.get("metadata") and state.get("schema"):
        return "data_direct"
    return "full_setup"


def _route_after_data(state: PipelineState) -> str:
    """Court-circuiter insight+storytelling+qa selon l'intent.

    - simple_query  → direct layout (table depuis aggregates)
    - chart_request → viz puis layout (pas d'insight ni de qa)
    - full_report   → pipeline complet
    """
    intent = state.get("intent", "full_report")
    if intent == "simple_query":
        return "direct_layout"
    if intent == "chart_request":
        return "direct_viz"
    return "full_pipeline"


def build_pipeline() -> StateGraph:
    """Construit et compile le graphe LangGraph du pipeline BrouxAI.

    Topologie :
        intent → (metadata → schema →)? data
        → insight → (HITL?) → storytelling → viz → qa → (HITL?) → layout → END
    """
    graph = StateGraph(PipelineState)

    # ── Instanciation des agents ────────────────────────────────────────────
    intent_agent = IntentAgent()
    metadata_agent = MetadataAgent()
    schema_agent = SchemaLinkingAgent()
    data_agent = DataAgent()
    insight_agent = InsightAgent()
    storytelling_agent = StorytellingAgent()
    viz_agent = VizAgent()
    qa_agent = QAAgent()
    layout_agent = LayoutAgent()

    # ── Ajout des nœuds ─────────────────────────────────────────────────────
    graph.add_node(NODE_INTENT, intent_agent)
    graph.add_node(NODE_METADATA, metadata_agent)
    graph.add_node(NODE_SCHEMA, schema_agent)
    graph.add_node(NODE_DATA, data_agent)
    graph.add_node(NODE_INSIGHT, insight_agent)
    graph.add_node(NODE_STORYTELLING, storytelling_agent)
    graph.add_node(NODE_VIZ, viz_agent)
    graph.add_node(NODE_QA, qa_agent)
    graph.add_node(NODE_LAYOUT, layout_agent)
    graph.add_node(NODE_HITL_WAIT, _hitl_wait_node)

    # ── Point d'entrée ──────────────────────────────────────────────────────
    graph.set_entry_point(NODE_INTENT)

    # intent → metadata (full setup) ou data (session existante)
    graph.add_conditional_edges(
        NODE_INTENT,
        _route_after_intent,
        {"data_direct": NODE_DATA, "full_setup": NODE_METADATA},
    )

    # metadata → (HITL?) → schema
    graph.add_conditional_edges(
        NODE_METADATA,
        should_trigger_hitl,
        {HITL_WAIT: NODE_HITL_WAIT, CONTINUE: NODE_SCHEMA, END: END},
    )

    # schema → (HITL?) → data
    graph.add_conditional_edges(
        NODE_SCHEMA,
        should_trigger_hitl,
        {HITL_WAIT: NODE_HITL_WAIT, CONTINUE: NODE_DATA, END: END},
    )

    # data → (simple_query → layout) | (chart_request → viz → layout) | (full_report → insight → ...)
    graph.add_conditional_edges(
        NODE_DATA,
        _route_after_data,
        {
            "direct_layout": NODE_LAYOUT,
            "direct_viz": NODE_VIZ,
            "full_pipeline": NODE_INSIGHT,
        },
    )

    graph.add_conditional_edges(
        NODE_INSIGHT,
        should_trigger_hitl,
        {HITL_WAIT: NODE_HITL_WAIT, CONTINUE: NODE_STORYTELLING, END: END},
    )

    # storytelling → viz (HITL CP4 optionnel, géré dans l'agent lui-même)
    graph.add_edge(NODE_STORYTELLING, NODE_VIZ)
    graph.add_edge(NODE_VIZ, NODE_QA)

    graph.add_conditional_edges(
        NODE_QA,
        should_trigger_hitl,
        {HITL_WAIT: NODE_HITL_WAIT, CONTINUE: NODE_LAYOUT, END: END},
    )

    # layout → END
    graph.add_conditional_edges(
        NODE_LAYOUT,
        should_trigger_hitl,
        {HITL_WAIT: NODE_HITL_WAIT, CONTINUE: END, END: END},
    )

    # hitl_wait est un nœud terminal (ne reboucle pas — reprise via API)
    graph.add_edge(NODE_HITL_WAIT, END)

    return graph.compile()
