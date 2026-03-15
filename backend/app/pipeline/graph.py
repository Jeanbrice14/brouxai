from __future__ import annotations

from langgraph.graph import END, StateGraph

from app.agents.data_agent import DataAgent
from app.agents.insight_agent import InsightAgent
from app.agents.layout_agent import LayoutAgent
from app.agents.metadata_agent import MetadataAgent
from app.agents.qa_agent import QAAgent
from app.agents.schema_linking_agent import SchemaLinkingAgent
from app.agents.storytelling_agent import StorytellingAgent
from app.agents.viz_agent import VizAgent
from app.pipeline.router import CONTINUE, HITL_WAIT, should_trigger_hitl
from app.pipeline.state import PipelineState

# Noms des nœuds du graphe
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
    state["status"] = "hitl_required"
    return state


def build_pipeline() -> StateGraph:
    """Construit et compile le graphe LangGraph du pipeline BrouxAI.

    Topologie :
        metadata → (HITL?) → schema → (HITL?) → data
        → insight → (HITL?) → storytelling → viz → qa → (HITL?) → layout → END
    """
    graph = StateGraph(PipelineState)

    # ── Instanciation des agents ────────────────────────────────────────────
    metadata_agent = MetadataAgent()
    schema_agent = SchemaLinkingAgent()
    data_agent = DataAgent()
    insight_agent = InsightAgent()
    storytelling_agent = StorytellingAgent()
    viz_agent = VizAgent()
    qa_agent = QAAgent()
    layout_agent = LayoutAgent()

    # ── Ajout des nœuds ─────────────────────────────────────────────────────
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
    graph.set_entry_point(NODE_METADATA)

    # ── Edges conditionnels après agents HITL-eligible ──────────────────────
    _hitl_edges = {
        HITL_WAIT: NODE_HITL_WAIT,
        CONTINUE: NODE_SCHEMA,
        END: END,
    }
    graph.add_conditional_edges(NODE_METADATA, should_trigger_hitl, _hitl_edges)

    _hitl_edges_schema = {
        HITL_WAIT: NODE_HITL_WAIT,
        CONTINUE: NODE_DATA,
        END: END,
    }
    graph.add_conditional_edges(NODE_SCHEMA, should_trigger_hitl, _hitl_edges_schema)

    # data_agent → insight (pas de HITL sauf anomalie volumétrie, géré en interne)
    graph.add_edge(NODE_DATA, NODE_INSIGHT)

    _hitl_edges_insight = {
        HITL_WAIT: NODE_HITL_WAIT,
        CONTINUE: NODE_STORYTELLING,
        END: END,
    }
    graph.add_conditional_edges(NODE_INSIGHT, should_trigger_hitl, _hitl_edges_insight)

    # storytelling → viz (HITL CP4 optionnel, géré dans l'agent lui-même)
    graph.add_edge(NODE_STORYTELLING, NODE_VIZ)
    graph.add_edge(NODE_VIZ, NODE_QA)

    _hitl_edges_qa = {
        HITL_WAIT: NODE_HITL_WAIT,
        CONTINUE: NODE_LAYOUT,
        END: END,
    }
    graph.add_conditional_edges(NODE_QA, should_trigger_hitl, _hitl_edges_qa)

    # layout → END
    graph.add_conditional_edges(
        NODE_LAYOUT,
        should_trigger_hitl,
        {HITL_WAIT: NODE_HITL_WAIT, CONTINUE: END, END: END},
    )

    # hitl_wait est un nœud terminal (ne reboucle pas — reprise via API)
    graph.add_edge(NODE_HITL_WAIT, END)

    return graph.compile()
