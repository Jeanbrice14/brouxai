from __future__ import annotations

import structlog
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

logger = structlog.get_logger(__name__)

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
NODE_SETUP_COMPLETE = "setup_complete_node"

# Retourné par _route_after_metadata/_route_after_schema quand state["setup_only"] est vrai
# et qu'aucun HITL n'est en attente — voir _setup_complete_node.
SETUP_COMPLETE = "setup_complete"


async def _hitl_wait_node(state: PipelineState) -> PipelineState:
    """Nœud terminal HITL — le pipeline s'arrête ici jusqu'à reprise humaine."""
    from app.services.report_store import save_report_state

    state["status"] = "hitl_required"
    if state.get("report_id"):
        await save_report_state(state["report_id"], dict(state))
    return state


async def _setup_complete_node(state: PipelineState) -> PipelineState:
    """Nœud terminal de la connexion initiale (upload CSV / connect Power BI, state["setup_only"]).

    Le prompt de connexion ("Connecte-toi au modèle...") ne correspond à aucune vraie question
    utilisateur — lancer data_agent/insight_agent dessus produirait une analyse DAX/insights
    inventée sur un prompt générique, pouvant même redéclencher un CP3 avant que l'utilisateur
    ait posé sa première vraie question. On s'arrête donc ici, une fois metadata (+ schema en
    mode csv) validés, sans jamais toucher data_agent/insight/storytelling/viz/qa/layout.

    Le frontend détecte ce cas via `response.type == "setup_complete"` (déjà attendu côté
    app/chat/[sessionId]/page.tsx avant même ce correctif).
    """
    from app.services.report_store import save_report_state

    state["status"] = "complete"
    state["response"] = {"type": "setup_complete"}
    if state.get("report_id"):
        await save_report_state(state["report_id"], dict(state))
    return state


def _route_after_intent(state: PipelineState) -> str:
    """Court-circuiter metadata+schema si déjà calculés (session existante).

    En mode powerbi_local, schema_linking_agent n'est jamais exécuté (cf.
    _route_after_metadata) — state["schema"] reste donc TOUJOURS {} pour ce data_source,
    même après une session pleinement validée. Exiger schema en plus de metadata forcerait
    un full_setup (donc un nouveau CP1) à chaque message powerbi_local, peu importe la
    réutilisation de base_report_id.
    """
    has_schema_or_exempt = state.get("schema") or state.get("data_source") == "powerbi_local"
    if state.get("metadata") and has_schema_or_exempt:
        return "data_direct"
    return "full_setup"


def _route_after_metadata(state: PipelineState) -> str:
    """HITL éventuel, puis routage schema/data selon data_source.

    - "csv"          → schema_linking_agent (détection FK/relations, HITL CP2 potentiel).
    - "powerbi_local" → data_agent directement. Le modèle sémantique Power BI a déjà ses
      relations définies par un humain dans Power BI Desktop (state["semantic_model_info"]) —
      schema_linking_agent n'a rien à y détecter, et planterait sur des raw_data_refs qui ne
      pointent vers aucun fichier Blob Storage (voir metadata_agent.py, mode powerbi_local).
    - Toute autre valeur (futur data_source non prévu) → schema_linking_agent par défaut,
      comportement conservateur (un agent inutile est préférable à un agent manquant), avec
      un warning loggé pour signaler l'absence de branche explicite.
    """
    hitl_decision = should_trigger_hitl(state)
    if hitl_decision != CONTINUE:
        return hitl_decision

    data_source = state.get("data_source", "csv")
    if data_source == "csv":
        # schema_linking_agent tourne toujours pendant le setup (détection de relations,
        # CP2 potentiel) — c'est _route_after_schema qui arrête ensuite si setup_only.
        return "to_schema"
    if data_source == "powerbi_local":
        # powerbi_local ne passe jamais par schema_linking_agent (cf. docstring ci-dessus) —
        # c'est donc ici, juste après metadata, que le setup initial doit s'arrêter.
        return SETUP_COMPLETE if state.get("setup_only") else "to_data"

    logger.warning(
        "route_after_metadata_unknown_data_source",
        data_source=data_source,
        fallback="schema_linking_agent",
    )
    return "to_schema"


def _route_after_schema(state: PipelineState) -> str:
    """Après schema_linking_agent (mode csv) : HITL éventuel (CP2), puis arrêt si setup_only
    (le setup initial s'arrête ici, sans lancer data_agent/insight/etc — cf.
    _setup_complete_node), sinon poursuite normale vers data_agent.
    """
    hitl_decision = should_trigger_hitl(state)
    if hitl_decision != CONTINUE:
        return hitl_decision
    return SETUP_COMPLETE if state.get("setup_only") else CONTINUE


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
    graph.add_node(NODE_SETUP_COMPLETE, _setup_complete_node)

    # ── Point d'entrée ──────────────────────────────────────────────────────
    graph.set_entry_point(NODE_INTENT)

    # intent → metadata (full setup) ou data (session existante)
    graph.add_conditional_edges(
        NODE_INTENT,
        _route_after_intent,
        {"data_direct": NODE_DATA, "full_setup": NODE_METADATA},
    )

    # metadata → (HITL?) → schema (csv) | data ou setup_complete (powerbi_local, cf.
    # _route_after_metadata — schema_linking_agent toujours sauté dans ce mode)
    graph.add_conditional_edges(
        NODE_METADATA,
        _route_after_metadata,
        {
            HITL_WAIT: NODE_HITL_WAIT,
            END: END,
            "to_schema": NODE_SCHEMA,
            "to_data": NODE_DATA,
            SETUP_COMPLETE: NODE_SETUP_COMPLETE,
        },
    )

    # schema → (HITL?) → data, ou arrêt si setup_only (cf. _route_after_schema)
    graph.add_conditional_edges(
        NODE_SCHEMA,
        _route_after_schema,
        {HITL_WAIT: NODE_HITL_WAIT, CONTINUE: NODE_DATA, END: END, SETUP_COMPLETE: NODE_SETUP_COMPLETE},
    )

    graph.add_edge(NODE_SETUP_COMPLETE, END)

    # data → insight, pour tout intent — le insight/narrative doit apparaître sur chaque
    # prompt, y compris simple_query (qui repartait directement en layout auparavant, sans
    # aucune narration). Seul le nombre de phrases de la narration varie ensuite selon
    # response_type (voir StorytellingAgent : table=1 phrase, chart=2-3, report=JSON complet).
    graph.add_edge(NODE_DATA, NODE_INSIGHT)

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
