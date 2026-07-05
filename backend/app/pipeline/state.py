from __future__ import annotations

from typing import TypedDict


class PipelineState(TypedDict):
    # Contexte
    tenant_id: str
    user_id: str
    report_id: str
    prompt: str

    # Source de données
    data_source: str  # "csv" (défaut, upload + DuckDB) | "powerbi_local" (Power BI Desktop + MCP)

    # Connexion initiale (upload CSV / connect Power BI) sans question réelle de l'utilisateur —
    # le pipeline s'arrête après metadata_agent (+ schema_linking_agent en mode csv) sans lancer
    # data_agent/insight_agent/etc, tant qu'aucune vraie question n'a été posée. Sans ce flag, le
    # prompt générique de connexion ("Connecte-toi au modèle...") déclenchait une analyse complète
    # (DAX + insights), pouvant même redéclencher un CP3 avant la première vraie question.
    setup_only: bool

    # Références données CSV/Excel (JAMAIS les données brutes) — vide en mode powerbi_local
    raw_data_refs: list[str]  # pointeurs Blob Storage
    brand_kit: dict  # logo, couleurs, typographie tenant

    # Mode powerbi_local uniquement (voir services/powerbi_local_mcp.py)
    pbix_file_name: str  # nom du fichier Power BI Desktop ciblé (recherche auto par le serveur MCP)
    semantic_model_info: dict  # get_model_metadata() en cache pour la session (tables/mesures/relations)
    dax_queries: list[dict]  # requêtes DAX générées + exécutées durant le pipeline (traçabilité/debug)

    # Enrichi progressivement par chaque agent
    metadata: dict  # → Metadata Agent
    schema: dict  # → Schema Linking Agent
    aggregates: dict  # → Data Agent
    insights: list[dict]  # → Insight Agent
    narrative: str  # → Storytelling Agent (résumé exécutif 2 phrases)
    narrative_title: str  # → Storytelling Agent — titre reformulé, jamais le prompt brut
    recommendations: list[str]  # → Storytelling Agent (1-2 actions concrètes)
    viz_specs: list[dict]  # → Viz Agent (specs JSON)
    qa_report: dict  # → QA Agent
    report_urls: dict  # → Layout Agent (html_url uniquement en v0)

    # HITL (Human-in-the-Loop)
    hitl_pending: bool
    hitl_checkpoint: str | None  # cp1|cp2|cp3|cp4|cp5
    hitl_corrections: list[dict]  # historique corrections humaines

    # Pipeline meta
    status: str  # pending|running|hitl_required|complete|error
    errors: list[str]
    current_agent: str

    # Intent & response mode
    intent: str  # "simple_query" | "chart_request" | "full_report"
    intent_confidence: float
    response_type: str  # "table" | "chart" | "report"

    # Session & chat
    session_id: str
    chat_history: list[dict]

    # Formatted response (populated by LayoutAgent / ResponseFormatter)
    response: dict


def initial_state(
    tenant_id: str,
    user_id: str,
    report_id: str,
    prompt: str,
    raw_data_refs: list[str] | None = None,
    brand_kit: dict | None = None,
    data_source: str = "csv",
    pbix_file_name: str = "",
    setup_only: bool = False,
) -> PipelineState:
    """Crée un PipelineState initial avec les valeurs par défaut.

    `raw_data_refs` reste requis en mode data_source="csv" ; ignoré (liste vide)
    en mode "powerbi_local", où `pbix_file_name` pilote la connexion MCP.
    """
    return PipelineState(
        tenant_id=tenant_id,
        user_id=user_id,
        report_id=report_id,
        prompt=prompt,
        data_source=data_source,
        setup_only=setup_only,
        raw_data_refs=raw_data_refs or [],
        brand_kit=brand_kit or {},
        pbix_file_name=pbix_file_name,
        semantic_model_info={},
        dax_queries=[],
        metadata={},
        schema={},
        aggregates={},
        insights=[],
        narrative="",
        narrative_title="",
        recommendations=[],
        viz_specs=[],
        qa_report={},
        report_urls={},
        hitl_pending=False,
        hitl_checkpoint=None,
        hitl_corrections=[],
        status="pending",
        errors=[],
        current_agent="",
        intent="",
        intent_confidence=0.0,
        response_type="report",
        session_id="",
        chat_history=[],
        response={},
    )
