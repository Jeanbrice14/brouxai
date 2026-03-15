from __future__ import annotations

from typing import TypedDict


class PipelineState(TypedDict):
    # Contexte
    tenant_id: str
    user_id: str
    report_id: str
    prompt: str

    # Références données (JAMAIS les données brutes)
    raw_data_refs: list[str]  # pointeurs Blob Storage
    brand_kit: dict  # logo, couleurs, typographie tenant

    # Enrichi progressivement par chaque agent
    metadata: dict  # → Metadata Agent
    schema: dict  # → Schema Linking Agent
    aggregates: dict  # → Data Agent
    insights: list[dict]  # → Insight Agent
    narrative: str  # → Storytelling Agent
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


def initial_state(
    tenant_id: str,
    user_id: str,
    report_id: str,
    prompt: str,
    raw_data_refs: list[str],
    brand_kit: dict | None = None,
) -> PipelineState:
    """Crée un PipelineState initial avec les valeurs par défaut."""
    return PipelineState(
        tenant_id=tenant_id,
        user_id=user_id,
        report_id=report_id,
        prompt=prompt,
        raw_data_refs=raw_data_refs,
        brand_kit=brand_kit or {},
        metadata={},
        schema={},
        aggregates={},
        insights=[],
        narrative="",
        viz_specs=[],
        qa_report={},
        report_urls={},
        hitl_pending=False,
        hitl_checkpoint=None,
        hitl_corrections=[],
        status="pending",
        errors=[],
        current_agent="",
    )
