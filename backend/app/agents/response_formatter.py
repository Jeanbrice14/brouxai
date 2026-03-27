from __future__ import annotations

import structlog

from app.pipeline.state import PipelineState

logger = structlog.get_logger(__name__)

_MAX_TABLE_ROWS = 20


def format_table_response(state: PipelineState) -> dict:
    """Formate la réponse pour response_type='table'."""
    aggregates = state.get("aggregates", {})
    narrative = state.get("narrative", "")
    prompt = state.get("prompt", "")

    # Prendre le premier agrégat disponible
    rows: list[dict] = []
    for key, val in aggregates.items():
        if isinstance(val, list) and val:
            rows = val[:_MAX_TABLE_ROWS]
            break

    columns = list(rows[0].keys()) if rows else []

    # Tronquer la narration à 1 phrase (mode table)
    summary = ""
    if narrative:
        sentences = [s.strip() for s in narrative.replace("\n", " ").split(".") if s.strip()]
        summary = sentences[0] + "." if sentences else narrative[:120]

    return {
        "type": "table",
        "title": prompt[:80] if prompt else "Résultats",
        "columns": columns,
        "rows": rows,
        "summary": summary,
        "total_rows": len(rows),
    }


def format_chart_response(state: PipelineState) -> dict:
    """Formate la réponse pour response_type='chart'."""
    viz_specs = state.get("viz_specs", [])
    aggregates = state.get("aggregates", {})
    narrative = state.get("narrative", "")
    prompt = state.get("prompt", "")

    viz_spec = viz_specs[0] if viz_specs else {}
    data_key = viz_spec.get("data_key", "")
    data = aggregates.get(data_key, [])
    if not isinstance(data, list):
        data = []

    # Tronquer la narration à 2-3 phrases max
    caption = ""
    if narrative:
        sentences = [s.strip() for s in narrative.replace("\n", " ").split(".") if s.strip()]
        caption = ". ".join(sentences[:3]) + ("." if sentences[:3] else "")

    return {
        "type": "chart",
        "title": viz_spec.get("title", prompt[:80] or "Graphique"),
        "viz_spec": viz_spec,
        "data": data,
        "caption": caption,
        "chart_type": viz_spec.get("chart_type", "bar"),
    }


def format_report_response(state: PipelineState) -> dict:
    """Formate la réponse pour response_type='report'."""
    return {
        "type": "report",
        "title": state.get("prompt", "Rapport analytique")[:80],
        "narrative": state.get("narrative", ""),
        "viz_specs": state.get("viz_specs", []),
        "aggregates": state.get("aggregates", {}),
        "qa_score": state.get("qa_report", {}).get("confidence_score", 1.0),
        "html_url": state.get("report_urls", {}).get("html_url", ""),
    }
