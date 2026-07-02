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
    data = aggregates.get(data_key, []) if data_key else []
    if not isinstance(data, list):
        data = []

    # Fallback: no viz_spec or empty data → build a minimal spec from aggregates
    if (not viz_spec or not data) and aggregates:
        for key, rows in aggregates.items():
            if not isinstance(rows, list) or not rows or not isinstance(rows[0], dict):
                continue
            cols = list(rows[0].keys())
            numeric_cols = [c for c in cols if isinstance(rows[0].get(c), (int, float))]
            str_cols = [c for c in cols if c not in numeric_cols]
            if not numeric_cols:
                continue
            is_temporal = any(
                t in c.lower() for c in cols for t in ("mois", "date", "month", "semaine", "year")
            )
            chart_type = "line" if is_temporal and len(rows) > 2 else "bar"
            viz_spec = {
                "chart_type": chart_type,
                "title": prompt[:80] or key.replace("_", " ").capitalize(),
                "data_key": key,
                "x": str_cols[0] if str_cols else cols[0],
                "y": numeric_cols[0],
                "color_by": None,
                "colors": {"primary": "#1E3A8A", "positive": "#16A34A", "negative": "#DC2626"},
                "annotations": [],
            }
            data = rows
            break

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
    aggregates = state.get("aggregates", {})
    viz_specs = state.get("viz_specs", [])

    # Injecter les données agrégées dans chaque viz_spec
    viz_specs_with_data = []
    for spec in viz_specs:
        spec_copy = dict(spec)
        data_key = spec_copy.get("data_key", "")
        rows = aggregates.get(data_key, [])
        spec_copy["data"] = rows if isinstance(rows, list) else []
        viz_specs_with_data.append(spec_copy)

    return {
        "type": "report",
        "title": state.get("prompt", "Rapport analytique")[:80],
        "narrative": state.get("narrative", ""),
        "insights": state.get("insights", []),
        "recommendations": state.get("recommendations", []),
        "viz_specs": viz_specs_with_data,
        "aggregates": aggregates,
        "qa_score": state.get("qa_report", {}).get("confidence_score", 1.0),
        "html_url": state.get("report_urls", {}).get("html_url", ""),
    }
