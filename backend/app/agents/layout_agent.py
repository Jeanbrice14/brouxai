from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import structlog
from jinja2 import Environment, FileSystemLoader, select_autoescape

from app.agents.base_agent import BaseAgent
from app.config import settings
from app.pipeline.state import PipelineState
from app.services.storage import upload_file

logger = structlog.get_logger(__name__)

# Répertoire des templates Jinja2
_TEMPLATES_DIR = Path(__file__).parent.parent / "templates"

_DEFAULT_COLORS = {
    "primary": "#1E3A8A",
    "positive": "#16A34A",
    "negative": "#DC2626",
}

# Noms de mois en français
_MONTHS_FR = [
    "",
    "janvier",
    "février",
    "mars",
    "avril",
    "mai",
    "juin",
    "juillet",
    "août",
    "septembre",
    "octobre",
    "novembre",
    "décembre",
]


def _format_date(language: str = "fr") -> str:
    """Formate la date courante selon la langue."""
    now = datetime.now()
    if language == "fr":
        return f"{now.day} {_MONTHS_FR[now.month]} {now.year}"
    return now.strftime("%B %d, %Y")


def _split_paragraphs(narrative: str) -> list[str]:
    """Découpe le narrative en paragraphes sur double newline."""
    paragraphs = [p.strip() for p in narrative.split("\n\n") if p.strip()]
    # Fallback : si pas de double newline, retourner le texte entier
    if not paragraphs and narrative.strip():
        paragraphs = [narrative.strip()]
    return paragraphs


def _extract_colors(brand_kit: dict) -> dict:
    """Extrait les couleurs du brand_kit avec les valeurs par défaut."""
    bk_colors = brand_kit.get("colors", {})
    return {
        "primary": bk_colors.get("primary", _DEFAULT_COLORS["primary"]),
        "positive": bk_colors.get("positive", _DEFAULT_COLORS["positive"]),
        "negative": bk_colors.get("negative", _DEFAULT_COLORS["negative"]),
    }


def _build_report_ref(tenant_id: str, report_id: str) -> str:
    """Construit la référence S3 pour le rapport HTML."""
    return f"s3://{settings.storage_bucket}/{tenant_id}/reports/{report_id}/report.html"


def _build_html_url(tenant_id: str, report_id: str) -> str:
    """Construit l'URL publique du rapport HTML."""
    key = f"{tenant_id}/reports/{report_id}/report.html"
    endpoint = settings.storage_endpoint.rstrip("/")
    bucket = settings.storage_bucket
    return f"{endpoint}/{bucket}/{key}"


class LayoutAgent(BaseAgent):
    """Agent 8 — Assemble le rendu final selon response_type.

    Mode "table" ou "chart" : pas d'HTML — appelle response_formatter directement.
    Mode "report" : comportement original — génère HTML complet + upload MinIO.
    """

    name = "layout_agent"

    def __init__(self) -> None:
        self._jinja_env = Environment(
            loader=FileSystemLoader(str(_TEMPLATES_DIR)),
            autoescape=select_autoescape(["html", "j2"]),
        )

    async def run(self, state: PipelineState) -> PipelineState:
        from app.agents.response_formatter import (
            format_chart_response,
            format_report_response,
            format_table_response,
        )

        log = logger.bind(report_id=state.get("report_id"))
        response_type = state.get("response_type", "report")

        # ── Mode simple (table ou chart) ─────────────────────────────────────
        if response_type == "table":
            state["response"] = format_table_response(state)
            state["report_urls"] = {}
            state["status"] = "complete"
            log.info("layout_simple_complete", response_type="table")
            return state

        if response_type == "chart":
            state["response"] = format_chart_response(state)
            state["report_urls"] = {}
            state["status"] = "complete"
            log.info("layout_simple_complete", response_type="chart")
            return state

        # ── Mode rapport complet ──────────────────────────────────────────────
        brand_kit = state.get("brand_kit", {})
        language = brand_kit.get("language", "fr")

        colors = _extract_colors(brand_kit)
        paragraphs = _split_paragraphs(state.get("narrative", ""))
        viz_specs = state.get("viz_specs", [])
        aggregates = state.get("aggregates", {})

        viz_specs_with_data = []
        for spec in viz_specs:
            spec_copy = dict(spec)
            data_key = spec_copy.get("data_key", "")
            rows = aggregates.get(data_key, [])
            spec_copy["data"] = rows if isinstance(rows, list) else []
            viz_specs_with_data.append(spec_copy)

        viz_specs_json = json.dumps(viz_specs_with_data, ensure_ascii=False, default=str)

        context = {
            "language": language,
            "prompt": state.get("prompt", "Rapport analytique"),
            "report_id": state.get("report_id", ""),
            "report_date": _format_date(language),
            "colors": colors,
            "logo_url": brand_kit.get("logo_url", ""),
            "company_name": brand_kit.get("company_name", ""),
            "paragraphs": paragraphs,
            "viz_specs": viz_specs_with_data,
            "viz_specs_json": viz_specs_json,
        }

        template = self._jinja_env.get_template("report.html.j2")
        html_content = template.render(**context)

        log.info("layout_html_rendered", size=len(html_content), paragraphs=len(paragraphs))

        tenant_id = state.get("tenant_id", "")
        report_id = state.get("report_id", "")
        ref = _build_report_ref(tenant_id, report_id)
        html_bytes = html_content.encode("utf-8")
        await upload_file(ref, html_bytes, content_type="text/html; charset=utf-8")

        html_url = _build_html_url(tenant_id, report_id)
        log.info("layout_uploaded", html_url=html_url)

        state["report_urls"] = {"html_url": html_url}
        state["response"] = format_report_response(state)
        state["status"] = "complete"
        return state
