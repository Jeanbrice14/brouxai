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
    """Agent 8 — Assemble le rendu HTML interactif final.

    Étapes :
        A. Préparer le contexte template (couleurs, date, paragraphes, viz_specs JSON).
        B. Rendre le template Jinja2 report.html.j2.
        C. Stocker le HTML dans MinIO : {tenant_id}/reports/{report_id}/report.html
        D. Écrire state["report_urls"] = {"html_url": ...} et state["status"] = "complete".

    Input  : state["narrative"] + state["viz_specs"] + state["aggregates"] + state["brand_kit"]
    Output : state["report_urls"]  — {html_url} stocké dans MinIO/R2
    Outils : Jinja2 (templates HTML), Recharts (rendu frontend interactif)
    V0     : HTML uniquement — export PDF reporté v1
    """

    name = "layout_agent"

    def __init__(self) -> None:
        self._jinja_env = Environment(
            loader=FileSystemLoader(str(_TEMPLATES_DIR)),
            autoescape=select_autoescape(["html", "j2"]),
        )

    async def run(self, state: PipelineState) -> PipelineState:
        log = logger.bind(report_id=state.get("report_id"))

        brand_kit = state.get("brand_kit", {})
        language = brand_kit.get("language", "fr")

        # ── Étape A : contexte template ──────────────────────────────────────
        colors = _extract_colors(brand_kit)
        paragraphs = _split_paragraphs(state.get("narrative", ""))
        viz_specs = state.get("viz_specs", [])

        # Sérialisation JSON sûre des viz_specs pour l'injection JS
        viz_specs_json = json.dumps(viz_specs, ensure_ascii=False)

        context = {
            "language": language,
            "prompt": state.get("prompt", "Rapport analytique"),
            "report_id": state.get("report_id", ""),
            "report_date": _format_date(language),
            "colors": colors,
            "logo_url": brand_kit.get("logo_url", ""),
            "company_name": brand_kit.get("company_name", ""),
            "paragraphs": paragraphs,
            "viz_specs": viz_specs,
            "viz_specs_json": viz_specs_json,
        }

        # ── Étape B : rendu Jinja2 ───────────────────────────────────────────
        template = self._jinja_env.get_template("report.html.j2")
        html_content = template.render(**context)

        log.info("layout_html_rendered", size=len(html_content), paragraphs=len(paragraphs))

        # ── Étape C : stockage MinIO ─────────────────────────────────────────
        tenant_id = state.get("tenant_id", "")
        report_id = state.get("report_id", "")

        ref = _build_report_ref(tenant_id, report_id)
        html_bytes = html_content.encode("utf-8")

        await upload_file(ref, html_bytes, content_type="text/html; charset=utf-8")

        html_url = _build_html_url(tenant_id, report_id)
        log.info("layout_uploaded", html_url=html_url)

        # ── Étape D : état final ─────────────────────────────────────────────
        state["report_urls"] = {"html_url": html_url}
        state["status"] = "complete"
        return state
