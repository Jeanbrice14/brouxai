"""Tests unitaires Sprint 7 — LayoutAgent complet.

Le storage est systématiquement mocké (aucun appel réseau réel).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.agents.layout_agent import LayoutAgent
from app.pipeline.state import initial_state

# ── Fixtures ──────────────────────────────────────────────────────────────────

_NARRATIVE_3_PARAS = (
    "L'analyse des ventes révèle des disparités importantes entre les régions.\n\n"
    "L'Île-de-France domine avec 42% du chiffre d'affaires total, "
    "portée par une base clientèle solide et diversifiée.\n\n"
    "Des recommandations ciblées permettront d'améliorer la performance des régions "
    "sous-représentées et d'optimiser l'allocation des ressources commerciales."
)

_VIZ_SPECS = [
    {
        "chart_type": "bar",
        "title": "CA par région",
        "data_key": "by_region",
        "x": "region",
        "y": "ca_ht",
        "color_by": None,
        "colors": {"primary": "#1E3A8A", "positive": "#16A34A", "negative": "#DC2626"},
        "annotations": [],
        "insight_ref": "Domination IDF",
    },
    {
        "chart_type": "line",
        "title": "Évolution mensuelle",
        "data_key": "by_month",
        "x": "month",
        "y": "ca_ht",
        "color_by": None,
        "colors": {"primary": "#1E3A8A", "positive": "#16A34A", "negative": "#DC2626"},
        "annotations": [],
        "insight_ref": "Croissance mensuelle",
    },
]


def _make_state(
    narrative: str = _NARRATIVE_3_PARAS,
    viz_specs: list | None = None,
    brand_kit: dict | None = None,
) -> dict:
    state = initial_state(
        tenant_id="tenant-test",
        user_id="user-test",
        report_id="report-test",
        prompt="Analyse les ventes par région",
        raw_data_refs=["s3://narr8-dev/uploads/ventes.csv"],
        brand_kit=brand_kit or {},
    )
    state["narrative"] = narrative
    state["viz_specs"] = viz_specs if viz_specs is not None else _VIZ_SPECS
    return state


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_generates_html_with_narrative():
    """State complet → le HTML contient le texte narrative et une structure valide."""
    state = _make_state()
    agent = LayoutAgent()

    with patch("app.agents.layout_agent.upload_file", AsyncMock()):
        result = await agent(state)

    assert result["status"] != "error", f"Erreurs pipeline: {result['errors']}"

    html_url = result["report_urls"].get("html_url", "")
    assert html_url, "html_url doit être renseigné"

    # Vérifier la structure HTML minimale en rendant directement le template
    agent2 = LayoutAgent()
    import json as _json

    from app.agents.layout_agent import _extract_colors, _format_date, _split_paragraphs

    brand_kit = {}
    language = "fr"
    colors = _extract_colors(brand_kit)
    paragraphs = _split_paragraphs(_NARRATIVE_3_PARAS)
    viz_specs = _VIZ_SPECS
    context = {
        "language": language,
        "prompt": "Analyse les ventes par région",
        "report_id": "report-test",
        "report_date": _format_date(language),
        "colors": colors,
        "logo_url": "",
        "company_name": "",
        "paragraphs": paragraphs,
        "viz_specs": viz_specs,
        "viz_specs_json": _json.dumps(viz_specs),
    }
    template = agent2._jinja_env.get_template("report.html.j2")
    html = template.render(**context)

    assert "<html" in html, "Le HTML doit contenir <html"
    assert "<body" in html, "Le HTML doit contenir <body"
    assert "</html>" in html, "Le HTML doit se terminer par </html>"
    # Vérifier la présence du contenu narratif (sans apostrophe pour éviter l'échappement HTML)
    assert "analyse des ventes" in html, "Le texte narrative doit être dans le HTML"
    assert len(paragraphs) == 3, f"Attendu 3 paragraphes, obtenu {len(paragraphs)}"


@pytest.mark.asyncio
async def test_injects_viz_specs_in_html():
    """HTML contient window.__VIZ_SPECS__ avec les 2 viz_specs sérialisées."""
    state = _make_state(viz_specs=_VIZ_SPECS)
    agent = LayoutAgent()

    captured_html: list[str] = []

    async def _capture_upload(ref: str, data: bytes, content_type: str = "") -> None:
        captured_html.append(data.decode("utf-8"))

    with patch("app.agents.layout_agent.upload_file", side_effect=_capture_upload):
        result = await agent(state)

    assert result["status"] != "error", f"Erreurs: {result['errors']}"
    assert captured_html, "upload_file n'a pas été appelé"
    html = captured_html[0]

    assert "window.__VIZ_SPECS__" in html, "__VIZ_SPECS__ doit être dans le HTML"

    # Vérifier que les 2 titres de viz sont dans le JSON injecté
    assert "CA par région" in html, "'CA par région' doit être dans le HTML"
    assert "Évolution mensuelle" in html, "'Évolution mensuelle' doit être dans le HTML"


@pytest.mark.asyncio
async def test_applies_brand_kit_colors():
    """La couleur primary du brand_kit apparaît dans le HTML généré."""
    brand_kit = {"colors": {"primary": "#FF0000", "positive": "#00FF00", "negative": "#0000FF"}}
    state = _make_state(brand_kit=brand_kit)
    agent = LayoutAgent()

    captured_html: list[str] = []

    async def _capture_upload(ref: str, data: bytes, content_type: str = "") -> None:
        captured_html.append(data.decode("utf-8"))

    with patch("app.agents.layout_agent.upload_file", side_effect=_capture_upload):
        result = await agent(state)

    assert result["status"] != "error", f"Erreurs: {result['errors']}"
    html = captured_html[0]

    assert "#FF0000" in html, "La couleur primary #FF0000 doit apparaître dans le HTML"
    assert "#00FF00" in html, "La couleur positive #00FF00 doit apparaître dans le HTML"
    assert "#0000FF" in html, "La couleur negative #0000FF doit apparaître dans le HTML"


@pytest.mark.asyncio
async def test_uploads_to_storage_and_returns_url():
    """upload_file est appelé avec text/html, html_url est renseigné, status == 'complete'."""
    state = _make_state()
    agent = LayoutAgent()

    mock_upload = AsyncMock()

    with patch("app.agents.layout_agent.upload_file", mock_upload):
        result = await agent(state)

    assert mock_upload.called, "upload_file aurait dû être appelé"

    # Vérifier le content_type
    call_kwargs = mock_upload.call_args
    content_type = call_kwargs.kwargs.get("content_type") or (
        call_kwargs.args[2] if len(call_kwargs.args) > 2 else None
    )
    assert content_type is not None, "content_type doit être passé à upload_file"
    assert "text/html" in content_type, (
        f"content_type doit contenir 'text/html', obtenu: {content_type}"
    )

    assert "html_url" in result["report_urls"], "html_url doit être dans report_urls"
    assert result["report_urls"]["html_url"], "html_url ne doit pas être vide"
    assert result["status"] == "complete", (
        f"status doit être 'complete', obtenu: {result['status']}"
    )
