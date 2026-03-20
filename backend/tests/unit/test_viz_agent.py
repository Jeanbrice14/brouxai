"""Tests unitaires Sprint 6 — VizAgent complet.

Le LLM est systématiquement mocké (aucun appel réseau réel).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.agents.viz_agent import VizAgent
from app.pipeline.state import initial_state

# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_AGGREGATES = {
    "by_region": [
        {"region": "Nord", "ca_ht": 12500.0},
        {"region": "Sud", "ca_ht": 8750.5},
        {"region": "IDF", "ca_ht": 42100.0},
    ],
    "by_month": [
        {"month": "2024-01", "ca_ht": 21000.0},
        {"month": "2024-02", "ca_ht": 23500.0},
    ],
}

SAMPLE_INSIGHTS = [
    {
        "title": "Domination IDF",
        "description": "L'IDF génère 42% du CA.",
        "type": "highlight",
        "confidence": 0.92,
        "supporting_data": "ca_ht IDF: 42100",
        "impact": "high",
    },
    {
        "title": "Croissance mensuelle",
        "description": "Hausse mensuelle constante.",
        "type": "trend",
        "confidence": 0.85,
        "supporting_data": "ca_ht Jan→Fev: +12%",
        "impact": "medium",
    },
    {
        "title": "Faiblesse Sud",
        "description": "Le Sud reste en retrait.",
        "type": "comparison",
        "confidence": 0.80,
        "supporting_data": "ca_ht Sud: 8750",
        "impact": "low",
    },
]


def _make_state(
    brand_kit: dict | None = None,
    insights: list | None = None,
    aggregates: dict | None = None,
) -> dict:
    state = initial_state(
        tenant_id="tenant-test",
        user_id="user-test",
        report_id="report-test",
        prompt="Analyse les ventes par région",
        raw_data_refs=["s3://narr8-dev/uploads/ventes.csv"],
        brand_kit=brand_kit or {},
    )
    state["insights"] = insights if insights is not None else SAMPLE_INSIGHTS
    state["aggregates"] = aggregates if aggregates is not None else SAMPLE_AGGREGATES
    return state


def _spec_for(insight_title: str, data_key: str = "by_region") -> dict:
    """Retourne une viz_spec valide pour le mock LLM."""
    # Colonnes selon le data_key pour respecter la validation
    x_col = "month" if data_key == "by_month" else "region"
    return {
        "chart_type": "bar",
        "title": f"Graphique — {insight_title}",
        "data_key": data_key,
        "x": x_col,
        "y": "ca_ht",
        "color_by": None,
        "colors": {
            "primary": "#1E3A8A",
            "positive": "#16A34A",
            "negative": "#DC2626",
        },
        "annotations": [],
        "insight_ref": insight_title,
    }


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_generates_viz_spec_per_insight():
    """3 insights → 3 viz_specs avec les champs obligatoires."""
    state = _make_state()
    agent = VizAgent()

    mock_llm = AsyncMock(
        side_effect=[
            _spec_for("Domination IDF"),
            _spec_for("Croissance mensuelle", "by_month"),
            _spec_for("Faiblesse Sud"),
        ]
    )

    with patch("app.agents.viz_agent.call_llm_json", mock_llm):
        result = await agent(state)

    assert result["status"] != "error", f"Erreurs: {result['errors']}"
    viz_specs = result["viz_specs"]
    assert len(viz_specs) == 3, f"Attendu 3 viz_specs, obtenu {len(viz_specs)}"

    required = ("chart_type", "title", "data_key", "x", "y")
    for spec in viz_specs:
        for field in required:
            assert field in spec, f"Champ '{field}' manquant dans viz_spec: {spec}"


@pytest.mark.asyncio
async def test_applies_brand_kit_colors():
    """Les couleurs du brand_kit sont injectées dans chaque viz_spec."""
    brand_kit = {"colors": {"primary": "#FF0000", "positive": "#00FF00", "negative": "#0000FF"}}
    state = _make_state(brand_kit=brand_kit, insights=SAMPLE_INSIGHTS[:2])
    agent = VizAgent()

    mock_llm = AsyncMock(
        side_effect=[
            _spec_for("Domination IDF"),
            _spec_for("Croissance mensuelle", "by_month"),
        ]
    )

    with patch("app.agents.viz_agent.call_llm_json", mock_llm):
        result = await agent(state)

    for spec in result["viz_specs"]:
        assert spec["colors"]["primary"] == "#FF0000", (
            f"Couleur primary incorrecte: {spec['colors']['primary']}"
        )
        assert spec["colors"]["positive"] == "#00FF00"
        assert spec["colors"]["negative"] == "#0000FF"


@pytest.mark.asyncio
async def test_default_colors_when_no_brand_kit():
    """Sans brand_kit, les couleurs par défaut sont appliquées."""
    state = _make_state(brand_kit={}, insights=SAMPLE_INSIGHTS[:1])
    agent = VizAgent()

    mock_llm = AsyncMock(return_value=_spec_for("Domination IDF"))

    with patch("app.agents.viz_agent.call_llm_json", mock_llm):
        result = await agent(state)

    assert result["viz_specs"], "viz_specs ne doit pas être vide"
    spec = result["viz_specs"][0]
    assert spec["colors"]["primary"] == "#1E3A8A", (
        f"Couleur primary par défaut incorrecte: {spec['colors']['primary']}"
    )
    assert spec["colors"]["positive"] == "#16A34A"
    assert spec["colors"]["negative"] == "#DC2626"


@pytest.mark.asyncio
async def test_skips_invalid_data_key():
    """viz_spec avec data_key inexistant → ignorée + warning dans state['errors']."""
    state = _make_state(insights=SAMPLE_INSIGHTS[:2])
    agent = VizAgent()

    invalid_spec = {
        "chart_type": "bar",
        "title": "Graphique invalide",
        "data_key": "inexistant_key",  # n'existe pas dans SAMPLE_AGGREGATES
        "x": "region",
        "y": "ca_ht",
        "color_by": None,
        "colors": _spec_for("x")["colors"],
        "annotations": [],
        "insight_ref": "Domination IDF",
    }
    valid_spec = _spec_for("Croissance mensuelle", "by_month")

    mock_llm = AsyncMock(side_effect=[invalid_spec, valid_spec])

    with patch("app.agents.viz_agent.call_llm_json", mock_llm):
        result = await agent(state)

    viz_specs = result["viz_specs"]
    assert len(viz_specs) == 1, (
        f"La spec avec data_key invalide aurait dû être ignorée. viz_specs: {viz_specs}"
    )
    assert viz_specs[0]["data_key"] == "by_month"

    errors = result.get("errors", [])
    assert any("viz_agent" in e for e in errors), (
        f"Un warning devrait être dans state['errors']. errors: {errors}"
    )


@pytest.mark.asyncio
async def test_maximum_5_viz_specs():
    """8 insights → maximum 5 viz_specs générées."""
    insights_8 = [
        {
            "title": f"Insight {i}",
            "description": f"Description {i}",
            "type": "highlight",
            "confidence": 0.90,
            "supporting_data": f"data {i}",
            "impact": "medium",
        }
        for i in range(8)
    ]
    state = _make_state(insights=insights_8)
    agent = VizAgent()

    mock_llm = AsyncMock(return_value=_spec_for("Insight 0"))

    with patch("app.agents.viz_agent.call_llm_json", mock_llm):
        result = await agent(state)

    viz_specs = result["viz_specs"]
    assert len(viz_specs) <= 5, f"Maximum 5 viz_specs autorisées, obtenu {len(viz_specs)}"
    assert mock_llm.call_count <= 5, (
        f"call_llm_json ne devrait pas être appelé plus de 5 fois, appelé {mock_llm.call_count} fois"
    )
