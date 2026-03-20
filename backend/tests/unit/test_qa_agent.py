"""Tests unitaires Sprint 6 — QAAgent complet.

Le LLM est systématiquement mocké (aucun appel réseau réel).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.agents.qa_agent import QAAgent
from app.pipeline.state import initial_state

# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_AGGREGATES = {
    "by_region": [
        {"region": "Nord", "ca_ht": 12500.0},
        {"region": "IDF", "ca_ht": 42100.0},
    ]
}

SAMPLE_INSIGHTS = [
    {
        "title": "Domination IDF",
        "description": "L'IDF génère 42% du CA.",
        "type": "highlight",
        "confidence": 0.92,
        "supporting_data": "ca_ht IDF: 42100",
        "impact": "high",
    }
]

SAMPLE_VIZ_SPECS = [
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
    }
]

# Narrative > 100 caractères sans trop de chiffres
_GOOD_NARRATIVE = (
    "L'analyse des ventes régionales révèle des disparités marquées entre les zones. "
    "L'Île-de-France génère la plus grande part du chiffre d'affaires total. "
    "Cette concentration mérite une attention particulière dans la stratégie commerciale. "
    "Des actions correctives sont recommandées pour rééquilibrer la performance régionale."
)


def _make_state(
    narrative: str = _GOOD_NARRATIVE,
    insights: list | None = None,
    aggregates: dict | None = None,
    viz_specs: list | None = None,
) -> dict:
    state = initial_state(
        tenant_id="tenant-test",
        user_id="user-test",
        report_id="report-test",
        prompt="Analyse les ventes par région",
        raw_data_refs=["s3://narr8-dev/uploads/ventes.csv"],
    )
    state["narrative"] = narrative
    state["insights"] = insights if insights is not None else SAMPLE_INSIGHTS
    state["aggregates"] = aggregates if aggregates is not None else SAMPLE_AGGREGATES
    state["viz_specs"] = viz_specs if viz_specs is not None else SAMPLE_VIZ_SPECS
    return state


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_nominal_all_checks_pass():
    """State complet et cohérent → confidence_score == 1.0, status == 'ok', pas de HITL."""
    state = _make_state()
    agent = QAAgent()

    mock_llm = AsyncMock(return_value={"issues": []})

    with patch("app.agents.qa_agent.call_llm_json", mock_llm):
        result = await agent(state)

    assert result["status"] != "error", f"Erreurs pipeline: {result['errors']}"
    qa = result["qa_report"]

    assert qa["confidence_score"] == 1.0, (
        f"confidence_score attendu 1.0, obtenu {qa['confidence_score']}"
    )
    assert qa["status"] == "ok", f"status attendu 'ok', obtenu {qa['status']}"
    assert result["hitl_pending"] is False, "HITL ne doit pas être déclenché"
    assert qa["issues"] == [], f"Aucune issue attendue, obtenu: {qa['issues']}"


@pytest.mark.asyncio
async def test_high_severity_issue_lowers_score():
    """Narrative vide → issue high severity → confidence_score ≤ 0.75."""
    state = _make_state(narrative="")  # narrative vide → check déterministe high
    agent = QAAgent()

    mock_llm = AsyncMock(return_value={"issues": []})

    with patch("app.agents.qa_agent.call_llm_json", mock_llm):
        result = await agent(state)

    qa = result["qa_report"]
    assert qa["confidence_score"] <= 0.75, (
        f"confidence_score devrait être ≤ 0.75 avec narrative vide, obtenu {qa['confidence_score']}"
    )
    assert qa["status"] in ("warning", "error"), (
        f"status devrait être 'warning' ou 'error', obtenu {qa['status']}"
    )


@pytest.mark.asyncio
async def test_triggers_hitl_below_threshold():
    """3 issues high severity → confidence_score < 0.80 → HITL déclenché."""
    state = _make_state()
    agent = QAAgent()

    llm_issues = [
        {"type": "hallucination", "passage": "Chiffre inventé.", "severity": "high"},
        {"type": "inconsistency", "passage": "Contradiction données.", "severity": "high"},
        {"type": "unsupported_claim", "passage": "Affirmation sans source.", "severity": "high"},
    ]
    mock_llm = AsyncMock(return_value={"issues": llm_issues})

    with patch("app.agents.qa_agent.call_llm_json", mock_llm):
        result = await agent(state)

    qa = result["qa_report"]
    assert qa["confidence_score"] < 0.80, (
        f"confidence_score devrait être < 0.80, obtenu {qa['confidence_score']}"
    )
    assert result["hitl_pending"] is True, "HITL devrait être déclenché"
    assert result["hitl_checkpoint"] == "cp3_insights"


@pytest.mark.asyncio
async def test_deterministic_checks_no_llm_cost():
    """check_narrative_not_empty détecte l'issue sans LLM si narrative vide."""
    state = _make_state(narrative="")
    agent = QAAgent()

    mock_llm = AsyncMock(return_value={"issues": []})

    with patch("app.agents.qa_agent.call_llm_json", mock_llm):
        result = await agent(state)

    qa = result["qa_report"]
    issue_checks = [i.get("check") for i in qa["issues"]]
    assert "check_narrative_not_empty" in issue_checks, (
        f"check_narrative_not_empty devrait être dans les issues. Checks détectés: {issue_checks}"
    )
    # Le LLM ne doit pas être appelé si narrative vide
    assert mock_llm.call_count == 0, (
        f"call_llm_json ne devrait pas être appelé si narrative vide. "
        f"Appelé {mock_llm.call_count} fois."
    )


@pytest.mark.asyncio
async def test_qa_report_structure():
    """qa_report contient exactement les champs requis."""
    state = _make_state()
    agent = QAAgent()

    mock_llm = AsyncMock(return_value={"issues": []})

    with patch("app.agents.qa_agent.call_llm_json", mock_llm):
        result = await agent(state)

    qa = result["qa_report"]
    required_fields = {"confidence_score", "status", "issues", "checks_run", "checks_passed"}
    for field in required_fields:
        assert field in qa, f"Champ '{field}' manquant dans qa_report: {qa}"

    assert isinstance(qa["confidence_score"], float), "confidence_score doit être un float"
    assert qa["status"] in ("ok", "warning", "error"), f"status invalide: {qa['status']}"
    assert isinstance(qa["issues"], list), "issues doit être une liste"
    assert isinstance(qa["checks_run"], int), "checks_run doit être un int"
    assert isinstance(qa["checks_passed"], int), "checks_passed doit être un int"
    assert 0.0 <= qa["confidence_score"] <= 1.0, (
        f"confidence_score hors bornes: {qa['confidence_score']}"
    )
