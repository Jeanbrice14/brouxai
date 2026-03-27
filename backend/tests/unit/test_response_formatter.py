from __future__ import annotations

from app.agents.response_formatter import (
    format_chart_response,
    format_report_response,
    format_table_response,
)
from app.pipeline.state import initial_state


def _make_state(**kwargs) -> dict:
    state = initial_state(
        tenant_id="test",
        user_id="test",
        report_id="test-id",
        prompt="Top 5 clients par CA",
        raw_data_refs=[],
    )
    state.update(kwargs)
    return state


def test_table_response_limited_to_20_rows():
    rows = [{"client": f"Client {i}", "ca": i * 1000} for i in range(30)]
    state = _make_state(aggregates={"ca_clients": rows}, narrative="Résumé court.")
    result = format_table_response(state)
    assert result["type"] == "table"
    assert len(result["rows"]) <= 20
    assert result["total_rows"] <= 20
    assert "columns" in result
    assert "summary" in result


def test_table_response_summary_is_one_sentence():
    narrative = "Première phrase. Deuxième phrase. Troisième phrase."
    state = _make_state(
        aggregates={"data": [{"a": 1}]},
        narrative=narrative,
    )
    result = format_table_response(state)
    # La summary doit être 1 seule phrase
    sentences = [s for s in result["summary"].split(".") if s.strip()]
    assert len(sentences) <= 1


def test_chart_response_has_viz_spec():
    viz_spec = {
        "chart_type": "bar",
        "title": "CA par région",
        "data_key": "ca_region",
        "x": "region",
        "y": "ca",
    }
    state = _make_state(
        viz_specs=[viz_spec],
        aggregates={"ca_region": [{"region": "Nord", "ca": 100}]},
        narrative="Tendance positive. Point notable.",
        response_type="chart",
    )
    result = format_chart_response(state)
    assert result["type"] == "chart"
    assert result["viz_spec"] == viz_spec
    assert result["chart_type"] == "bar"
    assert len(result["data"]) > 0
    assert "caption" in result


def test_report_response_has_narrative():
    state = _make_state(
        narrative="Narration complète du rapport.",
        viz_specs=[],
        qa_report={"confidence_score": 0.95},
        report_urls={"html_url": "http://localhost/report.html"},
    )
    result = format_report_response(state)
    assert result["type"] == "report"
    assert result["narrative"] == "Narration complète du rapport."
    assert result["qa_score"] == 0.95
    assert result["html_url"] == "http://localhost/report.html"


def test_storytelling_one_sentence_in_table_mode():
    """format_table_response tronque narrative à 1 phrase."""
    narrative = "Phrase un. Phrase deux. Phrase trois."
    state = _make_state(aggregates={}, narrative=narrative)
    result = format_table_response(state)
    # Doit avoir au plus 1 point final de phrase
    content = result["summary"].rstrip(".")
    assert "." not in content  # pas de point intérieur = 1 seule phrase
