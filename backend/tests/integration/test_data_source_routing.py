"""Test d'intégration léger — routage MetadataAgent → SchemaLinkingAgent/DataAgent
selon state["data_source"] (cf. graph.py::_route_after_metadata).

Vérifie sur le graphe réel (build_pipeline()), pas sur la fonction de routage isolée :
- mode "csv"          → SchemaLinkingAgent est invoqué (spy call_count == 1)
- mode "powerbi_local" → SchemaLinkingAgent n'est JAMAIS invoqué (spy call_count == 0),
                          et DataAgent l'est bien (spy call_count == 1), prouvant que le
                          routage va directement à DataAgent plutôt que de simplement
                          échouer avant.

Aucun vrai appel LLM ni MCP : tout est mocké aux frontières (read_dataframe, call_llm*,
get_powerbi_client, upload_file). Intent choisi volontairement "simple_query" (mots-clés
déterministes "quel est"/"total") — mais data_agent → insight_agent est désormais
inconditionnel quel que soit l'intent (le insight/narrative doit apparaître pour tout
prompt, cf. graph.py), donc insight/storytelling/viz/qa sont mockés ici aussi (réponses
"sûres" : haute confiance, pas de variation > 20 %, pour ne pas déclencher CP3/CP5 et
polluer les assertions qui portent sur metadata/schema/data uniquement).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from app.agents.data_agent import DataAgent
from app.agents.metadata_agent import MetadataAgent
from app.agents.schema_linking_agent import SchemaLinkingAgent
from app.pipeline.graph import build_pipeline
from app.pipeline.state import initial_state

_PROMPT = "Quel est le total des ventes par région ?"  # simple_query déterministe

# ── Mode csv ────────────────────────────────────────────────────────────────────

_CSV_DF = pd.DataFrame({"region": ["Nord", "Sud"], "ca_ht": [12500.0, 8750.5]})

_CSV_LLM_RESPONSE = {
    "semantic_name": "Valeur test",
    "description": "Description test",
    "type": "numeric",
    "unit": "",
    "confidence": 0.90,
    "is_key_candidate": False,
    "grain": "Une ligne représente une vente.",
}

_CSV_CACHED_AGGREGATES = {"by_region": [{"region": "Nord", "ca_ht": 12500.0}]}


# ── Mode powerbi_local ───────────────────────────────────────────────────────────

_PBI_MODEL_INFO = {
    "tables": {
        "Sales": {
            "columns": {
                "Region": {"dataType": "String", "description": "Région de vente"},
            },
            "description": "Table des ventes",
        }
    },
    "measures": {
        "Total Sales": {"description": "Somme des ventes", "dataType": "Double"},
    },
    "relations": [],
}
# Toutes les colonnes/mesures ont une description → confidence 0.95 ≥ seuil → pas de CP1,
# ce qui est nécessaire pour prouver le routage CONTINUE (sinon on ne teste que le
# chemin HITL_WAIT, qui saute schema_linking_agent de toute façon, HITL ou pas).

_PBI_DAX_RESPONSE = {
    "queries": [{"key": "ventes_par_region", "dax": "EVALUATE SUMMARIZECOLUMNS(...)"}]
}
_PBI_EXECUTE_RESULT = {"columns": ["Region", "Ventes"], "rows": [{"Region": "Nord", "Ventes": 100.0}]}

# ── Mocks partagés insight/storytelling/viz/qa (data_agent → insight_agent inconditionnel,
# cf. graph.py) — réponses "sûres" qui ne déclenchent jamais de HITL, pour garder ces tests
# concentrés sur le routage metadata/schema/data, pas sur le contenu insight/narrative.
_SAFE_INSIGHTS_RESPONSE = {
    "insights": [
        {
            "title": "Ventes par région",
            "description": "Le Nord contribue au chiffre d'affaires observé.",
            "type": "highlight",
            "confidence": 0.90,
            "supporting_data": "region: Nord",
            "impact": "medium",
        }
    ]
}
_SAFE_STORYTELLING_TABLE_RESPONSE = {"title": "Ventes par région", "summary": "Le Nord contribue au CA observé."}
_SAFE_VIZ_SPEC_RESPONSE = {
    "chart_type": "bar",
    "title": "Ventes par région",
    "data_key": "by_region",
    "x": "region",
    "y": "ca_ht",
    "color_by": None,
    "colors": {"primary": "#1E3A8A", "positive": "#16A34A", "negative": "#DC2626"},
    "annotations": [],
    "insight_ref": "Ventes par région",
}
_SAFE_QA_RESPONSE = {"issues": [], "confidence_score": 0.95}


def _make_state(data_source: str, **kwargs) -> dict:
    return initial_state(
        tenant_id="tenant-routing-test",
        user_id="user-routing-test",
        report_id="report-routing-test",
        prompt=_PROMPT,
        data_source=data_source,
        **kwargs,
    )


def _spy_on_run(agent_cls, counts: dict, key: str):
    """Patch `agent_cls.run` avec une fonction espionne qui délègue à l'implémentation réelle.

    Utilise une vraie fonction (pas un Mock) exprès : un Mock n'implémente pas le protocole
    descripteur, donc `self.run(state)` sur une instance ne le lierait pas correctement à
    `self` (TypeError "missing 1 required positional argument"). Une fonction plain Python
    est un descripteur — le binding self fonctionne normalement une fois patchée sur la classe.
    """
    original_run = agent_cls.run

    async def _spy(self, state):
        counts[key] += 1
        return await original_run(self, state)

    return patch.object(agent_cls, "run", _spy)


@pytest.mark.asyncio
async def test_csv_mode_invokes_schema_linking_agent():
    """Mode csv : SchemaLinkingAgent est bien appelé par le graphe (comportement inchangé)."""
    state = _make_state("csv", raw_data_refs=["s3://narr8-dev/uploads/ventes.csv"])
    counts = {"schema": 0}

    with (
        patch("app.agents.metadata_agent.read_dataframe", AsyncMock(return_value=_CSV_DF)),
        patch("app.agents.metadata_agent.call_llm_json", AsyncMock(return_value=_CSV_LLM_RESPONSE)),
        patch("app.agents.data_agent.get_cache", AsyncMock(return_value=_CSV_CACHED_AGGREGATES)),
        patch("app.agents.data_agent.set_cache", AsyncMock()),
        patch("app.agents.insight_agent.call_llm_json", AsyncMock(return_value=_SAFE_INSIGHTS_RESPONSE)),
        patch("app.agents.storytelling_agent.call_llm_json", AsyncMock(return_value=_SAFE_STORYTELLING_TABLE_RESPONSE)),
        patch("app.agents.viz_agent.call_llm_json", AsyncMock(return_value=_SAFE_VIZ_SPEC_RESPONSE)),
        patch("app.agents.qa_agent.call_llm_json", AsyncMock(return_value=_SAFE_QA_RESPONSE)),
        patch("app.agents.layout_agent.upload_file", AsyncMock()),
        _spy_on_run(SchemaLinkingAgent, counts, "schema"),
    ):
        result = await build_pipeline().ainvoke(state)

    assert counts["schema"] == 1, "SchemaLinkingAgent aurait dû être invoqué en mode csv"
    assert result["status"] != "error", f"Erreurs inattendues: {result['errors']}"
    assert result["schema"] != {}, "state['schema'] non écrit — SchemaLinkingAgent n'a pas tourné"


@pytest.mark.asyncio
async def test_powerbi_local_mode_never_invokes_schema_linking_agent():
    """Mode powerbi_local : SchemaLinkingAgent n'est JAMAIS invoqué, DataAgent si."""
    state = _make_state("powerbi_local", pbix_file_name="AdventureWorks")
    counts = {"schema": 0, "data": 0}

    fake_client = AsyncMock()
    fake_client.connect_to_desktop_file = AsyncMock(return_value={})
    fake_client.get_model_metadata = AsyncMock(return_value=_PBI_MODEL_INFO)
    fake_client.execute_dax = AsyncMock(return_value=_PBI_EXECUTE_RESULT)

    with (
        patch("app.agents.metadata_agent.get_powerbi_client", return_value=fake_client),
        patch("app.agents.data_agent.get_powerbi_client", return_value=fake_client),
        patch("app.agents.data_agent.call_llm_json", AsyncMock(return_value=_PBI_DAX_RESPONSE)),
        patch("app.agents.layout_agent.upload_file", AsyncMock()),
        patch("app.agents.metadata_agent.index_schema", AsyncMock(return_value={"reindexed": False, "n_fields": 0, "hash": "x"})),
        patch("app.agents.data_agent.retrieve_relevant_fields", AsyncMock(return_value=[])),
        patch("app.agents.insight_agent.call_llm_json", AsyncMock(return_value=_SAFE_INSIGHTS_RESPONSE)),
        patch("app.agents.storytelling_agent.call_llm_json", AsyncMock(return_value=_SAFE_STORYTELLING_TABLE_RESPONSE)),
        patch("app.agents.viz_agent.call_llm_json", AsyncMock(return_value=_SAFE_VIZ_SPEC_RESPONSE)),
        patch("app.agents.qa_agent.call_llm_json", AsyncMock(return_value=_SAFE_QA_RESPONSE)),
        _spy_on_run(SchemaLinkingAgent, counts, "schema"),
        _spy_on_run(DataAgent, counts, "data"),
    ):
        result = await build_pipeline().ainvoke(state)

    assert counts["schema"] == 0, (
        "SchemaLinkingAgent n'aurait JAMAIS dû être invoqué en mode powerbi_local "
        f"(appelé {counts['schema']} fois)"
    )
    assert counts["data"] == 1, "DataAgent aurait dû être invoqué directement après MetadataAgent"
    assert result["status"] != "error", f"Erreurs inattendues: {result['errors']}"
    assert result["hitl_pending"] is False, "Pas de CP1 attendu (descriptions présentes)"
    assert result["schema"] == {}, "state['schema'] doit rester vide — jamais écrit par SchemaLinkingAgent"
    assert result["aggregates"] != {}, "state['aggregates'] non écrit — DataAgent n'a pas tourné"


@pytest.mark.asyncio
async def test_powerbi_local_reused_metadata_skips_metadata_agent_entirely():
    """Régression CP1-à-chaque-message : quand metadata est déjà populé dans le state initial
    (simulateur de reports.py::generate_report_powerbi avec base_report_id) et data_source
    est powerbi_local, _route_after_intent doit envoyer directement à DataAgent — sans
    repasser par MetadataAgent (qui redéclencherait CP1 en recalculant les confidences à
    froid). Avant le fix, _route_after_intent exigeait aussi state['schema'], qui reste
    TOUJOURS vide en mode powerbi_local (schema_linking_agent jamais exécuté) — donc ce
    court-circuit ne se produisait jamais pour ce data_source."""
    state = _make_state("powerbi_local", pbix_file_name="AdventureWorks")
    # Simule reports.py::generate_report_powerbi avec base_report_id fourni — metadata et
    # semantic_model_info sont copiés depuis un rapport précédent AVANT que le pipeline démarre.
    state["metadata"] = {"files": {"powerbi://Sales": {"columns": {}}}}
    state["semantic_model_info"] = _PBI_MODEL_INFO
    counts = {"metadata": 0, "data": 0}

    fake_client = AsyncMock()
    fake_client.execute_dax = AsyncMock(return_value=_PBI_EXECUTE_RESULT)

    with (
        patch("app.agents.data_agent.get_powerbi_client", return_value=fake_client),
        patch("app.agents.data_agent.call_llm_json", AsyncMock(return_value=_PBI_DAX_RESPONSE)),
        patch("app.agents.layout_agent.upload_file", AsyncMock()),
        patch("app.agents.data_agent.retrieve_relevant_fields", AsyncMock(return_value=[])),
        patch("app.agents.insight_agent.call_llm_json", AsyncMock(return_value=_SAFE_INSIGHTS_RESPONSE)),
        patch("app.agents.storytelling_agent.call_llm_json", AsyncMock(return_value=_SAFE_STORYTELLING_TABLE_RESPONSE)),
        patch("app.agents.viz_agent.call_llm_json", AsyncMock(return_value=_SAFE_VIZ_SPEC_RESPONSE)),
        patch("app.agents.qa_agent.call_llm_json", AsyncMock(return_value=_SAFE_QA_RESPONSE)),
        _spy_on_run(MetadataAgent, counts, "metadata"),
        _spy_on_run(DataAgent, counts, "data"),
    ):
        result = await build_pipeline().ainvoke(state)

    assert counts["metadata"] == 0, (
        "MetadataAgent n'aurait pas dû être ré-invoqué : metadata déjà présent dans le "
        f"state initial (appelé {counts['metadata']} fois) — CP1 se serait redéclenché."
    )
    assert counts["data"] == 1, "DataAgent aurait dû être invoqué directement"
    assert result["status"] != "error", f"Erreurs inattendues: {result['errors']}"
    assert result["hitl_pending"] is False, "Aucun CP1 attendu — metadata_agent n'a jamais tourné"
