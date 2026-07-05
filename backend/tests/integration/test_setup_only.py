"""Tests d'intégration — state["setup_only"] (connexion initiale sans vraie question).

Régression réelle : après validation de CP1, le pipeline enchaînait data_agent/insight_agent
sur le prompt générique de connexion ("Connecte-toi au modèle et analyse le schéma"),
produisant une analyse inventée et pouvant même redéclencher un CP3 avant que l'utilisateur
ait posé sa première vraie question. setup_only=True fait s'arrêter le pipeline juste après
metadata_agent (+ schema_linking_agent en mode csv), avec state["response"] = {"type":
"setup_complete"} — signal déjà attendu côté frontend (app/chat/[sessionId]/page.tsx).

Aucun vrai appel LLM ni MCP : tout est mocké aux frontières.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from app.agents.data_agent import DataAgent
from app.agents.insight_agent import InsightAgent
from app.agents.schema_linking_agent import SchemaLinkingAgent
from app.pipeline.graph import build_pipeline
from app.pipeline.state import initial_state

_PROMPT = "Connecte-toi au modèle et analyse le schéma"

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

_PBI_MODEL_INFO = {
    "tables": {
        "Sales": {
            "columns": {"Region": {"dataType": "String", "description": "Région de vente"}},
            "description": "Table des ventes",
        }
    },
    "measures": {"Total Sales": {"description": "Somme des ventes", "dataType": "Double"}},
    "relations": [],
}
# Toutes les colonnes/mesures ont une description → confidence 0.95 ≥ seuil → pas de CP1
# (on veut ici tester le chemin "pas de HITL", pas le chemin hitl_wait).


def _spy_on_run(agent_cls, counts: dict, key: str):
    original_run = agent_cls.run

    async def _spy(self, state):
        counts[key] += 1
        return await original_run(self, state)

    return patch.object(agent_cls, "run", _spy)


@pytest.mark.asyncio
async def test_powerbi_local_setup_only_stops_after_metadata():
    """setup_only=True + powerbi_local : le pipeline s'arrête juste après metadata_agent,
    sans jamais invoquer data_agent/insight_agent."""
    state = initial_state(
        tenant_id="tenant-setup-test",
        user_id="user-setup-test",
        report_id="report-setup-test",
        prompt=_PROMPT,
        data_source="powerbi_local",
        pbix_file_name="AdventureWorks",
        setup_only=True,
    )
    counts = {"data": 0, "insight": 0}

    fake_client = AsyncMock()
    fake_client.connect_to_desktop_file = AsyncMock(return_value={})
    fake_client.get_model_metadata = AsyncMock(return_value=_PBI_MODEL_INFO)

    with (
        patch("app.agents.metadata_agent.get_powerbi_client", return_value=fake_client),
        patch("app.agents.metadata_agent.index_schema", AsyncMock(return_value={"reindexed": False, "n_fields": 0, "hash": "x"})),
        _spy_on_run(DataAgent, counts, "data"),
        _spy_on_run(InsightAgent, counts, "insight"),
    ):
        result = await build_pipeline().ainvoke(state)

    assert counts["data"] == 0, "DataAgent n'aurait jamais dû être invoqué (setup_only=True)"
    assert counts["insight"] == 0, "InsightAgent n'aurait jamais dû être invoqué (setup_only=True)"
    assert result["status"] == "complete", f"Erreurs: {result['errors']}"
    assert result["hitl_pending"] is False, "Pas de CP1 attendu (descriptions présentes)"
    assert result["response"] == {"type": "setup_complete"}
    assert result["aggregates"] == {}, "aggregates doit rester vide — data_agent n'a jamais tourné"


@pytest.mark.asyncio
async def test_csv_setup_only_stops_after_schema_linking():
    """setup_only=True + csv : metadata ET schema_linking_agent tournent (relations
    nécessaires pour CP2), mais data_agent n'est jamais invoqué."""
    state = initial_state(
        tenant_id="tenant-setup-test",
        user_id="user-setup-test",
        report_id="report-setup-test-csv",
        prompt=_PROMPT,
        data_source="csv",
        raw_data_refs=["s3://narr8-dev/uploads/ventes.csv"],
        setup_only=True,
    )
    counts = {"schema": 0, "data": 0, "insight": 0}

    with (
        patch("app.agents.metadata_agent.read_dataframe", AsyncMock(return_value=_CSV_DF)),
        patch("app.agents.metadata_agent.call_llm_json", AsyncMock(return_value=_CSV_LLM_RESPONSE)),
        patch("app.agents.schema_linking_agent.read_dataframe", AsyncMock(return_value=_CSV_DF)),
        _spy_on_run(SchemaLinkingAgent, counts, "schema"),
        _spy_on_run(DataAgent, counts, "data"),
        _spy_on_run(InsightAgent, counts, "insight"),
    ):
        result = await build_pipeline().ainvoke(state)

    assert counts["schema"] == 1, "SchemaLinkingAgent aurait dû tourner (relations pour CP2)"
    assert counts["data"] == 0, "DataAgent n'aurait jamais dû être invoqué (setup_only=True)"
    assert counts["insight"] == 0, "InsightAgent n'aurait jamais dû être invoqué (setup_only=True)"
    assert result["status"] == "complete", f"Erreurs: {result['errors']}"
    assert result["response"] == {"type": "setup_complete"}


@pytest.mark.asyncio
async def test_powerbi_local_without_setup_only_still_runs_full_analysis():
    """Contrôle négatif : sans setup_only (message normal), data_agent/insight_agent
    tournent bien — la régression inverse (tout court-circuiter par erreur) serait pire
    que le bug d'origine."""
    state = initial_state(
        tenant_id="tenant-setup-test",
        user_id="user-setup-test",
        report_id="report-setup-test-normal",
        prompt="Quel est le total des ventes par région ?",
        data_source="powerbi_local",
        pbix_file_name="AdventureWorks",
        setup_only=False,
    )
    counts = {"data": 0, "insight": 0}

    fake_client = AsyncMock()
    fake_client.connect_to_desktop_file = AsyncMock(return_value={})
    fake_client.get_model_metadata = AsyncMock(return_value=_PBI_MODEL_INFO)
    fake_client.execute_dax = AsyncMock(return_value={"columns": ["Region", "Ventes"], "rows": [{"Region": "Nord", "Ventes": 100.0}]})

    with (
        patch("app.agents.metadata_agent.get_powerbi_client", return_value=fake_client),
        patch("app.agents.data_agent.get_powerbi_client", return_value=fake_client),
        patch("app.agents.metadata_agent.index_schema", AsyncMock(return_value={"reindexed": False, "n_fields": 0, "hash": "x"})),
        patch("app.agents.data_agent.retrieve_relevant_fields", AsyncMock(return_value=[])),
        patch("app.agents.data_agent.call_llm_json", AsyncMock(return_value={"queries": [{"key": "ventes_par_region", "dax": "EVALUATE ..."}]})),
        patch("app.agents.insight_agent.call_llm_json", AsyncMock(return_value={"insights": []})),
        patch("app.agents.storytelling_agent.call_llm_json", AsyncMock(return_value={"title": "Ventes", "summary": "Résumé."})),
        patch("app.agents.viz_agent.call_llm_json", AsyncMock(return_value={})),
        patch("app.agents.qa_agent.call_llm_json", AsyncMock(return_value={"issues": []})),
        patch("app.agents.layout_agent.upload_file", AsyncMock()),
        _spy_on_run(DataAgent, counts, "data"),
        _spy_on_run(InsightAgent, counts, "insight"),
    ):
        result = await build_pipeline().ainvoke(state)

    assert counts["data"] == 1, "DataAgent aurait dû être invoqué (setup_only=False)"
    assert counts["insight"] == 1, "InsightAgent aurait dû être invoqué (setup_only=False)"
    assert result["response"] != {"type": "setup_complete"}
