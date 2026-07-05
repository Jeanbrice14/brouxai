"""Tests unitaires Sprint 4 — DataAgent (SQL/DuckDB).

LLM, storage et cache sont systématiquement mockés (aucun appel réseau réel).
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from app.agents.data_agent import DataAgent, _strip_order_by
from app.pipeline.state import initial_state

# ── Constantes ────────────────────────────────────────────────────────────────

REF_VENTES = "s3://narr8-dev/uploads/ventes.csv"

# DataFrame réaliste — contient des valeurs distinctives pour test_never_passes_raw_data_to_llm
DF_VENTES = pd.DataFrame(
    {
        "region": ["Nord", "Sud", "Est", "Ouest", "Ile-de-France"],
        "ca_ht": [12500.00, 8750.50, 21300.00, 5400.00, 16800.00],
        "client_id": ["CLI001", "CLI002", "CLI003", "CLI004", "CLI005"],
        "quantite": [10, 7, 18, 4, 14],
    }
)

# Valeurs brutes du CSV que le LLM NE doit PAS voir
_RAW_VALUES = ["12500", "8750", "CLI001", "CLI002", "CLI003"]

# Réponses LLM mockées (format DuckDB SQL)
_SIMPLE_LLM_RESPONSE = {
    "queries": [{"key": "summary", "sql": "SELECT COUNT(*) AS count FROM ventes"}]
}
# range(1000) est une table-function DuckDB — génère 1000 lignes sans données sources
_BIG_LLM_RESPONSE = {
    "queries": [{"key": "big_table", "sql": "SELECT * FROM range(1000)"}]
}
# Table inexistante → DuckDB lève une exception → déclenchement du fallback
_BROKEN_LLM_RESPONSE = {
    "queries": [{"key": "broken", "sql": "SELECT * FROM table_inexistante_xyz"}]
}


def _make_state(refs: list[str] | None = None) -> dict:
    return initial_state(
        tenant_id="tenant-test",
        user_id="user-test",
        report_id="report-test",
        prompt="Analyse les ventes par région",
        raw_data_refs=refs or [REF_VENTES],
    )


@contextmanager
def _patch_all(
    *,
    df: pd.DataFrame = DF_VENTES,
    llm_response: dict = _SIMPLE_LLM_RESPONSE,
    cached: dict | None = None,
):
    """Context manager qui mocke les 4 dépendances externes du DataAgent."""
    with (
        patch("app.agents.data_agent.read_dataframe", AsyncMock(return_value=df)),
        patch(
            "app.agents.data_agent.call_llm_json",
            AsyncMock(return_value=llm_response),
        ),
        patch("app.agents.data_agent.get_cache", AsyncMock(return_value=cached)),
        patch("app.agents.data_agent.set_cache", AsyncMock()),
    ):
        yield


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_nominal_generates_aggregates():
    """CSV standard : les agrégats sont produits et sérialisés en list-of-dicts."""
    state = _make_state()
    agent = DataAgent()

    with _patch_all(df=DF_VENTES, llm_response=_SIMPLE_LLM_RESPONSE, cached=None):
        result = await agent(state)

    assert result["status"] != "error", f"Erreurs inattendues: {result['errors']}"
    aggregates = result["aggregates"]
    assert aggregates, "state['aggregates'] ne doit pas être vide"

    # Les valeurs doivent être des list-of-dicts
    for key, rows in aggregates.items():
        assert isinstance(rows, list), f"{key} devrait être une liste"
        if rows:
            assert isinstance(rows[0], dict), f"{key}[0] devrait être un dict"


@pytest.mark.asyncio
async def test_never_passes_raw_data_to_llm():
    """Le prompt LLM ne contient JAMAIS les valeurs brutes du CSV."""
    state = _make_state()
    agent = DataAgent()

    mock_llm = AsyncMock(return_value={"queries": []})

    with (
        patch("app.agents.data_agent.read_dataframe", AsyncMock(return_value=DF_VENTES)),
        patch("app.agents.data_agent.call_llm_json", mock_llm),
        patch("app.agents.data_agent.get_cache", AsyncMock(return_value=None)),
        patch("app.agents.data_agent.set_cache", AsyncMock()),
    ):
        await agent(state)

    assert mock_llm.called, "call_llm_json aurait dû être appelé"
    call_kwargs = mock_llm.call_args
    prompt_sent: str = call_kwargs.kwargs.get("prompt") or call_kwargs.args[0]

    # Vérifier que les valeurs brutes ne sont pas dans le prompt
    for raw_val in _RAW_VALUES:
        assert raw_val not in prompt_sent, (
            f"Valeur brute '{raw_val}' trouvée dans le prompt LLM — règle violée !"
        )

    # Vérifier que le schéma structurel est présent (noms de colonnes et de tables)
    assert "ca_ht" in prompt_sent, "Le nom de colonne 'ca_ht' devrait être dans le prompt"
    assert "ventes" in prompt_sent, "Le nom de table 'ventes' devrait être dans le prompt"


@pytest.mark.asyncio
async def test_exec_failure_triggers_fallback():
    """SQL invalide → pas de crash, fallback basique, warning dans errors."""
    state = _make_state()
    agent = DataAgent()

    with _patch_all(df=DF_VENTES, llm_response=_BROKEN_LLM_RESPONSE, cached=None):
        result = await agent(state)

    # Le pipeline ne doit pas planter
    assert result["status"] != "error", "BaseAgent ne devrait pas avoir catchée une erreur"

    # Les agrégats de fallback doivent être présents
    aggregates = result["aggregates"]
    assert aggregates, "Le fallback devrait produire des agrégats non vides"

    # Au moins un agrégat de fallback (count, sums ou means)
    fallback_keys = [k for k in aggregates if "count" in k or "sums" in k or "means" in k]
    assert fallback_keys, f"Aucun agrégat de fallback trouvé : {list(aggregates.keys())}"

    # Un warning doit être enregistré dans errors
    assert result["errors"], "state['errors'] devrait contenir au moins un warning"
    errors_str = " ".join(result["errors"])
    assert "fallback" in errors_str.lower() or "sql" in errors_str.lower(), (
        f"Aucun warning fallback/sql dans errors: {result['errors']}"
    )


@pytest.mark.asyncio
async def test_cache_hit_skips_llm():
    """Cache hit → call_llm_json n'est PAS appelé, les agrégats du cache sont utilisés."""
    cached_aggregates = {"from_cache": [{"region": "Nord", "ca_ht": 12500.0}]}
    state = _make_state()
    agent = DataAgent()

    mock_llm = AsyncMock(return_value={"queries": []})

    with (
        patch("app.agents.data_agent.read_dataframe", AsyncMock(return_value=DF_VENTES)),
        patch("app.agents.data_agent.call_llm_json", mock_llm),
        patch("app.agents.data_agent.get_cache", AsyncMock(return_value=cached_aggregates)),
        patch("app.agents.data_agent.set_cache", AsyncMock()),
    ):
        result = await agent(state)

    # LLM ne doit PAS avoir été appelé
    mock_llm.assert_not_called()

    # Les agrégats du cache sont utilisés tels quels
    assert result["aggregates"] == cached_aggregates


@pytest.mark.asyncio
async def test_result_limited_to_500_rows():
    """Un résultat de 1000 lignes est limité à 500 lignes dans les agrégats."""
    state = _make_state()
    agent = DataAgent()

    # range(1000) est une table-function DuckDB valide, sans besoin de la table ventes
    with _patch_all(df=DF_VENTES, llm_response=_BIG_LLM_RESPONSE, cached=None):
        result = await agent(state)

    aggregates = result["aggregates"]
    assert "big_table" in aggregates, (
        f"'big_table' manquant dans aggregates. Clés: {list(aggregates.keys())}"
    )
    assert len(aggregates["big_table"]) == 500, (
        f"Attendu 500 lignes, obtenu {len(aggregates['big_table'])}"
    )


# ── Mode powerbi_local (DAX) ────────────────────────────────────────────────────

_PBI_METADATA_FILES = {
    "powerbi://Sales": {
        "row_count": None,
        "col_count": 1,
        "columns": {
            "Region": {"semantic_name": "Région", "type": "String", "kind": "column", "confidence": 0.95}
        },
        "grain": "",
        "avg_confidence": 0.95,
    },
    "powerbi://__measures__": {
        "row_count": None,
        "col_count": 1,
        "columns": {
            "Total Sales": {
                "semantic_name": "Total Sales",
                "type": "measure",
                "kind": "measure",
                "confidence": 0.95,
            }
        },
        "grain": "Mesures DAX du modèle sémantique",
        "avg_confidence": 0.95,
    },
}


def _make_powerbi_state() -> dict:
    state = initial_state(
        tenant_id="tenant-test",
        user_id="user-test",
        report_id="report-test",
        prompt="Total des ventes par région",
        data_source="powerbi_local",
        pbix_file_name="ventes.pbix",
    )
    state["metadata"] = {"files": _PBI_METADATA_FILES, "source": "powerbi_local"}
    state["semantic_model_info"] = {"tables": {}, "measures": {}, "relations": []}
    return state


@pytest.mark.asyncio
async def test_powerbi_mode_executes_generated_dax():
    """Mode powerbi_local : le DAX généré par le LLM est exécuté via le client MCP, pas DuckDB."""
    state = _make_powerbi_state()
    agent = DataAgent()

    fake_client = AsyncMock()
    fake_client.execute_dax = AsyncMock(
        return_value={"columns": ["region", "ca"], "rows": [{"region": "Nord", "ca": 12500.0}]}
    )
    dax_response = {
        "queries": [
            {
                "key": "ca_par_region",
                "dax": "EVALUATE SUMMARIZECOLUMNS('Sales'[Region], \"ca\", [Total Sales])",
            }
        ]
    }

    with (
        patch("app.agents.data_agent.get_powerbi_client", return_value=fake_client),
        patch("app.agents.data_agent.call_llm_json", AsyncMock(return_value=dax_response)),
        # RAG non mocké explicitement ferait un vrai appel embedding/DB (schema_rag.py
        # catch ses propres erreurs, mais on veut un test isolé, pas un fallback via échec réseau).
        patch("app.agents.data_agent.retrieve_relevant_fields", AsyncMock(return_value=[])),
    ):
        result = await agent(state)

    assert result["status"] != "error", f"Erreurs: {result['errors']}"
    assert result["aggregates"]["ca_par_region"] == [{"region": "Nord", "ca": 12500.0}]
    fake_client.execute_dax.assert_awaited_once()
    assert result["dax_queries"], "state['dax_queries'] devrait tracer la requête exécutée"
    assert result["dax_queries"][0]["status"] == "ok"


@pytest.mark.asyncio
async def test_powerbi_mode_reconnects_when_client_not_connected_to_this_file():
    """Régression réelle : quand metadata est réutilisé via base_report_id, MetadataAgent
    (qui appelle normalement connect_to_desktop_file) est sauté par _route_after_intent.
    Le client MCP singleton peut donc ne PAS être connecté (ex: process backend redémarré
    entre deux messages) — DataAgent doit reconnecter lui-même avant d'exécuter du DAX,
    sinon execute_dax échoue avec "no connectionName provided"."""
    state = _make_powerbi_state()
    agent = DataAgent()

    fake_client = AsyncMock()
    fake_client._connected_file = None  # simule un client singleton jamais connecté
    fake_client.connect_to_desktop_file = AsyncMock(return_value={})
    fake_client.execute_dax = AsyncMock(
        return_value={"columns": ["region", "ca"], "rows": [{"region": "Nord", "ca": 12500.0}]}
    )
    dax_response = {
        "queries": [{"key": "ca_par_region", "dax": "EVALUATE SUMMARIZECOLUMNS('Sales'[Region], \"ca\", [Total Sales])"}]
    }

    with (
        patch("app.agents.data_agent.get_powerbi_client", return_value=fake_client),
        patch("app.agents.data_agent.call_llm_json", AsyncMock(return_value=dax_response)),
        patch("app.agents.data_agent.retrieve_relevant_fields", AsyncMock(return_value=[])),
    ):
        result = await agent(state)

    fake_client.connect_to_desktop_file.assert_awaited_once_with(state["pbix_file_name"])
    assert result["status"] != "error", f"Erreurs: {result['errors']}"
    assert result["aggregates"]["ca_par_region"] == [{"region": "Nord", "ca": 12500.0}]


@pytest.mark.asyncio
async def test_powerbi_mode_skips_reconnect_when_already_connected_to_this_file():
    """Cas nominal (session déjà active) : pas de reconnexion redondante si le client est
    déjà connecté au bon fichier — évite un aller-retour MCP inutile à chaque message."""
    state = _make_powerbi_state()
    agent = DataAgent()

    fake_client = AsyncMock()
    fake_client._connected_file = state["pbix_file_name"]
    fake_client.connect_to_desktop_file = AsyncMock(return_value={})
    fake_client.execute_dax = AsyncMock(
        return_value={"columns": ["region", "ca"], "rows": [{"region": "Nord", "ca": 12500.0}]}
    )
    dax_response = {
        "queries": [{"key": "ca_par_region", "dax": "EVALUATE SUMMARIZECOLUMNS('Sales'[Region], \"ca\", [Total Sales])"}]
    }

    with (
        patch("app.agents.data_agent.get_powerbi_client", return_value=fake_client),
        patch("app.agents.data_agent.call_llm_json", AsyncMock(return_value=dax_response)),
        patch("app.agents.data_agent.retrieve_relevant_fields", AsyncMock(return_value=[])),
    ):
        result = await agent(state)

    fake_client.connect_to_desktop_file.assert_not_awaited()
    assert result["status"] != "error", f"Erreurs: {result['errors']}"


@pytest.mark.asyncio
async def test_powerbi_mode_falls_back_on_dax_error():
    """DAX invalide → pas de crash, fallback COUNTROWS/mesures, warning dans errors."""
    from app.services.powerbi_local_mcp import PowerBIDaxExecutionError

    state = _make_powerbi_state()
    agent = DataAgent()

    fake_client = AsyncMock()

    async def _execute_dax(query: str):
        if "COUNTROWS" in query or "Total Sales" in query:
            return {"columns": ["lignes"], "rows": [{"lignes": 42}]}
        raise PowerBIDaxExecutionError("DAX syntax error")

    fake_client.execute_dax = AsyncMock(side_effect=_execute_dax)
    dax_response = {"queries": [{"key": "broken", "dax": "EVALUATE BROKEN_SYNTAX"}]}

    with (
        patch("app.agents.data_agent.get_powerbi_client", return_value=fake_client),
        patch("app.agents.data_agent.call_llm_json", AsyncMock(return_value=dax_response)),
        patch("app.agents.data_agent.retrieve_relevant_fields", AsyncMock(return_value=[])),
    ):
        result = await agent(state)

    assert result["status"] != "error"
    assert result["aggregates"], "Le fallback devrait produire des agrégats non vides"
    errors_str = " ".join(result["errors"])
    assert "fallback" in errors_str.lower() or "dax" in errors_str.lower()


@pytest.mark.asyncio
async def test_powerbi_mode_retries_without_order_by_before_falling_back():
    """Régression réelle observée (AdventureWorks) : une requête DAX SUMMARIZECOLUMNS +
    ORDER BY sur une colonne de tri absente du résultat ('YearMonthSort' non incluse dans
    SUMMARIZECOLUMNS) échoue avec "impossible de déterminer une valeur unique". Plutôt que
    de tomber directement sur le fallback COUNTROWS (perte totale de l'information demandée),
    DataAgent doit retenter la MÊME requête sans son ORDER BY avant d'abandonner."""
    from app.services.powerbi_local_mcp import PowerBIDaxExecutionError

    state = _make_powerbi_state()
    agent = DataAgent()

    fake_client = AsyncMock()
    calls: list[str] = []

    async def _execute_dax(query: str):
        calls.append(query)
        if "ORDER BY" in query:
            raise PowerBIDaxExecutionError(
                "impossible de déterminer une valeur unique pour la colonne 'YearMonthSort'"
            )
        return {
            "columns": ["Year-Month", "ca"],
            "rows": [{"Year-Month": "2024-01", "ca": 1000.0}, {"Year-Month": "2024-02", "ca": 1200.0}],
        }

    fake_client.execute_dax = AsyncMock(side_effect=_execute_dax)
    dax_response = {
        "queries": [
            {
                "key": "evolution_ca",
                "dax": (
                    "EVALUATE SUMMARIZECOLUMNS('Calendar'[Year-Month], \"ca\", [Total Sales]) "
                    "ORDER BY 'Calendar'[YearMonthSort]"
                ),
            }
        ]
    }

    with (
        patch("app.agents.data_agent.get_powerbi_client", return_value=fake_client),
        patch("app.agents.data_agent.call_llm_json", AsyncMock(return_value=dax_response)),
        patch("app.agents.data_agent.retrieve_relevant_fields", AsyncMock(return_value=[])),
    ):
        result = await agent(state)

    assert len(calls) == 2, f"Attendu 2 tentatives (originale + sans ORDER BY), obtenu {len(calls)}"
    assert "ORDER BY" not in calls[1], "La 2e tentative doit avoir retiré le ORDER BY"
    assert result["aggregates"]["evolution_ca"] == [
        {"Year-Month": "2024-01", "ca": 1000.0},
        {"Year-Month": "2024-02", "ca": 1200.0},
    ], "Le résultat groupé de la 2e tentative doit être conservé, pas un fallback COUNTROWS"
    assert result["dax_queries"][-1]["status"] == "ok_retry_no_order_by"


@pytest.mark.asyncio
async def test_powerbi_mode_uses_rag_subset_when_available():
    """Quand le RAG renvoie un sous-ensemble non vide, le prompt DAX ne doit PAS contenir
    le schéma complet — seulement les champs retournés par retrieve_relevant_fields."""
    state = _make_powerbi_state()
    # Ajoute une deuxième table au schéma complet, absente du sous-ensemble RAG simulé,
    # pour distinguer sans ambiguïté "sous-ensemble utilisé" de "schéma complet utilisé".
    state["metadata"]["files"]["powerbi://Customers"] = {
        "row_count": None,
        "col_count": 1,
        "columns": {"Country": {"semantic_name": "Pays", "type": "String", "kind": "column", "confidence": 0.95}},
        "grain": "",
        "avg_confidence": 0.95,
    }
    agent = DataAgent()

    fake_client = AsyncMock()
    fake_client.execute_dax = AsyncMock(
        return_value={"columns": ["region", "ca"], "rows": [{"region": "Nord", "ca": 12500.0}]}
    )
    dax_response = {
        "queries": [
            {
                "key": "ca_par_region",
                "dax": "EVALUATE SUMMARIZECOLUMNS('Sales'[Region], \"ca\", [Total Sales])",
            }
        ]
    }
    rag_subset = [
        {
            "qualified_name": "Sales[Region]",
            "object_type": "column",
            "parent_table": "Sales",
            "description": "Région de vente",
        },
        {
            "qualified_name": "[Total Sales]",
            "object_type": "measure",
            "parent_table": None,
            "description": "Somme des ventes",
        },
    ]

    mock_dax_llm = AsyncMock(return_value=dax_response)

    with (
        patch("app.agents.data_agent.get_powerbi_client", return_value=fake_client),
        patch("app.agents.data_agent.call_llm_json", mock_dax_llm),
        patch("app.agents.data_agent.retrieve_relevant_fields", AsyncMock(return_value=rag_subset)),
    ):
        result = await agent(state)

    assert result["status"] != "error", f"Erreurs: {result['errors']}"

    prompt_sent = mock_dax_llm.call_args.kwargs.get("prompt") or mock_dax_llm.call_args.args[0]
    assert "Sales" in prompt_sent
    assert "Total Sales" in prompt_sent
    # La table Customers n'existe QUE dans le schéma complet, pas dans le sous-ensemble RAG —
    # son absence du prompt prouve que le sous-ensemble a bien remplacé le schéma complet.
    assert "Customers" not in prompt_sent, (
        "Le prompt contient 'Customers', absent du sous-ensemble RAG — le schéma complet "
        "a été utilisé au lieu du sous-ensemble retrouvé."
    )
    assert "Country" not in prompt_sent


# ── _strip_order_by ──────────────────────────────────────────────────────────────


def test_strip_order_by_removes_trailing_clause():
    dax = "EVALUATE SUMMARIZECOLUMNS('Calendar'[Month], \"ca\", [Total Sales]) ORDER BY 'Calendar'[MonthSort]"
    assert _strip_order_by(dax) == "EVALUATE SUMMARIZECOLUMNS('Calendar'[Month], \"ca\", [Total Sales])"


def test_strip_order_by_case_insensitive():
    dax = "EVALUATE ROW(\"x\", 1) order by 'T'[c]"
    assert _strip_order_by(dax) == 'EVALUATE ROW("x", 1)'


def test_strip_order_by_returns_none_when_absent():
    dax = "EVALUATE ROW(\"lignes\", COUNTROWS('Table'))"
    assert _strip_order_by(dax) is None
