"""Tests unitaires — client MCP Power BI local (powerbi_local_mcp.py).

Le serveur MCP (subprocess npx + Power BI Desktop) est systématiquement mocké,
avec des réponses calquées sur un test réel contre un serveur Power BI Modeling
MCP v0.5.0-beta.11 + Power BI Desktop (jeu de données AdventureWorks) :
- tous les tools exigent {"request": {...}}
- connexion en 2 étapes : ListLocalInstances puis Connect (pas de "connect par fichier" direct)
- column_operations/measure_operations "List" sont groupés par table
- dax_query_operations "Execute" renvoie ses lignes en CSV (EmbeddedResource), pas en JSON
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from app.services.powerbi_local_mcp import (
    PowerBIConnectionError,
    PowerBIDaxExecutionError,
    PowerBIDesktopNotFoundError,
    PowerBILocalMCPClient,
)

# ── Helpers ───────────────────────────────────────────────────────────────────


def _json_result(payload: dict, is_error: bool = False):
    """Simule un mcp.types.CallToolResult dont le contenu texte est du JSON."""
    import json

    content = [SimpleNamespace(type="text", text=json.dumps(payload))]
    return SimpleNamespace(content=content, structuredContent=None, isError=is_error)


def _error_result(message: str):
    content = [SimpleNamespace(type="text", text=message)]
    return SimpleNamespace(content=content, structuredContent=None, isError=True)


def _csv_result(csv_text: str):
    """Simule le CallToolResult réel d'un dax_query_operations Execute : content[0].text == '{}'
    et les données dans un bloc EmbeddedResource (content[1].resource.text, CSV)."""
    text_block = SimpleNamespace(type="text", text="{}")
    resource = SimpleNamespace(text=csv_text)
    resource_block = SimpleNamespace(type="resource", resource=resource)
    return SimpleNamespace(content=[text_block, resource_block], structuredContent=None, isError=False)


class _FakeSession:
    """Simule un mcp.ClientSession déjà initialisé."""

    def __init__(self, call_tool_side_effect=None, tools=None):
        self.initialize = AsyncMock()
        self.list_tools = AsyncMock(return_value=SimpleNamespace(tools=tools or []))
        self.call_tool = AsyncMock(side_effect=call_tool_side_effect or [])


def _fake_client_session_cls(session: _FakeSession):
    class _FakeClientSessionCtx:
        def __init__(self, *_args, **_kwargs):
            pass

        async def __aenter__(self):
            return session

        async def __aexit__(self, *_exc):
            return False

    return _FakeClientSessionCtx


@asynccontextmanager
async def _fake_stdio_client(_server_params):
    yield (AsyncMock(), AsyncMock())


def _patched(session: _FakeSession):
    return (
        patch("mcp.client.stdio.stdio_client", _fake_stdio_client),
        patch("mcp.ClientSession", _fake_client_session_cls(session)),
    )


_LIST_LOCAL_INSTANCES = {
    "message": "Found 1 local PowerBI Desktop and Analysis Services instances",
    "operation": "ListLocalInstances",
    "data": [
        {
            "processId": 10652,
            "port": 59991,
            "connectionString": "data source=localhost:59991;Application Name=MCP-PBIModeling",
            "parentProcessName": "PBIDesktop",
            "parentWindowTitle": "AdventureWorks Report_FINAL2 1 1",
            "startTime": "2026-07-02T18:45:33+02:00",
        }
    ],
}

_CONNECT_OK = {
    "message": "Connection established successfully",
    "operation": "Connect",
    "data": "PBIDesktop-AdventureWorks Report_FINAL2 1 1-59991",
}


# ── connect_to_desktop_file (flux réel : ListLocalInstances -> Connect) ────────


@pytest.mark.asyncio
async def test_connect_to_desktop_file_success():
    session = _FakeSession(
        call_tool_side_effect=[_json_result(_LIST_LOCAL_INSTANCES), _json_result(_CONNECT_OK)]
    )
    client = PowerBILocalMCPClient()

    p1, p2 = _patched(session)
    with p1, p2:
        result = await client.connect_to_desktop_file("AdventureWorks")

    assert result == _CONNECT_OK
    assert client._connected_file == "AdventureWorks"

    # Vérifie que Connect a bien reçu la connectionString de l'instance matchée, enveloppée
    # dans {"request": {...}} comme l'exige le serveur réel.
    connect_call = session.call_tool.await_args_list[1]
    assert connect_call.args[0] == "connection_operations"
    assert connect_call.args[1] == {
        "request": {
            "operation": "Connect",
            "connectionString": "data source=localhost:59991;Application Name=MCP-PBIModeling",
        }
    }


@pytest.mark.asyncio
async def test_connect_to_desktop_file_no_matching_instance():
    session = _FakeSession(call_tool_side_effect=[_json_result(_LIST_LOCAL_INSTANCES)])
    client = PowerBILocalMCPClient()

    p1, p2 = _patched(session)
    with p1, p2, pytest.raises(PowerBIDesktopNotFoundError, match="doit être ouvert"):
        await client.connect_to_desktop_file("FichierInexistant")


@pytest.mark.asyncio
async def test_connect_to_desktop_file_connect_call_fails():
    session = _FakeSession(
        call_tool_side_effect=[
            _json_result(_LIST_LOCAL_INSTANCES),
            _error_result("Server name must be present in connection string"),
        ]
    )
    client = PowerBILocalMCPClient()

    p1, p2 = _patched(session)
    with p1, p2, pytest.raises(PowerBIDesktopNotFoundError):
        await client.connect_to_desktop_file("AdventureWorks")


@pytest.mark.asyncio
async def test_session_start_failure_mentions_npx():
    """Si le subprocess ne démarre pas (npx/node absent), le message doit être actionnable."""
    client = PowerBILocalMCPClient()

    with patch("mcp.client.stdio.stdio_client", Mock(side_effect=RuntimeError("command not found"))):
        with pytest.raises(PowerBIConnectionError, match="Node.js/npx"):
            await client.connect_to_desktop_file("AdventureWorks")


# ── get_model_metadata (réponses groupées par table, confirmé réel) ────────────


@pytest.mark.asyncio
async def test_get_model_metadata_transforms_tables_measures_relations():
    session = _FakeSession(
        call_tool_side_effect=[
            _json_result(
                {
                    "message": "Found 2 tables",
                    "operation": "List",
                    "data": [
                        {"name": "Sales", "columnCount": 1},
                        {"name": "Customers", "columnCount": 1},
                    ],
                }
            ),
            _json_result(
                {
                    "message": "Found columns",
                    "operation": "List",
                    "data": [
                        {
                            "tableName": "Sales",
                            "columns": [{"name": "Amount", "dataType": "Double"}],
                        },
                        {
                            "tableName": "Customers",
                            "columns": [{"name": "Region", "dataType": "String"}],
                        },
                    ],
                }
            ),
            _json_result(
                {
                    "message": "Found measures",
                    "operation": "List",
                    "data": [
                        {
                            "tableName": "Sales",
                            "measures": [{"name": "Total Sales", "formatString": "#,0"}],
                        }
                    ],
                }
            ),
            _json_result(
                {
                    "message": "Found 1 relationships",
                    "operation": "List",
                    "data": [
                        {
                            "fromTable": "Sales",
                            "fromColumn": "CustomerId",
                            "toTable": "Customers",
                            "toColumn": "Id",
                        }
                    ],
                }
            ),
        ]
    )
    client = PowerBILocalMCPClient()
    client._connected_file = "AdventureWorks"  # connexion testée séparément

    p1, p2 = _patched(session)
    with p1, p2:
        metadata = await client.get_model_metadata()

    assert set(metadata["tables"].keys()) == {"Sales", "Customers"}
    assert metadata["tables"]["Sales"]["columns"]["Amount"]["dataType"] == "Double"
    assert metadata["tables"]["Customers"]["columns"]["Region"]["dataType"] == "String"
    assert "Total Sales" in metadata["measures"]
    assert metadata["relations"] == [
        {"fromTable": "Sales", "fromColumn": "CustomerId", "toTable": "Customers", "toColumn": "Id"}
    ]

    # column_operations doit recevoir toutes les tables en un seul appel batché
    columns_call = session.call_tool.await_args_list[1]
    assert columns_call.args[1] == {
        "request": {"operation": "List", "filter": {"tableNames": ["Sales", "Customers"]}}
    }


@pytest.mark.asyncio
async def test_get_model_metadata_requires_prior_connection():
    client = PowerBILocalMCPClient()
    with pytest.raises(PowerBIConnectionError, match="connect_to_desktop_file"):
        await client.get_model_metadata()


# ── execute_dax (résultat réel en CSV, pas en JSON) ────────────────────────────


@pytest.mark.asyncio
async def test_execute_dax_parses_csv_result():
    session = _FakeSession(
        call_tool_side_effect=[_csv_result("Sales[Region],[Total Sales]\r\nNord,12500\r\nSud,8750.5\r\n")]
    )
    client = PowerBILocalMCPClient()

    p1, p2 = _patched(session)
    with p1, p2:
        result = await client.execute_dax(
            "EVALUATE SUMMARIZECOLUMNS('Sales'[Region], \"Total Sales\", [Total Sales])"
        )

    assert result["columns"] == ["Region", "Total Sales"]
    assert result["rows"] == [
        {"Region": "Nord", "Total Sales": 12500},
        {"Region": "Sud", "Total Sales": 8750.5},
    ]


@pytest.mark.asyncio
async def test_execute_dax_coerces_locale_decimal_comma():
    """Confirmé contre un vrai serveur (AdventureWorks) : la culture de la connexion
    Analysis Services formate les décimaux avec une virgule ("23642495,0952"), pas un
    point — la valeur CSV est correctement quotée par le serveur pour désambiguïser
    de la virgule séparateur de champ (repr brut capturé : `Bikes,"23642495,0952"`).
    Sans coercition, ces valeurs restaient des chaînes Python — inexploitables par un
    graphique en aval (Number("23642495,0952") vaut NaN en JS)."""
    session = _FakeSession(
        call_tool_side_effect=[
            _csv_result(
                'Product Categories Lookup[CategoryName],[Ventes]\r\n'
                'Bikes,"23642495,0952"\r\n'
                'Clothing,"365418,6171"\r\n'
            )
        ]
    )
    client = PowerBILocalMCPClient()

    p1, p2 = _patched(session)
    with p1, p2:
        result = await client.execute_dax(
            "EVALUATE SUMMARIZECOLUMNS('Product Categories Lookup'[CategoryName], "
            '"Ventes", SUM(\'Sales Data\'[Revenue]))'
        )

    assert result["rows"] == [
        {"CategoryName": "Bikes", "Ventes": 23642495.0952},
        {"CategoryName": "Clothing", "Ventes": 365418.6171},
    ]
    assert all(isinstance(row["Ventes"], float) for row in result["rows"])


@pytest.mark.asyncio
async def test_execute_dax_single_row_measure():
    session = _FakeSession(call_tool_side_effect=[_csv_result("[lignes]\r\n10\r\n")])
    client = PowerBILocalMCPClient()

    p1, p2 = _patched(session)
    with p1, p2:
        result = await client.execute_dax("EVALUATE ROW(\"lignes\", COUNTROWS('Territory Lookup'))")

    assert result == {"columns": ["lignes"], "rows": [{"lignes": 10}]}


@pytest.mark.asyncio
async def test_execute_dax_failure_raises_explicit_error():
    session = _FakeSession(call_tool_side_effect=[Exception("DAX syntax error near EVALUATE")])
    client = PowerBILocalMCPClient()

    p1, p2 = _patched(session)
    with p1, p2, pytest.raises(PowerBIDaxExecutionError, match="Échec d'exécution DAX"):
        await client.execute_dax("EVALUATE BROKEN")


@pytest.mark.asyncio
async def test_execute_dax_reports_tool_level_error():
    session = _FakeSession(call_tool_side_effect=[_error_result("invalid measure")])
    client = PowerBILocalMCPClient()

    p1, p2 = _patched(session)
    with p1, p2, pytest.raises(PowerBIDaxExecutionError):
        await client.execute_dax("EVALUATE ROW(\"x\", [Inexistant])")


# ── Session réutilisée ─────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_session_started_once_across_multiple_calls():
    session = _FakeSession(
        call_tool_side_effect=[
            _csv_result("[count]\r\n1\r\n"),
            _csv_result("[count]\r\n2\r\n"),
        ]
    )
    client = PowerBILocalMCPClient()

    p1, p2 = _patched(session)
    with p1, p2:
        await client.execute_dax("EVALUATE ROW(\"count\", 1)")
        await client.execute_dax("EVALUATE ROW(\"count\", 2)")

    session.initialize.assert_awaited_once()
    assert session.call_tool.await_count == 2


# ── Toutes les requêtes sont enveloppées dans {"request": ...} ─────────────────


@pytest.mark.asyncio
async def test_all_calls_wrap_arguments_in_request_key():
    """Confirmé réel : chaque tool échoue avec 'missing required parameter request' sinon."""
    session = _FakeSession(call_tool_side_effect=[_csv_result("[x]\r\n1\r\n")])
    client = PowerBILocalMCPClient()

    p1, p2 = _patched(session)
    with p1, p2:
        await client.execute_dax("EVALUATE ROW(\"x\", 1)")

    call_args = session.call_tool.await_args
    assert "request" in call_args.args[1]
    assert call_args.args[1]["request"]["operation"] == "Execute"
