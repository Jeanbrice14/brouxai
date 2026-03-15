"""Tests unitaires Sprint 3 — SchemaLinkingAgent complet.

LLM et storage sont systématiquement mockés (aucun appel réseau réel).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from app.agents.schema_linking_agent import SchemaLinkingAgent
from app.pipeline.state import initial_state

# ── Constantes ────────────────────────────────────────────────────────────────

REF_VENTES = "s3://narr8-dev/uploads/ventes.csv"
REF_CLIENTS = "s3://narr8-dev/uploads/clients.csv"

# DataFrame ventes : contient client_id qui référence clients.id
DF_VENTES = pd.DataFrame(
    {
        "date": pd.to_datetime(
            ["2024-01-05", "2024-02-03", "2024-03-01", "2024-04-02", "2024-05-07"]
        ),
        "client_id": ["CLI001", "CLI002", "CLI003", "CLI004", "CLI005"],
        "ca_ht": [12500.0, 8750.5, 21300.0, 5400.0, 16800.0],
        "region": ["Nord", "Sud", "Est", "Ouest", "Ile-de-France"],
    }
)

# DataFrame clients : id est la PK
DF_CLIENTS = pd.DataFrame(
    {
        "id": ["CLI001", "CLI002", "CLI003", "CLI004", "CLI005", "CLI006", "CLI007"],
        "nom": [
            "Dupont SA",
            "Martin & Fils",
            "Bernard Tech",
            "Petit Commerce",
            "Grand IDF",
            "Lopez",
            "Moreau",
        ],
        "region": ["Nord", "Ouest", "Est", "Ouest", "Ile-de-France", "Sud", "Nord"],
        "segment": ["PME", "ETI", "PME", "TPE", "ETI", "PME", "ETI"],
    }
)

LLM_DESCRIPTION = {"description": "Une vente appartient à un client."}


def _make_state(refs: list[str], metadata_files: dict | None = None) -> dict:
    state = initial_state(
        tenant_id="tenant-test",
        user_id="user-test",
        report_id="report-test",
        prompt="Analyse les ventes par client",
        raw_data_refs=refs,
    )
    if metadata_files is not None:
        state["metadata"] = {"files": metadata_files}
    return state


def _df_side_effect(df_map: dict):
    """Retourne une fonction side_effect pour mocker read_dataframe."""

    def _fn(ref: str) -> pd.DataFrame:
        return df_map.get(ref, pd.DataFrame({"col": ["a", "b"]}))

    return _fn


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_single_file_no_relations():
    """Fichier unique → pas de relations, pas de HITL."""
    state = _make_state(
        refs=[REF_VENTES],
        metadata_files={REF_VENTES: {"row_count": 5, "col_count": 4, "columns": {}, "grain": ""}},
    )
    agent = SchemaLinkingAgent()

    # Pas d'appel externe attendu en mode fichier unique
    result = await agent(state)

    assert result["status"] != "error", f"Erreurs: {result['errors']}"
    assert result["schema"]["relations"] == []
    assert result["schema"]["multi_table"] is False
    assert result["hitl_pending"] is False
    assert result["hitl_checkpoint"] is None


@pytest.mark.asyncio
async def test_detects_foreign_key():
    """Relation évidente client_id → id détectée avec coverage 100% et cardinalité N:1."""
    state = _make_state(refs=[REF_VENTES, REF_CLIENTS])
    agent = SchemaLinkingAgent()

    with (
        patch(
            "app.agents.schema_linking_agent.read_dataframe",
            AsyncMock(
                side_effect=_df_side_effect({REF_VENTES: DF_VENTES, REF_CLIENTS: DF_CLIENTS})
            ),
        ),
        patch(
            "app.agents.schema_linking_agent.call_llm_json",
            AsyncMock(return_value=LLM_DESCRIPTION),
        ),
    ):
        result = await agent(state)

    assert result["status"] != "error", f"Erreurs: {result['errors']}"
    relations = result["schema"]["relations"]
    assert len(relations) > 0, "Aucune relation détectée alors que client_id → id est évidente"

    # Chercher la relation client_id → id
    fk_rel = next(
        (r for r in relations if r["col_a"] == "client_id" and r["col_b"] == "id"),
        None,
    )
    assert fk_rel is not None, f"Relation client_id→id non trouvée. Relations: {relations}"
    assert fk_rel["coverage"] > 0.70, f"Coverage insuffisant: {fk_rel['coverage']}"
    assert fk_rel["cardinality"] == "N:1", f"Cardinalité inattendue: {fk_rel['cardinality']}"
    assert fk_rel["description"] != "", "Description LLM manquante"


@pytest.mark.asyncio
async def test_triggers_hitl_on_multi_files():
    """Multi-fichiers → HITL toujours déclenché."""
    state = _make_state(refs=[REF_VENTES, REF_CLIENTS])
    agent = SchemaLinkingAgent()

    with (
        patch(
            "app.agents.schema_linking_agent.read_dataframe",
            AsyncMock(
                side_effect=_df_side_effect({REF_VENTES: DF_VENTES, REF_CLIENTS: DF_CLIENTS})
            ),
        ),
        patch(
            "app.agents.schema_linking_agent.call_llm_json",
            AsyncMock(return_value=LLM_DESCRIPTION),
        ),
    ):
        result = await agent(state)

    assert result["hitl_pending"] is True
    assert result["hitl_checkpoint"] == "cp2_schema"


@pytest.mark.asyncio
async def test_high_orphan_rate_flagged():
    """Relation avec 20% de valeurs orphelines → high_orphan_rate == True."""
    # client_id contient CLI999 qui n'existe pas dans clients → orphan_rate ≈ 1/6 ≈ 16.7%
    df_ventes_orphan = pd.DataFrame(
        {
            "client_id": ["CLI001", "CLI002", "CLI003", "CLI004", "CLI005", "CLI999"],
            "ca_ht": [1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0],
        }
    )
    df_clients_small = pd.DataFrame(
        {
            "id": ["CLI001", "CLI002", "CLI003", "CLI004", "CLI005"],
            "nom": ["A", "B", "C", "D", "E"],
        }
    )

    state = _make_state(refs=[REF_VENTES, REF_CLIENTS])
    agent = SchemaLinkingAgent()

    with (
        patch(
            "app.agents.schema_linking_agent.read_dataframe",
            AsyncMock(
                side_effect=_df_side_effect(
                    {REF_VENTES: df_ventes_orphan, REF_CLIENTS: df_clients_small}
                )
            ),
        ),
        patch(
            "app.agents.schema_linking_agent.call_llm_json",
            AsyncMock(return_value=LLM_DESCRIPTION),
        ),
    ):
        result = await agent(state)

    relations = result["schema"]["relations"]
    fk_rel = next(
        (r for r in relations if r["col_a"] == "client_id" and r["col_b"] == "id"),
        None,
    )
    assert fk_rel is not None, "Relation client_id→id non détectée"
    assert fk_rel["high_orphan_rate"] is True, (
        f"high_orphan_rate devrait être True, orphan_rate={fk_rel['orphan_rate']}"
    )
    assert fk_rel["orphan_rate"] > 0.05


@pytest.mark.asyncio
async def test_no_relation_detected_different_types():
    """DataFrames sans valeurs compatibles → aucune relation, HITL quand même (multi-fichiers)."""
    # Colonnes purement métriques vs colonnes catégorielles sans intersection de valeurs
    df_meteo = pd.DataFrame(
        {
            "temperature": [20.5, 21.3, 22.1, 23.0, 24.5],
            "pressure": [1013.0, 1014.0, 1015.0, 1016.0, 1017.0],
        }
    )
    df_geo = pd.DataFrame(
        {
            "city": ["Paris", "Lyon", "Marseille", "Bordeaux", "Nice"],
            "country": ["France", "France", "France", "France", "France"],
        }
    )

    state = _make_state(refs=[REF_VENTES, REF_CLIENTS])
    agent = SchemaLinkingAgent()

    with (
        patch(
            "app.agents.schema_linking_agent.read_dataframe",
            AsyncMock(side_effect=_df_side_effect({REF_VENTES: df_meteo, REF_CLIENTS: df_geo})),
        ),
        patch(
            "app.agents.schema_linking_agent.call_llm_json",
            AsyncMock(return_value=LLM_DESCRIPTION),
        ),
    ):
        result = await agent(state)

    assert result["status"] != "error", f"Erreurs: {result['errors']}"
    assert result["schema"]["relations"] == [], (
        f"Des relations ont été détectées à tort: {result['schema']['relations']}"
    )
    # HITL quand même car multi-fichiers
    assert result["hitl_pending"] is True
    assert result["hitl_checkpoint"] == "cp2_schema"
