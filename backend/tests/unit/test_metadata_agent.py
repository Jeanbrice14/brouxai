"""Tests unitaires Sprint 2 — MetadataAgent complet.

LLM et storage sont systématiquement mockés (aucun appel réseau réel).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from app.agents.metadata_agent import MetadataAgent
from app.pipeline.state import initial_state

# ── Fixtures ──────────────────────────────────────────────────────────────────

REF_CSV = "s3://narr8-dev/uploads/sample_ventes.csv"
REF_XLSX = "s3://narr8-dev/uploads/sample_ventes.xlsx"

SAMPLE_DF = pd.DataFrame(
    {
        "date": pd.to_datetime(
            ["2024-01-05", "2024-02-03", "2024-03-01", "2024-04-02", "2024-05-07"]
        ),
        "region": ["Nord", "Sud", "Est", "Ouest", "Ile-de-France"],
        "ca_ht": [12500.0, 8750.5, 21300.0, 5400.0, 16800.0],
        "client_id": ["CLI001", "CLI002", "CLI003", "CLI004", "CLI005"],
        "produit_ref": ["PROD-A42", "PROD-B17", "PROD-A42", "PROD-C05", "PROD-D91"],
        "quantite": [10, 7, 18, 4, 14],
    }
)

COLUMNS = list(SAMPLE_DF.columns)  # 6 colonnes


def _make_state(ref: str) -> dict:
    return initial_state(
        tenant_id="tenant-test",
        user_id="user-test",
        report_id="report-test",
        prompt="Analyse les ventes par région",
        raw_data_refs=[ref],
    )


def _col_llm_response(confidence: float = 0.95) -> dict:
    """Réponse LLM simulée pour l'inférence d'une colonne."""
    return {
        "semantic_name": "Valeur test",
        "description": "Description de test",
        "type": "numeric",
        "unit": "EUR",
        "confidence": confidence,
        "is_key_candidate": False,
    }


def _grain_response() -> dict:
    return {"grain": "Une ligne représente une vente par client et par date."}


def _build_side_effects(confidence: float = 0.95) -> list[dict]:
    """Construit la liste des retours LLM simulés :
    - 6 appels de colonnes (un par colonne du SAMPLE_DF)
    - 1 appel de grain
    """
    return [_col_llm_response(confidence)] * len(COLUMNS) + [_grain_response()]


# ── Tests ─────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_nominal_case():
    """CSV standard : les bonnes clés sont présentes dans metadata."""
    state = _make_state(REF_CSV)
    agent = MetadataAgent()

    with (
        patch("app.agents.metadata_agent.read_dataframe", AsyncMock(return_value=SAMPLE_DF)),
        patch(
            "app.agents.metadata_agent.call_llm_json",
            AsyncMock(side_effect=_build_side_effects(0.95)),
        ),
    ):
        result = await agent(state)

    assert result["status"] != "error", f"Erreurs: {result['errors']}"
    meta = result["metadata"]

    # Structure de haut niveau
    assert "files" in meta
    file_meta = meta["files"][REF_CSV]

    # Infos fichier
    assert file_meta["row_count"] == len(SAMPLE_DF)
    assert file_meta["col_count"] == len(COLUMNS)

    # Grain renseigné
    assert file_meta["grain"] != ""
    assert "ligne" in file_meta["grain"].lower()

    # avg_confidence calculé
    assert "avg_confidence" in file_meta
    assert 0.0 <= file_meta["avg_confidence"] <= 1.0

    # Toutes les colonnes présentes avec les bonnes clés
    for col in COLUMNS:
        assert col in file_meta["columns"], f"Colonne manquante: {col}"
        col_info = file_meta["columns"][col]
        for key in (
            "dtype",
            "n_unique",
            "null_pct",
            "sample",
            "semantic_name",
            "description",
            "type",
            "unit",
            "confidence",
            "is_key_candidate",
        ):
            assert key in col_info, f"Clé manquante '{key}' pour colonne '{col}'"


@pytest.mark.asyncio
async def test_triggers_hitl_on_low_confidence():
    """Confidence basse (0.40) → le HITL est déclenché."""
    state = _make_state(REF_CSV)
    agent = MetadataAgent()

    with (
        patch("app.agents.metadata_agent.read_dataframe", AsyncMock(return_value=SAMPLE_DF)),
        patch(
            "app.agents.metadata_agent.call_llm_json",
            AsyncMock(side_effect=_build_side_effects(0.40)),
        ),
    ):
        result = await agent(state)

    assert result["hitl_pending"] is True
    assert result["hitl_checkpoint"] == "cp1_metadata"


@pytest.mark.asyncio
async def test_no_hitl_on_high_confidence():
    """Confidence élevée (0.95) → pas de HITL."""
    state = _make_state(REF_CSV)
    agent = MetadataAgent()

    with (
        patch("app.agents.metadata_agent.read_dataframe", AsyncMock(return_value=SAMPLE_DF)),
        patch(
            "app.agents.metadata_agent.call_llm_json",
            AsyncMock(side_effect=_build_side_effects(0.95)),
        ),
    ):
        result = await agent(state)

    assert result["hitl_pending"] is False
    assert result["hitl_checkpoint"] is None


@pytest.mark.asyncio
async def test_handles_excel_file():
    """Fichier .xlsx : l'agent fonctionne identiquement (read_dataframe délègue le format)."""
    state = _make_state(REF_XLSX)
    agent = MetadataAgent()

    with (
        patch(
            "app.agents.metadata_agent.read_dataframe",
            AsyncMock(return_value=SAMPLE_DF),
        ),
        patch(
            "app.agents.metadata_agent.call_llm_json",
            AsyncMock(side_effect=_build_side_effects(0.92)),
        ),
    ):
        result = await agent(state)

    assert result["status"] != "error", f"Erreurs: {result['errors']}"
    assert REF_XLSX in result["metadata"]["files"]

    file_meta = result["metadata"]["files"][REF_XLSX]
    assert file_meta["col_count"] == len(COLUMNS)
    assert file_meta["grain"] != ""
    assert file_meta["avg_confidence"] > 0.0
