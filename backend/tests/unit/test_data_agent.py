"""Tests unitaires Sprint 4 — DataAgent complet.

LLM, storage et cache sont systématiquement mockés (aucun appel réseau réel).
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from app.agents.data_agent import DataAgent
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

# Code simple que le mock LLM retourne (résultat 1 ligne → toujours < 500)
_SIMPLE_CODE = "result = {'summary': pd.DataFrame({'metric': ['count'], 'value': [len(dfs[list(dfs.keys())[0]])]})}"
# Code qui crée un DataFrame de 1000 lignes
_BIG_CODE = "result = {'big_table': pd.DataFrame({'val': range(1000)})}"
# Code invalide → NameError
_BROKEN_CODE = "result = undefined_variable_that_does_not_exist"


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
    llm_code: str = _SIMPLE_CODE,
    cached: dict | None = None,
):
    """Context manager qui mocke les 4 dépendances externes du DataAgent."""
    with (
        patch("app.agents.data_agent.read_dataframe", AsyncMock(return_value=df)),
        patch(
            "app.agents.data_agent.call_llm_json",
            AsyncMock(return_value={"code": llm_code}),
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

    with _patch_all(df=DF_VENTES, llm_code=_SIMPLE_CODE, cached=None):
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

    # Capturer le prompt envoyé au LLM
    mock_llm = AsyncMock(return_value={"code": "result = {}"})

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

    # Vérifier que le schéma structurel est présent (noms de colonnes)
    assert "ca_ht" in prompt_sent, "Le nom de colonne 'ca_ht' devrait être dans le prompt"
    assert "ventes" in prompt_sent, "Le nom de table 'ventes' devrait être dans le prompt"


@pytest.mark.asyncio
async def test_exec_failure_triggers_fallback():
    """Code invalide → pas de crash, fallback basique, warning dans errors."""
    state = _make_state()
    agent = DataAgent()

    with _patch_all(df=DF_VENTES, llm_code=_BROKEN_CODE, cached=None):
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
    assert "fallback" in errors_str.lower() or "exec" in errors_str.lower(), (
        f"Aucun warning fallback/exec dans errors: {result['errors']}"
    )


@pytest.mark.asyncio
async def test_cache_hit_skips_llm():
    """Cache hit → call_llm_json n'est PAS appelé, les agrégats du cache sont utilisés."""
    cached_aggregates = {"from_cache": [{"region": "Nord", "ca_ht": 12500.0}]}
    state = _make_state()
    agent = DataAgent()

    mock_llm = AsyncMock(return_value={"code": _SIMPLE_CODE})

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
    """Un DataFrame de 1000 lignes est limité à 500 lignes dans les agrégats."""
    state = _make_state()
    agent = DataAgent()

    with _patch_all(df=DF_VENTES, llm_code=_BIG_CODE, cached=None):
        result = await agent(state)

    aggregates = result["aggregates"]
    assert "big_table" in aggregates, (
        f"'big_table' manquant dans aggregates. Clés: {list(aggregates.keys())}"
    )
    assert len(aggregates["big_table"]) == 500, (
        f"Attendu 500 lignes, obtenu {len(aggregates['big_table'])}"
    )
