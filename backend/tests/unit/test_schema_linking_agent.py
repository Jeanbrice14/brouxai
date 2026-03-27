"""Tests unitaires — SchemaLinkingAgent (star schema detection).

LLM et storage sont systématiquement mockés (aucun appel réseau réel).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from app.agents.schema_linking_agent import (
    SchemaLinkingAgent,
    _auto_generate_date_table,
    _detect_cycles,
    _detect_star_relations,
    classify_table,
    is_sequential_pk,
    is_valid_fk_cardinality,
    name_similarity,
)
from app.pipeline.state import initial_state

# ── Constantes ────────────────────────────────────────────────────────────────

REF_VENTES = "s3://narr8-dev/uploads/ventes.csv"
REF_CLIENTS = "s3://narr8-dev/uploads/clients.csv"
REF_PRODUITS = "s3://narr8-dev/uploads/produits.csv"

# Fixture ventes : 198 rows shape — fact table with metrics + FK cols
# Using a smaller fixture for unit tests but with the right shape
DF_VENTES_FIXTURE = pd.DataFrame(
    {
        "vente_id": list(range(1, 199)),
        "date": pd.to_datetime(["2024-01-05"] * 99 + ["2024-02-03"] * 99),
        "client_id": [i % 30 + 1 for i in range(198)],
        "produit_ref": [f"PRD-{(i % 5) + 1:03d}" for i in range(198)],
        "ca_ht": [float(100 + i * 10) for i in range(198)],
        "statut": ["facturé"] * 198,
    }
)

# Fixture clients : 30 rows — dimension table (1 PK, no metrics)
# nom has repeated values so only client_id is a key_candidate (unique_ratio > 0.85)
DF_CLIENTS_FIXTURE = pd.DataFrame(
    {
        "client_id": [f"C{i:03d}" for i in range(1, 31)],
        "nom": ["Dupont", "Martin", "Bernard", "Petit", "Lopez"] * 6,
        "region": ["Nord", "Sud", "Est", "Ouest", "Ile-de-France"] * 6,
        "segment": ["PME", "ETI", "TPE"] * 10,
    }
)

# DataFrame ventes : client_id FK avec valeurs répétées (unique_ratio = 4/8 = 0.50)
DF_VENTES = pd.DataFrame(
    {
        "date": pd.to_datetime(
            [
                "2024-01-05",
                "2024-02-03",
                "2024-03-01",
                "2024-04-02",
                "2024-05-07",
                "2024-06-01",
                "2024-07-15",
                "2024-08-10",
            ]
        ),
        "client_id": [
            "CLI001",
            "CLI002",
            "CLI001",
            "CLI003",
            "CLI002",
            "CLI004",
            "CLI001",
            "CLI003",
        ],
        "ca_ht": [
            12500.0,
            8750.5,
            21300.0,
            5400.0,
            16800.0,
            9000.0,
            14000.0,
            7500.0,
        ],
        "region": [
            "Nord",
            "Sud",
            "Est",
            "Ouest",
            "Ile-de-France",
            "Nord",
            "Sud",
            "Est",
        ],
    }
)

# DataFrame clients : client_id PK (nom identique → name_sim = 1.0 après normalisation)
DF_CLIENTS = pd.DataFrame(
    {
        "client_id": ["CLI001", "CLI002", "CLI003", "CLI004", "CLI005", "CLI006"],
        "nom": [
            "Dupont SA",
            "Martin & Fils",
            "Bernard Tech",
            "Petit Commerce",
            "Grand IDF",
            "Lopez",
        ],
        "region": ["Nord", "Ouest", "Est", "Ouest", "Ile-de-France", "Sud"],
        "segment": ["PME", "ETI", "PME", "TPE", "ETI", "PME"],
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
    def _fn(ref: str) -> pd.DataFrame:
        return df_map.get(ref, pd.DataFrame({"col": ["a", "b"]}))

    return _fn


# ══════════════════════════════════════════════════════════════════════════════
# Tests classify_table
# ══════════════════════════════════════════════════════════════════════════════


class TestClassifyTable:
    def test_classifies_ventes_as_fact(self):
        """ventes fixture (198 rows, ca_ht metric, multiple FK cols) → 'fact'"""
        result = classify_table(DF_VENTES_FIXTURE, {})
        assert result == "fact"

    def test_classifies_clients_as_dimension(self):
        """clients fixture (30 rows, 1 PK, no numeric metrics) → 'dimension'"""
        result = classify_table(DF_CLIENTS_FIXTURE, {})
        assert result == "dimension"

    def test_classifies_calendar_table(self):
        """Table avec date + ≤4 cols → 'calendar'"""
        df_cal = pd.DataFrame(
            {
                "date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]),
                "mois": [1, 1, 1],
                "annee": [2024, 2024, 2024],
            }
        )
        result = classify_table(df_cal, {})
        assert result == "calendar"

    def test_small_table_unknown(self):
        """Petite table sans critères clairs → 'unknown' ou 'dimension'"""
        df_small = pd.DataFrame({"col_a": ["x", "y"], "col_b": [1, 2]})
        result = classify_table(df_small, {})
        # Small table: len <= 100 → unknown or dimension depending on structure
        assert result in ("unknown", "dimension")

    def test_large_table_with_metrics_is_fact(self):
        """Table > 100 rows avec métriques numériques → 'fact'"""
        df = pd.DataFrame(
            {
                "id": list(range(1, 201)),
                "client_id": [i % 20 for i in range(200)],
                "montant": [float(i) for i in range(200)],
            }
        )
        result = classify_table(df, {})
        assert result == "fact"


# ══════════════════════════════════════════════════════════════════════════════
# Tests name_similarity
# ══════════════════════════════════════════════════════════════════════════════


class TestNameSimilarity:
    def test_identical_names(self):
        assert name_similarity("client_id", "client_id") >= 0.95

    def test_false_positive_client_vente_not_high(self):
        # "client" vs "vente" after normalization → SequenceMatcher ~0.545
        # This is above the threshold (0.50) but below 0.60 — important for regression tests
        sim = name_similarity("client_id", "vente_id")
        assert sim < 0.60, f"Similarity {sim} too high — may cause false positives"

    def test_produit_ref_vs_ref_low(self):
        # "produit_ref" normalizes to "produit", "ref" stays "ref"
        # SequenceMatcher("produit", "ref") ≈ 0.20 — correctly low
        assert name_similarity("produit_ref", "ref") < 0.50

    def test_id_prefix_vs_suffix(self):
        assert name_similarity("id_client", "client_id") >= 0.70

    def test_sap_code_vs_client_id_rejected(self):
        assert name_similarity("KUNNR", "client_id") < 0.50

    def test_same_after_normalization(self):
        # "client_id" → "client",  "id_client" → "client"  → 1.0
        assert name_similarity("client_id", "id_client") >= 0.95

    def test_completely_different(self):
        assert name_similarity("xyz_abc", "pqr_def") < 0.50

    def test_empty_after_normalization(self):
        # Edge case: very short names that normalize to empty
        result = name_similarity("_id", "_id")
        # Both normalize to "" → returns 0.0
        assert result == 0.0


# ══════════════════════════════════════════════════════════════════════════════
# Tests is_sequential_pk
# ══════════════════════════════════════════════════════════════════════════════


class TestIsSequentialPk:
    def test_sequential_starting_at_one(self):
        s = pd.Series([1, 2, 3, 4, 5])
        assert is_sequential_pk(s) is True

    def test_not_starting_at_one(self):
        s = pd.Series([5, 6, 7, 8, 9])
        assert is_sequential_pk(s) is False

    def test_with_gap(self):
        s = pd.Series([1, 2, 4, 5])
        assert is_sequential_pk(s) is False

    def test_string_ids(self):
        s = pd.Series(["CLI001", "CLI002", "CLI003"])
        assert is_sequential_pk(s) is False

    def test_single_value(self):
        s = pd.Series([1])
        assert is_sequential_pk(s) is False

    def test_low_unique_ratio(self):
        # Repeated values → unique_ratio ≤ 0.95 → False
        s = pd.Series([1, 2, 3, 1, 2, 3, 1, 2, 3, 1])
        assert is_sequential_pk(s) is False

    def test_empty_series(self):
        s = pd.Series([], dtype=int)
        assert is_sequential_pk(s) is False


# ══════════════════════════════════════════════════════════════════════════════
# Tests is_valid_fk_cardinality
# ══════════════════════════════════════════════════════════════════════════════


class TestIsValidFkCardinality:
    def test_valid_fk(self):
        """FK has low unique_ratio, PK has high unique_ratio."""
        fact_col = pd.Series([1, 2, 1, 3, 2, 4, 1, 3])  # unique_ratio = 4/8 = 0.50
        dim_col = pd.Series([1, 2, 3, 4, 5, 6])  # unique_ratio = 6/6 = 1.0
        assert is_valid_fk_cardinality(fact_col, dim_col) is True

    def test_invalid_fact_too_unique(self):
        """fact_col quasi-unique → not a valid FK."""
        fact_col = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])  # unique_ratio = 1.0
        dim_col = pd.Series([1, 2, 3, 4, 5, 6])
        assert is_valid_fk_cardinality(fact_col, dim_col) is False

    def test_invalid_dim_not_unique(self):
        """dim_col not unique → not a valid PK."""
        fact_col = pd.Series([1, 2, 1, 3, 2, 4])
        dim_col = pd.Series([1, 1, 2, 2, 3, 3])  # unique_ratio = 3/6 = 0.50
        assert is_valid_fk_cardinality(fact_col, dim_col) is False


# ══════════════════════════════════════════════════════════════════════════════
# Tests _detect_star_relations
# ══════════════════════════════════════════════════════════════════════════════


class TestDetectStarRelations:
    def test_rejects_pk_to_pk_relation(self):
        """clients.client_id [1..30] → ventes.vente_id [1..198] must be REJECTED."""
        df_clients = pd.DataFrame({"client_id": list(range(1, 31)), "nom": ["A"] * 30})
        df_ventes = pd.DataFrame(
            {
                "vente_id": list(range(1, 199)),
                "client_id": [i % 30 + 1 for i in range(198)],
                "ca_ht": [100.0] * 198,
            }
        )
        # From spec: clients is "dim" ref, ventes is "fact" ref
        # But testing _detect_star_relations directly (fact→dim direction)
        # Here we pass dim as fact_ref and fact as dim_ref to verify rejection
        result = _detect_star_relations(REF_CLIENTS, df_clients, REF_VENTES, df_ventes)
        bad = [r for r in result if r["col_a"] == "client_id" and r["col_b"] == "vente_id"]
        assert bad == [], f"Faux positif non rejeté: {bad}"

    def test_detects_correct_fk_relation(self):
        """ventes.client_id → clients.client_id MUST be detected."""
        df_ventes = pd.DataFrame(
            {
                "vente_id": list(range(1, 199)),
                "client_id": [i % 30 + 1 for i in range(198)],
                "ca_ht": [float(100 + i) for i in range(198)],
            }
        )
        df_clients = pd.DataFrame(
            {
                "client_id": list(range(1, 31)),
                "nom": [f"Client {i}" for i in range(1, 31)],
                "region": ["Nord", "Sud"] * 15,
            }
        )
        result = _detect_star_relations(REF_VENTES, df_ventes, REF_CLIENTS, df_clients)
        fk_rels = [r for r in result if r["col_a"] == "client_id" and r["col_b"] == "client_id"]
        assert len(fk_rels) > 0, "Relation ventes.client_id → clients.client_id non détectée"
        rel = fk_rels[0]
        assert rel["coverage"] >= 0.70
        assert rel["composite_score"] >= 0.60

    def test_filter_name_similarity_rejects_dissimilar(self):
        """Colonnes sans rapport de noms → aucun candidat même si valeurs partagées."""
        df_a = pd.DataFrame({"prix_vente": [100, 100, 200, 200, 300, 300]})
        df_b = pd.DataFrame({"latitude": [100, 200, 300, 400, 500]})
        result = _detect_star_relations(REF_VENTES, df_a, REF_CLIENTS, df_b)
        assert result == []

    def test_filter_coverage_rejects_low_overlap(self):
        """Coverage < 0.70 → relation rejetée."""
        df_a = pd.DataFrame(
            {
                "client_id": [
                    "CLI001",
                    "CLI002",
                    "CLI001",
                    "ZZZZZ",
                    "YYYYY",
                    "XXXXX",
                ]
            }
        )
        df_b = pd.DataFrame({"client_id": ["CLI001", "CLI002", "CLI003", "CLI004"]})
        result = _detect_star_relations(REF_VENTES, df_a, REF_CLIENTS, df_b)
        # coverage = 2/4 unique in fact covered = 50% < 70% → rejected
        assert result == []

    def test_composite_score_stored(self):
        """composite_score et name_sim présents dans chaque relation détectée."""
        df_a = pd.DataFrame({"client_id": ["C1", "C2", "C1", "C3", "C2", "C4"]})
        df_b = pd.DataFrame({"client_id": ["C1", "C2", "C3", "C4", "C5"]})
        result = _detect_star_relations(REF_VENTES, df_a, REF_CLIENTS, df_b)
        assert len(result) == 1
        rel = result[0]
        assert "composite_score" in rel
        assert "name_sim" in rel
        assert "unique_ratio" in rel
        assert rel["composite_score"] >= 0.60
        assert rel["name_sim"] == pytest.approx(1.0)

    def test_top_k_candidates_limit(self):
        """Au maximum _TOP_K_CANDIDATES (5) relations retournées par paire."""
        # Create many matching columns
        df_a = pd.DataFrame({f"col_{i}": ["C1", "C2", "C1", "C3"] for i in range(10)})
        df_b = pd.DataFrame({f"col_{i}": ["C1", "C2", "C3", "C4", "C5"] for i in range(10)})
        result = _detect_star_relations(REF_VENTES, df_a, REF_CLIENTS, df_b)
        assert len(result) <= 5


# ══════════════════════════════════════════════════════════════════════════════
# Tests _detect_cycles
# ══════════════════════════════════════════════════════════════════════════════


class TestDetectCycles:
    def test_no_cycles_star_schema(self):
        """Star schema (fact → dim only) → no cycles."""
        relations = [
            {"table_a": "ventes", "table_b": "clients"},
            {"table_a": "ventes", "table_b": "produits"},
        ]
        warnings = _detect_cycles(relations)
        assert warnings == []

    def test_detects_simple_cycle(self):
        """A→B→A cycle detected."""
        relations = [
            {"table_a": "table_a", "table_b": "table_b"},
            {"table_a": "table_b", "table_b": "table_a"},
        ]
        warnings = _detect_cycles(relations)
        assert len(warnings) > 0
        assert "circulaire" in warnings[0]

    def test_no_cycles_empty(self):
        """Empty relations → no warnings."""
        assert _detect_cycles([]) == []


# ══════════════════════════════════════════════════════════════════════════════
# Tests _auto_generate_date_table
# ══════════════════════════════════════════════════════════════════════════════


class TestAutoGenerateDateTable:
    def test_generates_expected_columns(self):
        df = pd.DataFrame({"date": pd.to_datetime(["2024-01-01", "2024-01-15", "2024-02-01"])})
        cal = _auto_generate_date_table(df, "date")
        assert "date" in cal.columns
        assert "annee" in cal.columns
        assert "mois" in cal.columns
        assert "trimestre" in cal.columns
        assert "nom_mois" in cal.columns
        assert "est_weekend" in cal.columns

    def test_correct_date_range(self):
        df = pd.DataFrame({"date": pd.to_datetime(["2024-01-01", "2024-01-10"])})
        cal = _auto_generate_date_table(df, "date")
        assert len(cal) == 10  # 10 days inclusive


# ══════════════════════════════════════════════════════════════════════════════
# Tests d'intégration de l'agent (avec mocks LLM + storage)
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_single_file_no_relations():
    """Fichier unique → schema vide avec nouveaux champs, pas de HITL."""
    state = _make_state(
        refs=[REF_VENTES],
        metadata_files={
            REF_VENTES: {
                "row_count": 8,
                "col_count": 4,
                "columns": {},
                "grain": "",
            }
        },
    )
    result = await SchemaLinkingAgent()(state)

    assert result["status"] != "error", f"Erreurs: {result['errors']}"
    assert result["schema"]["relations"] == []
    assert result["schema"]["multi_table"] is False
    assert result["schema"]["table_types"] == {}
    assert result["schema"]["auto_generated_tables"] == []
    assert result["schema"]["warnings"] == []
    assert result["schema"]["schema_type"] == "unknown"
    assert result["hitl_pending"] is False
    assert result["hitl_checkpoint"] is None


@pytest.mark.asyncio
async def test_detects_foreign_key():
    """Relation évidente client_id→client_id détectée avec score composite élevé."""
    state = _make_state(refs=[REF_VENTES, REF_CLIENTS])

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
        result = await SchemaLinkingAgent()(state)

    assert result["status"] != "error", f"Erreurs: {result['errors']}"
    relations = result["schema"]["relations"]
    assert len(relations) > 0, "Aucune relation détectée"

    fk_rel = next(
        (r for r in relations if r["col_a"] == "client_id" and r["col_b"] == "client_id"),
        None,
    )
    assert fk_rel is not None, f"Relation client_id→client_id non trouvée. Relations: {relations}"
    assert fk_rel["coverage"] >= 0.70
    assert fk_rel["cardinality"] == "N:1"
    assert fk_rel["composite_score"] >= 0.60
    assert fk_rel["name_sim"] == pytest.approx(1.0)
    assert fk_rel["unique_ratio"] < 0.90
    assert fk_rel["description"] != ""


@pytest.mark.asyncio
async def test_triggers_hitl_on_multi_files():
    """Multi-fichiers → HITL CP2 toujours déclenché."""
    state = _make_state(refs=[REF_VENTES, REF_CLIENTS])

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
        result = await SchemaLinkingAgent()(state)

    assert result["hitl_pending"] is True
    assert result["hitl_checkpoint"] == "cp2_schema"


@pytest.mark.asyncio
async def test_high_orphan_rate_flagged():
    """Relation avec orphelins > 5% → high_orphan_rate == True."""
    # CLI999 n'existe pas dans clients → orphan_rate = 1/4 = 0.25
    df_ventes_orphan = pd.DataFrame(
        {
            "client_id": [
                "CLI001",
                "CLI002",
                "CLI001",
                "CLI003",
                "CLI002",
                "CLI999",
            ],
            "ca_ht": [1000.0, 2000.0, 3000.0, 4000.0, 5000.0, 6000.0],
        }
    )
    df_clients_small = pd.DataFrame(
        {
            "client_id": ["CLI001", "CLI002", "CLI003", "CLI004", "CLI005"],
            "nom": ["A", "B", "C", "D", "E"],
        }
    )

    state = _make_state(refs=[REF_VENTES, REF_CLIENTS])

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
        result = await SchemaLinkingAgent()(state)

    relations = result["schema"]["relations"]
    fk_rel = next(
        (r for r in relations if r["col_a"] == "client_id" and r["col_b"] == "client_id"),
        None,
    )
    assert fk_rel is not None, "Relation client_id→client_id non détectée"
    assert fk_rel["high_orphan_rate"] is True
    assert fk_rel["orphan_rate"] > 0.05


@pytest.mark.asyncio
async def test_no_relation_detected_different_types():
    """DataFrames sans valeurs compatibles → aucune relation, HITL quand même."""
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
        result = await SchemaLinkingAgent()(state)

    assert result["status"] != "error", f"Erreurs: {result['errors']}"
    assert result["schema"]["relations"] == []
    assert result["hitl_pending"] is True
    assert result["hitl_checkpoint"] == "cp2_schema"


@pytest.mark.asyncio
async def test_sequential_pk_not_detected_as_fk():
    """Filtre 2 via agent : fact_col séquentielle [1..N] → relation rejetée."""
    df_orders = pd.DataFrame(
        {
            # Sequential PK (1-8): is_sequential_pk → True
            "order_id": list(range(1, 9)),
            "montant": [100, 200, 150, 120, 210, 300, 160, 280],
        }
    )
    df_order_ref = pd.DataFrame(
        {
            # PK séquentielle : [1,2,3,4,5,6,7,8] → exclue comme cible
            "order_id": list(range(1, 9)),
            "statut": ["livré"] * 8,
        }
    )

    state = _make_state(refs=[REF_VENTES, REF_CLIENTS])

    with (
        patch(
            "app.agents.schema_linking_agent.read_dataframe",
            AsyncMock(
                side_effect=_df_side_effect({REF_VENTES: df_orders, REF_CLIENTS: df_order_ref})
            ),
        ),
        patch(
            "app.agents.schema_linking_agent.call_llm_json",
            AsyncMock(return_value=LLM_DESCRIPTION),
        ),
    ):
        result = await SchemaLinkingAgent()(state)

    assert result["schema"]["relations"] == [], (
        f"PK séquentielle acceptée à tort comme cible : {result['schema']['relations']}"
    )


@pytest.mark.asyncio
async def test_false_positive_client_id_vente_id_rejected():
    """Régression : client_id → vente_id ne doit PAS être proposé."""
    df_ventes = pd.DataFrame(
        {
            "vente_id": ["V001", "V002", "V003", "V004", "V005", "V006"],
            "client_id": [
                "CLI001",
                "CLI002",
                "CLI001",
                "CLI003",
                "CLI002",
                "CLI003",
            ],
            "montant": [100.0, 200.0, 150.0, 300.0, 250.0, 180.0],
        }
    )
    df_clients = pd.DataFrame(
        {
            "client_id": ["CLI001", "CLI002", "CLI003", "CLI004"],
            "nom": ["Dupont", "Martin", "Bernard", "Petit"],
        }
    )

    state = _make_state(refs=[REF_VENTES, REF_CLIENTS])

    with (
        patch(
            "app.agents.schema_linking_agent.read_dataframe",
            AsyncMock(
                side_effect=_df_side_effect({REF_VENTES: df_ventes, REF_CLIENTS: df_clients})
            ),
        ),
        patch(
            "app.agents.schema_linking_agent.call_llm_json",
            AsyncMock(return_value=LLM_DESCRIPTION),
        ),
    ):
        result = await SchemaLinkingAgent()(state)

    relations = result["schema"]["relations"]
    false_positive = next(
        (r for r in relations if r["col_a"] == "client_id" and r["col_b"] == "vente_id"),
        None,
    )
    assert false_positive is None, (
        f"Faux positif client_id→vente_id détecté à tort : {false_positive}"
    )
    # La vraie relation client_id→client_id doit toujours être détectée
    true_rel = next(
        (r for r in relations if r["col_a"] == "client_id" and r["col_b"] == "client_id"),
        None,
    )
    assert true_rel is not None, "Vraie relation client_id→client_id non détectée"


@pytest.mark.asyncio
async def test_pk_source_unique_ratio_rejected():
    """Filtre 3 via agent : unique_ratio(col_a) >= 0.90 → col est une PK, pas une FK."""
    # 10 valeurs uniques / 10 lignes = 1.0 ≥ 0.90
    df_pk = pd.DataFrame({"product_id": [f"P{i}" for i in range(10)]})
    df_ref = pd.DataFrame({"product_id": [f"P{i}" for i in range(15)]})

    state = _make_state(refs=[REF_VENTES, REF_CLIENTS])

    with (
        patch(
            "app.agents.schema_linking_agent.read_dataframe",
            AsyncMock(side_effect=_df_side_effect({REF_VENTES: df_pk, REF_CLIENTS: df_ref})),
        ),
        patch(
            "app.agents.schema_linking_agent.call_llm_json",
            AsyncMock(return_value=LLM_DESCRIPTION),
        ),
    ):
        result = await SchemaLinkingAgent()(state)

    assert result["schema"]["relations"] == [], (
        f"Colonne PK (unique_ratio=1.0) acceptée à tort : {result['schema']['relations']}"
    )


@pytest.mark.asyncio
async def test_auto_generates_date_table():
    """Fact table with datetime column in metadata → __date_table__ auto-generated.

    The fact table must have > 100 rows + metrics so classify_table returns 'fact'.
    The metadata must declare the column as datetime type.
    The DataFrame column dtype must be datetime64 so 'datetime' appears in str(dtype).
    """
    n = 150
    df_fact = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-05", "2024-02-03", "2024-03-01"] * 50),
            "client_id": [i % 20 + 1 for i in range(n)],
            "ca_ht": [float(100 + i) for i in range(n)],
        }
    )
    # Ensure dtype is datetime64
    df_fact["date"] = pd.to_datetime(df_fact["date"])

    df_dim = pd.DataFrame(
        {
            "client_id": list(range(1, 21)),
            "nom": ["Dupont", "Martin"] * 10,
            "region": ["Nord", "Sud"] * 10,
        }
    )

    state = _make_state(
        refs=[REF_VENTES, REF_CLIENTS],
        metadata_files={
            REF_VENTES: {
                "columns": {
                    "date": {"type": "datetime"},
                    "client_id": {"type": "integer"},
                    "ca_ht": {"type": "numeric"},
                }
            }
        },
    )

    with (
        patch(
            "app.agents.schema_linking_agent.read_dataframe",
            AsyncMock(side_effect=_df_side_effect({REF_VENTES: df_fact, REF_CLIENTS: df_dim})),
        ),
        patch(
            "app.agents.schema_linking_agent.call_llm_json",
            AsyncMock(return_value=LLM_DESCRIPTION),
        ),
    ):
        result = await SchemaLinkingAgent()(state)

    assert result["status"] != "error", f"Erreurs: {result['errors']}"
    schema = result["schema"]
    assert "__date_table__" in schema["auto_generated_tables"], (
        f"__date_table__ non généré. auto_generated_tables={schema['auto_generated_tables']}"
        f"\ntable_types={schema['table_types']}"
    )
    assert schema["table_types"].get("__date_table__") == "calendar"


@pytest.mark.asyncio
async def test_detects_star_schema():
    """3 files: ventes (fact) + clients (dim) + produits (dim) → schema_type == 'star'."""
    # Build a proper fact table (198+ rows, metrics, FK cols)
    df_ventes = pd.DataFrame(
        {
            "vente_id": list(range(1, 199)),
            "client_id": [i % 30 + 1 for i in range(198)],
            "produit_id": [i % 10 + 1 for i in range(198)],
            "ca_ht": [float(100 + i) for i in range(198)],
        }
    )
    # Dimension tables: nom must be repeated (not all-unique) so only PK is key_candidate
    df_clients = pd.DataFrame(
        {
            "client_id": list(range(1, 31)),
            "nom": ["Dupont", "Martin", "Bernard", "Petit", "Lopez"] * 6,
            "region": ["Nord", "Sud"] * 15,
        }
    )
    df_produits = pd.DataFrame(
        {
            "produit_id": list(range(1, 11)),
            "nom": ["Standard", "Premium"] * 5,
            "categorie": ["Licence", "Service"] * 5,
        }
    )

    state = _make_state(refs=[REF_VENTES, REF_CLIENTS, REF_PRODUITS])

    with (
        patch(
            "app.agents.schema_linking_agent.read_dataframe",
            AsyncMock(
                side_effect=_df_side_effect(
                    {
                        REF_VENTES: df_ventes,
                        REF_CLIENTS: df_clients,
                        REF_PRODUITS: df_produits,
                    }
                )
            ),
        ),
        patch(
            "app.agents.schema_linking_agent.call_llm_json",
            AsyncMock(return_value=LLM_DESCRIPTION),
        ),
    ):
        result = await SchemaLinkingAgent()(state)

    assert result["status"] != "error", f"Erreurs: {result['errors']}"
    schema = result["schema"]
    assert schema["schema_type"] == "star"
    assert len(schema["relations"]) >= 2, (
        f"Attendu ≥ 2 relations (client_id + produit_id), obtenu : {schema['relations']}"
    )


@pytest.mark.asyncio
async def test_schema_has_new_fields_on_multi_file():
    """Multi-fichiers → state['schema'] contient tous les nouveaux champs."""
    state = _make_state(refs=[REF_VENTES, REF_CLIENTS])

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
        result = await SchemaLinkingAgent()(state)

    schema = result["schema"]
    assert "table_types" in schema
    assert "auto_generated_tables" in schema
    assert "warnings" in schema
    assert "schema_type" in schema
    assert isinstance(schema["table_types"], dict)
    assert isinstance(schema["auto_generated_tables"], list)
    assert isinstance(schema["warnings"], list)
    assert schema["schema_type"] in ("star", "unknown")
