"""Tests unitaires — SchemaLinkingAgent (4 filtres + score composite).

LLM et storage sont systématiquement mockés (aucun appel réseau réel).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pandas as pd
import pytest

from app.agents.schema_linking_agent import (
    SchemaLinkingAgent,
    _is_sequential,
    _name_similarity,
    _normalize_col_name,
    _passes_name_filter,
    _detect_candidates,
)
from app.pipeline.state import initial_state

# ── Constantes ────────────────────────────────────────────────────────────────

REF_VENTES = "s3://narr8-dev/uploads/ventes.csv"
REF_CLIENTS = "s3://narr8-dev/uploads/clients.csv"

# DataFrame ventes : client_id FK avec valeurs répétées (unique_ratio = 4/8 = 0.50)
DF_VENTES = pd.DataFrame(
    {
        "date": pd.to_datetime(
            ["2024-01-05", "2024-02-03", "2024-03-01", "2024-04-02",
             "2024-05-07", "2024-06-01", "2024-07-15", "2024-08-10"]
        ),
        "client_id": ["CLI001", "CLI002", "CLI001", "CLI003",
                      "CLI002", "CLI004", "CLI001", "CLI003"],
        "ca_ht": [12500.0, 8750.5, 21300.0, 5400.0, 16800.0, 9000.0, 14000.0, 7500.0],
        "region": ["Nord", "Sud", "Est", "Ouest", "Ile-de-France", "Nord", "Sud", "Est"],
    }
)

# DataFrame clients : client_id PK (nom identique → name_sim = 1.0 après normalisation)
DF_CLIENTS = pd.DataFrame(
    {
        "client_id": ["CLI001", "CLI002", "CLI003", "CLI004", "CLI005", "CLI006"],
        "nom": ["Dupont SA", "Martin & Fils", "Bernard Tech",
                "Petit Commerce", "Grand IDF", "Lopez"],
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
# Tests des fonctions utilitaires (pures, sans I/O)
# ══════════════════════════════════════════════════════════════════════════════


class TestNormalizeColName:
    def test_strips_id_suffix(self):
        assert _normalize_col_name("client_id") == "client"

    def test_strips_ref_suffix(self):
        assert _normalize_col_name("order_ref") == "order"

    def test_strips_id_prefix(self):
        assert _normalize_col_name("id_client") == "client"

    def test_strips_ref_prefix(self):
        assert _normalize_col_name("ref_commande") == "commande"

    def test_no_change_plain_name(self):
        assert _normalize_col_name("region") == "region"

    def test_no_change_standalone_id(self):
        # "id" seul ne doit pas être vidé
        assert _normalize_col_name("id") == "id"

    def test_lowercased(self):
        assert _normalize_col_name("Client_ID") == "client"


class TestNameSimilarity:
    def test_identical_names(self):
        assert _name_similarity("client_id", "client_id") == 1.0

    def test_same_after_normalization(self):
        # "client_id" → "client",  "id_client" → "client"  → 1.0
        assert _name_similarity("client_id", "id_client") == 1.0

    def test_similar_names(self):
        # "order" vs "order" → 1.0
        assert _name_similarity("order_id", "order_ref") == 1.0

    def test_false_positive_client_vente(self):
        # "client" vs "vente" partagent "ent" → sim ≈ 0.545 < 0.55
        sim = _name_similarity("client_id", "vente_id")
        assert sim < 0.55

    def test_dissimilar_names_below_threshold(self):
        assert _name_similarity("prix_vente", "latitude") < 0.55

    def test_completely_different(self):
        assert _name_similarity("xyz_abc", "pqr_def") < 0.55


class TestPassesNameFilter:
    def test_identical_columns_pass(self):
        assert _passes_name_filter("client_id", "client_id", 1.0) is True

    def test_same_entity_different_suffix_passes(self):
        # "order_id" vs "order_ref" → entity "order"/"order" sim=1.0 ≥ 0.75
        assert _passes_name_filter("order_id", "order_ref", 1.0) is True

    def test_false_positive_client_vente_blocked(self):
        # Les deux ont _id → règle même-suffixe : entity sim ≈ 0.545 < 0.75 → rejeté
        sim = _name_similarity("client_id", "vente_id")  # ≈ 0.545
        assert _passes_name_filter("client_id", "vente_id", sim) is False

    def test_general_threshold_blocks_unrelated(self):
        assert _passes_name_filter("prix_vente", "latitude", 0.30) is False

    def test_plain_columns_use_general_threshold(self):
        # Colonnes sans suffixe → seuil général (0.55) seulement
        assert _passes_name_filter("region", "region", 1.0) is True
        assert _passes_name_filter("region", "pays", 0.30) is False

    def test_same_suffix_low_entity_sim_blocked(self):
        # user_id vs commande_id → "user" vs "commande" sim très faible → bloqué
        sim = _name_similarity("user_id", "commande_id")
        assert _passes_name_filter("user_id", "commande_id", sim) is False

    def test_similar_entities_same_suffix_passes(self):
        # categorie_id vs category_id → "categorie" vs "category" ≈ 0.94 ≥ 0.75
        sim = _name_similarity("categorie_id", "category_id")
        assert _passes_name_filter("categorie_id", "category_id", sim) is True


class TestIsSequential:
    def test_sequential_ints(self):
        assert _is_sequential({"1", "2", "3", "4", "5"}) is True

    def test_sequential_not_starting_at_one(self):
        assert _is_sequential({"10", "11", "12", "13"}) is True

    def test_non_sequential_with_gap(self):
        assert _is_sequential({"1", "2", "4", "5"}) is False

    def test_non_sequential_strings(self):
        assert _is_sequential({"CLI001", "CLI002", "CLI003"}) is False

    def test_single_value_not_sequential(self):
        # Moins de 2 valeurs → False (pas assez pour confirmer une séquence)
        assert _is_sequential({"1"}) is False

    def test_mixed_type_strings(self):
        assert _is_sequential({"1", "deux", "3"}) is False


# ══════════════════════════════════════════════════════════════════════════════
# Tests de _detect_candidates (sans agent, sans I/O)
# ══════════════════════════════════════════════════════════════════════════════


class TestDetectCandidates:
    def test_filter1_name_similarity_rejects_dissimilar_columns(self):
        """Filtre 1 : colonnes à noms sans rapport → aucun candidat même si valeurs partagées."""
        df_a = pd.DataFrame({"prix_vente": [100, 100, 200, 200, 300, 300]})
        df_b = pd.DataFrame({"latitude": [100, 200, 300, 400, 500]})
        result = _detect_candidates(REF_VENTES, df_a, REF_CLIENTS, df_b)
        assert result == [], f"Des candidats ont été détectés à tort : {result}"

    def test_filter2_sequential_pk_excluded_as_target(self):
        """Filtre 2 : col_b = [1,2,3..N] → exclue comme cible même si coverage = 100%."""
        df_a = pd.DataFrame({"order_id": [1, 2, 3, 1, 2, 4]})       # FK, valeurs répétées
        df_b = pd.DataFrame({"order_id": [1, 2, 3, 4, 5, 6, 7]})    # PK séquentielle
        result = _detect_candidates(REF_VENTES, df_a, REF_CLIENTS, df_b)
        assert result == [], f"PK séquentielle non filtrée : {result}"

    def test_filter2_sequential_source_allowed(self):
        """Filtre 2 : col_a séquentielle peut rester SOURCE (seule la cible est exclue)."""
        # df_a a une col séquentielle, df_b a une col non-séquentielle avec mêmes valeurs
        df_a = pd.DataFrame({"seq_id": [1, 2, 3, 4, 5, 6]})
        df_b = pd.DataFrame({"seq_id": ["1", "2", "3", "4", "5", "6", "7", "8"]})
        # vals_b n'est pas séquentielle en entier (strings) → pas filtré par filtre 2
        # mais unique_ratio(df_a.seq_id) = 6/6 = 1.0 → filtré par filtre 3
        result = _detect_candidates(REF_VENTES, df_a, REF_CLIENTS, df_b)
        # Filtré par unique_ratio, pas par la règle source
        assert result == []

    def test_filter3_high_unique_ratio_rejected(self):
        """Filtre 3 : unique_ratio(FK) >= 0.90 → la colonne source est une PK, pas une FK."""
        # 10 valeurs uniques sur 10 lignes → unique_ratio = 1.0
        df_a = pd.DataFrame({"product_id": [f"P{i}" for i in range(10)]})
        df_b = pd.DataFrame({"product_id": [f"P{i}" for i in range(12)]})
        result = _detect_candidates(REF_VENTES, df_a, REF_CLIENTS, df_b)
        assert result == [], f"Colonne PK non filtrée : {result}"

    def test_filter4_composite_score_stored(self):
        """Filtre 4 : composite_score et name_sim présents dans chaque relation détectée."""
        df_a = pd.DataFrame({"client_id": ["C1", "C2", "C1", "C3", "C2", "C4"]})
        df_b = pd.DataFrame({"client_id": ["C1", "C2", "C3", "C4", "C5"]})
        result = _detect_candidates(REF_VENTES, df_a, REF_CLIENTS, df_b)
        assert len(result) == 1
        rel = result[0]
        assert "composite_score" in rel
        assert "name_sim" in rel
        assert "unique_ratio" in rel
        assert rel["composite_score"] >= 0.60
        assert rel["name_sim"] == 1.0   # "client_id" vs "client_id" après norm

    def test_filter4_composite_score_formula(self):
        """Filtre 4 : vérification numérique du score = cov×0.40 + sim×0.40 + card×0.20."""
        df_a = pd.DataFrame({"client_id": ["C1", "C2", "C1", "C3", "C2", "C4"]})
        df_b = pd.DataFrame({"client_id": ["C1", "C2", "C3", "C4", "C5"]})
        result = _detect_candidates(REF_VENTES, df_a, REF_CLIENTS, df_b)
        assert result
        rel = result[0]
        expected = round(rel["coverage"] * 0.40 + rel["name_sim"] * 0.40 + 1.0 * 0.20, 4)
        assert rel["composite_score"] == pytest.approx(expected, abs=1e-4)

    def test_all_filters_pass_clear_fk(self):
        """Cas nominal : tous les filtres passent sur une FK évidente."""
        result = _detect_candidates(REF_VENTES, DF_VENTES, REF_CLIENTS, DF_CLIENTS)
        cols = [(r["col_a"], r["col_b"]) for r in result]
        assert ("client_id", "client_id") in cols, f"Relation FK non détectée : {cols}"


# ══════════════════════════════════════════════════════════════════════════════
# Tests d'intégration de l'agent (avec mocks LLM + storage)
# ══════════════════════════════════════════════════════════════════════════════


@pytest.mark.asyncio
async def test_single_file_no_relations():
    """Fichier unique → pas de relations, pas de HITL."""
    state = _make_state(
        refs=[REF_VENTES],
        metadata_files={REF_VENTES: {"row_count": 8, "col_count": 4, "columns": {}, "grain": ""}},
    )
    result = await SchemaLinkingAgent()(state)

    assert result["status"] != "error", f"Erreurs: {result['errors']}"
    assert result["schema"]["relations"] == []
    assert result["schema"]["multi_table"] is False
    assert result["hitl_pending"] is False
    assert result["hitl_checkpoint"] is None


@pytest.mark.asyncio
async def test_detects_foreign_key():
    """Relation évidente client_id→client_id détectée avec score composite élevé."""
    state = _make_state(refs=[REF_VENTES, REF_CLIENTS])

    with (
        patch(
            "app.agents.schema_linking_agent.read_dataframe",
            AsyncMock(side_effect=_df_side_effect({REF_VENTES: DF_VENTES, REF_CLIENTS: DF_CLIENTS})),
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
            AsyncMock(side_effect=_df_side_effect({REF_VENTES: DF_VENTES, REF_CLIENTS: DF_CLIENTS})),
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
            "client_id": ["CLI001", "CLI002", "CLI001", "CLI003", "CLI002", "CLI999"],
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
    """Filtre 2 via agent : col_b séquentielle [1..N] → relation rejetée."""
    df_orders = pd.DataFrame(
        {
            # FK : unique_ratio = 4/8 = 0.5, valeurs répétées
            "order_id": [1, 2, 3, 1, 2, 4, 3, 4],
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
    """Régression : client_id → vente_id ne doit PAS être proposé (entités différentes)."""
    df_ventes = pd.DataFrame(
        {
            "vente_id": ["V001", "V002", "V003", "V004", "V005", "V006"],
            "client_id": ["CLI001", "CLI002", "CLI001", "CLI003", "CLI002", "CLI003"],
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
            AsyncMock(side_effect=_df_side_effect({REF_VENTES: df_ventes, REF_CLIENTS: df_clients})),
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
