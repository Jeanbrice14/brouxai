"""Tests unitaires — RAG schéma sémantique Power BI (schema_rag.py).

Stockage Redis (get_cache/set_cache) plutôt que Postgres/pgvector — voir la docstring de
index_schema() dans schema_rag.py pour le pourquoi. Aucun vrai Redis ni vrai appel
d'embedding : un store Python en mémoire simule get_cache/set_cache. Couvre : préservation
des entrées human_verified lors d'une réindexation, fallback silencieux (jamais
d'exception) sur échec d'embedding/Redis, la logique de hash structurel
(compute_model_hash), et le classement par similarité cosinus du retrieval.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from unittest.mock import AsyncMock, patch

import pytest

from app.services.schema_rag import (
    _cosine_similarity,
    _embedding_text,
    _flatten_fields,
    _question_has_temporal_marker,
    compute_model_hash,
    index_schema,
    retrieve_relevant_fields,
    update_field_from_hitl,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

_MODEL_INFO = {
    "tables": {
        "Sales": {
            "description": "Table des ventes",
            "columns": {
                "Region": {"dataType": "String", "description": "Région de vente"},
                "Amount": {"dataType": "Double", "description": ""},
            },
        },
        "Customers": {
            "description": "",
            "columns": {"Name": {"dataType": "String", "description": ""}},
        },
    },
    "measures": {
        "Total Sales": {"dataType": "Double", "description": "Somme des ventes"},
    },
    "relations": [
        {"fromTable": "Sales", "fromColumn": "CustomerId", "toTable": "Customers", "toColumn": "Id"},
    ],
}


class _FakeCache:
    """Simule get_cache/set_cache — un dict Python en mémoire, keyed par clé Redis."""

    def __init__(self, initial: dict | None = None):
        self.store: dict[str, dict] = dict(initial or {})

    async def get(self, key: str):
        return self.store.get(key)

    async def set(self, key: str, value: dict, ttl: int = 3600):
        self.store[key] = value


@contextmanager
def _patched(cache: _FakeCache, **extra):
    """Context manager patchant get_cache/set_cache (+ patches additionnels) sur schema_rag."""
    patches = [
        patch("app.services.schema_rag.get_cache", AsyncMock(side_effect=cache.get)),
        patch("app.services.schema_rag.set_cache", AsyncMock(side_effect=cache.set)),
    ]
    for target, mock in extra.items():
        patches.append(patch(f"app.services.schema_rag.{target}", mock))
    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        yield


# ── _flatten_fields / _embedding_text / compute_model_hash ─────────────────────


def test_flatten_fields_covers_tables_columns_measures_relations():
    fields = _flatten_fields(_MODEL_INFO)
    types = {f["object_type"] for f in fields}
    assert types == {"table", "column", "measure", "relationship"}

    qnames = {f["qualified_name"] for f in fields}
    assert "Sales" in qnames
    assert "Sales[Region]" in qnames
    assert "[Total Sales]" in qnames
    assert "Sales[CustomerId]->Customers[Id]" in qnames


def test_embedding_text_includes_description_and_type():
    text = _embedding_text(
        {"qualified_name": "Sales[Region]", "dtype": "String", "description": "Région de vente"}
    )
    assert "Sales[Region]" in text
    assert "String" in text
    assert "Région de vente" in text


def test_compute_model_hash_ignores_description_changes():
    """Le hash ne doit PAS changer si seule une description change (CP1 sans impact structurel)."""
    model_with_desc = {
        "tables": {"Sales": {"columns": {"Region": {"dataType": "String", "description": "X"}}}},
        "measures": {},
        "relations": [],
    }
    model_without_desc = {
        "tables": {"Sales": {"columns": {"Region": {"dataType": "String", "description": ""}}}},
        "measures": {},
        "relations": [],
    }
    assert compute_model_hash(model_with_desc) == compute_model_hash(model_without_desc)


def test_compute_model_hash_changes_on_structural_change():
    base = {
        "tables": {"Sales": {"columns": {"Region": {"dataType": "String", "description": ""}}}},
        "measures": {},
        "relations": [],
    }
    changed_type = {
        "tables": {"Sales": {"columns": {"Region": {"dataType": "Int64", "description": ""}}}},
        "measures": {},
        "relations": [],
    }
    added_column = {
        "tables": {
            "Sales": {
                "columns": {
                    "Region": {"dataType": "String", "description": ""},
                    "NewCol": {"dataType": "Int64", "description": ""},
                }
            }
        },
        "measures": {},
        "relations": [],
    }
    assert compute_model_hash(base) != compute_model_hash(changed_type)
    assert compute_model_hash(base) != compute_model_hash(added_column)


# ── _cosine_similarity ──────────────────────────────────────────────────────────


def test_cosine_similarity_identical_vectors_is_one():
    assert _cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors_is_zero():
    assert _cosine_similarity([1.0, 0.0], [0.0, 1.0]) == pytest.approx(0.0)


def test_cosine_similarity_opposite_vectors_is_minus_one():
    assert _cosine_similarity([1.0, 0.0], [-1.0, 0.0]) == pytest.approx(-1.0)


def test_cosine_similarity_zero_vector_returns_zero_not_nan():
    assert _cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0


# ── _question_has_temporal_marker ────────────────────────────────────────────


def test_question_has_temporal_marker_without_accent_on_evolution():
    """Régression réelle (cf. test_intent_agent.py) : "evolution" sans accent doit être
    reconnu au même titre que "évolution" — sinon le boost RAG temporel (et l'override
    ligne de VizAgent) ne se déclenche pas pour une question tapée sans accents."""
    assert _question_has_temporal_marker("quelle est l'evolution des commandes en 2022") is True


def test_question_has_temporal_marker_without_accent_on_annee():
    assert _question_has_temporal_marker("ventilation par annee") is True


# ── index_schema ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_index_schema_skips_when_hash_unchanged():
    expected_hash = compute_model_hash(_MODEL_INFO)
    cache = _FakeCache({"schema_rag:hash:tenant-1:AdventureWorks": {"hash": expected_hash}})
    mock_embed = AsyncMock()

    with _patched(cache, call_embedding=mock_embed):
        result = await index_schema("tenant-1", "AdventureWorks", _MODEL_INFO)

    assert result["reindexed"] is False
    mock_embed.assert_not_called()


@pytest.mark.asyncio
async def test_index_schema_preserves_human_verified_fields():
    """Un champ human_verified=true ne doit JAMAIS être ré-embeddé ni écrasé."""
    cache = _FakeCache(
        {
            "schema_rag:fields:tenant-1:AdventureWorks": {
                "fields": [
                    {
                        "qualified_name": "Sales[Region]",
                        "object_type": "column",
                        "parent_table": "Sales",
                        "description": "Région de vente (validée humainement)",
                        "embedding": [0.9, 0.9, 0.9],
                        "human_verified": True,
                        "confidence": 1.0,
                        "is_temporal": False,
                    }
                ]
            }
        }
    )
    embed_calls: list[list[str]] = []

    async def _fake_embed(texts):
        embed_calls.append(texts)
        return [[0.1, 0.2, 0.3] for _ in texts]

    with _patched(cache, call_embedding=AsyncMock(side_effect=_fake_embed)):
        result = await index_schema("tenant-1", "AdventureWorks", _MODEL_INFO)

    assert result["reindexed"] is True
    all_embedded_texts = [t for batch in embed_calls for t in batch]
    assert not any("Sales[Region]" in t for t in all_embedded_texts), (
        f"Sales[Region] (human_verified) n'aurait pas dû être ré-embeddé : {all_embedded_texts}"
    )
    assert any("Sales[Amount]" in t for t in all_embedded_texts)

    stored = cache.store["schema_rag:fields:tenant-1:AdventureWorks"]["fields"]
    preserved = next(f for f in stored if f["qualified_name"] == "Sales[Region]")
    assert preserved["description"] == "Région de vente (validée humainement)"
    assert preserved["human_verified"] is True


@pytest.mark.asyncio
async def test_index_schema_never_raises_on_embedding_failure():
    """Une panne d'embedding pendant l'indexation ne doit jamais lever d'exception."""
    cache = _FakeCache()

    with _patched(
        cache, call_embedding=AsyncMock(side_effect=RuntimeError("embedding API down"))
    ):
        result = await index_schema("tenant-1", "AdventureWorks", _MODEL_INFO)

    assert result["reindexed"] is False
    assert "error" in result


# ── retrieve_relevant_fields ────────────────────────────────────────────────────


def _stored_field(qualified_name, object_type, parent_table, description, embedding, is_temporal=False):
    return {
        "qualified_name": qualified_name,
        "object_type": object_type,
        "parent_table": parent_table,
        "description": description,
        "embedding": embedding,
        "human_verified": False,
        "confidence": 0.6,
        "is_temporal": is_temporal,
    }


@pytest.mark.asyncio
async def test_retrieve_relevant_fields_returns_formatted_rows():
    stored = [_stored_field("Sales[Region]", "column", "Sales", "Région de vente", [1.0, 0.0])]
    cache = _FakeCache({"schema_rag:fields:tenant-1:AdventureWorks": {"fields": stored}})

    with _patched(cache, call_embedding=AsyncMock(return_value=[[1.0, 0.0]])):
        fields = await retrieve_relevant_fields("tenant-1", "AdventureWorks", "ventes par région", k=5)

    assert fields == [
        {
            "qualified_name": "Sales[Region]",
            "object_type": "column",
            "parent_table": "Sales",
            "description": "Région de vente",
        }
    ]


@pytest.mark.asyncio
async def test_retrieve_relevant_fields_ranks_by_cosine_similarity():
    """Le champ le plus proche du vecteur de la question doit arriver en premier."""
    stored = [
        _stored_field("Sales[Region]", "column", "Sales", "peu pertinent", [0.0, 1.0]),
        _stored_field("Sales[Amount]", "column", "Sales", "très pertinent", [1.0, 0.0]),
    ]
    cache = _FakeCache({"schema_rag:fields:tenant-1:AdventureWorks": {"fields": stored}})

    with _patched(cache, call_embedding=AsyncMock(return_value=[[1.0, 0.0]])):
        fields = await retrieve_relevant_fields("tenant-1", "AdventureWorks", "montant des ventes", k=5)

    assert [f["qualified_name"] for f in fields] == ["Sales[Amount]", "Sales[Region]"]


@pytest.mark.asyncio
async def test_retrieve_relevant_fields_respects_k():
    stored = [
        _stored_field(f"Sales[Col{i}]", "column", "Sales", "", [1.0, float(i)]) for i in range(10)
    ]
    cache = _FakeCache({"schema_rag:fields:tenant-1:AdventureWorks": {"fields": stored}})

    with _patched(cache, call_embedding=AsyncMock(return_value=[[1.0, 0.0]])):
        fields = await retrieve_relevant_fields("tenant-1", "AdventureWorks", "question", k=3)

    assert len(fields) == 3


@pytest.mark.asyncio
async def test_retrieve_relevant_fields_temporal_question_forces_calendar_fields():
    """Question avec marqueur temporel ('par mois') → les champs is_temporal=true sont
    inclus même s'ils sont absents du top-k sémantique (régression confirmée contre
    AdventureWorks : "évolution des ventes par mois" ne remontait pas Calendar Lookup)."""
    stored = [
        _stored_field("[Total Sales]", "measure", None, "Somme des ventes", [1.0, 0.0]),
        _stored_field(
            "Calendar Lookup", "table", None, "", [0.0, 1.0], is_temporal=True
        ),
        _stored_field(
            "Calendar Lookup[Date]", "column", "Calendar Lookup", "", [0.0, 0.9], is_temporal=True
        ),
        _stored_field(
            "Calendar Lookup[Month]", "column", "Calendar Lookup", "", [0.0, 0.8], is_temporal=True
        ),
    ]
    cache = _FakeCache({"schema_rag:fields:tenant-1:AdventureWorks": {"fields": stored}})

    with _patched(cache, call_embedding=AsyncMock(return_value=[[1.0, 0.0]])):
        fields = await retrieve_relevant_fields(
            "tenant-1", "AdventureWorks", "évolution des ventes par mois", k=1
        )

    qualified_names = {f["qualified_name"] for f in fields}
    assert "[Total Sales]" in qualified_names, "Le résultat sémantique original doit être conservé"
    assert "Calendar Lookup" in qualified_names
    assert "Calendar Lookup[Date]" in qualified_names
    assert "Calendar Lookup[Month]" in qualified_names


@pytest.mark.asyncio
async def test_retrieve_relevant_fields_non_temporal_question_skips_boost():
    """Sans marqueur temporel dans la question, seul le top-k sémantique est retourné."""
    stored = [
        _stored_field("Sales[Region]", "column", "Sales", "Région de vente", [1.0, 0.0]),
        _stored_field(
            "Calendar Lookup", "table", None, "", [0.0, 1.0], is_temporal=True
        ),
    ]
    cache = _FakeCache({"schema_rag:fields:tenant-1:AdventureWorks": {"fields": stored}})

    with _patched(cache, call_embedding=AsyncMock(return_value=[[1.0, 0.0]])):
        fields = await retrieve_relevant_fields(
            "tenant-1", "AdventureWorks", "répartition par catégorie", k=1
        )

    assert [f["qualified_name"] for f in fields] == ["Sales[Region]"]


@pytest.mark.asyncio
async def test_retrieve_relevant_fields_falls_back_to_empty_on_embedding_failure():
    with patch(
        "app.services.schema_rag.call_embedding",
        AsyncMock(side_effect=RuntimeError("embedding API down")),
    ):
        fields = await retrieve_relevant_fields("tenant-1", "AdventureWorks", "ventes par région")

    assert fields == []


@pytest.mark.asyncio
async def test_retrieve_relevant_fields_falls_back_to_empty_on_redis_failure():
    with (
        patch("app.services.schema_rag.call_embedding", AsyncMock(return_value=[[1.0, 0.0]])),
        patch(
            "app.services.schema_rag.get_cache",
            AsyncMock(side_effect=RuntimeError("redis unreachable")),
        ),
    ):
        fields = await retrieve_relevant_fields("tenant-1", "AdventureWorks", "ventes par région")

    assert fields == []


@pytest.mark.asyncio
async def test_retrieve_relevant_fields_no_index_returns_empty():
    """Aucune entrée indexée pour ce modèle (jamais indexé, ou Redis vide) → liste vide,
    pas d'exception — l'appelant retombe sur le schéma complet."""
    cache = _FakeCache()

    with _patched(cache, call_embedding=AsyncMock(return_value=[[1.0, 0.0]])):
        fields = await retrieve_relevant_fields("tenant-1", "AdventureWorks", "ventes par région")

    assert fields == []


# ── update_field_from_hitl ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_update_field_from_hitl_marks_verified():
    stored = [_stored_field("Sales[Region]", "column", "Sales", "", [0.1, 0.1])]
    cache = _FakeCache({"schema_rag:fields:tenant-1:AdventureWorks": {"fields": stored}})

    with _patched(cache, call_embedding=AsyncMock(return_value=[[0.9, 0.9]])):
        updated = await update_field_from_hitl(
            "tenant-1", "AdventureWorks", "Sales[Region]", "Région de vente validée"
        )

    assert updated is True
    updated_field = cache.store["schema_rag:fields:tenant-1:AdventureWorks"]["fields"][0]
    assert updated_field["description"] == "Région de vente validée"
    assert updated_field["human_verified"] is True
    assert updated_field["confidence"] == 1.0
    assert updated_field["embedding"] == [0.9, 0.9]


@pytest.mark.asyncio
async def test_update_field_from_hitl_returns_false_when_field_not_found():
    cache = _FakeCache({"schema_rag:fields:tenant-1:AdventureWorks": {"fields": []}})

    with _patched(cache, call_embedding=AsyncMock(return_value=[[0.9, 0.9]])):
        updated = await update_field_from_hitl(
            "tenant-1", "AdventureWorks", "Sales[Inexistant]", "description"
        )

    assert updated is False


@pytest.mark.asyncio
async def test_update_field_from_hitl_never_raises_on_failure():
    with patch(
        "app.services.schema_rag.call_embedding",
        AsyncMock(side_effect=RuntimeError("embedding API down")),
    ):
        updated = await update_field_from_hitl(
            "tenant-1", "AdventureWorks", "Sales[Region]", "description"
        )

    assert updated is False
