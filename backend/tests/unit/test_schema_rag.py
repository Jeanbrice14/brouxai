"""Tests unitaires — RAG schéma sémantique Power BI (schema_rag.py).

Aucune vraie connexion Postgres ni vrai appel d'embedding : la session SQLAlchemy et
call_embedding sont mockés. Couvre : préservation des entrées human_verified lors d'une
réindexation, fallback silencieux (jamais d'exception) sur échec d'embedding/DB, et la
logique de hash structurel (compute_model_hash).
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from app.services.schema_rag import (
    _embedding_text,
    _flatten_fields,
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


class _FakeResult:
    """Simule un sqlalchemy.Result — supporte .all()/.scalars().all()/.scalar_one_or_none()."""

    def __init__(self, rows=None, scalar=None):
        self._rows = rows if rows is not None else []
        self._scalar = scalar

    def all(self):
        return self._rows

    def scalars(self):
        return self

    def scalar_one_or_none(self):
        return self._scalar


class _FakeSession:
    """Simule une AsyncSession — enregistre les statements exécutés pour assertions."""

    def __init__(self, execute_results=None):
        self._execute_results = list(execute_results or [])
        self.executed = []
        self.committed = False

    async def execute(self, stmt):
        self.executed.append(stmt)
        if self._execute_results:
            return self._execute_results.pop(0)
        return _FakeResult()

    async def commit(self):
        self.committed = True

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def _session_factory(session: _FakeSession):
    """AsyncSessionLocal() doit être un callable retournant un context manager async."""
    return lambda: session


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


# ── index_schema ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_index_schema_skips_when_hash_unchanged():
    expected_hash = compute_model_hash(_MODEL_INFO)

    with (
        patch("app.services.schema_rag.get_cache", AsyncMock(return_value={"hash": expected_hash})),
        patch("app.services.schema_rag.call_embedding", AsyncMock()) as mock_embed,
        patch("app.services.schema_rag.AsyncSessionLocal") as mock_session_local,
    ):
        result = await index_schema("tenant-1", "AdventureWorks", _MODEL_INFO)

    assert result["reindexed"] is False
    mock_embed.assert_not_called()
    mock_session_local.assert_not_called()


@pytest.mark.asyncio
async def test_index_schema_preserves_human_verified_fields():
    """Un champ human_verified=true ne doit JAMAIS être ré-embeddé ni écrasé."""
    session = _FakeSession(
        execute_results=[
            # SELECT qualified_name WHERE human_verified=true
            _FakeResult(rows=[("Sales[Region]",)]),
            # DELETE stale (pas de résultat utile)
            _FakeResult(),
        ]
    )
    embed_calls: list[list[str]] = []

    async def _fake_embed(texts):
        embed_calls.append(texts)
        return [[0.1, 0.2, 0.3] for _ in texts]

    with (
        patch("app.services.schema_rag.get_cache", AsyncMock(return_value=None)),
        patch("app.services.schema_rag.set_cache", AsyncMock()),
        patch("app.services.schema_rag.call_embedding", side_effect=_fake_embed),
        patch("app.services.schema_rag.AsyncSessionLocal", _session_factory(session)),
    ):
        result = await index_schema("tenant-1", "AdventureWorks", _MODEL_INFO)

    assert result["reindexed"] is True
    # "Sales[Region]" est human_verified → ne doit apparaître dans AUCUN texte embeddé
    all_embedded_texts = [t for batch in embed_calls for t in batch]
    assert not any("Sales[Region]" in t for t in all_embedded_texts), (
        f"Sales[Region] (human_verified) n'aurait pas dû être ré-embeddé : {all_embedded_texts}"
    )
    # Les autres champs (non vérifiés) doivent bien être embeddés
    assert any("Sales[Amount]" in t for t in all_embedded_texts)
    assert session.committed is True


@pytest.mark.asyncio
async def test_index_schema_never_raises_on_embedding_failure():
    """Une panne d'embedding pendant l'indexation ne doit jamais lever d'exception."""
    session = _FakeSession(execute_results=[_FakeResult(rows=[])])

    with (
        patch("app.services.schema_rag.get_cache", AsyncMock(return_value=None)),
        patch("app.services.schema_rag.set_cache", AsyncMock()),
        patch(
            "app.services.schema_rag.call_embedding",
            AsyncMock(side_effect=RuntimeError("embedding API down")),
        ),
        patch("app.services.schema_rag.AsyncSessionLocal", _session_factory(session)),
    ):
        result = await index_schema("tenant-1", "AdventureWorks", _MODEL_INFO)

    assert result["reindexed"] is False
    assert "error" in result


# ── retrieve_relevant_fields ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_retrieve_relevant_fields_returns_formatted_rows():
    class _Row:
        def __init__(self, qualified_name, object_type, parent_table, description):
            self.qualified_name = qualified_name
            self.object_type = object_type
            self.parent_table = parent_table
            self.description = description

    row = _Row("Sales[Region]", "column", "Sales", "Région de vente")
    session = _FakeSession(execute_results=[_FakeResult(rows=[row])])

    with (
        patch("app.services.schema_rag.call_embedding", AsyncMock(return_value=[[0.1, 0.2]])),
        patch("app.services.schema_rag.AsyncSessionLocal", _session_factory(session)),
    ):
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
async def test_retrieve_relevant_fields_temporal_question_forces_calendar_fields():
    """Question avec marqueur temporel ('par mois') → les champs is_temporal=true sont
    inclus même s'ils sont absents du top-k sémantique (régression confirmée contre
    AdventureWorks : "évolution des ventes par mois" ne remontait pas Calendar Lookup)."""

    class _Row:
        def __init__(self, qualified_name, object_type, parent_table, description):
            self.qualified_name = qualified_name
            self.object_type = object_type
            self.parent_table = parent_table
            self.description = description

    # Résultat sémantique "normal" (top-k) : ne contient PAS la table calendrier.
    semantic_rows = [_Row("[Total Sales]", "measure", None, "Somme des ventes")]
    # Résultat du boost temporel (is_temporal=true) : la table + colonnes calendrier.
    temporal_rows = [
        _Row("Calendar Lookup", "table", None, ""),
        _Row("Calendar Lookup[Date]", "column", "Calendar Lookup", ""),
        _Row("Calendar Lookup[Month]", "column", "Calendar Lookup", ""),
    ]
    session = _FakeSession(
        execute_results=[_FakeResult(rows=semantic_rows), _FakeResult(rows=temporal_rows)]
    )

    with (
        patch("app.services.schema_rag.call_embedding", AsyncMock(return_value=[[0.1, 0.2]])),
        patch("app.services.schema_rag.AsyncSessionLocal", _session_factory(session)),
    ):
        fields = await retrieve_relevant_fields(
            "tenant-1", "AdventureWorks", "évolution des ventes par mois", k=5
        )

    qualified_names = {f["qualified_name"] for f in fields}
    assert "[Total Sales]" in qualified_names, "Le résultat sémantique original doit être conservé"
    assert "Calendar Lookup" in qualified_names
    assert "Calendar Lookup[Date]" in qualified_names
    assert "Calendar Lookup[Month]" in qualified_names
    # 2 requêtes exécutées : la recherche sémantique + le boost temporel.
    assert len(session.executed) == 2


@pytest.mark.asyncio
async def test_retrieve_relevant_fields_non_temporal_question_skips_boost():
    """Sans marqueur temporel dans la question, une seule requête (sémantique) est exécutée."""

    class _Row:
        def __init__(self, qualified_name, object_type, parent_table, description):
            self.qualified_name = qualified_name
            self.object_type = object_type
            self.parent_table = parent_table
            self.description = description

    row = _Row("Sales[Region]", "column", "Sales", "Région de vente")
    session = _FakeSession(execute_results=[_FakeResult(rows=[row])])

    with (
        patch("app.services.schema_rag.call_embedding", AsyncMock(return_value=[[0.1, 0.2]])),
        patch("app.services.schema_rag.AsyncSessionLocal", _session_factory(session)),
    ):
        await retrieve_relevant_fields("tenant-1", "AdventureWorks", "répartition par catégorie")

    assert len(session.executed) == 1, "Pas de marqueur temporel → pas de requête de boost"


@pytest.mark.asyncio
async def test_retrieve_relevant_fields_falls_back_to_empty_on_embedding_failure():
    with patch(
        "app.services.schema_rag.call_embedding",
        AsyncMock(side_effect=RuntimeError("embedding API down")),
    ):
        fields = await retrieve_relevant_fields("tenant-1", "AdventureWorks", "ventes par région")

    assert fields == []


@pytest.mark.asyncio
async def test_retrieve_relevant_fields_falls_back_to_empty_on_db_failure():
    class _FailingSession(_FakeSession):
        async def execute(self, stmt):
            raise RuntimeError("db unreachable")

    with (
        patch("app.services.schema_rag.call_embedding", AsyncMock(return_value=[[0.1, 0.2]])),
        patch("app.services.schema_rag.AsyncSessionLocal", _session_factory(_FailingSession())),
    ):
        fields = await retrieve_relevant_fields("tenant-1", "AdventureWorks", "ventes par région")

    assert fields == []


# ── update_field_from_hitl ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_update_field_from_hitl_marks_verified():
    class _FakeRow:
        def __init__(self):
            self.description = ""
            self.embedding = None
            self.human_verified = False
            self.confidence = 0.6

    row = _FakeRow()
    session = _FakeSession(execute_results=[_FakeResult(scalar=row)])

    with (
        patch("app.services.schema_rag.call_embedding", AsyncMock(return_value=[[0.9, 0.9]])),
        patch("app.services.schema_rag.AsyncSessionLocal", _session_factory(session)),
    ):
        updated = await update_field_from_hitl(
            "tenant-1", "AdventureWorks", "Sales[Region]", "Région de vente validée"
        )

    assert updated is True
    assert row.description == "Région de vente validée"
    assert row.human_verified is True
    assert row.confidence == 1.0
    assert session.committed is True


@pytest.mark.asyncio
async def test_update_field_from_hitl_returns_false_when_field_not_found():
    session = _FakeSession(execute_results=[_FakeResult(scalar=None)])

    with (
        patch("app.services.schema_rag.call_embedding", AsyncMock(return_value=[[0.9, 0.9]])),
        patch("app.services.schema_rag.AsyncSessionLocal", _session_factory(session)),
    ):
        updated = await update_field_from_hitl(
            "tenant-1", "AdventureWorks", "Sales[Inexistant]", "description"
        )

    assert updated is False
    assert session.committed is False


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
