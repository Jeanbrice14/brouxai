from __future__ import annotations

import hashlib

import structlog
from sqlalchemy import delete, func, select
from sqlalchemy.dialects.postgresql import insert

from app.config import settings
from app.db import AsyncSessionLocal
from app.models.semantic_field_embedding import SemanticFieldEmbedding
from app.services.cache import get_cache, set_cache
from app.services.llm import call_embedding

logger = structlog.get_logger(__name__)

_HASH_CACHE_PREFIX = "schema_rag:hash:"

# ── Détection temporelle (retrieval hybride, voir retrieve_relevant_fields) ────

# Types Power BI qui indiquent sans ambiguïté un champ temporel.
_TEMPORAL_DTYPES = {"datetime", "date", "time"}

# Mots-clés sur le nom qualifié/table — capture les colonnes calendrier typées
# Int64/String (ex: AdventureWorks "Month": Int64, "Month Name": String) que le dtype
# seul raterait, ainsi que les tables calendrier elles-mêmes.
_TEMPORAL_FIELD_KEYWORDS = [
    "calendar", "calendrier", "date", "time", "période", "periode",
    "month", "mois", "year", "année", "annee", "quarter", "trimestre",
    "week", "semaine", "day", "jour",
]

# Marqueurs dans la QUESTION utilisateur qui indiquent un besoin d'évolution temporelle —
# déclenche le boost hybride (inclusion forcée des champs is_temporal).
_TEMPORAL_QUESTION_KEYWORDS = [
    "mois", "mensuel", "année", "annuel", "trimestre", "trimestriel",
    "évolution", "evolution", "tendance", "historique", "dans le temps",
    "au fil du", "semaine", "hebdomadaire", "progression", "par mois",
    "par année", "sur l'année", "quotidien", "chronologique",
]


def _is_temporal_field(qualified_name: str, parent_table: str | None, dtype: str) -> bool:
    """Heuristique combinée : dtype réel (fiable, disponible seulement à l'indexation)
    OU nom de table/champ évoquant un calendrier (capture aussi les colonnes calendrier
    typées Int64/String, ex: "Month" en Int64 dans AdventureWorks)."""
    if dtype.lower() in _TEMPORAL_DTYPES:
        return True
    haystack = f"{qualified_name} {parent_table or ''}".lower()
    return any(kw in haystack for kw in _TEMPORAL_FIELD_KEYWORDS)


def _question_has_temporal_marker(question: str) -> bool:
    q = question.lower()
    return any(kw in q for kw in _TEMPORAL_QUESTION_KEYWORDS)


# ── Helpers de transformation semantic_model_info ──────────────────────────────


def _flatten_fields(semantic_model_info: dict) -> list[dict]:
    """Aplati tables/colonnes/mesures/relations en une liste de champs indexables.

    Chaque champ : {qualified_name, object_type, parent_table, description, dtype,
    is_temporal}.
    """
    fields: list[dict] = []

    tables = semantic_model_info.get("tables", {}) or {}
    for table_name, table_info in tables.items():
        if not isinstance(table_info, dict):
            table_info = {}
        fields.append(
            {
                "qualified_name": table_name,
                "object_type": "table",
                "parent_table": None,
                "description": table_info.get("description") or "",
                "dtype": "",
                "is_temporal": _is_temporal_field(table_name, None, ""),
            }
        )
        columns = table_info.get("columns", {}) or {}
        for col_name, col_info in columns.items():
            if not isinstance(col_info, dict):
                col_info = {}
            dtype = col_info.get("dataType", col_info.get("type", "")) or ""
            qualified_name = f"{table_name}[{col_name}]"
            fields.append(
                {
                    "qualified_name": qualified_name,
                    "object_type": "column",
                    "parent_table": table_name,
                    "description": col_info.get("description") or "",
                    "dtype": dtype,
                    "is_temporal": _is_temporal_field(qualified_name, table_name, dtype),
                }
            )

    measures = semantic_model_info.get("measures", {}) or {}
    for measure_name, measure_info in measures.items():
        if not isinstance(measure_info, dict):
            measure_info = {}
        dtype = measure_info.get("dataType", "") or ""
        qualified_name = f"[{measure_name}]"
        fields.append(
            {
                "qualified_name": qualified_name,
                "object_type": "measure",
                "parent_table": None,
                "description": measure_info.get("description") or "",
                "dtype": dtype,
                "is_temporal": _is_temporal_field(qualified_name, None, dtype),
            }
        )

    relations = semantic_model_info.get("relations", []) or []
    for rel in relations:
        if not isinstance(rel, dict):
            continue
        from_table = rel.get("fromTable", "?")
        from_col = rel.get("fromColumn", "?")
        to_table = rel.get("toTable", "?")
        to_col = rel.get("toColumn", "?")
        qualified_name = f"{from_table}[{from_col}]->{to_table}[{to_col}]"
        fields.append(
            {
                "qualified_name": qualified_name,
                "object_type": "relationship",
                "parent_table": from_table,
                "description": f"Relation {from_table}.{from_col} vers {to_table}.{to_col}",
                "dtype": "",
                "is_temporal": _is_temporal_field(qualified_name, from_table, ""),
            }
        )

    return fields


def _embedding_text(field: dict) -> str:
    """Texte source de l'embedding — nom qualifié + type + description si disponible."""
    parts = [field["qualified_name"]]
    if field.get("dtype"):
        parts.append(f"type: {field['dtype']}")
    if field.get("description"):
        parts.append(field["description"])
    return " — ".join(parts)


def compute_model_hash(semantic_model_info: dict) -> str:
    """Hash structurel du schéma (noms + types), volontairement SANS les descriptions.

    Les descriptions peuvent changer via CP1 sans que le schéma structurel change —
    un hash qui les inclurait forcerait une réindexation complète à chaque correction
    humaine, ce qui est exactement ce qu'on veut éviter.
    """
    fields = _flatten_fields(semantic_model_info)
    structural = sorted(f"{f['object_type']}:{f['qualified_name']}:{f['dtype']}" for f in fields)
    payload = "|".join(structural)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _hash_cache_key(tenant_id: str, model_id: str) -> str:
    return f"{_HASH_CACHE_PREFIX}{tenant_id}:{model_id}"


# ── Indexation ──────────────────────────────────────────────────────────────────


async def index_schema(
    tenant_id: str,
    model_id: str,
    semantic_model_info: dict,
) -> dict:
    """Indexe (ou réutilise) le schéma sémantique Power BI dans pgvector.

    Compare un hash structurel à celui stocké en Redis (par tenant+modèle, pas par
    session — l'index doit survivre à travers plusieurs sessions de chat sur le même
    fichier Power BI) :
    - Hash identique → skip la réindexation complète (ne touche à rien).
    - Hash différent → réindexe, mais préserve intégralement les entrées
      human_verified=true déjà en base (ne jamais écraser une description validée par
      un humain avec une description vide issue d'un nouveau fetch MCP).

    Ne bloque jamais le pipeline : les erreurs sont catchées et loggées, jamais levées —
    l'appelant (MetadataAgent) doit pouvoir continuer même si l'indexation RAG échoue
    (DataAgent retombera sur le schéma complet via son propre fallback).

    Returns:
        {"reindexed": bool, "n_fields": int, "hash": str}
    """
    new_hash = compute_model_hash(semantic_model_info)
    cache_key = _hash_cache_key(tenant_id, model_id)

    try:
        cached = await get_cache(cache_key)
        if cached and cached.get("hash") == new_hash:
            logger.info("schema_rag_hash_unchanged", tenant_id=tenant_id, model_id=model_id)
            return {"reindexed": False, "n_fields": 0, "hash": new_hash}

        fields = _flatten_fields(semantic_model_info)
        qualified_names = {f["qualified_name"] for f in fields}

        async with AsyncSessionLocal() as session:
            existing_verified = await session.execute(
                select(SemanticFieldEmbedding.qualified_name).where(
                    SemanticFieldEmbedding.tenant_id == tenant_id,
                    SemanticFieldEmbedding.model_id == model_id,
                    SemanticFieldEmbedding.human_verified.is_(True),
                )
            )
            verified_names = {row[0] for row in existing_verified.all()}

            to_embed = [f for f in fields if f["qualified_name"] not in verified_names]
            texts = [_embedding_text(f) for f in to_embed]
            vectors = await call_embedding(texts) if texts else []

            for field, vector in zip(to_embed, vectors):
                stmt = insert(SemanticFieldEmbedding).values(
                    tenant_id=tenant_id,
                    model_id=model_id,
                    qualified_name=field["qualified_name"],
                    object_type=field["object_type"],
                    parent_table=field["parent_table"],
                    description=field["description"],
                    embedding=vector,
                    human_verified=False,
                    confidence=0.6 if field["description"] else 0.3,
                    is_temporal=field["is_temporal"],
                )
                stmt = stmt.on_conflict_do_update(
                    index_elements=["tenant_id", "model_id", "qualified_name"],
                    set_={
                        "object_type": stmt.excluded.object_type,
                        "parent_table": stmt.excluded.parent_table,
                        "description": stmt.excluded.description,
                        "embedding": stmt.excluded.embedding,
                        "confidence": stmt.excluded.confidence,
                        "is_temporal": stmt.excluded.is_temporal,
                        "updated_at": func.now(),
                    },
                    # Ne réécrit jamais une ligne human_verified=true (garde-fou supplémentaire
                    # au-delà du filtre `to_embed` ci-dessus, en cas de course entre 2 indexations).
                    where=SemanticFieldEmbedding.human_verified.is_(False),
                )
                await session.execute(stmt)

            # Supprime les entrées non-vérifiées devenues obsolètes (champ disparu du nouveau
            # schéma) — jamais les entrées human_verified, même orphelines.
            if qualified_names:
                await session.execute(
                    delete(SemanticFieldEmbedding).where(
                        SemanticFieldEmbedding.tenant_id == tenant_id,
                        SemanticFieldEmbedding.model_id == model_id,
                        SemanticFieldEmbedding.human_verified.is_(False),
                        SemanticFieldEmbedding.qualified_name.notin_(qualified_names),
                    )
                )

            await session.commit()

        await set_cache(cache_key, {"hash": new_hash}, ttl=settings.schema_rag_hash_ttl_seconds)
        logger.info(
            "schema_rag_indexed",
            tenant_id=tenant_id,
            model_id=model_id,
            n_fields=len(fields),
            n_embedded=len(to_embed),
            n_preserved_verified=len(verified_names),
        )
        return {"reindexed": True, "n_fields": len(fields), "hash": new_hash}
    except Exception as exc:
        logger.warning("schema_rag_index_failed", tenant_id=tenant_id, model_id=model_id, error=str(exc))
        return {"reindexed": False, "n_fields": 0, "hash": new_hash, "error": str(exc)}


# ── Retrieval ─────────────────────────────────────────────────────────────────


_MAX_TEMPORAL_BOOST_FIELDS = 30


def _row_to_field(row: SemanticFieldEmbedding) -> dict:
    return {
        "qualified_name": row.qualified_name,
        "object_type": row.object_type,
        "parent_table": row.parent_table,
        "description": row.description,
    }


async def retrieve_relevant_fields(
    tenant_id: str,
    model_id: str,
    question: str,
    k: int | None = None,
) -> list[dict]:
    """Retourne les k champs de schéma les plus pertinents pour `question`.

    Recherche par similarité cosine dans pgvector. Ne bloque JAMAIS la génération DAX :
    retourne une liste vide (jamais d'exception) si l'embedding échoue, si la requête
    pgvector échoue, ou si aucune entrée n'existe encore pour ce modèle — l'appelant
    (DataAgent) doit alors retomber sur le schéma complet.

    Retrieval hybride : le retrieval sémantique pur rate systématiquement les champs
    calendrier sur les questions d'évolution temporelle (ex: "évolution des ventes par
    mois" ne remonte pas la table calendrier dans le top k — testé et confirmé contre
    AdventureWorks). Si la question contient un marqueur temporel
    (_question_has_temporal_marker), les champs `is_temporal=true` du modèle sont
    ajoutés au résultat sémantique (union dédupliquée par qualified_name, jamais un
    remplacement) — une panne de ce boost n'invalide pas le résultat sémantique déjà
    obtenu, elle est juste loggée et ignorée.
    """
    _k = k or settings.schema_rag_k

    try:
        query_vector = (await call_embedding([question]))[0]
    except Exception as exc:
        logger.warning("schema_rag_embedding_failed", error=str(exc))
        return []

    try:
        async with AsyncSessionLocal() as session:
            stmt = (
                select(SemanticFieldEmbedding)
                .where(
                    SemanticFieldEmbedding.tenant_id == tenant_id,
                    SemanticFieldEmbedding.model_id == model_id,
                )
                .order_by(SemanticFieldEmbedding.embedding.cosine_distance(query_vector))
                .limit(_k)
            )
            result = await session.execute(stmt)
            rows = result.scalars().all()
            fields = [_row_to_field(r) for r in rows]

            if _question_has_temporal_marker(question):
                try:
                    temporal_stmt = (
                        select(SemanticFieldEmbedding)
                        .where(
                            SemanticFieldEmbedding.tenant_id == tenant_id,
                            SemanticFieldEmbedding.model_id == model_id,
                            SemanticFieldEmbedding.is_temporal.is_(True),
                        )
                        .limit(_MAX_TEMPORAL_BOOST_FIELDS)
                    )
                    temporal_result = await session.execute(temporal_stmt)
                    temporal_rows = temporal_result.scalars().all()

                    seen = {f["qualified_name"] for f in fields}
                    n_added = 0
                    for row in temporal_rows:
                        field = _row_to_field(row)
                        if field["qualified_name"] not in seen:
                            fields.append(field)
                            seen.add(field["qualified_name"])
                            n_added += 1
                    logger.info(
                        "schema_rag_temporal_boost_applied",
                        question=question,
                        n_temporal_candidates=len(temporal_rows),
                        n_added=n_added,
                    )
                except Exception as exc:
                    logger.warning("schema_rag_temporal_boost_failed", error=str(exc))
    except Exception as exc:
        logger.warning("schema_rag_retrieval_failed", error=str(exc))
        return []

    return fields


# ── Mise à jour ponctuelle depuis CP1 ────────────────────────────────────────────


async def update_field_from_hitl(
    tenant_id: str,
    model_id: str,
    qualified_name: str,
    verified_description: str,
) -> bool:
    """Ré-embed et met à jour UN SEUL champ suite à une validation humaine CP1.

    Marque human_verified=true et confidence=1.0. N'effectue jamais de réindexation
    complète du schéma. Silencieux (retourne False) si le champ est introuvable — un
    champ inconnu ne doit pas faire échouer la reprise HITL.
    """
    try:
        vector = (
            await call_embedding(
                [_embedding_text({"qualified_name": qualified_name, "dtype": "", "description": verified_description})]
            )
        )[0]

        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(SemanticFieldEmbedding).where(
                    SemanticFieldEmbedding.tenant_id == tenant_id,
                    SemanticFieldEmbedding.model_id == model_id,
                    SemanticFieldEmbedding.qualified_name == qualified_name,
                )
            )
            row = result.scalar_one_or_none()
            if row is None:
                logger.warning(
                    "schema_rag_hitl_field_not_found",
                    tenant_id=tenant_id,
                    model_id=model_id,
                    qualified_name=qualified_name,
                )
                return False

            row.description = verified_description
            row.embedding = vector
            row.human_verified = True
            row.confidence = 1.0
            await session.commit()

        logger.info("schema_rag_hitl_updated", qualified_name=qualified_name)
        return True
    except Exception as exc:
        logger.warning(
            "schema_rag_hitl_update_failed", qualified_name=qualified_name, error=str(exc)
        )
        return False
