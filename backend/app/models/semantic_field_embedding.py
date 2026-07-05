from __future__ import annotations

import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, Float, String, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column

from app.models.base import Base

# Dimension de text-embedding-3-small (voir services/llm.py::call_embedding).
EMBEDDING_DIM = 1536


class SemanticFieldEmbedding(Base):
    """Un champ de schéma Power BI (table/colonne/mesure/relation) indexé pour le RAG.

    Clé d'unicité : (tenant_id, model_id, qualified_name). `model_id` identifie le modèle
    Power BI (aujourd'hui : `pbix_file_name`) — volontairement pas `session_id` : l'index
    et les corrections humaines (`human_verified`) doivent survivre à travers plusieurs
    sessions de chat sur le même fichier, pas être reconstruits à chaque nouvelle session.
    """

    __tablename__ = "semantic_field_embeddings"
    __table_args__ = (
        UniqueConstraint(
            "tenant_id", "model_id", "qualified_name", name="uq_semantic_field_tenant_model_qname"
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    tenant_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    model_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    qualified_name: Mapped[str] = mapped_column(String, nullable=False)  # "Table[Champ]" / "[Mesure]"
    object_type: Mapped[str] = mapped_column(String, nullable=False)  # table|column|measure|relationship
    parent_table: Mapped[str | None] = mapped_column(String, nullable=True)
    description: Mapped[str] = mapped_column(String, nullable=False, default="")
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM), nullable=False)
    human_verified: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    # Calculé une fois à l'indexation (dtype DateTime/Date/Time OU nom de table/champ
    # évoquant un calendrier) — persisté plutôt que re-dérivé à chaque retrieval : plus
    # robuste (utilise le dtype réel disponible seulement à l'indexation) et corrigible
    # manuellement plus tard si l'heuristique se trompe. Voir schema_rag.py::_is_temporal_field.
    is_temporal: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    updated_at: Mapped[datetime] = mapped_column(
        server_default=func.now(), onupdate=func.now(), nullable=False
    )
