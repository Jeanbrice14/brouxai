"""create semantic_field_embeddings table

Revision ID: 0001
Revises:
Create Date: 2026-07-03

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_EMBEDDING_DIM = 1536  # text-embedding-3-small


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "semantic_field_embeddings",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("tenant_id", sa.String(), nullable=False),
        sa.Column("model_id", sa.String(), nullable=False),
        sa.Column("qualified_name", sa.String(), nullable=False),
        sa.Column("object_type", sa.String(), nullable=False),
        sa.Column("parent_table", sa.String(), nullable=True),
        sa.Column("description", sa.String(), nullable=False, server_default=""),
        sa.Column("embedding", Vector(_EMBEDDING_DIM), nullable=False),
        sa.Column("human_verified", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("confidence", sa.Float(), nullable=False, server_default="0"),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.UniqueConstraint(
            "tenant_id", "model_id", "qualified_name", name="uq_semantic_field_tenant_model_qname"
        ),
    )
    op.create_index(
        "ix_semantic_field_embeddings_tenant_id", "semantic_field_embeddings", ["tenant_id"]
    )
    op.create_index(
        "ix_semantic_field_embeddings_model_id", "semantic_field_embeddings", ["model_id"]
    )
    op.create_index(
        "ix_semantic_field_embeddings_tenant_model",
        "semantic_field_embeddings",
        ["tenant_id", "model_id"],
    )
    # Pas d'index ANN (ivfflat/hnsw) sur `embedding` : un modèle Power BI compte typiquement
    # quelques dizaines à ~200 champs, un scan séquentiel avec l'opérateur cosine suffit très
    # largement. À reconsidérer seulement si le volume de champs par (tenant_id, model_id)
    # devient significativement plus grand.


def downgrade() -> None:
    op.drop_index("ix_semantic_field_embeddings_tenant_model", table_name="semantic_field_embeddings")
    op.drop_index("ix_semantic_field_embeddings_model_id", table_name="semantic_field_embeddings")
    op.drop_index("ix_semantic_field_embeddings_tenant_id", table_name="semantic_field_embeddings")
    op.drop_table("semantic_field_embeddings")
