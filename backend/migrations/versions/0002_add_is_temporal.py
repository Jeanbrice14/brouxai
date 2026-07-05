"""add is_temporal to semantic_field_embeddings

Revision ID: 0002
Revises: 0001
Create Date: 2026-07-03

"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "semantic_field_embeddings",
        sa.Column("is_temporal", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    # Sert la requête de boost hybride (retrieve_relevant_fields) : tenant+modèle+is_temporal.
    op.create_index(
        "ix_semantic_field_embeddings_temporal",
        "semantic_field_embeddings",
        ["tenant_id", "model_id", "is_temporal"],
    )


def downgrade() -> None:
    op.drop_index("ix_semantic_field_embeddings_temporal", table_name="semantic_field_embeddings")
    op.drop_column("semantic_field_embeddings", "is_temporal")
