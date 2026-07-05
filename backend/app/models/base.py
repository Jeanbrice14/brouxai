from __future__ import annotations

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Classe de base déclarative SQLAlchemy — point d'ancrage unique pour Alembic."""
