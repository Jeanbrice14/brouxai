from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.config import settings

# Premier point d'entrée SQLAlchemy du projet — jusqu'ici (RAG schéma pgvector mis à part)
# aucune connexion Postgres n'était établie depuis le code applicatif.
engine = create_async_engine(settings.database_url, pool_pre_ping=True)

AsyncSessionLocal = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
