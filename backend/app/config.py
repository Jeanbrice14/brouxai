from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # LLM
    openai_api_key: str = ""
    litellm_default_model: str = "gpt-4o"
    litellm_cheap_model: str = "gpt-4o-mini"

    # Base de données
    database_url: str = "postgresql+asyncpg://narr8:narr8dev@localhost:5432/narr8"
    database_url_sync: str = "postgresql://narr8:narr8dev@localhost:5432/narr8"

    # Redis
    redis_url: str = "redis://localhost:6379/0"

    # Storage
    storage_endpoint: str = "http://localhost:9000"
    storage_key: str = "minioadmin"
    storage_secret: str = "minioadmin"
    storage_bucket: str = "narr8-dev"

    # App
    app_env: str = "development"
    secret_key: str = "dev-secret-change-in-prod"
    debug: bool = True

    # HITL seuils
    hitl_confidence_threshold: float = 0.80
    hitl_metadata_confidence_threshold: float = 0.85
    hitl_orphan_rate_threshold: float = 0.05


settings = Settings()
