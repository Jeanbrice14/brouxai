from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Chemin absolu vers le .env (peu importe le répertoire de lancement)
_ENV_FILE = Path(__file__).parent.parent / ".env"
# Surcharges locales, non commitées (ex: bypass dev-only comme POWERBI_MCP_SKIP_CONFIRMATION).
# Chargé après .env — ses valeurs sont prioritaires. Absent par défaut, jamais requis.
_ENV_LOCAL_FILE = Path(__file__).parent.parent / ".env.local"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(str(_ENV_FILE), str(_ENV_LOCAL_FILE)), extra="ignore"
    )

    # LLM
    openai_api_key: str = ""
    litellm_default_model: str = "gpt-5.4"
    litellm_cheap_model: str = "gpt-5.4-mini"

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

    # Power BI Desktop local (MCP) — v1+, source de données alternative au CSV/DuckDB
    powerbi_mcp_command: str = "npx"
    powerbi_mcp_args: str = "-y @microsoft/powerbi-modeling-mcp@latest --start"
    powerbi_pbix_file_name: str = ""
    powerbi_mcp_skip_confirmation: bool = False
    # "llm" : notre prompt LiteLLM génère le DAX. "mcp_native" : délègue au tool
    # de génération DAX du serveur MCP si disponible (repli automatique sur "llm" sinon).
    powerbi_dax_generation_mode: str = "llm"

    # RAG schéma sémantique (embeddings stockés dans Redis, cf. app/services/schema_rag.py)
    # — mode powerbi_local uniquement
    schema_rag_embedding_model: str = "text-embedding-3-small"
    # Nombre de champs de schéma retournés par retrieve_relevant_fields — configurable
    # pour ajustement empirique sans redéploiement (valeur de départ, à affiner).
    schema_rag_k: int = 15
    # TTL du hash de schéma en cache Redis (secondes) — 30 jours : le schéma d'un modèle
    # Power BI change rarement ; expirer force juste un recalcul de hash, jamais un blocage.
    schema_rag_hash_ttl_seconds: int = 2_592_000

    # Mémoire de conversation — nombre de tours (question, réponse résumée) conservés par
    # session pour la résolution de références ("cette catégorie", "et le mois dernier ?").
    # Résumé texte seulement (jamais les agrégats bruts) — coût négligeable même injecté
    # systématiquement, pas besoin de détection heuristique de pertinence.
    chat_history_k: int = 3


settings = Settings()
