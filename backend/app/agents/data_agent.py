from __future__ import annotations

import asyncio
from pathlib import Path
from urllib.parse import urlparse

import duckdb
import pandas as pd
import structlog

from app.agents.base_agent import BaseAgent
from app.config import settings
from app.pipeline.state import PipelineState
from app.services.cache import get_cache, make_cache_key, set_cache
from app.services.llm import call_llm_json
from app.services.storage import read_dataframe

logger = structlog.get_logger(__name__)

_MAX_ROWS = 500
_EXEC_TIMEOUT = 30.0

_SQL_SYSTEM_PROMPT = (
    "Tu es un expert SQL. "
    "Génère des requêtes SQL courtes et correctes pour DuckDB. "
    "Les tables sont enregistrées par leur nom de fichier (sans extension). "
    "Retourne UNIQUEMENT du JSON valide avec une clé 'queries' : "
    "liste d'objets {\"key\": \"nom_agrégat\", \"sql\": \"SELECT ...\"}."
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _table_name(ref: str) -> str:
    return Path(urlparse(ref).path).stem


def _build_schema_summary(state: PipelineState, dfs: dict[str, pd.DataFrame]) -> str:
    """Résumé structurel du schéma — jamais de données brutes, uniquement des métadonnées.

    Contient : noms de tables, noms de colonnes, types sémantiques, unités, grain, relations.
    Limité à 2000 caractères.
    """
    lines: list[str] = []
    files_meta = state.get("metadata", {}).get("files", {})

    for tname, df in dfs.items():
        lines.append(f"Table: {tname} ({len(df)} lignes, {len(df.columns)} colonnes)")

        meta_info: dict = {}
        for ref, meta in files_meta.items():
            if _table_name(ref) == tname:
                meta_info = meta
                break

        col_meta = meta_info.get("columns", {})
        for col in df.columns:
            info = col_meta.get(col, {})
            semantic = info.get("semantic_name", col)
            col_type = info.get("type", str(df[col].dtype))
            unit = info.get("unit", "")
            unit_str = f" [{unit}]" if unit else ""
            lines.append(f"  - {col} ({semantic}, {col_type}{unit_str})")

        grain = meta_info.get("grain", "")
        if grain:
            lines.append(f"  Grain: {grain}")

    relations = state.get("schema", {}).get("relations", [])
    if relations:
        lines.append("\nRelations détectées :")
        for rel in relations[:5]:
            desc = rel.get("description", "")
            lines.append(
                f"  {rel['table_a']}.{rel['col_a']} -> {rel['table_b']}.{rel['col_b']}: {desc}"
            )

    summary = "\n".join(lines)
    return summary[:2000]


def _auto_join_hints(dfs: dict[str, pd.DataFrame]) -> str:
    """Detect potential JOIN keys by finding column names shared across two or more tables."""
    table_names = list(dfs.keys())
    hints: list[str] = []
    for i, t1 in enumerate(table_names):
        for t2 in table_names[i + 1:]:
            shared = sorted(set(dfs[t1].columns) & set(dfs[t2].columns))
            for col in shared:
                hints.append(f"  {t1}.{col} = {t2}.{col}")
    return "\n".join(hints) if hints else "  (aucune détectée automatiquement)"


def _build_sql_prompt(user_prompt: str, schema_summary: str, dfs: dict) -> str:
    """Prompt LLM pour la génération SQL — ne contient que des noms de colonnes, pas de valeurs."""
    table_list = "\n".join(
        f"  - {name}: colonnes = {list(df.columns)}" for name, df in dfs.items()
    )
    join_hints = _auto_join_hints(dfs)
    return (
        f"Demande utilisateur : {user_prompt}\n\n"
        f"Schéma :\n{schema_summary}\n\n"
        f"Tables disponibles dans DuckDB (noms exacts à utiliser dans FROM/JOIN) :\n{table_list}\n\n"
        f"Clés de jointure détectées (colonnes partagées entre tables) :\n{join_hints}\n\n"
        "Génère des requêtes SQL DuckDB qui répondent à la demande.\n"
        "Règles :\n"
        "  - Utilise uniquement les tables listées ci-dessus\n"
        "  - Préfère une seule requête GROUP BY qui calcule toutes les métriques demandées à la fois\n"
        "  - Chaque requête retourne un agrégat (jamais les données brutes)\n"
        "  - Utilise des alias explicites en français pour toutes les colonnes calculées (ex: AS mois, AS ca_total, AS atteinte_pct, AS commercial)\n"
        "  - Comparaison cross-tables (ex: comparer le nombre de ventes vs le nombre de clients) : utilise UNION ALL retournant exactement deux colonnes (metric TEXT, valeur NUMERIC)\n"
        "    Ex: SELECT 'nb_ventes' AS metric, COUNT(*) AS valeur FROM ventes UNION ALL SELECT 'nb_clients' AS metric, COUNT(*) AS valeur FROM clients\n"
        "  - KPI ou valeur unique (moyenne, somme d'UNE seule métrique) : SELECT uniquement la colonne demandée avec un alias français. Ne calcule PAS la moyenne de toutes les colonnes numériques.\n"
        "    Correct  : SELECT AVG(ca_ht) AS ca_ht_moyen FROM ventes\n"
        "    Incorrect: SELECT AVG(vente_id) AS vente_id_mean, AVG(client_id) AS client_id_mean, AVG(ca_ht) AS ca_ht_mean FROM ventes\n"
        "  - Joins : utilise les clés de jointure listées ci-dessus\n"
        "  - Évolution temporelle (tendance, évolution mensuelle/annuelle) : GROUP BY DATE_TRUNC('month', col_date) avec alias français 'mois'. Toujours ORDER BY mois.\n"
        "    Ex : SELECT DATE_TRUNC('month', date) AS mois, SUM(ca_ht) AS ca_mensuel FROM ventes GROUP BY mois ORDER BY mois\n"
        "  - Dates : les colonnes date/mois sont déjà converties en TIMESTAMP par le moteur. "
        "Utilise DATE_TRUNC('month', colonne) AS mois (alias TOUJOURS en français minuscule).\n"
        "    Pour les colonnes de type 'mois' (ex: budget.mois), utilise aussi DATE_TRUNC car elles sont en TIMESTAMP.\n"
        "    Jointure date↔mois : DATE_TRUNC('month', ventes.date) = DATE_TRUNC('month', budget.mois)\n"
        '  - Ex correct : {"key": "revenue_by_subcat", "sql": "SELECT s.SubcategoryName, SUM(sa.Revenue) AS TotalRevenue FROM sales sa JOIN products p ON sa.ProductKey = p.ProductKey JOIN subcategory s ON p.ProductSubcategoryKey = s.ProductSubcategoryKey GROUP BY s.SubcategoryName"}\n'
        'Retourne : {"queries": [{"key": "nom_aggregat", "sql": "SELECT ..."}, ...]}'
    )


def _preprocess_df(df: pd.DataFrame) -> pd.DataFrame:
    """Convert object-typed date-like columns to datetime so DuckDB can apply DATE_TRUNC.

    Uses format='mixed' (pandas >= 2.0) which handles formats like RFC 2822.
    """
    import re
    _DATE_PATTERN = re.compile(r"date|day|month|year|time|period|mois|periode", re.I)
    df = df.copy()
    for col in df.columns:
        if _DATE_PATTERN.search(col) and not pd.api.types.is_datetime64_any_dtype(df[col]):
            parsed = None
            # Try YYYY-MM format first (e.g. budget.mois = "2024-01")
            try:
                candidate = pd.to_datetime(df[col], format="%Y-%m", errors="coerce")
                if candidate.notna().mean() > 0.7:
                    parsed = candidate
            except Exception:
                pass
            # Fall back to mixed format (handles ISO 8601, RFC 2822, etc.)
            if parsed is None:
                try:
                    candidate = pd.to_datetime(df[col], format="mixed", dayfirst=False, errors="coerce")
                    if candidate.notna().mean() > 0.7:
                        parsed = candidate
                except Exception:
                    try:
                        candidate = pd.to_datetime(df[col], errors="coerce")
                        if candidate.notna().mean() > 0.7:
                            parsed = candidate
                    except Exception:
                        pass
            if parsed is not None:
                df[col] = parsed
    return df


def _run_sql_sync(queries: list[dict], dfs: dict[str, pd.DataFrame]) -> dict:
    """Exécute les requêtes SQL dans un connexion DuckDB locale (synchrone, appelé via to_thread)."""
    conn = duckdb.connect()
    try:
        for name, df in dfs.items():
            conn.register(name, _preprocess_df(df))
        result: dict = {}
        for q in queries:
            key = q.get("key") or "query_result"
            sql = q.get("sql", "").strip()
            if sql:
                result[key] = conn.execute(sql).df()
        return result
    finally:
        conn.close()


def _serialize_result(raw_result: dict) -> dict:
    """Convertit les DataFrames en list-of-dicts, 500 lignes max par agrégat."""
    aggregates: dict = {}
    for key, value in raw_result.items():
        if isinstance(value, pd.DataFrame):
            rows = value.head(_MAX_ROWS).to_dict(orient="records")
        else:
            rows = [{"value": value}]
        aggregates[key] = _clean_records(rows)
    return aggregates


def _clean_records(records: list[dict]) -> list[dict]:
    """Normalise les types numpy/NaN/Timestamp pour la sérialisation JSON."""
    import datetime
    cleaned = []
    for row in records:
        clean_row = {}
        for k, v in row.items():
            if hasattr(v, "item"):
                clean_row[k] = v.item()
            elif isinstance(v, float) and pd.isna(v):
                clean_row[k] = None
            elif isinstance(v, pd.Timestamp):
                clean_row[k] = v.strftime("%Y-%m")
            elif isinstance(v, (datetime.date, datetime.datetime)):
                clean_row[k] = v.strftime("%Y-%m")
            else:
                clean_row[k] = v
        cleaned.append(clean_row)
    return cleaned


def _compute_fallback(dfs: dict[str, pd.DataFrame]) -> dict:
    """Génère des agrégats basiques via DuckDB si la requête LLM échoue.

    Calcule COUNT, SUM, AVG, et série temporelle mensuelle si colonne date disponible.
    Ne plante jamais — même sur DataFrames vides.
    """
    import re as _re
    _DATE_COL_RE = _re.compile(r"date|jour|time|period|mois|periode", _re.I)

    aggregates: dict = {}
    conn = duckdb.connect()
    try:
        for name, df in dfs.items():
            if df.empty:
                aggregates[f"{name}_count"] = [{"count": 0}]
                continue
            conn.register(name, _preprocess_df(df))
            numeric_cols = df.select_dtypes(include="number").columns.tolist()

            # Sums & means (une seule colonne par agrégat pour éviter les colonnes parasites)
            if numeric_cols:
                cols_sum = ", ".join(f'SUM("{c}") AS "{c}_sum"' for c in numeric_cols)
                sums_df = conn.execute(f'SELECT {cols_sum} FROM "{name}"').df()
                aggregates[f"{name}_sums"] = _clean_records(sums_df.to_dict(orient="records"))

                for c in numeric_cols:
                    means_df = conn.execute(
                        f'SELECT ROUND(AVG("{c}"), 4) AS {c}_moyen FROM "{name}"'
                    ).df()
                    aggregates[f"{name}_{c}_moyen"] = _clean_records(means_df.to_dict(orient="records"))

            count_df = conn.execute(f'SELECT COUNT(*) AS count FROM "{name}"').df()
            aggregates[f"{name}_count"] = _clean_records(count_df.to_dict(orient="records"))

            # Série temporelle mensuelle si une colonne date existe
            date_cols = [c for c in df.columns if _DATE_COL_RE.search(c)]
            for date_col in date_cols[:1]:  # une seule colonne date
                try:
                    for metric in numeric_cols[:3]:  # au plus 3 métriques
                        ts_df = conn.execute(
                            f'SELECT DATE_TRUNC(\'month\', "{date_col}") AS mois, '
                            f'SUM("{metric}") AS {metric}_mensuel '
                            f'FROM "{name}" GROUP BY mois ORDER BY mois'
                        ).df()
                        if len(ts_df) > 1:
                            aggregates[f"{name}_{metric}_par_mois"] = _clean_records(
                                ts_df.to_dict(orient="records")
                            )
                except Exception:
                    pass
    finally:
        conn.close()
    return aggregates


# ── Agent ─────────────────────────────────────────────────────────────────────


class DataAgent(BaseAgent):
    """Agent 3 — Interprète le prompt, génère du SQL DuckDB, exécute les agrégations.

    RÈGLE ABSOLUE : les données brutes ne passent JAMAIS dans le LLM.
    Le LLM ne reçoit que : schema_summary (noms + types) + noms des colonnes disponibles.

    Étapes :
        A. Chargement des DataFrames depuis storage.
        B. Construction du schema_summary (métadonnées uniquement).
        C. Génération des requêtes SQL via LLM.
        D. Exécution DuckDB avec timeout 30s.
        E. Sérialisation des résultats (max 500 lignes).
        F. Fallback basique si l'exécution échoue.

    Cache Redis : TTL 1h par (tenant, datasets, prompt).

    Input  : state["prompt"] + state["schema"] + state["metadata"]
    Output : state["aggregates"]
    Modèle : settings.litellm_cheap_model (gpt-4o-mini)
    """

    name = "data_agent"

    async def run(self, state: PipelineState) -> PipelineState:
        log = logger.bind(report_id=state.get("report_id"))

        # ── Cache check ──────────────────────────────────────────────────────
        cache_key = await make_cache_key(
            state["tenant_id"], state["raw_data_refs"], state["prompt"]
        )
        cached = await get_cache(cache_key)
        if cached is not None:
            log.info("data_agent_cache_hit", key=cache_key)
            state["aggregates"] = cached
            return state

        # ── Étape A : chargement des DataFrames ──────────────────────────────
        dfs: dict[str, pd.DataFrame] = {}
        for ref in state["raw_data_refs"]:
            tname = _table_name(ref)
            dfs[tname] = await read_dataframe(ref)
            log.info("data_agent_loaded_df", table=tname, rows=len(dfs[tname]))

        # ── Étape B : schema_summary (aucune donnée brute) ───────────────────
        schema_summary = _build_schema_summary(state, dfs)

        # ── Étape C : génération SQL via LLM ──────────────────────────────────
        sql_prompt = _build_sql_prompt(state["prompt"], schema_summary, dfs)
        llm_result = await call_llm_json(
            prompt=sql_prompt,
            system=_SQL_SYSTEM_PROMPT,
            model=settings.litellm_cheap_model,
        )
        queries = llm_result.get("queries", [])
        if not isinstance(queries, list):
            queries = []
        log.info("data_agent_sql_generated", n_queries=len(queries))

        # ── Étapes D + E : exécution + sérialisation ─────────────────────────
        aggregates: dict = {}
        exec_success = False

        if queries:
            try:
                raw_result = await asyncio.wait_for(
                    asyncio.to_thread(_run_sql_sync, queries, dfs),
                    timeout=_EXEC_TIMEOUT,
                )
                aggregates = _serialize_result(raw_result)
                exec_success = True
                log.info("data_agent_sql_ok", n_keys=len(aggregates))
            except TimeoutError:
                msg = f"DataAgent: timeout SQL ({_EXEC_TIMEOUT}s)"
                state["errors"] = state.get("errors", []) + [msg]
                log.warning("data_agent_sql_timeout")
            except Exception as exc:
                msg = f"DataAgent: SQL error: {type(exc).__name__}: {exc}"
                state["errors"] = state.get("errors", []) + [msg]
                log.warning("data_agent_sql_error", error=str(exc))

        # ── Étape F : fallback ────────────────────────────────────────────────
        if not exec_success or not aggregates:
            if dfs:
                aggregates = _compute_fallback(dfs)
                warn = "DataAgent: fallback to basic aggregates"
                state["errors"] = state.get("errors", []) + [warn]
                log.warning("data_agent_fallback_used")

        state["aggregates"] = aggregates

        # ── Cache store ──────────────────────────────────────────────────────
        if aggregates:
            await set_cache(cache_key, aggregates, ttl=3600)

        return state
