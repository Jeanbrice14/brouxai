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
from app.services.conversation_memory import format_history_for_prompt
from app.services.llm import call_llm_json
from app.services.powerbi_local_mcp import PowerBIDaxExecutionError, get_powerbi_client
from app.services.schema_rag import retrieve_relevant_fields
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

_DAX_SYSTEM_PROMPT = (
    "Tu es un expert DAX (Power BI / Analysis Services). "
    "Génère des requêtes DAX courtes et correctes (EVALUATE ...). "
    "Utilise UNIQUEMENT les tables, colonnes et mesures listées — n'en invente jamais. "
    "Retourne UNIQUEMENT du JSON valide avec une clé 'queries' : "
    "liste d'objets {\"key\": \"nom_agrégat\", \"dax\": \"EVALUATE ...\"}."
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


def _build_sql_prompt(
    user_prompt: str, schema_summary: str, dfs: dict, chat_history: list[dict] | None = None
) -> str:
    """Prompt LLM pour la génération SQL — ne contient que des noms de colonnes, pas de valeurs."""
    table_list = "\n".join(
        f"  - {name}: colonnes = {list(df.columns)}" for name, df in dfs.items()
    )
    join_hints = _auto_join_hints(dfs)
    history_block = format_history_for_prompt(chat_history or [])
    history_section = (
        f"Questions précédentes de cette conversation (résumées, pour résoudre les "
        f"références comme \"cette catégorie\"/\"ce mois\" — ignore cette section si la "
        f"demande actuelle est autonome) :\n{history_block}\n\n"
        if history_block
        else ""
    )
    return (
        f"Demande utilisateur : {user_prompt}\n\n"
        f"{history_section}"
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


def _pbi_table_name(ref: str) -> str:
    """Extrait le nom de table depuis une ref 'powerbi://<table>' (cf. metadata_agent)."""
    return ref.removeprefix("powerbi://")


def _build_dax_schema_summary(state: PipelineState) -> str:
    """Résumé du modèle sémantique Power BI (mesures/tables/colonnes/relations), limité à 6000 chars.

    Les mesures sont listées EN PREMIER, avant le détail table par table — pas dans l'ordre
    d'itération d'origine (où "Mesures DAX disponibles" arrive après toutes les tables).
    Régression réelle (AdventureWorks, RAG indisponible donc ce repli complet est utilisé) :
    sur un modèle à nombreuses tables/colonnes, la troncature coupait la section mesures
    entièrement AVANT qu'elle apparaisse, laissant le LLM générer SUMMARIZECOLUMNS(...,
    [Revenue]) en traitant une colonne brute comme une mesure, faute de savoir laquelle des
    58 mesures réelles existait. Mettre les mesures en premier garantit qu'une troncature
    coupe en priorité le détail des colonnes (moins critique), jamais la liste des mesures.
    """
    files_meta = state.get("metadata", {}).get("files", {})

    measure_lines: list[str] = []
    table_lines: list[str] = []
    for ref, meta in files_meta.items():
        table_name = _pbi_table_name(ref)
        if table_name == "__measures__":
            measure_lines.append("Mesures DAX disponibles :")
            for col, info in meta.get("columns", {}).items():
                semantic = info.get("semantic_name", col)
                measure_lines.append(f"  - {col}: {semantic}")
            continue
        table_lines.append(f"Table: {table_name}")
        for col, info in meta.get("columns", {}).items():
            semantic = info.get("semantic_name", col)
            marker = "[mesure]" if info.get("kind") == "measure" else f"[{info.get('type', 'unknown')}]"
            table_lines.append(f"  - {col}: {semantic} {marker}")

    lines = measure_lines + table_lines

    relations = state.get("semantic_model_info", {}).get("relations", [])
    if relations:
        lines.append("\nRelations du modèle :")
        for rel in relations[:10]:
            if not isinstance(rel, dict):
                continue
            from_table = rel.get("fromTable", rel.get("table_a", "?"))
            from_col = rel.get("fromColumn", rel.get("col_a", "?"))
            to_table = rel.get("toTable", rel.get("table_b", "?"))
            to_col = rel.get("toColumn", rel.get("col_b", "?"))
            lines.append(f"  {from_table}.{from_col} -> {to_table}.{to_col}")

    return "\n".join(lines)[:6000]


def _build_dax_schema_summary_from_rag(fields: list[dict]) -> str:
    """Résumé du schéma construit depuis un sous-ensemble RAG (retrieve_relevant_fields),
    au même format que _build_dax_schema_summary — mesures listées en premier (voir sa
    docstring pour le pourquoi), colonnes groupées par table, relations en dernier."""
    by_table: dict[str, list[dict]] = {}
    measures: list[dict] = []
    relations: list[dict] = []

    for f in fields:
        if f["object_type"] == "measure":
            measures.append(f)
        elif f["object_type"] == "relationship":
            relations.append(f)
        elif f["object_type"] == "column":
            by_table.setdefault(f["parent_table"] or "?", []).append(f)
        # object_type == "table" : entrée informationnelle seule, rien à lister dessous

    lines: list[str] = []
    if measures:
        lines.append("Mesures DAX disponibles :")
        for m in measures:
            name = m["qualified_name"].strip("[]")
            desc = f" — {m['description']}" if m.get("description") else ""
            lines.append(f"  - {name}{desc}")

    for table_name, cols in by_table.items():
        lines.append(f"Table: {table_name}")
        for c in cols:
            qname = c["qualified_name"]
            field_name = qname.split("[", 1)[1][:-1] if "[" in qname else qname
            desc = f" — {c['description']}" if c.get("description") else ""
            lines.append(f"  - {field_name}{desc}")

    if relations:
        lines.append("\nRelations du modèle :")
        for r in relations:
            lines.append(f"  {r['qualified_name']}")

    return "\n".join(lines)[:6000]


def _build_dax_prompt(
    user_prompt: str, schema_summary: str, chat_history: list[dict] | None = None
) -> str:
    """Prompt LLM pour la génération DAX — ne contient que des noms de tables/colonnes/mesures."""
    history_block = format_history_for_prompt(chat_history or [])
    history_section = (
        f"Questions précédentes de cette conversation (résumées, pour résoudre les "
        f"références comme \"cette catégorie\"/\"ce mois\" — ignore cette section si la "
        f"demande actuelle est autonome) :\n{history_block}\n\n"
        if history_block
        else ""
    )
    return (
        f"Demande utilisateur : {user_prompt}\n\n"
        f"{history_section}"
        f"Modèle sémantique Power BI :\n{schema_summary}\n\n"
        "Génère des requêtes DAX qui répondent à la demande.\n"
        "Règles :\n"
        "  - Utilise uniquement les tables, colonnes et mesures listées ci-dessus\n"
        "  - Préfère une mesure existante à un recalcul manuel si elle correspond exactement\n"
        "  - Chaque requête commence par EVALUATE et retourne une table\n"
        "  - Agrégation par catégorie : EVALUATE SUMMARIZECOLUMNS('Table'[Colonne], \"Alias\", [Mesure])\n"
        "  - Évolution temporelle : EVALUATE SUMMARIZECOLUMNS('Calendrier'[Mois], \"Alias\", [Mesure]) "
        "trié par la colonne temporelle\n"
        "  - RÈGLE CRITIQUE ORDER BY : si vous triez par une colonne de tri dédiée différente de "
        "la colonne affichée (ex: un champ '*Sort'/'*Order' séparé du libellé, courant dans les "
        "tables calendrier — 'Année-Mois' affiché mais trié par 'AnneeMoisTri'), cette colonne de "
        "tri DOIT AUSSI figurer dans SUMMARIZECOLUMNS, sinon DAX échoue avec \"impossible de "
        "déterminer une valeur unique\". "
        "Incorrect : SUMMARIZECOLUMNS('Calendrier'[Mois], \"CA\", [Total Ventes]) ORDER BY "
        "'Calendrier'[MoisTri] — 'MoisTri' n'est pas dans la table résultat. "
        "Correct : SUMMARIZECOLUMNS('Calendrier'[Mois], 'Calendrier'[MoisTri], \"CA\", "
        "[Total Ventes]) ORDER BY 'Calendrier'[MoisTri]\n"
        "  - RÈGLE CRITIQUE FILTRE vs GROUPEMENT : si la demande mentionne une valeur EXPLICITE "
        "d'une dimension (ex: \"en 2022\", \"pour la France\", \"pour Bikes\"), c'est un FILTRE, "
        "pas un groupement — n'ajoutez PAS cette colonne dans SUMMARIZECOLUMNS, filtrez-la via "
        "CALCULATE sur la mesure. Sinon vous obtenez une ligne par combinaison (dimension "
        "demandée × valeur filtrée) au lieu d'une ligne par dimension demandée. "
        "Incorrect (\"ventes par catégorie en 2022\") : SUMMARIZECOLUMNS('Catégorie'[Nom], "
        "'Calendrier'[Année], \"CA\", [Total Ventes]) — regroupe par année au lieu de filtrer "
        "dessus. "
        "Correct : SUMMARIZECOLUMNS('Catégorie'[Nom], \"CA\", CALCULATE([Total Ventes], "
        "'Calendrier'[Année] = 2022))\n"
        "  - RÈGLE CRITIQUE MESURE vs COLONNE : n'utilisez [NomEntreCrochets] QUE pour un nom "
        "figurant EXACTEMENT dans la liste \"Mesures DAX disponibles\" ci-dessus. Toute autre "
        "valeur chiffrée est une colonne d'une table — agrégez-la explicitement : "
        "SUM('Table'[Colonne]), AVERAGE('Table'[Colonne]), etc. "
        "Incorrect : SUMMARIZECOLUMNS('Catégorie'[Nom], \"CA\", [Revenue]) si \"Revenue\" "
        "n'apparaît pas dans les mesures listées — DAX échoue avec \"impossible de "
        "déterminer la valeur\". "
        "Correct : SUMMARIZECOLUMNS('Catégorie'[Nom], \"CA\", SUM('Ventes'[Revenue]))\n"
        "  - KPI ou valeur unique : EVALUATE ROW(\"Alias\", [Mesure])\n"
        "  - Alias explicites en français entre guillemets\n\n"
        'Retourne : {"queries": [{"key": "nom_aggregat", "dax": "EVALUATE ..."}]}'
    )


def _strip_order_by(dax: str) -> str | None:
    """Retire la clause ORDER BY finale d'une requête DAX, ou None si absente.

    Repli léger avant le fallback COUNTROWS : un résultat groupé correct mais non trié
    reste bien plus exploitable (pour un insight/graphique) qu'un simple comptage de lignes.
    Couvre le cas fréquent où le LLM trie par une colonne absente de SUMMARIZECOLUMNS
    (ex: une colonne '*Sort' dédiée d'une table calendrier) — cf. _build_dax_prompt.
    """
    import re as _re

    match = _re.search(r"\border\s+by\b", dax, flags=_re.IGNORECASE)
    if not match:
        return None
    return dax[: match.start()].rstrip().rstrip(",")


def _compute_dax_fallback_queries(state: PipelineState) -> list[dict]:
    """Requêtes DAX basiques (COUNTROWS par table, valeur des mesures) si la génération LLM échoue."""
    queries: list[dict] = []
    files_meta = state.get("metadata", {}).get("files", {})
    for ref, meta in files_meta.items():
        table_name = _pbi_table_name(ref)
        if table_name == "__measures__":
            for measure_name in meta.get("columns", {}):
                queries.append(
                    {"key": f"{measure_name}_total", "dax": f'EVALUATE ROW("{measure_name}", [{measure_name}])'}
                )
            continue
        queries.append(
            {"key": f"{table_name}_count", "dax": f'EVALUATE ROW("lignes", COUNTROWS(\'{table_name}\'))'}
        )
    return queries[:5]


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
    """Agent 3 — Interprète le prompt et exécute les agrégations, via SQL/DuckDB (CSV) ou DAX (Power BI local).

    RÈGLE ABSOLUE : les données brutes ne passent JAMAIS dans le LLM.
    Le LLM ne reçoit que : schema_summary (noms + types) + noms des colonnes/mesures disponibles.

    Mode "csv" (défaut) :
        A. Chargement des DataFrames depuis storage.
        B. Construction du schema_summary (métadonnées uniquement).
        C. Génération des requêtes SQL via LLM.
        D. Exécution DuckDB avec timeout 30s.
        E. Sérialisation des résultats (max 500 lignes).
        F. Fallback basique si l'exécution échoue.
        Cache Redis : TTL 1h par (tenant, datasets, prompt).

    Mode "powerbi_local" :
        A. Génération DAX — settings.powerbi_dax_generation_mode :
           "llm" (défaut, prompt LiteLLM) ou "mcp_native" (tool DAX du serveur MCP,
           repli automatique sur "llm" si indisponible/échoue).
        B. Exécution via powerbi_local_mcp.execute_dax() (pas de DuckDB).
        C. Fallback COUNTROWS/mesures si la génération échoue.
        D. Traçabilité des requêtes dans state["dax_queries"].
        Pas de cache Redis (le modèle Power BI peut changer entre deux appels).

    Input  : state["prompt"] + state["schema"] + state["metadata"] (csv)
             state["prompt"] + state["metadata"] + state["semantic_model_info"] (powerbi_local)
    Output : state["aggregates"] (+ state["dax_queries"] en mode powerbi_local)
    Modèle : settings.litellm_cheap_model (gpt-4o-mini)
    """

    name = "data_agent"

    async def run(self, state: PipelineState) -> PipelineState:
        log = logger.bind(report_id=state.get("report_id"))

        if state.get("data_source") == "powerbi_local":
            return await self._run_powerbi(state, log)

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
        sql_prompt = _build_sql_prompt(state["prompt"], schema_summary, dfs, state.get("chat_history"))
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

    async def _run_powerbi(self, state: PipelineState, log) -> PipelineState:
        """Branche Power BI local : génère et exécute des requêtes DAX via MCP (pas de DuckDB)."""
        client = get_powerbi_client()

        # Le client MCP est un singleton process-wide (get_powerbi_client()) et ne conserve sa
        # connexion que tant que le process backend tourne. Normalement, MetadataAgent l'a déjà
        # établie ; mais quand metadata est réutilisé via base_report_id (_route_after_intent
        # saute alors MetadataAgent), rien d'autre n'appelle connect_to_desktop_file() — un
        # redémarrage backend entre deux messages de la même session ferait alors échouer
        # execute_dax() avec "no connectionName provided". Reconnecter ici est idempotent côté
        # MCP et sans coût si déjà connecté au bon fichier.
        pbix_file_name = state.get("pbix_file_name", "")
        if pbix_file_name and client._connected_file != pbix_file_name:
            await client.connect_to_desktop_file(pbix_file_name)

        schema_summary = await self._build_schema_summary_with_rag(state, log)

        queries: list[dict] = []
        if settings.powerbi_dax_generation_mode == "mcp_native":
            queries = await self._generate_dax_native(client, state["prompt"], log)

        if not queries:
            dax_prompt = _build_dax_prompt(state["prompt"], schema_summary, state.get("chat_history"))
            llm_result = await call_llm_json(
                prompt=dax_prompt,
                system=_DAX_SYSTEM_PROMPT,
                model=settings.litellm_cheap_model,
            )
            queries = llm_result.get("queries", [])
            if not isinstance(queries, list):
                queries = []

        log.info("data_agent_dax_generated", n_queries=len(queries))

        aggregates: dict = {}
        dax_log: list[dict] = []
        exec_success = False

        for q in queries:
            key = q.get("key") or "query_result"
            dax = q.get("dax", "").strip()
            if not dax:
                continue
            try:
                result = await client.execute_dax(dax)
                aggregates[key] = result.get("rows", [])[:_MAX_ROWS]
                dax_log.append({"key": key, "dax": dax, "status": "ok"})
                exec_success = True
                log.info("data_agent_dax_ok", key=key)
                continue
            except PowerBIDaxExecutionError as exc:
                msg = f"DataAgent: DAX error ({key}): {exc}"
                state["errors"] = state.get("errors", []) + [msg]
                dax_log.append({"key": key, "dax": dax, "status": "error", "error": str(exc)})
                log.warning("data_agent_dax_error", key=key, error=str(exc))

            # Repli léger : retirer un ORDER BY qui référence une colonne hors résultat
            # (cause d'échec la plus fréquente) avant d'abandonner cette requête — un résultat
            # groupé non trié reste bien plus utile qu'un COUNTROWS générique.
            dax_no_order = _strip_order_by(dax)
            if dax_no_order and dax_no_order != dax:
                try:
                    result = await client.execute_dax(dax_no_order)
                    aggregates[key] = result.get("rows", [])[:_MAX_ROWS]
                    dax_log.append({"key": key, "dax": dax_no_order, "status": "ok_retry_no_order_by"})
                    exec_success = True
                    log.info("data_agent_dax_retry_no_order_by_ok", key=key)
                except PowerBIDaxExecutionError as exc:
                    dax_log.append(
                        {"key": key, "dax": dax_no_order, "status": "retry_no_order_by_error", "error": str(exc)}
                    )
                    log.warning("data_agent_dax_retry_no_order_by_failed", key=key, error=str(exc))

        # ── Fallback : COUNTROWS/mesures directes si la génération/exécution échoue ──
        if not exec_success or not aggregates:
            fallback_queries = _compute_dax_fallback_queries(state)
            for q in fallback_queries:
                try:
                    result = await client.execute_dax(q["dax"])
                    aggregates[q["key"]] = result.get("rows", [])[:_MAX_ROWS]
                    dax_log.append({"key": q["key"], "dax": q["dax"], "status": "fallback_ok"})
                except PowerBIDaxExecutionError as exc:
                    dax_log.append(
                        {"key": q["key"], "dax": q["dax"], "status": "fallback_error", "error": str(exc)}
                    )
            if aggregates:
                warn = "DataAgent: fallback DAX utilisé"
                state["errors"] = state.get("errors", []) + [warn]
                log.warning("data_agent_dax_fallback_used")

        state["aggregates"] = aggregates
        state["dax_queries"] = state.get("dax_queries", []) + dax_log
        return state

    async def _build_schema_summary_with_rag(self, state: PipelineState, log) -> str:
        """Construit le schema_summary via RAG (sous-ensemble pertinent de champs), avec
        repli explicite sur le schéma complet si le retrieval échoue ou est vide.

        Ne remplace que la SOURCE du schéma injecté dans le prompt DAX — la logique de
        génération DAX elle-même (_build_dax_prompt) est inchangée.
        """
        tenant_id = state.get("tenant_id", "")
        model_id = state.get("pbix_file_name", "")

        try:
            fields = await retrieve_relevant_fields(tenant_id, model_id, state["prompt"])
        except Exception as exc:
            log.warning("data_agent_schema_rag_failed_fallback_full", error=str(exc))
            return _build_dax_schema_summary(state)

        if not fields:
            log.warning("data_agent_schema_rag_empty_fallback_full")
            return _build_dax_schema_summary(state)

        log.info("data_agent_schema_rag_used", n_fields=len(fields))
        return _build_dax_schema_summary_from_rag(fields)

    async def _generate_dax_native(self, client, prompt: str, log) -> list[dict]:
        """Tente d'utiliser un tool MCP de génération DAX depuis du langage naturel.

        Confirmé contre un vrai serveur (Power BI Modeling MCP v0.5.0-beta.11, via
        list_tools()) : dax_query_operations ne supporte que "Help, Execute, Validate,
        ClearCache" — il n'existe PAS d'opération de génération DAX depuis du langage
        naturel dans cette version du serveur. Cet appel échouera donc systématiquement
        aujourd'hui ; il est conservé pour basculer automatiquement dessus si Microsoft
        ajoute cette capacité plus tard, sans changement de code applicatif.
        Chaque retombée sur le mode "llm" est loggée en warning explicite (jamais silencieuse).
        """
        try:
            result = await client.call_tool(
                "dax_query_operations",
                {"operation": "Generate", "naturalLanguageQuery": prompt},
            )
        except Exception as exc:
            log.warning("data_agent_dax_native_unavailable_fallback_llm", error=str(exc))
            return []

        if isinstance(result, dict):
            queries = result.get("queries") or result.get("dax")
            if isinstance(queries, list) and queries:
                return queries
            if isinstance(queries, str) and queries:
                return [{"key": "result", "dax": queries}]

        log.warning(
            "data_agent_dax_native_unexpected_response_fallback_llm",
            result_type=type(result).__name__,
        )
        return []
