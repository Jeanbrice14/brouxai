from __future__ import annotations

import asyncio
from pathlib import Path
from urllib.parse import urlparse

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

_CODE_SYSTEM_PROMPT = (
    "Tu es un expert Python/pandas. "
    "Génère du code pandas court et correct. "
    "Retourne UNIQUEMENT du JSON valide avec une clé 'code'."
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

    for table_name, df in dfs.items():
        lines.append(f"Table: {table_name} ({len(df)} lignes, {len(df.columns)} colonnes)")

        # Retrouver les métadonnées de ce fichier par correspondance de nom
        meta_info: dict = {}
        for ref, meta in files_meta.items():
            if _table_name(ref) == table_name:
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
                f"  {rel['table_a']}.{rel['col_a']} → {rel['table_b']}.{rel['col_b']}: {desc}"
            )

    summary = "\n".join(lines)
    return summary[:2000]


def _build_code_prompt(user_prompt: str, schema_summary: str, dfs: dict) -> str:
    """Prompt LLM pour la génération de code — ne contient que des noms de colonnes, pas de valeurs."""
    # Uniquement les noms de colonnes — JAMAIS les données
    df_list = "\n".join(
        f"  - dfs['{name}']: colonnes = {list(df.columns)}" for name, df in dfs.items()
    )
    return (
        f"Demande utilisateur : {user_prompt}\n\n"
        f"Schéma :\n{schema_summary}\n\n"
        f"DataFrames disponibles (variable `dfs`, dict nom→DataFrame) :\n{df_list}\n\n"
        "Génère du code pandas qui répond à la demande.\n"
        "Règles :\n"
        "  - `pd` (pandas) et `dfs` sont dans le namespace\n"
        "  - Assigne le résultat à `result` (dict de DataFrames)\n"
        "  - Uniquement des agrégats — jamais les données brutes\n\n"
        'Retourne : {"code": "...code Python..."}'
    )


def _run_code_sync(code: str, dfs: dict[str, pd.DataFrame]) -> dict:
    """Exécute le code pandas dans un namespace contrôlé (synchrone, appelé via to_thread)."""
    namespace: dict = {"pd": pd, "dfs": dfs, "result": {}}
    exec(code, namespace)  # noqa: S102
    return namespace.get("result", {})


def _serialize_result(raw_result: dict) -> dict:
    """Convertit les DataFrames/Series en list-of-dicts, 500 lignes max par agrégat."""
    aggregates: dict = {}
    for key, value in raw_result.items():
        if isinstance(value, pd.DataFrame):
            rows = value.head(_MAX_ROWS).to_dict(orient="records")
        elif isinstance(value, pd.Series):
            rows = value.head(_MAX_ROWS).reset_index().to_dict(orient="records")
        else:
            rows = [{"value": value}]
        aggregates[key] = _clean_records(rows)
    return aggregates


def _clean_records(records: list[dict]) -> list[dict]:
    """Normalise les types numpy/NaN pour la sérialisation JSON."""
    cleaned = []
    for row in records:
        clean_row = {}
        for k, v in row.items():
            if hasattr(v, "item"):
                clean_row[k] = v.item()
            elif isinstance(v, float) and pd.isna(v):
                clean_row[k] = None
            else:
                clean_row[k] = v
        cleaned.append(clean_row)
    return cleaned


def _compute_fallback(dfs: dict[str, pd.DataFrame]) -> dict:
    """Génère des agrégats basiques si le code LLM échoue.

    Calcule : sommes, moyennes et count sur colonnes numériques.
    Ne plante jamais — même sur DataFrames vides.
    """
    aggregates: dict = {}
    for name, df in dfs.items():
        if df.empty:
            aggregates[f"{name}_count"] = [{"count": 0}]
            continue

        numeric_cols = df.select_dtypes(include="number").columns.tolist()

        if numeric_cols:
            aggregates[f"{name}_sums"] = [{col: _safe_float(df[col].sum()) for col in numeric_cols}]
            aggregates[f"{name}_means"] = [
                {col: round(_safe_float(df[col].mean()), 4) for col in numeric_cols}
            ]

        aggregates[f"{name}_count"] = [{"count": len(df)}]

    return aggregates


def _safe_float(v) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return 0.0


# ── Agent ─────────────────────────────────────────────────────────────────────


class DataAgent(BaseAgent):
    """Agent 3 — Interprète le prompt, génère du code pandas, exécute les agrégations.

    RÈGLE ABSOLUE : les données brutes ne passent JAMAIS dans le LLM.
    Le LLM ne reçoit que : schema_summary (noms + types) + noms des colonnes disponibles.

    Étapes :
        A. Chargement des DataFrames depuis storage.
        B. Construction du schema_summary (métadonnées uniquement).
        C. Génération du code pandas via LLM.
        D. Exécution sandbox avec timeout 30s.
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
            name = _table_name(ref)
            dfs[name] = await read_dataframe(ref)
            log.info("data_agent_loaded_df", table=name, rows=len(dfs[name]))

        # ── Étape B : schema_summary (aucune donnée brute) ───────────────────
        schema_summary = _build_schema_summary(state, dfs)

        # ── Étape C : génération du code via LLM ─────────────────────────────
        code_prompt = _build_code_prompt(state["prompt"], schema_summary, dfs)
        llm_result = await call_llm_json(
            prompt=code_prompt,
            system=_CODE_SYSTEM_PROMPT,
            model=settings.litellm_cheap_model,
        )
        code = llm_result.get("code", "")
        log.info("data_agent_code_generated", code_len=len(code))

        # ── Étapes D + E : exécution + sérialisation ─────────────────────────
        aggregates: dict = {}
        exec_success = False

        if code:
            try:
                raw_result = await asyncio.wait_for(
                    asyncio.to_thread(_run_code_sync, code, dfs),
                    timeout=_EXEC_TIMEOUT,
                )
                aggregates = _serialize_result(raw_result)
                exec_success = True
                log.info("data_agent_exec_ok", n_keys=len(aggregates))
            except TimeoutError:
                msg = f"DataAgent: timeout exec ({_EXEC_TIMEOUT}s)"
                state["errors"] = state.get("errors", []) + [msg]
                log.warning("data_agent_exec_timeout")
            except Exception as exc:
                msg = f"DataAgent: exec error: {type(exc).__name__}: {exc}"
                state["errors"] = state.get("errors", []) + [msg]
                log.warning("data_agent_exec_error", error=str(exc))

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
