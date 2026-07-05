from __future__ import annotations

import math

import pandas as pd
import structlog

from app.agents.base_agent import BaseAgent
from app.config import settings
from app.pipeline.state import PipelineState
from app.services.llm import call_llm_json
from app.services.powerbi_local_mcp import get_powerbi_client
from app.services.schema_rag import index_schema
from app.services.storage import read_dataframe

logger = structlog.get_logger(__name__)

# Confiance appliquée aux colonnes/mesures Power BI (mode powerbi_local)
_PBI_CONFIDENCE_WITH_DESCRIPTION = 0.95
_PBI_CONFIDENCE_WITHOUT_DESCRIPTION = 0.60

# Nombre de valeurs d'échantillon envoyées au LLM
_SAMPLE_SIZE = 5

_COLUMN_SYSTEM_PROMPT = (
    "Tu es un expert en data engineering. "
    "Analyse le profil d'une colonne et retourne UNIQUEMENT un objet JSON valide."
)

_GRAIN_SYSTEM_PROMPT = (
    "Tu es un expert en data engineering. "
    "Retourne UNIQUEMENT un objet JSON valide avec une seule clé 'grain'."
)


def _build_column_profile(series: pd.Series) -> dict:
    """Calcule le profil statistique d'une Series pandas.

    Ne transmet au LLM que des agrégats — jamais les données brutes.
    """
    n = len(series)
    null_count = int(series.isna().sum())
    null_pct = round(null_count / n, 4) if n > 0 else 0.0
    n_unique = int(series.nunique(dropna=True))

    # Échantillon de valeurs non-nulles (dédoublonnées)
    sample_vals = series.dropna().unique().tolist()[:_SAMPLE_SIZE]
    # Sérialiser les types non-JSON-natifs (Timestamp, numpy, ...)
    sample = [_safe_scalar(v) for v in sample_vals]

    profile: dict = {
        "dtype": str(series.dtype),
        "n_unique": n_unique,
        "null_pct": null_pct,
        "sample": sample,
        "min": None,
        "max": None,
        "mean": None,
    }

    # Statistiques numériques uniquement
    if pd.api.types.is_numeric_dtype(series):
        numeric = series.dropna()
        if len(numeric) > 0:
            profile["min"] = _safe_scalar(numeric.min())
            profile["max"] = _safe_scalar(numeric.max())
            profile["mean"] = round(float(numeric.mean()), 4)

    return profile


def _safe_scalar(v) -> float | int | str | bool | None:
    """Convertit les types numpy/pandas en scalaires Python natifs JSON-sérialisables."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return None
    if hasattr(v, "item"):  # numpy scalar
        return v.item()
    if hasattr(v, "isoformat"):  # datetime / Timestamp
        return v.isoformat()
    return v


def _build_column_prompt(col_name: str, profile: dict) -> str:
    lines = [
        "Analyse cette colonne de données :",
        f'- Nom technique : "{col_name}"',
        f"- Type Python   : {profile['dtype']}",
        f"- Valeurs uniques : {profile['n_unique']}",
        f"- % nulles : {profile['null_pct']:.1%}",
    ]
    if profile["min"] is not None:
        lines.append(
            f"- Min : {profile['min']},  Max : {profile['max']},  Moyenne : {profile['mean']}"
        )
    lines.append(f"- Échantillon de valeurs : {profile['sample']}")
    lines.append("")
    lines.append(
        "Retourne un JSON avec exactement ces champs :\n"
        "{\n"
        '  "semantic_name": "nom lisible en français",\n'
        '  "description": "description courte du contenu",\n'
        '  "type": "categorical|numeric|date|text|identifier",\n'
        '  "unit": "unité ou chaîne vide si sans unité",\n'
        '  "confidence": 0.0 à 1.0,\n'
        '  "is_key_candidate": true|false\n'
        "}"
    )
    return "\n".join(lines)


def _build_grain_prompt(col_summaries: list[str]) -> str:
    cols_str = "\n".join(f"  - {s}" for s in col_summaries)
    return (
        f"Voici les colonnes d'un fichier de données :\n{cols_str}\n\n"
        "En une phrase concise, qu'est-ce que représente UNE ligne dans ce fichier ?\n\n"
        'Retourne : {"grain": "..."}'
    )


_UNIT_SYMBOLS = ["€", "$", "£", "¥", "%"]


def _extract_unit_from_format_string(format_string: str) -> str:
    """Dérive une unité d'affichage courte depuis un formatString Power BI.

    formatString est un code de formatage Excel/PBI (ex: currency avec parenthèses pour les
    négatifs : "\\$#,0.###############;(\\$#,0.###############);\\$#,0.###############") —
    pas une unité au sens métier. Afficher ce code brut dans la colonne "Unité" du CP1 induit
    en erreur. On se limite à détecter un symbole simple et reconnaissable (devise, %) ;
    sinon on retourne une chaîne vide plutôt que le code de formatage brut.
    """
    if not format_string:
        return ""
    for symbol in _UNIT_SYMBOLS:
        if symbol in format_string:
            return symbol
    return ""


def _pbi_column_meta(col_name: str, col_info: dict, kind: str) -> dict:
    """Transforme une colonne/mesure Power BI en entrée du format `metadata.files.*.columns`.

    Confiance : haute si une description est renseignée dans le modèle, réduite sinon
    (pas d'inférence LLM ici — les métadonnées viennent déjà du modèle sémantique).
    """
    description = str(col_info.get("description") or "").strip()
    confidence = (
        _PBI_CONFIDENCE_WITH_DESCRIPTION if description else _PBI_CONFIDENCE_WITHOUT_DESCRIPTION
    )
    # Type réel (Int64, String, DateTime, Double...) pour colonnes ET mesures — "kind"
    # (déjà présent séparément) distingue mesure/colonne, "type" ne doit jamais perdre le
    # type de donnée réel derrière le mot générique "measure".
    data_type = col_info.get("dataType", col_info.get("type", "unknown"))
    return {
        "dtype": data_type,
        "n_unique": None,
        "null_pct": None,
        "sample": [],
        "min": None,
        "max": None,
        "mean": None,
        "semantic_name": col_info.get("displayName", col_name),
        "description": description,
        "type": data_type,
        "unit": _extract_unit_from_format_string(col_info.get("formatString", "") or ""),
        "confidence": confidence,
        "is_key_candidate": bool(col_info.get("isKey", False)),
        "kind": kind,  # "column" | "measure"
    }


def _transform_pbi_metadata(model_info: dict) -> tuple[dict, bool]:
    """Transforme le résultat de get_model_metadata() en state['metadata']['files'].

    Une "table" virtuelle par table Power BI (ref = "powerbi://<table>"), colonnes
    ET mesures fusionnées dans le même dict `columns` (les mesures portent kind="measure").
    Les mesures sans table d'appartenance connue sont regroupées dans une table
    virtuelle "__measures__".

    Returns:
        (files_meta, trigger_hitl)
    """
    files_meta: dict = {}
    trigger_hitl = False

    tables: dict = model_info.get("tables", {}) or {}
    measures: dict = model_info.get("measures", {}) or {}

    for table_name, table_info in tables.items():
        ref = f"powerbi://{table_name}"
        columns_meta: dict = {}
        for col_name, col_info in (table_info.get("columns", {}) or {}).items():
            entry = _pbi_column_meta(col_name, col_info if isinstance(col_info, dict) else {}, "column")
            columns_meta[col_name] = entry
            if entry["confidence"] < settings.hitl_metadata_confidence_threshold:
                trigger_hitl = True

        confidences = [c["confidence"] for c in columns_meta.values()]
        avg_confidence = round(sum(confidences) / len(confidences), 4) if confidences else 0.0

        files_meta[ref] = {
            "row_count": None,  # inconnu sans exécuter de DAX — pas de lecture brute ici
            "col_count": len(columns_meta),
            "columns": columns_meta,
            "grain": table_info.get("description", "") if isinstance(table_info, dict) else "",
            "avg_confidence": avg_confidence,
        }

    if measures:
        measures_ref = "powerbi://__measures__"
        columns_meta = {}
        for measure_name, measure_info in measures.items():
            entry = _pbi_column_meta(
                measure_name, measure_info if isinstance(measure_info, dict) else {}, "measure"
            )
            columns_meta[measure_name] = entry
            if entry["confidence"] < settings.hitl_metadata_confidence_threshold:
                trigger_hitl = True

        confidences = [c["confidence"] for c in columns_meta.values()]
        avg_confidence = round(sum(confidences) / len(confidences), 4) if confidences else 0.0
        files_meta[measures_ref] = {
            "row_count": None,
            "col_count": len(columns_meta),
            "columns": columns_meta,
            "grain": "Mesures DAX du modèle sémantique",
            "avg_confidence": avg_confidence,
        }

    return files_meta, trigger_hitl


class MetadataAgent(BaseAgent):
    """Agent 1 — Construit le Data Dictionary, depuis un upload CSV/Excel ou un modèle Power BI local.

    Mode "csv" (défaut) :
        A. Profil statistique de chaque colonne (sans données brutes).
        B. Inférence sémantique via LLM (semantic_name, type, unit, confidence…).
        C. Calcul avg_confidence par fichier.
        D. Inférence du grain du fichier via LLM.
        E. Déclenchement HITL si confidence < seuil.

    Mode "powerbi_local" :
        A. Connexion à l'instance Analysis Services locale via powerbi_local_mcp
           (state["pbix_file_name"], recherche auto par le serveur MCP).
        B. get_model_metadata() → tables/colonnes/mesures/relations, mis en cache
           dans state["semantic_model_info"].
        C. Transformation vers le format state["metadata"]["files"] attendu par
           SchemaLinkingAgent/DataAgent — confiance déduite de la présence d'une
           description (pas d'inférence LLM : les métadonnées viennent déjà du modèle).
        D. Déclenchement HITL si confidence < seuil (colonnes/mesures sans description).

    Input  : state["raw_data_refs"] (csv) | state["pbix_file_name"] (powerbi_local)
    Output : state["metadata"] (+ state["semantic_model_info"] en mode powerbi_local)
    Modèle : settings.litellm_cheap_model (gpt-4o-mini) — mode csv uniquement
    HITL CP1 : confidence colonne/mesure < settings.hitl_metadata_confidence_threshold (0.85)
    """

    name = "metadata_agent"

    async def run(self, state: PipelineState) -> PipelineState:
        log = logger.bind(report_id=state.get("report_id"))

        if state.get("data_source") == "powerbi_local":
            return await self._run_powerbi(state, log)

        files_meta: dict = {}
        trigger_hitl = False

        for ref in state["raw_data_refs"]:
            log.info("metadata_processing_file", ref=ref)
            df = await read_dataframe(ref)
            file_meta = await self._process_file(df, ref, log)
            files_meta[ref] = file_meta

            # Vérifier si le HITL doit être déclenché pour ce fichier
            for col_info in file_meta["columns"].values():
                if col_info["confidence"] < settings.hitl_metadata_confidence_threshold:
                    trigger_hitl = True
                    log.warning(
                        "metadata_low_confidence",
                        ref=ref,
                        col=col_info.get("semantic_name"),
                        confidence=col_info["confidence"],
                    )

        state["metadata"] = {"files": files_meta}

        if trigger_hitl:
            state["hitl_pending"] = True
            state["hitl_checkpoint"] = "cp1_metadata"
            log.info("metadata_hitl_triggered")

        return state

    async def _run_powerbi(self, state: PipelineState, log) -> PipelineState:
        """Branche Power BI local : connexion MCP + get_model_metadata() (pas de LLM)."""
        file_name = state.get("pbix_file_name", "")
        if not file_name:
            raise ValueError(
                "pbix_file_name manquant — requis en mode data_source='powerbi_local'."
            )

        # PowerBIConnectionError (message déjà actionnable, cf. powerbi_local_mcp) remonte
        # telle quelle jusqu'à BaseAgent.__call__, qui l'enregistre dans state["errors"].
        client = get_powerbi_client()
        await client.connect_to_desktop_file(file_name)
        model_info = await client.get_model_metadata()

        state["semantic_model_info"] = model_info

        # Indexation RAG du schéma (une fois par tenant+modèle, pas par requête — skip
        # automatique si le hash structurel est inchangé). N'échoue jamais bruyamment :
        # index_schema() catch ses propres erreurs, DataAgent retombe sur le schéma complet
        # si le RAG est indisponible.
        rag_result = await index_schema(state["tenant_id"], file_name, model_info)
        log.info("metadata_schema_rag", **rag_result)

        files_meta, trigger_hitl = _transform_pbi_metadata(model_info)
        state["metadata"] = {"files": files_meta, "source": "powerbi_local"}

        # raw_data_refs reste vide : c'est la valeur sémantiquement correcte en mode
        # powerbi_local, pas un contournement. Ce champ documente des pointeurs Blob
        # Storage (cf. state.py) — il n'en existe aucun ici, la source de données étant
        # une connexion live au modèle Power BI. SchemaLinkingAgent n'est de toute façon
        # plus invoqué dans ce mode (graph.py::_route_after_metadata route directement
        # vers DataAgent), donc le risque historique de crash sur read_dataframe() avec
        # des refs "powerbi://..." ne s'applique plus — ce n'est donc plus la raison de
        # laisser cette liste vide. Les relations du modèle (déjà connues, pas inférées)
        # sont dans semantic_model_info et consommées directement par DataAgent.

        if trigger_hitl:
            state["hitl_pending"] = True
            state["hitl_checkpoint"] = "cp1_metadata"
            log.info("metadata_hitl_triggered", reason="powerbi_missing_descriptions")

        log.info(
            "metadata_powerbi_processed",
            file_name=file_name,
            n_tables=len(files_meta),
        )
        return state

    async def _process_file(self, df: pd.DataFrame, ref: str, log) -> dict:
        """Traite un fichier et retourne ses métadonnées complètes."""
        columns_meta: dict = {}

        for col_name in df.columns:
            profile = _build_column_profile(df[col_name])
            prompt = _build_column_prompt(col_name, profile)

            llm_result = await call_llm_json(
                prompt=prompt,
                system=_COLUMN_SYSTEM_PROMPT,
                model=settings.litellm_cheap_model,
            )

            # Fusionner profil statistique + inférence LLM
            columns_meta[col_name] = {
                **profile,
                "semantic_name": llm_result.get("semantic_name", col_name),
                "description": llm_result.get("description", ""),
                "type": llm_result.get("type", "unknown"),
                "unit": llm_result.get("unit", ""),
                "confidence": float(llm_result.get("confidence", 0.0)),
                "is_key_candidate": bool(llm_result.get("is_key_candidate", False)),
            }

        # Calcul de la confiance moyenne du fichier
        confidences = [c["confidence"] for c in columns_meta.values()]
        avg_confidence = round(sum(confidences) / len(confidences), 4) if confidences else 0.0

        # Inférence du grain
        col_summaries = [
            f"{col}: {info['semantic_name']} ({info['type']})" for col, info in columns_meta.items()
        ]
        grain_result = await call_llm_json(
            prompt=_build_grain_prompt(col_summaries),
            system=_GRAIN_SYSTEM_PROMPT,
            model=settings.litellm_cheap_model,
        )
        grain = grain_result.get("grain", "")

        log.info(
            "metadata_file_processed",
            ref=ref,
            rows=len(df),
            cols=len(df.columns),
            avg_confidence=avg_confidence,
            grain=grain,
        )

        return {
            "row_count": len(df),
            "col_count": len(df.columns),
            "columns": columns_meta,
            "grain": grain,
            "avg_confidence": avg_confidence,
        }
