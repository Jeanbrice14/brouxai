from __future__ import annotations

import structlog

from app.agents.base_agent import BaseAgent
from app.config import settings
from app.pipeline.state import PipelineState
from app.services.llm import call_llm_json

logger = structlog.get_logger(__name__)

# ── Règles de sélection de graphique ─────────────────────────────────────────

_CHART_RULES = """
Règles de sélection du type de graphique :
- Comparaison entre catégories → "bar"
- Évolution temporelle (colonne date détectée) → "line"
- Distribution d'une variable continue → "histogram"
- Proportion entre catégories (max 5 catégories) → "pie"
- Corrélation entre deux variables numériques → "scatter"
- Données géographiques (région, pays, ville) → "bar" (map reporté v1+)
"""

_VALID_CHART_TYPES = {"bar", "line", "pie", "scatter", "histogram"}

# Couleurs par défaut
_DEFAULT_COLORS = {
    "primary": "#1E3A8A",
    "positive": "#16A34A",
    "negative": "#DC2626",
}

_MAX_VIZ_SPECS = 5

_VIZ_SYSTEM_PROMPT = (
    "Tu es un expert en data visualisation. "
    "Tu sélectionnes le type de graphique optimal pour chaque insight analytique. "
    "Tu retournes UNIQUEMENT du JSON valide correspondant exactement au format demandé. "
    "Tu utilises UNIQUEMENT les clés et colonnes fournies — tu n'inventes jamais de données."
)


def _extract_agg_keys(aggregates: dict) -> dict[str, list[str]]:
    """Retourne {data_key: [colonnes]} pour chaque clé d'agrégat."""
    result: dict[str, list[str]] = {}
    for key, rows in aggregates.items():
        if isinstance(rows, list) and rows and isinstance(rows[0], dict):
            result[key] = list(rows[0].keys())
    return result


def _build_viz_prompt(
    insight: dict,
    agg_keys: dict[str, list[str]],
    colors: dict,
) -> str:
    agg_summary = "\n".join(f"  - {key}: colonnes {cols}" for key, cols in agg_keys.items())
    return (
        f"Insight à visualiser :\n"
        f"  title: {insight.get('title', '')}\n"
        f"  description: {insight.get('description', '')}\n"
        f"  type: {insight.get('type', '')}\n"
        f"  supporting_data: {insight.get('supporting_data', '')}\n\n"
        f"Agrégats disponibles (data_key → colonnes) :\n{agg_summary}\n\n"
        f"Couleurs du brand_kit :\n"
        f"  primary: {colors['primary']}\n"
        f"  positive: {colors['positive']}\n"
        f"  negative: {colors['negative']}\n\n"
        f"{_CHART_RULES}\n"
        "Génère la viz_spec JSON avec ce format EXACT :\n"
        "{\n"
        '  "chart_type": "bar|line|pie|scatter|histogram",\n'
        '  "title": "titre lisible du graphique",\n'
        '  "data_key": "clé exacte dans les agrégats ci-dessus",\n'
        '  "x": "nom de la colonne pour axe X",\n'
        '  "y": "nom de la colonne pour axe Y",\n'
        '  "color_by": null,\n'
        '  "colors": {"primary": "...", "positive": "...", "negative": "..."},\n'
        '  "annotations": [],\n'
        f'  "insight_ref": "{insight.get("title", "")}"\n'
        "}"
    )


def _validate_spec(spec: dict, aggregates: dict) -> tuple[bool, str]:
    """Retourne (valid, raison_si_invalide)."""
    data_key = spec.get("data_key")
    if not data_key or data_key not in aggregates:
        return False, f"data_key '{data_key}' absent dans aggregates"

    rows = aggregates[data_key]
    if not isinstance(rows, list) or not rows:
        return False, f"aggregates['{data_key}'] est vide"

    cols = set(rows[0].keys()) if isinstance(rows[0], dict) else set()
    x_col = spec.get("x")
    y_col = spec.get("y")
    if x_col and x_col not in cols:
        return False, f"colonne x='{x_col}' absente dans {data_key} (colonnes: {cols})"
    if y_col and y_col not in cols:
        return False, f"colonne y='{y_col}' absente dans {data_key} (colonnes: {cols})"

    return True, ""


def _apply_colors(spec: dict, colors: dict) -> dict:
    """Injecte les couleurs du brand_kit dans la viz_spec."""
    spec["colors"] = {
        "primary": colors.get("primary", _DEFAULT_COLORS["primary"]),
        "positive": colors.get("positive", _DEFAULT_COLORS["positive"]),
        "negative": colors.get("negative", _DEFAULT_COLORS["negative"]),
    }
    return spec


class VizAgent(BaseAgent):
    """Agent 6 — Sélectionne le type de graphique et génère les viz_spec JSON.

    Étapes :
        A. Règles de sélection de graphique définies en constantes.
        B. Pour chaque insight : appel LLM pour générer la viz_spec.
        C. Validation : data_key + colonnes x/y doivent exister dans aggregates.
        D. Application des couleurs brand_kit (défaut si absent).
        E. Écriture dans state["viz_specs"] (max 5).

    Input  : state["insights"] + state["aggregates"] + state["brand_kit"]
    Output : state["viz_specs"]
    Modèle : settings.litellm_cheap_model (gpt-4o-mini)
    V0     : viz_spec JSON pour Recharts uniquement — export PNG/PDF reporté v1
    """

    name = "viz_agent"

    async def run(self, state: PipelineState) -> PipelineState:
        log = logger.bind(report_id=state.get("report_id"))

        insights = state.get("insights", [])
        aggregates = state.get("aggregates", {})
        brand_kit = state.get("brand_kit", {})

        # ── Étape D : couleurs ───────────────────────────────────────────────
        bk_colors = brand_kit.get("colors", {})
        colors = {
            "primary": bk_colors.get("primary", _DEFAULT_COLORS["primary"]),
            "positive": bk_colors.get("positive", _DEFAULT_COLORS["positive"]),
            "negative": bk_colors.get("negative", _DEFAULT_COLORS["negative"]),
        }

        # ── Étape B : génération des viz_specs ───────────────────────────────
        agg_keys = _extract_agg_keys(aggregates)
        viz_specs: list[dict] = []

        for insight in insights:
            if len(viz_specs) >= _MAX_VIZ_SPECS:
                break

            prompt = _build_viz_prompt(insight, agg_keys, colors)
            try:
                spec = await call_llm_json(
                    prompt=prompt,
                    system=_VIZ_SYSTEM_PROMPT,
                    model=settings.litellm_cheap_model,
                )
            except Exception as exc:
                log.warning("viz_llm_error", insight=insight.get("title"), error=str(exc))
                continue

            # Normaliser chart_type
            if spec.get("chart_type") not in _VALID_CHART_TYPES:
                spec["chart_type"] = "bar"

            # ── Étape C : validation ─────────────────────────────────────────
            valid, reason = _validate_spec(spec, aggregates)
            if not valid:
                log.warning("viz_spec_invalid", reason=reason, insight=insight.get("title"))
                state["errors"] = state.get("errors", []) + [
                    f"viz_agent: viz_spec ignorée — {reason}"
                ]
                continue

            # ── Étape D : appliquer couleurs ─────────────────────────────────
            spec = _apply_colors(spec, colors)
            viz_specs.append(spec)

        log.info("viz_specs_generated", count=len(viz_specs))
        state["viz_specs"] = viz_specs
        return state
