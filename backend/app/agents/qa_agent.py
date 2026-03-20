from __future__ import annotations

import re

import structlog

from app.agents.base_agent import BaseAgent
from app.config import settings
from app.pipeline.state import PipelineState
from app.services.llm import call_llm_json

logger = structlog.get_logger(__name__)

_QA_SYSTEM_PROMPT = (
    "Tu es un expert en quality assurance de rapports analytiques. "
    "Tu détectes les hallucinations LLM, incohérences et affirmations non supportées. "
    "Tu retournes UNIQUEMENT du JSON valide. "
    "Sois conservateur : ne signale que les problèmes évidents et clairement identifiables."
)

# Pénalités par sévérité
_PENALTY = {"high": 0.25, "medium": 0.10, "low": 0.05}

# Seuil de surcharge cognitive (nombre de chiffres dans le narrative)
_MAX_NUMBERS_IN_NARRATIVE = 30


# ── Checks déterministes ──────────────────────────────────────────────────────


def _check_insights_have_aggregates(insights: list[dict], aggregates: dict) -> list[dict]:
    """Chaque insight doit référencer des données dans aggregates."""
    issues: list[dict] = []
    if not aggregates:
        if insights:
            issues.append(
                {
                    "type": "inconsistency",
                    "passage": "Insights présents mais aggregates vides.",
                    "severity": "high",
                    "check": "check_insights_have_aggregates",
                }
            )
    return issues


def _check_viz_specs_valid(viz_specs: list[dict], aggregates: dict) -> list[dict]:
    """Chaque viz_spec doit avoir data_key valide dans aggregates."""
    issues: list[dict] = []
    for spec in viz_specs:
        data_key = spec.get("data_key")
        if data_key and data_key not in aggregates:
            issues.append(
                {
                    "type": "inconsistency",
                    "passage": f"viz_spec data_key='{data_key}' absent dans aggregates.",
                    "severity": "medium",
                    "check": "check_viz_specs_valid",
                }
            )
    return issues


def _check_narrative_not_empty(narrative: str) -> list[dict]:
    """state['narrative'] doit exister et contenir > 100 caractères."""
    issues: list[dict] = []
    if not narrative or len(narrative.strip()) <= 100:
        issues.append(
            {
                "type": "inconsistency",
                "passage": "Narrative absente ou trop courte (≤ 100 caractères).",
                "severity": "high",
                "check": "check_narrative_not_empty",
            }
        )
    return issues


def _check_numbers_in_narrative(narrative: str) -> list[dict]:
    """Plus de 30 nombres dans le narrative → surcharge cognitive."""
    issues: list[dict] = []
    numbers = re.findall(r"\b\d+(?:[.,]\d+)?\b", narrative)
    if len(numbers) > _MAX_NUMBERS_IN_NARRATIVE:
        issues.append(
            {
                "type": "unsupported_claim",
                "passage": f"{len(numbers)} nombres détectés (seuil: {_MAX_NUMBERS_IN_NARRATIVE}).",
                "severity": "low",
                "check": "check_numbers_in_narrative",
            }
        )
    return issues


def _run_deterministic_checks(state: PipelineState) -> list[dict]:
    """Exécute tous les checks sans appel LLM. Retourne la liste d'issues."""
    issues: list[dict] = []
    issues += _check_narrative_not_empty(state.get("narrative", ""))
    issues += _check_insights_have_aggregates(
        state.get("insights", []), state.get("aggregates", {})
    )
    issues += _check_viz_specs_valid(state.get("viz_specs", []), state.get("aggregates", {}))
    issues += _check_numbers_in_narrative(state.get("narrative", ""))
    return issues


# ── Check LLM ─────────────────────────────────────────────────────────────────


def _build_llm_check_prompt(insights: list[dict], narrative: str) -> str:
    insights_summary = "\n".join(
        f"- {i.get('title', '')}: {i.get('description', '')} "
        f"[données: {i.get('supporting_data', '')}]"
        for i in insights[:3]
    )
    narrative_excerpt = narrative[:500] if narrative else "(vide)"
    return (
        "Vérifie la cohérence entre ces insights analytiques et le début de la narration.\n\n"
        f"Insights (sources de vérité) :\n{insights_summary}\n\n"
        f"Début de la narration (500 premiers caractères) :\n{narrative_excerpt}\n\n"
        "Identifie uniquement les affirmations de la narration qui :\n"
        "  - Contredisent directement un insight (hallucination)\n"
        "  - Citent des chiffres absents des insights (unsupported_claim)\n"
        "  - Sont incohérentes avec les données sources (inconsistency)\n\n"
        'Si tout est cohérent, retourne {"issues": []}.\n\n'
        'Format attendu : {"issues": [{"type": "hallucination|inconsistency|unsupported_claim", '
        '"passage": "extrait du texte problématique", "severity": "high|medium|low"}]}'
    )


# ── Scoring ───────────────────────────────────────────────────────────────────


def _compute_score(issues: list[dict]) -> float:
    score = 1.0
    for issue in issues:
        score -= _PENALTY.get(issue.get("severity", "low"), 0.05)
    return max(0.0, score)


def _status_from_score(score: float) -> str:
    if score >= 0.80:
        return "ok"
    if score >= 0.50:
        return "warning"
    return "error"


class QAAgent(BaseAgent):
    """Agent 7 — Vérifie la cohérence et détecte les hallucinations LLM.

    Étapes :
        A. Checks déterministes (0 coût LLM) : narrative non vide, viz_specs valides,
           insights ont des agrégats, surcharge cognitive.
        B. Check LLM : cohérence insights ↔ narrative (gpt-4o-mini, une seule fois).
        C. Calcul du confidence_score (pénalités par sévérité).
        D. HITL si confidence_score < settings.hitl_confidence_threshold.
        E. Écriture dans state["qa_report"].

    Input  : rapport complet + state["aggregates"]
    Output : state["qa_report"]
    Modèle : settings.litellm_cheap_model (gpt-4o-mini)
    HITL   : si confidence_score < 0.80 → HITL obligatoire avant publication
    Pattern : Constitutional AI
    """

    name = "qa_agent"

    async def run(self, state: PipelineState) -> PipelineState:
        log = logger.bind(report_id=state.get("report_id"))

        # ── Étape A : checks déterministes ───────────────────────────────────
        det_issues = _run_deterministic_checks(state)
        checks_run = 4  # narrative_not_empty, insights_have_aggregates, viz_specs_valid, numbers
        checks_passed = checks_run - sum(1 for i in det_issues if i.get("severity") == "high")

        # Si narrative vide, inutile d'appeler le LLM
        narrative = state.get("narrative", "")
        llm_issues: list[dict] = []

        if narrative and len(narrative.strip()) > 100:
            # ── Étape B : check LLM ──────────────────────────────────────────
            prompt = _build_llm_check_prompt(state.get("insights", []), narrative)
            try:
                llm_result = await call_llm_json(
                    prompt=prompt,
                    system=_QA_SYSTEM_PROMPT,
                    model=settings.litellm_cheap_model,
                )
                raw_issues = llm_result.get("issues", [])
                if isinstance(raw_issues, list):
                    for issue in raw_issues:
                        if isinstance(issue, dict):
                            # Normaliser severity
                            if issue.get("severity") not in _PENALTY:
                                issue["severity"] = "low"
                            llm_issues.append(issue)
            except Exception as exc:
                log.warning("qa_llm_check_failed", error=str(exc))
            checks_run += 1

        all_issues = det_issues + llm_issues
        checks_passed = checks_run - sum(1 for i in all_issues if i.get("severity") == "high")
        checks_passed = max(0, checks_passed)

        # ── Étape C : calcul du score ────────────────────────────────────────
        confidence_score = _compute_score(all_issues)
        status = _status_from_score(confidence_score)

        log.info(
            "qa_complete",
            confidence_score=confidence_score,
            status=status,
            issues=len(all_issues),
        )

        # ── Étape D : HITL ───────────────────────────────────────────────────
        if confidence_score < settings.hitl_confidence_threshold:
            state["hitl_pending"] = True
            state["hitl_checkpoint"] = "cp3_insights"
            log.warning("qa_hitl_triggered", confidence_score=confidence_score)

        # ── Étape E ──────────────────────────────────────────────────────────
        state["qa_report"] = {
            "confidence_score": confidence_score,
            "status": status,
            "issues": all_issues,
            "checks_run": checks_run,
            "checks_passed": checks_passed,
        }
        return state
