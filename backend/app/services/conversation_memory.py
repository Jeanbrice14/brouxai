from __future__ import annotations

import json

import structlog

from app.config import settings

logger = structlog.get_logger(__name__)

# Même durée de vie que l'index de sessions (report_store.py::_SESSION_TTL) — perdre
# l'historique de conversation ne devrait pas arriver plus vite que perdre la session
# elle-même dans la sidebar.
_HISTORY_TTL = 30 * 24 * 3600  # 30 jours


def _history_key(tenant_id: str, session_id: str) -> str:
    return f"chat_history:{tenant_id}:{session_id}"


async def _get_client():
    """Crée un client Redis async. Retourne None si Redis non disponible."""
    try:
        import redis.asyncio as aioredis

        client = aioredis.from_url(settings.redis_url, decode_responses=True)
        await client.ping()
        return client
    except Exception as exc:
        logger.warning("conversation_memory_redis_unavailable", error=str(exc))
        return None


async def get_conversation_history(
    tenant_id: str, session_id: str, k: int | None = None
) -> list[dict]:
    """Retourne les k derniers tours (question, réponse résumée) d'une session, du plus
    ancien au plus récent (ordre chronologique de lecture pour un prompt LLM).

    Ne bloque jamais : liste vide si Redis indisponible, session inconnue, ou aucun historique.
    """
    if not tenant_id or not session_id:
        return []
    _k = k or settings.chat_history_k
    client = await _get_client()
    if client is None:
        return []
    try:
        raw_items = await client.lrange(_history_key(tenant_id, session_id), 0, _k - 1)
        turns = [json.loads(item) for item in raw_items]
        turns.reverse()  # LPUSH place le plus récent en tête — on veut l'ordre chronologique
        return turns
    except Exception as exc:
        logger.warning("conversation_history_get_error", session_id=session_id, error=str(exc))
        return []
    finally:
        await client.aclose()


async def append_conversation_turn(
    tenant_id: str,
    session_id: str,
    question: str,
    answer_summary: str,
    k: int | None = None,
) -> None:
    """Ajoute un tour à l'historique de conversation, borné aux k derniers.

    Ne stocke qu'un résumé texte (jamais les agrégats/insights bruts) — le coût d'injection
    dans les prompts SQL/DAX reste négligeable même en l'incluant systématiquement, ce qui
    évite d'avoir à détecter par heuristique si la question courante référence la précédente.
    Ne plante jamais.
    """
    if not tenant_id or not session_id or not question:
        return
    _k = k or settings.chat_history_k
    client = await _get_client()
    if client is None:
        return
    try:
        key = _history_key(tenant_id, session_id)
        turn = json.dumps({"question": question, "answer_summary": answer_summary})
        await client.lpush(key, turn)
        await client.ltrim(key, 0, _k - 1)
        await client.expire(key, _HISTORY_TTL)
        logger.info("conversation_turn_recorded", session_id=session_id)
    except Exception as exc:
        logger.warning("conversation_turn_append_error", session_id=session_id, error=str(exc))
    finally:
        await client.aclose()


def format_history_for_prompt(history: list[dict]) -> str:
    """Formate l'historique en texte court pour injection dans un prompt LLM.

    Chaîne vide si l'historique est vide — au prompt appelant d'omettre entièrement la
    section dans ce cas plutôt que d'afficher un bloc "Contexte : (vide)".
    """
    if not history:
        return ""
    lines: list[str] = []
    for i, turn in enumerate(history, 1):
        question = turn.get("question", "")
        answer = turn.get("answer_summary", "")
        if not question:
            continue
        lines.append(f"  Q{i}: {question}")
        if answer:
            lines.append(f"  R{i}: {answer}")
    return "\n".join(lines)


def should_record_turn(state: dict) -> bool:
    """Un tour ne doit être enregistré que s'il correspond à une vraie question aboutie —
    jamais le prompt générique de connexion (setup_only), ni un état encore en attente
    HITL ou en erreur (la narration pourrait être absente ou porter sur des données
    incomplètes)."""
    return (
        state.get("status") == "complete"
        and not state.get("setup_only")
        and bool(state.get("narrative"))
    )


async def maybe_record_turn(state: dict) -> None:
    """Enregistre le tour courant dans l'historique de conversation si pertinent (cf.
    should_record_turn). Point d'entrée unique, appelé après complétion du pipeline
    (reports.py::_run_pipeline et hitl.py::_run_resume) — jamais bloquant."""
    if not should_record_turn(state):
        return
    await append_conversation_turn(
        tenant_id=state.get("tenant_id", ""),
        session_id=state.get("session_id", ""),
        question=state.get("prompt", ""),
        answer_summary=state.get("narrative", ""),
    )
