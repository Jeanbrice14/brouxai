from __future__ import annotations

import json
import re

import litellm
import structlog

from app.config import settings

logger = structlog.get_logger(__name__)

# Désactiver les logs verbeux de litellm
litellm.suppress_debug_info = True


async def call_llm(
    prompt: str,
    system: str = "",
    model: str | None = None,
    temperature: float = 0.0,
) -> str:
    """Appelle le LLM et retourne le texte brut de la réponse.

    Args:
        prompt: Message utilisateur.
        system: Instruction système (optionnelle).
        model: Identifiant du modèle LiteLLM. Défaut : settings.litellm_cheap_model.
        temperature: Température de génération (0.0 = déterministe).

    Returns:
        Contenu texte de la réponse.
    """
    _model = model or settings.litellm_cheap_model

    messages: list[dict] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = await litellm.acompletion(
        model=_model,
        messages=messages,
        temperature=temperature,
    )

    content: str = response.choices[0].message.content or ""
    usage = getattr(response, "usage", None)
    tokens = usage.total_tokens if usage else None

    logger.info("llm_call", model=_model, tokens=tokens, prompt_len=len(prompt))
    return content


async def call_llm_json(
    prompt: str,
    system: str = "Retourne UNIQUEMENT du JSON valide, sans markdown, sans explication.",
    model: str | None = None,
) -> dict:
    """Appelle le LLM et garantit un dict Python parsé depuis le JSON retourné.

    Args:
        prompt: Message utilisateur.
        system: Instruction système. Défaut : instruction JSON strict.
        model: Identifiant du modèle LiteLLM.

    Returns:
        dict Python parsé depuis la réponse JSON.

    Raises:
        ValueError: Si le LLM ne retourne pas de JSON parsable.
    """
    content = await call_llm(prompt=prompt, system=system, model=model, temperature=0.0)

    # Tentative 1 : JSON brut
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        pass

    # Tentative 2 : extraire un bloc ```json ... ``` ou ``` ... ```
    block_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", content, re.DOTALL)
    if block_match:
        try:
            return json.loads(block_match.group(1))
        except json.JSONDecodeError:
            pass

    # Tentative 3 : extraire le premier objet JSON { ... } dans le texte
    obj_match = re.search(r"\{.*\}", content, re.DOTALL)
    if obj_match:
        try:
            return json.loads(obj_match.group())
        except json.JSONDecodeError:
            pass

    raise ValueError(
        f"LLM n'a pas retourné de JSON valide. "
        f"Modèle: {model or settings.litellm_cheap_model}. "
        f"Réponse (200 premiers chars): {content[:200]}"
    )
