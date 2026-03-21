from __future__ import annotations

import traceback
from abc import ABC, abstractmethod

import structlog

from app.pipeline.state import PipelineState
from app.services.report_store import save_report_state

logger = structlog.get_logger(__name__)


class BaseAgent(ABC):
    """Classe abstraite dont héritent tous les agents du pipeline BrouxAI.

    Responsabilités :
    - Journalisation structurée (entrée / sortie / durée)
    - Capture et enregistrement des erreurs dans state["errors"]
    - Mise à jour de state["current_agent"] et state["status"]

    Les agents concrets n'ont qu'à implémenter `run()` — ils ne doivent JAMAIS
    gérer les exceptions eux-mêmes : c'est le rôle de `__call__`.
    """

    # Nom de l'agent, à redéfinir dans chaque sous-classe
    name: str = "base_agent"

    async def __call__(self, state: PipelineState) -> PipelineState:
        """Point d'entrée appelé par LangGraph.

        Orchestre : logging → exécution → gestion d'erreur.
        """
        log = logger.bind(
            agent=self.name,
            report_id=state.get("report_id"),
            tenant_id=state.get("tenant_id"),
        )
        log.info("agent_start")

        state["current_agent"] = self.name
        state["status"] = "running"

        try:
            state = await self.run(state)
            log.info("agent_complete")
        except Exception as exc:
            error_msg = f"{self.name}: {type(exc).__name__}: {exc}"
            log.error("agent_error", error=error_msg, traceback=traceback.format_exc())
            state["errors"] = state.get("errors", []) + [error_msg]
            state["status"] = "error"

        # Persister le state dans Redis après chaque agent (pour WebSocket + reprise HITL)
        if state.get("report_id"):
            await save_report_state(state["report_id"], dict(state))

        return state

    @abstractmethod
    async def run(self, state: PipelineState) -> PipelineState:
        """Logique métier de l'agent.

        Doit lire depuis `state`, écrire dans sa clé dédiée, et retourner `state`.
        Ne doit PAS gérer les exceptions — elles remontent à `__call__`.
        """
        ...
