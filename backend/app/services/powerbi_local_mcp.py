from __future__ import annotations

import csv
import io
import json
from contextlib import AsyncExitStack
from typing import Any

import structlog

from app.config import settings

logger = structlog.get_logger(__name__)


class PowerBIConnectionError(Exception):
    """Erreur générique de communication avec le Power BI Modeling MCP Server."""


class PowerBIDesktopNotFoundError(PowerBIConnectionError):
    """Power BI Desktop n'est pas ouvert avec le fichier attendu."""


class PowerBIDaxExecutionError(Exception):
    """Échec d'exécution (ou de validation) d'une requête DAX."""


def _extract_text(result: Any) -> str:
    for block in result.content or []:
        text = getattr(block, "text", None)
        if text:
            return text
    return ""


def _parse_content(result: Any) -> Any:
    text = _extract_text(result)
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {"raw": text}


def _as_list(value: Any, *keys: str) -> list:
    """Normalise un résultat de tool MCP en liste.

    Confirmé contre un vrai serveur : toutes les réponses "List" enveloppent leur
    liste sous la clé "data" (ex: {"message": "...", "operation": "List", "data": [...]}).
    Les clés candidates supplémentaires (`*keys`) restent en repli défensif.
    """
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for key in ("data", *keys, "items", "value"):
            if isinstance(value.get(key), list):
                return value[key]
    return []


def _coerce_csv_value(raw: str) -> Any:
    """Convertit une valeur CSV texte en int/float natif quand possible (CSV n'a pas de types).

    Confirmé contre un vrai serveur (AdventureWorks) : les nombres décimaux sont formatés
    selon la culture de la connexion Analysis Services — virgule décimale observée
    (ex: "23642495,0952"), pas point décimal. Sans ce cas, la valeur restait une chaîne
    Python (`float()` lève sur une virgule), ce qui aurait rendu le graphique final
    inexploitable côté frontend (`Number("23642495,0952")` vaut `NaN` en JS).

    Limite connue : si un futur serveur utilise le format US avec virgule comme séparateur
    de milliers ET point décimal (ex: "1,234.56"), cette valeur reste inchangée (elle
    contient déjà un ".", donc le remplacement virgule→point n'est pas tenté) — non
    rencontré en pratique, mais à garder en tête si la culture de connexion change.
    """
    if raw == "":
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    if raw.count(",") == 1 and "." not in raw:
        try:
            return float(raw.replace(",", "."))
        except ValueError:
            pass
    return raw


def _clean_dax_header(header: str) -> str:
    """"Table[Colonne]" ou "[Alias]" -> "Colonne"/"Alias" (nettoie le préfixe de qualification DAX)."""
    if "[" in header and header.endswith("]"):
        return header.split("[", 1)[1][:-1]
    return header


def _extract_csv_rows(result: Any) -> list[dict]:
    """Le résultat d'une requête DAX (dax_query_operations, Execute) est renvoyé en CSV

    dans un bloc EmbeddedResource — PAS en JSON dans le texte principal (confirmé contre
    un vrai serveur : `content[0].text` vaut littéralement "{}", les données sont dans
    `content[1].resource.text` avec mimeType "text/csv").
    """
    for block in result.content or []:
        resource = getattr(block, "resource", None)
        text = getattr(resource, "text", None) if resource is not None else None
        if text:
            reader = csv.DictReader(io.StringIO(text))
            return [
                {_clean_dax_header(k): _coerce_csv_value(v) for k, v in row.items()}
                for row in reader
            ]
    return []


class PowerBILocalMCPClient:
    """Client MCP stdio vers le Power BI Modeling MCP Server (microsoft/powerbi-modeling-mcp).

    Le serveur tourne en subprocess local (npx ou binaire .exe téléchargé) et
    communique en JSON-RPC sur stdio.

    Une seule session est maintenue tant que le process backend tourne :
    relancer le subprocess à chaque appel redéclencherait le démarrage npx
    (plusieurs secondes) ET la confirmation MCP Elicitation à chaque requête.

    Comportements confirmés en testant contre un vrai serveur (AdventureWorks,
    Power BI Desktop, serveur v0.5.0-beta.11) :
    - TOUS les tools exigent leurs arguments enveloppés dans {"request": {...}}
      (sinon : "The arguments dictionary is missing a value for the required
      parameter 'request'").
    - Les valeurs de `operation` sont en PascalCase exact ("List", "Get", "Execute",
      "Connect", "ListLocalInstances", ...).
    - Il n'existe PAS de connexion "par nom de fichier" directe : il faut d'abord
      `connection_operations` "ListLocalInstances" (retourne les instances Power BI
      Desktop locales avec leur `connectionString` et `parentWindowTitle`), puis
      matcher par nom de fichier sur `parentWindowTitle`, puis "Connect" avec la
      `connectionString` trouvée.
    - Les réponses "List" de column_operations/measure_operations sont groupées par
      table : {"data": [{"tableName": ..., "columns": [...]}]} — pas une liste plate.
    - dax_query_operations "Execute" renvoie ses lignes en CSV dans un bloc
      EmbeddedResource (pas en JSON) — voir `_extract_csv_rows`.
    - dax_query_operations ne propose PAS d'opération de génération DAX depuis du
      langage naturel (seules "Help, Execute, Validate, ClearCache" existent) —
      le mode `powerbi_dax_generation_mode="mcp_native"` retombe donc systématiquement
      sur le mode "llm" avec le serveur actuel (voir data_agent.py).
    - Le flag qui contrôle la confirmation des écritures est `--require-confirmation`
      (opt-in) — **pas** `--skipconfirmation`. Confirmé via `powerbi-modeling-mcp.exe
      --help` : "Require confirmation prompts for write operations (default: skipped)".
      Sans ce flag, le serveur exécute les modifications SANS jamais solliciter le
      protocole MCP Elicitation, quelle que soit la capacité annoncée côté client.
      `_ensure_session()` l'ajoute donc par défaut (opt-out via
      `powerbi_mcp_skip_confirmation=true`).
    - En complément, `_elicitation_callback` déclare la capacité `elicitation` côté
      client et décline par défaut toute confirmation demandée par le serveur — un
      second niveau de garde-fou, utile si `--require-confirmation` est actif mais
      insuffisant seul (voir avertissement ci-dessus : sans lui, un client MCP qui
      n'annonce pas la capacité `elicitation` ferait échouer/bloquer différemment
      la demande de confirmation du serveur plutôt que de la décliner proprement).
    """

    def __init__(self) -> None:
        self._stack: AsyncExitStack | None = None
        self._session: Any = None
        self._connected_file: str | None = None
        self._tools_cache: dict[str, Any] | None = None

    async def _elicitation_callback(self, context: Any, params: Any) -> Any:
        """Répond aux demandes MCP Elicitation (avant 1ère modification et 1ère requête).

        Passer ce callback à ClientSession est ce qui fait que le SDK `mcp` annonce la
        capacité `elicitation` au serveur (cf. mcp.client.session.ClientSession.initialize
        — `elicitation` reste `None` sans callback non-défaut). Root cause confirmée du
        "Skip Confirmation: Enabled" observé initialement, malgré son nom trompeur : ce
        N'ÉTAIT PAS un problème de capacité client, mais l'absence du flag serveur
        `--require-confirmation` (confirmations SKIPPÉES PAR DÉFAUT, voir `--help` du
        binaire — il n'existe pas de flag `--skipconfirmation`). `_ensure_session()`
        ajoute maintenant `--require-confirmation` par défaut ; CE callback reste un
        second niveau de défense indépendant, pour le cas où le serveur solliciterait
        malgré tout une confirmation MCP Elicitation (plutôt que de la bloquer par un
        autre mécanisme serveur).

        BrouxAI est un backend headless : aucun humain synchrone ne peut répondre à une
        invite interactive pendant un appel de tool. Comportement fail-closed par défaut :
        - settings.powerbi_mcp_skip_confirmation == False (défaut) → on décline. Toute
          tentative de modification (ou de 1ère requête si le serveur l'exige) échoue
          explicitement plutôt que de s'exécuter silencieusement — y compris si un futur
          bug de notre côté appelle un tool de modification par erreur.
        - settings.powerbi_mcp_skip_confirmation == True (opt-in explicite de
          l'opérateur) → on accepte automatiquement.
        """
        from mcp import types

        message = getattr(params, "message", "")
        log = logger.bind(elicitation_message=message, elicitation_mode=getattr(params, "mode", None))

        if settings.powerbi_mcp_skip_confirmation:
            log.info("powerbi_mcp_elicitation_auto_approved")
            return types.ElicitResult(action="accept")

        log.warning("powerbi_mcp_elicitation_declined")
        return types.ElicitResult(action="decline")

    async def _ensure_session(self) -> Any:
        if self._session is not None:
            return self._session

        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        args = settings.powerbi_mcp_args.split()
        # Confirmé via `powerbi-modeling-mcp.exe --help` : le flag réel est
        # `--require-confirmation` (opt-in) — les confirmations sont SKIPPÉES PAR DÉFAUT
        # côté serveur ("default: skipped"). Il n'existe PAS de flag `--skipconfirmation`
        # (contrairement à ce que suggérait la doc publique résumée). On ajoute donc
        # `--require-confirmation` par défaut pour respecter l'intention initiale
        # (confirmations actives par défaut) ; l'opérateur peut l'omettre explicitement
        # via powerbi_mcp_skip_confirmation=true.
        if not settings.powerbi_mcp_skip_confirmation and "--require-confirmation" not in args:
            args = [*args, "--require-confirmation"]

        server_params = StdioServerParameters(command=settings.powerbi_mcp_command, args=args)

        stack = AsyncExitStack()
        try:
            read_stream, write_stream = await stack.enter_async_context(stdio_client(server_params))
            session = await stack.enter_async_context(
                ClientSession(
                    read_stream,
                    write_stream,
                    elicitation_callback=self._elicitation_callback,
                )
            )
            await session.initialize()
        except Exception as exc:
            await stack.aclose()
            raise PowerBIConnectionError(
                "Impossible de démarrer le Power BI Modeling MCP Server "
                f"(commande: {settings.powerbi_mcp_command} {' '.join(args)}). "
                "Vérifiez que Node.js/npx est installé et accessible dans le PATH. "
                f"Détail : {exc}"
            ) from exc

        self._stack = stack
        self._session = session
        logger.info("powerbi_mcp_session_started", command=settings.powerbi_mcp_command)
        return session

    async def _list_tools(self) -> dict[str, Any]:
        """Découvre les tools exposés (une fois par session) — utile pour debug/ajustement."""
        if self._tools_cache is not None:
            return self._tools_cache
        session = await self._ensure_session()
        result = await session.list_tools()
        self._tools_cache = {tool.name: tool.inputSchema for tool in result.tools}
        logger.info("powerbi_mcp_tools_discovered", tools=list(self._tools_cache.keys()))
        return self._tools_cache

    async def _call_tool_raw(self, name: str, arguments: dict) -> Any:
        """Appelle un tool MCP et retourne le CallToolResult brut (non parsé).

        Enveloppe automatiquement `arguments` dans {"request": {...}} — confirmé
        requis par tous les tools de ce serveur.
        """
        session = await self._ensure_session()
        try:
            result = await session.call_tool(name, {"request": arguments})
        except Exception as exc:
            raise PowerBIConnectionError(f"Échec de l'appel au tool MCP '{name}' : {exc}") from exc

        if result.isError:
            raise PowerBIConnectionError(
                f"Tool MCP '{name}' a retourné une erreur : {_extract_text(result)}"
            )
        return result

    async def call_tool(self, name: str, arguments: dict) -> Any:
        """Échappatoire générique JSON : appelle un tool MCP par son nom exact.

        Retourne `structuredContent` si présent, sinon parse le premier bloc
        texte de `content` comme JSON. Ne convient PAS pour dax_query_operations
        "Execute" (résultat en CSV, voir `execute_dax`).
        """
        result = await self._call_tool_raw(name, arguments)
        if result.structuredContent is not None:
            return result.structuredContent
        return _parse_content(result)

    async def connect_to_desktop_file(self, file_name: str) -> dict:
        """Connecte le serveur MCP à l'instance Analysis Services locale de `file_name`.

        Flux réel (confirmé, pas de "connect par nom de fichier" direct) :
        1. `connection_operations` "ListLocalInstances" → instances Desktop locales.
        2. Match par sous-chaîne insensible à la casse sur `parentWindowTitle`.
        3. `connection_operations` "Connect" avec la `connectionString` trouvée.

        Raises:
            PowerBIDesktopNotFoundError: si Power BI Desktop n'est pas ouvert avec
                ce fichier chargé — message actionnable plutôt qu'une stack trace brute.
        """
        log = logger.bind(file_name=file_name)

        # Erreur de démarrage du subprocess (npx/node manquant) → message dédié, pas englobé
        # dans le message "Desktop non ouvert" qui serait trompeur ici.
        await self._ensure_session()
        await self._list_tools()

        try:
            instances_result = await self.call_tool(
                "connection_operations", {"operation": "ListLocalInstances"}
            )
            instances = _as_list(instances_result)
            match = next(
                (
                    inst
                    for inst in instances
                    if isinstance(inst, dict)
                    and file_name.lower() in str(inst.get("parentWindowTitle", "")).lower()
                ),
                None,
            )
            if match is None:
                found = [
                    inst.get("parentWindowTitle") for inst in instances if isinstance(inst, dict)
                ]
                raise PowerBIConnectionError(
                    f"Aucune instance Power BI Desktop locale ne correspond à '{file_name}' "
                    f"(instances ouvertes trouvées : {found})"
                )

            result = await self.call_tool(
                "connection_operations",
                {"operation": "Connect", "connectionString": match["connectionString"]},
            )
        except PowerBIConnectionError as exc:
            log.error("powerbi_connect_failed", error=str(exc))
            raise PowerBIDesktopNotFoundError(
                f"Power BI Desktop doit être ouvert avec le fichier '{file_name}' chargé "
                f"pour que BrouxAI puisse s'y connecter. Détail : {exc}"
            ) from exc

        self._connected_file = file_name
        log.info("powerbi_connected", matched_window_title=match.get("parentWindowTitle"))
        return result if isinstance(result, dict) else {"raw": result}

    async def get_model_metadata(self) -> dict:
        """Récupère tables, colonnes, mesures et relations du modèle sémantique connecté.

        Returns:
            {"tables": {nom: {"columns": {...}, "description": str}},
             "measures": {nom: {...}},
             "relations": [...]}
        """
        if not self._connected_file:
            raise PowerBIConnectionError(
                "Aucune connexion active — appelez connect_to_desktop_file() d'abord."
            )

        metadata: dict[str, Any] = {"tables": {}, "measures": {}, "relations": []}

        tables_result = await self.call_tool("table_operations", {"operation": "List"})
        table_names: list[str] = []
        for table in _as_list(tables_result):
            table_name = table.get("name") if isinstance(table, dict) else str(table)
            if not table_name:
                continue
            table_names.append(table_name)
            # table_operations "List" ne renvoie pas de description (nécessiterait "Get" par table).
            metadata["tables"][table_name] = {"columns": {}, "description": ""}

        if table_names:
            # Un seul appel batché pour toutes les tables (filter.tableNames accepte une liste),
            # plutôt qu'un appel par table.
            columns_result = await self.call_tool(
                "column_operations", {"operation": "List", "filter": {"tableNames": table_names}}
            )
            for entry in _as_list(columns_result):
                if not isinstance(entry, dict):
                    continue
                table_name = entry.get("tableName")
                if table_name not in metadata["tables"]:
                    continue
                for col in entry.get("columns", []) or []:
                    if isinstance(col, dict) and col.get("name"):
                        metadata["tables"][table_name]["columns"][col["name"]] = col

        measures_result = await self.call_tool("measure_operations", {"operation": "List"})
        for entry in _as_list(measures_result):
            if not isinstance(entry, dict):
                continue
            for measure in entry.get("measures", []) or []:
                if isinstance(measure, dict) and measure.get("name"):
                    metadata["measures"][measure["name"]] = measure

        relations_result = await self.call_tool("relationship_operations", {"operation": "List"})
        metadata["relations"] = _as_list(relations_result)

        logger.info(
            "powerbi_metadata_fetched",
            n_tables=len(metadata["tables"]),
            n_measures=len(metadata["measures"]),
            n_relations=len(metadata["relations"]),
        )
        return metadata

    async def execute_dax(self, query: str) -> dict:
        """Exécute une requête DAX et retourne un résultat structuré.

        Le résultat réel est renvoyé en CSV par le serveur (pas en JSON) — voir
        `_extract_csv_rows`. Les valeurs sont converties en int/float quand possible.

        Returns:
            {"columns": [...], "rows": [{col: val, ...}, ...]}

        Raises:
            PowerBIDaxExecutionError: si l'exécution échoue (jamais de crash silencieux).
        """
        try:
            result = await self._call_tool_raw(
                "dax_query_operations", {"operation": "Execute", "query": query}
            )
        except PowerBIConnectionError as exc:
            raise PowerBIDaxExecutionError(f"Échec d'exécution DAX : {exc}") from exc

        rows = _extract_csv_rows(result)
        columns = list(rows[0].keys()) if rows else []
        return {"columns": columns, "rows": rows}

    async def validate_dax(self, query: str) -> bool:
        """Valide une requête DAX sans nécessairement en récupérer le résultat complet."""
        try:
            result = await self.call_tool(
                "dax_query_operations", {"operation": "Validate", "query": query}
            )
        except PowerBIConnectionError:
            return False
        if isinstance(result, dict) and "valid" in result:
            return bool(result["valid"])
        return True

    async def close(self) -> None:
        """Ferme la session MCP et le subprocess associé."""
        if self._stack is not None:
            await self._stack.aclose()
        self._stack = None
        self._session = None
        self._connected_file = None
        self._tools_cache = None


_client: PowerBILocalMCPClient | None = None


def get_powerbi_client() -> PowerBILocalMCPClient:
    """Retourne le client MCP Power BI singleton du process backend."""
    global _client
    if _client is None:
        _client = PowerBILocalMCPClient()
    return _client
