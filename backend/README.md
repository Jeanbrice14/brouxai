# BrouxAI — Backend

Pipeline multi-agent FastAPI + LangGraph. Voir `../CLAUDE.md` pour l'architecture complète.

## Sources de données

BrouxAI supporte deux sources de données, sélectionnées via `PipelineState["data_source"]` :

| Mode | `data_source` | Entrée | Exécution des requêtes |
|------|----------------|--------|--------------------------|
| Upload CSV/Excel (défaut) | `"csv"` | `POST /api/v1/reports/generate` (fichiers) | SQL via DuckDB (in-process) |
| Power BI Desktop local | `"powerbi_local"` | `POST /api/v1/reports/generate-powerbi` (`pbix_file_name`) | DAX via le Power BI Modeling MCP Server |

Les deux modes partagent le même pipeline (InsightAgent, StorytellingAgent, VizAgent, QAAgent,
LayoutAgent) — seuls MetadataAgent et DataAgent changent de comportement selon la source.

### Routage du pipeline (`app/pipeline/graph.py`)

Depuis que `SchemaLinkingAgent` est court-circuité en mode `powerbi_local` (relations déjà
connues côté Power BI, rien à y redétecter), le graphe bifurque juste après `MetadataAgent`
selon `state["data_source"]` — voir `_route_after_metadata()` :

```
IntentAgent
    │
    ├─ metadata+schema déjà en cache (session existante) ────────────────────► DataAgent
    │
    ▼ (sinon : full_setup)
MetadataAgent ──(HITL CP1 ?)──► hitl_wait (pause — reprise via POST /reports/{id}/review)
    │
    │ data_source == "csv"           ──► SchemaLinkingAgent ──(HITL CP2 ?)──┐
    │ data_source == "powerbi_local" ─────────────────────────────────────┐│
    │ data_source inconnu (futur)    ──► SchemaLinkingAgent (repli         ││
    │                                    conservateur + warning loggé)     ││
    │                                                                     ▼▼
    │                                                                  DataAgent
    │                                                                     │
    │              ┌──────────────────────────────────────────────────────┤
    │        intent=simple_query                          intent=chart_request | full_report
    │              ▼                                                      ▼
    │         LayoutAgent ◄──────────────────────────  InsightAgent ──(HITL CP3 ?)──► StorytellingAgent
    │                                                                                        │
    │                                                                                        ▼
    │                                                                   QAAgent ◄── VizAgent
    │                                                                      │
    │                                                                      ▼
    └────────────────────────────────────────────────────────────────► LayoutAgent ──► END
```

Points clés :
- `SchemaLinkingAgent` n'est **jamais invoqué** en mode `powerbi_local` — pas un no-op interne
  à l'agent, il est absent du chemin d'exécution du graphe (vérifié par test, voir
  `tests/integration/test_data_source_routing.py`).
- La reprise après CP1 (`app/pipeline/checkpoints.py::resume_pipeline`) suit la même règle :
  reprend vers `schema_linking_agent` en mode `csv`, directement vers `data_agent` en mode
  `powerbi_local`.
- Un `data_source` futur sans branche explicite route par défaut vers `SchemaLinkingAgent`
  (comportement conservateur — un agent inutile est préférable à un agent manquant) avec un
  warning loggé (`route_after_metadata_unknown_data_source`).
- Le routage post-`DataAgent` (`_route_after_data`, basé sur `intent`) est indépendant de cette
  bifurcation `data_source` — les deux dimensions ne se chevauchent qu'au niveau de `DataAgent`,
  point de convergence commun aux deux chemins.

## Power BI Desktop local (MCP) — v1+

Cette intégration connecte BrouxAI à un modèle sémantique **déjà ouvert dans Power BI Desktop**,
sans upload de fichier. Elle utilise le
[Power BI Modeling MCP Server](https://github.com/microsoft/powerbi-modeling-mcp) officiel de
Microsoft, qui communique en stdio (JSON-RPC) et se connecte automatiquement à l'instance
Analysis Services locale que Power BI Desktop expose quand un `.pbix`/`.pbip` est ouvert.

### Prérequis

- **Windows uniquement** pour cette phase (connexion à l'instance Analysis Services locale de
  Power BI Desktop).
- **Power BI Desktop** installé, avec le fichier cible **ouvert** avant de lancer une session
  BrouxAI dessus (le serveur recherche l'instance par nom de fichier).
- **Node.js/npx** installé et accessible dans le `PATH` (le serveur MCP se lance via
  `npx -y @microsoft/powerbi-modeling-mcp@latest --start`, pas d'installation manuelle requise).
- Aucune variable d'environnement de type `client_id`/`client_secret` : l'authentification se
  fait via l'identité locale déjà connectée (`--authmode interactive`, valeur par défaut).

### Configuration (`.env`)

```env
POWERBI_MCP_COMMAND=npx
POWERBI_MCP_ARGS=-y @microsoft/powerbi-modeling-mcp@latest --start
POWERBI_PBIX_FILE_NAME=
POWERBI_MCP_SKIP_CONFIRMATION=false
POWERBI_DAX_GENERATION_MODE=llm
```

- `POWERBI_MCP_SKIP_CONFIRMATION` : le serveur applique le protocole MCP Elicitation — une
  confirmation utilisateur est demandée avant la première modification et avant la première
  requête exécutée sur le modèle. Laisser à `false` pour voir ce comportement en action
  (cohérent avec notre philosophie HITL). Ne l'activer qu'après validation du flux de base.
- `POWERBI_DAX_GENERATION_MODE` : `"llm"` (notre prompt LiteLLM génère le DAX, seul mode
  fonctionnel aujourd'hui) ou `"mcp_native"` (délègue à un tool de génération DAX du serveur
  MCP — **confirmé absent** de la version actuelle du serveur, voir Limitations ci-dessous ;
  repli automatique et loggé en warning sur `"llm"`).

### Utiliser une session Power BI

1. Ouvrir le fichier `.pbix`/`.pbip` cible dans Power BI Desktop.
2. Appeler `POST /api/v1/reports/generate-powerbi` avec `prompt` + `pbix_file_name` (nom exact
   du fichier, sans nécessairement le chemin complet — le serveur MCP le retrouve).
3. Le pipeline se comporte ensuite comme le mode CSV (statut consultable via
   `GET /api/v1/reports/{report_id}`).

### Comportement du serveur MCP — confirmé contre un vrai serveur

Testé en conditions réelles (Power BI Modeling MCP v0.5.0-beta.11, Power BI Desktop, jeu de
données AdventureWorks). Points qui différaient de la documentation publique (catégories de
tools sans schéma d'arguments) et qui sont maintenant reflétés dans le code :

- **Tous** les tools exigent leurs arguments enveloppés dans `{"request": {...}}` (sinon :
  `"missing a value for the required parameter 'request'"`).
- Les valeurs de `operation` sont en **PascalCase exact** (`"List"`, `"Get"`, `"Execute"`,
  `"Connect"`, `"ListLocalInstances"`, ...).
- **Pas de connexion directe par nom de fichier.** Le flux réel est en 2 étapes :
  1. `connection_operations` `"ListLocalInstances"` → liste des instances Power BI Desktop
     locales avec leur `connectionString` et `parentWindowTitle` (titre de fenêtre, pas le nom
     de fichier exact).
  2. Match par sous-chaîne insensible à la casse sur `parentWindowTitle`, puis
     `connection_operations` `"Connect"` avec la `connectionString` trouvée.
- Les réponses `"List"` de `column_operations`/`measure_operations` sont **groupées par table**
  (`{"data": [{"tableName": ..., "columns": [...]}]}`), pas une liste plate de colonnes/mesures.
- `column_operations`/`measure_operations` `"List"` **ne renvoient pas la description** des
  colonnes/mesures (champ présent uniquement via `"Get"`, testé et confirmé vide sur
  AdventureWorks faute de documentation dans ce jeu de données). En mode `powerbi_local`,
  `MetadataAgent` verra donc généralement une confiance basse et déclenchera CP1 — cohérent
  avec l'absence réelle de documentation dans le modèle, mais implique un coût HITL plus élevé
  que prévu initialement sur des modèles non documentés.
- `dax_query_operations` `"Execute"` renvoie ses lignes en **CSV** dans un bloc
  `EmbeddedResource` (mimeType `text/csv`) — **pas en JSON** dans le texte principal (celui-ci
  vaut littéralement `"{}"`). `services/powerbi_local_mcp.py` parse ce CSV et nettoie les
  en-têtes DAX du style `Table[Colonne]`/`[Alias]`.
- `dax_query_operations` ne supporte que `"Help, Execute, Validate, ClearCache"` — **il n'existe
  pas d'opération de génération DAX depuis du langage naturel** dans cette version du serveur.
  `POWERBI_DAX_GENERATION_MODE=mcp_native` retombera donc systématiquement sur `"llm"`
  (warning explicite loggé à chaque fois, jamais silencieux) tant que Microsoft n'ajoute pas
  cette capacité.

### Confirmations MCP (write-ops) — root cause trouvée et corrigée

Le serveur a démarré avec `"Skip Confirmation: Enabled"` dans nos tests initiaux, sans jamais
passer `--skipconfirmation` — ce flag **n'existe même pas**. Confirmé via
`powerbi-modeling-mcp.exe --help` (binaire réel, pas la doc résumée) : le vrai flag est
**`--require-confirmation`** (opt-in), documenté "default: skipped". Autrement dit, **les
confirmations sont désactivées par défaut côté serveur**, quel que soit le client MCP utilisé.

Vérifié en conditions réelles : un premier test consistant à appeler `table_operations`
`"Create"` sans ce flag a effectivement créé une vraie table dans le modèle AdventureWorks
ouvert, sans aucune sollicitation MCP Elicitation — confirmant que le risque était réel, pas
théorique (la table a été supprimée immédiatement après, aucune persistance car non sauvegardée
dans le `.pbix`).

Deux niveaux de garde-fou sont maintenant en place :
1. **`_ensure_session()`** ajoute `--require-confirmation` par défaut aux arguments du serveur
   (omis uniquement si `POWERBI_MCP_SKIP_CONFIRMATION=true`, opt-out explicite de l'opérateur).
2. **`_elicitation_callback`** déclare la capacité `elicitation` côté client (le SDK `mcp` ne
   l'annonce que si un callback non-défaut est fourni à `ClientSession`) et **décline par
   défaut** toute confirmation demandée par le serveur — deuxième filet de sécurité si jamais
   le serveur sollicite une confirmation MCP Elicitation malgré tout.

`POWERBI_MCP_SKIP_CONFIRMATION=false` (défaut) doit donc maintenant effectivement bloquer toute
tentative de modification tant que l'opérateur ne l'active pas explicitement.

En dev local, `.env.local` (non commité, absent de `.env.exemple`) surcharge `.env` avec
`POWERBI_MCP_SKIP_CONFIRMATION=true` pour débloquer les tests d'intégration sans avoir à modifier
`.env` ni les valeurs par défaut du code — voir `app/config.py` (`_ENV_LOCAL_FILE`).

### HITL et elicitation MCP (piste future, non implémenté)

Aujourd'hui, un décliné d'elicitation (que ce soit `_elicitation_callback` qui décline, ou un
vrai refus humain côté serveur) atterrit dans `DataAgent` comme n'importe quelle autre erreur
DAX : `PowerBIDaxExecutionError` levée par `execute_dax()`, catchée par `BaseAgent.__call__`
comme une exception générique → `state["status"] = "error"`. Il n'y a aucun moyen de distinguer
"le serveur a demandé une confirmation et on l'a refusée" d'un vrai bug DAX (syntaxe invalide,
colonne inexistante, etc.) — les deux ont la même forme d'erreur pour l'instant.

**Idée pour une itération future** : faire remonter cet événement comme un HITL checkpoint
dédié plutôt que de dépendre uniquement du flag global `POWERBI_MCP_SKIP_CONFIRMATION`, pour
permettre une vraie approbation humaine au cas par cas (cohérent avec la philosophie HITL du
reste du pipeline — cf. CP1-CP5 dans `CLAUDE.md`, sachant que CP5/QA ne déclenche déjà pas
réellement de HITL en v0, voir `qa_agent.py`).

**Ce qu'il faudrait, à haut niveau** (aucun code ci-dessous n'est implémenté) :

1. **Détecter l'événement précisément.** `_elicitation_callback` sait déjà quand il décline
   (il reçoit `params.message`, ex: `"Are you sure you want to execute dax queries..."`). Il
   faudrait mémoriser ce fait (ex: `self._last_declined_elicitation: dict | None` sur le client,
   ou une exception dédiée `PowerBIElicitationDeclinedError(PowerBIDaxExecutionError)` portant le
   message d'elicitation), pour que `execute_dax()` puisse la distinguer d'un échec DAX ordinaire.
2. **Décider où l'exposer dans `PipelineState`.** Deux options, à trancher plus tard :
   - **Option A — nouveau `cp0_powerbi_connection`** : checkpoint dédié, déclenché dans
     `DataAgent` (ou même plus tôt, dans `MetadataAgent` au moment de `connect_to_desktop_file`)
     dès qu'une elicitation est déclinée. Sémantiquement le plus juste — ce n'est pas un problème
     de qualité d'insight (CP3) mais d'autorisation d'accès à une source de données externe live.
     Coût : nécessite un nouveau composant frontend de review (à côté de `MetadataReview`,
     `SchemaReview`, `InsightReview`) et une entrée dans le vocabulaire de checkpoints existant.
     `pipeline/router.py`/`pipeline/graph.py` n'auraient pas besoin de changer (le routeur ne
     regarde que `hitl_pending`, pas la valeur de `hitl_checkpoint`).
   - **Option B — fusionner dans CP3 (`InsightAgent`)** : traiter un décliné d'elicitation comme
     une raison supplémentaire de déclencher `cp3_insights` (au même titre que confidence basse
     ou anomalie détectée), en propageant l'info depuis `state["errors"]` jusqu'à `InsightAgent`.
     Plus simple (pas de nouveau checkpoint ni de nouvel écran), mais sémantiquement bancal —
     CP3 valide la qualité des insights, pas l'autorisation d'accès à la donnée source, et
     `InsightAgent` tourne après `DataAgent` alors que le problème existe déjà au niveau des
     agrégats (potentiellement vides si toutes les requêtes DAX ont été déclinées).
3. **Le point dur, indépendant du choix A/B : la reprise (resume).** Les checkpoints CP1-CP4
   existants mettent en pause le graphe LangGraph lui-même (`hitl_wait`, nœud terminal) — la
   reprise relance simplement l'agent suivant avec un state enrichi par la correction humaine.
   Une elicitation MCP, elle, se produit **à l'intérieur d'un seul appel de tool synchrone**
   (`session.call_tool(...)` attend la réponse du callback avant de retourner) : au moment où
   `DataAgent` récupère l'exception, l'appel a déjà échoué et la session MCP est passée à autre
   chose. Reprendre voudrait dire **relancer la même requête DAX** en signalant au client MCP
   d'approuver *cette fois-ci* — ce qui suppose un mécanisme d'approbation "au coup par coup"
   (ex: un flag transmis depuis l'API HITL jusqu'à `execute_dax()` pour ce report_id précis),
   distinct du flag global `powerbi_mcp_skip_confirmation` qui approuverait sinon TOUT,
   indéfiniment, pour toutes les requêtes futures — une plomberie différente de la reprise
   HITL actuelle, à concevoir spécifiquement si cette piste est retenue.

## RAG schéma sémantique (pgvector) — mode powerbi_local

`DataAgent` injectait jusqu'ici tout `semantic_model_info` (toutes les tables/colonnes/mesures
du modèle Power BI) dans le prompt de génération DAX, même pour des questions ne portant que
sur un sous-ensemble — coûteux en tokens et risque de disperser le LLM. Un système RAG sur
pgvector indexe le schéma une fois par (tenant, modèle) et ne récupère que les k champs les
plus pertinents pour chaque question.

### Mise en place

```bash
docker compose up -d postgres   # pgvector/pgvector:pg16
cd backend
alembic upgrade head            # active l'extension vector + crée semantic_field_embeddings
```

### Blocage réseau Windows → Postgres : résolu (backend en conteneur)

**Diagnostic (session précédente)** : la connexion asyncpg/psycopg2 depuis un backend lancé
nativement sur Windows (`uvicorn` en local) vers le Postgres dockerisé échouait
systématiquement (`ConnectionResetError` / `UnicodeDecodeError` selon le driver) — le handshake
protocole Postgres complet fonctionnait pourtant en socket brut manuel (SSL negotiation,
SCRAM/MD5 auth request tous reçus correctement), la connexion étant réinitialisée précisément
quand le driver client écrit sa réponse d'authentification. Cause probable : interaction Docker
Desktop (proxy réseau Windows) / pile asyncio Proactor.

**Hypothèse testée et confirmée** : faire tourner le **backend lui-même dans un conteneur
Docker**, sur le même réseau que Postgres, plutôt que nativement sur Windows. Résultat : connexion
immédiate et stable, `index_schema()`/`retrieve_relevant_fields()` réels validés de bout en bout
(147 champs indexés, retrieval hybride confirmé avec le vrai pgvector — voir ci-dessus). Le
problème est donc bien localisé au chemin réseau Windows-hôte → proxy Docker Desktop → conteneur,
pas à pgvector, asyncpg, ni au code applicatif.

**Méthode recommandée sur Windows** (`backend/Dockerfile` + service `backend` dans
`docker-compose.yml`, profil `backend` — n'affecte pas `docker compose up -d` par défaut) :

```bash
docker compose --profile backend up -d --build backend
curl http://localhost:8000/health
```

Le service surcharge `DATABASE_URL`/`DATABASE_URL_SYNC`/`REDIS_URL`/`STORAGE_ENDPOINT` de
`backend/.env` (qui pointent vers `localhost`, valables pour un usage natif) avec les noms de
service Docker (`postgres`/`redis`/`minio`) via `environment:` — le reste (clés API, etc.) est
chargé depuis `backend/.env` via `env_file:`. `backend/.dockerignore` exclut `.env`/`.env.local`
de l'image (ne jamais bake un secret dans l'image ; il est injecté à l'exécution par
`docker-compose`).

Continuer à lancer `uvicorn` nativement sur Windows reste possible pour tout ce qui ne touche
pas Postgres (le reste du pipeline — Redis, MinIO, MCP Power BI — fonctionne nativement sans
problème, confirmé sur plusieurs sessions).

### Modèle d'embedding

**`text-embedding-3-small`** (OpenAI, 1536 dimensions) via `litellm.aembedding()` — même clé
`OPENAI_API_KEY` que le reste du projet, pas de nouveau provider/credential. Choix par défaut
faute d'alternative déjà configurée dans le repo ; `SCHEMA_RAG_EMBEDDING_MODEL` reste
overridable si besoin.

### Schéma pgvector (`semantic_field_embeddings`)

Aucune infrastructure SQLAlchemy/Alembic n'existait avant cette session (dépendances déjà
déclarées dans `pyproject.toml` mais jamais câblées) — bootstrap complet : `app/db.py`
(moteur async), `app/models/base.py` + `app/models/semantic_field_embedding.py`, `alembic.ini` +
`migrations/`.

| Colonne | Type | Rôle |
|---|---|---|
| `id` | UUID (PK) | |
| `tenant_id`, `model_id` | String | `model_id` = `pbix_file_name` — **pas** `session_id` : l'index et les corrections humaines doivent survivre à travers plusieurs sessions de chat sur le même fichier Power BI, pas être reconstruits à chaque nouvelle session |
| `qualified_name` | String | `"Table[Colonne]"` / `"[Mesure]"` / `"TableA[ColA]->TableB[ColB]"` pour les relations |
| `object_type` | String | `table`\|`column`\|`measure`\|`relationship` |
| `parent_table` | String? | Table propriétaire (colonnes uniquement) |
| `description` | String | Description du modèle, ou validée par CP1 |
| `embedding` | `vector(1536)` | pgvector |
| `human_verified` | Boolean | `true` si confirmé/corrigé via CP1 — jamais réécrit par une réindexation |
| `confidence` | Float | 0.95 (description présente), 0.60 (absente), 1.0 (human_verified) |
| `updated_at` | timestamptz | |

Contrainte d'unicité `(tenant_id, model_id, qualified_name)`. Pas d'index ANN (ivfflat/hnsw) —
un modèle Power BI compte typiquement quelques dizaines à ~200 champs, un scan séquentiel avec
l'opérateur `<=>` (cosine) suffit largement.

### `k` par défaut : 15

`SCHEMA_RAG_K=15`, configurable sans redéploiement. Choisi empiriquement (voir validation
ci-dessous) : sur les questions testées contre AdventureWorks, les résultats réellement
pertinents (mesure/table/colonnes clés + relation de jointure nécessaire) se concentrent dans
les 7-10 premiers rangs — 15 laisse une marge sans injecter un volume proche du schéma complet.
À affiner avec plus de retours d'usage réel.

### Retrieval hybride pour les questions temporelles

Le retrieval sémantique pur ratait systématiquement les champs calendrier sur les questions
d'évolution/tendance (ex: "évolution des ventes par mois" ne remontait pas `Calendar Lookup`
dans le top 15, alors qu'ils sont indispensables au `GROUP BY` DAX correct). Corrigé par un
**boost hybride** dans `retrieve_relevant_fields()` :

1. Détection par mot-clé sur la **question** (`mois`, `mensuel`, `année`, `trimestre`,
   `évolution`, `tendance`, `historique`, `dans le temps`, `au fil du`, `semaine`, `progression`,
   etc. — voir `_TEMPORAL_QUESTION_KEYWORDS`).
2. Si détecté, les champs marqués **`is_temporal=true`** en base sont ajoutés au résultat
   sémantique (union dédupliquée par `qualified_name`, jamais un remplacement — plafonné à 30
   candidats temporels).
3. `is_temporal` est calculé une fois à l'indexation (`_is_temporal_field`), pas re-dérivé à
   chaque retrieval : combine le `dtype` réel (`DateTime`/`Date`/`Time`, fiable mais absent pour
   les colonnes calendrier typées `Int64`/`String` comme `Month`) ET une heuristique de nom
   (`calendar`, `calendrier`, `date`, `month`, `mois`, `year`, `année`, `quarter`, `trimestre`,
   `week`, `semaine`...) — persisté plutôt que re-calculé pour rester correct même si un futur
   retrieval ne repasse pas par `_flatten_fields`.

**Validé contre le vrai pgvector** (voir section Docker ci-dessous) : pour "évolution des ventes
par mois", `Calendar Lookup` (table + 16 colonnes `Date`/`Month`/`Year`/...) et la relation
`Sales Data[OrderDate]->Calendar Lookup[Date]` sont maintenant bien présents dans le résultat —
absents avant ce fix.

### Validation réelle contre AdventureWorks

Résultats sur le schéma réel (147 champs indexables, AdventureWorks), **retrieval sémantique
pur** (avant boost hybride) :

- **"ventes par catégorie"** : `[Total Sales]` (mesure) en tête, puis `Product Categories
  Lookup` (table + colonnes `CategoryName`/`ProductCategoryKey`), puis la relation de jointure
  `Product Subcategories Lookup[ProductCategoryKey]->Product Categories Lookup[...]` — tout ce
  qu'il faut pour écrire le `SUMMARIZECOLUMNS` correct est présent dans le top 10.
- **"quelles sont les ventes totales par catégorie de produit ?"** (formulation complète) :
  résultats encore plus nets, `[Total Sales]` avec une similarité nettement plus haute (0.53
  vs 0.44).
- **"taux de retour par produit"** : la table `Return Rate` et les mesures de retour dominent
  le top 10 — pertinent.
- **"évolution des ventes par mois"** : voir "Retrieval hybride" ci-dessus — corrigé.

### Fallback schéma complet

Jamais déclenché sur les cas réels testés (retrieval toujours non vide, y compris contre le
vrai pgvector — voir section Docker). Les seuls déclenchements observés sont dans les tests
unitaires où l'échec est simulé exprès (panne d'embedding, panne DB). En usage réel, le
fallback ne se déclenchera que si pgvector est indisponible ou si le modèle n'a jamais été
indexé.

### Autres limitations connues (v1, première itération)

- `SchemaLinkingAgent` n'est plus invoqué du tout en mode `powerbi_local` (voir section
  "Routage du pipeline" ci-dessus) — pas adapté à ce mode de toute façon (relations déjà
  connues côté Power BI, rien à y redétecter). `raw_data_refs` reste vide dans ce mode, mais
  c'est désormais la valeur sémantiquement correcte (aucun fichier Blob Storage dans ce mode),
  plus un contournement anti-crash. Les relations du modèle sont dans
  `state["semantic_model_info"]["relations"]`, consommées directement par `DataAgent`.
- DuckDB reste disponible en fallback local pour le mode CSV (et pour les tests unitaires sans
  dépendre de Power BI Desktop) — il n'est plus sur le chemin critique du mode `powerbi_local`.
