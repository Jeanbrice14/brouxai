# BrouxAI — Claude Code Project Instructions

## Présentation du Projet

**BrouxAI** est une plateforme SaaS no-code de **data storytelling multi-agent**.
Elle permet à des utilisateurs non-techniciens de transformer des fichiers de données bruts
en rapports narratifs visuels complets, via un simple prompt en langage naturel.

**Répertoire principal** : `BrouxAI/`

```
BrouxAI/
├── backend/         ← FastAPI + LangGraph agents (Python)
├── frontend/        ← Next.js 14 + Tailwind + Recharts (TypeScript)
├── infra/           ← Docker Compose + Kubernetes manifests
├── docs/            ← Documentation technique
├── CLAUDE.md        ← Ce fichier
└── .env.example
```

---

## Stack Technique

### Backend (Python 3.11+)
| Rôle | Technologie |
|------|-------------|
| API REST + WebSocket | FastAPI |
| Orchestration agents | LangGraph |
| Abstraction LLM multi-provider | LiteLLM |
| Task queue async | Celery + Redis |
| ORM base de données | SQLAlchemy 2.0 (async) |
| Validation données | Pydantic v2 |
| Logging structuré | structlog |

### LLM Providers (via LiteLLM)
- **Par défaut** : `gpt-4o` (agents critiques : Insight, Storytelling)
- **Économique** : `gpt-4o-mini` (agents simples : Metadata, Viz, QA)
- **Configurable** via `LITELLM_DEFAULT_MODEL` et `LITELLM_CHEAP_MODEL`
- **Cloud-agnostic** : switcher vers Anthropic Claude, Mistral, Ollama sans changer le code

### Infrastructure locale (Docker)
| Service | Image | Port |
|---------|-------|------|
| PostgreSQL + pgvector | `pgvector/pgvector:pg16` | 5432 |
| Redis | `redis:7-alpine` | 6379 |
| MinIO (S3-compatible) | `minio/minio` | 9000 / 9001 |

### Frontend (TypeScript)
| Rôle | Technologie |
|------|-------------|
| Framework | Next.js 14 (App Router) |
| Styling | Tailwind CSS |
| Composants UI | shadcn/ui |
| Graphiques interactifs | Recharts |
| Graphiques statiques (v1) | Plotly — reporté après v0 |
| Auth | Clerk |
| Billing | Stripe |

---

## Architecture Multi-Agent

### Principe Fondamental
**Aucun agent ne passe le dataset brut à un autre agent.**
Chaque agent lit uniquement ce dont il a besoin depuis le `PipelineState` (State Store central).
Les données brutes restent dans le Blob Storage (MinIO/R2) — seuls des **pointeurs** et **agrégats** circulent.

### PipelineState (State Store LangGraph)
```python
class PipelineState(TypedDict):
    # Contexte
    tenant_id:        str
    user_id:          str
    report_id:        str
    prompt:           str

    # Références données (JAMAIS les données brutes)
    raw_data_refs:    list[str]     # pointeurs Blob Storage
    brand_kit:        dict          # logo, couleurs, typographie tenant

    # Enrichi progressivement par chaque agent
    metadata:         dict          # → Metadata Agent
    schema:           dict          # → Schema Linking Agent
    aggregates:       dict          # → Data Agent
    insights:         list[dict]    # → Insight Agent
    narrative:        str           # → Storytelling Agent
    viz_specs:        list[dict]    # → Viz Agent (specs JSON)
    qa_report:        dict          # → QA Agent
    report_urls:      dict          # → Layout Agent (html_url uniquement en v0)

    # HITL (Human-in-the-Loop)
    hitl_pending:     bool
    hitl_checkpoint:  Optional[str]   # cp1|cp2|cp3|cp4|cp5
    hitl_corrections: list[dict]      # historique corrections humaines

    # Pipeline meta
    status:           str           # pending|running|hitl_required|complete|error
    errors:           list[str]
    current_agent:    str
```

### Les 8 Agents

#### Agent 1 — `MetadataAgent`
- **Fichier** : `backend/app/agents/metadata_agent.py`
- **Rôle** : Analyser chaque fichier uploadé, inférer la sémantique des colonnes (nom, type, unité, description), construire le Data Dictionary par tenant.
- **Input** : `raw_data_refs` (pointeurs fichiers)
- **Output** : `state["metadata"]` — Data Dictionary enrichi avec scores de confiance
- **Modèle LLM** : `gpt-4o-mini` (économique)
- **HITL CP1** : déclenché si confiance colonne < 0.85
- **Cache** : Oui — le Data Dictionary est réutilisé sur tout rapport ultérieur du même dataset

#### Agent 2 — `SchemaLinkingAgent`
- **Fichier** : `backend/app/agents/schema_linking_agent.py`
- **Rôle** : Détecter les relations implicites entre plusieurs fichiers (clés étrangères, jointures) via analyse statistique d'intersection + confirmation LLM.
- **Input** : `raw_data_refs` + `state["metadata"]`
- **Output** : `state["schema"]` — relations, cardinalités, taux orphelins, alertes qualité
- **Modèle LLM** : `gpt-4o-mini`
- **HITL CP2** : toujours sur multi-fichiers + si taux orphelins > 5%
- **Cache** : Oui — schéma persisté par combinaison de datasets

#### Agent 3 — `DataAgent`
- **Fichier** : `backend/app/agents/data_agent.py`
- **Rôle** : Interpréter le prompt utilisateur, générer du code Python/pandas, exécuter les agrégations. Ne passe JAMAIS les données brutes au LLM — seulement le résultat agrégé (quelques centaines de lignes max).
- **Input** : `state["prompt"]` + `state["schema"]` + `state["metadata"]`
- **Output** : `state["aggregates"]` — DataFrames sérialisés en dicts
- **Modèle LLM** : `gpt-4o-mini` (génère du code, pas de raisonnement complexe)
- **HITL** : automatique, pas de HITL sauf anomalie volumétrie
- **Cache** : partiel (1h pour prompts similaires sur même dataset)

#### Agent 4 — `InsightAgent`
- **Fichier** : `backend/app/agents/insight_agent.py`
- **Rôle** : Analyser les agrégats, détecter tendances, anomalies, corrélations significatives. Produire des insights hiérarchisés avec score de confiance.
- **Input** : `state["aggregates"]` + `state["metadata"]`
- **Output** : `state["insights"]` — liste d'insights avec type, description, confiance
- **Modèle LLM** : `gpt-4o` (raisonnement analytique complexe)
- **HITL CP3** : si confiance insight < 0.80 OU anomalie détectée OU variation > ±20%

#### Agent 5 — `StorytellingAgent`
- **Fichier** : `backend/app/agents/storytelling_agent.py`
- **Rôle** : Construire la narration data-driven structurée. Respecte le ton (formel/neutre/synthétique) et la langue du brand_kit. Ne doit JAMAIS inventer de chiffres.
- **Input** : `state["insights"]` + `state["brand_kit"]` (ton, langue) + template narratif
- **Output** : `state["narrative"]` — texte structuré (résumé exécutif, analyse, alertes, recommandations)
- **Modèle LLM** : `gpt-4o` (qualité narrative)
- **HITL CP4** : optionnel selon plan tarifaire (obligatoire sur Business+)

#### Agent 6 — `VizAgent`
- **Fichier** : `backend/app/agents/viz_agent.py`
- **Rôle** : Sélectionner le type de graphique optimal pour chaque insight. Générer les **viz_spec JSON** — source de vérité unique utilisée par Recharts (frontend interactif). Export PDF reporté à v1.
- **Input** : `state["insights"]` + `state["aggregates"]` + `state["brand_kit"]` (couleurs)
- **Output** : `state["viz_specs"]` — liste de specs JSON (type, données, axes, couleurs, annotations)
- **Modèle LLM** : `gpt-4o-mini`
- **Règles chart** : bar→comparaison, line→temporel, pie→proportion (max 5 catégories), scatter→corrélation

#### Agent 7 — `QAAgent`
- **Fichier** : `backend/app/agents/qa_agent.py`
- **Rôle** : Vérifier la cohérence entre données sources, insights et narration. Détecter les hallucinations LLM et erreurs numériques. Pattern Constitutional AI.
- **Input** : rapport complet + `state["aggregates"]` (données sources)
- **Output** : `state["qa_report"]` — confidence_score (0-1) + liste issues + statut (ok/warning/error)
- **Modèle LLM** : `gpt-4o-mini`
- **HITL** : si confidence_score < 0.80 → HITL obligatoire avant publication

#### Agent 8 — `LayoutAgent`
- **Fichier** : `backend/app/agents/layout_agent.py`
- **Rôle** : Assembler narration + viz_specs + brand_kit dans un rendu HTML interactif. **V0 : HTML uniquement** — export PDF reporté à v1.
- **Input** : `state["narrative"]` + `state["viz_specs"]` + `state["aggregates"]` + `state["brand_kit"]`
- **Output** : `state["report_urls"]` — {html_url} stocké dans MinIO/R2
- **Outils** : Jinja2 (templates HTML), Recharts (rendu frontend interactif)
- **⚠️ V0 — exclus** : Plotly images PNG, WeasyPrint PDF, kaleido

---

## Structure Fichiers Backend

```
backend/
├── app/
│   ├── __init__.py
│   ├── main.py                        ← FastAPI app + routes
│   ├── config.py                      ← Settings via pydantic-settings
│   ├── dependencies.py                ← Auth, DB, tenant context
│   │
│   ├── api/
│   │   └── v1/
│   │       ├── __init__.py
│   │       ├── reports.py             ← POST /reports/generate, GET /reports/{id}
│   │       ├── datasets.py            ← POST /datasets/upload
│   │       ├── hitl.py                ← POST /reports/{id}/review
│   │       ├── brand.py               ← GET/PUT /settings/brand
│   │       └── health.py              ← GET /health
│   │
│   ├── agents/
│   │   ├── __init__.py
│   │   ├── base_agent.py              ← Classe abstraite (logging, error handling)
│   │   ├── metadata_agent.py
│   │   ├── schema_linking_agent.py
│   │   ├── data_agent.py
│   │   ├── insight_agent.py
│   │   ├── storytelling_agent.py
│   │   ├── viz_agent.py
│   │   ├── qa_agent.py
│   │   └── layout_agent.py
│   │
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── state.py                   ← PipelineState TypedDict
│   │   ├── graph.py                   ← LangGraph graph definition
│   │   ├── checkpoints.py             ← HITL interrupt/resume logic
│   │   └── router.py                  ← Conditional edges (HITL triggers)
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── llm.py                     ← LiteLLM wrapper (call_llm, call_llm_json)
│   │   ├── storage.py                 ← S3-compatible abstraction (MinIO/R2/S3)
│   │   ├── vector_store.py            ← pgvector operations
│   │   ├── cache.py                   ← Redis cache layer
│   │   ├── pdf_generator.py           ← WeasyPrint + Plotly (reporté v1 — ne pas créer en v0)
│   │   └── notification.py            ← Email/Slack HITL alerts
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── tenant.py                  ← Tenant, BrandKit
│   │   ├── user.py
│   │   ├── dataset.py                 ← Dataset, DataDictionary
│   │   ├── report.py                  ← Report, VizSpec
│   │   └── audit_log.py               ← HITLReview, AuditLog
│   │
│   └── workers/
│       ├── __init__.py
│       ├── celery_app.py
│       ├── pipeline_task.py           ← Task Celery : lancement pipeline
│       └── export_task.py             ← Task Celery : génération HTML async (PDF reporté v1)
│
├── migrations/                        ← Alembic
├── tests/
│   ├── unit/                          ← Tests par agent (mock LLM + storage)
│   ├── integration/                   ← Tests pipeline complet
│   └── fixtures/                      ← Datasets CSV de test
├── pyproject.toml
├── .env.example
└── Dockerfile
```

---

## Structure Fichiers Frontend

```
frontend/
├── app/                               ← Next.js 14 App Router
│   ├── (auth)/                        ← Pages auth (Clerk)
│   │   ├── sign-in/
│   │   └── sign-up/
│   ├── dashboard/                     ← Liste des rapports
│   │   └── page.tsx
│   ├── datasets/                      ← Upload + gestion fichiers
│   │   └── page.tsx
│   ├── report/
│   │   └── [id]/
│   │       ├── page.tsx               ← Viewer rapport
│   │       └── edit/page.tsx          ← Éditeur no-code
│   ├── review/
│   │   └── [id]/page.tsx              ← Interface HITL
│   ├── settings/
│   │   └── page.tsx                   ← Brand Kit + billing
│   ├── layout.tsx
│   └── page.tsx                       ← Landing page
│
├── components/
│   ├── ui/                            ← shadcn/ui base
│   ├── charts/
│   │   ├── DynamicChart.tsx           ← Recharts depuis viz_spec JSON
│   │   ├── ChartEditor.tsx            ← Panneau édition no-code (sans LLM)
│   │   └── ChartTypeSelector.tsx
│   ├── pipeline/
│   │   ├── PipelineStatus.tsx         ← Avancement temps réel (WebSocket)
│   │   └── AgentCard.tsx
│   ├── hitl/
│   │   ├── MetadataReview.tsx         ← CP1 : validation colonnes
│   │   ├── SchemaReview.tsx           ← CP2 : validation relations
│   │   └── InsightReview.tsx          ← CP3 : validation insights
│   ├── brand/
│   │   ├── ColorPicker.tsx
│   │   ├── LogoUploader.tsx
│   │   └── BrandPreview.tsx
│   └── layout/
│       ├── Sidebar.tsx
│       └── TopNav.tsx
│
├── lib/
│   ├── api.ts                         ← Fetch wrapper + auth headers
│   ├── ws.ts                          ← WebSocket pipeline status
│   └── viz-spec.ts                    ← Types viz_spec + helpers Recharts
│
├── types/
│   └── index.ts                       ← Types TypeScript partagés
│
└── package.json
```

---

## Variables d'Environnement

```env
# LLM
OPENAI_API_KEY=sk-...
LITELLM_DEFAULT_MODEL=gpt-4o
LITELLM_CHEAP_MODEL=gpt-4o-mini

# Base de données
DATABASE_URL=postgresql+asyncpg://narr8:narr8dev@localhost:5432/narr8
DATABASE_URL_SYNC=postgresql://narr8:narr8dev@localhost:5432/narr8

# Redis
REDIS_URL=redis://localhost:6379/0

# Storage (MinIO en dev, Cloudflare R2 en prod)
STORAGE_ENDPOINT=http://localhost:9000
STORAGE_KEY=minioadmin
STORAGE_SECRET=minioadmin
STORAGE_BUCKET=narr8-dev

# App
APP_ENV=development
SECRET_KEY=dev-secret-change-in-prod
DEBUG=true

# HITL seuils
HITL_CONFIDENCE_THRESHOLD=0.80
HITL_METADATA_CONFIDENCE_THRESHOLD=0.85
HITL_ORPHAN_RATE_THRESHOLD=0.05
```

---

## Commandes Essentielles

```bash
# ── Démarrer les services Docker ──────────────────────────────
docker compose up -d
docker compose ps
docker compose logs -f

# ── Backend ───────────────────────────────────────────────────
cd backend
source .venv/bin/activate           # activer env virtuel
uvicorn app.main:app --reload --port 8000

# ── Tests ─────────────────────────────────────────────────────
pytest tests/ -v                    # tous les tests
pytest tests/unit/ -v               # tests unitaires seulement
pytest tests/unit/test_metadata_agent.py -v  # un agent précis
pytest --cov=app tests/             # avec couverture

# ── Linter / Formatter ────────────────────────────────────────
ruff check .                        # vérifier
ruff format .                       # formater

# ── Base de données ───────────────────────────────────────────
alembic revision --autogenerate -m "description"
alembic upgrade head

# ── Frontend ──────────────────────────────────────────────────
cd frontend
npm install
npm run dev                         # http://localhost:3000
npm run build
npm run lint
```

---

## Conventions de Code

### Python
- **Style** : Ruff (PEP8 + imports triés)
- **Types** : Annotations obligatoires sur toutes les fonctions publiques
- **Async** : Tout le code I/O est `async/await`
- **Erreurs** : Toujours catcher dans `BaseAgent.__call__` — jamais dans les agents eux-mêmes
- **LLM calls** : Utiliser exclusivement `app.services.llm.call_llm` et `call_llm_json` — jamais directement OpenAI ou LangChain
- **Storage** : Utiliser exclusivement `app.services.storage` — jamais boto3 directement
- **Config** : Utiliser exclusivement `app.config.settings` — jamais `os.environ` directement

### Nommage
```python
# Agents : NomAgent (PascalCase)
class MetadataAgent(BaseAgent): ...

# Services : fonctions snake_case
async def call_llm_json(prompt: str) -> dict: ...

# State keys : snake_case
state["raw_data_refs"]
state["viz_specs"]

# Routes API : kebab-case
POST /api/v1/reports/generate
GET  /api/v1/reports/{report_id}
```

### TypeScript (Frontend)
- **Style** : ESLint + Prettier
- **Types** : Strict mode, pas de `any`
- **Components** : PascalCase, un fichier = un composant
- **API calls** : Toujours via `lib/api.ts`

---

## Patterns Importants

### Appel LLM avec JSON garanti
```python
from app.services.llm import call_llm_json

result = await call_llm_json(
    prompt="...",
    system="Retourne UNIQUEMENT du JSON valide.",
    model=settings.litellm_cheap_model,
)
# result est toujours un dict Python parsé
```

### Lire un fichier depuis le storage
```python
from app.services.storage import read_dataframe

df = await read_dataframe(state["raw_data_refs"][0])
# df est un pandas DataFrame
```

### Déclencher le HITL
```python
# Dans n'importe quel agent :
if condition_requiring_human_review:
    state["hitl_pending"]    = True
    state["hitl_checkpoint"] = "cp1_metadata"  # ou cp2, cp3, cp4, cp5
    return state  # Le pipeline s'arrête ici, attend validation humaine
```

### viz_spec JSON (format standard)
```json
{
  "chart_type": "bar",
  "title": "CA par région — Q3 2024",
  "data_key": "ca_by_region",
  "x": "region",
  "y": "ca_ht",
  "color_by": "performance_vs_budget",
  "colors": {
    "primary": "#1E3A8A",
    "positive": "#16A34A",
    "negative": "#DC2626"
  },
  "annotations": [
    {"x": "Nord", "text": "▲ +23% vs budget"}
  ]
}
```

---

## Ordre de Développement (Sprints)

```
Sprint 0  ✅  Setup environnement (Docker, venv, FastAPI health)
Sprint 1  →   Pipeline squelette LangGraph (tous les agents vides)
Sprint 2  →   MetadataAgent complet + testé
Sprint 3  →   SchemaLinkingAgent complet + testé
Sprint 4  →   DataAgent complet + testé
Sprint 5  →   InsightAgent + StorytellingAgent
Sprint 6  →   VizAgent + QAAgent
Sprint 7  →   LayoutAgent + export HTML interactif (v0) — PDF reporté v1
Sprint 8  →   FastAPI endpoints + WebSocket pipeline status
Sprint 9  →   HITL endpoints + workflow review
Sprint 10 →   Frontend Next.js (Studio no-code, Dashboard, HITL UI)
Sprint 11 →   Auth (Clerk) + Billing (Stripe) + Multi-tenant
Sprint 12 →   Tests E2E + optimisation coûts + déploiement
```

**Règle absolue** : à la fin de chaque sprint, quelque chose tourne de bout en bout et les tests passent.

---

## Tests

### Structure des tests unitaires d'un agent
```python
# tests/unit/test_{nom}_agent.py

@pytest.mark.asyncio
async def test_{agent}_nominal_case():
    """Cas nominal : l'agent produit le bon output."""
    agent = NomAgent()
    with patch("app.agents.nom_agent.read_dataframe", AsyncMock(...)), \
         patch("app.agents.nom_agent.call_llm_json", AsyncMock(...)):
        result = await agent.run(initial_state)
    assert result["expected_key"] == expected_value

@pytest.mark.asyncio
async def test_{agent}_triggers_hitl():
    """L'agent déclenche le HITL dans les bonnes conditions."""
    ...
    assert result["hitl_pending"] == True
    assert result["hitl_checkpoint"] == "cpX_..."

@pytest.mark.asyncio
async def test_{agent}_handles_error():
    """L'agent gère les erreurs sans planter le pipeline."""
    ...
    assert result["status"] == "error"
    assert len(result["errors"]) > 0
```

---

## Périmètre V0 (MVP strict)

```
✅ INCLUS en v0                        ❌ REPORTÉ à v1+
──────────────────────────────         ──────────────────────────────
Pipeline 8 agents complet              Export PDF (bouton sur rendu HTML)
Upload CSV / Excel                     Intégration Power BI (Mode 2 + 3)
Connexion SQL basique                  Connecteur champ sémantique PBI
Interface HITL (CP1, CP2, CP3)         Google Sheets, Snowflake
Export HTML interactif (Recharts)      White-label agences
Brand Kit (couleurs, logo, ton)        Fine-tuning sectoriel
Auth multi-tenant (Clerk)              API publique documentée
Billing Stripe (3 plans)               SSO SAML Enterprise
Dashboard rapports                     Intégration Tableau / Qlik
```

**Librairies exclues de v0** (ne pas installer, ne pas importer) :
- `weasyprint` — génération PDF
- `kaleido` — export Plotly → PNG
- `plotly` — utilisé uniquement pour PDF (le rendu interactif est Recharts côté frontend)

### Export PDF — Bouton v1+
```
Rendu HTML affiché dans l'interface
              │
   ┌──────────────────────┐
   │  📄 Exporter en PDF  │  ← bouton visible uniquement en v1+
   └──────────────────────┘
              │
              ▼
   POST /api/v1/reports/{id}/export/pdf
   Génération WeasyPrint côté serveur
   depuis le HTML déjà stocké en MinIO
              │
              ▼
   Téléchargement direct navigateur
```
Implémentation v1+ : `backend/app/services/pdf_generator.py`

---

## Intégration Power BI (v1+)

**BrouxAI est autonome — Power BI est une destination optionnelle.**
Le rendu final de BrouxAI est toujours son propre HTML interactif.
Power BI n'est jamais le moteur de rendu de BrouxAI.

### Les 2 modes d'intégration retenus (v1+)

```
Mode 1 — Embedded Visual (v1+)
────────────────────────────────
Le rapport HTML BrouxAI est affiché
dans un iframe Power BI Embedded

Cas d'usage : intégrer la narration
BrouxAI dans un rapport Power BI
existant comme un visuel custom
```

```
Mode 2 — Bouton dans Power BI Desktop (v1+)
────────────────────────────────────────────
Extension custom dans Power BI Desktop
Un bouton appelle l'API BrouxAI
avec les données du rapport ouvert
       ↓
BrouxAI génère la narration HTML
       ↓
Résultat affiché dans BrouxAI ou injecté

Cas d'usage : analyste BI qui génère
la narration sans quitter Power BI
```

**Mode supprimé :**
- ~~Mode Push Dataset API~~ — pousser de la donnée depuis BrouxAI vers Power BI n'est pas pertinent. BrouxAI consomme de la donnée, il ne l'alimente pas.

### Connecteur Champ Sémantique Power BI (v1+)
```
Connexion OAuth2 au workspace Power BI
       ↓
Import automatique via Power BI REST API :
├── Tables + colonnes (métadonnées déjà définies)
├── Relations entre tables (schéma déjà connu)
├── Mesures DAX ([CA HT], [Marge], [Churn%]...)
└── Hiérarchies (Année > Trimestre > Mois)
       ↓
Metadata Agent + Schema Agent → mode "validation"
(import au lieu d'inférence → HITL réduit à ~1 min)
Data Agent → génère du DAX au lieu de pandas
```

---

**Nom commercial** : BrouxAI
**Positionnement** : Painkiller pour contrôleurs de gestion, DAF en PME et consultants BI
**Pain résolu** : 2-3 jours/mois perdus à rédiger manuellement des commentaires de gestion
**Cibles MVP** : PME/ETI (équipes data internes) + Agences de conseil BI
**Modèle économique** : Abonnement mensuel
  - Starter  : 99€/mois — 20 rapports, 1 user, CSV/Excel
  - Business : 299€/mois — 100 rapports, 5 users, + SQL + Power BI
  - Agency   : 799€/mois — illimité, 20 workspaces clients, white-label
  - Enterprise : sur devis — SSO, tenant dédié, fine-tuning

**Intégration Power BI** : push dataset API + embedded visuels (Phase 2)