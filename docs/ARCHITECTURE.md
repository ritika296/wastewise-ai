# WasteWise AI — Architecture (Phase 1)

## Core flow (unchanged from the brief)

```
Historical Sales Data → Data Quality & Validation → Feature Engineering
→ Demand Forecasting ML → Waste/Shortage Risk Detection
→ Recommendation/Optimization Engine → LLM Business Copilot
→ Manager Decision → Preparation/Ordering Action → Business Outcome
```

## Personas & what each can see/do

| Persona | Primary use |
|---|---|
| **Restaurant Operations Manager** (primary) | Dashboard, Forecast, Risk, Recommendations, Item Detail, Copilot, What-If — the full daily decision loop |
| Regional Operations Manager | Executive Dashboard across locations, Business Impact |
| Finance Manager | Business Impact / ROI Calculator, Cost Monitoring |
| Data/ML Ops Team | MLOps, LLMOps, Data Quality, Latency/Observability |
| Business Executive | Executive Dashboard, Business Impact (summary level only) |

## Technology choices, and why

| Layer | Choice | Rationale |
|---|---|---|
| **ML** | scikit-learn (Random Forest, Gradient Boosting) + XGBoost, baseline = seasonal naive/moving average | Matches the brief exactly; these are the right tool for tabular, moderate-size, interpretable demand data — no justification needed to deviate |
| **MLOps tracking** | **Real MLflow**, local file-store backend (`./mlruns`, no tracking server to stand up) | The brief says "use MLflow if practical" — for a local PoC, MLflow's file-store mode has zero server overhead (`mlflow.set_tracking_uri("file:./mlruns")`) and gives genuine experiment tracking, not a simulation of it. This is a deliberate difference from a lighter JSON-only registry: it's practical *here* because there's no deployment cost to it. |
| **Model registry (serving)** | A thin JSON pointer (`champion_model_version`) read by the API, backed by MLflow's own run history as the source of truth | The API shouldn't have a live dependency on the MLflow tracking store for every prediction; it reads a small, fast pointer file that Phase 5's promotion logic writes after evaluation |
| **Backend** | FastAPI + SQLite (SQLAlchemy models, Postgres-DSN-compatible) | Matches the brief's own "PostgreSQL or SQLite for PoC" option; SQLite keeps the PoC runnable with zero external services, and the ORM layer makes a later Postgres swap a one-line config change |
| **LLM provider** | A provider-abstraction interface (`llm/client.py`) supporting OpenAI, Groq, or Anthropic via one config var, **plus a deterministic template fallback** so the PoC runs and is fully testable with zero API key | Directly matches Section 25's "must allow OpenAI / Groq / Ollama / or another compatible provider, do not hard-code" — Ollama support is included as a local-inference option for a zero-cost demo path |
| **Frontend — proposed deviation, flagged for your decision** | Dependency-free HTML/JS + Tailwind (no build step) rather than a React/Next.js project, *unless you'd rather I build the React/Next.js version as originally specified* | Reasoning below |

### On the frontend choice specifically

The brief asks for React/Next.js. My default recommendation for a PoC this size is a zero-build-step HTML/Tailwind/vanilla-JS frontend instead, for one reason: it removes an entire category of setup friction (Node version, `npm install`, build tooling, dev server config) for a grader or client stakeholder who just wants to open the product and click through it — and the API contract (`api.js`, a thin fetch wrapper) is framework-agnostic, so migrating to React/Next.js for a Pilot/MVP later is a frontend-only rewrite with zero backend impact.

This is a real trade-off, not a cost-cutting shortcut dressed up as one: you lose component reusability and React's ecosystem (form libraries, state management) that would matter at Pilot/MVP scale, and you should weigh that against the setup-friction savings. **Tell me which you'd prefer before Phase 11 (frontend build)** — I can build either; I'm flagging the decision now rather than making it silently, since last time I made this same call by default and want it to be your choice this time, not a repeated assumption.

## Folder structure (scaffolded, empty — filled in phase by phase)

```
wastewise-ai/
├── frontend/
├── backend/
├── ml/
│   ├── data/            Phase 2 — synthetic data generator
│   ├── preprocessing/   Phase 3 — validation & cleaning
│   ├── features/        Phase 3 — feature engineering
│   ├── training/        Phase 4 — model training + MLflow logging
│   ├── evaluation/       Phase 4 — metrics, model comparison
│   ├── inference/        Phase 6 — serving-time prediction + risk + recommendation
│   └── monitoring/       Phase 5 — drift detection
├── llm/
│   ├── prompts/          Phase 7 — versioned prompt templates
│   ├── chains/           Phase 7 — question → context → LLM pipeline
│   ├── evaluation/       Phase 8 — grounding/quality eval suite
│   └── monitoring/       Phase 8 — cost/latency/quality logging
├── data/
│   ├── raw/              Phase 2 output
│   └── processed/        Phase 3 output
├── models/                Phase 4 — serialized model artifacts
├── monitoring/            Phase 9 — cost + latency engines (cross-cutting)
├── notebooks/             optional, exploratory only — not part of the served product
├── tests/                 Phase 13, but tests are added as each phase ships, not deferred
├── docs/
│   ├── POC_CHARTER.md     done (this phase)
│   ├── ARCHITECTURE.md    this file
│   ├── DATASET.md         Phase 2
│   ├── ML_PIPELINE.md     Phase 4
│   └── LLMOPS.md          Phase 8
├── .env.example
├── requirements.txt
├── README.md
└── docker-compose.yml
```

## Database design (tables, per Section 29 — implemented starting Phase 10)

`restaurants`, `menu_items`, `sales`, `inventory`, `promotions`, `weather`,
`forecasts`, `risk_predictions`, `recommendations`, `model_versions`,
`model_metrics`, `llm_requests`, `llm_evaluations`, `cost_metrics`,
`latency_metrics`, `data_quality_metrics` — all timestamped and
version-tagged, per the brief. Defined as SQLAlchemy models in Phase 10;
not created early since they'd sit empty until the pipeline exists to fill
them.

## What Phase 1 deliberately does not include

No code beyond folder structure yet — no dataset, no models, no API. Per
your instruction to work in phases and not skip foundational steps, the
charter and architecture are the Phase 1 deliverable; everything else is
built and *tested* in its own phase before the next begins.
