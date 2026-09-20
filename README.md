# WasteWise AI

**Predict demand. Reduce waste. Optimize every meal.**

A working AI decision-support platform for restaurant operations, built
as a controlled Pre-Sales PoC. Answers one specific business question:

> **Can historical restaurant sales data accurately forecast near-term
> item-level demand sufficiently well to support better food-preparation
> decisions?**

Every number this product shows is either computed live by a real
trained model, or explicitly labeled as an illustrative assumption.
Nothing is hardcoded to look impressive — see "Real bugs found and
fixed," below, for what that discipline actually caught.

📄 **Start here**: `docs/POC_CHARTER.md` (scope, success criteria) and
`docs/ARCHITECTURE.md` (design decisions and why).

## Quick start (local, no Docker)

Requires Python 3.11+.

```bash
pip install -r backend/requirements.txt

# 1. Generate data, validate/clean/engineer features, train, register, drift-check
python ml/data/generate_dataset.py
python ml/preprocessing/run_pipeline.py
python ml/training/train_models.py
python ml/monitoring/run_mlops_pipeline.py

# 2. Start the API
cd backend && uvicorn app.main:app --reload --port 8001
# -> http://127.0.0.1:8001/docs
```

In a second terminal:

```bash
cd frontend && python3 -m http.server 5173
# -> open http://127.0.0.1:5173
```

**To enable a real LLM** (rather than the honest, deterministic template
fallback): `export LLM_PROVIDER=groq` (or `openai`/`anthropic`/`ollama`)
and the matching `*_API_KEY` before starting the backend.

## Quick start (Docker)

```bash
docker compose up --build
# backend  -> http://localhost:8001
# frontend -> http://localhost:5173
```

First boot trains the full pipeline automatically (a few minutes);
subsequent restarts with the same volumes skip straight to serving. **Note:**
the Docker Compose setup was written and its entrypoint logic verified by
running the exact same script directly (not `docker build`/`docker
compose up`) — no Docker daemon was available in the environment this was
built in. The Dockerfiles and compose file follow the same patterns
already proven to work in the non-Docker quick start above, but a real
`docker compose up` run has not been observed end-to-end. Flagging this
plainly rather than claiming a test that didn't happen.

## Running the tests

```bash
python -m pytest tests/ -v
```

**167 tests, all passing** — including two that actually matter most:
`tests/test_integration.py` walks the full PREDICT → DETECT → EXPLAIN →
SIMULATE → RECOMMEND → DECIDE → MEASURE loop as **one continuous flow**
through the live API, asserting the same forecast number stays identical
across every step, rather than testing each piece in isolation and hoping
they agree in production. `tests/test_failure_scenarios.py` (Phase 13)
targets the failure modes Section 35 asks for specifically — LLM
timeout, model-file-missing, corrupted registry, invalid/extreme what-if
parameters, concurrent requests, and cost-threshold alerting — and found
one genuine bug in the process (below).

**Reproducibility, verified, not assumed**: the entire pipeline (data
generation → training → MLOps) was re-run from a completely clean
directory (no pre-existing data/models/anything) during Phase 12 and
produced byte-identical metrics to every prior incremental run — proving
the `seed=42` reproducibility claim rather than just stating it.

## What's built (phases 1-12)

| Phase | What | Docs |
|---|---|---|
| 1 | Requirements & architecture | `docs/ARCHITECTURE.md` |
| 2 | Synthetic dataset (49K rows, real seasonality/censoring) | `docs/DATASET.md` |
| 3 | Validation, cleaning, leak-safe feature engineering | (tested in `tests/test_preprocessing.py`) |
| 4 | Champion/challenger ML (baseline, RF, GB, XGBoost) | `docs/ML_PIPELINE.md` |
| 5 | MLOps: registry, promotion policy, drift detection | `docs/MLOPS.md` |
| 6 | Risk & recommendation engine (deterministic) | `docs/ML_PIPELINE.md` (Phase 6 section) |
| 7 | LLM copilot + What-If simulator | `docs/LLMOPS.md` |
| 8 | LLMOps: request logging, prompt eval suite | `docs/LLMOPS.md` |
| 9 | AI cost + latency monitoring | `docs/MONITORING.md` |
| 10 | FastAPI backend (full REST API) | `docs/API.md` |
| 11 | Frontend (13 pages, zero-build HTML/JS/Tailwind) | `docs/FRONTEND.md` |
| 12 | Integration: end-to-end test, reproducibility, Docker | this file |
| 13 | Targeted failure-scenario testing (Section 35) | `tests/test_failure_scenarios.py` |

## Real bugs found and fixed along the way (not smoothed over)

This project surfaced more real, substantive bugs than most — each
caught by actually running the code and checking the output, not by
inspection. A representative sample, full details in each phase's docs:

- **Dataset calibration**: first-pass stockout rate was 42% (unrealistic) — recalibrated to 17%
- **Champion selection**: two different, both-defensible "champions" (Gradient Boosting from an ad hoc score, XGBoost from the formal promotion policy) — reconciled explicitly, not hidden
- **Drift false alarm**: naive train-vs-test comparison flagged `temperature` at PSI 4.20 ("critical") purely from calendar-window mismatch — fixed with a season-matched comparison
- **Risk-engine miscalibration**: 89/90 items showed "conflicting" HIGH shortage + HIGH waste simultaneously due to a structural threshold coupling — found, explained mathematically, fixed
- **Cross-process non-determinism**: a "reproducible" simulated inventory feed used Python's randomized `hash()`, silently drifting between separate runs — fixed and verified with a real subprocess test
- **Hallucination-checker false positives**: the LLM eval framework itself flagged real, grounded numbers (from prose and dict keys) as "hallucinated" — fixed, verified down to 0.0%
- **Cold-start latency**: 14x first-request penalty from lazy-loaded artifacts, found in monitoring, fixed proactively in the API's startup lifespan
- **A copy-paste import bug** that prevented the API from booting at all, caught immediately and fixed with a proper reconciliation of two data sources rather than a quick patch
- **Unclipped negative what-if forecast**: an extreme negative `demand_pct_change` (e.g. -500%) produced a forecast of -372.6 portions — the demand-override arithmetic wasn't clipped to zero the way every other prediction path already was. Found by Phase 13's failure-scenario testing, fixed with a one-line `max(..., 0)`.

## Limitations (stated, not discovered later)

See `docs/POC_CHARTER.md`'s Out-of-Scope section for the full list.
Briefly: no production auth/RBAC, no live POS/inventory integration
(inventory is a documented, labeled simulation), no real cloud billing
(cost figures beyond LLM spend are illustrative), and — as noted above —
the Docker path is written and logic-verified but not build-tested.
