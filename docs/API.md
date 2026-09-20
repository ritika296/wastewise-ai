# WasteWise AI — Backend API (Phase 10)

Run: `cd backend && uvicorn app.main:app --reload --port 8001` (interactive
docs at `/docs`). Every module built in Phases 2-9 is wired in directly —
nothing here is re-implemented, only exposed over HTTP.

## Startup warm-up — built in from the start this time

Phase 9 measured a 14x cold-start penalty (132ms → 9ms) from lazy-loading
the model and parquet files on first use — the same pattern found earlier
in the RetainIQ project's SHAP explainer. This backend's `lifespan`
handler warms the shared artifacts (model, features, volatility stats)
and the recommendation path **before accepting any request**:

```
[startup] Warmed shared artifacts in 160-172ms
```

No user ever pays that cost — it's absorbed once, at boot.

## Endpoints (Section 28)

| Endpoint | Notes |
|---|---|
| `GET /dashboard?date=` | Real KPIs from the live recommendation engine |
| `GET /forecast?date=&restaurant_id=` | Item-level forecast list |
| `GET /forecast/{item_id}?date=&restaurant_id=` | Single-item forecast + real XGBoost TreeSHAP explanation |
| `GET /risk?date=&restaurant_id=&tier=` | Shortage/waste risk classification |
| `GET /recommendations?date=&restaurant_id=` | Full recommendation table |
| `POST /what-if` | Real model re-inference + deterministic recalculation |
| `POST /copilot/chat` | Grounded LLM copilot (logs every call) |
| `GET /mlops/models` | Candidates + **both** champions, reconciled (see below) |
| `GET /mlops/drift?simulated=` | Season-matched real drift, or labeled simulated scenario |
| `GET /llmops/metrics` | Real logged request metrics |
| `GET /prompts?date=` | v1/v2/v3 comparison |
| `GET /cost` / `GET /cost/tiers` | Cost dashboard + tier trade-off table |
| `GET /latency?n_requests=&include_llm=` | Real, on-demand latency benchmark |
| `GET /data-quality` | Real validation report on the raw dataset |
| `POST /feedback` / `GET /feedback` | Human-in-the-loop, persisted (SQLite) |

## A real bug caught here: two different "champions"

`GET /mlops/models` surfaces **both** Phase 4's naive weighted-score pick
(Gradient Boosting) and Phase 5's formal, threshold-gated registry
champion (XGBoost) explicitly — this endpoint was originally written to
read a function (`load_registry()`) that didn't exist on the Phase 5
registry module at all (a copy-paste import error from a different
project's module shape), caught immediately by the server failing to
boot. Fixed by reading each source correctly and stating the reconciliation
in the response itself (`"note"` field), rather than picking one silently.

## Error handling (Section 34)

A custom `ValueError` handler converts internal "no data for this
date/item" errors into clean `404` responses — verified no endpoint ever
leaks a Python traceback (`test_no_endpoint_ever_returns_raw_stack_trace`).
Pydantic schema validation handles malformed requests automatically
(`422`, with field-level detail) — verified for missing required fields,
an out-of-range `human_decision` value, and a too-short copilot question.

## What Phase 10 does not include

Authentication/RBAC (Section 30 — explicitly deferred to "what would be
required before production," documented in the companion pre-sales
material), rate limiting, a real production ASGI deployment (Gunicorn +
multiple Uvicorn workers) — this is a `--reload` dev server appropriate
for a PoC demo, not a production process manager.
