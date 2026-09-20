# WasteWise AI — Cost & Latency Monitoring (Phase 9)

## Latency Engine (`monitoring/latency_engine.py`)

Every number is measured by actually running the pipeline with
`time.perf_counter()` — not invented. Stage breakdown per Section 16:
Data Retrieval -> Feature Processing -> ML Prediction -> Recommendation
-> (LLM, tracked separately).

### A cold-start finding, same pattern as the RetainIQ project, found and handled the same way

The first request in a fresh process took **132ms**; every subsequent
request took **~9ms** — a 14x difference, entirely from
`_load_shared_artifacts()` loading the model and parquet files once (it's
`@lru_cache`d). This is real, measured behavior, not a bug to hide:

```
call 1: data_retrieval=124.0ms total=132.2ms   <- cold
call 2: data_retrieval=1.8ms  total=9.9ms      <- warm
call 3: data_retrieval=1.6ms  total=9.2ms
call 4: data_retrieval=1.5ms  total=9.1ms
call 5: data_retrieval=1.5ms  total=9.1ms
```

**Implication for Phase 10**: the FastAPI backend must warm the shared
artifacts at startup (the same fix applied to RetainIQ's SHAP explainer),
not on the first real user request — otherwise the first user of the day
pays this cost, not a load-tester who conveniently warms it first.

### Real benchmark (30 requests, ML path only, no LLM)

| Stage | Mean | P50 | P95 | P99 |
|---|---|---|---|---|
| Data retrieval | 1.48ms | 1.47ms | 1.60ms | 1.65ms |
| Feature processing | 0.68ms | 0.67ms | 0.76ms | 0.82ms |
| ML prediction | 6.28ms | 6.23ms | 6.56ms | 7.14ms |
| Recommendation | 0.98ms | 0.96ms | 1.08ms | 1.18ms |
| **Total** | **9.43ms** | **9.37ms** | **9.76ms** | **10.50ms** |

**P95 total (9.76ms) is ~200x under the 2-second SLA** — expected for an
in-process, no-network ML path; this comfortable margin is exactly why
the LLM stage (which legitimately runs in the hundreds-of-milliseconds-
to-seconds range for a live API call) is tracked against its own budget,
not blended into the same SLA.

## Cost Engine (`monitoring/cost_engine.py`)

| Component | Status |
|---|---|
| LLM cost | **REAL** — pulled from `llm/monitoring/request_logger.py`'s logged calls |
| Embedding cost | **N/A** — WasteWise AI has no RAG/embedding layer (unlike a document-grounded copilot); stated explicitly, not silently reported as a suspicious zero |
| ML inference cost | **Illustrative** — real measured inference time x a documented assumed compute rate (₹0.008/CPU-second) |
| Infrastructure cost | **Illustrative** — a documented flat monthly assumption (₹4,500/month) amortized over an assumed request volume |

Every illustrative figure carries a `"basis"` field saying so, and the
dashboard's top-level `"note"` states plainly: *"LLM cost is REAL... ML
inference and infrastructure costs are ILLUSTRATIVE... not measured cloud
billing, since this PoC runs locally with no cloud account attached."*

### LLM tier comparison (Section 28)

Four tiers compared on real cost-rate-card figures (the same rates
`llm/client.py` actually uses) and latency, with unmeasured live-provider
latencies explicitly labeled `"PUBLISHED TYPICAL — not measured in this
environment"` (no API keys are configured to measure them directly) versus the
template tier's genuinely `"MEASURED"` figure. The comparison deliberately
does **not** declare a universal winner — Section 28 explicitly warns
against defaulting to the largest model, and the recommendation here is
scoped to WasteWise AI's actual question mix (mostly structured,
template-answerable) rather than presented as a general ranking.

## What Phase 9 does not include

Real cloud billing integration (no cloud account is attached to this PoC
— see the cost engine's repeated "illustrative" labeling), a scheduled/
continuous benchmark runner (this runs on-demand; Phase 10's API exposes
it over HTTP), and multi-tenant cost attribution (documented as a
Pilot/MVP concern in the cost dashboard's `cost_per_user` assumption
note).
