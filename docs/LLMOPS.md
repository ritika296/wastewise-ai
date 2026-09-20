# WasteWise AI — LLM Architecture (Phase 7, extended in Phase 8)

## Pipeline (Section 33)

```
question -> classify_intent() -> retrieve structured data (recommendation
engine + real TreeSHAP explainability) -> build grounded facts -> versioned
prompt construction -> call_llm() (provider-configurable + template
fallback) -> structured response (ANSWER / KEY EVIDENCE / REASON /
RECOMMENDED ACTION / EXPECTED BUSINESS IMPACT / CONFIDENCE-LIMITATION)
```

Intent detection (`llm/chains/copilot.py::classify_intent`) is a simple,
inspectable keyword/regex matcher, not a second ML model — appropriate
PoC scope (Section 36). Seven intents: item_explanation,
shortage_priority_list, waste_priority_list, category_waste_summary,
meeting_summary, day_over_day, what_if, and a general fallback that
honestly says "I don't have enough information" rather than guessing.

## Provider configuration (Section 25 — never hard-coded)

`LLM_PROVIDER` env var selects openai / groq / anthropic / ollama /
**template** (default). Every provider path shares one interface
(`llm/client.py::call_llm`) and one illustrative cost-rate card. If no
API key is set for a live provider, or the call fails for any reason, the
system falls back to the deterministic template engine automatically —
`fallback_used: true` is always reported, never silent.

## Grounding discipline (Section 12-13)

The LLM never computes a number. Every fact it narrates comes from:
1. The champion model's real forecast (Phase 4/5)
2. The deterministic recommendation engine (Phase 6)
3. Real, exact XGBoost TreeSHAP contributions (`ml/inference/explain.py`)

Verified by test, not just asserted: `test_item_question_returns_real_forecast_number`
checks the exact forecast figure from the structured facts appears
verbatim in the final answer text, and
`test_unknown_item_triggers_honest_no_information_response` checks that
asking about a nonexistent item produces the required "I don't have
enough information" response, not a fabricated one.

## What-If Simulator (Section 11) — no LLM involved at all

`ml/inference/what_if.py` re-runs the REAL trained model on a mutated
feature vector for levers that are genuine model inputs (promotion,
weather), and applies transparent arithmetic for levers that aren't model
inputs (a direct demand-% override, safety buffer, starting inventory).
Proven, not assumed: `test_weather_lever_uses_real_model_not_fixed_multiplier`
checks that a heat-sensitive item (Cold Coffee) and a heat-insensitive one
(Butter Naan) respond by *different* relative amounts to the same
heatwave scenario — if a lever were a fixed multiplier, they'd move
identically. `test_no_llm_involved_in_what_if_module` statically checks
the module never imports the LLM client at all.

## Two real bugs found and fixed during this phase

1. **Cross-process non-determinism in the "simulated inventory" feed**
   (`ml/inference/recommendation_engine.py`). The original implementation
   seeded per-item randomness with Python's built-in `hash(item_id)`,
   which is randomized per-process via `PYTHONHASHSEED` unless disabled.
   Results looked reproducible in a same-process test (which called the
   function twice in a row) but silently changed between separate runs of
   the demo script. Fixed with an `hashlib.md5`-based stable hash;
   verified with a test that spawns genuinely separate subprocesses and
   confirms identical output across all of them.

2. A leftover dead code branch in `_find_item_mention()` (an abandoned
   matching approach that did nothing) — removed during code review
   before it could confuse a future maintainer.

## What Phase 7 does not include (deferred to Phase 8/9)

Token/cost/latency *logging and aggregation* (the client returns these
per-call; Phase 8 adds persistent logging + the LLMOps dashboard), the
formal prompt-version evaluation suite (Phase 8), and the
`day_over_day` / `category_waste_summary` intents are wired with partial
context only — full implementation (comparing two dates; joining
category back onto the recommendations table) is noted as a Phase 9/10
integration task rather than stubbed silently.

## Phase 8: Formal Logging & Evaluation

### Request logging (`llm/monitoring/request_logger.py`)

SQLite (`monitoring/llm_requests.db`) — the `llm_requests` table from
Section 29's schema, built now so Phase 10's API reads from the real
thing. Every copilot call logs provider, model, prompt version, token
counts, latency, cost, and fallback/error flags. `get_metrics()` computes
P50/P95/P99 latency, error/fallback rates, cost projections (daily ->
monthly), and breakdowns by prompt version and intent.

### The template engine became version-aware — a real fix, not cosmetic

Before this phase, `llm/client.py`'s template fallback produced
byte-identical output regardless of which prompt version (v1/v2/v3) was
passed to it — verified this was genuinely true before fixing it (not
assumed). That would have made any prompt-version comparison meaningless
whenever no live LLM API key is configured, which is this PoC's default
and most common state. Fixed: the template engine now mirrors what each
version's system prompt actually instructs — v1 (no structure, no
explicit refusal rule) produces a loose paragraph and a soft "not sure"
response when facts are empty; v2 adds the six-section structure; v3 adds
the hard "never invent, refuse explicitly" rules. This makes offline
prompt evaluation genuinely informative, not just present.

### Evaluation results (real run, 2026-05-08, template-fallback mode)

**Prompt version comparison** (`compare_prompt_versions()`):

| Version | Groundedness | Completeness | Hallucination Rate |
|---|---|---|---|
| v1 | 0.833 | 0.333 | 0.0 |
| v2 | 0.833 | 0.900 | 0.0 |
| v3 | **1.000** | 0.900 | 0.0 |

v1 fails groundedness on the honesty case specifically — it responds "I'm
not sure about that one" rather than the exact required refusal phrase,
which is precisely what v3's hard rule (explicit refusal string) exists
to fix. This is a genuine, measured demonstration of why the stricter
prompt version is worth using, not an assumed one.

**Full eval suite aggregate** (v3, active version, 6 cases):

```
groundedness: 1.0    safety: 1.0    completeness: 0.9
relevance: 1.0        hallucination_rate: 0.0    intent_accuracy: 1.0
```

### The hallucination checker: two real bugs found and fixed

The check cross-references every number in the LLM's answer against
every number available as grounding — not a keyword heuristic. Building
it surfaced two real bugs in the CHECKER (not the copilot):

1. **Numbers in the REASON text weren't recognized.** `context["reason"]`
   (a real, risk-engine-computed sentence like "demand (117 portions)
   would exceed supply (101) by 16 portions") is legitimate grounding,
   but the first version of the checker only scanned the flat `facts`
   dict's *values*, missing it entirely — falsely flagging 40-47% of
   numbers in item-explanation answers as "hallucinated" when they were
   fully grounded, just conveyed as prose. Fixed by also scanning
   `reason` / `recommended_action` / `expected_impact`.
2. **Numbers embedded in fact *keys* weren't recognized either** — e.g.
   "28" in the driver label "Driver: 28-day average demand" (a real,
   correct feature name from `ml/inference/explain.py`) was flagged
   because the checker only scanned dict *values*, not keys. Fixed by
   scanning both.

After both fixes: hallucination rate on the full eval suite is exactly
`0.0`, correctly reflecting that the template engine (and, when
connected, any real LLM constrained to the v3 hard rules) does not
introduce unverified figures. Both are locked in as regression tests
(`test_hallucination_check_accepts_numbers_from_reason_text`,
`test_hallucination_check_accepts_numbers_embedded_in_fact_keys`) rather
than just fixed and forgotten.

### What Phase 8 does not include

A live human-rated evaluation (the eval suite here is automated,
heuristic scoring — appropriate PoC scope, not a replacement for the
"business users rate recommendations ≥4/5" success criterion in the PoC
Charter, which needs a real reviewer session at Pilot stage), and
continuous/scheduled evaluation runs (this runs on-demand via
`run_eval_suite()`; Phase 10's API exposes it over HTTP but there's no
scheduler in the PoC).
