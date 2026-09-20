# WasteWise AI — PoC Charter

**"Predict demand. Reduce waste. Optimize every meal."**

## Problem

Restaurant managers decide how much of each menu item to prepare using
historical intuition and manual estimation. This produces two chronic,
opposite failure modes on any given day: over-preparation (food waste,
higher food cost) and under-preparation (stockouts, lost sales). Decisions
are inconsistent between managers and impossible to explain after the fact.

## PoC Question

**"Can historical restaurant sales data accurately forecast near-term
item-level demand sufficiently well to support better food-preparation
decisions?"**

This is a feasibility question, not a deployment question. The PoC does
not ask "will this reduce waste by X%" — that requires a Pilot, once
feasibility is proven (same evidence-ladder discipline as any PoC: Demo
creates belief, PoC creates evidence, Pilot creates operational
confidence, MVP creates usable value).

## Objective

Assess whether item-level, day-ahead demand can be forecast from
historical sales, calendar, promotion, and weather signals accurately
enough (WAPE against a defined threshold) to generate a preparation
recommendation a manager would find more useful than their current
manual estimate — and to do so with a transparent, non-hallucinating
explanation layer, at an observable cost and latency.

## Scope (In)

- Historical sales data for 2–5 simulated restaurant locations, 20–50 menu
  items, 12–24 months daily — synthetic, seeded, documented as such
- Item-level demand forecasting (7-day horizon; primary decision =
  tomorrow's preparation quantity)
- A deterministic waste/shortage risk layer built on forecast + inventory
- A deterministic, transparent preparation recommendation engine
- An LLM explanation/copilot layer, grounded only in structured ML +
  business-rule output — never inventing a number
- A What-If simulator (deterministic recalculation, not LLM-driven)
- MLOps: model registry, champion/challenger comparison, drift monitoring
  (lightweight, PoC-appropriate — not a full MLflow deployment unless it
  adds real value at this scale)
- LLMOps: prompt versioning, an evaluation dataset, cost + latency logging
- Illustrative, assumption-labelled ROI calculator
- A working, enterprise-styled UI covering all 13 modules (Section 4)

## Scope (Out)

- Production ERP / POS integration
- Automated purchase ordering or any irreversible automated action
- Real-time (sub-daily) restaurant operations
- Payment processing
- Fully autonomous decisions — the manager always approves
- Production-scale, multi-tenant, or multi-country deployment
- A guarantee of waste reduction or revenue improvement — the PoC produces
  *illustrative* estimates, explicitly labelled, never a promise

## Data Requirements

Synthetic dataset (Phase 2), seeded (`random_state=42`), documented in
`docs/DATASET.md` — schema per Section 6 of the product brief, with
genuine seasonality, weekly patterns, promotions, holidays, occasional
stockouts, and item-level heterogeneity. No claim of real-client data.

## Timeline (PoC — matches the 14 build phases, condensed)

Architecture & charter → dataset → pipeline/validation → ML models
→ MLOps → risk/recommendation engine → LLM copilot → LLMOps → cost/latency
→ backend → frontend → integration → testing → demo prep.
Each phase ships working, tested code before the next begins — no phase
is "planned" without being built and verified.

## Success Criteria (four dimensions, thresholds configurable — not
pre-claimed as met)

| Dimension | Criterion | Threshold (PoC target, to be measured) |
|---|---|---|
| Business | Preparation recommendations rated more useful than manual estimate | Qualitative — requires a human reviewer session, out of scope for automated PoC self-certification |
| Analytics | Forecast accuracy | WAPE ≤ 25% on held-out time-based test set (typical retail/food-service PoC bar; will be reported exactly, not rounded favorably) |
| Technical | Reliable end-to-end prediction | 0 unhandled errors across defined test scenarios, including missing/invalid input |
| AI | Grounded LLM responses | ≥90% of eval-suite responses contain zero unsupported claims; explicit "I don't have enough information" on out-of-scope questions |
| Operational | Cost & latency observable | Every request logged with cost + stage-level latency; P95 latency target ≤2s (ML path), LLM path may run longer and is tracked separately |

## Risks

| Risk | Mitigation |
|---|---|
| Synthetic data too easy/hard to forecast, giving a misleading feasibility signal | Explicit noise injection + documented generation logic (Phase 2), reviewed before modeling begins |
| LLM invents a number not present in structured output | Hard architectural rule: LLM prompt never contains permission to compute — only to narrate given facts; eval suite checks this |
| Scope creep toward a real ordering/ERP system | This charter's Out-of-Scope list is the standing reference; any request outside it is flagged, not silently built |
| Over-claiming ROI | Every business-impact number carries an "Illustrative PoC estimate" label and adjustable assumptions, never a bare number |

## Dependencies

Python 3.11+, a local machine capable of training scikit-learn/XGBoost
models on ~500K-row tabular data (trivial on any laptop), optionally an
LLM API key (the system must run end-to-end without one, per the
established fallback pattern).

## Assumptions

Restaurant operates a fixed daily menu (no dynamic menu changes mid-PoC);
"waste" is defined as prepared-but-unsold at day close; a 7-day forecast
horizon is sufficient for the preparation decision this PoC targets (a
same-day/intraday re-forecast is a Pilot-stage extension, not in scope).

## Expected Outputs

A working, demoable product (backend + frontend), a trained and evaluated
forecasting pipeline with an honest model-comparison table, a risk and
recommendation engine, a grounded LLM copilot, full MLOps/LLMOps/cost/
latency/data-quality visibility, and this charter's success criteria
measured — not asserted — against the built system.

## Business Value (framed, not promised)

If the PoC's forecast accuracy and grounding rate clear their thresholds,
the evidence supports a Pilot: a single-location, live trial measuring
actual waste and stockout reduction against a control period — the only
stage that can honestly measure real business value.
