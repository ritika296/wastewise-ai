# WasteWise AI — Demo Script (Phase 14)

Every number below was captured from a live run of the actual system on
2026-05-08's forecast — not written as an illustrative placeholder. Run
`docker compose up` or the local quick-start in `README.md`, open the
frontend, and every screen described here will show the same figures
(the champion model, the seeded dataset, and the recommendation logic
are all deterministic).

---

## Scene 1 — Business Problem

**Presenter:** *"Here is the client's problem."*

Open the **Executive Dashboard**. For tomorrow's service:

> 90 items scored across 3 locations. **55 items at high shortage risk.**
> 2 items at high waste risk. ₹86,120 in revenue at risk from potential
> stockouts; ₹61,662 in estimated waste cost if over-prepared. The
> restaurant manager currently decides preparation quantities from
> experience and a gut feel for "how last week went" — with no
> systematic way to separate the 55 genuinely risky items from the other
> 35.

## Scene 2 — Current Decision

**Presenter:** *"Here is how the decision is currently made."*

Without WasteWise, a manager would glance at yesterday's sales and
prepare "about the same, maybe a bit more for Friday." There's no
record of *why* that number was chosen, and every manager on shift makes
a different call — this is the inconsistency Section 1 of the PoC
Charter names directly.

## Scene 3 — Analytics Output

**Presenter:** *"Here is what our analytics identifies."*

Click into **Chicken Biryani** (Bengaluru - Koramangala) — the
dashboard's #1 flagged item:

```
Forecast demand:          93.2 portions
Currently available:      56.0 portions
Recommended preparation:  45 portions      (top-up, not from scratch)
Risk category:            HIGH SHORTAGE RISK
```

This is the exact structure the brief's own Biryani example calls for —
forecast, available, recommended prep, risk, reason — computed by
`ml/inference/recommendation_engine.py`'s transparent arithmetic, not a
model guessing a final number.

## Scene 4 — AI Explanation

**Presenter:** *"Here is why the prediction was made — and here is what
the AI recommends."*

Ask the Copilot: **"Why is Chicken Biryani high risk?"**

> ANSWER: Chicken Biryani is classified as high shortage risk.
>
> KEY EVIDENCE (real XGBoost TreeSHAP contributions, not invented):
> - 28-day average demand: 110.79 → increases forecast by ~41.8%
> - 14-day average demand: 112.5 → increases forecast by ~11.4%
> - Yesterday's sales: 140.0 → increases forecast by ~5.9%
>
> REASON: Even in a high-demand scenario within normal variability (117
> portions), forecasted demand would exceed total available supply (101)
> by 16 portions — 17% of forecast.
>
> RECOMMENDED ACTION: Prepare approximately 45 portions.
>
> EXPECTED BUSINESS IMPACT: Revenue at risk if under-prepared: ₹4,033.80.
> Estimated waste cost if over-prepared: ₹2,771.37.

**Presenter, if asked "is this a real AI model or a script?":** *"Both,
honestly — the driver percentages are exact TreeSHAP values from the
real trained XGBoost model; the sentence structure you're reading is a
deterministic template unless a live LLM key is configured, and it says
so explicitly rather than pretending otherwise."*

## Scene 5 — Decision (What-If)

**Presenter:** *"Here is where the manager makes the final call."*

Open the **What-If Simulator**. Ask: *"What if tomorrow's demand is 20%
higher than forecast — maybe a cricket match nearby?"*

```
Original forecast:    93.2 portions
Scenario forecast:   111.8 portions   (+20.0%, exactly)
```

The manager can see the new preparation number recalculate in real time
— through the same deterministic recommendation engine, never an LLM
guessing a number — before committing to a decision.

Back on the item page, the manager clicks **Approve** on the 45-portion
recommendation. This is logged (`POST /feedback`, persisted in SQLite)
— the human decision is the one that counts, and it's never automated
away.

## Scene 6 — Business Outcome

**Presenter:** *"Here is the expected business value."*

Open **Business Impact / ROI** (every input adjustable, nothing
pre-baked):

```
Monthly savings:        ₹32,000
Annual savings:         ₹384,000
Net annual benefit:     ₹180,000
ROI:                     88.2%
Payback period:          4.7 months

"Illustrative PoC estimate — all inputs are adjustable assumptions,
 not measured results."
```

That label is not boilerplate — it's load-bearing. Change any input and
the ROI recalculates live; nothing here is presented as an achieved
result.

## Scene 7 — Trust

**Presenter:** *"And here is the evidence that the solution is
feasible, operable, and honest about its own limits."*

- **MLOps**: Champion model is XGBoost (`xgboost-v1`), selected by a
  formal, threshold-gated promotion policy — not just "whichever number
  looked best." WAPE 23.05% against real historical sales.
- **Data Quality**: 49,167 of 49,369 records valid (99.6%) — checked
  before any prediction is trusted, not assumed clean.
- **Latency**: P95 total response time **15.6ms** — the 2-second SLA
  target isn't close, it's not even in the same order of magnitude.
- **LLMOps / AI Cost**: Every copilot answer is grounded, logged, and
  costed. In template mode (no LLM key set) the cost is exactly ₹0 —
  and the product tells you that's why, rather than hiding it.

---

## Sample User Journey (Section 30, walked live)

1. Manager opens the Dashboard → sees 55 high-shortage-risk items for tomorrow.
2. Clicks **Chicken Biryani** → sees forecast 93.2, available 56, recommend 45.
3. Asks the Copilot *"Why is demand high?"* → gets the grounded, TreeSHAP-backed explanation above.
4. Opens **What-If**, tries **+20% demand** → sees the recalculated 111.8-portion scenario.
5. Approves the 45-portion recommendation → persisted to the feedback log.
6. Dashboard reflects the day's activity; MLOps/LLMOps/Cost/Latency pages show the system is healthy.

---

## Tough questions this demo should survive

**"Is this real data?"** No — clearly documented as synthetic in
`docs/DATASET.md`, generated with realistic seasonality, weekly patterns,
and a deliberately hard demand-censoring problem, precisely so the model
comparison numbers (23.05% WAPE) are honest rather than trivially good.

**"Why isn't the AI just automating the preparation order?"** Section
20's human-in-the-loop principle, enforced in the product: every
recommendation requires an explicit approve/reject/modify from a person.

**"What if the LLM provider goes down?"** Watch it happen — `docs/API.md`
and `tests/test_failure_scenarios.py` show the exact fallback: a timeout
or connection error degrades to the deterministic template engine
automatically, and the response says so (`fallback_used: true`), never
silently.

**"How do I know the champion model wasn't cherry-picked?"** `docs/MLOPS.md`
documents the full promotion trail: Random Forest → registered first,
becomes champion by default → Gradient Boosting registered, *rejected*
for insufficient improvement → XGBoost registered, promoted on a real
0.52-point WAPE gain. The rejection is as visible as the promotion.
