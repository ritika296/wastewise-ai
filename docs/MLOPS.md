# WasteWise AI — MLOps (Phase 5)

## Model Registry (`ml/monitoring/registry.py`)

A formal version registry with an explicit **promotion policy** — no
candidate becomes the serving champion just by being registered. A
challenger must clear ALL of:

| Rule | Threshold |
|---|---|
| WAPE improvement over current champion | ≥ 0.5 percentage points |
| Inference latency | ≤ 2.0x the current champion's latency |
| Forecast bias magnitude regression | ≤ 1.0 percentage point worse |

**Live trace from this run** (`ml/monitoring/run_mlops_pipeline.py`):

```
random_forest    -> promoted (no existing champion)
gradient_boosting -> REJECTED (0.43pp improvement < 0.5pp threshold)
xgboost          -> promoted (0.52pp improvement, latency & bias within bounds)
```

**Formal champion: XGBoost.** See `docs/ML_PIPELINE.md`'s reconciliation
note for why this differs from Phase 4's hand-rolled pick (Gradient
Boosting) — this is the intended outcome of moving from an ad hoc
heuristic to a testable policy, not an inconsistency to paper over.

## Drift Monitoring (`ml/monitoring/drift.py`)

Two kinds tracked: **feature drift** (has the input distribution shifted
from what the model was trained on?) and **prediction drift** (has the
model's own output distribution shifted?), both via Population Stability
Index (PSI): <0.10 normal, 0.10–0.25 warning, ≥0.25 critical.

### A real finding worth stating plainly

The first version of this check compared the full-year training set
against the 4-month test window (Apr–Jul) directly, and it reported
**`temperature` PSI = 4.20 — "critical."** That's not a real problem: a
4-month test window can never calendar-match a 12-month training window,
so a strongly seasonal feature is *guaranteed* to show a large marginal
shift with zero informational content. A drift monitor that fired an
alert here would train the operations team to ignore it — the classic
failure mode of a noisy monitor.

**Fix applied**: the real (non-simulated) drift check now compares the
test window against the *same calendar months from the prior year*
(season-matched), not the full training set. Re-run:

```
[NAIVE: full-year train vs. 4-month test]         max PSI: 4.20 (misleading)
[SEASON-MATCHED: same months, prior year]          max PSI: 0.20 (warning — genuinely mild, real)
Retrain trigger (real data): healthy, no trigger
```

`month` itself is excluded from the trigger determination entirely
(`CALENDAR_STRUCTURAL_FEATURES`) since it's a deterministic calendar
index — its "drift" across non-overlapping windows is guaranteed and
uninformative by construction, not a signal.

### Simulated scenario (clearly labeled, per Section 19)

`simulate_drift_scenario()` artificially shifts `temperature` by +12°C
(an unseasonal heatwave) to prove the detector catches a *real* shift
when one occurs — every field in its output carries `"simulated": true`
and a `[SIMULATED SCENARIO]` tag propagates through to the retrain-trigger
reason text, so this can never be mistaken for a live finding downstream.

```
[SIMULATED +12C heatwave] max PSI: 4.40 — critical
Retrain trigger (SIMULATED): triggered
  - Feature drift critical: temperature=4.40 [SIMULATED SCENARIO]
  - Forecast accuracy degraded: WAPE 23.05% -> 31.12% (simulated), 35% relative increase
```

## Retraining Trigger (`ml/monitoring/retrain_trigger.py`)

A simple, inspectable rule engine — not a black-box score. Three rules,
each stating exactly which one fired:

1. Feature drift critical (max PSI ≥ 0.25, excluding calendar-structural features)
2. Prediction drift critical (PSI ≥ 0.25)
3. Forecast accuracy degraded ≥20% relative WAPE increase vs. training-time value

On the real (season-matched) data: **no trigger fires — status healthy.**
On the simulated heatwave scenario: **triggers, for two independent, clearly-labeled reasons.**

## What Phase 5 does not include

Automated retraining execution (the trigger *recommends*, per Section 40's
"keep human approval in the decision loop" — a real deployment would
route this to a human-reviewed retraining job, not auto-retrain), a live
feature store, and continuous/scheduled monitoring (this PoC runs
monitoring on-demand via the pipeline script; Phase 10's API exposes the
same checks over HTTP, but there's no scheduler/cron in the PoC).
