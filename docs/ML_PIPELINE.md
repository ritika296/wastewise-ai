# WasteWise AI — ML Pipeline (Phase 4)

## Model comparison (test split: 2026-04-01 to 2026-07-02, 8,370 rows)

| Model | MAE | RMSE | WAPE% | MAPE% | Bias% | Latency (ms/1k rows) |
|---|---|---|---|---|---|---|
| Baseline: Naive (lag_7) | 16.55 | 22.88 | 32.37 | 39.19 | -0.63 | 0.02 |
| Baseline: Moving Avg (7d) | 13.21 | 18.36 | 25.85 | 32.15 | -0.14 | 0.01 |
| Random Forest | 12.05 | 16.54 | 23.57 | 29.07 | -1.44 | 16.09 |
| Gradient Boosting | 11.83 | 15.93 | 23.14 | 29.68 | -0.22 | 1.21 |
| **XGBoost** | 11.79 | 15.93 | **23.05** | 28.93 | -1.13 | 3.31 |

All numbers above are from an actual run of `ml/training/train_models.py`
against the real processed dataset — not illustrative.

## Champion: Gradient Boosting (not XGBoost, despite XGBoost's lower WAPE)

This is deliberate, not an error. **XGBoost has the best raw WAPE (23.05%
vs. 23.14%)** — a 0.09-point difference. The selection logic in
`train_models.py` scores candidates on `WAPE + 0.002×latency +
0.5×|bias|`, not WAPE alone, per the brief's explicit instruction not to
select a model on accuracy alone:

- **Latency**: Random Forest is ~13x slower than Gradient Boosting per
  1,000 rows (16.09ms vs. 1.21ms) — at the scale of daily, multi-item,
  multi-location scoring this compounds; RF is penalized for it.
- **Bias**: Random Forest (-1.44%) and XGBoost (-1.13%) both
  systematically *under*-forecast more than Gradient Boosting (-0.22%).
  In this business, under-forecasting bias is the shortage-risk direction
  — arguably the more expensive failure mode (lost sales vs. wasted
  food), so a smaller bias magnitude is weighted in.
- Net: Gradient Boosting's negligible accuracy cost against XGBoost buys
  meaningfully lower forecast bias, at a latency far better than Random
  Forest's. This is a documented trade-off, not a hidden one — see
  `models/champion_model.json` for the exact rationale string the system
  itself records.

**All three ML models clear the "must beat baseline" gate** — champion
selection only runs among models that already beat the better baseline
(Moving Average, 25.85% WAPE). This gate is enforced in code
(`viable = [... if wape < baseline_wape]`); if no model had cleared it,
the pipeline would report a NO-GO signal rather than silently promoting
the best of a bad set.

## The honest finding: censored demand costs real accuracy

Every model's WAPE is meaningfully worse when scored against the latent
`true_demand` instead of the censored `units_sold`:

| Model | WAPE vs. units_sold | WAPE vs. true_demand | Gap |
|---|---|---|---|
| Baseline: Naive | 32.37% | 34.34% | +1.97 |
| Baseline: Moving Avg | 25.85% | 28.53% | +2.68 |
| Random Forest | 23.57% | 26.45% | +2.88 |
| Gradient Boosting | 23.14% | 26.02% | +2.88 |
| XGBoost | 23.05% | 25.96% | +2.91 |

This ~2.7–2.9 point gap is the accuracy cost of the fact that, on
stockout days, `units_sold` under-represents what people actually wanted
to buy — a real, common food-service data problem (see `docs/DATASET.md`).
`true_demand` was never available as a training feature (see
`ml/features/build_features.py`); this comparison exists purely to report
the gap honestly, not to imply it was somehow closed.

**Implication for the PoC charter's success criterion** (WAPE ≤ 25%,
Section 39): Gradient Boosting clears this threshold against the
production-real target (`units_sold`, 23.14%) but **does not** clear it
against true demand (26.02%). This is reported as-is in the PoC decision
gate (Phase 14) rather than picking whichever number looks better.

## Time-aware validation

No shuffling. Train/val/test are chronologically disjoint (verified by
`tests/test_preprocessing.py::test_split_is_strictly_chronological` and
`test_split_is_not_random`). Every lag/rolling feature is computed with a
`.shift(1)` before any window operation, so no model ever sees same-day or
future information — verified by
`tests/test_preprocessing.py::test_no_feature_uses_same_or_future_day_target`,
which recomputes expected feature values from scratch on random series
and compares, rather than trusting the implementation by inspection alone.

## MLflow tracking

Local SQLite backend (`mlruns.db` — MLflow 3.x deprecated the plain
file-store backend used in earlier versions; SQLite requires no server and
is the currently-recommended local option). Every run logs: hyperparameters,
val/test WAPE/MAE/RMSE/bias, inference latency, fit time, and the model
artifact itself. View with:

```bash
mlflow ui --backend-store-uri sqlite:///mlruns.db
```

## Note: Phase 5 supersedes this file's champion pointer

`models/champion_model.json` (written by this phase) used a hand-rolled
weighted score (`WAPE + 0.002×latency + 0.5×|bias|`) to pick Gradient
Boosting. Phase 5 (`ml/monitoring/registry.py`) replaces this with a
formal, threshold-gated promotion policy applied to the same three
models, registered in the order they'd realistically be built:

1. Random Forest registered first → becomes champion by default (no incumbent to beat)
2. Gradient Boosting registered second → **rejected**: only 0.43 percentage-point WAPE improvement over Random Forest, below the 0.5pp promotion threshold
3. XGBoost registered third → **promoted**: 0.52pp WAPE improvement over the current champion (Random Forest), clears the threshold, latency and bias regression both within bounds

**Formal registry champion: XGBoost**, not Gradient Boosting. This is not
a contradiction — it's the difference between an ad hoc weighted-average
heuristic (Phase 4) and an explicit, testable promotion policy (Phase 5).
From Phase 6 onward, `ml/monitoring/registry.py`'s `get_champion()` is
the single source of truth the system serves from; Phase 4's
`champion_model.json` is kept only as a historical record of the first
pass. See `docs/ML_PIPELINE.md`'s companion note in the MLOps section
(Phase 5) for the full promotion trace.

## What Phase 4 does not include

Hyperparameter search beyond sensible defaults (a PoC-scale grid, not an
exhaustive tune — the brief prioritizes "working end-to-end flow over
dozens of incomplete features"), SHAP-based explainability (Phase 6, tied
to the risk/recommendation engine where explanations are actually
consumed), and production model-serving infrastructure (Phase 10's
FastAPI layer loads the champion artifact directly — no separate serving
container for a PoC this size).

## Phase 6: Risk & Recommendation Engine — a second real bug, found and fixed

The risk engine (`ml/inference/risk_engine.py`) classifies shortage and
waste risk from the forecast plus a demand-uncertainty band. Two issues
surfaced during development, both fixed rather than tuned away cosmetically:

**Bug 1 — uncertainty band used raw demand variance, not residual error.**
The first version computed each item's coefficient of variation from its
raw historical `units_sold` (mean ~0.33, driven largely by weekday/
seasonal/promotional swings the model already explains). That's not the
right basis for a *forecast* uncertainty band -- it double-counts variation
the model accounts for. Fixed: `compute_item_volatility()` now uses the
champion model's own **residuals on the validation set**
(`|actual - predicted| / actual`, per item), which correctly dropped the
mean CV to ~0.24 -- consistent with the champion's real ~23% test WAPE.

**Bug 2 -- shortage/waste thresholds mechanically co-triggered.** For any
item, `waste_gap - shortage_gap = 2 x safety_buffer` by construction (both
are measured from the same forecast +/- band around one buffer). With an
8% buffer (16-point fixed gap) and thresholds only 15 points apart
(10%/25%), *every* item with CV above ~18% -- i.e. nearly all 90 items at
this model's real accuracy -- triggered both "HIGH shortage" and "HIGH
waste" simultaneously, collapsing the whole priority list into a single
useless "REVIEW" bucket. Fixed by widening the threshold gap (15%/40%,
now 25 points) to exceed the mechanical 16-point coupling. Locked in by
`tests/test_recommendations.py::test_thresholds_dont_co_trigger_for_typical_champion_model_cv`
so it can't silently regress.

**Resulting demo distribution** (2026-05-08, 90 item-location rows):
53 HIGH shortage risk, 35 moderate, 2 genuine dual-risk items (both
low-volume, high-CV: Soup of the Day, Gajar Halwa) -- a distribution a
manager could actually act on, not one undifferentiated bucket.

**One more honest finding, not smoothed over**: zero items land in "LOW"
shortage risk at the default 8% safety buffer -- because the champion
model's real residual CV floor (~18%) already exceeds that buffer. This
is a genuine business signal, not a display bug: at this forecast
accuracy, an 8% buffer is simply too thin to fully de-risk any item. The
What-If Simulator (Phase 7) is where a manager would explore raising the
buffer and see this trade-off directly, rather than the system quietly
picking a buffer size that makes the output look better.
