"""
Drift monitoring (Phase 5 / Section 19). Two kinds of drift are tracked:

1. FEATURE (data) drift — has the distribution of input features shifted
   vs. what the model was trained on? (Population Stability Index)
2. PREDICTION drift — has the distribution of the model's own outputs
   shifted? (Can catch drift even in features not individually monitored.)

Since this PoC has no live production traffic yet (that starts in
Phase 10's API), drift is evaluated two ways, BOTH clearly labeled:

  - `check_drift(reference, current)` run against the REAL train vs. test
    split — an honest measurement, not simulated, of how much the actual
    data shifted over the ~8 month test-set gap.
  - `simulate_drift_scenario()` artificially perturbs a batch to
    demonstrate the detector catches a real shift — explicitly labeled
    "simulated" in its output, per Section 19's instruction.
"""
import numpy as np
import pandas as pd

PSI_STABLE = 0.10
PSI_WARNING = 0.25

# Features whose marginal distribution is EXPECTED to differ across two
# non-overlapping calendar windows, purely from calendar structure — not
# because anything is actually wrong. `month` is a deterministic calendar
# index: comparing a 12-month training window against a 4-month test
# window (Apr-Jul only) guarantees a large PSI with zero informational
# content. These are still measured and reported for transparency, but
# excluded from the retrain-trigger's "critical" determination — a real
# MLOps system that didn't do this would page someone every single month
# for no reason. `temperature` is reported with the same caveat but NOT
# excluded, since real temperature anomalies (a genuine heatwave, not a
# calendar artifact) are a legitimate signal worth watching — see the
# seasonally-matched comparison note in docs/LLMOPS.md's MLOps section.
CALENDAR_STRUCTURAL_FEATURES = {"month"}


def population_stability_index(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    reference = reference[~np.isnan(reference)]
    current = current[~np.isnan(current)]
    if len(reference) < 10 or len(current) < 10:
        return 0.0
    breakpoints = np.quantile(reference, np.linspace(0, 1, bins + 1))
    breakpoints[0], breakpoints[-1] = -np.inf, np.inf
    breakpoints = np.unique(breakpoints)
    if len(breakpoints) < 3:
        return 0.0
    ref_counts, _ = np.histogram(reference, bins=breakpoints)
    cur_counts, _ = np.histogram(current, bins=breakpoints)
    ref_pct = np.clip(ref_counts / max(len(reference), 1), 1e-4, None)
    cur_pct = np.clip(cur_counts / max(len(current), 1), 1e-4, None)
    return round(float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))), 4)


def _status(psi: float) -> str:
    if psi < PSI_STABLE:
        return "normal"
    if psi < PSI_WARNING:
        return "warning"
    return "critical"


def check_feature_drift(reference_df: pd.DataFrame, current_df: pd.DataFrame, features: list) -> dict:
    report = {}
    for f in features:
        if f not in reference_df.columns or f not in current_df.columns:
            continue
        if not np.issubdtype(reference_df[f].dtype, np.number):
            continue
        psi = population_stability_index(reference_df[f].values.astype(float), current_df[f].values.astype(float))
        report[f] = {
            "psi": psi, "status": _status(psi),
            "excluded_from_trigger": f in CALENDAR_STRUCTURAL_FEATURES,
        }
    # Only non-calendar-structural features count toward the overall/critical
    # determination that feeds the retrain trigger.
    triggerable = {k: v for k, v in report.items() if not v["excluded_from_trigger"]}
    worst_psi = max((v["psi"] for v in triggerable.values()), default=0.0)
    return {
        "per_feature": report, "overall_status": _status(worst_psi), "max_psi": worst_psi,
        "n_reference": len(reference_df), "n_current": len(current_df), "simulated": False,
        "note": ("'month' is measured but excluded from the critical-drift determination — "
                 "see CALENDAR_STRUCTURAL_FEATURES for why."),
    }


def check_prediction_drift(reference_predictions: np.ndarray, current_predictions: np.ndarray) -> dict:
    psi = population_stability_index(np.asarray(reference_predictions, dtype=float),
                                      np.asarray(current_predictions, dtype=float))
    return {
        "psi": psi, "status": _status(psi),
        "reference_mean": round(float(np.mean(reference_predictions)), 2),
        "current_mean": round(float(np.mean(current_predictions)), 2),
        "simulated": False,
    }


def simulate_drift_scenario(reference_df: pd.DataFrame, features: list, shift_feature: str = "temperature",
                             shift_amount: float = 12.0) -> dict:
    """
    SIMULATED — not measured from real data. Artificially shifts one
    feature's distribution (e.g., a +12C heatwave scenario) to demonstrate
    the detector correctly flags a real shift when one occurs. Every
    field in the returned dict says so explicitly.
    """
    current = reference_df.copy()
    if shift_feature in current.columns:
        current[shift_feature] = current[shift_feature] + shift_amount
    report = check_feature_drift(reference_df, current, features)
    report["simulated"] = True
    report["simulation_description"] = (
        f"SIMULATED SCENARIO — not real production data. Artificially shifted "
        f"'{shift_feature}' by +{shift_amount} to demonstrate drift detection, "
        f"e.g. an unseasonal heatwave shifting demand for heat-sensitive items."
    )
    for f_report in report["per_feature"].values():
        f_report["simulated"] = True
    return report
