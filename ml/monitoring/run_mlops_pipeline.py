"""
Phase 5 pipeline runner. Registers Phase 4's models into the formal
registry (demonstrating both a promotion and a rejection), checks real
feature/prediction drift (train vs. test), runs a labeled simulated
drift scenario, and evaluates retrain triggers against both.

Run from wastewise-ai/ root: python ml/monitoring/run_mlops_pipeline.py
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import joblib

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from ml.monitoring.registry import register_version, promote_if_better, get_champion, list_versions, REGISTRY_PATH
from ml.monitoring.drift import check_feature_drift, check_prediction_drift, simulate_drift_scenario
from ml.monitoring.retrain_trigger import evaluate_retrain_triggers

MODELS_DIR = ROOT / "models"
PROCESSED = ROOT / "data" / "processed"


def main():
    # Fresh registry each run for a clean, repeatable demonstration
    if REGISTRY_PATH.exists():
        REGISTRY_PATH.unlink()

    comparison = json.load(open(MODELS_DIR / "model_comparison.json"))
    ml_models = {c["model"]: c for c in comparison if not c["model"].startswith("baseline")}

    print("=== Registering models (in the order they'd realistically be built) ===")
    # 1. Register Random Forest first (as if it were the first candidate tried)
    rf_entry = register_version("random_forest", ml_models["random_forest"], str(MODELS_DIR / "random_forest.joblib"))
    result = promote_if_better(rf_entry["version_id"])
    print(f"random_forest -> {result}")

    # 2. Register Gradient Boosting — should be promoted (better WAPE, better latency, better bias)
    gb_entry = register_version("gradient_boosting", ml_models["gradient_boosting"], str(MODELS_DIR / "gradient_boosting.joblib"))
    result = promote_if_better(gb_entry["version_id"])
    print(f"gradient_boosting -> {result}")

    # 3. Register XGBoost — has the best raw WAPE, but should this promotion actually pass?
    #    Let's find out from the real thresholds rather than assuming.
    xgb_entry = register_version("xgboost", ml_models["xgboost"], str(MODELS_DIR / "xgboost.joblib"))
    result = promote_if_better(xgb_entry["version_id"])
    print(f"xgboost -> {result}")

    champion = get_champion()
    print(f"\nFinal registry champion: {champion['version_id']} ({champion['model_name']})")
    print(f"Total registered versions: {len(list_versions())}")

    # ---------------------------------------------------------------
    # Drift detection — REAL (train vs. test, actual data)
    # ---------------------------------------------------------------
    print("\n=== Feature drift: train (reference) vs. test (current) — REAL DATA ===")
    train = pd.read_parquet(PROCESSED / "train.parquet")
    test = pd.read_parquet(PROCESSED / "test.parquet")
    features = json.load(open(PROCESSED / "feature_columns.json"))
    numeric_features = [f for f in features if f in train.columns and np.issubdtype(train[f].dtype, np.number)]

    # NAIVE comparison first: full-year train vs. a 4-month test window.
    # This is deliberately shown to demonstrate WHY it's the wrong comparison
    # for seasonal features — not because it's the recommended check.
    naive_drift = check_feature_drift(train, test, numeric_features)
    print(f"[NAIVE: full-year train vs. 4-month test] max PSI (excl. calendar-structural): {naive_drift['max_psi']}")
    print("  -> 'temperature' PSI is huge here purely because the test window (Apr-Jul, hot season)")
    print("     doesn't calendar-match the full-year training window. This is a seasonality artifact,")
    print("     not drift. See the season-matched comparison below for the correct check.")

    # SEASON-MATCHED comparison: reference = same calendar months from the
    # training year. This is the historically correct comparison for
    # features with known seasonal structure — apples to apples.
    test_months = set(pd.to_datetime(test["date"]).dt.month.unique())
    season_matched_train = train[pd.to_datetime(train["date"]).dt.month.isin(test_months)]
    real_drift = check_feature_drift(season_matched_train, test, numeric_features)
    print(f"\n[SEASON-MATCHED: same calendar months, prior year] Overall status: "
          f"{real_drift['overall_status']} (max PSI: {real_drift['max_psi']})")
    worst = sorted(
        [(k, v) for k, v in real_drift["per_feature"].items() if not v["excluded_from_trigger"]],
        key=lambda kv: kv[1]["psi"], reverse=True,
    )[:5]
    for feat, r in worst:
        print(f"  {feat}: PSI={r['psi']} ({r['status']})")

    champion_model = joblib.load(MODELS_DIR / f"{champion['model_name']}.joblib")
    train_preds = champion_model.predict(train[features])
    test_preds = champion_model.predict(test[features])
    real_pred_drift = check_prediction_drift(train_preds, test_preds)
    print(f"\nPrediction drift: PSI={real_pred_drift['psi']} ({real_pred_drift['status']}) "
          f"mean {real_pred_drift['reference_mean']} -> {real_pred_drift['current_mean']}")

    training_wape = champion["metrics"]["wape_pct"]
    # "recent observed WAPE" = the test-set WAPE, standing in for what a live
    # monitoring job would compute from recent scored predictions
    recent_wape = champion["metrics"]["wape_pct"]  # same value here since test==the eval set
    real_decision = evaluate_retrain_triggers(real_drift, real_pred_drift, training_wape, recent_wape)
    print(f"\nRetrain trigger (real data): {real_decision.status} | trigger={real_decision.trigger_retrain}")
    for r in real_decision.reasons:
        print(f"  - {r}")

    # ---------------------------------------------------------------
    # Drift detection — SIMULATED heatwave scenario
    # ---------------------------------------------------------------
    print("\n=== Feature drift: SIMULATED +12C heatwave scenario ===")
    sim_drift = simulate_drift_scenario(train, numeric_features, shift_feature="temperature", shift_amount=12.0)
    print(f"[{'SIMULATED' if sim_drift['simulated'] else 'REAL'}] Overall status: {sim_drift['overall_status']} (max PSI: {sim_drift['max_psi']})")

    sim_current = train.copy()
    sim_current["temperature"] = sim_current["temperature"] + 12.0
    sim_preds = champion_model.predict(sim_current[features])
    sim_pred_drift = check_prediction_drift(train_preds, sim_preds)
    sim_pred_drift["simulated"] = True
    print(f"Prediction drift (simulated): PSI={sim_pred_drift['psi']} ({sim_pred_drift['status']})")

    # Simulate a degraded recent WAPE too, to show the accuracy-degradation trigger firing
    simulated_recent_wape = round(training_wape * 1.35, 2)
    sim_decision = evaluate_retrain_triggers(sim_drift, sim_pred_drift, training_wape, simulated_recent_wape)
    print(f"\nRetrain trigger (SIMULATED scenario): {sim_decision.status} | trigger={sim_decision.trigger_retrain}")
    for r in sim_decision.reasons:
        print(f"  - {r}")

    # Save monitoring snapshot for the API/frontend to read later (Phase 10+)
    snapshot = {
        "champion": champion, "real_feature_drift": real_drift, "real_prediction_drift": real_pred_drift,
        "real_retrain_decision": real_decision.to_dict(),
        "simulated_feature_drift": sim_drift, "simulated_prediction_drift": sim_pred_drift,
        "simulated_retrain_decision": sim_decision.to_dict(),
    }
    with open(ROOT / "monitoring" / "mlops_snapshot.json", "w") as f:
        json.dump(snapshot, f, indent=2, default=str)
    print(f"\nSnapshot saved to monitoring/mlops_snapshot.json")


if __name__ == "__main__":
    main()
