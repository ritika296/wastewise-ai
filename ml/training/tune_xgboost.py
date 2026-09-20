"""
Phase 15a — Bayesian hyperparameter tuning for the XGBoost champion.

Uses Optuna (TPE sampler) to search the XGBoost hyperparameter space
against VALIDATION WAPE only. The tuned model is then evaluated on the
untouched TEST set exactly once, so the reported test result cannot be
cherry-picked by the search itself.

This is an honesty-first script: if the tuned model's test WAPE does not
improve on the existing baseline champion's test WAPE, that is reported
plainly via a `verdict` field rather than smoothed over. Overfitting a
validation set during hyperparameter search is a real and common failure
mode, not a bug in this script.

Run from wastewise-ai/ root: python ml/training/tune_xgboost.py
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from ml.evaluation.metrics import evaluate

PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
TARGET = "units_sold"
N_TRIALS = 40

optuna.logging.set_verbosity(optuna.logging.WARNING)


def load_data():
    train = pd.read_parquet(PROCESSED / "train.parquet")
    val = pd.read_parquet(PROCESSED / "val.parquet")
    test = pd.read_parquet(PROCESSED / "test.parquet")
    features = json.load(open(PROCESSED / "feature_columns.json"))
    return train, val, test, features


def load_baseline_test_wape() -> float:
    """The existing (untuned) XGBoost champion's test WAPE, from Phase 4's
    full_results.json, used as the bar the tuned model must clear."""
    full_results = json.loads((MODELS_DIR / "full_results.json").read_text())
    rows = [r for r in full_results if r["model"] == "xgboost" and r["split"] == "test"]
    if not rows:
        return None
    return rows[0]["vs_units_sold"]["wape_pct"]


def make_objective(X_train, y_train, X_val, y_val):
    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 9),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        }
        model = XGBRegressor(
            **params, objective="reg:squarederror", random_state=42, n_jobs=-1,
        )
        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        return evaluate(y_val.values, pred)["wape_pct"]

    return objective


def main():
    train, val, test, features = load_data()
    X_train, y_train = train[features], train[TARGET]
    X_val, y_val = val[features], val[TARGET]
    X_test, y_test = test[features], test[TARGET]

    baseline_test_wape = load_baseline_test_wape()
    print(f"Baseline (untuned) XGBoost champion — test WAPE: {baseline_test_wape}%")
    print(f"Running Optuna search: {N_TRIALS} trials, TPE sampler, optimizing VAL WAPE only...")

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    t0 = time.perf_counter()
    study.optimize(make_objective(X_train, y_train, X_val, y_val), n_trials=N_TRIALS, show_progress_bar=False)
    search_seconds = round(time.perf_counter() - t0, 1)

    best_params = study.best_params
    best_val_wape = round(study.best_value, 2)
    print(f"Search done in {search_seconds}s. Best val WAPE: {best_val_wape}% with params: {best_params}")

    # Refit the best-found config on train, then evaluate ONCE on the
    # untouched test set. No further trials are run after seeing this number.
    tuned_model = XGBRegressor(
        **best_params, objective="reg:squarederror", random_state=42, n_jobs=-1,
    )
    tuned_model.fit(X_train, y_train)
    tuned_val_pred = tuned_model.predict(X_val)
    tuned_test_pred = tuned_model.predict(X_test)

    tuned_val_metrics = evaluate(y_val.values, tuned_val_pred)
    tuned_test_metrics = evaluate(y_test.values, tuned_test_pred)

    # Baseline val WAPE (untuned defaults) for a like-for-like val comparison.
    baseline_model = XGBRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.06,
        subsample=0.85, colsample_bytree=0.85,
        objective="reg:squarederror", random_state=42, n_jobs=-1,
    )
    baseline_model.fit(X_train, y_train)
    baseline_val_metrics = evaluate(y_val.values, baseline_model.predict(X_val))

    val_improved = tuned_val_metrics["wape_pct"] < baseline_val_metrics["wape_pct"]
    test_improved = (
        baseline_test_wape is not None and tuned_test_metrics["wape_pct"] < baseline_test_wape
    )

    if test_improved:
        verdict = (
            f"Tuning improved test WAPE from {baseline_test_wape}% to "
            f"{tuned_test_metrics['wape_pct']}% — promote the tuned model."
        )
    elif val_improved and not test_improved:
        verdict = (
            f"Tuning improved val WAPE ({baseline_val_metrics['wape_pct']}% -> "
            f"{tuned_val_metrics['wape_pct']}%) but did NOT meaningfully improve test WAPE "
            f"({baseline_test_wape}% -> {tuned_test_metrics['wape_pct']}%). This is a validation-set "
            f"overfit: the search found hyperparameters that fit val's specific noise, not a "
            f"generalizable improvement. The untuned baseline champion remains in production."
        )
    else:
        verdict = (
            f"Tuning did not improve on the baseline (val {baseline_val_metrics['wape_pct']}% -> "
            f"{tuned_val_metrics['wape_pct']}%, test {baseline_test_wape}% -> "
            f"{tuned_test_metrics['wape_pct']}%). The untuned baseline champion remains in production."
        )

    result = {
        "phase": "15a_hyperparameter_tuning",
        "method": "Optuna TPE sampler, 40 trials, optimizing validation WAPE",
        "search_seconds": search_seconds,
        "n_trials": N_TRIALS,
        "best_params": best_params,
        "baseline": {
            "val_wape_pct": baseline_val_metrics["wape_pct"],
            "test_wape_pct": baseline_test_wape,
        },
        "tuned": {
            "val_wape_pct": tuned_val_metrics["wape_pct"],
            "test_wape_pct": tuned_test_metrics["wape_pct"],
            "test_metrics_full": tuned_test_metrics,
        },
        "val_improved": bool(val_improved),
        "test_improved": bool(test_improved),
        "promoted": bool(test_improved),
        "verdict": verdict,
    }

    out_path = MODELS_DIR / "tuning_result.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\nSaved: {out_path}")
    print(f"\nVERDICT: {verdict}")
    return result


if __name__ == "__main__":
    main()
