"""
Phase 4 — model training, comparison, and champion selection.

Trains:
  - Baseline A: Naive (lag_7 — "same day last week")
  - Baseline B: Moving Average (rolling_mean_7)
  - Random Forest
  - Gradient Boosting
  - XGBoost

All three ML models are trained to predict `units_sold` (the real,
available-in-production target) — NOT `true_demand`, which would not
exist at prediction time. Every model is then evaluated against BOTH
`units_sold` and `true_demand` on val/test, so the comparison table shows
honestly how much accuracy is lost to demand censoring.

MLflow (local file store, `./mlruns` — no tracking server needed) logs
every run's params, metrics, and model artifact. A thin JSON pointer
(`models/champion_model.json`) records which run is the serving champion,
written only after the evaluation criteria below are applied — never an
automatic "lowest error wins."

Run from wastewise-ai/ root: python ml/training/train_models.py
"""
import json
import time
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import mlflow
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
from ml.evaluation.metrics import evaluate

PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlruns.db'}")
mlflow.set_experiment("wastewise-demand-forecasting")

TARGET = "units_sold"


def load_data():
    train = pd.read_parquet(PROCESSED / "train.parquet")
    val = pd.read_parquet(PROCESSED / "val.parquet")
    test = pd.read_parquet(PROCESSED / "test.parquet")
    features = json.load(open(PROCESSED / "feature_columns.json"))
    return train, val, test, features


def eval_both_targets(y_pred, split_df, split_name, model_name, timings):
    """Evaluate against units_sold (production-real) AND true_demand (latent, honest check)."""
    result = {
        "model": model_name, "split": split_name,
        "vs_units_sold": evaluate(split_df[TARGET].values, y_pred),
        "vs_true_demand": evaluate(split_df["true_demand"].values, y_pred),
        **timings,
    }
    return result


def time_inference(model_or_fn, X, n_repeats=3):
    times = []
    for _ in range(n_repeats):
        t0 = time.perf_counter()
        model_or_fn(X)
        times.append((time.perf_counter() - t0) * 1000)
    per_1000 = (np.mean(times) / len(X)) * 1000
    return {"inference_ms_per_1000_rows": round(per_1000, 3)}


def main():
    train, val, test, features = load_data()
    print(f"Train: {len(train)} | Val: {len(val)} | Test: {len(test)} | Features: {len(features)}")

    results = []
    comparison_rows = []

    # ---------------------------------------------------------------
    # Baseline A: Naive (lag_7)
    # ---------------------------------------------------------------
    with mlflow.start_run(run_name="baseline_naive_lag7"):
        mlflow.log_param("model_type", "naive_lag7")
        pred_fn = lambda df: df["lag_7"].values
        timing = time_inference(pred_fn, val)
        for split_name, split_df in [("val", val), ("test", test)]:
            r = eval_both_targets(pred_fn(split_df), split_df, split_name, "baseline_naive_lag7", timing)
            results.append(r)
            mlflow.log_metrics({f"{split_name}_wape": r["vs_units_sold"]["wape_pct"],
                                 f"{split_name}_mae": r["vs_units_sold"]["mae"]})
        mlflow.log_metric("inference_ms_per_1000_rows", timing["inference_ms_per_1000_rows"])

    # ---------------------------------------------------------------
    # Baseline B: Moving Average (rolling_mean_7)
    # ---------------------------------------------------------------
    with mlflow.start_run(run_name="baseline_moving_avg_7"):
        mlflow.log_param("model_type", "moving_average_7")
        pred_fn = lambda df: df["rolling_mean_7"].values
        timing = time_inference(pred_fn, val)
        for split_name, split_df in [("val", val), ("test", test)]:
            r = eval_both_targets(pred_fn(split_df), split_df, split_name, "baseline_moving_avg_7", timing)
            results.append(r)
            mlflow.log_metrics({f"{split_name}_wape": r["vs_units_sold"]["wape_pct"],
                                 f"{split_name}_mae": r["vs_units_sold"]["mae"]})
        mlflow.log_metric("inference_ms_per_1000_rows", timing["inference_ms_per_1000_rows"])

    # ---------------------------------------------------------------
    # ML Models
    # ---------------------------------------------------------------
    X_train, y_train = train[features], train[TARGET]
    X_val, y_val = val[features], val[TARGET]
    X_test, y_test = test[features], test[TARGET]

    model_specs = {
        "random_forest": RandomForestRegressor(
            n_estimators=300, max_depth=10, min_samples_leaf=5,
            random_state=42, n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingRegressor(
            n_estimators=200, max_depth=3, learning_rate=0.08,
            subsample=0.85, random_state=42,
        ),
        "xgboost": XGBRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.06,
            subsample=0.85, colsample_bytree=0.85,
            objective="reg:squarederror", random_state=42, n_jobs=-1,
        ),
    }

    fitted_models = {}
    for name, model in model_specs.items():
        with mlflow.start_run(run_name=name):
            mlflow.log_params({k: v for k, v in model.get_params().items() if isinstance(v, (int, float, str, bool)) or v is None})
            t0 = time.perf_counter()
            model.fit(X_train, y_train)
            fit_time = time.perf_counter() - t0
            mlflow.log_metric("fit_time_s", round(fit_time, 2))

            timing = time_inference(lambda X: model.predict(X), X_val)
            mlflow.log_metric("inference_ms_per_1000_rows", timing["inference_ms_per_1000_rows"])

            for split_name, X_split, split_df in [("val", X_val, val), ("test", X_test, test)]:
                y_pred = np.clip(model.predict(X_split), 0, None)
                r = eval_both_targets(y_pred, split_df, split_name, name, timing)
                results.append(r)
                mlflow.log_metrics({
                    f"{split_name}_wape": r["vs_units_sold"]["wape_pct"],
                    f"{split_name}_mae": r["vs_units_sold"]["mae"],
                    f"{split_name}_rmse": r["vs_units_sold"]["rmse"],
                    f"{split_name}_bias_pct": r["vs_units_sold"]["forecast_bias_pct"],
                })

            model_path = MODELS_DIR / f"{name}.joblib"
            joblib.dump(model, model_path)
            mlflow.log_artifact(str(model_path))
            mlflow.set_tag("model_size_kb", round(model_path.stat().st_size / 1024, 1))
            fitted_models[name] = model

    # ---------------------------------------------------------------
    # Comparison table (test split, vs. units_sold — the real target)
    # ---------------------------------------------------------------
    print("\n" + "=" * 100)
    print(f"{'Model':<22}{'MAE':>8}{'RMSE':>8}{'WAPE%':>8}{'MAPE%':>8}{'Bias%':>8}{'Latency(ms/1k)':>16}")
    print("=" * 100)
    test_results = [r for r in results if r["split"] == "test"]
    comparison = []
    for r in test_results:
        m = r["vs_units_sold"]
        print(f"{r['model']:<22}{m['mae']:>8}{m['rmse']:>8}{m['wape_pct']:>8}{m['mape_pct']:>8}"
              f"{m['forecast_bias_pct']:>8}{r['inference_ms_per_1000_rows']:>16}")
        comparison.append({
            "model": r["model"], **m,
            "inference_ms_per_1000_rows": r["inference_ms_per_1000_rows"],
            "vs_true_demand_wape_pct": r["vs_true_demand"]["wape_pct"],
        })
    print("=" * 100)

    with open(MODELS_DIR / "model_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)
    with open(MODELS_DIR / "full_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # ---------------------------------------------------------------
    # Champion selection — multi-criteria, NOT "lowest WAPE wins" alone
    # ---------------------------------------------------------------
    ml_only = [c for c in comparison if not c["model"].startswith("baseline")]
    # Filter: must beat both baselines on WAPE to even be considered "adds value"
    baseline_wape = min(c["wape_pct"] for c in comparison if c["model"].startswith("baseline"))
    viable = [c for c in ml_only if c["wape_pct"] < baseline_wape]
    if not viable:
        champion = min(ml_only, key=lambda c: c["wape_pct"])
        rationale = (f"No ML model beat the best baseline (WAPE {baseline_wape}%) on this run; "
                     f"selecting {champion['model']} as the closest candidate. This would be a "
                     f"NO-GO signal for the PoC decision gate, not silently promoted.")
    else:
        # Among viable models, balance accuracy against latency and forecast bias magnitude
        # (a model that's 1% more accurate but 50x slower, or badly biased toward
        # over/under-forecasting, is not automatically the right choice).
        def score(c):
            return c["wape_pct"] + 0.002 * c["inference_ms_per_1000_rows"] + 0.5 * abs(c["forecast_bias_pct"])
        champion = min(viable, key=score)
        rationale = (
            f"'{champion['model']}' selected: WAPE {champion['wape_pct']}% (beats baseline "
            f"{baseline_wape}%), inference {champion['inference_ms_per_1000_rows']}ms/1k rows, "
            f"forecast bias {champion['forecast_bias_pct']}%. Chosen over other viable candidates "
            f"by balancing accuracy, latency, and bias magnitude — not WAPE alone."
        )

    registry = {
        "champion": champion["model"],
        "selection_rationale": rationale,
        "champion_metrics": champion,
        "baseline_wape_pct": baseline_wape,
        "trained_at": pd.Timestamp.now("UTC").isoformat(),
        "train_rows": len(train), "val_rows": len(val), "test_rows": len(test),
        "feature_count": len(features),
    }
    with open(MODELS_DIR / "champion_model.json", "w") as f:
        json.dump(registry, f, indent=2)

    print(f"\nCHAMPION: {champion['model']}")
    print(rationale)


if __name__ == "__main__":
    main()
