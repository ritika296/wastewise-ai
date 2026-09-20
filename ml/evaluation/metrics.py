"""
Forecast evaluation metrics (Phase 4 / Module 8). No metric here is
computed only against `units_sold` (the censored series) — every model
is also scored against `true_demand` (the latent, uncensored series),
so the comparison table can show, honestly, how much of the reported
accuracy is an artefact of forecasting a demand-capped signal.
"""
import numpy as np


def mae(y_true, y_pred):
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true, y_pred, epsilon=1e-6):
    """Only computed over rows where y_true > 0 — MAPE is undefined/explosive at y_true=0."""
    mask = y_true > 0
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / (y_true[mask] + epsilon))) * 100)


def wape(y_true, y_pred):
    """Weighted Absolute Percentage Error — robust to zero-demand days, the
    standard metric for intermittent/low-volume demand series."""
    denom = np.sum(np.abs(y_true))
    if denom == 0:
        return float("nan")
    return float(np.sum(np.abs(y_true - y_pred)) / denom * 100)


def forecast_bias(y_true, y_pred):
    """Positive = systematic over-forecasting (waste risk); negative = under-forecasting (shortage risk)."""
    denom = np.sum(y_true)
    if denom == 0:
        return float("nan")
    return float(np.sum(y_pred - y_true) / denom * 100)


def evaluate(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_pred = np.clip(y_pred, 0, None)  # a negative predicted demand is not meaningful
    return {
        "mae": round(mae(y_true, y_pred), 3),
        "rmse": round(rmse(y_true, y_pred), 3),
        "mape_pct": round(mape(y_true, y_pred), 2),
        "wape_pct": round(wape(y_true, y_pred), 2),
        "forecast_bias_pct": round(forecast_bias(y_true, y_pred), 2),
        "n": int(len(y_true)),
    }
