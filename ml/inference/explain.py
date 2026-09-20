"""
Explainability (Section 20). Uses the champion model's NATIVE, EXACT
feature attribution — XGBoost's built-in TreeSHAP (`pred_contribs=True`)
when the champion is XGBoost, which is exact (not approximated, not
sampled) and directly supported by the library. If a future champion is
NOT a tree model with native contributions, `explain_prediction()` falls
back to reporting global feature importance only, with an explicit note
that per-instance percentage contributions are not available for that
model type — per Section 20's explicit instruction never to fabricate
feature contributions the method doesn't actually support.
"""
import numpy as np
import pandas as pd
import xgboost as xgb

FRIENDLY_NAMES = {
    "lag_1": "Yesterday's sales", "lag_7": "Same day last week", "lag_14": "Same day two weeks ago",
    "rolling_mean_7": "7-day average demand", "rolling_mean_14": "14-day average demand",
    "rolling_mean_28": "28-day average demand", "selling_price": "Price", "discount": "Discount",
    "promotion_flag": "Promotion running", "day_of_week": "Day of week", "weekend_flag": "Weekend",
    "holiday_flag": "Holiday", "temperature": "Temperature", "special_event": "Special event", "month": "Month",
}


def _friendly(feature_name: str) -> str:
    if feature_name in FRIENDLY_NAMES:
        return FRIENDLY_NAMES[feature_name]
    for prefix, label in [("restaurant_id_", "Location: "), ("item_id_", "Item: "),
                           ("category_", "Category: "), ("weather_condition_", "Weather: ")]:
        if feature_name.startswith(prefix):
            return label + feature_name[len(prefix):]
    return feature_name


def explain_prediction(model, feature_row: pd.DataFrame, features: list, top_k: int = 5) -> dict:
    """
    Returns real, exact per-feature contributions (in units of the
    prediction, i.e. "portions") for XGBoost champions. Each contribution
    is also expressed as a % of the model's base value + contributions
    (the actual predicted total) — these percentages ARE mathematically
    real, derived from TreeSHAP's additive decomposition, not estimated.
    """
    if not isinstance(model, xgb.XGBRegressor):
        return {
            "method": "global_importance_only",
            "note": ("Per-instance contribution percentages are not available for this model "
                     "type with the method used here — showing global feature importance only, "
                     "not fabricated per-instance percentages."),
            "top_features": _global_importance(model, features, top_k),
        }

    booster = model.get_booster()
    dmatrix = xgb.DMatrix(feature_row[features], feature_names=features)
    contribs = booster.predict(dmatrix, pred_contribs=True)[0]  # shape: (n_features + 1,) last = bias
    bias = contribs[-1]
    feature_contribs = contribs[:-1]
    total = bias + feature_contribs.sum()  # == model's raw prediction

    pairs = sorted(zip(features, feature_contribs), key=lambda x: abs(x[1]), reverse=True)[:top_k]
    top_features = []
    for feat, contrib in pairs:
        pct_of_total = (contrib / total * 100) if total != 0 else 0.0
        top_features.append({
            "feature": feat, "label": _friendly(feat),
            "value": round(float(feature_row[feat].iloc[0]), 2),
            "contribution_portions": round(float(contrib), 2),
            "contribution_pct_of_forecast": round(float(pct_of_total), 1),
            "direction": "increases" if contrib > 0 else "decreases",
        })
    return {
        "method": "xgboost_native_treeshap_exact",
        "base_value_portions": round(float(bias), 2),
        "predicted_total_portions": round(float(total), 2),
        "top_features": top_features,
    }


def _global_importance(model, features: list, top_k: int) -> list:
    if not hasattr(model, "feature_importances_"):
        return []
    pairs = sorted(zip(features, model.feature_importances_), key=lambda x: x[1], reverse=True)[:top_k]
    return [{"feature": f, "label": _friendly(f), "importance": round(float(v), 4)} for f, v in pairs]
