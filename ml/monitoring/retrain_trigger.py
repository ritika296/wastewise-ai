"""
Retraining trigger (Phase 5 / Section 17). A simple, explicit rule engine
— not a black-box score — that decides whether current evidence justifies
flagging the champion model for retraining. Every trigger fired states
exactly which rule fired and why, so a human reviewing this can agree or
disagree with the specific reason, not just a verdict.
"""
from dataclasses import dataclass, field

FEATURE_DRIFT_CRITICAL_PSI = 0.25
PREDICTION_DRIFT_CRITICAL_PSI = 0.25
WAPE_DEGRADATION_RELATIVE_THRESHOLD = 0.20  # 20% relative increase in WAPE vs. training-time value


@dataclass
class RetrainDecision:
    trigger_retrain: bool
    reasons: list = field(default_factory=list)
    status: str = "healthy"

    def to_dict(self):
        return {"trigger_retrain": self.trigger_retrain, "reasons": self.reasons, "status": self.status}


def evaluate_retrain_triggers(feature_drift_report: dict, prediction_drift_report: dict,
                               training_time_wape: float, recent_observed_wape: float) -> RetrainDecision:
    reasons = []

    if feature_drift_report["max_psi"] >= FEATURE_DRIFT_CRITICAL_PSI:
        worst_features = sorted(
            feature_drift_report["per_feature"].items(), key=lambda kv: kv[1]["psi"], reverse=True
        )[:3]
        worst_summary = ", ".join(f"{k}={v['psi']}" for k, v in worst_features)
        reasons.append(
            f"Feature drift critical: max PSI {feature_drift_report['max_psi']} >= {FEATURE_DRIFT_CRITICAL_PSI} "
            f"(worst: {worst_summary})"
            + (" [SIMULATED SCENARIO]" if feature_drift_report.get("simulated") else "")
        )

    if prediction_drift_report["psi"] >= PREDICTION_DRIFT_CRITICAL_PSI:
        reasons.append(
            f"Prediction drift critical: PSI {prediction_drift_report['psi']} >= {PREDICTION_DRIFT_CRITICAL_PSI} "
            f"(mean shifted {prediction_drift_report['reference_mean']} -> {prediction_drift_report['current_mean']})"
            + (" [SIMULATED SCENARIO]" if prediction_drift_report.get("simulated") else "")
        )

    if training_time_wape and recent_observed_wape:
        relative_change = (recent_observed_wape - training_time_wape) / training_time_wape
        if relative_change >= WAPE_DEGRADATION_RELATIVE_THRESHOLD:
            reasons.append(
                f"Forecast accuracy degraded: WAPE {training_time_wape}% (training) -> "
                f"{recent_observed_wape}% (recent), a {relative_change:.0%} relative increase "
                f">= {WAPE_DEGRADATION_RELATIVE_THRESHOLD:.0%} threshold"
            )

    trigger = len(reasons) > 0
    status = "critical" if trigger else "healthy"
    return RetrainDecision(trigger_retrain=trigger, reasons=reasons, status=status)
