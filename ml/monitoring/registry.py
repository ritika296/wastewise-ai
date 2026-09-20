"""
Model registry (Phase 5 / Section 17). Wraps the champion_model.json
pointer written by Phase 4 with formal versioning and a PROMOTION RULE
that a new candidate must clear before it can replace the serving
champion — "do not automatically promote models without evaluation
criteria" is enforced here in code, not just stated in a doc.

Versions are recorded in models/registry.json (append-only history);
models/champion_model.json (Phase 4's output) is treated as the *current*
pointer this module manages going forward.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = ROOT / "models"
REGISTRY_PATH = MODELS_DIR / "registry.json"

# Promotion thresholds — a challenger must clear ALL of these to be promoted.
# Deliberately conservative: a marginal accuracy win doesn't automatically win
# if it comes with materially worse latency or bias.
MIN_WAPE_IMPROVEMENT_PCT = 0.5     # challenger's WAPE must be at least this many points better...
MAX_LATENCY_REGRESSION_MULT = 2.0  # ...or not regress latency by more than 2x...
MAX_BIAS_REGRESSION_PP = 1.0       # ...or not regress |bias| by more than 1 percentage point


def _load_registry() -> dict:
    if REGISTRY_PATH.exists():
        return json.loads(REGISTRY_PATH.read_text())
    return {"versions": [], "current_champion_version": None}


def _save_registry(reg: dict):
    REGISTRY_PATH.write_text(json.dumps(reg, indent=2, default=str))


def register_version(model_name: str, metrics: dict, artifact_path: str,
                      dataset_version: str = "v1", notes: str = "") -> dict:
    """Records a new trained candidate. Does NOT promote it — that's a separate,
    explicit step via promote_if_better()."""
    reg = _load_registry()
    version_id = f"{model_name}-v{sum(1 for v in reg['versions'] if v['model_name'] == model_name) + 1}"
    entry = {
        "version_id": version_id, "model_name": model_name, "metrics": metrics,
        "artifact_path": artifact_path, "dataset_version": dataset_version,
        "notes": notes, "registered_at": datetime.now(timezone.utc).isoformat(),
        "status": "registered",  # registered -> champion (via promotion) | retired
    }
    reg["versions"].append(entry)
    _save_registry(reg)
    return entry


def get_champion() -> dict | None:
    reg = _load_registry()
    if not reg["current_champion_version"]:
        return None
    return next((v for v in reg["versions"] if v["version_id"] == reg["current_champion_version"]), None)


def promote_if_better(candidate_version_id: str) -> dict:
    """
    Evaluates a registered candidate against the current champion using the
    thresholds above. Returns {"promoted": bool, "reason": str}. This is the
    ONLY path by which current_champion_version changes.
    """
    reg = _load_registry()
    candidate = next((v for v in reg["versions"] if v["version_id"] == candidate_version_id), None)
    if candidate is None:
        return {"promoted": False, "reason": f"No such version: {candidate_version_id}"}

    champion = get_champion()
    if champion is None:
        # First model ever registered — becomes champion automatically only
        # because there is no incumbent to compare against, not as a shortcut.
        reg["current_champion_version"] = candidate["version_id"]
        candidate["status"] = "champion"
        _save_registry(reg)
        return {"promoted": True, "reason": "No existing champion — first registered model becomes champion by default."}

    c_wape = candidate["metrics"].get("wape_pct")
    ch_wape = champion["metrics"].get("wape_pct")
    c_latency = candidate["metrics"].get("inference_ms_per_1000_rows", 0)
    ch_latency = champion["metrics"].get("inference_ms_per_1000_rows", 0.001)
    c_bias = abs(candidate["metrics"].get("forecast_bias_pct", 0))
    ch_bias = abs(champion["metrics"].get("forecast_bias_pct", 0))

    wape_improved = (ch_wape - c_wape) >= MIN_WAPE_IMPROVEMENT_PCT
    latency_ok = c_latency <= ch_latency * MAX_LATENCY_REGRESSION_MULT
    bias_ok = (c_bias - ch_bias) <= MAX_BIAS_REGRESSION_PP

    if wape_improved and latency_ok and bias_ok:
        for v in reg["versions"]:
            if v["version_id"] == champion["version_id"]:
                v["status"] = "retired"
        candidate["status"] = "champion"
        reg["current_champion_version"] = candidate["version_id"]
        _save_registry(reg)
        return {
            "promoted": True,
            "reason": (f"WAPE improved {ch_wape}% -> {c_wape}% (>= {MIN_WAPE_IMPROVEMENT_PCT}pp threshold), "
                       f"latency {c_latency}ms within {MAX_LATENCY_REGRESSION_MULT}x of champion, "
                       f"bias regression {c_bias - ch_bias:+.2f}pp within {MAX_BIAS_REGRESSION_PP}pp threshold."),
        }

    reasons = []
    if not wape_improved:
        reasons.append(f"WAPE improvement {ch_wape - c_wape:.2f}pp < required {MIN_WAPE_IMPROVEMENT_PCT}pp")
    if not latency_ok:
        reasons.append(f"latency {c_latency}ms exceeds {MAX_LATENCY_REGRESSION_MULT}x champion's {ch_latency}ms")
    if not bias_ok:
        reasons.append(f"bias regression {c_bias - ch_bias:.2f}pp exceeds {MAX_BIAS_REGRESSION_PP}pp threshold")
    _save_registry(reg)  # candidate stays "registered", not promoted
    return {"promoted": False, "reason": "Not promoted: " + "; ".join(reasons)}


def list_versions(model_name: str = None) -> list:
    reg = _load_registry()
    versions = reg["versions"]
    if model_name:
        versions = [v for v in versions if v["model_name"] == model_name]
    return versions
