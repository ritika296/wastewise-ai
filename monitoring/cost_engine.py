"""
AI cost engine (Phase 9 / Section 15). Combines:
  - LLM cost: REAL, pulled from llm/monitoring/request_logger.py's logged calls
  - Embedding cost: N/A for this product — WasteWise AI has no RAG/embedding
    layer (unlike a document-grounded copilot), stated explicitly rather
    than silently reported as zero without explanation
  - ML inference cost: an ILLUSTRATIVE compute-time-based estimate (real
    measured inference time x a documented assumed compute rate)
  - Infrastructure cost: an ILLUSTRATIVE flat assumption (documented)

Every illustrative figure is labeled as such in its own field name and in
`assumptions`, per Section 21/22's repeated instruction never to present
an assumption as a measurement.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from llm.monitoring.request_logger import get_metrics as get_llm_metrics
from monitoring.latency_engine import run_latency_benchmark

# --- Documented cost assumptions (illustrative — adjust for a real deployment) ---
ASSUMED_COMPUTE_RATE_INR_PER_CPU_SECOND = 0.008   # a small cloud VM's approximate cost
ASSUMED_INFRA_MONTHLY_INR = 4500                   # small API server + SQLite/Postgres + storage
ASSUMED_DAILY_REQUEST_VOLUME = 500                 # for infra-cost-per-request amortization (illustrative)

# Section 35: "cost threshold exceeded" must be a real, checkable
# scenario, not just a dashboard number nobody watches.
DAILY_COST_ALERT_THRESHOLD_INR = 500.0


def check_cost_threshold(daily_cost_inr: float, threshold_inr: float = DAILY_COST_ALERT_THRESHOLD_INR) -> dict:
    """Simple threshold alert — mirrors check_for_abnormal_latency's shape
    in monitoring/latency_engine.py, so cost and latency alerting follow
    the same pattern rather than two different alerting philosophies."""
    exceeded = daily_cost_inr > threshold_inr
    return {
        "status": "critical" if exceeded else "normal",
        "daily_cost_inr": daily_cost_inr, "threshold_inr": threshold_inr,
        "alerts": [f"Daily AI cost ₹{daily_cost_inr} exceeds the ₹{threshold_inr} alert threshold"] if exceeded else [],
    }


def compute_ml_inference_cost(n_requests: int, avg_ml_latency_ms: float) -> dict:
    total_cpu_seconds = (avg_ml_latency_ms / 1000) * n_requests
    cost = round(total_cpu_seconds * ASSUMED_COMPUTE_RATE_INR_PER_CPU_SECOND, 5)
    return {
        "total_cost_inr": cost, "cost_per_request_inr": round(cost / max(n_requests, 1), 6),
        "basis": "illustrative — measured inference time x assumed compute rate",
        "assumed_rate_inr_per_cpu_second": ASSUMED_COMPUTE_RATE_INR_PER_CPU_SECOND,
    }


def compute_infra_cost_per_request(daily_volume: int = ASSUMED_DAILY_REQUEST_VOLUME) -> dict:
    monthly_requests = daily_volume * 30
    per_request = round(ASSUMED_INFRA_MONTHLY_INR / max(monthly_requests, 1), 6)
    return {
        "monthly_infra_cost_inr": ASSUMED_INFRA_MONTHLY_INR, "assumed_daily_volume": daily_volume,
        "cost_per_request_inr": per_request,
        "basis": "illustrative — flat assumed monthly infra spend / assumed monthly request volume",
    }


def get_ai_cost_dashboard(n_recommendations_generated: int = 90) -> dict:
    """
    n_recommendations_generated: real count from the last recommendation
    batch (Phase 6/7), used for the "cost per recommendation" metric.
    """
    llm_metrics = get_llm_metrics(since_hours=24)
    llm_requests = llm_metrics.get("total_requests", 0)

    latency_bench = run_latency_benchmark(n_requests=20, include_llm=False)
    avg_ml_latency = latency_bench.get("ml_prediction", {}).get("mean_ms", 0) if latency_bench.get("status") != "no_data" else 0

    ml_cost = compute_ml_inference_cost(max(n_recommendations_generated, 1), avg_ml_latency)
    infra_cost = compute_infra_cost_per_request()

    llm_cost_total = llm_metrics.get("total_cost_inr", 0.0)
    embedding_cost_total = 0.0  # N/A — see module docstring

    total_ai_cost = round(llm_cost_total + ml_cost["total_cost_inr"] + infra_cost["cost_per_request_inr"] * max(llm_requests, 1), 5)
    total_requests = max(llm_requests, 1)

    return {
        "breakdown": {
            "llm_cost_inr": llm_cost_total,
            "embedding_cost_inr": embedding_cost_total,
            "ml_inference_cost_inr": ml_cost["total_cost_inr"],
            "infrastructure_cost_inr": round(infra_cost["cost_per_request_inr"] * total_requests, 5),
        },
        "total_ai_cost_inr": total_ai_cost,
        "unit_economics": {
            "cost_per_request_inr": round(total_ai_cost / total_requests, 6),
            "cost_per_recommendation_inr": round(
                (ml_cost["total_cost_inr"] + infra_cost["cost_per_request_inr"] * n_recommendations_generated)
                / max(n_recommendations_generated, 1), 6
            ),
            "cost_per_1000_requests_inr": round(total_ai_cost / total_requests * 1000, 3),
            "cost_per_user_inr": total_ai_cost,  # single-tenant PoC — see note
        },
        "cost_today_inr": llm_metrics.get("total_cost_inr", 0.0),
        "cost_alert": check_cost_threshold(
            llm_metrics.get("daily_cost_projection_inr", 0.0) + infra_cost["monthly_infra_cost_inr"] / 30
        ),
        "daily_cost_projection_inr": llm_metrics.get("daily_cost_projection_inr", 0.0) + (
            infra_cost["monthly_infra_cost_inr"] / 30
        ),
        "monthly_cost_projection_inr": llm_metrics.get("monthly_cost_projection_inr", 0.0) + infra_cost["monthly_infra_cost_inr"],
        "assumptions": {
            "ml_inference": ml_cost["basis"], "infrastructure": infra_cost["basis"],
            "embedding": "Not applicable — this product has no RAG/embedding layer",
            "cost_per_user": "Single-tenant PoC — 'user' == the one restaurant client; a multi-tenant "
                              "deployment would divide by actual active manager accounts",
        },
        "note": ("LLM cost is REAL (from logged requests). ML inference and infrastructure costs are "
                 "ILLUSTRATIVE, computed from real measured latency x a documented assumed rate — not "
                 "measured cloud billing, since this PoC runs locally with no cloud account attached."),
    }


def compare_llm_tiers() -> dict:
    """
    Section 28's cost/quality/latency trade-off table. Cost figures are
    REAL (the same rate card llm/client.py actually uses). Latency figures
    for live providers are PUBLISHED TYPICAL figures (explicitly labeled),
    since no API keys are configured in this environment to measure them
    directly — the template tier's latency IS measured (it's what this
    PoC actually runs).
    """
    from llm.client import COST_RATES

    tiers = [
        {
            "tier": "template (deterministic, no LLM)", "provider": "template",
            "typical_input_cost_per_1k_inr": COST_RATES["template"]["input"],
            "typical_output_cost_per_1k_inr": COST_RATES["template"]["output"],
            "typical_p95_latency_ms": "~5 (MEASURED — this PoC's actual default)",
            "quality_note": "Fully grounded by construction (echoes structured facts); no natural-language flexibility for open-ended questions.",
        },
        {
            "tier": "small (Groq Llama-3.1-8B class)", "provider": "groq",
            "typical_input_cost_per_1k_inr": COST_RATES["groq"]["input"],
            "typical_output_cost_per_1k_inr": COST_RATES["groq"]["output"],
            "typical_p95_latency_ms": "~400-800 (PUBLISHED TYPICAL — not measured in this environment, no API key configured)",
            "quality_note": "Fast and cheap; may need stricter prompting (v3-style hard rules) to stay reliably grounded.",
        },
        {
            "tier": "medium (OpenAI gpt-4o-mini class)", "provider": "openai",
            "typical_input_cost_per_1k_inr": COST_RATES["openai"]["input"],
            "typical_output_cost_per_1k_inr": COST_RATES["openai"]["output"],
            "typical_p95_latency_ms": "~800-1500 (PUBLISHED TYPICAL — not measured in this environment)",
            "quality_note": "Stronger natural-language handling for open-ended Ask-AI questions than the small tier.",
        },
        {
            "tier": "large (Anthropic Claude Haiku class)", "provider": "anthropic",
            "typical_input_cost_per_1k_inr": COST_RATES["anthropic"]["input"],
            "typical_output_cost_per_1k_inr": COST_RATES["anthropic"]["output"],
            "typical_p95_latency_ms": "~1000-2000 (PUBLISHED TYPICAL — not measured in this environment)",
            "quality_note": "Highest cost tier here; best suited if open-ended question variety is high and grounding must hold under looser prompting.",
        },
    ]
    return {
        "tiers": tiers,
        "recommendation": (
            "For WasteWise AI's actual usage pattern — mostly structured, template-answerable "
            "questions (item risk, priority lists, summaries) with occasional open-ended Ask-AI "
            "queries — the template engine already covers the majority of traffic at zero cost. "
            "If a live LLM is adopted, the small/Groq tier is the reasonable default given v3's "
            "hard-rule prompting was shown (Phase 8) to substantially close the quality gap "
            "without it. The larger tiers are worth the added cost mainly if Ask-AI's free-text "
            "question variety grows beyond what strict prompting reliably handles."
        ),
        "caveat": "Do not treat this as a universal ranking — Section 28 explicitly warns against "
                  "always picking the largest model; the right tier depends on actual question mix, "
                  "measured (not assumed) quality at each tier, and the latency budget for the use case.",
    }
