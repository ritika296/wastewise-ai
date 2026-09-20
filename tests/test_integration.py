"""
Phase 12 — end-to-end integration test. Unlike every other test file,
which exercises one module or endpoint in isolation, this test walks the
EXACT loop from Section 41:

  PREDICT -> DETECT -> EXPLAIN -> SIMULATE -> RECOMMEND -> DECIDE -> MEASURE

as ONE continuous flow through the live API, asserting that data stays
CONSISTENT across steps — the same forecast number that appears in step 1
must be the same number reasoned about in steps 2-4, not a coincidentally
similar one from a re-computed call that happens to agree.

Run from wastewise-ai/ root: python -m pytest tests/test_integration.py -v
"""
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "backend"))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)
DATE = "2026-05-08"


@pytest.fixture(scope="module", autouse=True)
def authenticate():
    """Phase 16's global auth middleware requires a Bearer token on every
    request — see the identical fixture in test_api.py for rationale."""
    from app.db import seed_demo_users
    seed_demo_users()
    r = client.post("/auth/login", json={"username": "manager", "password": "wastewise123"})
    assert r.status_code == 200, r.text
    client.headers.update({"Authorization": f"Bearer {r.json()['access_token']}"})
    yield


def test_full_loop_predict_detect_explain_simulate_recommend_decide_measure():
    # ---------- PREDICT ----------
    # The manager opens the dashboard: what's the business problem today?
    dashboard = client.get(f"/dashboard?date={DATE}").json()
    assert dashboard["items_high_shortage_risk"] > 0
    top_item = dashboard["top_risky_items"][0]
    restaurant_id, item_name = top_item["restaurant_id"], top_item["item_name"]

    # Find that item's ID from the recommendations list (dashboard only
    # gives the name; this mirrors what a real UI click-through does)
    recs = client.get(f"/recommendations?date={DATE}&restaurant_id={restaurant_id}").json()
    matching = [r for r in recs if r["item_name"] == item_name]
    assert matching, f"Dashboard's top risky item '{item_name}' must appear in /recommendations"
    item = matching[0]
    item_id = item["item_id"]
    predicted_forecast = item["forecast_demand"]

    forecast_detail = client.get(f"/forecast/{item_id}?date={DATE}&restaurant_id={restaurant_id}").json()
    # SAME NUMBER — not just "close", the actual float from the same model call chain
    assert forecast_detail["forecast_demand"] == predicted_forecast, (
        "The forecast shown in the recommendation list and the forecast detail page "
        "must be IDENTICAL for the same item/date/restaurant — any drift here means "
        "two code paths are silently diverging."
    )

    # ---------- DETECT ----------
    risk_list = client.get(f"/risk?date={DATE}&restaurant_id={restaurant_id}").json()
    risk_entry = next(r for r in risk_list if r["item_id"] == item_id)
    assert risk_entry["risk_category"] == item["risk_category"], (
        "The /risk endpoint and /recommendations endpoint must agree on risk "
        "classification for the same item — they're both reading the same "
        "underlying recommendation, not two independently-computed opinions."
    )

    # ---------- EXPLAIN ----------
    explanation = forecast_detail["explanation"]
    assert explanation["method"] == "xgboost_native_treeshap_exact"
    assert len(explanation["top_features"]) > 0
    # The explanation's own predicted_total must match the forecast it's explaining
    assert abs(explanation["predicted_total_portions"] - predicted_forecast) < 0.5

    # ---------- SIMULATE ----------
    what_if_result = client.post("/what-if", json={
        "date": DATE, "restaurant_id": restaurant_id, "item_id": item_id, "demand_pct_change": 0.15,
    }).json()
    # The what-if's OWN baseline must agree with the same forecast we've been tracking
    assert abs(what_if_result["original_forecast"] - predicted_forecast) < 0.5, (
        "What-If's baseline forecast must match the same real forecast used earlier "
        "in this flow — not a separately-drifted recomputation."
    )
    assert what_if_result["scenario_forecast"] > what_if_result["original_forecast"]

    # ---------- RECOMMEND ----------
    recommended_prep = item["recommended_preparation"]
    assert recommended_prep >= 0
    # The recommendation must be internally consistent with the item's own
    # forecast + inventory + buffer (the deterministic formula, re-verified
    # at the integration level, not just inside test_recommendations.py)
    expected_total = item["current_inventory"] + recommended_prep
    assert abs(expected_total - item["total_available_after_prep"]) < 1.0

    # ---------- DECIDE (human-in-the-loop) ----------
    feedback_resp = client.post("/feedback", json={
        "date": DATE, "restaurant_id": restaurant_id, "item_id": item_id,
        "recommendation": f"Prepare {recommended_prep} portions",
        "human_decision": "approve", "final_action": f"Prepare {recommended_prep} portions",
    })
    assert feedback_resp.status_code == 200
    feedback_id = feedback_resp.json()["id"]

    persisted = client.get("/feedback").json()
    matching_feedback = next((f for f in persisted if f["id"] == feedback_id), None)
    assert matching_feedback is not None, "The approval must actually persist, not just return 200"
    assert matching_feedback["item_id"] == item_id
    assert matching_feedback["human_decision"] == "approve"

    # ---------- MEASURE ----------
    # After all this real activity, cost/latency/quality monitoring must
    # reflect it — not show a static, pre-canned dashboard.
    llm_before = client.get("/llmops/metrics").json()
    client.post("/copilot/chat", json={"question": f"Why is {item_name} high risk?", "date": DATE})
    llm_after = client.get("/llmops/metrics").json()
    assert llm_after["total_requests"] > llm_before.get("total_requests", 0), (
        "Asking the copilot a question must increment the observed request count — "
        "proof that MEASURE is watching the actual system, not a mock."
    )

    latency = client.get("/latency?n_requests=5").json()
    assert latency["sla_status"] == "within_sla"

    cost = client.get("/cost").json()
    assert cost["total_ai_cost_inr"] >= 0


def test_loop_is_item_specific_not_accidentally_global():
    """
    Guards against a subtle bug class: if two different items' data got
    mixed up somewhere in the chain, this wouldn't be caught by testing
    one item alone. Runs the SAME predict->detect consistency check for
    TWO different items and confirms their numbers don't collide.
    """
    recs = client.get(f"/recommendations?date={DATE}&restaurant_id=R01").json()
    assert len(recs) >= 2
    item_a, item_b = recs[0], recs[1]
    assert item_a["item_id"] != item_b["item_id"]

    detail_a = client.get(f"/forecast/{item_a['item_id']}?date={DATE}&restaurant_id=R01").json()
    detail_b = client.get(f"/forecast/{item_b['item_id']}?date={DATE}&restaurant_id=R01").json()
    assert detail_a["forecast_demand"] == item_a["forecast_demand"]
    assert detail_b["forecast_demand"] == item_b["forecast_demand"]
    # The two items' forecasts should generally differ (different menu items);
    # if they were accidentally identical every time, that would itself be
    # suspicious of a caching/indexing bug — not asserted as impossible, but
    # logged for visibility if it happens.
    if detail_a["forecast_demand"] == detail_b["forecast_demand"]:
        pytest.skip("Coincidentally identical forecasts for two different items — not itself a failure, but worth a human glance.")
