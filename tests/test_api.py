"""
Phase 10 tests (API layer). Run from wastewise-ai/backend/: python -m pytest ../tests/test_api.py -v
Uses FastAPI's TestClient — no live server needed to run these.
"""
import sys
from pathlib import Path
import pytest

ROOT = Path(__file__).resolve().parent.parent
BACKEND = ROOT / "backend"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(BACKEND))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)
DEMO_DATE = "2026-05-08"


@pytest.fixture(scope="module", autouse=True)
def authenticate():
    """
    Phase 16 added a global auth middleware — every one of these
    previously-passing tests now needs a Bearer token. Seeds the demo
    accounts directly (idempotent — safe even if the app's own startup
    lifespan already did it) rather than relying on TestClient's lifespan
    context, then logs in and attaches the token to every request this
    module's shared client makes.
    """
    from app.db import seed_demo_users
    seed_demo_users()
    r = client.post("/auth/login", json={"username": "manager", "password": "wastewise123"})
    assert r.status_code == 200, r.text
    client.headers.update({"Authorization": f"Bearer {r.json()['access_token']}"})
    yield


def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["product"] == "WasteWise AI"


def test_dashboard_returns_real_kpis():
    r = client.get(f"/dashboard?date={DEMO_DATE}")
    assert r.status_code == 200
    body = r.json()
    assert body["items_scored"] == 90
    assert body["items_high_shortage_risk"] > 0
    assert "model_wape_pct" in body


def test_dashboard_missing_date_returns_422():
    r = client.get("/dashboard")
    assert r.status_code == 422


def test_forecast_list():
    r = client.get(f"/forecast?date={DEMO_DATE}")
    assert r.status_code == 200
    assert len(r.json()) == 90


def test_forecast_filtered_by_restaurant():
    r = client.get(f"/forecast?date={DEMO_DATE}&restaurant_id=R01")
    body = r.json()
    assert all(row["restaurant_id"] == "R01" for row in body)
    assert len(body) == 30  # 30 menu items at one restaurant


def test_forecast_detail_has_real_explanation():
    r = client.get(f"/forecast/I01?date={DEMO_DATE}&restaurant_id=R01")
    assert r.status_code == 200
    body = r.json()
    assert body["explanation"]["method"] == "xgboost_native_treeshap_exact"
    assert len(body["explanation"]["top_features"]) > 0


def test_forecast_detail_unknown_item_returns_404_not_500():
    r = client.get(f"/forecast/NONEXISTENT?date={DEMO_DATE}&restaurant_id=R01")
    assert r.status_code == 404
    assert "detail" in r.json()
    assert "Traceback" not in str(r.json())  # never expose a stack trace


def test_risk_endpoint():
    r = client.get(f"/risk?date={DEMO_DATE}")
    assert r.status_code == 200
    assert len(r.json()) == 90


def test_risk_filtered_by_tier():
    r = client.get(f"/risk?date={DEMO_DATE}&tier=HIGH")
    body = r.json()
    assert all(row["shortage_risk"] == "HIGH" or row["waste_risk"] == "HIGH" for row in body)


def test_recommendations_endpoint():
    r = client.get(f"/recommendations?date={DEMO_DATE}")
    assert r.status_code == 200
    assert len(r.json()) == 90


def test_what_if_endpoint():
    r = client.post("/what-if", json={
        "date": DEMO_DATE, "restaurant_id": "R01", "item_id": "I01", "demand_pct_change": 0.2,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["forecast_delta_pct"] == pytest.approx(20.0, abs=0.5)


def test_what_if_missing_required_field_returns_422():
    r = client.post("/what-if", json={"date": DEMO_DATE})
    assert r.status_code == 422


def test_what_if_unknown_item_returns_404():
    r = client.post("/what-if", json={
        "date": DEMO_DATE, "restaurant_id": "R01", "item_id": "NOPE",
    })
    assert r.status_code == 404


def test_copilot_chat_endpoint():
    r = client.post("/copilot/chat", json={"question": "Which items should I prepare more of?", "date": DEMO_DATE})
    assert r.status_code == 200
    assert r.json()["intent"] == "shortage_priority_list"


def test_copilot_chat_question_too_short_returns_422():
    r = client.post("/copilot/chat", json={"question": "hi", "date": DEMO_DATE})
    assert r.status_code == 422


def test_mlops_models_shows_both_champions_honestly():
    r = client.get("/mlops/models")
    body = r.json()
    assert body["phase4_naive_pick"] == "gradient_boosting"
    assert body["formal_registry_champion"] == "xgboost"
    assert body["phase4_naive_pick"] != body["formal_registry_champion"]  # the documented discrepancy


def test_mlops_drift_real_by_default():
    r = client.get("/mlops/drift")
    assert r.status_code == 200
    assert r.json()["drift_report"]["simulated"] is False


def test_mlops_drift_simulated_when_requested():
    r = client.get("/mlops/drift?simulated=true")
    body = r.json()
    assert body["drift_report"]["simulated"] is True
    assert "SIMULATED" in body["drift_report"]["simulation_description"]


def test_llmops_metrics_endpoint():
    client.post("/copilot/chat", json={"question": "Give me a summary.", "date": DEMO_DATE})
    r = client.get("/llmops/metrics")
    assert r.status_code == 200
    assert r.json()["total_requests"] >= 1


def test_cost_endpoint():
    r = client.get("/cost")
    assert r.status_code == 200
    assert "illustrative" in r.json()["assumptions"]["ml_inference"].lower()


def test_cost_tiers_endpoint():
    r = client.get("/cost/tiers")
    assert len(r.json()["tiers"]) == 4


def test_latency_endpoint():
    r = client.get("/latency?n_requests=5")
    assert r.status_code == 200
    assert r.json()["n_requests_measured"] == 5


def test_data_quality_endpoint():
    r = client.get("/data-quality")
    assert r.status_code == 200
    assert r.json()["total_records"] > 40000


def test_feedback_roundtrip():
    post_resp = client.post("/feedback", json={
        "date": DEMO_DATE, "restaurant_id": "R01", "item_id": "I01",
        "recommendation": "Prepare 45", "human_decision": "approve", "final_action": "Prepare 45",
    })
    assert post_resp.status_code == 200
    feedback_id = post_resp.json()["id"]

    get_resp = client.get("/feedback")
    assert get_resp.status_code == 200
    assert any(f["id"] == feedback_id for f in get_resp.json())


def test_feedback_invalid_decision_rejected():
    r = client.post("/feedback", json={
        "date": DEMO_DATE, "restaurant_id": "R01", "item_id": "I01",
        "recommendation": "x", "human_decision": "maybe", "final_action": "x",
    })
    assert r.status_code == 422  # pattern validation: must be approve|reject|modify


def test_no_endpoint_ever_returns_raw_stack_trace():
    """Section 34: never expose a stack trace. Spot-check a few error-inducing calls."""
    responses = [
        client.get(f"/forecast/BAD?date={DEMO_DATE}&restaurant_id=R01"),
        client.post("/what-if", json={"date": DEMO_DATE, "restaurant_id": "BAD", "item_id": "BAD"}),
    ]
    for r in responses:
        assert "Traceback" not in r.text
        assert "File \"" not in r.text


def test_dashboard_trend_returns_real_multi_day_points():
    r = client.get(f"/dashboard/trend?days=5&end_date={DEMO_DATE}")
    assert r.status_code == 200
    body = r.json()
    assert body["n_days"] == 5
    dates = [p["date"] for p in body["points"]]
    assert dates == sorted(dates)  # chronological order
    assert dates[-1] == DEMO_DATE


def test_business_impact_roi_formula_is_correct():
    r = client.get("/business-impact?monthly_food_waste_inr=100000&waste_reduction_pct=10"
                    "&monthly_stockout_loss_inr=0&stockout_reduction_pct=0"
                    "&ai_monthly_cost_inr=1000&implementation_cost_inr=10000")
    body = r.json()
    # monthly savings = 100000 * 0.10 = 10000; annual = 120000
    assert body["estimated_monthly_savings_inr"] == pytest.approx(10000)
    assert body["estimated_annual_savings_inr"] == pytest.approx(120000)
    annual_investment = 1000 * 12 + 10000
    expected_roi = (120000 - annual_investment) / annual_investment * 100
    assert body["roi_pct"] == pytest.approx(expected_roi, rel=0.01)


def test_business_impact_labeled_illustrative():
    r = client.get("/business-impact")
    assert "Illustrative PoC estimate" in r.json()["label"]


# --- N. Bring Your Own Data upload endpoint -------------------------------

def test_data_upload_rejects_non_csv():
    r = client.post("/data/upload", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 400


def test_data_upload_rejects_missing_columns():
    r = client.post("/data/upload", files={"file": ("bad.csv", b"date,item_name\n2026-01-01,X\n", "text/csv")})
    assert r.status_code == 400
    assert "units_sold" in r.json()["detail"]


def test_data_upload_returns_forecast_for_valid_csv():
    csv_content = (
        "date,item_name,units_sold,restaurant_location\n"
        + "\n".join(f"2026-01-{d:02d},Tea,{20 + (d % 5)},Downtown" for d in range(1, 26))
    )
    r = client.post("/data/upload", files={"file": ("sales.csv", csv_content.encode(), "text/csv")})
    assert r.status_code == 200
    body = r.json()
    assert body["n_items"] == 1
    assert body["items"][0]["item_name"] == "Tea"
    assert body["items"][0]["forecast_units"] is not None


def test_data_upload_sample_template_is_valid_csv_with_required_columns():
    r = client.get("/data/upload/sample-template")
    assert r.status_code == 200
    header = r.text.splitlines()[0]
    for col in ["date", "item_name", "units_sold"]:
        assert col in header


def test_data_upload_requires_auth():
    r = requests_no_auth_post()
    assert r.status_code == 401


def requests_no_auth_post():
    # Uses a fresh client with no Authorization header, unlike the module-level
    # `client` which has a Bearer token attached by the `authenticate` fixture.
    from fastapi.testclient import TestClient
    from app.main import app as _app
    bare_client = TestClient(_app)
    return bare_client.post("/data/upload", files={"file": ("x.csv", b"date,item_name,units_sold\n2026-01-01,X,5\n", "text/csv")})
