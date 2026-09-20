"""
Phase 13 — failure scenario tests (Section 35). Targets scenarios not
exercised by any earlier phase's tests: LLM timeout, model unavailable,
invalid what-if parameters, concurrent requests, and cost threshold
alerting. Every test here checks GRACEFUL degradation — a clean error or
a sensible fallback — never an unhandled exception or a hang.
"""
import sys
import time
import threading
from pathlib import Path
from unittest.mock import patch
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


# --- LLM timeout / unavailable ---------------------------------------

def test_llm_timeout_falls_back_gracefully():
    """A real (simulated) network timeout must not crash the request —
    it must fall back to the template engine and say so."""
    import llm.client as llm_client

    def raise_timeout(*args, **kwargs):
        import httpx
        raise httpx.TimeoutException("simulated timeout")

    # A real API key must be present, or the code takes the earlier
    # "no key configured" fallback branch and never reaches _call_provider
    # at all — which is what happened the first time this test was
    # written, silently testing the wrong code path.
    with patch.object(llm_client, "_call_provider", side_effect=raise_timeout):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "fake-key-for-test"}):
            result = llm_client.call_llm(
                "system", "user", {"facts": {"a": 1}}, provider="openai", prompt_version="v3"
            )
    assert result["fallback_used"] is True
    assert result["error"] is True
    assert "timeout" in result["error_detail"].lower()
    assert len(result["text"]) > 0  # still produced a real answer, not empty


def test_copilot_chat_endpoint_survives_llm_provider_exception():
    """End-to-end: even if the underlying LLM call blows up entirely, the
    /copilot/chat endpoint must return 200 with a usable answer, not 500."""
    import llm.client as llm_client

    def raise_connection_error(*args, **kwargs):
        raise ConnectionError("simulated connection refused")

    with patch.object(llm_client, "_call_provider", side_effect=raise_connection_error):
        with patch.object(llm_client, "LLM_PROVIDER", "openai"):
            with patch.dict("os.environ", {"OPENAI_API_KEY": "fake-key-for-test"}):
                r = client.post("/copilot/chat", json={"question": "Which items should I prepare more of?", "date": DATE})
    assert r.status_code == 200
    assert len(r.json()["answer"]) > 0


# --- Model unavailable --------------------------------------------------

def test_missing_model_artifact_raises_informative_error_not_crash():
    """If the champion model file is missing entirely, the failure must
    be a clear, catchable error — not an unhandled crash deep in joblib.
    Model loading in THIS project happens in
    ml.inference.pipeline._load_shared_artifacts(), not a
    ml.monitoring.registry function of that name (that was an incorrect
    assumption carried over from a different project's module shape —
    corrected here rather than left in)."""
    import joblib
    from ml.inference import pipeline as pl
    with patch.object(joblib, "load", side_effect=FileNotFoundError("model file not found")):
        pl._load_shared_artifacts.cache_clear()
        with pytest.raises(FileNotFoundError):
            pl._load_shared_artifacts()
    pl._load_shared_artifacts.cache_clear()  # restore real state for other tests


def test_corrupted_registry_file_does_not_silently_return_garbage():
    """A malformed registry.json must fail loudly (JSONDecodeError), not
    return None or an empty dict that downstream code silently accepts.
    Uses a real temporarily-corrupted file rather than mocking `open`,
    since Path.read_text() doesn't reliably route through a
    builtins.open patch (found the hard way — the mock-based first
    attempt at this test silently didn't intercept anything)."""
    import json
    from ml.monitoring import registry as reg
    original = reg.REGISTRY_PATH.read_text()
    try:
        reg.REGISTRY_PATH.write_text("{not valid json")
        with pytest.raises(json.JSONDecodeError):
            reg._load_registry()
    finally:
        reg.REGISTRY_PATH.write_text(original)


# --- Invalid what-if parameters ------------------------------------------

def test_what_if_extreme_negative_demand_does_not_produce_negative_forecast():
    r = client.post("/what-if", json={
        "date": DATE, "restaurant_id": "R01", "item_id": "I01", "demand_pct_change": -5.0,
    })
    assert r.status_code == 200
    assert r.json()["scenario_forecast"] >= 0


def test_what_if_extreme_positive_demand_does_not_crash():
    r = client.post("/what-if", json={
        "date": DATE, "restaurant_id": "R01", "item_id": "I01", "demand_pct_change": 100.0,
    })
    assert r.status_code == 200
    assert r.json()["scenario_forecast"] > r.json()["original_forecast"]


def test_what_if_invalid_weather_scenario_ignored_not_crashed():
    r = client.post("/what-if", json={
        "date": DATE, "restaurant_id": "R01", "item_id": "I01", "weather_scenario": "Tornado",
    })
    assert r.status_code == 200
    assert not any("Tornado" in c for c in r.json()["applied_changes"])


def test_what_if_negative_safety_buffer_handled():
    r = client.post("/what-if", json={
        "date": DATE, "restaurant_id": "R01", "item_id": "I01", "safety_buffer_pct": -0.5,
    })
    assert r.status_code == 200
    assert r.json()["scenario_recommendation"]["recommended_preparation"] >= 0


def test_what_if_malicious_string_item_id_handled_safely():
    r = client.post("/what-if", json={
        "date": DATE, "restaurant_id": "R01", "item_id": "'; DROP TABLE feedback_log; --",
    })
    assert r.status_code == 404
    assert "Traceback" not in r.text


def test_what_if_empty_string_fields_rejected_or_handled():
    r = client.post("/what-if", json={"date": "", "restaurant_id": "", "item_id": ""})
    assert r.status_code in (404, 422)
    assert "Traceback" not in r.text


# --- Concurrent requests --------------------------------------------------

def test_concurrent_requests_do_not_corrupt_shared_cached_state():
    results = []
    errors = []

    def make_request(item_id):
        try:
            r = client.get(f"/forecast/{item_id}?date={DATE}&restaurant_id=R01")
            results.append((item_id, r.status_code, r.json() if r.status_code == 200 else None))
        except Exception as e:
            errors.append(str(e))

    items = [f"I{str(i).zfill(2)}" for i in range(1, 21)]
    threads = [threading.Thread(target=make_request, args=(item,)) for item in items]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert not errors, f"Concurrent requests raised exceptions: {errors}"
    assert len(results) == 20
    for item_id, status, body in results:
        assert status == 200
        assert body["item_id"] == item_id


def test_concurrent_feedback_writes_all_persist():
    def submit(i):
        client.post("/feedback", json={
            "date": DATE, "restaurant_id": "R01", "item_id": "I01",
            "recommendation": f"test rec {i}", "human_decision": "approve", "final_action": f"test action {i}",
        })

    before = len(client.get("/feedback?limit=200").json())
    threads = [threading.Thread(target=submit, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    after = len(client.get("/feedback?limit=200").json())
    assert after - before == 20


# --- Cost threshold ----------------------------------------------------

def test_cost_threshold_alert_fires_when_exceeded():
    from monitoring.cost_engine import check_cost_threshold
    result = check_cost_threshold(daily_cost_inr=1000.0, threshold_inr=500.0)
    assert result["status"] == "critical"
    assert len(result["alerts"]) == 1


def test_cost_threshold_alert_silent_when_under():
    from monitoring.cost_engine import check_cost_threshold
    result = check_cost_threshold(daily_cost_inr=50.0, threshold_inr=500.0)
    assert result["status"] == "normal"
    assert result["alerts"] == []


def test_cost_endpoint_includes_alert_field():
    r = client.get("/cost")
    assert "cost_alert" in r.json()
    assert r.json()["cost_alert"]["status"] in ("normal", "critical")


# --- High latency simulation ----------------------------------------------

def test_slow_llm_call_still_returns_and_reports_real_latency():
    import llm.client as llm_client

    def slow_call(*args, **kwargs):
        time.sleep(0.3)
        return "A deliberately slow response."

    with patch.object(llm_client, "_call_provider", side_effect=slow_call):
        with patch.dict("os.environ", {"OPENAI_API_KEY": "fake-key"}):
            result = llm_client.call_llm("system", "user", {"facts": {}}, provider="openai", prompt_version="v3")
    assert result["latency_ms"] >= 300
    assert result["fallback_used"] is False
    assert result["error"] is False


# --- Missing / bad data at the serving layer ------------------------------

def test_forecast_for_date_outside_dataset_range_returns_clean_404():
    r = client.get(f"/forecast/I01?date=1999-01-01&restaurant_id=R01")
    assert r.status_code == 404
    assert "Traceback" not in r.text


def test_dashboard_for_nonexistent_date_returns_clean_404_not_500():
    r = client.get("/dashboard?date=2099-12-31")
    assert r.status_code == 404
    assert "Traceback" not in r.text
