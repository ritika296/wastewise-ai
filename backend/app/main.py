"""
WasteWise AI backend — Section 28's API surface, wired to every module
built in Phases 2-9. The champion model and shared artifacts are warmed
at STARTUP (see `warm_up()`), not on the first user request — this is
the fix Phase 9 found necessary for RetainIQ and flagged for here; it's
built in from the start this time, not discovered after the fact.

Run: uvicorn app.main:app --reload --port 8001   (from backend/)
"""
import sys
import time
from pathlib import Path
from datetime import date as date_cls

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, Depends, HTTPException, Query, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from app.schemas import WhatIfRequest, CopilotChatRequest, FeedbackRequest, LoginRequest, RegisterRequest
from app.db import init_db, get_session, seed_demo_users, FeedbackLog, User
from app.auth import (
    hash_password, verify_password, create_access_token, decode_token,
    get_current_user, require_role,
)

from ml.inference.pipeline import generate_recommendations_for_date, champion_model_and_features, get_feature_row
from ml.inference.explain import explain_prediction
from ml.inference.what_if import run_what_if as _run_what_if
from ml.inference.custom_upload_pipeline import run_upload_pipeline, UploadValidationError
from ml.monitoring.registry import list_versions, get_champion
from ml.monitoring.drift import check_feature_drift, simulate_drift_scenario
from ml.monitoring.retrain_trigger import evaluate_retrain_triggers
from ml.preprocessing.validate import validate as validate_data_quality

from llm.chains.copilot import answer_question as _answer_question
from llm.monitoring.request_logger import get_metrics as get_llm_metrics
from llm.evaluation.eval_suite import compare_prompt_versions

from monitoring.cost_engine import get_ai_cost_dashboard, compare_llm_tiers
from monitoring.latency_engine import run_latency_benchmark

import json
import numpy as np
import pandas as pd

from contextlib import asynccontextmanager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Loads the champion model, feature-engineered parquet files, and item
    volatility stats ONCE at process startup, not on the first real
    request. Phase 9 measured a 14x cold-start penalty (132ms -> 9ms)
    from exactly this kind of lazy loading in RetainIQ's SHAP explainer
    and again in this project's own latency benchmark — this fixes it
    proactively rather than waiting to rediscover it in production.
    """
    t0 = time.perf_counter()
    champion_model_and_features()  # triggers ml.inference.pipeline._load_shared_artifacts()
    generate_recommendations_for_date("2026-05-08")  # also warms the recommendation path
    seed_demo_users()  # Phase 16 — idempotent, seeds 4 demo accounts only if users table is empty
    print(f"[startup] Warmed shared artifacts in {(time.perf_counter() - t0) * 1000:.0f}ms")
    yield


app = FastAPI(title="WasteWise AI API", version="1.0.0",
              description="Predict demand. Reduce waste. Optimize every meal.",
              lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

init_db()

PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"

# --------------------------------------------------------------------- #
# Phase 16 — global authentication middleware.
#
# Every endpoint requires a valid Bearer token EXCEPT the ones listed here
# (the API root, docs, and the two auth endpoints themselves — you can't
# require a token to get your first token). This is an allow-list, not a
# deny-list, deliberately: a new endpoint added later is protected by
# default unless someone explicitly opts it out here.
# --------------------------------------------------------------------- #
PUBLIC_PATHS = {"/", "/docs", "/openapi.json", "/redoc", "/auth/login", "/auth/register"}


@app.middleware("http")
async def auth_middleware(request: Request, call_next):
    if request.method == "OPTIONS" or request.url.path in PUBLIC_PATHS:
        return await call_next(request)
    auth_header = request.headers.get("Authorization", "")
    if not auth_header.startswith("Bearer "):
        return JSONResponse(status_code=401, content={"detail": "Not authenticated — missing Bearer token"})
    token = auth_header.removeprefix("Bearer ").strip()
    try:
        payload = decode_token(token)
        request.state.user = {"username": payload.get("sub"), "role": payload.get("role", "manager")}
    except HTTPException as e:
        return JSONResponse(status_code=e.status_code, content={"detail": e.detail})
    return await call_next(request)


# --------------------------------------------------------------------- #
# Authentication endpoints
# --------------------------------------------------------------------- #
@app.post("/auth/login")
def login(req: LoginRequest, db: Session = Depends(get_session)):
    user = db.query(User).filter(User.username == req.username).first()
    # Login-enumeration protection: whether the username doesn't exist, or
    # exists but the password is wrong, the client sees the SAME error
    # message. Verifying against a hash even on a "no such user" outcome
    # (rather than short-circuiting immediately) keeps the response time
    # profile close between the two cases as well.
    if user is None:
        verify_password(req.password, hash_password("dummy-constant-time-padding"))
        raise HTTPException(status_code=401, detail="Incorrect username or password")
    if not verify_password(req.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Incorrect username or password")

    token = create_access_token(user.username, user.role)
    return {"access_token": token, "token_type": "bearer", "username": user.username, "role": user.role}


@app.post("/auth/register")
def register(req: RegisterRequest, db: Session = Depends(get_session)):
    existing = db.query(User).filter(User.username == req.username).first()
    if existing is not None:
        raise HTTPException(status_code=409, detail="Username already taken")
    user = User(username=req.username, password_hash=hash_password(req.password), role=req.role or "manager")
    db.add(user)
    db.commit()
    db.refresh(user)
    token = create_access_token(user.username, user.role)
    return {"access_token": token, "token_type": "bearer", "username": user.username, "role": user.role}


@app.get("/auth/me")
def me(user: dict = Depends(get_current_user)):
    return user


@app.exception_handler(ValueError)
async def value_error_handler(request, exc: ValueError):
    """
    Section 34: handle errors properly, never expose a stack trace to a
    normal user. Our internal modules raise ValueError for "no data for
    this date/item" — translated here into a clean 404, not a 500.
    """
    return JSONResponse(status_code=404, content={"detail": str(exc)})


@app.get("/")
def root():
    return {"product": "WasteWise AI", "tagline": "Predict demand. Reduce waste. Optimize every meal.", "status": "ok"}


# --------------------------------------------------------------------- #
# A. Executive Dashboard
# --------------------------------------------------------------------- #
@app.get("/dashboard")
def dashboard(date: str = Query(..., description="YYYY-MM-DD")):
    recs = generate_recommendations_for_date(date)
    champion = get_champion()
    llm_metrics = get_llm_metrics(since_hours=24)

    total_forecast_sales_inr = float((recs["forecast_demand"] * recs.get("selling_price", 0)).sum()) if "selling_price" in recs.columns else None

    return {
        "date": date,
        "forecasted_units_tomorrow": round(float(recs["forecast_demand"].sum()), 1),
        "items_scored": len(recs),
        "items_high_shortage_risk": int((recs.shortage_risk == "HIGH").sum()),
        "items_high_waste_risk": int((recs.waste_risk == "HIGH").sum()),
        "total_recommended_preparation_units": int(recs["recommended_preparation"].sum()),
        "estimated_revenue_at_risk_inr": round(float(recs["estimated_revenue_at_risk_inr"].sum()), 2),
        "estimated_waste_cost_inr": round(float(recs["estimated_waste_cost_inr"].sum()), 2),
        "model_version": champion["version_id"] if champion else None,
        "model_wape_pct": champion["metrics"]["wape_pct"] if champion else None,
        "ai_requests_today": llm_metrics.get("total_requests", 0),
        "avg_ai_latency_ms": llm_metrics.get("avg_latency_ms", 0),
        "top_risky_items": recs.sort_values("expected_shortfall_units", ascending=False)
            .head(5)[["item_name", "restaurant_id", "risk_category", "forecast_demand"]].to_dict(orient="records"),
    }


# --------------------------------------------------------------------- #
# B. Demand Forecast
# --------------------------------------------------------------------- #
@app.get("/forecast")
def forecast(date: str = Query(...), restaurant_id: str = Query(None)):
    recs = generate_recommendations_for_date(date)
    if restaurant_id:
        recs = recs[recs.restaurant_id == restaurant_id]
    return recs[["restaurant_id", "item_id", "item_name", "forecast_demand"]].to_dict(orient="records")


@app.get("/forecast/{item_id}")
def forecast_detail(item_id: str, date: str = Query(...), restaurant_id: str = Query(...)):
    feat_row, model, features = get_feature_row(date, restaurant_id, item_id)
    if len(feat_row) == 0:
        raise ValueError(f"No forecast available for {item_id} at {restaurant_id} on {date}")
    forecast_value = float(np.clip(model.predict(feat_row[features]), 0, None)[0])
    explanation = explain_prediction(model, feat_row, features)
    return {
        "item_id": item_id, "restaurant_id": restaurant_id, "date": date,
        "forecast_demand": round(forecast_value, 1), "explanation": explanation,
    }


# --------------------------------------------------------------------- #
# C. Waste & Shortage Risk / D. Preparation Recommendations
# --------------------------------------------------------------------- #
@app.get("/risk")
def risk(date: str = Query(...), restaurant_id: str = Query(None), tier: str = Query(None)):
    recs = generate_recommendations_for_date(date)
    if restaurant_id:
        recs = recs[recs.restaurant_id == restaurant_id]
    if tier:
        recs = recs[(recs.shortage_risk == tier) | (recs.waste_risk == tier)]
    return recs[["restaurant_id", "item_id", "item_name", "risk_category", "shortage_risk", "waste_risk",
                  "shortage_reason", "waste_reason"]].to_dict(orient="records")


@app.get("/recommendations")
def recommendations(date: str = Query(...), restaurant_id: str = Query(None)):
    recs = generate_recommendations_for_date(date)
    if restaurant_id:
        recs = recs[recs.restaurant_id == restaurant_id]
    return recs.sort_values("expected_shortfall_units", ascending=False).to_dict(orient="records")


# --------------------------------------------------------------------- #
# G. What-If Simulator
# --------------------------------------------------------------------- #
@app.post("/what-if")
def what_if(req: WhatIfRequest):
    overrides = {}
    if req.demand_pct_change:
        overrides["demand_pct_change"] = req.demand_pct_change
    if req.promotion:
        overrides["promotion"] = True
        overrides["discount_pct"] = req.discount_pct
    if req.weather_scenario:
        overrides["weather_scenario"] = req.weather_scenario
    if req.price_change_pct is not None:
        overrides["price_change_pct"] = req.price_change_pct
    if req.safety_buffer_pct is not None:
        overrides["safety_buffer_pct"] = req.safety_buffer_pct
    if req.current_inventory_override is not None:
        overrides["current_inventory_override"] = req.current_inventory_override

    return _run_what_if(req.date, req.restaurant_id, req.item_id, overrides)


# --------------------------------------------------------------------- #
# N. Bring Your Own Data — upload a real sales CSV, get a real forecast
# trained on it, in the same request. See ml/inference/custom_upload_pipeline.py
# for the honest constraints (min history for ML vs moving-average fallback).
# --------------------------------------------------------------------- #
UPLOAD_MAX_BYTES = 10 * 1024 * 1024  # 10 MB — generous for a CSV, protects the demo session


@app.post("/data/upload")
async def data_upload(file: UploadFile = File(...)):
    contents = await file.read()
    if len(contents) > UPLOAD_MAX_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large — limit is {UPLOAD_MAX_BYTES // (1024*1024)} MB.")
    try:
        result = run_upload_pipeline(contents, file.filename or "upload.csv")
    except UploadValidationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result


@app.get("/data/upload/sample-template")
def data_upload_sample_template():
    """Returns a small, valid sample CSV (as text) so a restaurant/demo
    viewer knows exactly what format to prepare — never leaves them
    guessing at column names."""
    from fastapi.responses import PlainTextResponse
    sample = (
        "date,item_name,units_sold,restaurant_location\n"
        "2026-01-01,Butter Chicken,42,Downtown\n"
        "2026-01-02,Butter Chicken,38,Downtown\n"
        "2026-01-03,Butter Chicken,51,Downtown\n"
        "2026-01-01,Veg Biryani,30,Downtown\n"
        "2026-01-02,Veg Biryani,27,Downtown\n"
        "2026-01-03,Veg Biryani,33,Downtown\n"
    )
    return PlainTextResponse(content=sample, media_type="text/csv")


# --------------------------------------------------------------------- #
# F. AI Restaurant Copilot
# --------------------------------------------------------------------- #
@app.post("/copilot/chat")
def copilot_chat(req: CopilotChatRequest):
    return _answer_question(req.question, req.date, prompt_version=req.prompt_version)


# --------------------------------------------------------------------- #
# H. MLOps Monitoring
# --------------------------------------------------------------------- #
@app.get("/mlops/models")
def mlops_models():
    """
    Candidate comparison comes from Phase 4's model_comparison.json (the
    champion/challenger training run). The ACTUAL serving champion comes
    from Phase 5's formal registry (ml/monitoring/registry.py), which —
    per its own documented promotion policy — selected a different model
    (XGBoost) than Phase 4's hand-rolled weighted score (Gradient
    Boosting). See docs/ML_PIPELINE.md's reconciliation note. Both are
    surfaced here rather than silently picking one.
    """
    comparison = json.load(open(MODELS_DIR / "model_comparison.json"))
    formal_champion = get_champion()
    return {
        "candidates": comparison,
        "phase4_naive_pick": json.load(open(MODELS_DIR / "champion_model.json"))["champion"],
        "formal_registry_champion": formal_champion["model_name"] if formal_champion else None,
        "formal_registry_version": formal_champion["version_id"] if formal_champion else None,
        "note": "The formal registry champion (Phase 5's threshold-gated promotion policy) is what's actually served — see docs/MLOPS.md.",
    }


@app.get("/models")
def models():
    champion = get_champion()
    return {"registered_versions": [v["version_id"] for v in list_versions()],
            "champion": champion["version_id"] if champion else None}


@app.get("/experiments")
def experiments():
    comparison = json.load(open(MODELS_DIR / "model_comparison.json"))
    formal_champion = get_champion()
    return {"champion_challenger_run": comparison,
            "champion_selected": formal_champion["model_name"] if formal_champion else None}


@app.get("/mlops/drift")
def mlops_drift(simulated: bool = Query(False, description="Set true to see a labeled synthetic drift scenario")):
    train = pd.read_parquet(PROCESSED / "train.parquet")
    test = pd.read_parquet(PROCESSED / "test.parquet")
    features = json.load(open(PROCESSED / "feature_columns.json"))
    numeric_features = [f for f in features if f in train.columns and np.issubdtype(train[f].dtype, np.number)]

    if simulated:
        report = simulate_drift_scenario(train, numeric_features, shift_feature="temperature", shift_amount=12.0)
    else:
        test_months = set(pd.to_datetime(test["date"]).dt.month.unique())
        season_matched_train = train[pd.to_datetime(train["date"]).dt.month.isin(test_months)]
        report = check_feature_drift(season_matched_train, test, numeric_features)
        report["comparison_method"] = "season-matched (same calendar months, prior period) — see docs/MLOPS.md"

    champion = get_champion()
    training_wape = champion["metrics"]["wape_pct"] if champion else None
    decision = evaluate_retrain_triggers(report, {"psi": 0.0, "status": "normal", "reference_mean": 0,
                                                   "current_mean": 0, "simulated": simulated},
                                          training_wape, training_wape)
    return {"drift_report": report, "retrain_decision": decision.to_dict()}


# --------------------------------------------------------------------- #
# I. LLMOps Monitoring
# --------------------------------------------------------------------- #
@app.get("/llmops/metrics")
def llmops_metrics(since_hours: int = Query(24)):
    return get_llm_metrics(since_hours=since_hours)


@app.get("/prompts")
def prompts(date: str = Query("2026-05-08")):
    return compare_prompt_versions(date)


# --------------------------------------------------------------------- #
# J. AI Cost / K. Latency
# --------------------------------------------------------------------- #
@app.get("/cost")
def cost():
    return get_ai_cost_dashboard()


@app.get("/cost/tiers")
def cost_tiers():
    return compare_llm_tiers()


@app.get("/latency")
def latency(n_requests: int = Query(15, le=50), include_llm: bool = Query(False)):
    return run_latency_benchmark(n_requests=n_requests, include_llm=include_llm)


# --------------------------------------------------------------------- #
# L. Data Quality
# --------------------------------------------------------------------- #
@app.get("/data-quality")
def data_quality():
    raw = pd.read_csv(ROOT / "data" / "raw" / "sales_data.csv")
    report = validate_data_quality(raw, as_of=pd.Timestamp(raw["date"].max()).to_pydatetime())
    return report.to_dict()


# --------------------------------------------------------------------- #
# Human-in-the-loop feedback (Section 20)
# --------------------------------------------------------------------- #
@app.get("/dashboard/trend")
def dashboard_trend(days: int = Query(14, le=30), end_date: str = Query("2026-05-08")):
    """
    Real, computed trend — not simulated. Runs the actual recommendation
    pipeline across `days` consecutive dates ending at `end_date` and
    aggregates each day's totals. Slower than a cached read (genuine
    per-day computation), kept to a modest default window for response time.
    """
    end = pd.Timestamp(end_date)
    dates = [(end - pd.Timedelta(days=i)).strftime("%Y-%m-%d") for i in range(days - 1, -1, -1)]
    points = []
    for d in dates:
        try:
            recs = generate_recommendations_for_date(d)
        except ValueError:
            continue
        points.append({
            "date": d,
            "total_forecast_units": round(float(recs["forecast_demand"].sum()), 1),
            "high_shortage_risk_items": int((recs.shortage_risk == "HIGH").sum()),
            "high_waste_risk_items": int((recs.waste_risk == "HIGH").sum()),
            "estimated_waste_cost_inr": round(float(recs["estimated_waste_cost_inr"].sum()), 2),
            "estimated_revenue_at_risk_inr": round(float(recs["estimated_revenue_at_risk_inr"].sum()), 2),
        })
    return {"points": points, "n_days": len(points)}


# --------------------------------------------------------------------- #
# M. Business Impact / ROI
# --------------------------------------------------------------------- #
@app.get("/business-impact")
def business_impact(
    monthly_food_waste_inr: float = Query(200000),
    waste_reduction_pct: float = Query(10.0),
    monthly_stockout_loss_inr: float = Query(80000),
    stockout_reduction_pct: float = Query(15.0),
    ai_monthly_cost_inr: float = Query(4500),
    implementation_cost_inr: float = Query(150000),
):
    """
    Section 21/22 — illustrative ROI calculator. Every input is a
    configurable assumption (query params), never a hardcoded "achieved"
    result. Formula: ROI = (Net Annual Benefit / Total Annual Investment) x 100.
    """
    monthly_waste_savings = monthly_food_waste_inr * (waste_reduction_pct / 100)
    monthly_stockout_savings = monthly_stockout_loss_inr * (stockout_reduction_pct / 100)
    monthly_savings = monthly_waste_savings + monthly_stockout_savings
    annual_savings = monthly_savings * 12
    annual_ai_cost = ai_monthly_cost_inr * 12
    total_annual_investment = annual_ai_cost + implementation_cost_inr  # implementation amortized in year 1
    net_annual_benefit = annual_savings - total_annual_investment
    roi_pct = (net_annual_benefit / total_annual_investment * 100) if total_annual_investment else 0
    payback_months = (implementation_cost_inr / monthly_savings) if monthly_savings > 0 else None

    return {
        "inputs": {
            "monthly_food_waste_inr": monthly_food_waste_inr, "waste_reduction_pct": waste_reduction_pct,
            "monthly_stockout_loss_inr": monthly_stockout_loss_inr, "stockout_reduction_pct": stockout_reduction_pct,
            "ai_monthly_cost_inr": ai_monthly_cost_inr, "implementation_cost_inr": implementation_cost_inr,
        },
        "estimated_monthly_savings_inr": round(monthly_savings, 2),
        "estimated_annual_savings_inr": round(annual_savings, 2),
        "estimated_annual_ai_cost_inr": round(annual_ai_cost, 2),
        "net_annual_benefit_inr": round(net_annual_benefit, 2),
        "roi_pct": round(roi_pct, 1),
        "payback_period_months": round(payback_months, 1) if payback_months else None,
        "label": "Illustrative PoC estimate — all inputs are adjustable assumptions, not measured results",
    }


@app.post("/feedback")
def feedback(req: FeedbackRequest, db: Session = Depends(get_session)):
    entry = FeedbackLog(
        date=req.date, restaurant_id=req.restaurant_id, item_id=req.item_id,
        recommendation=req.recommendation, human_decision=req.human_decision,
        final_action=req.final_action, notes=req.notes,
    )
    db.add(entry)
    db.commit()
    return {"status": "recorded", "id": entry.id}


@app.get("/feedback")
def list_feedback(db: Session = Depends(get_session), limit: int = Query(50, le=200)):
    rows = db.query(FeedbackLog).order_by(FeedbackLog.created_at.desc()).limit(limit).all()
    return [{"id": r.id, "date": r.date, "item_id": r.item_id, "restaurant_id": r.restaurant_id,
             "recommendation": r.recommendation, "human_decision": r.human_decision,
             "final_action": r.final_action, "outcome": r.outcome,
             "created_at": r.created_at.isoformat() if r.created_at else None} for r in rows]
