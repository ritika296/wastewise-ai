"""Pydantic schemas (Phase 10 / Section 25 — 'Use Pydantic schemas')."""
from typing import Optional
from pydantic import BaseModel, Field


class WhatIfRequest(BaseModel):
    date: str = Field(..., description="YYYY-MM-DD, must exist in the processed dataset")
    restaurant_id: str
    item_id: str
    demand_pct_change: float = 0.0
    promotion: bool = False
    discount_pct: float = 15.0
    weather_scenario: Optional[str] = None
    price_change_pct: Optional[float] = None
    safety_buffer_pct: Optional[float] = None
    current_inventory_override: Optional[float] = None


class CopilotChatRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=500)
    date: str = Field(..., description="YYYY-MM-DD")
    prompt_version: Optional[str] = None


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1)
    password: str = Field(..., min_length=1)


class RegisterRequest(BaseModel):
    # Password lives in the request BODY, never a query parameter — a
    # query-param password would leak into server access logs and browser
    # history. Minimums are PoC-level sanity checks, not a full password
    # policy.
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8, max_length=200)
    role: Optional[str] = Field("manager", pattern="^(manager|regional|finance|admin)$")


class FeedbackRequest(BaseModel):
    date: str
    restaurant_id: str
    item_id: str
    recommendation: str
    human_decision: str = Field(..., pattern="^(approve|reject|modify)$")
    final_action: str
    notes: Optional[str] = None
