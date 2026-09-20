"""
Database layer (Phase 10 / Section 25 & 29). SQLite by default (PoC —
zero external services); RETAINIQ_STYLE swap to Postgres via
WASTEWISE_DB_URL for Pilot/MVP, same pattern as the earlier RetainIQ
project. Holds the human-in-the-loop feedback table (Section 20) — every
other data source (forecasts, model registry, LLM requests) already has
its own storage from earlier phases and is read directly, not duplicated
here.
"""
import os
from datetime import datetime, timezone
from pathlib import Path
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text
from sqlalchemy.orm import declarative_base, sessionmaker

ROOT = Path(__file__).resolve().parent.parent.parent
DB_URL = os.getenv("WASTEWISE_DB_URL", f"sqlite:///{ROOT / 'backend' / 'wastewise.db'}")

engine = create_engine(DB_URL, connect_args={"check_same_thread": False} if "sqlite" in DB_URL else {})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base = declarative_base()


def now():
    return datetime.now(timezone.utc)


class User(Base):
    """Phase 16 — real user accounts, replacing the PoC's decorative
    'type any name' login. Passwords are stored ONLY as bcrypt hashes."""
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    role = Column(String, default="manager")  # manager | regional | finance | admin
    created_at = Column(DateTime, default=now)


class FeedbackLog(Base):
    """Human-in-the-loop: AI recommendation -> human decision -> outcome (Section 20)."""
    __tablename__ = "feedback_log"
    id = Column(Integer, primary_key=True)
    date = Column(String, index=True)
    restaurant_id = Column(String)
    item_id = Column(String, index=True)
    recommendation = Column(Text)
    human_decision = Column(String)  # approve | reject | modify
    final_action = Column(Text)
    notes = Column(Text)
    outcome = Column(String, default="pending")  # pending | as_expected | different — filled in later
    created_at = Column(DateTime, default=now)


def init_db():
    Base.metadata.create_all(engine)


def get_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


DEMO_USERS = [
    {"username": "manager", "password": "wastewise123", "role": "manager"},
    {"username": "regional", "password": "wastewise123", "role": "regional"},
    {"username": "finance", "password": "wastewise123", "role": "finance"},
    {"username": "admin", "password": "wastewise123", "role": "admin"},
]


def seed_demo_users():
    """
    Seeds 4 demo accounts (one per role, all password 'wastewise123') ONLY
    if the users table is currently empty — never overwrites real accounts
    a user might register on top of this PoC. Idempotent: safe to call on
    every startup.
    """
    from app.auth import hash_password  # local import avoids a circular import at module load time

    db = SessionLocal()
    try:
        if db.query(User).count() > 0:
            return
        for demo in DEMO_USERS:
            db.add(User(username=demo["username"], password_hash=hash_password(demo["password"]), role=demo["role"]))
        db.commit()
    finally:
        db.close()
