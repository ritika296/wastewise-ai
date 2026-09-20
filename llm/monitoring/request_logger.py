"""
LLM request logging (Phase 8 / Section 14 & 29). Every LLM call —
real or template-fallback — is persisted here. This is the `llm_requests`
table from Section 29's database design, implemented now (SQLite) so
Phase 10's FastAPI backend can read from it directly rather than
re-inventing logging at integration time.
"""
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = ROOT / "monitoring" / "llm_requests.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_requests (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    question TEXT,
    intent TEXT,
    provider TEXT,
    model TEXT,
    prompt_version TEXT,
    input_tokens INTEGER,
    output_tokens INTEGER,
    total_tokens INTEGER,
    latency_ms REAL,
    cost_inr REAL,
    fallback_used INTEGER,
    error INTEGER,
    error_detail TEXT
);
"""


def _connect():
    DB_PATH.parent.mkdir(exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute(SCHEMA)
    return conn


def log_request(question: str, intent: str, llm_meta: dict):
    conn = _connect()
    conn.execute(
        """INSERT INTO llm_requests
           (timestamp, question, intent, provider, model, prompt_version, input_tokens,
            output_tokens, total_tokens, latency_ms, cost_inr, fallback_used, error, error_detail)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            datetime.now(timezone.utc).isoformat(), question, intent,
            llm_meta.get("provider"), llm_meta.get("model"), llm_meta.get("prompt_version"),
            llm_meta.get("input_tokens"), llm_meta.get("output_tokens"), llm_meta.get("total_tokens"),
            llm_meta.get("latency_ms"), llm_meta.get("cost_inr"),
            int(bool(llm_meta.get("fallback_used"))), int(bool(llm_meta.get("error"))),
            llm_meta.get("error_detail"),
        ),
    )
    conn.commit()
    conn.close()


def get_metrics(since_hours: int = 24) -> dict:
    conn = _connect()
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM llm_requests WHERE timestamp >= datetime('now', ?)", (f"-{since_hours} hours",)
    ).fetchall()
    conn.close()

    if not rows:
        return {"total_requests": 0, "window_hours": since_hours}

    latencies = sorted(r["latency_ms"] for r in rows if r["latency_ms"] is not None)
    total_cost = sum(r["cost_inr"] or 0 for r in rows)
    n = len(rows)

    def pct(p):
        idx = min(int(len(latencies) * p), len(latencies) - 1)
        return latencies[idx] if latencies else 0

    by_version = {}
    for r in rows:
        v = r["prompt_version"] or "unknown"
        by_version.setdefault(v, []).append(r)

    return {
        "total_requests": n, "window_hours": since_hours,
        "total_input_tokens": sum(r["input_tokens"] or 0 for r in rows),
        "total_output_tokens": sum(r["output_tokens"] or 0 for r in rows),
        "total_tokens": sum(r["total_tokens"] or 0 for r in rows),
        "total_cost_inr": round(total_cost, 4),
        "cost_per_request_inr": round(total_cost / n, 5),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else 0,
        "p50_latency_ms": round(pct(0.5), 1), "p95_latency_ms": round(pct(0.95), 1),
        "p99_latency_ms": round(pct(0.99), 1),
        "error_rate": round(sum(r["error"] for r in rows) / n, 4),
        "fallback_rate": round(sum(r["fallback_used"] for r in rows) / n, 4),
        "requests_by_prompt_version": {v: len(rs) for v, rs in by_version.items()},
        "requests_by_intent": _count_by(rows, "intent"),
        "daily_cost_projection_inr": round(total_cost / max(since_hours, 1) * 24, 2),
        "monthly_cost_projection_inr": round(total_cost / max(since_hours, 1) * 24 * 30, 2),
    }


def _count_by(rows, field):
    counts = {}
    for r in rows:
        key = r[field] or "unknown"
        counts[key] = counts.get(key, 0) + 1
    return counts


def clear_log():
    """Testing/demo utility — not exposed via any API."""
    conn = _connect()
    conn.execute("DELETE FROM llm_requests")
    conn.commit()
    conn.close()
