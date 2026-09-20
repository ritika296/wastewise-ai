"""
Phase 16 tests — real authentication (bcrypt password hashing, JWT
sessions, global auth middleware, registration).

Run from wastewise-ai/ root: python -m pytest tests/test_auth.py -v
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
from app.auth import hash_password, verify_password, create_access_token, decode_token
from app.db import seed_demo_users, SessionLocal, User
from fastapi import HTTPException

client = TestClient(app)


@pytest.fixture(scope="module", autouse=True)
def ensure_demo_users():
    seed_demo_users()
    yield


# --- Password hashing --------------------------------------------------

def test_password_hash_is_not_plaintext():
    hashed = hash_password("mypassword123")
    assert hashed != "mypassword123"
    assert hashed.startswith("$2")  # bcrypt hash prefix


def test_password_hash_verifies_correct_password():
    hashed = hash_password("mypassword123")
    assert verify_password("mypassword123", hashed) is True


def test_password_hash_rejects_wrong_password():
    hashed = hash_password("mypassword123")
    assert verify_password("wrongpassword", hashed) is False


def test_same_password_hashes_differently_each_time():
    """bcrypt salts each hash — two hashes of the same password must differ,
    even though both verify correctly."""
    h1 = hash_password("samepassword")
    h2 = hash_password("samepassword")
    assert h1 != h2
    assert verify_password("samepassword", h1)
    assert verify_password("samepassword", h2)


def test_verify_password_handles_malformed_hash_gracefully():
    """A corrupted/garbage stored hash must return False, never raise."""
    assert verify_password("anything", "not-a-real-bcrypt-hash") is False


# --- JWT round-trip ------------------------------------------------------

def test_jwt_round_trip():
    token = create_access_token("testuser", "manager")
    payload = decode_token(token)
    assert payload["sub"] == "testuser"
    assert payload["role"] == "manager"


def test_jwt_rejects_tampered_token():
    token = create_access_token("testuser", "manager")
    tampered = token[:-4] + "abcd"
    with pytest.raises(HTTPException) as exc_info:
        decode_token(tampered)
    assert exc_info.value.status_code == 401


def test_jwt_rejects_garbage_token():
    with pytest.raises(HTTPException) as exc_info:
        decode_token("this-is-not-a-jwt-at-all")
    assert exc_info.value.status_code == 401


def test_jwt_error_message_never_leaks_internal_exception_text():
    """Real bug found and fixed: a malformed token's raw JWTError text
    could include fragments of the token/payload. The error message
    returned to the client must always be the same generic string."""
    with pytest.raises(HTTPException) as exc_info:
        decode_token("garbage.token.value")
    assert exc_info.value.detail == "Invalid or expired token"
    # Also true for a subtly tampered (but structurally valid-looking) token
    token = create_access_token("someone", "manager")
    with pytest.raises(HTTPException) as exc_info2:
        decode_token(token + "x")
    assert exc_info2.value.detail == "Invalid or expired token"


# --- Login flow ----------------------------------------------------------

def test_login_with_correct_demo_credentials_succeeds():
    r = client.post("/auth/login", json={"username": "manager", "password": "wastewise123"})
    assert r.status_code == 200
    body = r.json()
    assert "access_token" in body
    assert body["username"] == "manager"
    assert body["role"] == "manager"


def test_login_with_wrong_password_fails():
    r = client.post("/auth/login", json={"username": "manager", "password": "wrongpassword"})
    assert r.status_code == 401


def test_login_with_nonexistent_user_fails():
    r = client.post("/auth/login", json={"username": "does-not-exist-xyz", "password": "anything"})
    assert r.status_code == 401


def test_login_enumeration_protection_same_error_message():
    """A wrong password for a real user and a login attempt for a
    nonexistent user must return the SAME error message — otherwise an
    attacker could enumerate valid usernames from the error text alone."""
    r1 = client.post("/auth/login", json={"username": "manager", "password": "wrongpassword"})
    r2 = client.post("/auth/login", json={"username": "totally-nonexistent-user", "password": "wrongpassword"})
    assert r1.status_code == r2.status_code == 401
    assert r1.json()["detail"] == r2.json()["detail"]


# --- Protected endpoints ---------------------------------------------------

def test_protected_endpoint_blocks_missing_token():
    r = client.get("/dashboard?date=2026-05-08")
    assert r.status_code == 401


def test_protected_endpoint_blocks_invalid_token():
    r = client.get("/dashboard?date=2026-05-08", headers={"Authorization": "Bearer not-a-real-token"})
    assert r.status_code == 401


def test_protected_endpoint_allows_valid_token():
    login_r = client.post("/auth/login", json={"username": "manager", "password": "wastewise123"})
    token = login_r.json()["access_token"]
    r = client.get("/dashboard?date=2026-05-08", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200


def test_public_paths_do_not_require_auth():
    assert client.get("/").status_code == 200


# --- Registration ----------------------------------------------------------

def test_register_new_user_succeeds():
    r = client.post("/auth/register", json={"username": "newtestuser_phase16", "password": "securepass123"})
    assert r.status_code == 200
    assert "access_token" in r.json()
    # cleanup
    db = SessionLocal()
    db.query(User).filter(User.username == "newtestuser_phase16").delete()
    db.commit()
    db.close()


def test_register_duplicate_username_fails():
    client.post("/auth/register", json={"username": "duplicate_phase16", "password": "securepass123"})
    r = client.post("/auth/register", json={"username": "duplicate_phase16", "password": "anotherpass456"})
    assert r.status_code == 409
    db = SessionLocal()
    db.query(User).filter(User.username == "duplicate_phase16").delete()
    db.commit()
    db.close()


def test_register_rejects_short_password():
    r = client.post("/auth/register", json={"username": "shortpwuser", "password": "short"})
    assert r.status_code == 422  # pydantic validation error


def test_register_rejects_short_username():
    r = client.post("/auth/register", json={"username": "ab", "password": "validpassword123"})
    assert r.status_code == 422
