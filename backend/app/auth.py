"""
Phase 16 — real authentication (bcrypt password hashing + JWT sessions).

Replaces the PoC's "type any name to log in" decorative screen with an
actual login flow: passwords are hashed with bcrypt (never stored or
logged in plaintext), sessions are signed JWTs (HS256), and every
non-public endpoint requires a valid Bearer token (wired in main.py's
auth middleware).

Security notes (each one a real bug this phase found and fixed, not a
hypothetical):
  - decode_token() must NEVER leak the underlying exception's text back to
    a client — a malformed/tampered token's raw error can include partial
    payload bytes. Every failure path returns the same generic message.
  - Login and "user not found" must return the SAME error message and
    (as close as practical) the same latency profile, so a client cannot
    enumerate valid usernames by the error text alone.
  - Registration takes the password in the REQUEST BODY (see schemas.py's
    RegisterRequest), never as a query parameter — a query-param password
    ends up in server access logs and browser history.

Run from wastewise-ai/ root — this module is imported by backend/app/main.py.
"""
import os
from datetime import datetime, timedelta, timezone

import bcrypt
from jose import jwt, JWTError
from fastapi import Depends, HTTPException, Request, status

SECRET_KEY = os.getenv("WASTEWISE_JWT_SECRET", "wastewise-poc-dev-secret-do-not-use-in-real-production")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 8  # 8-hour session, reasonable for a work shift


def hash_password(plain_password: str) -> str:
    salt = bcrypt.gensalt()
    return bcrypt.hashpw(plain_password.encode("utf-8"), salt).decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    try:
        return bcrypt.checkpw(plain_password.encode("utf-8"), hashed_password.encode("utf-8"))
    except (ValueError, TypeError):
        # Malformed hash (shouldn't happen with data we wrote ourselves) — treat as no match,
        # never raise past this boundary so a bad stored hash can't 500 the login endpoint.
        return False


def create_access_token(username: str, role: str) -> str:
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {"sub": username, "role": role, "exp": expire}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    """
    Every failure path — expired, tampered signature, malformed structure,
    wrong algorithm — returns the SAME generic 401 with no internal detail.
    This was a real bug: the original version let the underlying JWTError's
    text (which can echo back fragments of a malformed token) reach the
    client directly.
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )


def get_current_user(request: Request) -> dict:
    """
    FastAPI dependency for endpoints that want the authenticated user
    directly (e.g. to check `role`). Most endpoints are protected globally
    by main.py's middleware instead; this is for the few that need the
    identity itself (e.g. /auth/me).
    """
    user = getattr(request.state, "user", None)
    if user is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not authenticated")
    return user


def require_role(*allowed_roles: str):
    """Dependency factory: require_role('admin', 'finance') restricts an
    endpoint to specific roles, on top of the base authentication check."""
    def _check(user: dict = Depends(get_current_user)) -> dict:
        if user.get("role") not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{user.get('role')}' is not permitted to access this resource",
            )
        return user
    return _check
