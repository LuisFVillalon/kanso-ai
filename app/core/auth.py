"""
Supabase JWT verification for the AI service.

The frontend calls this service directly with the same Supabase access token
it sends the main backend (Authorization: Bearer <jwt>). Every LLM call costs
money, so only signed-in users (including demo sandbox users) may use it.

Mirrors kanso-backend/app/core/auth.py: ES256/RS256 tokens are verified with
the project's public keys from the JWKS endpoint, and legacy HS256 tokens
with SUPABASE_JWT_SECRET when it is set.

Environment variables
─────────────────────
SUPABASE_URL         — project URL (e.g. https://xyz.supabase.co); required
SUPABASE_JWT_SECRET  — only if the project still signs tokens with HS256
"""

import logging
import os
from functools import lru_cache

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import PyJWKClient

log = logging.getLogger(__name__)

_bearer = HTTPBearer(auto_error=False)
_LEEWAY_SECONDS = 10  # clock skew between Supabase Auth and the Fly VM


def _auth_error(detail: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=detail,
        headers={"WWW-Authenticate": "Bearer"},
    )


@lru_cache(maxsize=1)
def _jwks_client() -> PyJWKClient:
    supabase_url = os.getenv("SUPABASE_URL", "").rstrip("/")
    if not supabase_url:
        raise RuntimeError("SUPABASE_URL is not set; can't verify Supabase JWTs.")
    return PyJWKClient(f"{supabase_url}/auth/v1/.well-known/jwks.json", cache_keys=True)


def _decode(token: str) -> dict:
    options = {"verify_aud": False}  # Supabase sets aud="authenticated"; not needed here
    if jwt.get_unverified_header(token).get("alg") == "HS256":
        secret = os.getenv("SUPABASE_JWT_SECRET", "")
        if not secret:
            raise jwt.InvalidTokenError("HS256 token but SUPABASE_JWT_SECRET is not configured.")
        return jwt.decode(token, secret, algorithms=["HS256"], options=options, leeway=_LEEWAY_SECONDS)

    try:
        key = _jwks_client().get_signing_key_from_jwt(token).key
    except jwt.PyJWKClientError as exc:
        raise jwt.InvalidTokenError(f"Could not fetch signing key: {exc}") from exc
    return jwt.decode(token, key, algorithms=["ES256", "RS256"], options=options, leeway=_LEEWAY_SECONDS)


def get_current_user_id(
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> str:
    """FastAPI dependency: verify the Bearer JWT and return the Supabase user id."""
    if credentials is None:
        raise _auth_error("Missing Authorization header")
    try:
        payload = _decode(credentials.credentials)
    except jwt.ExpiredSignatureError:
        raise _auth_error("Token has expired") from None
    except jwt.InvalidTokenError as exc:
        log.warning("AUTH | rejected — invalid token: %s", exc)
        raise _auth_error("Invalid token") from None

    user_id = payload.get("sub")
    if not user_id:
        raise _auth_error("Token payload missing 'sub' claim")
    return user_id
