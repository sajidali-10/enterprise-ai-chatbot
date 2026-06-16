"""
JWT token utilities for authentication.

Provides create_access_token and decode_token using PyJWT.
"""

from datetime import datetime, timezone, timedelta
from typing import Optional, Any

import jwt
from jwt import PyJWTError

from app.core.config import settings


def create_access_token(
    subject: str,
    extra_claims: Optional[dict[str, Any]] = None,
    expires_delta: Optional[timedelta] = None,
    token_version: Optional[int] = None,
) -> str:
    """
    Create a JWT access token.

    Args:
        subject: The subject claim (typically user_id as string).
        extra_claims: Additional claims to include in the payload.
        expires_delta: Override default token expiry.
        token_version: User's current token_version for session invalidation.

    Returns:
        Encoded JWT string.
    """
    if expires_delta is None:
        expires_delta = timedelta(minutes=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)

    expire = datetime.now(timezone.utc) + expires_delta
    payload = {
        "sub": subject,
        "exp": expire,
        "iat": datetime.now(timezone.utc),
        "type": "access",
    }
    if token_version is not None:
        payload["tv"] = token_version
    if extra_claims:
        payload.update(extra_claims)

    encoded = jwt.encode(
        payload,
        settings.JWT_SECRET_KEY,
        algorithm=settings.JWT_ALGORITHM,
    )
    return encoded


def decode_token(token: str) -> dict[str, Any]:
    """
    Decode and verify a JWT access token.

    Args:
        token: The JWT string to decode.

    Returns:
        Decoded token payload dictionary.

    Raises:
        ValueError: If JWT_SECRET_KEY is not configured.
        jwt.ExpiredSignatureError: If token is expired.
        jwt.InvalidTokenError: If token is invalid.
    """
    if not settings.JWT_SECRET_KEY:
        raise ValueError("JWT_SECRET_KEY is not configured")

    payload = jwt.decode(
        token,
        settings.JWT_SECRET_KEY,
        algorithms=[settings.JWT_ALGORITHM],
        options={"require": ["exp", "sub"]},
    )
    return payload
