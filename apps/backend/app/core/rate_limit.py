"""
Rate Limiting Utility

Provides rate limiting with Redis backend and in-memory fallback.
Used to protect sensitive endpoints against abuse and brute-force attacks.

Configuration:
- RATE_LIMIT_ENABLED=true/false (default: true)
- REDIS_HOST, REDIS_PORT for Redis connection (default: redis:6379)

Limitations of in-memory fallback:
- Not shared across multiple backend processes/containers
- Resets on application restart
- Sufficient for single-container deployments only
"""

import logging
import time
from typing import Optional
from collections import defaultdict
from dataclasses import dataclass, field

from fastapi import Request, HTTPException

from app.core.config import settings

logger = logging.getLogger(__name__)

# Default rate limit: 5 requests per 60 seconds
DEFAULT_REQUESTS = 5
DEFAULT_WINDOW = 60

_redis_client: Optional[object] = None


def _get_redis_client():
    """Lazy-load Redis client with connection caching."""
    global _redis_client
    if _redis_client is not None:
        return _redis_client

    try:
        import redis as redis_lib
        _redis_client = redis_lib.Redis(
            host=settings.REDIS_HOST,
            port=settings.REDIS_PORT,
            socket_connect_timeout=2,
            socket_timeout=2,
            decode_responses=True,
        )
        _redis_client.ping()
        logger.info("Rate limiting: using Redis backend (%s:%s)", settings.REDIS_HOST, settings.REDIS_PORT)
        return _redis_client
    except Exception as e:
        logger.warning("Rate limiting: Redis unavailable (%s). Using in-memory fallback.", e)
        _redis_client = None
        return None


@dataclass
class _InMemoryBucket:
    """Simple fixed-window in-memory bucket for rate limiting."""
    requests: list[float] = field(default_factory=list)


_in_memory_store: dict[str, _InMemoryBucket] = defaultdict(_InMemoryBucket)


def _is_rate_limited_memory(key: str, max_requests: int, window: int) -> bool:
    """Check if key is rate limited using in-memory store (fixed window)."""
    now = time.time()
    bucket = _in_memory_store[key]
    # Remove requests outside the current window
    bucket.requests = [t for t in bucket.requests if now - t < window]
    if len(bucket.requests) >= max_requests:
        return True
    bucket.requests.append(now)
    return False


def _is_rate_limited_redis(key: str, max_requests: int, window: int) -> bool:
    """Check if key is rate limited using Redis (sliding window via sorted set)."""
    redis = _get_redis_client()
    if redis is None:
        return _is_rate_limited_memory(key, max_requests, window)

    now = time.time()
    pipe = redis.pipeline()
    # Remove entries outside the window
    pipe.zremrangebyscore(key, 0, now - window)
    # Count entries in the window
    pipe.zcard(key)
    # Add current request timestamp
    pipe.zadd(key, {str(now): now})
    # Set expiry on the key (auto-cleanup)
    pipe.expire(key, window)
    results = pipe.execute()

    count = results[1]
    return count >= max_requests


def is_rate_limited(key: str, max_requests: int = DEFAULT_REQUESTS, window: int = DEFAULT_WINDOW) -> bool:
    """
    Check if a key has exceeded its rate limit.

    Args:
        key: Unique identifier (e.g. "login:ip:192.168.1.1").
        max_requests: Maximum allowed requests in the window.
        window: Time window in seconds.

    Returns:
        True if rate limit exceeded, False otherwise.
    """
    if not settings.RATE_LIMIT_ENABLED:
        return False

    try:
        return _is_rate_limited_redis(key, max_requests, window)
    except Exception as e:
        logger.warning("Rate limiting error for key %s: %s. Falling back to in-memory.", key, e)
        return _is_rate_limited_memory(key, max_requests, window)


def rate_limit(
    max_requests: int = DEFAULT_REQUESTS,
    window: int = DEFAULT_WINDOW,
    key_func=None,
):
    """
    FastAPI dependency for rate limiting.

    Args:
        max_requests: Maximum allowed requests in the window.
        window: Time window in seconds.
        key_func: Callable that takes a Request and returns a rate limit key.
                  Defaults to "{endpoint}:ip:{client_ip}".

    Returns:
        A FastAPI dependency that raises HTTPException(429) when rate limited.
    """
    def _default_key_func(request: Request) -> str:
        endpoint = request.url.path
        client_ip = request.client.host if request.client else "unknown"
        return f"{endpoint}:ip:{client_ip}"

    _key_func = key_func or _default_key_func

    def _check(request: Request):
        key = _key_func(request)
        if is_rate_limited(key, max_requests=max_requests, window=window):
            raise HTTPException(
                status_code=429,
                detail="Too Many Requests. Please try again later.",
            )
        return request

    return _check
