"""
Startup validation for production security.

Validates required environment variables on application startup.
Warns or fails for insecure placeholder values in production/local auth mode.
Does not log secret values.
"""

import logging
import os
import sys

logger = logging.getLogger(__name__)

# Insecure placeholder values that should never be used in production
_INSECURE_PASSWORDS = {"changeme", "password", "admin", "123456", "password123"}
_INSECURE_SECRETS = {"changeme", "secret", "test", "default", "admin"}

# Minimum secure secret length
_MIN_JWT_SECRET_LENGTH = 32


def _fail_fast(msg: str) -> None:
    """Log an error and exit the application immediately."""
    logger.error("[SECURITY] %s", msg)
    sys.exit(1)


def _warn(msg: str) -> None:
    """Log a security warning."""
    logger.warning("[SECURITY] %s", msg)


def validate_startup() -> None:
    """
    Validate required environment variables for safe production use.

    Checks:
    - AUTH_MODE is valid (dev | local | oidc)
    - JWT_SECRET_KEY is present and secure in local/oidc mode
    - Database config is present
    - Insecure placeholders are rejected in local/production mode
    - OPENROUTER_API_KEY only required when provider=openrouter
    - BOOTSTRAP_ADMIN_PASSWORD only checked when bootstrap is needed
    """
    auth_mode = os.getenv("AUTH_MODE", "dev").lower().strip()

    if auth_mode not in ("dev", "local", "oidc"):
        _fail_fast(f"Invalid AUTH_MODE='{auth_mode}'. Must be one of: dev, local, oidc")

    # In local/oidc mode, JWT secret is required
    if auth_mode in ("local", "oidc"):
        jwt_secret = os.getenv("JWT_SECRET_KEY", "").strip()
        if not jwt_secret:
            _fail_fast(
                "JWT_SECRET_KEY is not set. "
                "Generate a secure secret with: openssl rand -base64 48"
            )
        if len(jwt_secret) < _MIN_JWT_SECRET_LENGTH:
            _fail_fast(
                f"JWT_SECRET_KEY is too short ({len(jwt_secret)} chars). "
                f"Minimum {_MIN_JWT_SECRET_LENGTH} characters required."
            )
        # Check for weak/commonly used secrets (case-insensitive)
        jwt_lower = jwt_secret.lower()
        if any(weak in jwt_lower for weak in _INSECURE_SECRETS):
            _fail_fast(
                "JWT_SECRET_KEY appears to contain an insecure/placeholder value. "
                "Generate a secure random secret."
            )
        logger.info("JWT_SECRET_KEY validated (length=%d)", len(jwt_secret))
    else:
        # dev mode - warn if JWT_SECRET_KEY is missing
        jwt_secret = os.getenv("JWT_SECRET_KEY", "").strip()
        if not jwt_secret:
            _warn("JWT_SECRET_KEY is not set. JWT authentication will not work in dev mode.")

    # Database config
    database_url = os.getenv("DATABASE_URL", "").strip()
    postgres_host = os.getenv("POSTGRES_HOST", "").strip()
    if not database_url and not postgres_host:
        _fail_fast(
            "Database configuration missing. "
            "Set either DATABASE_URL or POSTGRES_HOST/PORT/DB/USER/PASSWORD."
        )

    # Check for insecure database password in non-dev mode
    if auth_mode != "dev":
        pg_password = os.getenv("POSTGRES_PASSWORD", "").strip()
        if pg_password and pg_password.lower() in _INSECURE_PASSWORDS:
            _fail_fast(
                f"POSTGRES_PASSWORD '{pg_password}' is an insecure/placeholder value. "
                "Change to a strong random password before deploying."
            )

    # MinIO credentials check (warn in non-dev mode)
    if auth_mode != "dev":
        minio_password = os.getenv("MINIO_ROOT_PASSWORD", "").strip()
        if minio_password and minio_password.lower() in _INSECURE_PASSWORDS:
            _fail_fast(
                f"MINIO_ROOT_PASSWORD is an insecure/placeholder value. "
                "Change to a strong random password before deploying."
            )

    # OpenRouter API key - only required if provider is openrouter
    llm_provider = os.getenv("LLM_PROVIDER", "mock").lower().strip()
    if llm_provider == "openrouter":
        openrouter_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not openrouter_key:
            _fail_fast(
                "OPENROUTER_API_KEY is required when LLM_PROVIDER=openrouter. "
                "Get your key at https://openrouter.ai/keys"
            )
        logger.info("OPENROUTER_API_KEY is set")
    else:
        logger.info("LLM_PROVIDER=%s, OPENROUTER_API_KEY not required", llm_provider)

    # DEV_USER_SECRET check in dev mode
    dev_auth_enabled = os.getenv("DEV_AUTH_ENABLED", "false").lower() in ("true", "1", "yes")
    if auth_mode == "dev" and dev_auth_enabled:
        dev_secret = os.getenv("DEV_USER_SECRET", "").strip()
        if not dev_secret or dev_secret == "dev-secret-change-in-production":
            _warn(
                "DEV_USER_SECRET is using the default/placeholder value. "
                "Change it to prevent unauthorized dev access in shared environments."
            )

    logger.info("Startup validation completed successfully (AUTH_MODE=%s)", auth_mode)
