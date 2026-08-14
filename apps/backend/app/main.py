import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.api import chat as chat_router
from app.api import chat_sessions as chat_sessions_router
from app.api import documents as documents_router
from app.api import evaluation as evaluation_router
from app.api import auth as auth_router
from app.api import admin as admin_router
from app.api import admin_documents as admin_documents_router
from app.api import admin_audit as admin_audit_router
from app.api import admin_security as admin_security_router
from app.api import admin_status as admin_status_router
from app.api import admin_rag_config as admin_rag_config_router
from app.db.base import Base, engine
from app.core.config import settings
from app.core.startup_validation import validate_startup
from app.core.security_headers import SecurityHeadersMiddleware

logger = logging.getLogger(__name__)

app = FastAPI(
    title="AI Chatbot API",
    description="Enterprise AI Chatbot Backend",
    version="0.1.0",
)

# Global exception handler — prevents stack traces from leaking to clients
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger = logging.getLogger(__name__)
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred. Please try again later."},
    )

# Security headers middleware — injects X-Content-Type-Options, X-Frame-Options,
# Referrer-Policy, Permissions-Policy, CSP, and Cache-Control for auth endpoints.
app.add_middleware(SecurityHeadersMiddleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_cors_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(chat_router.router)
app.include_router(chat_router._provider_router)
app.include_router(chat_sessions_router.router)
app.include_router(documents_router.router)
app.include_router(evaluation_router.router)
app.include_router(auth_router.router)
app.include_router(admin_router.router)
app.include_router(admin_documents_router.router)
app.include_router(admin_audit_router.router)
app.include_router(admin_status_router.router)
app.include_router(admin_rag_config_router.router)
app.include_router(admin_security_router.router)


def _should_autocreate_schema() -> bool:
    """
    Phase 34A.1 — Database Migration Hardening.

    Production schema MUST be owned by Alembic. Application startup
    must NOT silently create missing production tables.

    `Base.metadata.create_all(bind=engine)` is gated behind a
    combination of:

      1. Explicit opt-in via the DB_AUTO_CREATE_SCHEMA env var (or
         equivalent settings flag), AND/OR
      2. A test-shaped DATABASE_URL (sqlite:///./test.db,
         sqlite:///:memory:, or sqlite:///./test_*.db).

    For any other DATABASE_URL (e.g. a real PostgreSQL connection),
    `create_all` is suppressed so a missing Alembic migration can
    never be silently masked by a startup-side schema creation.
    Tests opt in via their own fixture-driven create_all (see
    `tests/conftest.py:db_session`) and do not depend on this hook.
    """
    try:
        if settings.DB_AUTO_CREATE_SCHEMA:
            return True
    except Exception:
        pass

    try:
        url = (settings.DATABASE_URL or "").lower()
    except Exception:
        url = ""
    if url.startswith("sqlite"):
        # SQLite is only used for the local pytest harness. Production
        # and staging always run against PostgreSQL.
        return True
    return False


@app.on_event("startup")
def startup():
    validate_startup()
    if _should_autocreate_schema():
        try:
            Base.metadata.create_all(bind=engine)
        except Exception:
            logger.exception("Base.metadata.create_all failed during startup.")
    else:
        logger.info(
            "DB_AUTO_CREATE_SCHEMA=false; relying on Alembic for schema management."
        )
    _bootstrap_admin_user()


def _bootstrap_admin_user():
    """
    Bootstrap an admin user from environment variables on startup.

    Only runs when AUTH_MODE=local, BOOTSTRAP_ADMIN_PASSWORD is provided,
    and no admin user exists in the database.
    """
    if settings.AUTH_MODE != "local":
        return

    if not settings.BOOTSTRAP_ADMIN_PASSWORD:
        logger.info("BOOTSTRAP_ADMIN_PASSWORD not set; skipping admin bootstrap.")
        return

    from sqlalchemy.orm import Session
    from app.db.session import SessionLocal
    from app.security.models import User, UserRole
    from app.security.password import hash_password

    db: Session = SessionLocal()
    try:
        existing_admin = db.query(User).filter(User.role == UserRole.admin).first()
        if existing_admin:
            logger.info("Admin user already exists; skipping bootstrap.")
            return

        admin_user = User(
            username=settings.BOOTSTRAP_ADMIN_USERNAME,
            email=settings.BOOTSTRAP_ADMIN_EMAIL,
            hashed_password=hash_password(settings.BOOTSTRAP_ADMIN_PASSWORD),
            role=UserRole.admin,
            is_active=True,
            is_external=False,
        )
        db.add(admin_user)
        db.commit()
        logger.info(
            "Bootstrap admin user created: username=%s, email=%s",
            settings.BOOTSTRAP_ADMIN_USERNAME,
            settings.BOOTSTRAP_ADMIN_EMAIL,
        )
    except Exception:
        logger.exception("Failed to bootstrap admin user.")
        db.rollback()
    finally:
        db.close()


@app.get("/", tags=["Root"])
def root():
    return {"message": "AI Chatbot API"}


@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "healthy", "service": "backend"}