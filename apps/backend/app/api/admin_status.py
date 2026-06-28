"""
Admin System Status API

Provides a safe, admin-only endpoint for operations dashboard data.
No secrets, no tokens, no environment dumps.
"""

import os
from datetime import datetime, timedelta
from typing import Any

import redis
from fastapi import APIRouter, Depends, Request
from qdrant_client import QdrantClient
from sqlalchemy import desc, text
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.minio_client import get_minio_client
from app.db.session import get_db
from app.security.auth import AuthContext
from app.security.dependencies import require_admin
from app.security.models import AuditAction, AuditLog, User

router = APIRouter(prefix="/api/admin/system", tags=["Admin System Status"])

# ---------------------------------------------------------------------------
# Safe helpers — each subsection is fully isolated: on any DB error the
# transaction is rolled back and a safe degraded response is returned.
# ---------------------------------------------------------------------------

def _check_postgres(db: Session) -> dict:
    try:
        db.execute(text("SELECT 1"))
        return {"status": "healthy", "label": "PostgreSQL DB"}
    except Exception as exc:
        db.rollback()
        return {"status": "unhealthy", "label": "PostgreSQL DB", "detail": "Connection failed"}


def _check_redis() -> dict:
    try:
        r = redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, socket_connect_timeout=2)
        r.ping()
        return {"status": "healthy", "label": "Redis Cache"}
    except Exception:
        return {"status": "unhealthy", "label": "Redis Cache", "detail": "Connection failed"}


def _check_qdrant() -> dict:
    try:
        client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT, timeout=2)
        client.get_collections()
        return {"status": "healthy", "label": "Qdrant Vector DB"}
    except Exception:
        return {"status": "unhealthy", "label": "Qdrant Vector DB", "detail": "Connection failed"}


def _check_minio() -> dict:
    try:
        client = get_minio_client()
        client.list_buckets()
        return {"status": "healthy", "label": "MinIO Object Storage"}
    except Exception:
        return {"status": "unhealthy", "label": "MinIO Object Storage", "detail": "Connection failed"}


def _get_provider_info() -> dict:
    provider_name = os.getenv("LLM_PROVIDER", "mock").lower().strip()
    gateway_mode = provider_name == "litellm"
    litellm_enabled = os.getenv("LITELLM_ENABLED", "false").lower() in ("true", "1", "yes")

    if provider_name == "openrouter":
        model = os.getenv("OPENROUTER_MODEL", "unknown")
    elif provider_name == "litellm":
        model = os.getenv("LITELLM_MODEL", "unknown")
    elif provider_name == "openai":
        model = os.getenv("OPENAI_MODEL", "unknown")
    elif provider_name == "ollama":
        model = os.getenv("OLLAMAMODEL", "unknown")
    else:
        model = "unknown"

    return {
        "provider": provider_name,
        "model": model,
        "gateway_mode": gateway_mode,
        "litellm_enabled": litellm_enabled,
    }


def _get_rag_summary(db: Session) -> dict:
    from app.models.evaluation import EvaluationRun
    try:
        latest = db.query(EvaluationRun).order_by(desc(EvaluationRun.created_at)).first()
        if latest:
            return {
                "last_run": latest.created_at.isoformat() if latest.created_at else None,
                "total_tests": latest.total_tests,
                "passed_tests": latest.passed_tests,
                "failed_tests": latest.failed_tests,
                "pass_percentage": round(latest.pass_percentage, 2),
                "status": latest.status,
            }
        return {
            "last_run": None,
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "pass_percentage": 0.0,
            "status": "no_runs",
        }
    except Exception:
        db.rollback()
        return {
            "last_run": None,
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "pass_percentage": 0.0,
            "status": "unavailable",
        }


def _get_document_summary(db: Session) -> dict:
    from app.models.document import Document
    try:
        total = db.query(Document).count()
        indexed = db.query(Document).filter(Document.status == "indexed").count()
        failed = db.query(Document).filter(Document.status == "failed").count()
        pending = db.query(Document).filter(Document.status == "pending").count()
        return {
            "total_documents": total,
            "indexed": indexed,
            "failed": failed,
            "pending": pending,
            "collection_name": settings.QDRANT_COLLECTION,
            "embedding_provider": settings.EMBEDDING_PROVIDER,
            "embedding_dimension": settings.EMBEDDING_DIMENSION,
        }
    except Exception:
        db.rollback()
        return {
            "total_documents": 0,
            "indexed": 0,
            "failed": 0,
            "pending": 0,
            "collection_name": settings.QDRANT_COLLECTION or "unknown",
            "embedding_provider": settings.EMBEDDING_PROVIDER or "unknown",
            "embedding_dimension": settings.EMBEDDING_DIMENSION or 0,
        }


def _get_security_summary(db: Session, auth: AuthContext) -> dict:
    since = datetime.utcnow() - timedelta(days=1)

    try:
        # PostgreSQL stores enum values as UPPERCASE names (e.g. "LOGIN_FAILURE").
        # Use .name to get the uppercase string matching the DB representation.
        failed_logins = db.query(AuditLog).filter(
            AuditLog.action.in_((AuditAction.LOGIN_FAILURE.name, AuditAction.LOGIN_SUCCESS.name)),
            AuditLog.status == "failure",
            AuditLog.created_at >= since,
        ).count()
    except Exception:
        db.rollback()
        failed_logins = 0

    try:
        recent_audits = db.query(AuditLog).filter(AuditLog.created_at >= since).count()
    except Exception:
        db.rollback()
        recent_audits = 0

    admin_action_names = (
        AuditAction.USER_CREATED.name,
        AuditAction.USER_UPDATED.name,
        AuditAction.USER_DEACTIVATED.name,
        AuditAction.PASSWORD_RESET.name,
        AuditAction.ROLE_CHANGED.name,
    )
    last_admin_action = None
    try:
        last_admin_action = (
            db.query(AuditLog)
            .filter(AuditLog.action.in_(admin_action_names))
            .order_by(desc(AuditLog.created_at))
            .first()
        )
    except Exception:
        db.rollback()

    try:
        total_users = db.query(User).filter(User.is_active == True).count()
    except Exception:
        db.rollback()
        total_users = 0

    return {
        "auth_mode": settings.AUTH_MODE,
        "current_user_role": auth.role.value if hasattr(auth.role, "value") else str(auth.role),
        "total_active_users": total_users,
        "recent_failed_logins_24h": failed_logins,
        "recent_audit_events_24h": recent_audits,
        "last_admin_action": {
            "action": last_admin_action.action.name if last_admin_action else None,
            "username": last_admin_action.username if last_admin_action else None,
            "timestamp": last_admin_action.created_at.isoformat() if last_admin_action and last_admin_action.created_at else None,
        } if last_admin_action else None,
    }


def _get_recent_activity(db: Session) -> dict:
    from app.models.document import Document
    from app.models.evaluation import EvaluationRun
    from app.models.observability import ChatObservation

    recent_documents: list[dict[str, Any]] = []
    latest_evaluation: dict[str, Any] | None = None
    recent_audit_events: list[dict[str, Any]] = []
    recent_feedback: list[dict[str, Any]] = []

    try:
        docs = db.query(Document).order_by(desc(Document.created_at)).limit(5).all()
        recent_documents = [
            {
                "id": d.id,
                "filename": d.original_name or d.filename,
                "status": d.status,
                "created_at": d.created_at.isoformat() if d.created_at else None,
            }
            for d in docs
        ]
    except Exception:
        db.rollback()

    try:
        latest_eval = db.query(EvaluationRun).order_by(desc(EvaluationRun.created_at)).first()
        if latest_eval:
            latest_evaluation = {
                "run_id": latest_eval.id,
                "created_at": latest_eval.created_at.isoformat() if latest_eval.created_at else None,
                "pass_percentage": round(latest_eval.pass_percentage, 2),
                "status": latest_eval.status,
            }
    except Exception:
        db.rollback()

    try:
        audits = db.query(AuditLog).order_by(desc(AuditLog.created_at)).limit(5).all()
        recent_audit_events = [
            {
                "id": a.id,
                # Use .name to get the uppercase DB representation of the enum.
                "action": a.action.name if hasattr(a.action, "name") else str(a.action),
                "username": a.username,
                "status": a.status,
                "timestamp": a.created_at.isoformat() if a.created_at else None,
            }
            for a in audits
        ]
    except Exception:
        db.rollback()

    try:
        feedback = (
            db.query(ChatObservation)
            .filter(ChatObservation.feedback_rating.isnot(None))
            .order_by(desc(ChatObservation.created_at))
            .limit(5)
            .all()
        )
        recent_feedback = [
            {
                "id": f.id,
                "rating": f.feedback_rating,
                "question": (f.question[:100] + "...") if f.question and len(f.question) > 100 else f.question,
                "timestamp": f.created_at.isoformat() if f.created_at else None,
            }
            for f in feedback
        ]
    except Exception:
        db.rollback()

    return {
        "recent_documents": recent_documents,
        "latest_evaluation": latest_evaluation,
        "recent_audit_events": recent_audit_events,
        "recent_feedback": recent_feedback,
    }


# ---------------------------------------------------------------------------
# Endpoint — fully isolated subsections: one failing never 500s the whole
# ---------------------------------------------------------------------------

@router.get("/status")
def get_system_status(
    request: Request,
    db: Session = Depends(get_db),
    auth: AuthContext = Depends(require_admin),
):
    """
    Get comprehensive system status for the admin operations dashboard.

    Returns safe, non-sensitive data about:
    - Gateway / HTTPS status
    - Application service health
    - Data store connectivity
    - AI provider configuration
    - RAG evaluation summary
    - Document/indexing summary
    - Security / audit summary
    - Recent activity

    No secrets, API keys, tokens, or raw .env values are exposed.
    Every subsection is fully isolated — a failure in any one of them
    returns a safe degraded response and never 500s the whole endpoint.
    """
    # Infer gateway info from request headers
    forwarded_proto = request.headers.get("X-Forwarded-Proto", "")
    is_https = forwarded_proto == "https" or request.url.scheme == "https"
    host = request.headers.get("Host", request.url.hostname or "unknown")

    # Connectivity checks — each is already self-contained
    pg = _check_postgres(db)
    rd = _check_redis()
    qd = _check_qdrant()
    mn = _check_minio()

    all_ok = all(s["status"] == "healthy" for s in [pg, rd, qd, mn])
    provider = _get_provider_info()

    # Each subsection is fully isolated: try/except + rollback ensures a
    # failure in one query does NOT abort the transaction for the rest.
    rag_quality: dict[str, Any] = {
        "last_run": None,
        "total_tests": 0,
        "passed_tests": 0,
        "failed_tests": 0,
        "pass_percentage": 0.0,
        "status": "unavailable",
    }
    documents: dict[str, Any] = {
        "total_documents": 0,
        "indexed": 0,
        "failed": 0,
        "pending": 0,
        "collection_name": "unknown",
        "embedding_provider": "unknown",
        "embedding_dimension": 0,
    }
    security: dict[str, Any] = {
        "auth_mode": "unknown",
        "current_user_role": "unknown",
        "total_active_users": 0,
        "recent_failed_logins_24h": 0,
        "recent_audit_events_24h": 0,
        "last_admin_action": None,
    }
    recent_activity: dict[str, Any] = {
        "recent_documents": [],
        "latest_evaluation": None,
        "recent_audit_events": [],
        "recent_feedback": [],
    }

    try:
        rag_quality = _get_rag_summary(db)
    except Exception:
        db.rollback()

    try:
        documents = _get_document_summary(db)
    except Exception:
        db.rollback()

    try:
        security = _get_security_summary(db, auth)
    except Exception:
        db.rollback()

    try:
        recent_activity = _get_recent_activity(db)
    except Exception:
        db.rollback()

    # Phase 31A — LangSmith status. Each subsection is isolated: if the
    # tracing service raises (e.g. langsmith not installed) we fall back
    # to a safe degraded response that never includes API key material.
    langsmith: dict[str, Any] = {"enabled": False, "project": None, "endpoint_host": None, "sample_rate": None, "privacy_flags": {}, "last_trace": None, "warnings": []}
    try:
        from app.services.langsmith_tracing import get_langsmith_status
        status = get_langsmith_status()
        langsmith = {
            "enabled": bool(status.get("langsmith_tracing")),
            "available": bool(status.get("langsmith_available", True)),
            "project": status.get("project"),
            "endpoint_host": status.get("endpoint_host"),
            "sample_rate": status.get("sample_rate"),
            "api_key_configured": bool(status.get("api_key_configured")),
            "privacy_mode": status.get("privacy_mode"),
            "privacy_flags": {
                "log_full_prompt": bool(status.get("log_full_prompt")),
                "log_document_text": bool(status.get("log_document_text")),
                "log_user_input": bool(status.get("log_user_input")),
                "log_llm_output": bool(status.get("log_llm_output")),
                "log_retrieved_context": bool(status.get("log_retrieved_context")),
                "redact_metadata": bool(status.get("redact_metadata")),
                "max_context_chars": status.get("max_context_chars"),
            },
            "last_trace": status.get("last_trace") or {},
            "warning": status.get("warning"),
            "warnings": [
                w for w in [status.get("warning")] if w
            ],
        }
    except Exception:
        # Never break /status because tracing status failed
        langsmith = {"enabled": False, "available": False, "error": "status_unavailable"}

    return {
        "gateway": {
            "nginx_proxy": "healthy",
            "https_active": is_https,
            "domain": host,
        },
        "application": {
            "backend_api": {"status": "healthy", "label": "Backend API"},
            "frontend": {"status": "healthy", "label": "Frontend"},
        },
        "data": {
            "postgres": pg,
            "redis": rd,
            "qdrant": qd,
            "minio": mn,
        },
        "ai_provider": {
            "active_provider": provider["provider"],
            "model": provider["model"],
            "gateway_mode": provider["gateway_mode"],
            "litellm_enabled": provider["litellm_enabled"],
        },
        "rag_quality": rag_quality,
        "documents": documents,
        "security": security,
        "recent_activity": recent_activity,
        "langsmith": langsmith,
        "overall_healthy": all_ok,
        "status_source_note": "Status is based on application connectivity checks, not raw Docker container state.",
    }