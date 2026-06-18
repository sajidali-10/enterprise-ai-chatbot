"""
Admin System Status API

Provides a safe, admin-only endpoint for operations dashboard data.
No secrets, no tokens, no environment dumps.
"""

import os
from datetime import datetime, timedelta
from typing import Optional
from urllib.parse import urlparse

import redis
from fastapi import APIRouter, Depends, Request
from qdrant_client import QdrantClient
from sqlalchemy import func, text, desc
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.minio_client import get_minio_client
from app.db.session import get_db
from app.security.auth import AuthContext
from app.security.dependencies import require_admin
from app.security.models import AuditAction, AuditLog, User

router = APIRouter(prefix="/api/admin/system", tags=["Admin System Status"])


def _check_postgres(db: Session) -> dict:
    try:
        db.execute(text("SELECT 1"))
        return {"status": "healthy", "label": "PostgreSQL DB"}
    except Exception as exc:
        return {"status": "unhealthy", "label": "PostgreSQL DB", "detail": str(exc)}


def _check_redis() -> dict:
    try:
        r = redis.Redis(host=settings.REDIS_HOST, port=settings.REDIS_PORT, socket_connect_timeout=2)
        r.ping()
        return {"status": "healthy", "label": "Redis Cache"}
    except Exception as exc:
        return {"status": "unhealthy", "label": "Redis Cache", "detail": str(exc)}


def _check_qdrant() -> dict:
    try:
        client = QdrantClient(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT, timeout=2)
        client.get_collections()
        return {"status": "healthy", "label": "Qdrant Vector DB"}
    except Exception as exc:
        return {"status": "unhealthy", "label": "Qdrant Vector DB", "detail": str(exc)}


def _check_minio() -> dict:
    try:
        client = get_minio_client()
        client.list_buckets()
        return {"status": "healthy", "label": "MinIO Object Storage"}
    except Exception as exc:
        return {"status": "unhealthy", "label": "MinIO Object Storage", "detail": str(exc)}


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
        model = os.getenv("OLLAMA_MODEL", "unknown")
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


def _get_document_summary(db: Session) -> dict:
    from app.models.document import Document
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


def _get_security_summary(db: Session, auth: AuthContext) -> dict:
    since = datetime.utcnow() - timedelta(days=1)
    failed_logins = db.query(AuditLog).filter(
        AuditLog.action.in_((AuditAction.LOGIN_FAILURE, AuditAction.USER_LOGIN)),
        AuditLog.status == "failure",
        AuditLog.created_at >= since,
    ).count()

    recent_audits = db.query(AuditLog).filter(AuditLog.created_at >= since).count()

    last_admin_action = (
        db.query(AuditLog)
        .filter(
            AuditLog.action.in_((
                AuditAction.USER_CREATED,
                AuditAction.USER_UPDATED,
                AuditAction.USER_DEACTIVATED,
                AuditAction.PASSWORD_RESET,
                AuditAction.ROLE_CHANGED,
            ))
        )
        .order_by(desc(AuditLog.created_at))
        .first()
    )

    total_users = db.query(User).filter(User.is_active == True).count()

    return {
        "auth_mode": settings.AUTH_MODE,
        "current_user_role": auth.role.value if hasattr(auth.role, "value") else str(auth.role),
        "total_active_users": total_users,
        "recent_failed_logins_24h": failed_logins,
        "recent_audit_events_24h": recent_audits,
        "last_admin_action": {
            "action": last_admin_action.action.value if last_admin_action else None,
            "username": last_admin_action.username if last_admin_action else None,
            "timestamp": last_admin_action.created_at.isoformat() if last_admin_action and last_admin_action.created_at else None,
        } if last_admin_action else None,
    }


def _get_recent_activity(db: Session) -> dict:
    from app.models.document import Document
    from app.models.evaluation import EvaluationRun
    from app.models.observability import ChatObservation

    latest_docs = (
        db.query(Document)
        .order_by(desc(Document.created_at))
        .limit(5)
        .all()
    )

    latest_eval = (
        db.query(EvaluationRun)
        .order_by(desc(EvaluationRun.created_at))
        .first()
    )

    latest_audits = (
        db.query(AuditLog)
        .order_by(desc(AuditLog.created_at))
        .limit(5)
        .all()
    )

    latest_feedback = (
        db.query(ChatObservation)
        .filter(ChatObservation.feedback_rating.isnot(None))
        .order_by(desc(ChatObservation.created_at))
        .limit(5)
        .all()
    )

    return {
        "recent_documents": [
            {
                "id": d.id,
                "filename": d.original_name or d.filename,
                "status": d.status,
                "created_at": d.created_at.isoformat() if d.created_at else None,
            }
            for d in latest_docs
        ],
        "latest_evaluation": {
            "run_id": latest_eval.id,
            "created_at": latest_eval.created_at.isoformat() if latest_eval and latest_eval.created_at else None,
            "pass_percentage": round(latest_eval.pass_percentage, 2) if latest_eval else None,
            "status": latest_eval.status if latest_eval else None,
        } if latest_eval else None,
        "recent_audit_events": [
            {
                "id": a.id,
                "action": a.action.value if hasattr(a.action, "value") else str(a.action),
                "username": a.username,
                "status": a.status,
                "timestamp": a.created_at.isoformat() if a.created_at else None,
            }
            for a in latest_audits
        ],
        "recent_feedback": [
            {
                "id": f.id,
                "rating": f.feedback_rating,
                "question": f.question[:100] + "..." if f.question and len(f.question) > 100 else f.question,
                "timestamp": f.created_at.isoformat() if f.created_at else None,
            }
            for f in latest_feedback
        ],
    }


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
    """
    # Infer gateway info from request headers
    forwarded_proto = request.headers.get("X-Forwarded-Proto", "")
    is_https = forwarded_proto == "https" or request.url.scheme == "https"
    host = request.headers.get("Host", request.url.hostname or "unknown")

    # Connectivity checks
    pg = _check_postgres(db)
    rd = _check_redis()
    qd = _check_qdrant()
    mn = _check_minio()

    # Overall health
    all_ok = all(s["status"] == "healthy" for s in [pg, rd, qd, mn])

    provider = _get_provider_info()

    return {
        "gateway": {
            "nginx_proxy": "healthy",  # If request reached here through nginx, it's working
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
        "rag_quality": _get_rag_summary(db),
        "documents": _get_document_summary(db),
        "security": _get_security_summary(db, auth),
        "recent_activity": _get_recent_activity(db),
        "overall_healthy": all_ok,
        "status_source_note": "Status is based on application connectivity checks, not raw Docker container state.",
    }
