"""
Admin Security Operations API

Provides admin-only endpoints for security visibility, audit investigation,
failed-login tracking, admin actions, document access changes, and RAG safety.

No secrets, tokens, API keys, or environment values are exposed.
Every subsection is isolated: a failed query is rolled back and returns safe
degraded defaults instead of 500ing the whole endpoint.
"""

import csv
import io
import json as json_mod
from datetime import datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response
from sqlalchemy import desc, func
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.security.auth import AuthContext
from app.security.dependencies import require_admin
from app.security.models import AuditAction, AuditLog

router = APIRouter(prefix="/api/admin/security", tags=["Admin Security Operations"])

# ---------------------------------------------------------------------------
# Safe SQL helpers — each subsection is fully isolated.
# ---------------------------------------------------------------------------

def _now() -> datetime:
    """Return current UTC datetime."""
    return datetime.utcnow()


def _cutoff(hours: int = 24) -> datetime:
    """Return the cutoff datetime for the given hour window."""
    return _now() - timedelta(hours=hours)


def _parse_since(hours: int = 24) -> datetime:
    """Parse hours back into a datetime cutoff."""
    return _cutoff(hours)


_ACTION_FAILED_LOGINS = (AuditAction.LOGIN_FAILURE.name,)
_ACTION_SUCCESS_LOGINS = (AuditAction.LOGIN_SUCCESS.name,)

_ACTION_ADMIN_ACTIONS = (
    AuditAction.USER_CREATED.name,
    AuditAction.USER_UPDATED.name,
    AuditAction.USER_DEACTIVATED.name,
    AuditAction.PASSWORD_RESET.name,
    AuditAction.ROLE_CHANGED.name,
)

_ACTION_DOCUMENT_SEC = (
    AuditAction.DOCUMENT_UPLOAD.name,
    AuditAction.DOCUMENT_DELETE.name,
    AuditAction.DOCUMENT_ACCESS_CHANGE.name,
)

_ACTION_RAG_SAFETY = (
    AuditAction.RAG_ACCESS_DENIED.name,
    AuditAction.CHAT_RAG.name,
)

_ACTION_PERMISSION_DENIED = (
    AuditAction.RAG_ACCESS_DENIED.name,
    AuditAction.ADMIN_ACCESS_DENIED.name,
    AuditAction.PERMISSION_REVOKE.name,
)


# ---------------------------------------------------------------------------
# Summary helpers — each isolated.
# ---------------------------------------------------------------------------

def _count_by_action_status(
    db: Session,
    actions: tuple[str, ...],
    since: datetime,
    status: str | None = None,
) -> int:
    """Count audit log entries matching actions and optional status."""
    try:
        query = db.query(AuditLog).filter(
            AuditLog.action.in_(actions),
            AuditLog.created_at >= since,
        )
        if status is not None:
            query = query.filter(AuditLog.status == status)
        return query.count()
    except Exception:
        db.rollback()
        return 0


def _get_failed_logins(db: Session, since: datetime, limit: int = 20) -> list[dict[str, Any]]:
    """Recent failed login attempts with IP and user-agent."""
    try:
        rows = (
            db.query(AuditLog)
            .filter(
                AuditLog.action == AuditAction.LOGIN_FAILURE.name,
                AuditLog.created_at >= since,
            )
            .order_by(desc(AuditLog.created_at))
            .limit(limit)
            .all()
        )
        result = []
        for row in rows:
            user_agent = row.request_user_agent or ""
            # Truncate user agent for display safety (max 80 chars)
            if len(user_agent) > 80:
                user_agent = user_agent[:77] + "..."
            result.append({
                "id": row.id,
                "timestamp": row.created_at.isoformat() if row.created_at else None,
                "username": row.username,
                "request_ip": row.request_ip,
                "user_agent": user_agent,
                "status": row.status,
                "error_message": (row.error_message or ""),
            })
        return result
    except Exception:
        db.rollback()
        return []


def _get_login_counts_by_ip(db: Session, since: datetime) -> list[dict[str, Any]]:
    """Count failed login attempts grouped by IP address."""
    try:
        rows = (
            db.query(
                AuditLog.request_ip,
                func.count(AuditLog.id).label("count"),
            )
            .filter(
                AuditLog.action == AuditAction.LOGIN_FAILURE.name,
                AuditLog.created_at >= since,
            )
            .group_by(AuditLog.request_ip)
            .order_by(desc("count"))
            .limit(10)
            .all()
        )
        return [
            {"request_ip": ip or "unknown", "count": count}
            for ip, count in rows
        ]
    except Exception:
        db.rollback()
        return []


def _get_admin_actions(db: Session, since: datetime, limit: int = 20) -> list[dict[str, Any]]:
    """Recent admin actions (user management, role changes, password resets)."""
    try:
        rows = (
            db.query(AuditLog)
            .filter(
                AuditLog.action.in_(_ACTION_ADMIN_ACTIONS),
                AuditLog.created_at >= since,
            )
            .order_by(desc(AuditLog.created_at))
            .limit(limit)
            .all()
        )
        result = []
        for row in rows:
            # Safely parse details JSON without exposing secrets
            target = ""
            details_str = row.details
            if details_str:
                try:
                    parsed = json_mod.loads(details_str)
                    if isinstance(parsed, dict):
                        # Look for safe descriptive fields
                        target = parsed.get("target_username") or parsed.get("target") or ""
                except (json_mod.JSONDecodeError, TypeError):
                    pass
            result.append({
                "id": row.id,
                "timestamp": row.created_at.isoformat() if row.created_at else None,
                "admin_username": row.username,
                "action": row.action if isinstance(row.action, str) else row.action.name,
                "target": target,
                "status": row.status,
            })
        return result
    except Exception:
        db.rollback()
        return []


def _get_document_security(db: Session, since: datetime, limit: int = 20) -> list[dict[str, Any]]:
    """Recent document-related security events (upload, delete, access change)."""
    try:
        rows = (
            db.query(AuditLog)
            .filter(
                AuditLog.action.in_(_ACTION_DOCUMENT_SEC),
                AuditLog.created_at >= since,
            )
            .order_by(desc(AuditLog.created_at))
            .limit(limit)
            .all()
        )
        result = []
        for row in rows:
            doc_name = ""
            details_str = row.details
            if details_str:
                try:
                    parsed = json_mod.loads(details_str)
                    if isinstance(parsed, dict):
                        doc_name = parsed.get("filename") or parsed.get("document_name") or ""
                except (json_mod.JSONDecodeError, TypeError):
                    pass
            result.append({
                "id": row.id,
                "timestamp": row.created_at.isoformat() if row.created_at else None,
                "username": row.username,
                "action": row.action if isinstance(row.action, str) else row.action.name,
                "document_name": doc_name,
                "status": row.status,
            })
        return result
    except Exception:
        db.rollback()
        return []


def _get_rag_safety(db: Session, since: datetime, limit: int = 20) -> list[dict[str, Any]]:
    """RAG safety events: fallbacks, blocked questions, low-confidence retrieval."""

    def _rag_status_from_details(row: AuditLog) -> str:
        """Infer the RAG safety status from audit log details."""
        if row.status == "failure" and row.error_message:
            msg = row.error_message.lower()
            if "fallback" in msg:
                return "fallback"
            if "block" in msg or "unsupported" in msg:
                return "blocked"
            if "confiden" in msg:
                return "low_confidence"
            if "citation" in msg or "no_context" in msg:
                return "no_citation"
            if "legal" in msg or "medical" in msg or "financial" in msg:
                return "high_risk"
        return row.status

    try:
        # Also include failures in CHAT_RAG which may indicate fallback events
        rag_action_names = (*_ACTION_RAG_SAFETY, AuditAction.CHAT_RAG.name)
        rows = (
            db.query(AuditLog)
            .filter(
                AuditLog.action.in_(rag_action_names),
                AuditLog.created_at >= since,
                AuditLog.status.in_(("failure", "error")),
            )
            .order_by(desc(AuditLog.created_at))
            .limit(limit)
            .all()
        )
        result = []
        for row in rows:
            q = row.question or ""
            if len(q) > 120:
                q = q[:117] + "..."
            result.append({
                "id": row.id,
                "timestamp": row.created_at.isoformat() if row.created_at else None,
                "username": row.username,
                "action": row.action if isinstance(row.action, str) else row.action.name,
                "question_preview": q,
                "status": _rag_status_from_details(row),
                "error_message": (row.error_message or "")[:200],
                "request_ip": row.request_ip,
            })
        return result
    except Exception:
        db.rollback()
        return []


def _get_risky_activity(db: Session, since: datetime) -> dict[str, Any]:
    """Identify suspicious patterns: repeated IPs with failed logins/per-denied access."""
    repeated_ips = []
    repeated_users = []
    try:
        # Repeated failed-login IPs (>= 3 attempts)
        ip_rows = (
            db.query(
                AuditLog.request_ip,
                func.count(AuditLog.id).label("count"),
            )
            .filter(
                AuditLog.action == AuditAction.LOGIN_FAILURE.name,
                AuditLog.created_at >= since,
            )
            .group_by(AuditLog.request_ip)
            .having(func.count(AuditLog.id) >= 3)
            .order_by(desc("count"))
            .limit(10)
            .all()
        )
        repeated_ips = [
            {"ip": ip or "unknown", "failed_attempts": count}
            for ip, count in ip_rows
        ]
    except Exception:
        db.rollback()

    try:
        # Repeated permission-denied by username
        user_rows = (
            db.query(
                AuditLog.username,
                func.count(AuditLog.id).label("count"),
            )
            .filter(
                AuditLog.action.in_(_ACTION_PERMISSION_DENIED),
                AuditLog.created_at >= since,
            )
            .group_by(AuditLog.username)
            .having(func.count(AuditLog.id) >= 2)
            .order_by(desc("count"))
            .limit(10)
            .all()
        )
        repeated_users = [
            {"username": username, "denied_count": count}
            for username, count in user_rows
        ]
    except Exception:
        db.rollback()

    return {
        "repeated_failed_login_ips": repeated_ips,
        "repeated_permission_denied_users": repeated_users,
    }


def _build_summary_cards(db: Session, since: datetime) -> dict[str, int]:
    """Build count-only summary cards."""
    failed_logins = _count_by_action_status(
        db, _ACTION_FAILED_LOGINS, since, status="failure"
    )
    successful_logins = _count_by_action_status(
        db, _ACTION_SUCCESS_LOGINS, since, status="success"
    )
    admin_actions = _count_by_action_status(
        db, _ACTION_ADMIN_ACTIONS, since
    )
    permission_denied = _count_by_action_status(
        db, _ACTION_PERMISSION_DENIED, since
    )
    doc_access_changes = _count_by_action_status(
        db, (AuditAction.DOCUMENT_ACCESS_CHANGE.name,), since
    )
    rag_fallbacks = _count_by_action_status(
        db, _ACTION_RAG_SAFETY, since, status="failure"
    )

    return {
        "failed_logins_24h": failed_logins,
        "successful_logins_24h": successful_logins,
        "admin_actions_24h": admin_actions,
        "permission_denied_24h": permission_denied,
        "document_access_changes_24h": doc_access_changes,
        "rag_fallbacks_24h": rag_fallbacks,
    }


# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------

def _build_filtered_audit_query(
    db: Session,
    since: datetime | None = None,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    action: str | None = None,
    status: str | None = None,
    username: str | None = None,
    ip: str | None = None,
    event_type: str | None = None,
) -> Any:
    """Build a filtered audit log query for export."""
    query = db.query(AuditLog)

    if since is not None:
        query = query.filter(AuditLog.created_at >= since)

    if start_date is not None:
        query = query.filter(AuditLog.created_at >= start_date)

    if end_date is not None:
        query = query.filter(AuditLog.created_at <= end_date)

    if action is not None:
        query = query.filter(AuditLog.action == action)

    if status is not None:
        query = query.filter(AuditLog.status == status)

    if username is not None:
        query = query.filter(AuditLog.username.ilike(f"%{username}%"))

    if ip is not None:
        query = query.filter(AuditLog.request_ip == ip)

    if event_type is not None:
        # event_type maps to action family (e.g. "login", "admin", "document", "rag")
        mapping: dict[str, tuple[str, ...]] = {
            "login": _ACTION_FAILED_LOGINS + _ACTION_SUCCESS_LOGINS,
            "admin": _ACTION_ADMIN_ACTIONS,
            "document": _ACTION_DOCUMENT_SEC,
            "rag": _ACTION_RAG_SAFETY,
            "permission_denied": _ACTION_PERMISSION_DENIED,
        }
        if event_type in mapping:
            query = query.filter(AuditLog.action.in_(mapping[event_type]))

    return query.order_by(desc(AuditLog.created_at))


def _audit_rows_to_dicts(rows: list[AuditLog]) -> list[dict[str, Any]]:
    """Convert AuditLog ORM rows to safe dicts."""
    out = []
    for row in rows:
        details = None
        if row.details:
            try:
                details = json_mod.loads(row.details)
            except (json_mod.JSONDecodeError, TypeError):
                details = row.details
        out.append({
            "id": row.id,
            "timestamp": row.created_at.isoformat() if row.created_at else None,
            "username": row.username,
            "action": row.action if isinstance(row.action, str) else row.action.name,
            "status": row.status,
            "request_ip": row.request_ip,
            "user_agent": row.request_user_agent,
            "details": details,
            "error_message": row.error_message,
        })
    return out


def _generate_csv(rows: list[dict[str, Any]]) -> str:
    """Generate CSV string from audit log dicts."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "id", "timestamp", "username", "action", "status",
        "request_ip", "user_agent", "error_message",
    ])
    for row in rows:
        writer.writerow([
            row["id"],
            row["timestamp"],
            row["username"],
            row["action"],
            row["status"],
            row.get("request_ip") or "",
            row.get("user_agent") or "",
            (row.get("error_message") or "")[:500],
        ])
    buf.seek(0)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# HTTP Endpoints
# ---------------------------------------------------------------------------

@router.get("/overview")
def get_security_overview(
    request: Request,
    hours: int = Query(24, ge=1, le=720, description="Number of hours back to look for events"),
    db: Session = Depends(get_db),
    auth: AuthContext = Depends(require_admin),
):
    """
    Get summarized security operations data (admin only).

    Returns safe, non-sensitive data about:
    - Failed / successful logins
    - Admin actions
    - Permission denied events
    - Document access changes
    - RAG fallback / safety events
    - Risky patterns (repeated IPs, repeated denied access)

    No secrets, tokens, or stack traces are exposed.
    Every subsection is isolated: a failure returns safe defaults.
    """
    since = _parse_since(hours)

    # Isolated subsections — each failure is caught and rolled back
    summary_cards: dict[str, Any] = {
        "failed_logins_24h": 0,
        "successful_logins_24h": 0,
        "admin_actions_24h": 0,
        "permission_denied_24h": 0,
        "document_access_changes_24h": 0,
        "rag_fallbacks_24h": 0,
    }
    failed_logins_list: list[dict[str, Any]] = []
    login_counts_by_ip: list[dict[str, Any]] = []
    admin_actions_list: list[dict[str, Any]] = []
    document_security_list: list[dict[str, Any]] = []
    rag_safety_list: list[dict[str, Any]] = []
    risky: dict[str, Any] = {
        "repeated_failed_login_ips": [],
        "repeated_permission_denied_users": [],
    }

    try:
        summary_cards = _build_summary_cards(db, since)
    except Exception:
        db.rollback()

    try:
        failed_logins_list = _get_failed_logins(db, since)
    except Exception:
        db.rollback()

    try:
        login_counts_by_ip = _get_login_counts_by_ip(db, since)
    except Exception:
        db.rollback()

    try:
        admin_actions_list = _get_admin_actions(db, since)
    except Exception:
        db.rollback()

    try:
        document_security_list = _get_document_security(db, since)
    except Exception:
        db.rollback()

    try:
        rag_safety_list = _get_rag_safety(db, since)
    except Exception:
        db.rollback()

    try:
        risky = _get_risky_activity(db, since)
    except Exception:
        db.rollback()

    return {
        "summary_cards": summary_cards,
        "failed_login_activity": {
            "events": failed_logins_list,
            "login_counts_by_ip": login_counts_by_ip,
        },
        "admin_actions": admin_actions_list,
        "document_security": document_security_list,
        "rag_safety": rag_safety_list,
        "risky_activity": risky,
        "export_options": {
            "csv_supported": True,
            "json_supported": True,
            "query_params": [
                "start_date", "end_date", "action", "status",
                "username", "ip", "event_type",
            ],
        },
        "query_window_hours": hours,
        "queried_since": since.isoformat(),
    }


@router.get("/export.json")
def export_json(
    start_date: datetime | None = Query(None, description="Filter from date (ISO 8601)"),
    end_date: datetime | None = Query(None, description="Filter to date (ISO 8601)"),
    action: str | None = Query(None, description="Filter by action name"),
    status: str | None = Query(None, description="Filter by result status"),
    username: str | None = Query(None, description="Filter by username (partial match)"),
    ip: str | None = Query(None, description="Filter by request IP"),
    event_type: str | None = Query(None, description="Event family: login | admin | document | rag | permission_denied"),
    db: Session = Depends(get_db),
    auth: AuthContext = Depends(require_admin),
):
    """
    Export audit logs as JSON (admin only).

    Supports filtering by date range, action, status, username, IP, and event type.
    Returns up to 5000 records to keep response sizes safe.
    """
    try:
        query = _build_filtered_audit_query(
            db, start_date=start_date, end_date=end_date,
            action=action, status=status, username=username, ip=ip,
            event_type=event_type,
        )
        rows = query.limit(5000).all()
        data = _audit_rows_to_dicts(rows)
        return {"exported_at": _now().isoformat(), "count": len(data), "records": data}
    except Exception:
        db.rollback()
        raise HTTPException(status_code=500, detail="Export failed. Please try again.")


@router.get("/export.csv")
def export_csv(
    start_date: datetime | None = Query(None, description="Filter from date (ISO 8601)"),
    end_date: datetime | None = Query(None, description="Filter to date (ISO 8601)"),
    action: str | None = Query(None, description="Filter by action name"),
    status: str | None = Query(None, description="Filter by result status"),
    username: str | None = Query(None, description="Filter by username (partial match)"),
    ip: str | None = Query(None, description="Filter by request IP"),
    event_type: str | None = Query(None, description="Event family: login | admin | document | rag | permission_denied"),
    db: Session = Depends(get_db),
    auth: AuthContext = Depends(require_admin),
):
    """
    Export audit logs as CSV (admin only).

    Supports filtering by date range, action, status, username, IP, and event type.
    Returns up to 5000 records to keep response sizes safe.
    """
    try:
        query = _build_filtered_audit_query(
            db, start_date=start_date, end_date=end_date,
            action=action, status=status, username=username, ip=ip,
            event_type=event_type,
        )
        rows = query.limit(5000).all()
        data = _audit_rows_to_dicts(rows)
        csv_str = _generate_csv(data)
        return Response(
            content=csv_str,
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=audit-export.csv"},
        )
    except Exception:
        db.rollback()
        raise HTTPException(status_code=500, detail="Export failed. Please try again.")
