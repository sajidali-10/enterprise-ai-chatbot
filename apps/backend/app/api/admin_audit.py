"""
Admin Audit Log API

Provides admin-only access to audit logs with filtering and pagination.
"""

from typing import Optional
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func, desc

from app.db.session import get_db
from app.security.dependencies import require_admin
from app.security.auth import AuthContext
from app.security.models import AuditLog, AuditAction
from app.schemas.admin_audit import AuditLogListResponse, AuditLogEntry

router = APIRouter(prefix="/api/admin/audit-logs", tags=["Admin Audit Logs"])


@router.get("", response_model=AuditLogListResponse)
def list_audit_logs(
    action: Optional[str] = Query(None, description="Filter by action name"),
    username: Optional[str] = Query(None, description="Filter by actor username"),
    target_type: Optional[str] = Query(None, description="Filter by target type (stored in details JSON)"),
    result: Optional[str] = Query(None, description="Filter by result: success or failure"),
    date_from: Optional[datetime] = Query(None, description="Filter from date (ISO 8601)"),
    date_to: Optional[datetime] = Query(None, description="Filter to date (ISO 8601)"),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    db: Session = Depends(get_db),
    auth: AuthContext = Depends(require_admin),
):
    """
    List audit logs with filtering and pagination (admin only).

    Supports filtering by action, actor username, result, and date range.
    Returns paginated results with total count.
    """
    query = db.query(AuditLog)

    if action:
        query = query.filter(AuditLog.action == action)

    if username:
        query = query.filter(AuditLog.username.ilike(f"%{username}%"))

    if result:
        query = query.filter(AuditLog.status == result)

    if date_from:
        query = query.filter(AuditLog.created_at >= date_from)

    if date_to:
        query = query.filter(AuditLog.created_at <= date_to)

    # Get total count before pagination
    total = query.count()

    # Order by newest first and paginate
    query = query.order_by(desc(AuditLog.created_at))
    offset = (page - 1) * page_size
    logs = query.offset(offset).limit(page_size).all()

    entries = []
    for log in logs:
        import json
        details = None
        if log.details:
            try:
                details = json.loads(log.details)
            except json.JSONDecodeError:
                details = log.details

        retrieved_doc_ids = None
        if log.retrieved_document_ids:
            try:
                retrieved_doc_ids = json.loads(log.retrieved_document_ids)
            except json.JSONDecodeError:
                pass

        entries.append(AuditLogEntry(
            id=log.id,
            user_id=log.user_id,
            username=log.username,
            action=log.action if isinstance(log.action, str) else log.action.value,
            request_ip=log.request_ip,
            request_user_agent=log.request_user_agent,
            details=details,
            question=log.question,
            retrieved_document_ids=retrieved_doc_ids,
            status=log.status,
            error_message=log.error_message,
            model_provider=log.model_provider,
            model_name=log.model_name,
            duration_ms=log.duration_ms,
            created_at=log.created_at.isoformat() if log.created_at else None,
        ))

    return AuditLogListResponse(
        entries=entries,
        total=total,
        page=page,
        page_size=page_size,
        total_pages=(total + page_size - 1) // page_size,
    )
