"""
Admin document access management endpoints (Phase 13).

Routes (all admin-only, mounted under /api/admin/documents):

  GET    /api/admin/documents                       — list all documents with owner/visibility/shares
  GET    /api/admin/documents/{document_id}/access  — fetch access summary for one document
  PATCH  /api/admin/documents/{document_id}/access  — update visibility + ownership + shares in one txn

All endpoints require the caller to be an admin. Non-admins get 403.
All mutations log an AuditLog row with action=DOCUMENT_ACCESS_CHANGE.
"""
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.document import Document
from app.security.auth import AuthContext
from app.security.audit import AuditEvent, get_audit_logger
from app.security.dependencies import require_permission
from app.core.rate_limit import rate_limit
from app.security.models import (
    AuditAction,
    DocumentPermission,
    DocumentRoleAccess,
    User,
    UserRole,
)


router = APIRouter(prefix="/api/admin/documents", tags=["Admin Documents"])


VALID_VISIBILITIES = {"private", "shared", "global"}
VALID_ACCESS_LEVELS = {"view", "manage"}
VALID_ROLES = {"admin", "user", "viewer"}


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

class AccessSummary(BaseModel):
    document_id: int
    visibility: str
    owner_user_id: Optional[int]
    owner_username: Optional[str]
    shared_user_ids: List[int] = []
    shared_role_access: List[dict] = []  # [{role, access_level}]
    created_at: Optional[str] = None
    updated_at: Optional[str] = None


class AdminDocumentRow(BaseModel):
    id: int
    original_name: str
    mime_type: str
    size_bytes: int
    status: str
    visibility: str
    owner_user_id: Optional[int]
    owner_username: Optional[str]
    created_at: Optional[str]


class AccessUpdateRequest(BaseModel):
    visibility: Optional[str] = None
    owner_user_id: Optional[int] = None
    # Replace the entire user-share set with these IDs (access_level='view' by default).
    # Pass an explicit list to replace, omit to leave unchanged.
    shared_user_ids: Optional[List[int]] = None
    # Replace the entire role-share set with these role names (each becomes access_level='view').
    # Pass an explicit list to replace, omit to leave unchanged.
    shared_roles: Optional[List[str]] = None
    # Optional: when provided, share rows use this access level (defaults to 'view').
    # Applies to both shared_user_ids and shared_roles in this request.
    access_level: Optional[str] = "view"


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("", response_model=List[AdminDocumentRow])
def list_admin_documents(
    db: Session = Depends(get_db),
    auth: AuthContext = Depends(require_permission("can_manage_users")),
):
    """Admin-only: list every document with owner + visibility metadata."""
    rows = (
        db.query(Document, User.username)
        .outerjoin(User, Document.owner_user_id == User.id)
        .order_by(Document.created_at.desc())
        .all()
    )
    return [
        AdminDocumentRow(
            id=d.id,
            original_name=d.original_name,
            mime_type=d.mime_type,
            size_bytes=d.size_bytes,
            status=d.status,
            visibility=d.visibility,
            owner_user_id=d.owner_user_id,
            owner_username=owner_username,
            created_at=d.created_at.isoformat() if d.created_at else None,
        )
        for d, owner_username in rows
    ]


@router.get("/{document_id}/access", response_model=AccessSummary)
def get_document_access(
    document_id: int,
    db: Session = Depends(get_db),
    auth: AuthContext = Depends(require_permission("can_manage_users")),
):
    """Admin-only: return the access summary for one document."""
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    owner_username = None
    if doc.owner_user_id is not None:
        owner = db.query(User).filter(User.id == doc.owner_user_id).first()
        if owner:
            owner_username = owner.username

    shared_user_ids = [
        row[0]
        for row in db.query(DocumentPermission.user_id)
        .filter(DocumentPermission.document_id == document_id)
        .all()
    ]
    role_rows = (
        db.query(DocumentRoleAccess.role, DocumentRoleAccess.access_level)
        .filter(DocumentRoleAccess.document_id == document_id)
        .order_by(DocumentRoleAccess.role.asc(), DocumentRoleAccess.access_level.asc())
        .all()
    )
    shared_role_access = [
        {"role": r, "access_level": lvl} for r, lvl in role_rows
    ]

    return AccessSummary(
        document_id=doc.id,
        visibility=doc.visibility,
        owner_user_id=doc.owner_user_id,
        owner_username=owner_username,
        shared_user_ids=shared_user_ids,
        shared_role_access=shared_role_access,
        created_at=doc.created_at.isoformat() if doc.created_at else None,
        updated_at=doc.updated_at.isoformat() if doc.updated_at else None,
    )


@router.patch("/{document_id}/access", response_model=AccessSummary)
def update_document_access(
    request: Request,
    document_id: int,
    payload: AccessUpdateRequest,
    db: Session = Depends(get_db),
    auth: AuthContext = Depends(require_permission("can_manage_users")),
    _=Depends(rate_limit(max_requests=20, window=60)),
):
    """
    Admin-only: update visibility, ownership, and/or shares in a single transaction.

    Fields omitted from the request body are left unchanged. To remove all user
    shares, pass shared_user_ids=[]. To remove all role shares, pass shared_roles=[].

    Logs AuditAction.DOCUMENT_ACCESS_CHANGE with the before/after values.
    """
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # Validate visibility if provided
    if payload.visibility is not None and payload.visibility not in VALID_VISIBILITIES:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid visibility '{payload.visibility}'. Must be one of: {sorted(VALID_VISIBILITIES)}",
        )

    # Validate access_level if provided
    access_level = payload.access_level or "view"
    if access_level not in VALID_ACCESS_LEVELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid access_level '{access_level}'. Must be one of: {sorted(VALID_ACCESS_LEVELS)}",
        )

    # Validate role names
    if payload.shared_roles is not None:
        bad_roles = [r for r in payload.shared_roles if r not in VALID_ROLES]
        if bad_roles:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid role(s): {bad_roles}. Must be one of: {sorted(VALID_ROLES)}",
            )

    # Validate owner_user_id if provided
    if payload.owner_user_id is not None:
        owner = db.query(User).filter(User.id == payload.owner_user_id).first()
        if not owner:
            raise HTTPException(
                status_code=400,
                detail=f"Owner user id {payload.owner_user_id} not found",
            )

    # Validate shared_user_ids if provided
    if payload.shared_user_ids is not None:
        existing_ids = {
            row[0]
            for row in db.query(User.id)
            .filter(User.id.in_(payload.shared_user_ids))
            .all()
        }
        missing = set(payload.shared_user_ids) - existing_ids
        if missing:
            raise HTTPException(
                status_code=400,
                detail=f"Shared user id(s) not found: {sorted(missing)}",
            )

    # Capture before-state for audit log
    before = {
        "visibility": doc.visibility,
        "owner_user_id": doc.owner_user_id,
        "shared_user_ids": [
            row[0]
            for row in db.query(DocumentPermission.user_id)
            .filter(DocumentPermission.document_id == document_id)
            .all()
        ],
        "shared_roles": [
            row[0]
            for row in db.query(DocumentRoleAccess.role)
            .filter(DocumentRoleAccess.document_id == document_id)
            .all()
        ],
    }

    try:
        # 1) Visibility
        if payload.visibility is not None:
            doc.visibility = payload.visibility

        # 2) Ownership
        if payload.owner_user_id is not None:
            doc.owner_user_id = payload.owner_user_id

        # 3) User shares (replace-all semantics)
        if payload.shared_user_ids is not None:
            db.query(DocumentPermission).filter(
                DocumentPermission.document_id == document_id
            ).delete()
            for uid in payload.shared_user_ids:
                share = DocumentPermission(
                    user_id=uid,
                    document_id=document_id,
                    can_read=True,
                    can_write=(access_level == "manage"),
                    access_level=access_level,
                    granted_by=auth.user_id,
                )
                db.add(share)

        # 4) Role shares (replace-all semantics)
        if payload.shared_roles is not None:
            db.query(DocumentRoleAccess).filter(
                DocumentRoleAccess.document_id == document_id
            ).delete()
            for role_name in payload.shared_roles:
                rshare = DocumentRoleAccess(
                    document_id=document_id,
                    role=role_name,
                    access_level=access_level,
                    granted_by=auth.user_id,
                )
                db.add(rshare)

        db.commit()
        db.refresh(doc)
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Failed to update access: {str(e)}")

    # Build after-state for audit log
    after = {
        "visibility": doc.visibility,
        "owner_user_id": doc.owner_user_id,
        "shared_user_ids": [
            row[0]
            for row in db.query(DocumentPermission.user_id)
            .filter(DocumentPermission.document_id == document_id)
            .all()
        ],
        "shared_roles": [
            row[0]
            for row in db.query(DocumentRoleAccess.role)
            .filter(DocumentRoleAccess.document_id == document_id)
            .all()
        ],
    }

    # Audit log
    try:
        audit_logger = get_audit_logger()
        event = AuditEvent(
            action=AuditAction.DOCUMENT_ACCESS_CHANGE,
            username=auth.username,
            user_id=auth.user_id,
            status="success",
            request_ip=request.client.host if request.client else None,
            request_user_agent=request.headers.get("user-agent", "")[:500],
            details={
                "document_id": document_id,
                "before": before,
                "after": after,
                "access_level": access_level,
            },
        )
        audit_logger.log(event)
    except Exception:
        pass  # Don't fail the request if audit logging fails

    # Return the new access summary (reuse the GET logic)
    return get_document_access(document_id=document_id, db=db, auth=auth)
