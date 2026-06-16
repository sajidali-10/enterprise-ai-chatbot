"""
Admin User Management API

Endpoints under /api/admin/users require an authenticated admin user.

Provides:
- GET    /api/admin/users                    — list users
- GET    /api/admin/users/{user_id}          — get user + permissions
- POST   /api/admin/users                    — create user
- PATCH  /api/admin/users/{user_id}          — update email/full_name/role/is_active
- POST   /api/admin/users/{user_id}/reset-password
- DELETE /api/admin/users/{user_id}          — soft delete (deactivate)

All endpoints require:
- Valid JWT bearer token
- Active user
- Admin role (or can_manage_users permission)

Safeguards:
- Never return hashed_password
- Prevent deactivating / demoting / hard-removing the last active admin
"""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.security.dependencies import require_admin
from app.security.auth import AuthContext, get_role_permissions
from app.security.models import User, UserRole, AuditAction
from app.security.password import hash_password, validate_password_policy
from app.security.audit import AuditEvent, get_audit_logger
from app.core.rate_limit import rate_limit
from app.schemas.admin import (
    CreateUserRequest,
    MessageResponse,
    ResetPasswordRequest,
    SafeUser,
    SafeUserWithPermissions,
    UpdateUserRequest,
)


router = APIRouter(prefix="/api/admin/users", tags=["Admin Users"])


def _count_active_admins(db: Session, exclude_user_id: Optional[int] = None) -> int:
    """Count active admin users, optionally excluding one (for self-edit checks)."""
    q = db.query(User).filter(User.role == UserRole.ADMIN, User.is_active == True)  # noqa: E712
    if exclude_user_id is not None:
        q = q.filter(User.id != exclude_user_id)
    return q.count()


def _ensure_not_last_admin(
    db: Session,
    user: User,
    new_role: Optional[UserRole] = None,
    new_is_active: Optional[bool] = None,
) -> None:
    """
    Prevent modifications that would leave the system without any active admin.

    `new_role` and `new_is_active` represent the PROPOSED values after the change.
    If either is None, the current value is used.

    Raises HTTPException 400 if the change would remove the last active admin.
    """
    current_role = user.role if isinstance(user.role, UserRole) else UserRole(user.role)
    is_currently_admin = current_role == UserRole.ADMIN and user.is_active

    if not is_currently_admin:
        # User is not currently an active admin; no risk to remove admin capability
        return

    target_role = new_role if new_role is not None else current_role
    target_active = new_is_active if new_is_active is not None else user.is_active

    will_still_be_active_admin = (
        target_role == UserRole.ADMIN and target_active
    )

    if will_still_be_active_admin:
        return

    remaining = _count_active_admins(db, exclude_user_id=user.id)
    if remaining == 0:
        raise HTTPException(
            status_code=400,
            detail="Cannot demote or deactivate the last active admin user.",
        )


def _to_safe_user(user: User) -> SafeUser:
    return SafeUser(
        id=user.id,
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        role=user.role.value if isinstance(user.role, UserRole) else str(user.role),
        is_active=user.is_active,
        created_at=user.created_at,
        updated_at=user.updated_at,
        last_login=user.last_login,
    )


def _to_safe_user_with_perms(user: User) -> SafeUserWithPermissions:
    role = user.role if isinstance(user.role, UserRole) else UserRole(user.role)
    perms = get_role_permissions(role)
    return SafeUserWithPermissions(
        id=user.id,
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        role=role.value,
        is_active=user.is_active,
        created_at=user.created_at,
        updated_at=user.updated_at,
        last_login=user.last_login,
        permissions=perms,
    )


@router.get("", response_model=list[SafeUser])
def list_users(
    db: Session = Depends(get_db),
    auth: AuthContext = Depends(require_admin),
):
    """List all users (admin only). Does not include hashed_password."""
    users = db.query(User).order_by(User.created_at.desc()).all()
    return [_to_safe_user(u) for u in users]


@router.get("/{user_id}", response_model=SafeUserWithPermissions)
def get_user(
    user_id: int,
    db: Session = Depends(get_db),
    auth: AuthContext = Depends(require_admin),
):
    """Get a single user with effective permissions (admin only)."""
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return _to_safe_user_with_perms(user)


@router.post("", response_model=SafeUser, status_code=201)
def create_user(
    payload: CreateUserRequest,
    db: Session = Depends(get_db),
    auth: AuthContext = Depends(require_admin),
    _=Depends(rate_limit(max_requests=20, window=60)),
):
    """Create a new user (admin only). Password is bcrypt-hashed."""
    # Validate password policy
    policy_error = validate_password_policy(payload.password)
    if policy_error:
        raise HTTPException(status_code=400, detail=policy_error)

    username = payload.username.strip()
    email = payload.email.strip().lower()

    # Uniqueness checks
    if db.query(User).filter(User.username == username).first():
        raise HTTPException(status_code=409, detail="Username already exists")
    if db.query(User).filter(User.email == email).first():
        raise HTTPException(status_code=409, detail="Email already exists")

    user = User(
        username=username,
        email=email,
        full_name=payload.full_name,
        hashed_password=hash_password(payload.password),
        role=payload.role,
        is_active=payload.is_active,
        is_external=False,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    # Audit log: user created
    try:
        audit = get_audit_logger()
        event = AuditEvent(
            action=AuditAction.USER_CREATED,
            username=auth.username,
            user_id=auth.user_id,
            status="success",
            details={"created_user_id": user.id, "created_username": user.username, "role": user.role.value},
        )
        audit.log(event)
    except Exception:
        pass

    return _to_safe_user(user)


@router.patch("/{user_id}", response_model=SafeUser)
def update_user(
    user_id: int,
    payload: UpdateUserRequest,
    db: Session = Depends(get_db),
    auth: AuthContext = Depends(require_admin),
    _=Depends(rate_limit(max_requests=20, window=60)),
):
    """Update email / full_name / role / is_active (admin only).

    Last-active-admin safeguard: cannot demote or deactivate the only active admin.
    """
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Last-admin safeguard BEFORE mutating the user
    if payload.role is not None or payload.is_active is not None:
        _ensure_not_last_admin(
            db,
            user,
            new_role=payload.role,
            new_is_active=payload.is_active,
        )

    # Apply updates
    if payload.email is not None:
        new_email = payload.email.strip().lower()
        existing = db.query(User).filter(User.email == new_email, User.id != user_id).first()
        if existing:
            raise HTTPException(status_code=409, detail="Email already exists")
        user.email = new_email

    if payload.full_name is not None:
        user.full_name = payload.full_name

    if payload.role is not None:
        user.role = payload.role

    if payload.is_active is not None:
        user.is_active = payload.is_active
        # Invalidate sessions on deactivation
        if not payload.is_active:
            user.token_version += 1

    db.commit()
    db.refresh(user)

    # Audit log: user updated
    try:
        audit = get_audit_logger()
        event = AuditEvent(
            action=AuditAction.USER_UPDATED,
            username=auth.username,
            user_id=auth.user_id,
            status="success",
            details={"target_user_id": user_id, "target_username": user.username},
        )
        audit.log(event)
    except Exception:
        pass

    return _to_safe_user(user)


@router.post("/{user_id}/reset-password", response_model=MessageResponse)
def reset_password(
    user_id: int,
    payload: ResetPasswordRequest,
    db: Session = Depends(get_db),
    auth: AuthContext = Depends(require_admin),
    _=Depends(rate_limit(max_requests=20, window=60)),
):
    """Reset a user's password (admin only). Returns success without echoing the password."""
    # Validate password policy
    policy_error = validate_password_policy(payload.new_password)
    if policy_error:
        raise HTTPException(status_code=400, detail=policy_error)

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    user.hashed_password = hash_password(payload.new_password)
    user.token_version += 1
    db.commit()

    # Audit log: password reset
    try:
        audit = get_audit_logger()
        event = AuditEvent(
            action=AuditAction.PASSWORD_RESET,
            username=auth.username,
            user_id=auth.user_id,
            status="success",
            details={"target_user_id": user_id, "target_username": user.username},
        )
        audit.log(event)
    except Exception:
        pass

    return MessageResponse(success=True, detail="Password reset successfully")


@router.delete("/{user_id}", response_model=SafeUser)
def deactivate_user(
    user_id: int,
    db: Session = Depends(get_db),
    auth: AuthContext = Depends(require_admin),
    _=Depends(rate_limit(max_requests=20, window=60)),
):
    """Soft-delete a user by deactivating them (admin only).

    Last-active-admin safeguard: cannot deactivate the only active admin.
    """
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    _ensure_not_last_admin(db, user, new_is_active=False)

    user.is_active = False
    user.token_version += 1
    db.commit()
    db.refresh(user)

    # Audit log: user deactivated
    try:
        audit = get_audit_logger()
        event = AuditEvent(
            action=AuditAction.USER_DEACTIVATED,
            username=auth.username,
            user_id=auth.user_id,
            status="success",
            details={"target_user_id": user_id, "target_username": user.username},
        )
        audit.log(event)
    except Exception:
        pass

    return _to_safe_user(user)
