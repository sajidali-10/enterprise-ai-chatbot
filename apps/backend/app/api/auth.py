"""
Authentication API Endpoints

Provides:
- POST /api/auth/login   — Authenticate with username/email + password, receive JWT
- GET  /api/auth/me      — Get current authenticated user info
- POST /api/auth/logout  — Client-side logout acknowledgment
"""

from datetime import timedelta
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, Header
from sqlalchemy.orm import Session
from jwt import ExpiredSignatureError, InvalidTokenError

from app.db.session import get_db
from app.core.config import settings
from app.core.rate_limit import rate_limit
from app.security.models import User, UserRole, AuditAction
from app.security.jwt import create_access_token, decode_token
from app.security.password import verify_password
from app.security.auth import get_role_permissions
from app.security.audit import AuditEvent, get_audit_logger
from app.schemas.auth import LoginRequest, LoginResponse, UserInfo, UserPermissions

router = APIRouter(prefix="/api/auth", tags=["Authentication"])


def _build_user_info(user: User) -> UserInfo:
    """Build a UserInfo response from a User model."""
    role = user.role
    if isinstance(role, str):
        role = UserRole(role)
    perms = get_role_permissions(role)
    return UserInfo(
        id=user.id,
        username=user.username,
        email=user.email,
        full_name=user.full_name,
        role=role.value,
        is_active=user.is_active,
        permissions=UserPermissions(**perms),
    )


def _login_rate_limit_key(request: Request) -> str:
    """Rate limit key for login: per-IP + per-username combination."""
    client_ip = request.client.host if request.client else "unknown"
    # Try to get username from request body for per-account limiting
    body = getattr(request.state, '_body', None)
    username = "unknown"
    if body and hasattr(body, 'get'):
        username = body.get('username_or_email', 'unknown')
    return f"login:ip:{client_ip}:user:{username}"


def _get_client_ip(request: Request) -> str:
    """Extract client IP from request, handling proxies."""
    forwarded_for = request.headers.get("X-Forwarded-For")
    if forwarded_for:
        return forwarded_for.split(",")[0].strip()
    real_ip = request.headers.get("X-Real-IP")
    if real_ip:
        return real_ip
    if request.client:
        return request.client.host
    return "unknown"


# Rate limit: 5 login attempts per 60 seconds per IP
@router.post("/login", response_model=LoginResponse)
def login(
    request: Request,
    credentials: LoginRequest,
    db: Session = Depends(get_db),
    _=Depends(rate_limit(max_requests=5, window=60)),
):
    """
    Authenticate a user with username/email and password.

    Returns a JWT access token and user profile with permissions.
    """
    # Look up user by username or email
    user: Optional[User] = (
        db.query(User)
        .filter(
            (User.username == credentials.username_or_email)
            | (User.email == credentials.username_or_email)
        )
        .first()
    )

    # Audit logging helpers
    def _audit_login(action: AuditAction, target_user_id: Optional[int] = None, success: bool = True, reason: Optional[str] = None):
        try:
            audit = get_audit_logger()
            event = AuditEvent(
                action=action,
                username=credentials.username_or_email,
                user_id=target_user_id,
                request_ip=_get_client_ip(request),
                request_user_agent=request.headers.get("user-agent", "")[:500],
                status="success" if success else "failure",
                error_message=reason,
                details={"username_or_email": credentials.username_or_email},
            )
            audit.log(event)
        except Exception:
            pass  # Don't fail the request if audit logging fails

    if not user:
        _audit_login(AuditAction.LOGIN_FAILURE, success=False, reason="User not found")
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if getattr(user, "is_deleted", False):
        _audit_login(AuditAction.LOGIN_FAILURE, target_user_id=user.id, success=False, reason="User account is deleted")
        raise HTTPException(status_code=401, detail="User account is deleted")

    if not user.is_active:
        _audit_login(AuditAction.LOGIN_FAILURE, target_user_id=user.id, success=False, reason="User account is inactive")
        raise HTTPException(status_code=401, detail="User account is inactive")

    if not user.hashed_password:
        _audit_login(AuditAction.LOGIN_FAILURE, target_user_id=user.id, success=False, reason="No password set")
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if not verify_password(credentials.password, user.hashed_password):
        _audit_login(AuditAction.LOGIN_FAILURE, target_user_id=user.id, success=False, reason="Invalid password")
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # Update last_login
    from sqlalchemy import func
    user.last_login = func.now()
    db.commit()

    # Log successful login
    _audit_login(AuditAction.LOGIN_SUCCESS, target_user_id=user.id, success=True)

    token = create_access_token(
        subject=str(user.id),
        extra_claims={"role": user.role.value if isinstance(user.role, UserRole) else user.role},
        token_version=user.token_version,
    )

    return LoginResponse(
        access_token=token,
        token_type="bearer",
        expires_in=settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user=_build_user_info(user),
    )


@router.get("/me", response_model=UserInfo)
def get_current_user_info(
    request: Request,
    authorization: Optional[str] = Header(None),
    db: Session = Depends(get_db),
):
    """
    Get information about the currently authenticated user.

    Requires a valid Bearer token in the Authorization header.
    """
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")

    token = authorization[7:]  # Remove "Bearer " prefix

    try:
        payload = decode_token(token)
    except ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")
    except InvalidTokenError:
        raise HTTPException(status_code=401, detail="Invalid token")
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))

    user_id_str = payload.get("sub")
    if not user_id_str:
        raise HTTPException(status_code=401, detail="Invalid token payload")

    try:
        user_id = int(user_id_str)
    except ValueError:
        raise HTTPException(status_code=401, detail="Invalid token payload")

    user: Optional[User] = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=401, detail="User not found")

    if not user.is_active:
        raise HTTPException(status_code=401, detail="User account is inactive")

    # Check token_version for session invalidation
    token_version_in_jwt = payload.get("tv")
    if token_version_in_jwt is not None and token_version_in_jwt != user.token_version:
        raise HTTPException(status_code=401, detail="Token invalidated")

    return _build_user_info(user)


@router.post("/logout")
def logout(request: Request):
    """
    Client-side logout acknowledgment.

    JWT tokens are stateless; the client is responsible for removing the token.
    """
    # Attempt to identify the user from the Authorization header for audit
    auth_header = request.headers.get("Authorization", "")
    if auth_header.lower().startswith("bearer "):
        token = auth_header[7:]
        try:
            from app.security.jwt import decode_token
            payload = decode_token(token)
            user_id = int(payload.get("sub", "0"))
            try:
                audit = get_audit_logger()
                event = AuditEvent(
                    action=AuditAction.USER_LOGOUT,
                    username="unknown",
                    user_id=user_id,
                    request_ip=_get_client_ip(request),
                    request_user_agent=request.headers.get("user-agent", "")[:500],
                    status="success",
                )
                audit.log(event)
            except Exception:
                pass
        except Exception:
            pass

    return {"success": True, "detail": "Logged out successfully"}


@router.get("/config")
def get_auth_config():
    """
    Return safe authentication configuration for the frontend.

    Does not expose secrets. Only reveals which authentication mode is active
    so the frontend can adjust its UI accordingly.
    """
    return {
        "auth_mode": settings.AUTH_MODE,
        "dev_auth_enabled": settings.DEV_AUTH_ENABLED,
    }
