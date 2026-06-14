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
from app.security.models import User, UserRole
from app.security.jwt import create_access_token, decode_token
from app.security.password import verify_password
from app.security.auth import get_role_permissions
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


@router.post("/login", response_model=LoginResponse)
def login(
    request: Request,
    credentials: LoginRequest,
    db: Session = Depends(get_db),
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

    if not user:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if not user.is_active:
        raise HTTPException(status_code=401, detail="User account is inactive")

    if not user.hashed_password:
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if not verify_password(credentials.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    # Update last_login
    from sqlalchemy import func
    user.last_login = func.now()
    db.commit()

    token = create_access_token(
        subject=str(user.id),
        extra_claims={"role": user.role.value if isinstance(user.role, UserRole) else user.role},
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

    return _build_user_info(user)


@router.post("/logout")
def logout(request: Request):
    """
    Client-side logout acknowledgment.

    JWT tokens are stateless; the client is responsible for removing the token.
    """
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
