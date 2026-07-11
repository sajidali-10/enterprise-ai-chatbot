"""
FastAPI dependencies for authentication and authorization.

These are designed to be used with Depends() on route handlers.
"""

from typing import Optional

from fastapi import Request, HTTPException, Depends

from app.security.auth import (
    AuthContext,
    authenticate_request,
    get_role_permissions,
    UserRole,
)


def get_auth_context(request: Request) -> Optional[AuthContext]:
    """
    Extract authentication context from the request.

    Returns None if the security module is unavailable (should not happen).
    """
    try:
        return authenticate_request(request)
    except Exception:
        # Defensive fallback
        return AuthContext(
            user_id=None,
            username="anonymous",
            role=UserRole.viewer,
            is_authenticated=False,
        )


def require_active_user(auth: AuthContext = Depends(get_auth_context)) -> AuthContext:
    """Require an authenticated and active user."""
    if not auth or not auth.is_authenticated:
        raise HTTPException(status_code=401, detail="Authentication required")
    # For JWT users, is_active is checked during token validation.
    # For dev users, we consider them always active.
    return auth


def require_role(required_role: UserRole):
    """
    Return a FastAPI dependency that enforces a minimum role.

    Admin bypasses all role checks.
    """
    def checker(auth: AuthContext = Depends(get_auth_context)) -> AuthContext:
        if not auth or not auth.is_authenticated:
            raise HTTPException(status_code=401, detail="Authentication required")

        if auth.is_admin():
            return auth

        role_hierarchy = {
            UserRole.sysadmin: 4,
            UserRole.admin: 3,
            UserRole.user: 2,
            UserRole.viewer: 1,
        }

        if role_hierarchy.get(auth.role, 0) < role_hierarchy.get(required_role, 0):
            raise HTTPException(status_code=403, detail="Insufficient permissions")

        return auth
    return checker


def require_permission(permission: str):
    """
    Return a FastAPI dependency that enforces a specific permission.

    Example:
        @router.post("/upload")
        def upload(..., auth: AuthContext = Depends(require_permission("can_upload_documents"))):
            ...
    """
    def checker(auth: AuthContext = Depends(get_auth_context)) -> AuthContext:
        if not auth or not auth.is_authenticated:
            raise HTTPException(status_code=401, detail="Authentication required")

        perms = get_role_permissions(auth.role)
        if not perms.get(permission):
            raise HTTPException(status_code=403, detail=f"Permission denied: {permission}")

        return auth
    return checker


def require_admin(auth: AuthContext = Depends(get_auth_context)) -> AuthContext:
    """Require an admin user."""
    if not auth or not auth.is_authenticated:
        raise HTTPException(status_code=401, detail="Authentication required")

    if not auth.is_admin():
        raise HTTPException(status_code=403, detail="Admin access required")

    return auth


def require_sysadmin(auth: AuthContext = Depends(get_auth_context)) -> AuthContext:
    """Require a sysadmin user."""
    if not auth or not auth.is_authenticated:
        raise HTTPException(status_code=401, detail="Authentication required")

    if not auth.is_sysadmin():
        raise HTTPException(status_code=403, detail="Sysadmin access required")

    return auth
