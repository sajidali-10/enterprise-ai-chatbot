"""
Authentication Layer

Provides authentication primitives supporting:
- Local development with dev user headers
- Future SSO/OIDC integration (placeholder)
- User identity extraction from requests

The auth layer is designed to be swapped out for enterprise SSO
without changing the rest of the application.
"""

import json
import secrets
from typing import Optional
from dataclasses import dataclass
from datetime import datetime

from fastapi import Request, HTTPException, Header
from sqlalchemy.orm import Session

from app.security.models import User, UserRole
from app.db.session import SessionLocal
from app.core.config import settings
from app.security.jwt import decode_token
from jwt import PyJWTError


# Dev mode: Secret header value for local development authentication bypass
DEV_USER_HEADER = "X-Dev-User"
DEV_USER_SECRET = "dev-secret-change-in-production"  # Configure via env in production


@dataclass
class AuthContext:
    """
    Authentication context passed through the request lifecycle.
    
    Contains the authenticated user information and session details.
    """
    user_id: Optional[int]
    username: str
    role: UserRole
    is_authenticated: bool
    is_external: bool = False
    session_id: Optional[str] = None
    is_protected: bool = False
    
    def is_admin(self) -> bool:
        return self.role in (UserRole.admin, UserRole.sysadmin)
    
    def is_sysadmin(self) -> bool:
        return self.role == UserRole.sysadmin
    
    def is_active(self) -> bool:
        """Check if user is active. Dev users are always considered active."""
        return True  # Dev users bypass this check
    
    def to_dict(self) -> dict:
        return {
            "user_id": self.user_id,
            "username": self.username,
            "role": self.role.value if isinstance(self.role, UserRole) else self.role,
            "is_authenticated": self.is_authenticated,
            "is_external": self.is_external,
            "session_id": self.session_id,
            "is_protected": self.is_protected,
        }


class AuthProviderBase:
    """
    Abstract base class for authentication providers.
    
    Implement this interface to add new authentication methods
    (e.g., OIDC, SAML, OAuth2).
    """
    
    def authenticate(self, request: Request) -> Optional[AuthContext]:
        """
        Authenticate a request and return the auth context.
        Returns None if authentication fails (request continues anonymously).
        """
        raise NotImplementedError


class DevAuthProvider(AuthProviderBase):
    """
    Development authentication provider.
    
    Reads a special header to create a dev user context without
    requiring real authentication. ONLY for local development!
    
    Usage:
        curl -H "X-Dev-User: admin" http://localhost:8000/api/chat
    """
    
    def authenticate(self, request: Request) -> Optional[AuthContext]:
        dev_username = request.headers.get(DEV_USER_HEADER)
        if not dev_username:
            return None
        
        # Determine role from username convention
        if dev_username.startswith("sysadmin_"):
            role = UserRole.sysadmin
            username = dev_username
        elif dev_username.startswith("admin_"):
            role = UserRole.admin
            username = dev_username
        elif dev_username.startswith("viewer_"):
            role = UserRole.viewer
            username = dev_username
        else:
            role = UserRole.user
            username = dev_username
        
        return AuthContext(
            user_id=None,  # Dev users don't have DB records
            username=username,
            role=role,
            is_authenticated=True,
            is_external=False,
        )


class JWTAuthProvider(AuthProviderBase):
    """
    JWT Bearer token authentication provider.

    Validates Authorization: Bearer <token> header, decodes the JWT,
    and looks up the user in the database.
    """

    def authenticate(self, request: Request) -> Optional[AuthContext]:
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.lower().startswith("bearer "):
            return None

        token = auth_header[7:]  # Remove "Bearer " prefix
        if not token:
            return None

        if not settings.JWT_SECRET_KEY:
            return None

        try:
            payload = decode_token(token)
        except PyJWTError:
            return None

        user_id_str = payload.get("sub")
        if not user_id_str:
            return None

        try:
            user_id = int(user_id_str)
        except ValueError:
            return None

        db = SessionLocal()
        try:
            user = db.query(User).filter(User.id == user_id).first()
            if not user or not user.is_active:
                return None

            # Check token_version for session invalidation
            token_version_in_jwt = payload.get("tv")
            if token_version_in_jwt is not None and token_version_in_jwt != user.token_version:
                return None

            return AuthContext(
                user_id=user.id,
                username=user.username,
                role=user.role if isinstance(user.role, UserRole) else UserRole(user.role),
                is_authenticated=True,
                is_external=False,
                is_protected=user.is_protected,
            )
        finally:
            db.close()


class DatabaseAuthProvider(AuthProviderBase):
    """
    Database-backed authentication provider.
    
    Looks up user from the users table based on session or token.
    Placeholder for when real auth is implemented.
    """
    
    def authenticate(self, request: Request) -> Optional[AuthContext]:
        """
        Authenticate using database user lookup.
        This is a placeholder that would verify JWT tokens, sessions, etc.
        """
        # Placeholder — JWT now handled by JWTAuthProvider
        return None


class AnonymousAuthProvider(AuthProviderBase):
    """
    Anonymous/proprietary (internal service) authentication.
    
    Used for service-to-service communication or when no auth is present.
    """
    
    def authenticate(self, request: Request) -> Optional[AuthContext]:
        return AuthContext(
            user_id=None,
            username="anonymous",
            role=UserRole.viewer,  # Limited permissions for anonymous
            is_authenticated=False,
            is_external=False,
        )


class CompositeAuthProvider(AuthProviderBase):
    """
    Composite auth provider that tries multiple auth methods in sequence.
    
    Auth methods are tried in order until one succeeds.
    """
    
    def __init__(self, providers: list[AuthProviderBase]):
        self.providers = providers
    
    def authenticate(self, request: Request) -> Optional[AuthContext]:
        last_result = None
        for provider in self.providers:
            result = provider.authenticate(request)
            if result and result.is_authenticated:
                return result
            last_result = result
        return last_result


def get_auth_provider() -> CompositeAuthProvider:
    """
    Get the configured auth provider composition.
    
    Order depends on AUTH_MODE:
    - local: JWT first, then anonymous
    - dev: dev headers first, then JWT, then anonymous
    """
    providers: list[AuthProviderBase] = []

    if settings.AUTH_MODE == "dev":
        providers.append(DevAuthProvider())
        providers.append(JWTAuthProvider())
    else:
        # Default to local auth (JWT-based)
        providers.append(JWTAuthProvider())
        # Only include dev auth as last resort if explicitly enabled
        if settings.DEV_AUTH_ENABLED:
            providers.append(DevAuthProvider())

    providers.append(DatabaseAuthProvider())
    providers.append(AnonymousAuthProvider())

    return CompositeAuthProvider(providers)


def authenticate_request(request: Request) -> AuthContext:
    """
    Authenticate an incoming request and return the auth context.
    
    This is the main entry point for auth in the application.
    """
    provider = get_auth_provider()
    context = provider.authenticate(request)
    
    if context is None:
        # Should not happen - AnonymousAuthProvider should always return a context
        context = AuthContext(
            user_id=None,
            username="anonymous",
            role=UserRole.viewer,
            is_authenticated=False,
        )
    
    return context


def require_auth(func):
    """
    Decorator to require authentication for an endpoint.
    
    Usage:
        @router.post("/protected")
        @require_auth
        def protected_endpoint(request: Request, auth: AuthContext = Depends(get_auth)):
            ...
    """
    def wrapper(*args, **kwargs):
        request = kwargs.get('request') or (len(args) > 0 and args[0] if isinstance(args[0], Request) else None)
        if not request:
            raise HTTPException(status_code=500, detail="Internal auth error")
        
        auth = authenticate_request(request)
        if not auth.is_authenticated:
            raise HTTPException(status_code=401, detail="Authentication required")
        
        kwargs['auth'] = auth
        return func(*args, **kwargs)
    return wrapper


def require_role(required_role: UserRole):
    """
    Decorator to require a specific role for an endpoint.
    
    Usage:
        @router.post("/admin")
        @require_role(UserRole.admin)
        def admin_endpoint(auth: AuthContext = Depends(get_auth)):
            ...
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            auth: AuthContext = kwargs.get('auth')
            if not auth:
                raise HTTPException(status_code=500, detail="Internal auth error")
            
            if not auth.is_authenticated:
                raise HTTPException(status_code=401, detail="Authentication required")
            
            # SYSADMIN and ADMIN can do anything
            if auth.is_sysadmin() or auth.is_admin():
                return func(*args, **kwargs)
            
            # Check role hierarchy
            role_hierarchy = {
                UserRole.sysadmin: 4,
                UserRole.admin: 3,
                UserRole.user: 2,
                UserRole.viewer: 1,
            }
            
            if role_hierarchy.get(auth.role, 0) < role_hierarchy.get(required_role, 0):
                raise HTTPException(status_code=403, detail="Insufficient permissions")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator


def get_role_permissions(role: UserRole) -> dict:
    """Return a permission map for a given user role."""
    perms = {
        "can_use_general_chat": False,
        "can_use_knowledge_base": False,
        "can_use_debug": False,
        "can_view_documents": False,
        "can_upload_documents": False,
        "can_reindex_documents": False,
        "can_delete_documents": False,
        "can_access_observability": False,
        "can_access_evaluations": False,
        "can_submit_feedback": False,
        "can_manage_users": False,
    }
    if role in (UserRole.sysadmin, UserRole.admin):
        perms.update({
            "can_use_general_chat": True,
            "can_use_knowledge_base": True,
            "can_use_debug": True,
            "can_view_documents": True,
            "can_upload_documents": True,
            "can_reindex_documents": True,
            "can_delete_documents": True,
            "can_access_observability": True,
            "can_access_evaluations": True,
            "can_submit_feedback": True,
            "can_manage_users": True,
        })
    elif role == UserRole.user:
        perms.update({
            "can_use_general_chat": True,
            "can_use_knowledge_base": True,
            "can_use_debug": False,
            "can_view_documents": True,
            "can_upload_documents": True,
            "can_reindex_documents": False,
            "can_delete_documents": False,
            "can_access_observability": False,
            "can_access_evaluations": False,
            "can_submit_feedback": True,
            "can_manage_users": False,
        })
    elif role == UserRole.viewer:
        perms.update({
            "can_use_general_chat": False,
            "can_use_knowledge_base": True,
            "can_use_debug": False,
            "can_view_documents": True,
            "can_upload_documents": False,
            "can_reindex_documents": False,
            "can_delete_documents": False,
            "can_access_observability": False,
            "can_access_evaluations": False,
            "can_submit_feedback": True,
            "can_manage_users": False,
        })
    return perms