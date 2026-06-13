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
    
    def is_admin(self) -> bool:
        return self.role == UserRole.ADMIN
    
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
        if dev_username.startswith("admin_"):
            role = UserRole.ADMIN
            username = dev_username
        elif dev_username.startswith("viewer_"):
            role = UserRole.VIEWER
            username = dev_username
        else:
            role = UserRole.USER
            username = dev_username
        
        return AuthContext(
            user_id=None,  # Dev users don't have DB records
            username=username,
            role=role,
            is_authenticated=True,
            is_external=False,
        )


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
        # TODO: Implement JWT/session validation
        # For now, return None to fall through to dev auth or anonymous
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
            role=UserRole.VIEWER,  # Limited permissions for anonymous
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
    
    In production, this would include OIDC/JWT validation.
    In development, it includes dev header bypass.
    """
    return CompositeAuthProvider([
        DevAuthProvider(),      # Try dev headers first (local dev)
        DatabaseAuthProvider(), # Then try DB-backed auth (placeholder)
        AnonymousAuthProvider(), # Finally fall back to anonymous
    ])


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
            role=UserRole.VIEWER,
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
        @require_role(UserRole.ADMIN)
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
            
            # Admin can do anything
            if auth.is_admin():
                return func(*args, **kwargs)
            
            # Check role hierarchy
            role_hierarchy = {
                UserRole.ADMIN: 3,
                UserRole.USER: 2,
                UserRole.VIEWER: 1,
            }
            
            if role_hierarchy.get(auth.role, 0) < role_hierarchy.get(required_role, 0):
                raise HTTPException(status_code=403, detail="Insufficient permissions")
            
            return func(*args, **kwargs)
        return wrapper
    return decorator