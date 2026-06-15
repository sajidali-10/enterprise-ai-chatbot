"""
Security Module

Provides authentication, authorization, and audit logging:
- auth: Authentication layer with dev mode and SSO placeholders
- permissions: Document-level permission checking and filtering
- audit: Audit logging for compliance and monitoring

Models:
- User: User accounts and roles
- DocumentPermission: Document-level access control
- AuditLog: Audit trail for all user actions
"""

from app.security.auth import (
    AuthContext,
    AuthProviderBase,
    DevAuthProvider,
    DatabaseAuthProvider,
    AnonymousAuthProvider,
    CompositeAuthProvider,
    get_auth_provider,
    authenticate_request,
    require_auth,
    require_role,
    DEV_USER_HEADER,
    DEV_USER_SECRET,
)
from app.security.permissions import (
    PermissionChecker,
    PermissionResult,
    get_permission_checker,
    check_document_access,
    filter_documents_by_permission,
)
from app.security.audit import (
    AuditLogger,
    AuditEvent,
    get_audit_logger,
    log_rag_query,
    timed_audit_log,
)
from app.security.models import (
    User,
    UserRole,
    DocumentPermission,
    DocumentRoleAccess,
    AuditLog as AuditLogModel,
    AuditAction,
)

__all__ = [
    # Auth
    "AuthContext",
    "AuthProviderBase",
    "DevAuthProvider",
    "DatabaseAuthProvider",
    "AnonymousAuthProvider",
    "CompositeAuthProvider",
    "get_auth_provider",
    "authenticate_request",
    "require_auth",
    "require_role",
    "DEV_USER_HEADER",
    "DEV_USER_SECRET",
    # Permissions
    "PermissionChecker",
    "PermissionResult",
    "get_permission_checker",
    "check_document_access",
    "filter_documents_by_permission",
    # Audit
    "AuditLogger",
    "AuditEvent",
    "get_audit_logger",
    "log_rag_query",
    "timed_audit_log",
    # Models
    "User",
    "UserRole",
    "DocumentPermission",
    "DocumentRoleAccess",
    "AuditLogModel",
    "AuditAction",
]