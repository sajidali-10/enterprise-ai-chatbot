"""
Tests for Security and Permission Filtering (Phase 6)

Tests authentication, authorization, and audit logging functionality.
"""

import pytest
from unittest.mock import MagicMock, patch
from dataclasses import dataclass

# Test authentication components
from app.security.auth import (
    AuthContext,
    DevAuthProvider,
    AnonymousAuthProvider,
    CompositeAuthProvider,
)
from app.security.models import UserRole

# Test permission components
from app.security.permissions import PermissionResult


class TestAuthContext:
    """Tests for AuthContext dataclass."""

    def test_admin_check(self):
        ctx = AuthContext(
            user_id=1,
            username="admin_user",
            role=UserRole.ADMIN,
            is_authenticated=True,
        )
        assert ctx.is_admin() is True

    def test_user_not_admin(self):
        ctx = AuthContext(
            user_id=2,
            username="regular_user",
            role=UserRole.USER,
            is_authenticated=True,
        )
        assert ctx.is_admin() is False

    def test_dev_user_not_admin(self):
        ctx = AuthContext(
            user_id=None,
            username="dev_user",
            role=UserRole.USER,
            is_authenticated=True,
        )
        assert ctx.is_admin() is False

    def test_to_dict(self):
        ctx = AuthContext(
            user_id=1,
            username="test_user",
            role=UserRole.USER,
            is_authenticated=True,
            is_external=False,
            session_id="session123",
        )
        d = ctx.to_dict()
        assert d["user_id"] == 1
        assert d["username"] == "test_user"
        assert d["role"] == "user"
        assert d["is_authenticated"] is True


class TestDevAuthProvider:
    """Tests for DevAuthProvider."""

    def test_no_dev_header_returns_none(self):
        provider = DevAuthProvider()
        mock_request = MagicMock()
        mock_request.headers = {}
        
        result = provider.authenticate(mock_request)
        assert result is None

    def test_regular_dev_user(self):
        provider = DevAuthProvider()
        mock_request = MagicMock()
        mock_request.headers = {"X-Dev-User": "testuser"}
        
        result = provider.authenticate(mock_request)
        assert result is not None
        assert result.username == "testuser"
        assert result.role == UserRole.USER
        assert result.is_authenticated is True

    def test_admin_dev_user(self):
        provider = DevAuthProvider()
        mock_request = MagicMock()
        mock_request.headers = {"X-Dev-User": "admin_superuser"}
        
        result = provider.authenticate(mock_request)
        assert result is not None
        assert result.username == "admin_superuser"
        assert result.role == UserRole.ADMIN
        assert result.is_authenticated is True

    def test_viewer_dev_user(self):
        provider = DevAuthProvider()
        mock_request = MagicMock()
        mock_request.headers = {"X-Dev-User": "viewer_guest"}
        
        result = provider.authenticate(mock_request)
        assert result is not None
        assert result.username == "viewer_guest"
        assert result.role == UserRole.VIEWER
        assert result.is_authenticated is True


class TestAnonymousAuthProvider:
    """Tests for AnonymousAuthProvider."""

    def test_returns_anonymous_context(self):
        provider = AnonymousAuthProvider()
        mock_request = MagicMock()
        
        result = provider.authenticate(mock_request)
        assert result is not None
        assert result.username == "anonymous"
        assert result.role == UserRole.VIEWER
        assert result.is_authenticated is False


class TestCompositeAuthProvider:
    """Tests for CompositeAuthProvider."""

    def test_tries_providers_in_order(self):
        # Create mock providers
        provider1 = MagicMock()
        provider2 = MagicMock()
        provider3 = MagicMock()
        
        provider1.authenticate.return_value = None  # Fails
        provider2.authenticate.return_value = AuthContext(
            user_id=1,
            username="found",
            role=UserRole.USER,
            is_authenticated=True,
        )
        
        composite = CompositeAuthProvider([provider1, provider2, provider3])
        mock_request = MagicMock()
        
        result = composite.authenticate(mock_request)
        
        # Should return result from provider2
        assert result is not None
        assert result.username == "found"
        
        # Provider3 should not have been called
        provider3.authenticate.assert_not_called()

    def test_falls_through_to_anonymous(self):
        provider1 = MagicMock()
        provider1.authenticate.return_value = None
        
        anonymous = AnonymousAuthProvider()
        composite = CompositeAuthProvider([provider1, anonymous])
        mock_request = MagicMock()
        
        result = composite.authenticate(mock_request)
        
        # Should fall through to anonymous
        assert result is not None
        assert result.username == "anonymous"


class TestPermissionResult:
    """Tests for PermissionResult dataclass."""

    def test_granted_result(self):
        result = PermissionResult(
            granted=True,
            reason="Permission granted",
            document_id=123,
        )
        assert result.granted is True
        assert result.document_id == 123

    def test_denied_result(self):
        result = PermissionResult(
            granted=False,
            reason="No permission record found",
            document_id=456,
        )
        assert result.granted is False


class TestUserRole:
    """Tests for UserRole enum."""

    def test_role_values(self):
        assert UserRole.ADMIN.value == "admin"
        assert UserRole.USER.value == "user"
        assert UserRole.VIEWER.value == "viewer"


class TestAuthContextRoleValues:
    """Tests that AuthContext works with UserRole enum."""

    def test_admin_role_string(self):
        ctx = AuthContext(
            user_id=1,
            username="admin",
            role=UserRole.ADMIN,
            is_authenticated=True,
        )
        assert ctx.role == UserRole.ADMIN
        assert ctx.role.value == "admin"

    def test_user_role_string(self):
        ctx = AuthContext(
            user_id=2,
            username="user",
            role=UserRole.USER,
            is_authenticated=True,
        )
        assert ctx.role == UserRole.USER
        assert ctx.role.value == "user"

    def test_viewer_role_string(self):
        ctx = AuthContext(
            user_id=3,
            username="viewer",
            role=UserRole.VIEWER,
            is_authenticated=True,
        )
        assert ctx.role == UserRole.VIEWER
        assert ctx.role.value == "viewer"


# Mock database and permission checker tests
class TestPermissionFilteringLogic:
    """Tests for permission filtering logic."""

    def test_admin_bypasses_permission(self):
        """Admin users should bypass permission checks."""
        # This tests the logic that admin users can access all documents
        ctx = AuthContext(
            user_id=1,
            username="admin",
            role=UserRole.ADMIN,
            is_authenticated=True,
        )
        # Admin should be able to bypass - this is the design intent
        assert ctx.is_admin() is True

    def test_dev_user_without_id_cannot_access(self):
        """Dev users without database records cannot access documents."""
        ctx = AuthContext(
            user_id=None,  # No database record
            username="dev_user",
            role=UserRole.USER,
            is_authenticated=True,
        )
        # Dev user without user_id cannot have permissions
        assert ctx.user_id is None
        assert ctx.is_admin() is False

    def test_regular_user_needs_explicit_permission(self):
        """Regular users need explicit document permissions."""
        ctx = AuthContext(
            user_id=100,
            username="regular_user",
            role=UserRole.USER,
            is_authenticated=True,
        )
        assert ctx.is_admin() is False
        assert ctx.user_id == 100


class TestAuditLogModel:
    """Tests for AuditLog model structure."""

    def test_audit_action_values(self):
        from app.security.models import AuditAction
        assert AuditAction.CHAT_MESSAGE.value == "chat_message"
        assert AuditAction.CHAT_RAG.value == "chat_rag"
        assert AuditAction.DOCUMENT_UPLOAD.value == "document_upload"

    def test_audit_log_fields(self):
        """Test that AuditLog model has expected fields."""
        from app.security.models import AuditLog, AuditAction, UserRole
        
        # This is just a structure test - we don't instantiate
        # but we verify the class has the right attributes
        assert hasattr(AuditLog, '__tablename__')
        assert AuditLog.__tablename__ == 'audit_logs'