"""
Permissions Layer

Provides document-level permission checking and access control.
"""

from typing import Optional
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.security.auth import AuthContext
from app.security.models import DocumentPermission, UserRole
from app.db.session import SessionLocal


@dataclass
class PermissionResult:
    """Result of a permission check."""
    granted: bool
    reason: str
    document_id: Optional[int] = None


class PermissionChecker:
    """
    Checks and enforces document-level permissions.
    
    Permission hierarchy:
    - Admin: Can access all documents (bypasses permission checks)
    - User: Can only access documents they have explicit permission for
    - Viewer: Read-only access to permitted documents
    """
    
    def __init__(self, db: Session):
        self.db = db
    
    def _check_admin_bypass(self, auth: AuthContext) -> bool:
        """Check if user is admin and can bypass permission checks."""
        return auth.is_admin()
    
    def _check_document_permission(
        self,
        user_id: int,
        document_id: int,
        require_write: bool = False,
    ) -> PermissionResult:
        """
        Check if user has permission to access a document.
        
        Args:
            user_id: The user's database ID
            document_id: The document to check access for
            require_write: If True, require write permission
            
        Returns:
            PermissionResult with granted=True if access allowed
        """
        permission = self.db.query(DocumentPermission).filter(
            DocumentPermission.user_id == user_id,
            DocumentPermission.document_id == document_id,
        ).first()
        
        if not permission:
            return PermissionResult(
                granted=False,
                reason=f"No permission record for user {user_id} on document {document_id}",
                document_id=document_id,
            )
        
        if require_write and not permission.can_write:
            return PermissionResult(
                granted=False,
                reason="User does not have write permission",
                document_id=document_id,
            )
        
        if not permission.can_read:
            return PermissionResult(
                granted=False,
                reason="User does not have read permission",
                document_id=document_id,
            )
        
        return PermissionResult(
            granted=True,
            reason="Permission granted",
            document_id=document_id,
        )
    
    def check_access(
        self,
        auth: AuthContext,
        document_id: int,
        require_write: bool = False,
    ) -> PermissionResult:
        """
        Check if user can access a specific document.
        
        Admin users bypass all permission checks.
        """
        # Admin bypass
        if self._check_admin_bypass(auth):
            return PermissionResult(
                granted=True,
                reason="Admin bypass",
                document_id=document_id,
            )
        
        # Dev users without user_id can't have permissions
        if auth.user_id is None:
            return PermissionResult(
                granted=False,
                reason="Dev user without database record cannot access documents",
                document_id=document_id,
            )
        
        return self._check_document_permission(auth.user_id, document_id, require_write)
    
    def filter_accessible_documents(
        self,
        auth: AuthContext,
        document_ids: list[int],
    ) -> list[int]:
        """
        Filter a list of document IDs to only those the user can access.
        
        Args:
            auth: Authentication context
            document_ids: List of document IDs to filter
            
        Returns:
            List of document IDs the user has access to
        """
        # Admin can access all
        if self._check_admin_bypass(auth):
            return list(document_ids)
        
        # Dev users without user_id can't access any documents
        if auth.user_id is None:
            return []
        
        # Query permissions for user's documents
        permissions = self.db.query(DocumentPermission).filter(
            DocumentPermission.user_id == auth.user_id,
            DocumentPermission.document_id.in_(document_ids),
            DocumentPermission.can_read == True,
        ).all()
        
        return [p.document_id for p in permissions]
    
    def check_chunk_access(
        self,
        auth: AuthContext,
        chunk_document_ids: list[int],
    ) -> list[int]:
        """
        Given a list of chunk document IDs, return only those accessible to the user.
        
        This is used to filter RAG retrieval results before presenting to the user.
        
        Args:
            auth: Authentication context
            chunk_document_ids: List of document IDs from retrieved chunks
            
        Returns:
            List of accessible document IDs
        """
        return self.filter_accessible_documents(auth, chunk_document_ids)


def get_permission_checker() -> PermissionChecker:
    """Get a permission checker instance."""
    db = SessionLocal()
    return PermissionChecker(db)


def check_document_access(
    auth: AuthContext,
    document_id: int,
    require_write: bool = False,
) -> PermissionResult:
    """
    Convenience function to check document access.
    
    Creates its own database session.
    """
    checker = get_permission_checker()
    try:
        return checker.check_access(auth, document_id, require_write)
    finally:
        checker.db.close()


def filter_documents_by_permission(
    auth: AuthContext,
    document_ids: list[int],
) -> list[int]:
    """
    Convenience function to filter document IDs by permission.
    
    Creates its own database session.
    """
    checker = get_permission_checker()
    try:
        return checker.filter_accessible_documents(auth, document_ids)
    finally:
        checker.db.close()