"""
Permissions Layer — Phase 13.

Document access control is governed by three orthogonal mechanisms:

1. ``Document.visibility`` (private | shared | global)
   - ``global``  — every authenticated user can view.
   - ``shared``  — only owner, admin, and explicitly shared users/roles can view.
   - ``private`` — only owner and admin can view.

2. ``Document.owner_user_id``
   - Whoever uploaded/owns the document. Owner always has full (manage) access.

3. Explicit share rows (per-document, additive):
   - ``DocumentPermission`` rows (one per user) carry ``access_level`` ∈ {view, manage}.
   - ``DocumentRoleAccess`` rows (one per role) carry ``access_level`` ∈ {view, manage}.

Resolution rules:
- Admin role bypasses all checks (admin sees everything, can manage everything).
- Dev users (X-Dev-User header in DEV_AUTH_ENABLED mode) bypass all checks.
- Otherwise, a user has *view* access to a doc if ANY of:
    * doc.visibility == 'global'
    * doc.owner_user_id == auth.user_id
    * DocumentPermission row exists for (doc.id, auth.user_id) — any access_level
    * DocumentRoleAccess row exists for (doc.id, auth.role) — any access_level
- A user has *manage* access if ANY of:
    * admin/dev bypass
    * doc.owner_user_id == auth.user_id
    * DocumentPermission row with access_level == 'manage'
    * DocumentRoleAccess row with access_level == 'manage'

The existing ``filter_documents_by_permission`` / ``check_document_access`` /
``check_access`` entry points remain in place so that call sites do not need to
change — they are now thin wrappers around the new resolver.
"""

from typing import Optional
from dataclasses import dataclass

from sqlalchemy.orm import Session
from sqlalchemy import or_, and_

from app.security.auth import AuthContext
from app.security.models import DocumentPermission, DocumentRoleAccess, UserRole
from app.models.document import Document
from app.db.session import SessionLocal
from app.core.config import settings


# Dev user detection: user_id=None means dev user (from X-Dev-User header).
# These users bypass permission checks in development mode.
def is_dev_user(auth: AuthContext) -> bool:
    """Check if this is a dev user (authenticated via X-Dev-User header)."""
    return (
        auth.is_authenticated
        and auth.user_id is None
        and auth.username != "anonymous"
    )


@dataclass
class PermissionResult:
    """Result of a permission check."""
    granted: bool
    reason: str
    document_id: Optional[int] = None


# ---------------------------------------------------------------------------
# Public helpers (new in Phase 13)
# ---------------------------------------------------------------------------

def _is_admin(auth: AuthContext) -> bool:
    return bool(getattr(auth, "is_admin", lambda: False)())


def _dev_bypass_active(auth: AuthContext) -> bool:
    if not getattr(settings, "DEV_AUTH_ENABLED", False):
        return False
    return is_dev_user(auth)


def can_access_document(
    auth: AuthContext,
    document_id: int,
    db: Optional[Session] = None,
) -> bool:
    """
    Return True iff ``auth`` can view ``document_id``.

    Resolution order (admin and dev bypass always grant):
      1. visibility == 'global'
      2. owner_user_id == auth.user_id
      3. document_permissions row for (doc_id, auth.user_id) with any access_level
      4. document_role_access row for (doc_id, auth.role) with any access_level

    A missing/unauthenticated auth returns False.
    """
    if auth is None or not getattr(auth, "is_authenticated", False):
        return False

    if _is_admin(auth) or _dev_bypass_active(auth):
        return True

    if getattr(auth, "user_id", None) is None:
        return False

    owns_session = db is None
    session = db or SessionLocal()
    try:
        return _has_view_access(session, document_id, auth.user_id, auth.role)
    finally:
        if owns_session:
            session.close()


def can_manage_document(
    auth: AuthContext,
    document_id: int,
    db: Optional[Session] = None,
) -> bool:
    """
    Return True iff ``auth`` can manage (delete, reindex, change visibility) ``document_id``.

    Resolution order (admin and dev bypass always grant):
      1. admin or dev bypass
      2. owner_user_id == auth.user_id
      3. document_permissions row with access_level == 'manage' for (doc_id, auth.user_id)
      4. document_role_access row with access_level == 'manage' for (doc_id, auth.role)
    """
    if auth is None or not getattr(auth, "is_authenticated", False):
        return False

    if _is_admin(auth) or _dev_bypass_active(auth):
        return True

    if getattr(auth, "user_id", None) is None:
        return False

    owns_session = db is None
    session = db or SessionLocal()
    try:
        return _has_manage_access(session, document_id, auth.user_id, auth.role)
    finally:
        if owns_session:
            session.close()


def get_accessible_document_ids(
    auth: AuthContext,
    db: Optional[Session] = None,
) -> list[int]:
    """
    Return the full set of document IDs visible to ``auth``.

    Admin and dev users get every doc ID; everyone else gets the union of:
      - docs with visibility='global'
      - docs owned by the user
      - docs shared directly with the user (DocumentPermission row)
      - docs shared with the user's role (DocumentRoleAccess row)
    """
    if auth is None or not getattr(auth, "is_authenticated", False):
        return []

    owns_session = db is None
    session = db or SessionLocal()
    try:
        if _is_admin(auth) or _dev_bypass_active(auth):
            return [d.id for d in session.query(Document.id).all()]

        user_id = getattr(auth, "user_id", None)
        role = getattr(auth, "role", None)
        if user_id is None:
            return []

        return _query_accessible_doc_ids(session, user_id, role)
    finally:
        if owns_session:
            session.close()


# ---------------------------------------------------------------------------
# Internal resolver helpers (DB-bound)
# ---------------------------------------------------------------------------

def _has_view_access(session: Session, document_id: int, user_id: int, role) -> bool:
    """Check view access for a single document."""
    # Single document lookup with all four conditions in one query.
    doc = session.query(Document).filter(Document.id == document_id).first()
    if doc is None:
        return False

    if doc.visibility == "global":
        return True
    if doc.owner_user_id == user_id:
        return True

    user_share = session.query(DocumentPermission.id).filter(
        DocumentPermission.user_id == user_id,
        DocumentPermission.document_id == document_id,
    ).first()
    if user_share is not None:
        return True

    role_value = role.value if hasattr(role, "value") else str(role) if role else None
    if role_value:
        role_share = session.query(DocumentRoleAccess.id).filter(
            DocumentRoleAccess.document_id == document_id,
            DocumentRoleAccess.role == role_value,
        ).first()
        if role_share is not None:
            return True

    return False


def _has_manage_access(session: Session, document_id: int, user_id: int, role) -> bool:
    """Check manage access for a single document."""
    doc = session.query(Document).filter(Document.id == document_id).first()
    if doc is None:
        return False

    if doc.owner_user_id == user_id:
        return True

    manage_user_share = session.query(DocumentPermission.id).filter(
        DocumentPermission.user_id == user_id,
        DocumentPermission.document_id == document_id,
        DocumentPermission.access_level == "manage",
    ).first()
    if manage_user_share is not None:
        return True

    role_value = role.value if hasattr(role, "value") else str(role) if role else None
    if role_value:
        manage_role_share = session.query(DocumentRoleAccess.id).filter(
            DocumentRoleAccess.document_id == document_id,
            DocumentRoleAccess.role == role_value,
            DocumentRoleAccess.access_level == "manage",
        ).first()
        if manage_role_share is not None:
            return True

    return False


def _query_accessible_doc_ids(session: Session, user_id: int, role) -> list[int]:
    """Return all doc IDs visible to the user (per the four-condition union)."""
    role_value = role.value if hasattr(role, "value") else str(role) if role else None

    # Subquery: docs the user is explicitly shared with via DocumentPermission.
    user_shared = (
        session.query(DocumentPermission.document_id)
        .filter(DocumentPermission.user_id == user_id)
        .subquery()
    )

    # Subquery: docs the user's role is explicitly shared with via DocumentRoleAccess.
    if role_value:
        role_shared = (
            session.query(DocumentRoleAccess.document_id)
            .filter(DocumentRoleAccess.role == role_value)
            .subquery()
        )
    else:
        role_shared = None

    query = session.query(Document.id).filter(
        or_(
            Document.visibility == "global",
            Document.owner_user_id == user_id,
            Document.id.in_(session.query(user_shared.c.document_id)),
            Document.id.in_(session.query(role_shared.c.document_id)) if role_shared is not None else False,
        )
    )

    return [row[0] for row in query.all()]


# ---------------------------------------------------------------------------
# Existing public API (preserved for call-site compatibility)
# ---------------------------------------------------------------------------

class PermissionChecker:
    """
    Backwards-compatible wrapper around the Phase 13 resolver.

    The old PermissionChecker.checked only DocumentPermission rows; the new
    resolver uses visibility + ownership + DocumentPermission + DocumentRoleAccess.
    """

    def __init__(self, db: Session):
        self.db = db

    def _check_admin_bypass(self, auth: AuthContext) -> bool:
        return _is_admin(auth)

    def _check_dev_bypass(self, auth: AuthContext) -> bool:
        return _dev_bypass_active(auth)

    def _check_document_permission(
        self,
        user_id: int,
        document_id: int,
        require_write: bool = False,
    ) -> PermissionResult:
        """Legacy single-doc check via the new resolver."""
        granted = can_access_document(
            # Build a minimal AuthContext-like object that has the methods we need.
            _LegacyAuthShim(user_id=user_id),
            document_id,
            db=self.db,
        )
        if not granted:
            return PermissionResult(
                granted=False,
                reason=f"No view access for user {user_id} on document {document_id}",
                document_id=document_id,
            )

        if require_write and not can_manage_document(
            _LegacyAuthShim(user_id=user_id),
            document_id,
            db=self.db,
        ):
            return PermissionResult(
                granted=False,
                reason="User does not have manage permission",
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
        """Check single-document access for the given auth context."""
        if _is_admin(auth) or _dev_bypass_active(auth):
            return PermissionResult(granted=True, reason="Admin/dev bypass", document_id=document_id)
        if not getattr(auth, "is_authenticated", False):
            return PermissionResult(granted=False, reason="Not authenticated", document_id=document_id)
        if getattr(auth, "user_id", None) is None:
            return PermissionResult(granted=False, reason="No user ID", document_id=document_id)

        return self._check_document_permission(auth.user_id, document_id, require_write)

    def filter_accessible_documents(
        self,
        auth: AuthContext,
        document_ids: list[int],
    ) -> list[int]:
        """Filter the given document IDs to only those visible to auth."""
        if not document_ids:
            return []
        if _is_admin(auth) or _dev_bypass_active(auth):
            return list(document_ids)
        if getattr(auth, "user_id", None) is None:
            return []

        user_id = auth.user_id
        role = getattr(auth, "role", None)

        # Bulk resolve: query the four conditions once, then intersect with the candidate set.
        candidates = set(document_ids)
        accessible = set()

        # Global docs are visible to everyone authenticated.
        global_ids = {
            row[0]
            for row in self.db.query(Document.id)
            .filter(Document.id.in_(document_ids), Document.visibility == "global")
            .all()
        }
        accessible.update(global_ids)

        # Owned docs.
        owned_ids = {
            row[0]
            for row in self.db.query(Document.id)
            .filter(Document.id.in_(document_ids), Document.owner_user_id == user_id)
            .all()
        }
        accessible.update(owned_ids)

        # User-shared docs.
        shared_user_ids = {
            row[0]
            for row in self.db.query(DocumentPermission.document_id)
            .filter(
                DocumentPermission.user_id == user_id,
                DocumentPermission.document_id.in_(document_ids),
            )
            .all()
        }
        accessible.update(shared_user_ids)

        # Role-shared docs.
        role_value = role.value if hasattr(role, "value") else str(role) if role else None
        if role_value:
            shared_role_ids = {
                row[0]
                for row in self.db.query(DocumentRoleAccess.document_id)
                .filter(
                    DocumentRoleAccess.role == role_value,
                    DocumentRoleAccess.document_id.in_(document_ids),
                )
                .all()
            }
            accessible.update(shared_role_ids)

        # Intersect with the candidate set so we never return docs the caller didn't ask about.
        return sorted(candidates & accessible)

    def check_chunk_access(
        self,
        auth: AuthContext,
        chunk_document_ids: list[int],
    ) -> list[int]:
        return self.filter_accessible_documents(auth, chunk_document_ids)


class _LegacyAuthShim:
    """
    Tiny adapter so PermissionChecker._check_document_permission can call the new
    resolvers without needing a real AuthContext (which is hard to construct in tests).
    """
    def __init__(self, user_id: int):
        self.user_id = user_id
        self.role = None
        self.is_authenticated = True
        self.is_admin = lambda: False

    def __getattr__(self, name):
        # Anything else the resolvers probe for → None / False.
        if name == "username":
            return None
        return None


def get_permission_checker() -> PermissionChecker:
    """Get a permission checker instance."""
    db = SessionLocal()
    return PermissionChecker(db)


def check_document_access(
    auth: AuthContext,
    document_id: int,
    require_write: bool = False,
) -> PermissionResult:
    """Convenience function to check document access. Creates its own DB session."""
    if auth is None:
        return PermissionResult(granted=False, reason="No auth")
    if _is_admin(auth) or _dev_bypass_active(auth):
        return PermissionResult(granted=True, reason="Admin/dev bypass", document_id=document_id)
    if not getattr(auth, "is_authenticated", False):
        return PermissionResult(granted=False, reason="Not authenticated", document_id=document_id)

    granted = can_access_document(auth, document_id)
    if not granted:
        return PermissionResult(
            granted=False,
            reason=f"No view access for user {auth.user_id} on document {document_id}",
            document_id=document_id,
        )
    if require_write and not can_manage_document(auth, document_id):
        return PermissionResult(granted=False, reason="No manage access", document_id=document_id)
    return PermissionResult(granted=True, reason="Permission granted", document_id=document_id)


def filter_documents_by_permission(
    auth: AuthContext,
    document_ids: list[int],
) -> list[int]:
    """Convenience function to filter document IDs by permission. Creates its own DB session."""
    if auth is None or not getattr(auth, "is_authenticated", False):
        return []
    if _is_admin(auth) or _dev_bypass_active(auth):
        return list(document_ids)

    db = SessionLocal()
    try:
        checker = PermissionChecker(db)
        return checker.filter_accessible_documents(auth, document_ids)
    finally:
        db.close()
