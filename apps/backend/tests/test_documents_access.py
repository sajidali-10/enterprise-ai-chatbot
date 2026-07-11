"""
Tests for Phase 13 — Document Access Control.

Covers:
- GET /api/documents: list filtering per role (admin=all, user=own+shared+global, viewer=global+shared+role)
- Unauthenticated list returns 401
- POST /api/documents/upload: ownership + default visibility per role
- DELETE /api/documents/{id}: per-doc manage perms
- POST /api/documents/{id}/index: per-doc manage perms
- GET /api/admin/documents, GET/PATCH /api/admin/documents/{id}/access: admin-only RBAC
"""

import io
import pytest
from fastapi.testclient import TestClient

from app.core.config import settings
from app.security.models import User, UserRole, DocumentPermission, DocumentRoleAccess
from app.security.password import hash_password
from app.models.document import Document, DocumentVersion, DocumentChunk
from sqlalchemy import text


settings.JWT_SECRET_KEY = "test-secret-key-for-jwt"
settings.JWT_ALGORITHM = "HS256"
settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES = 60


def _make_user(db, username, email, password, role=UserRole.user, is_active=True):
    u = User(
        username=username, email=email,
        hashed_password=hash_password(password),
        role=role, is_active=is_active,
    )
    db.add(u); db.commit(); db.refresh(u)
    return u


def _login(client, username_or_email, password):
    res = client.post("/api/auth/login", json={
        "username_or_email": username_or_email, "password": password,
    })
    assert res.status_code == 200, res.text
    return res.json()["access_token"]


def _make_doc(db, name, visibility="global", owner_user_id=None, status="indexed"):
    """Insert a Document row directly. Indexing/Qdrant not needed for access-control tests."""
    d = Document(
        filename=f"{name}.bin",
        original_name=f"{name}.bin",
        mime_type="text/plain",
        size_bytes=10,
        status=status,
        visibility=visibility,
        owner_user_id=owner_user_id,
    )
    db.add(d); db.commit(); db.refresh(d)
    return d


def _cleanup_doc(db, doc_id):
    """Cascade-delete a doc and its dependents (used by some tests)."""
    db.query(DocumentChunk).filter(DocumentChunk.document_id == doc_id).delete()
    db.query(DocumentVersion).filter(DocumentVersion.document_id == doc_id).delete()
    db.query(DocumentPermission).filter(DocumentPermission.document_id == doc_id).delete()
    db.query(DocumentRoleAccess).filter(DocumentRoleAccess.document_id == doc_id).delete()
    d = db.query(Document).filter(Document.id == doc_id).first()
    if d:
        db.delete(d); db.commit()


@pytest.fixture
def admin_user(db_session):
    return _make_user(db_session, "phase13admin", "p13admin@test.com", "adminpass", role=UserRole.admin)


@pytest.fixture
def regular_user(db_session):
    return _make_user(db_session, "phase13user", "p13user@test.com", "userpass", role=UserRole.user)


@pytest.fixture
def viewer_user(db_session):
    return _make_user(db_session, "phase13viewer", "p13viewer@test.com", "viewerpass", role=UserRole.viewer)


@pytest.fixture
def admin_client(client, admin_user):
    token = _login(client, "phase13admin", "adminpass")
    client.headers["Authorization"] = f"Bearer {token}"
    return client


@pytest.fixture
def user_client(client, regular_user):
    token = _login(client, "phase13user", "userpass")
    client.headers["Authorization"] = f"Bearer {token}"
    return client


@pytest.fixture
def viewer_client(client, viewer_user):
    token = _login(client, "phase13viewer", "viewerpass")
    client.headers["Authorization"] = f"Bearer {token}"
    return client


# ---------------------------------------------------------------------------
# GET /api/documents — list filtering per role
# ---------------------------------------------------------------------------

class TestListDocumentsFiltering:
    """Each role should see only documents it has access to."""

    def test_unauthenticated_returns_401(self, client):
        # Strip any auth header set by other fixtures
        client.headers.pop("Authorization", None)
        res = client.get("/api/documents")
        assert res.status_code in (401, 403), f"got {res.status_code}: {res.text}"

    def test_admin_sees_all_documents(self, admin_client, db_session, regular_user, viewer_user):
        # Seed three docs: one private owned by regular_user, one private owned by viewer_user, one global
        global_doc = _make_doc(db_session, "list_global")
        private_user = _make_doc(db_session, "list_private_user", visibility="private", owner_user_id=regular_user.id)
        private_viewer = _make_doc(db_session, "list_private_viewer", visibility="private", owner_user_id=viewer_user.id)
        try:
            res = admin_client.get("/api/documents")
            assert res.status_code == 200
            ids = {d["id"] for d in res.json()}
            assert global_doc.id in ids
            assert private_user.id in ids
            assert private_viewer.id in ids
        finally:
            _cleanup_doc(db_session, global_doc.id)
            _cleanup_doc(db_session, private_user.id)
            _cleanup_doc(db_session, private_viewer.id)

    def test_regular_user_sees_only_global_or_own_or_shared(self, user_client, db_session, regular_user, viewer_user):
        # Seed: global, own private, other user's private, shared with user, role-shared with role=user
        global_doc = _make_doc(db_session, "filter_global")
        own_private = _make_doc(db_session, "filter_own_private", visibility="private", owner_user_id=regular_user.id)
        other_private = _make_doc(db_session, "filter_other_private", visibility="private", owner_user_id=viewer_user.id)
        shared_doc = _make_doc(db_session, "filter_shared", visibility="shared", owner_user_id=viewer_user.id)
        db_session.add(DocumentPermission(
            user_id=regular_user.id, document_id=shared_doc.id,
            can_read=True, can_write=False, access_level="view",
        ))
        role_shared = _make_doc(db_session, "filter_role_shared", visibility="shared", owner_user_id=viewer_user.id)
        db_session.add(DocumentRoleAccess(
            document_id=role_shared.id, role="user", access_level="view",
        ))
        db_session.commit()
        try:
            res = user_client.get("/api/documents")
            assert res.status_code == 200
            ids = {d["id"] for d in res.json()}
            assert global_doc.id in ids, "user should see global"
            assert own_private.id in ids, "user should see own private"
            assert other_private.id not in ids, "user should NOT see other user's private"
            assert shared_doc.id in ids, "user should see user-shared"
            assert role_shared.id in ids, "user should see role-shared (user role)"
        finally:
            _cleanup_doc(db_session, global_doc.id)
            _cleanup_doc(db_session, own_private.id)
            _cleanup_doc(db_session, other_private.id)
            _cleanup_doc(db_session, shared_doc.id)
            _cleanup_doc(db_session, role_shared.id)

    def test_viewer_sees_global_and_shared_only(self, viewer_client, db_session, regular_user, viewer_user):
        global_doc = _make_doc(db_session, "view_global")
        user_private = _make_doc(db_session, "view_user_private", visibility="private", owner_user_id=regular_user.id)
        viewer_private = _make_doc(db_session, "view_viewer_private", visibility="private", owner_user_id=viewer_user.id)
        user_shared = _make_doc(db_session, "view_user_shared", visibility="shared", owner_user_id=regular_user.id)
        db_session.add(DocumentPermission(
            user_id=viewer_user.id, document_id=user_shared.id,
            can_read=True, can_write=False, access_level="view",
        ))
        role_shared = _make_doc(db_session, "view_role_shared", visibility="shared", owner_user_id=regular_user.id)
        db_session.add(DocumentRoleAccess(
            document_id=role_shared.id, role="viewer", access_level="view",
        ))
        db_session.commit()
        try:
            res = viewer_client.get("/api/documents")
            assert res.status_code == 200
            ids = {d["id"] for d in res.json()}
            assert global_doc.id in ids, "viewer should see global"
            assert user_private.id not in ids, "viewer should NOT see user's private"
            assert viewer_private.id in ids, "viewer should see own private"
            assert user_shared.id in ids, "viewer should see user-shared"
            assert role_shared.id in ids, "viewer should see role-shared (viewer role)"
        finally:
            _cleanup_doc(db_session, global_doc.id)
            _cleanup_doc(db_session, user_private.id)
            _cleanup_doc(db_session, viewer_private.id)
            _cleanup_doc(db_session, user_shared.id)
            _cleanup_doc(db_session, role_shared.id)

    def test_payload_includes_visibility_and_owner_username(self, admin_client, db_session, regular_user):
        doc = _make_doc(db_session, "fields_doc", visibility="private", owner_user_id=regular_user.id)
        try:
            res = admin_client.get("/api/documents")
            assert res.status_code == 200
            item = next(d for d in res.json() if d["id"] == doc.id)
            assert item["visibility"] == "private"
            assert item["owner_user_id"] == regular_user.id
            assert item["owner_username"] == regular_user.username
        finally:
            _cleanup_doc(db_session, doc.id)


# ---------------------------------------------------------------------------
# POST /api/documents/upload — ownership + default visibility per role
# ---------------------------------------------------------------------------

class TestUploadDocumentOwnership:
    """Upload sets owner_user_id and applies role-aware visibility default."""

    def test_admin_upload_defaults_to_global(self, admin_client, db_session):
        content = b"phase13 admin upload test content"
        res = admin_client.post(
            "/api/documents/upload",
            files={"file": ("upload_admin.txt", io.BytesIO(content), "text/plain")},
        )
        # Test app may fail on Qdrant indexing (no real provider in test) but should still
        # either succeed (200) or fail after the visibility/owner fields are committed.
        # If it succeeds, owner/visibility must be set; if it fails, we inspect the DB directly.
        doc_id = None
        if res.status_code == 200:
            body = res.json()
            doc_id = body["id"]
            assert body["visibility"] == "global"
            assert body["owner_user_id"] is not None
        else:
            # Find the most recently created doc
            d = db_session.query(Document).order_by(Document.id.desc()).first()
            assert d is not None
            doc_id = d.id
            assert d.visibility == "global"
            assert d.owner_user_id is not None
        _cleanup_doc(db_session, doc_id)

    def test_regular_user_upload_defaults_to_private(self, user_client, db_session):
        content = b"phase13 user upload test content"
        res = user_client.post(
            "/api/documents/upload",
            files={"file": ("upload_user.txt", io.BytesIO(content), "text/plain")},
        )
        doc_id = None
        if res.status_code == 200:
            body = res.json()
            doc_id = body["id"]
            assert body["visibility"] == "private", f"user upload must default to private; got {body}"
        else:
            d = db_session.query(Document).order_by(Document.id.desc()).first()
            assert d is not None
            doc_id = d.id
            assert d.visibility == "private", f"user upload must default to private; got {d.visibility}"
        _cleanup_doc(db_session, doc_id)

    def test_user_cannot_override_visibility_to_global(self, user_client, db_session):
        content = b"phase13 user override test"
        res = user_client.post(
            "/api/documents/upload",
            files={"file": ("upload_override.txt", io.BytesIO(content), "text/plain")},
            data={"visibility": "global"},
        )
        doc_id = None
        if res.status_code == 200:
            body = res.json()
            doc_id = body["id"]
            assert body["visibility"] == "private", "override must be blocked for regular users"
        else:
            d = db_session.query(Document).order_by(Document.id.desc()).first()
            assert d is not None
            doc_id = d.id
            assert d.visibility == "private", "override must be blocked for regular users"
        _cleanup_doc(db_session, doc_id)

    def test_viewer_upload_is_rejected_with_403(self, viewer_client):
        content = b"phase13 viewer upload test"
        res = viewer_client.post(
            "/api/documents/upload",
            files={"file": ("upload_viewer.txt", io.BytesIO(content), "text/plain")},
        )
        assert res.status_code == 403, f"viewer upload must be blocked; got {res.status_code}: {res.text}"


# ---------------------------------------------------------------------------
# DELETE /api/documents/{id} — per-doc manage perms
# ---------------------------------------------------------------------------

class TestDeleteDocumentPerms:
    """Delete enforces can_manage_document; role-level perms still apply."""

    def test_admin_can_delete_any_doc(self, admin_client, db_session, regular_user):
        doc = _make_doc(db_session, "del_admin", visibility="private", owner_user_id=regular_user.id)
        res = admin_client.delete(f"/api/documents/{doc.id}")
        assert res.status_code == 200
        assert db_session.query(Document).filter(Document.id == doc.id).first() is None

    def test_other_non_owner_cannot_delete_private_doc(self, user_client, db_session, viewer_user):
        # viewer_user owns a private doc; user_client is a different user trying to delete it
        doc = _make_doc(db_session, "del_other", visibility="private", owner_user_id=viewer_user.id)
        try:
            res = user_client.delete(f"/api/documents/{doc.id}")
            # user_client lacks can_delete_documents at role level → 403 (not per-doc)
            assert res.status_code == 403
            # The doc should still exist
            assert db_session.query(Document).filter(Document.id == doc.id).first() is not None
        finally:
            _cleanup_doc(db_session, doc.id)

    def test_viewer_cannot_delete_any_doc(self, viewer_client, db_session, admin_user):
        doc = _make_doc(db_session, "del_viewer", owner_user_id=admin_user.id)
        try:
            res = viewer_client.delete(f"/api/documents/{doc.id}")
            assert res.status_code == 403
        finally:
            _cleanup_doc(db_session, doc.id)


# ---------------------------------------------------------------------------
# POST /api/documents/{id}/index — per-doc manage perms
# ---------------------------------------------------------------------------

class TestReindexDocumentPerms:
    """Per-doc reindex endpoint enforces can_manage_document."""

    def test_admin_can_reindex_any_doc(self, admin_client, db_session, regular_user):
        doc = _make_doc(db_session, "idx_admin", visibility="private", owner_user_id=regular_user.id, status="extracted")
        try:
            res = admin_client.post(f"/api/documents/{doc.id}/index")
            # Either succeeds (200) or fails with a downstream Qdrant error — but never 403
            assert res.status_code != 403, f"admin must not be blocked; got {res.status_code}: {res.text}"
        finally:
            _cleanup_doc(db_session, doc.id)

    def test_non_owner_cannot_reindex_other_user_private_doc(self, user_client, db_session, viewer_user):
        doc = _make_doc(db_session, "idx_other", visibility="private", owner_user_id=viewer_user.id, status="extracted")
        try:
            res = user_client.post(f"/api/documents/{doc.id}/index")
            # user lacks can_reindex_documents at role level → 403
            assert res.status_code == 403
        finally:
            _cleanup_doc(db_session, doc.id)

    def test_viewer_cannot_reindex_any_doc(self, viewer_client, db_session, admin_user):
        doc = _make_doc(db_session, "idx_viewer", owner_user_id=admin_user.id, status="extracted")
        try:
            res = viewer_client.post(f"/api/documents/{doc.id}/index")
            assert res.status_code == 403
        finally:
            _cleanup_doc(db_session, doc.id)


# ---------------------------------------------------------------------------
# /api/admin/documents — admin-only RBAC
# ---------------------------------------------------------------------------

class TestAdminDocumentsRBAC:
    """All three admin endpoints require can_manage_users."""

    def test_unauthenticated_get_admin_documents_returns_401(self, client):
        client.headers.pop("Authorization", None)
        res = client.get("/api/admin/documents")
        assert res.status_code in (401, 403)

    def test_user_get_admin_documents_returns_403(self, user_client):
        res = user_client.get("/api/admin/documents")
        assert res.status_code == 403

    def test_viewer_get_admin_documents_returns_403(self, viewer_client):
        res = viewer_client.get("/api/admin/documents")
        assert res.status_code == 403

    def test_admin_can_list_all_documents(self, admin_client, db_session, regular_user):
        own = _make_doc(db_session, "adm_own", visibility="private", owner_user_id=regular_user.id)
        try:
            res = admin_client.get("/api/admin/documents")
            assert res.status_code == 200
            ids = {d["id"] for d in res.json()}
            assert own.id in ids
        finally:
            _cleanup_doc(db_session, own.id)

    def test_unauthenticated_get_access_returns_401(self, client, db_session):
        client.headers.pop("Authorization", None)
        doc = _make_doc(db_session, "adm_access_unauth")
        try:
            res = client.get(f"/api/admin/documents/{doc.id}/access")
            assert res.status_code in (401, 403)
        finally:
            _cleanup_doc(db_session, doc.id)

    def test_user_get_access_returns_403(self, user_client, db_session, regular_user):
        doc = _make_doc(db_session, "adm_access_user", owner_user_id=regular_user.id)
        try:
            res = user_client.get(f"/api/admin/documents/{doc.id}/access")
            assert res.status_code == 403
        finally:
            _cleanup_doc(db_session, doc.id)

    def test_admin_get_access_returns_200_with_summary(self, admin_client, db_session, regular_user):
        doc = _make_doc(db_session, "adm_access_admin", visibility="shared", owner_user_id=regular_user.id)
        db_session.add(DocumentPermission(
            user_id=regular_user.id, document_id=doc.id,
            can_read=True, can_write=False, access_level="manage",
        ))
        db_session.add(DocumentRoleAccess(
            document_id=doc.id, role="user", access_level="view",
        ))
        db_session.commit()
        try:
            res = admin_client.get(f"/api/admin/documents/{doc.id}/access")
            assert res.status_code == 200
            body = res.json()
            assert body["visibility"] == "shared"
            assert body["owner_user_id"] == regular_user.id
            assert body["owner_username"] == regular_user.username
            assert regular_user.id in body["shared_user_ids"]
            assert any(r["role"] == "user" for r in body["shared_role_access"])
        finally:
            _cleanup_doc(db_session, doc.id)

    def test_user_patch_access_returns_403(self, user_client, db_session, admin_user):
        doc = _make_doc(db_session, "adm_patch_user", owner_user_id=admin_user.id)
        try:
            res = user_client.patch(
                f"/api/admin/documents/{doc.id}/access",
                json={"visibility": "global"},
            )
            assert res.status_code == 403
        finally:
            _cleanup_doc(db_session, doc.id)

    def test_admin_patch_access_changes_visibility_and_shares(self, admin_client, db_session, regular_user, viewer_user):
        doc = _make_doc(db_session, "adm_patch_admin", visibility="private", owner_user_id=regular_user.id)
        try:
            res = admin_client.patch(
                f"/api/admin/documents/{doc.id}/access",
                json={
                    "visibility": "shared",
                    "shared_user_ids": [viewer_user.id],
                    "shared_roles": ["viewer"],
                    "access_level": "view",
                },
            )
            assert res.status_code == 200
            body = res.json()
            assert body["visibility"] == "shared"
            assert viewer_user.id in body["shared_user_ids"]
            assert any(r["role"] == "viewer" for r in body["shared_role_access"])
            # Verify persistence
            res2 = admin_client.get(f"/api/admin/documents/{doc.id}/access")
            assert res2.status_code == 200
            body2 = res2.json()
            assert body2["visibility"] == "shared"
            assert viewer_user.id in body2["shared_user_ids"]
        finally:
            _cleanup_doc(db_session, doc.id)

    def test_admin_patch_invalid_visibility_returns_400(self, admin_client, db_session, regular_user):
        doc = _make_doc(db_session, "adm_patch_bad_vis", owner_user_id=regular_user.id)
        try:
            res = admin_client.patch(
                f"/api/admin/documents/{doc.id}/access",
                json={"visibility": "public"},
            )
            assert res.status_code == 400
            assert "visibility" in res.json()["detail"].lower()
        finally:
            _cleanup_doc(db_session, doc.id)

    def test_admin_patch_invalid_role_returns_400(self, admin_client, db_session, regular_user):
        doc = _make_doc(db_session, "adm_patch_bad_role", owner_user_id=regular_user.id)
        try:
            res = admin_client.patch(
                f"/api/admin/documents/{doc.id}/access",
                json={"shared_roles": ["superuser"]},
            )
            assert res.status_code == 400
            assert "role" in res.json()["detail"].lower()
        finally:
            _cleanup_doc(db_session, doc.id)

    def test_admin_patch_nonexistent_user_returns_400(self, admin_client, db_session, regular_user):
        doc = _make_doc(db_session, "adm_patch_bad_user", owner_user_id=regular_user.id)
        try:
            res = admin_client.patch(
                f"/api/admin/documents/{doc.id}/access",
                json={"shared_user_ids": [999_999_999]},
            )
            assert res.status_code == 400
            assert "not found" in res.json()["detail"].lower()
        finally:
            _cleanup_doc(db_session, doc.id)

    def test_admin_patch_unknown_doc_returns_404(self, admin_client):
        res = admin_client.patch(
            "/api/admin/documents/9999999/access",
            json={"visibility": "global"},
        )
        assert res.status_code == 404
