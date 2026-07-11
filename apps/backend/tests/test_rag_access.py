"""
Tests for Phase 13 — RAG Retrieval Access Control (leakage prevention).

Covers:
- Resolver functions (can_access_document, can_manage_document,
  filter_documents_by_permission, get_accessible_document_ids) behave
  correctly against the real database across admin/user/viewer contexts.
- retrieve_chunks_with_auth correctly drops unauthorized chunks via the
  post-retrieval filter (defense-in-depth), even when the upstream hybrid
  retrieval returns chunks from documents the caller can't see.
- Citation paths: an unauthorized doc's title, chunk IDs, and doc_id do
  not appear in any post-filter artifact.
- No-access fallback: a user with zero accessible documents and a query
  produces a safe answer (no leakage of any unauthorized metadata).
- Admin/debug metadata does not leak unauthorized doc IDs/titles.
"""

import pytest
from unittest.mock import patch, MagicMock

from app.core.config import settings
from app.security.auth import AuthContext
from app.security.models import (
    User, UserRole, DocumentPermission, DocumentRoleAccess,
)
from app.security.password import hash_password
from app.models.document import Document, DocumentVersion, DocumentChunk
from app.security.permissions import (
    can_access_document,
    can_manage_document,
    filter_documents_by_permission,
    get_accessible_document_ids,
)


settings.JWT_SECRET_KEY = "test-secret-key-for-jwt"
settings.JWT_ALGORITHM = "HS256"
settings.JWT_ACCESS_TOKEN_EXPIRE_MINUTES = 60


def _make_user(db, username, email, role=UserRole.user, is_active=True):
    u = User(
        username=username, email=email,
        hashed_password=hash_password(f"{username}-pw"),
        role=role, is_active=is_active,
    )
    db.add(u); db.commit(); db.refresh(u)
    return u


def _make_doc(db, name, visibility="global", owner_user_id=None):
    d = Document(
        filename=f"{name}.bin",
        original_name=f"{name}.bin",
        mime_type="text/plain",
        size_bytes=10,
        status="indexed",
        visibility=visibility,
        owner_user_id=owner_user_id,
    )
    db.add(d); db.commit(); db.refresh(d)
    return d


def _cleanup_doc(db, doc_id):
    db.query(DocumentChunk).filter(DocumentChunk.document_id == doc_id).delete()
    db.query(DocumentVersion).filter(DocumentVersion.document_id == doc_id).delete()
    db.query(DocumentPermission).filter(DocumentPermission.document_id == doc_id).delete()
    db.query(DocumentRoleAccess).filter(DocumentRoleAccess.document_id == doc_id).delete()
    d = db.query(Document).filter(Document.id == doc_id).first()
    if d:
        db.delete(d); db.commit()


@pytest.fixture
def admin_user(db_session):
    return _make_user(db_session, "rag_admin", "rag_admin@test.com", role=UserRole.admin)


@pytest.fixture
def owner_user(db_session):
    return _make_user(db_session, "rag_owner", "rag_owner@test.com", role=UserRole.user)


@pytest.fixture
def other_user(db_session):
    return _make_user(db_session, "rag_other", "rag_other@test.com", role=UserRole.user)


@pytest.fixture
def viewer_user(db_session):
    return _make_user(db_session, "rag_viewer", "rag_viewer@test.com", role=UserRole.viewer)


def _admin_auth(user):
    return AuthContext(user_id=user.id, username=user.username, role=UserRole.admin, is_authenticated=True, is_external=False)


def _user_auth(user):
    return AuthContext(user_id=user.id, username=user.username, role=UserRole.user, is_authenticated=True, is_external=False)


def _viewer_auth(user):
    return AuthContext(user_id=user.id, username=user.username, role=UserRole.viewer, is_authenticated=True, is_external=False)


# ---------------------------------------------------------------------------
# Resolver-level tests (no RAG chain, just the permission function)
# ---------------------------------------------------------------------------

class TestResolverVisibility:
    """The resolver returns correct access decisions per visibility level."""

    def test_admin_can_access_any_visibility(self, db_session, admin_user, owner_user):
        admin = _admin_auth(admin_user)
        for vis in ("private", "shared", "global"):
            doc = _make_doc(db_session, f"rslv_admin_{vis}", visibility=vis, owner_user_id=owner_user.id)
            try:
                assert can_access_document(admin, doc.id, db=db_session), f"admin must access {vis}"
                assert can_manage_document(admin, doc.id, db=db_session), f"admin must manage {vis}"
            finally:
                _cleanup_doc(db_session, doc.id)

    def test_owner_can_access_own_private_doc(self, db_session, owner_user):
        doc = _make_doc(db_session, "rslv_owner_priv", visibility="private", owner_user_id=owner_user.id)
        try:
            auth = _user_auth(owner_user)
            assert can_access_document(auth, doc.id, db=db_session)
            assert can_manage_document(auth, doc.id, db=db_session)
        finally:
            _cleanup_doc(db_session, doc.id)

    def test_other_user_cannot_access_private_doc(self, db_session, owner_user, other_user):
        doc = _make_doc(db_session, "rslv_other_priv", visibility="private", owner_user_id=owner_user.id)
        try:
            auth = _user_auth(other_user)
            assert not can_access_document(auth, doc.id, db=db_session)
            assert not can_manage_document(auth, doc.id, db=db_session)
        finally:
            _cleanup_doc(db_session, doc.id)

    def test_other_user_cannot_access_private_doc_even_if_shared_with_someone_else(
        self, db_session, owner_user, other_user, viewer_user
    ):
        """A user-share to a third party does NOT leak to a different non-owner."""
        doc = _make_doc(db_session, "rslv_thirdparty", visibility="private", owner_user_id=owner_user.id)
        try:
            db_session.add(DocumentPermission(
                user_id=viewer_user.id, document_id=doc.id,
                can_read=True, can_write=False, access_level="view",
            ))
            db_session.commit()
            auth = _user_auth(other_user)
            assert not can_access_document(auth, doc.id, db=db_session), \
                "private doc shared with viewer_user MUST NOT be visible to other_user"
            # And viewer_user can see it
            v_auth = _viewer_auth(viewer_user)
            assert can_access_document(v_auth, doc.id, db=db_session), \
                "explicit user-share to viewer_user must grant access"
        finally:
            _cleanup_doc(db_session, doc.id)

    def test_viewer_cannot_access_private_doc(self, db_session, owner_user, viewer_user):
        doc = _make_doc(db_session, "rslv_viewer_priv", visibility="private", owner_user_id=owner_user.id)
        try:
            auth = _viewer_auth(viewer_user)
            assert not can_access_document(auth, doc.id, db=db_session)
            assert not can_manage_document(auth, doc.id, db=db_session)
        finally:
            _cleanup_doc(db_session, doc.id)

    def test_viewer_can_access_global_doc(self, db_session, viewer_user):
        doc = _make_doc(db_session, "rslv_viewer_global", visibility="global")
        try:
            auth = _viewer_auth(viewer_user)
            assert can_access_document(auth, doc.id, db=db_session)
            # Viewer cannot manage (no can_delete/reindex at role level anyway, but per-doc also no)
            assert not can_manage_document(auth, doc.id, db=db_session)
        finally:
            _cleanup_doc(db_session, doc.id)

    def test_viewer_can_access_shared_doc_via_role_share(self, db_session, owner_user, viewer_user):
        doc = _make_doc(db_session, "rslv_viewer_role_share", visibility="shared", owner_user_id=owner_user.id)
        db_session.add(DocumentRoleAccess(
            document_id=doc.id, role="viewer", access_level="view",
        ))
        db_session.commit()
        try:
            auth = _viewer_auth(viewer_user)
            assert can_access_document(auth, doc.id, db=db_session)
        finally:
            _cleanup_doc(db_session, doc.id)

    def test_viewer_can_access_shared_doc_via_user_share(self, db_session, owner_user, viewer_user):
        doc = _make_doc(db_session, "rslv_viewer_user_share", visibility="shared", owner_user_id=owner_user.id)
        db_session.add(DocumentPermission(
            user_id=viewer_user.id, document_id=doc.id,
            can_read=True, can_write=False, access_level="view",
        ))
        db_session.commit()
        try:
            auth = _viewer_auth(viewer_user)
            assert can_access_document(auth, doc.id, db=db_session)
        finally:
            _cleanup_doc(db_session, doc.id)


class TestFilterDocumentsByPermission:
    """Bulk filter returns only doc IDs visible to the caller."""

    def test_admin_filter_returns_all(self, db_session, admin_user, owner_user):
        docs = [
            _make_doc(db_session, "flt_a_global"),
            _make_doc(db_session, "flt_b_priv", visibility="private", owner_user_id=owner_user.id),
        ]
        try:
            admin = _admin_auth(admin_user)
            allowed = filter_documents_by_permission(admin, [d.id for d in docs], )
            assert set(allowed) == {d.id for d in docs}
        finally:
            for d in docs:
                _cleanup_doc(db_session, d.id)

    def test_other_user_filter_excludes_private(self, db_session, owner_user, other_user):
        global_doc = _make_doc(db_session, "flt_g")
        private_doc = _make_doc(db_session, "flt_p", visibility="private", owner_user_id=owner_user.id)
        try:
            auth = _user_auth(other_user)
            allowed = filter_documents_by_permission(auth, [global_doc.id, private_doc.id])
            assert global_doc.id in allowed
            assert private_doc.id not in allowed
        finally:
            _cleanup_doc(db_session, global_doc.id)
            _cleanup_doc(db_session, private_doc.id)

    def test_get_accessible_document_ids_includes_role_shared(
        self, db_session, owner_user, viewer_user
    ):
        """get_accessible_document_ids returns the union of all visible docs."""
        global_doc = _make_doc(db_session, "gad_global")
        priv_no_share = _make_doc(db_session, "gad_priv", visibility="private", owner_user_id=owner_user.id)
        role_shared = _make_doc(db_session, "gad_role", visibility="shared", owner_user_id=owner_user.id)
        db_session.add(DocumentRoleAccess(
            document_id=role_shared.id, role="viewer", access_level="view",
        ))
        db_session.commit()
        try:
            v_auth = _viewer_auth(viewer_user)
            ids = set(get_accessible_document_ids(v_auth, db=db_session))
            assert global_doc.id in ids
            assert priv_no_share.id not in ids
            assert role_shared.id in ids
        finally:
            _cleanup_doc(db_session, global_doc.id)
            _cleanup_doc(db_session, priv_no_share.id)
            _cleanup_doc(db_session, role_shared.id)


# ---------------------------------------------------------------------------
# retrieve_chunks_with_auth defense-in-depth
# ---------------------------------------------------------------------------

class TestRetrieveChunksWithAuthFiltering:
    """
    Mock the upstream hybrid retrieval to return chunks from BOTH accessible
    and inaccessible documents. Confirm the resolver-driven filter drops the
    unauthorized ones and the metadata reports the correct counts.
    """

    def test_unauthorized_chunks_dropped_post_retrieval(
        self, db_session, admin_user, owner_user, other_user
    ):
        private_doc = _make_doc(db_session, "rag_priv", visibility="private", owner_user_id=owner_user.id)
        global_doc = _make_doc(db_session, "rag_global")
        try:
            # Simulate upstream hybrid retrieval returning chunks from BOTH docs.
            fake_chunks = [
                {"chunk_id": "c1", "document_id": global_doc.id, "content": "global content A", "source_file_name": "rag_global.bin"},
                {"chunk_id": "c2", "document_id": private_doc.id, "content": "PRIVATE_SECRET_CONTENT", "source_file_name": "rag_priv.bin"},
                {"chunk_id": "c3", "document_id": global_doc.id, "content": "global content B", "source_file_name": "rag_global.bin"},
            ]
            fake_metadata = {"retrieval_status": "ok", "chunks_before_filter": 3}

            # Patch retrieve_chunks_with_settings to return the fake chunks.
            from app.rag import retriever as retriever_mod
            with patch.object(retriever_mod, "retrieve_chunks_with_settings",
                              return_value=(fake_chunks, fake_metadata)):
                other_auth = _user_auth(other_user)
                chunks, metadata = retriever_mod.retrieve_chunks_with_auth(
                    query="any query", auth=other_auth, debug=True,
                )

            # The private doc's chunk MUST NOT appear in the filtered results.
            kept_doc_ids = {c["document_id"] for c in chunks}
            assert private_doc.id not in kept_doc_ids, \
                "Private doc chunk leaked to other_user!"
            assert global_doc.id in kept_doc_ids, \
                "Global doc chunk was wrongly filtered out"
            # And the secret content string MUST NOT appear anywhere in the response.
            joined = str(chunks) + str(metadata)
            assert "PRIVATE_SECRET_CONTENT" not in joined
            assert "rag_priv.bin" not in joined
            # Metadata should report the filter applied.
            assert metadata.get("permission_filtered") is True
            assert metadata.get("filtered_count") == 1, \
                f"expected 1 chunk filtered, got {metadata.get('filtered_count')}"
            assert metadata.get("total_count") == 3
            assert metadata.get("accessible_count") == 2
        finally:
            _cleanup_doc(db_session, private_doc.id)
            _cleanup_doc(db_session, global_doc.id)

    def test_admin_sees_all_chunks(self, db_session, admin_user, owner_user):
        private_doc = _make_doc(db_session, "rag_priv_admin", visibility="private", owner_user_id=owner_user.id)
        global_doc = _make_doc(db_session, "rag_global_admin")
        try:
            fake_chunks = [
                {"chunk_id": "c1", "document_id": global_doc.id, "content": "g", "source_file_name": "g"},
                {"chunk_id": "c2", "document_id": private_doc.id, "content": "p", "source_file_name": "p"},
            ]
            fake_metadata = {}
            from app.rag import retriever as retriever_mod
            with patch.object(retriever_mod, "retrieve_chunks_with_settings",
                              return_value=(fake_chunks, fake_metadata)):
                chunks, metadata = retriever_mod.retrieve_chunks_with_auth(
                    query="any query", auth=_admin_auth(admin_user), debug=True,
                )
            kept_doc_ids = {c["document_id"] for c in chunks}
            assert private_doc.id in kept_doc_ids
            assert global_doc.id in kept_doc_ids
            assert metadata.get("filtered_count") == 0
        finally:
            _cleanup_doc(db_session, private_doc.id)
            _cleanup_doc(db_session, global_doc.id)

    def test_viewer_with_only_global_access_filters_out_private(self, db_session, viewer_user, owner_user):
        private_doc = _make_doc(db_session, "rag_priv_viewer", visibility="private", owner_user_id=owner_user.id)
        global_doc = _make_doc(db_session, "rag_global_viewer")
        try:
            fake_chunks = [
                {"chunk_id": "c1", "document_id": global_doc.id, "content": "g", "source_file_name": "g"},
                {"chunk_id": "c2", "document_id": private_doc.id, "content": "p", "source_file_name": "p"},
            ]
            fake_metadata = {}
            from app.rag import retriever as retriever_mod
            with patch.object(retriever_mod, "retrieve_chunks_with_settings",
                              return_value=(fake_chunks, fake_metadata)):
                chunks, metadata = retriever_mod.retrieve_chunks_with_auth(
                    query="any query", auth=_viewer_auth(viewer_user), debug=True,
                )
            kept_doc_ids = {c["document_id"] for c in chunks}
            assert private_doc.id not in kept_doc_ids
            assert global_doc.id in kept_doc_ids
            assert metadata.get("filtered_count") == 1
        finally:
            _cleanup_doc(db_session, private_doc.id)
            _cleanup_doc(db_session, global_doc.id)

    def test_zero_access_yields_empty_chunks(self, db_session, owner_user, other_user):
        private_doc = _make_doc(db_session, "rag_priv_zero", visibility="private", owner_user_id=owner_user.id)
        try:
            fake_chunks = [
                {"chunk_id": "c1", "document_id": private_doc.id, "content": "p", "source_file_name": "p"},
            ]
            fake_metadata = {}
            from app.rag import retriever as retriever_mod
            with patch.object(retriever_mod, "retrieve_chunks_with_settings",
                              return_value=(fake_chunks, fake_metadata)):
                chunks, metadata = retriever_mod.retrieve_chunks_with_auth(
                    query="any query", auth=_user_auth(other_user), debug=True,
                )
            assert chunks == []
            assert metadata.get("accessible_count") == 0
            assert metadata.get("total_count") == 1
            assert metadata.get("filtered_count") == 1
            # accessible_document_ids is the only place a doc_id would be echoed back;
            # it MUST NOT include the unauthorized private doc.
            accessible_ids = metadata.get("accessible_document_ids") or []
            assert private_doc.id not in accessible_ids
        finally:
            _cleanup_doc(db_session, private_doc.id)


class TestDebugMetadataDoesNotLeak:
    """The debug=True path must not include unauthorized doc IDs in metadata."""

    def test_debug_metadata_excludes_unauthorized_doc_id(self, db_session, admin_user, owner_user, other_user):
        private_doc = _make_doc(db_session, "rag_dbg_priv", visibility="private", owner_user_id=owner_user.id)
        try:
            fake_chunks = [
                {"chunk_id": "c1", "document_id": private_doc.id, "content": "SECRET", "source_file_name": "PRIVATE_FILE.bin"},
            ]
            fake_metadata = {"retrieval_status": "ok"}
            from app.rag import retriever as retriever_mod
            with patch.object(retriever_mod, "retrieve_chunks_with_settings",
                              return_value=(fake_chunks, fake_metadata)):
                chunks, metadata = retriever_mod.retrieve_chunks_with_auth(
                    query="any", auth=_user_auth(other_user), debug=True,
                )
            # Serialize the metadata. The only doc-id field in metadata is
            # 'accessible_document_ids'; it MUST NOT contain the unauthorized doc id.
            assert private_doc.id not in (metadata.get("accessible_document_ids") or [])
            assert "PRIVATE_FILE.bin" not in repr(metadata), \
                f"private file name leaked in metadata: {repr(metadata)}"
            assert "SECRET" not in repr(metadata)
            assert chunks == []
        finally:
            _cleanup_doc(db_session, private_doc.id)


# ---------------------------------------------------------------------------
# No-access safe fallback (RAG answer path)
# ---------------------------------------------------------------------------

class TestNoAccessSafeFallback:
    """When the resolver denies access to all candidate docs, the answer path
    must produce a safe fallback (no unauthorized content in the answer)."""

    def test_user_with_no_access_gets_empty_chunks_from_retriever(self, db_session, owner_user, other_user):
        """End-to-end: simulate a user who can see nothing, verify the chunk filter drops everything."""
        private_doc = _make_doc(db_session, "rag_nopriv", visibility="private", owner_user_id=owner_user.id)
        try:
            fake_chunks = [
                {"chunk_id": "c1", "document_id": private_doc.id, "content": "secret", "source_file_name": "PRIV.bin"},
            ]
            fake_metadata = {}
            from app.rag import retriever as retriever_mod
            with patch.object(retriever_mod, "retrieve_chunks_with_settings",
                              return_value=(fake_chunks, fake_metadata)):
                chunks, metadata = retriever_mod.retrieve_chunks_with_auth(
                    query="ask about anything", auth=_user_auth(other_user), debug=True,
                )
            assert chunks == []
            # Confirm the secret doesn't appear in any output field.
            joined = str(chunks) + str(metadata)
            assert "secret" not in joined.lower()
            assert "PRIV.bin" not in joined
        finally:
            _cleanup_doc(db_session, private_doc.id)
