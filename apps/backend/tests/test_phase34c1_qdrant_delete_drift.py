"""Phase 34C.1 -- Regression test for the Qdrant delete() API drift.

Background
----------

The Phase 34C ``indexer.py`` ``_delete_with_filter`` calls

    client.delete(collection_name=..., points=ids)

but ``qdrant-client==1.9.1`` (the version pinned in
``requirements.txt`` and installed in the project venv) exposes the
parameter as ``points_selector``, not ``points``. Passing ``points=``
raises ``TypeError: delete() got an unexpected keyword argument
'points'``.

The error is caught by the broad ``except Exception`` in
``_delete_with_filter`` and surfaces as
``DeleteResult(error="qdrant_delete_failed: ...")`` -- the function
returns ``deleted=0`` without anyone noticing. Document deletion and
image deletion both go through this code path, which is part of the
Phase 34C lifecycle contract.

These tests pin the fix so the drift cannot regress:

* ``test_delete_signature_uses_points_selector`` -- inspect the
  installed Qdrant client to confirm ``points_selector`` is the
  supported kwarg and ``points`` is not.

* ``test_delete_image_knowledge_succeeds_against_real_client`` --
  spin up the in-memory stub Qdrant client, exercise
  ``delete_image_knowledge`` through the indexer, and assert the
  call returned ``deleted=1`` with no error.

* ``test_delete_document_removes_only_image_knowledge`` -- assert
  ``delete_image_knowledge_by_document_id`` removes exactly the
  image-knowledge points (not other source types) -- regression for
  the same fix applied to the document-delete path.
"""

from __future__ import annotations

import inspect
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List

import pytest


def _ensure_settings_env() -> None:
    os.environ.setdefault("MULTIMODAL_KNOWLEDGE_ENABLED", "true")
    os.environ.setdefault("MULTIMODAL_KNOWLEDGE_SCHEMA_VERSION", "1")


_ensure_settings_env()


# ---------------------------------------------------------------------------
# Stubs -- mirror the pattern used in test_phase34c_indexer.py so the
# regression test exercises the same code path.
# ---------------------------------------------------------------------------


@dataclass
class StubPoint:
    id: int
    vector: List[float]
    payload: Dict[str, Any]


@dataclass
class StubEmbeddingProvider:
    model_name: str = "stub-embed-v1"
    dimension: int = 8

    def embed(self, texts):
        return [[0.1] * self.dimension for _ in texts]


class _QdrantClientLikeStub:
    """In-memory stub that follows the installed client's signature.

    The ``delete`` method accepts the NEW ``points_selector`` kwarg
    (not the old ``points`` kwarg) so the indexer's fix has to use
    the new name to call into this stub successfully.
    """

    def __init__(self) -> None:
        self.points: Dict[int, StubPoint] = {}
        self.delete_calls: List[List[int]] = []

    def upsert(self, *, collection_name: str, points: List[Any]) -> None:
        for p in points:
            self.points[p.id] = StubPoint(id=p.id, vector=p.vector, payload=p.payload)

    def scroll(self, *, collection_name: str, scroll_filter, limit: int,
               with_payload: bool, with_vectors: bool):
        out = []
        for pid, point in self.points.items():
            if _filter_matches(point.payload, scroll_filter):
                out.append(type("Point", (), {"id": pid, "payload": point.payload})())
            if len(out) >= limit:
                break
        return (out, None)

    def delete(self, *, collection_name: str, points_selector: List[int], **kwargs) -> Any:
        # Note: the parameter is ``points_selector`` -- the installed
        # qdrant-client==1.9.1 signature. The old ``points=`` name
        # raises TypeError on this stub (and on the real client).
        deleted_ids: List[int] = []
        for pid in points_selector:
            if pid in self.points:
                deleted_ids.append(pid)
                self.points.pop(pid, None)
        self.delete_calls.append(deleted_ids)
        return type("UpdateResult", (), {"operation_id": 1, "status": "completed"})()


def _filter_matches(payload: Dict[str, Any], flt) -> bool:
    try:
        must = list(flt.must or [])
    except Exception:
        return True
    for cond in must:
        try:
            key = cond.key
            value = cond.match.value
        except Exception:
            return True
        if str(payload.get(key)) != str(value):
            return False
    return True


# ---------------------------------------------------------------------------
# Signature test -- runs without any app/db setup.
# ---------------------------------------------------------------------------


def test_delete_signature_uses_points_selector():
    """The installed qdrant-client==1.9.1 exposes ``points_selector``.

    Pin this so a future bump that re-adds ``points=`` as an alias
    does not silently undo the fix.
    """
    from qdrant_client import QdrantClient

    sig = inspect.signature(QdrantClient.delete)
    params = list(sig.parameters.keys())
    assert "points_selector" in params, params
    # Confirm the old kwarg is NOT in the formal signature.
    assert "points" not in params, params


# ---------------------------------------------------------------------------
# Behaviour tests -- exercise the fixed code path end to end.
# ---------------------------------------------------------------------------


@pytest.fixture
def stubbed_qdrant(monkeypatch):
    """Wire the indexer + knowledge record to the in-memory stub."""
    client = _QdrantClientLikeStub()

    # The indexer resolves the embedding provider and Qdrant client
    # lazily from ``app.services.embeddings`` /
    # ``app.services.vector.qdrant_service`` at call time. Patch the
    # resolved attributes on those modules -- same pattern used by
    # the existing Phase 34C indexer test suite.
    monkeypatch.setattr(
        "app.services.vector.qdrant_service.get_qdrant_client",
        lambda: client,
    )
    monkeypatch.setattr(
        "app.services.embeddings.get_embedding_provider",
        lambda: StubEmbeddingProvider(),
    )

    from app.services.multimodal import indexer as _indexer_module

    # The indexer also asks Qdrant for the collection name -- patch
    # it to a constant so we don't hit the live collection.
    monkeypatch.setattr(_indexer_module, "_collection_name", lambda: "stub-collection")
    monkeypatch.setattr(_indexer_module, "_is_enabled", lambda: True)

    return client


def _make_record(**overrides):
    from app.services.multimodal.knowledge_record import (
        ImageKnowledgeRecord,
        build_knowledge_text,
        cap_knowledge_text,
    )

    base = dict(
        document_id=42,
        image_id=7,
        owner_user_id=1,
        visibility="global",
        original_filename="server_b.png",
        mime_type="image/png",
        ocr_text="Server A\nServer B\nFailed",
        ocr_confidence=92,
        ocr_status="success",
        knowledge_schema_version=1,
        is_ocr_only=True,
    )
    base.update(overrides)
    rec = ImageKnowledgeRecord(**base)
    rec.knowledge_text = build_knowledge_text(rec)
    rec.embedding_text = cap_knowledge_text(rec.knowledge_text)
    return rec


def test_delete_image_knowledge_succeeds_against_real_client(stubbed_qdrant):
    """``delete_image_knowledge`` must succeed end to end.

    Regression for the qdrant-client 1.9.1 ``points=`` -> ``points_selector=``
    rename. Before the fix this returned
    ``DeleteResult(error="qdrant_delete_failed: ..."`` and deleted=0.
    """
    from app.services.multimodal.indexer import (
        delete_image_knowledge,
        upsert_image_knowledge,
    )

    rec = _make_record()
    upsert_result = upsert_image_knowledge(rec)
    assert upsert_result.status == "upserted"
    point_id = upsert_result.point_id
    assert point_id in stubbed_qdrant.points

    del_result = delete_image_knowledge(rec.image_id)

    # The fix must produce a clean delete with no error.
    assert del_result.error is None, del_result.error
    assert del_result.deleted == 1, del_result.deleted
    assert point_id not in stubbed_qdrant.points


def test_delete_document_removes_only_image_knowledge(stubbed_qdrant):
    """``delete_image_knowledge_by_document_id`` removes ONLY image_knowledge points.

    Regression for the same fix applied to the document-level delete
    helper. A non-image-knowledge point with the same document_id
    must survive.
    """
    from app.services.multimodal.indexer import (
        delete_image_knowledge_by_document_id,
        upsert_image_knowledge,
    )

    # Two image-knowledge points on document 1, one on document 2.
    upsert_image_knowledge(_make_record(document_id=1, image_id=1))
    upsert_image_knowledge(_make_record(document_id=1, image_id=2))
    upsert_image_knowledge(_make_record(document_id=2, image_id=3))

    # Inject a foreign point under document_id=1 with a different source_type.
    stubbed_qdrant.points[999_999] = StubPoint(
        id=999_999,
        vector=[0.0] * 8,
        payload={"source_type": "native_text", "document_id": 1},
    )

    res = delete_image_knowledge_by_document_id(1)

    # Two image-knowledge points removed; the foreign point untouched.
    assert res.error is None, res.error
    assert res.deleted == 2
    assert 999_999 in stubbed_qdrant.points


def test_indexer_does_not_use_legacy_points_kwarg():
    """Static guarantee: the indexer never passes ``points=`` to ``client.delete``.

    Read the source of ``_delete_with_filter`` and assert the call
    site uses ``points_selector``. A future refactor that reverts
    to the old kwarg will fail this test.
    """
    import inspect

    from app.services.multimodal import indexer

    src = inspect.getsource(indexer._delete_with_filter)
    assert "points_selector=" in src, src
    # The legacy kwarg should not appear (other than inside the
    # comment that documents the fix).
    for line in src.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        assert "points=ids" not in stripped and "points=ids_list" not in stripped, line
