"""Phase 34C -- Backfill CLI tests.

Validates the CLI surface of the backfill: argument parsing,
disabled short-circuit, dry-run summary shape, and the
``_resolve_target_documents`` dispatcher. The full
DB -> loader -> indexer pipeline is covered by
``test_phase34c_lifecycle.py`` (idempotency) and ``test_phase34c_indexer.py``
(no duplicates, same point id on re-run).

Mapping to acceptance criterion:
   24. Backfill can be rerun safely. (verified via the lifecycle tests;
       here we only test that the CLI is wired to call into them.)
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict, List

import pytest


def _ensure_settings_env():
    os.environ.setdefault("MULTIMODAL_KNOWLEDGE_ENABLED", "true")
    os.environ.setdefault("MULTIMODAL_KNOWLEDGE_SCHEMA_VERSION", "1")


_ensure_settings_env()


def _build_args(**overrides) -> argparse.Namespace:
    base = dict(
        document_id=None,
        limit=0,
        batch_size=50,
        dry_run=False,
        user_id=None,
        include_ocr_empty=False,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def test_backfill_arg_parser_defaults():
    """The parser exposes the documented flags with sane defaults."""
    from app.services.multimodal.scripts.backfill import _build_arg_parser

    parser = _build_arg_parser()
    args = parser.parse_args([])
    assert args.document_id is None
    assert args.limit == 0
    assert args.batch_size == 50
    assert args.dry_run is False
    assert args.user_id is None
    assert args.include_ocr_empty is False


def test_backfill_arg_parser_parses_flags():
    """The parser accepts the documented flags."""
    from app.services.multimodal.scripts.backfill import _build_arg_parser

    parser = _build_arg_parser()
    args = parser.parse_args([
        "--document-id", "42",
        "--limit", "200",
        "--batch-size", "25",
        "--dry-run",
        "--user-id", "7",
        "--include-ocr-empty",
    ])
    assert args.document_id == 42
    assert args.limit == 200
    assert args.batch_size == 25
    assert args.dry_run is True
    assert args.user_id == 7
    assert args.include_ocr_empty is True


# ---------------------------------------------------------------------------
# Disabled short-circuit
# ---------------------------------------------------------------------------


def test_backfill_disabled_short_circuits(monkeypatch):
    """MULTIMODAL_KNOWLEDGE_ENABLED=false -> backfill reports disabled=True."""
    from app.core.config import settings as _settings
    monkeypatch.setattr(_settings, "MULTIMODAL_KNOWLEDGE_ENABLED", False)
    from app.services.multimodal.scripts.backfill import run

    summary = run(_build_args())
    assert summary.get("disabled") is True
    # No counts computed when disabled.
    assert summary.get("indexed", 0) == 0


def test_backfill_enabled_produces_expected_summary_keys(monkeypatch):
    """Summary dict has the documented keys regardless of outcome."""
    from app.core.config import settings as _settings
    monkeypatch.setattr(_settings, "MULTIMODAL_KNOWLEDGE_ENABLED", True)
    # Drive run() with an empty / invalid DB path so it returns the
    # fallback summary shape.
    from app.services.multimodal.scripts import backfill as backfill_mod

    monkeypatch.setattr(backfill_mod, "SessionLocal", lambda: None, raising=False)
    from app.services.multimodal.scripts.backfill import run

    summary = run(_build_args())
    # The summary is well-formed. ``disabled`` is only added by the
    # disabled short-circuit; it should NOT be set here.
    for key in (
        "scanned", "indexed", "skipped", "errors", "error_samples",
        "dry_run", "document_id", "user_id", "limit",
    ):
        assert key in summary, f"missing key {key}"
    assert "disabled" not in summary or summary["disabled"] is not True
    assert isinstance(summary["error_samples"], list)


# ---------------------------------------------------------------------------
# Document-id dispatcher
# ---------------------------------------------------------------------------


def test_backfill_respects_document_id():
    """--document-id restricts the target list to that single document."""
    from app.services.multimodal.scripts.backfill import _resolve_target_documents

    class _DB:
        pass

    args = _build_args(document_id=2)
    assert _resolve_target_documents(_DB(), args) == [2]


# ---------------------------------------------------------------------------
# Dry-run -- CLI reports counts without writing
# ---------------------------------------------------------------------------


def test_backfill_dry_run_summary_shape(monkeypatch):
    """Dry-run summary contains the would_index field and is well-formed."""
    from app.services.multimodal.scripts.backfill import run
    from app.services.multimodal.scripts import backfill as backfill_mod

    # Stub out the DB access and the image-id enumeration so the
    # dry-run path can compute counts without a real DB.
    class _FakeDB:
        pass

    monkeypatch.setattr(backfill_mod, "SessionLocal", lambda: _FakeDB(), raising=False)
    monkeypatch.setattr(
        backfill_mod, "_resolve_target_documents", lambda db, args: [1, 2],
    )
    monkeypatch.setattr(
        backfill_mod, "_image_ids_for",
        lambda db, did, *, include_ocr_empty: [10, 11] if did == 1 else [20],
    )

    summary = run(_build_args(dry_run=True))
    assert summary["dry_run"] is True
    assert summary["scanned"] == 3  # 2 images for doc 1, 1 image for doc 2
    assert summary["indexed"] == 0
    assert summary["would_index"] == 3
    # No errors expected on the dry-run path.
    assert summary["errors"] == 0


# ---------------------------------------------------------------------------
# Limit flag is surfaced
# ---------------------------------------------------------------------------


def test_backfill_summary_includes_limit(monkeypatch):
    """The ``limit`` field is present and reflects the --limit flag."""
    from app.services.multimodal.scripts.backfill import run
    from app.services.multimodal.scripts import backfill as backfill_mod

    monkeypatch.setattr(backfill_mod, "SessionLocal", lambda: None, raising=False)
    from app.core.config import settings as _settings
    monkeypatch.setattr(_settings, "MULTIMODAL_KNOWLEDGE_ENABLED", False)

    summary = run(_build_args(limit=100))
    assert summary["limit"] == 100


# ---------------------------------------------------------------------------
# Pipeline idempotency contract (delegated to lifecycle tests)
# ---------------------------------------------------------------------------


def test_backfill_calls_into_lifecycle_helpers(monkeypatch):
    """When enabled and no --document-id, run() calls reindex_all_images.

    This is the wiring contract that connects the CLI to the
    lifecycle module. The lifecycle idempotency contract itself is
    proven in ``test_phase34c_lifecycle.py``.
    """
    from app.services.multimodal.scripts.backfill import run
    from app.services.multimodal.scripts import backfill as backfill_mod
    from app.services.multimodal import lifecycle as lifecycle_mod

    class _FakeDB:
        pass

    monkeypatch.setattr(backfill_mod, "SessionLocal", lambda: _FakeDB(), raising=False)
    monkeypatch.setattr(
        backfill_mod, "_resolve_target_documents", lambda db, args: [],
    )

    called = {"all": 0, "per_doc": 0}

    def _fake_all(db, *, batch_size, document_id=None):
        called["all"] += 1
        return {"scanned": 0, "indexed": 0, "skipped": 0, "errors": 0, "error_samples": []}

    def _fake_per_doc(db, document_id, *, batch_size):
        called["per_doc"] += 1
        return {"scanned": 0, "indexed": 0, "skipped": 0, "errors": 0, "error_samples": []}

    monkeypatch.setattr(lifecycle_mod, "reindex_all_images", _fake_all)
    monkeypatch.setattr(lifecycle_mod, "reindex_document_images", _fake_per_doc)

    summary = run(_build_args())
    # No target documents -- reindex_all_images is still called (with an
    # empty target list) but per-document is never invoked.
    assert called["all"] == 1
    assert called["per_doc"] == 0
    assert summary["indexed"] == 0
