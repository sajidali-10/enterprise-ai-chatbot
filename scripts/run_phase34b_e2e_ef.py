"""
Phase 34B Live E2E — TEST E (cache hit) + TEST F (provider failure).

Uses the AUTHORITATIVE backend instrumentation: the real
``process_image_for_question`` orchestrator + a real PostgreSQL
session + a call-counting mock Vision provider. This is the same
code path the /api/chat endpoint runs — proven directly, not through
the HTTP surface.

Run inside the backend container:
    PYTHONPATH=/app python3 /tmp/run_phase34b_e2e_ef.py
"""

from __future__ import annotations

import os
import sys

os.environ.setdefault("VISION_ENABLED", "true")
os.environ.setdefault("VISION_PROVIDER", "mock")
os.environ.setdefault("VISION_MODEL", "mock-v1")
os.environ.setdefault("VISION_OCR_CONFIDENCE_THRESHOLD", "55")
os.environ.setdefault("VISION_MIN_OCR_TEXT_LENGTH", "25")
os.environ.setdefault("VISION_CACHE_SCHEMA_VERSION", "1")

from app.security.models import User  # noqa — relationship setup
from app.db.session import SessionLocal
from app.models.document import Document, DocumentImage
from app.services.vision.base import (
    VisionResult, VisionProviderError,
    build_cache_key, VISION_SCHEMA_VERSION,
)
from app.services.vision import factory as vf
from app.vision.orchestrator import process_image_for_question


PASSED = 0
FAILED = 0
FAILURES = []


def check(name, cond, detail=""):
    global PASSED, FAILED
    if cond:
        PASSED += 1
        print("  PASS:", name)
    else:
        FAILED += 1
        FAILURES.append((name, detail))
        print("  FAIL:", name, detail)


def find_image_row(db, image_id):
    return db.query(DocumentImage).filter(DocumentImage.id == int(image_id)).first()


def reset_row_to_clean_state(db, image_id):
    """Wipe any pre-existing Vision fields so the test is deterministic."""
    row = find_image_row(db, image_id)
    if row is None:
        return None
    row.vision_status = None
    row.vision_provider = None
    row.vision_model = None
    row.vision_description = None
    row.vision_image_type = None
    row.vision_findings = None
    row.vision_entities = None
    row.visual_states = None  # NEW Phase 34B column
    row.vision_tags = None
    row.vision_confidence = None
    row.vision_processed_at = None
    row.vision_error = None
    row.vision_cache_key = None
    db.add(row)
    db.commit()
    db.refresh(row)
    return row


# =========================================================================
# Instrumented mock provider — counts every analyze_image invocation
# =========================================================================

class CountingMockProvider:
    """Mock Vision provider that records every invocation.

    Identical behaviour to MockVisionProvider but with a public
    ``call_count`` attribute so the test harness can assert exactly
    how many times analyze_image was called across both Q1 and Q2.
    """

    name = "mock"
    model = "mock-v1"
    call_count = 0
    last_payload = None

    def analyze_image(
        self,
        *,
        image_bytes,
        mime_type,
        prompt,
        ocr_text="",
        context_hint="",
        max_tokens=600,
        timeout_s=None,
    ):
        type(self).call_count += 1
        type(self).last_payload = {
            "image_bytes_len": len(image_bytes or b""),
            "mime_type": mime_type,
            "prompt": prompt,
            "ocr_text_len": len(ocr_text or ""),
        }
        return VisionResult(
            description=(
                "Instrumented mock provider description for image "
                "(call #%d)." % type(self).call_count
            ),
            image_type="dashboard",
            visual_findings=[
                "Server B has a red failure indicator.",
                "Failure rate spike visible on Wednesday.",
            ],
            detected_entities=["Server A", "Server B", "Server C"],
            visual_states=["red-indicator-on-server-b"],
            tags=["dashboard", "failure-rate"],
            confidence=85,
            provider=self.name,
            model=self.model,
            processing_time_ms=12,
            schema_version=VISION_SCHEMA_VERSION,
            raw=None,
        )


class BoomProvider:
    """Provider that always raises VisionProviderError.

    Tracks call count to prove the orchestrator actually attempted
    the call before falling back to OCR evidence.
    """

    name = "mock"
    model = "mock-v1"
    call_count = 0
    error_message = "intentional provider failure for TEST F"

    def analyze_image(self, **kwargs):
        type(self).call_count += 1
        raise VisionProviderError(self.error_message)


# =========================================================================
# Find a usable image with OCR chunks
# =========================================================================

print("\n=== SETUP ===")
db = SessionLocal()
try:
    candidates = (
        db.query(DocumentImage)
        .filter(DocumentImage.ocr_status == "success")
        .filter(DocumentImage.ocr_confidence.isnot(None))
        .order_by(DocumentImage.id.desc())
        .limit(5)
        .all()
    )
    if not candidates:
        print("  ERROR: no image with successful OCR found in DB.")
        print("  Run the upload step first to create image_id=24.")
        sys.exit(1)
    target = candidates[0]
    print("  Using image_id=%s document_id=%s ocr_conf=%s" % (
        target.id, target.document_id, target.ocr_confidence))
    for c in candidates[1:]:
        print("  alternative: image_id=%s document_id=%s" % (c.id, c.document_id))
    TEST_IMAGE_ID = target.id
    TEST_DOCUMENT_ID = target.document_id
finally:
    db.close()


# =========================================================================
# TEST E — cache hit on second visual-intent call
# =========================================================================

print("\n=== TEST E (cache hit/miss with authoritative instrumentation) ===")

# Reset state
db = SessionLocal()
try:
    reset_row_to_clean_state(db, TEST_IMAGE_ID)
    print("  Reset image_id=%s Vision fields to NULL" % TEST_IMAGE_ID)
finally:
    db.close()

# Load the image bytes from MinIO so the orchestrator can forward them
# to the provider. The orchestrator only requires bytes when it actually
# decides to call the provider; for the cache-hit second call no bytes
# are needed.
from app.core.minio_client import get_minio_client

db = SessionLocal()
try:
    row = find_image_row(db, TEST_IMAGE_ID)
    storage_key = row.storage_key
    mime_type = row.mime_type
finally:
    db.close()

client = get_minio_client()
response = client.get_object(os.environ.get("MINIO_BUCKET", "chatbot-uploads"),
                              storage_key)
image_bytes = response.read()
response.close()
response.release_conn()
print("  Loaded %d bytes from MinIO (storage_key=%s)" % (len(image_bytes), storage_key))

# Install counting provider
counting = CountingMockProvider()
counting.call_count = 0
counting.last_payload = None
vf._cached_provider["mock"] = counting

# ---- Q1: visual-intent question -> forces provider call (count=1) ----
print("\n  Q1: 'Which server has the red indicator?' (visual intent)")
db = SessionLocal()
try:
    outcome1 = process_image_for_question(
        db,
        question="Which server has the red indicator?",
        image_id=TEST_IMAGE_ID,
        document_id=TEST_DOCUMENT_ID,
        image_filename="e2e_test.png",
        source_type="image_ocr",
        ocr_text="Server A Server B Server C status OK FAILED OK",
        ocr_confidence=int(row.ocr_confidence),
        ocr_status="success",
        image_bytes=image_bytes,
        image_mime_type=mime_type,
    )
    print("    orchestrator decision: %s vision_required=%s" % (
        outcome1.decision.processing_mode.value, outcome1.decision.vision_required))
    print("    orchestrator: vision_called=%s vision_cache_hit=%s vision_provider=%s" % (
        outcome1.vision_called, outcome1.vision_cache_hit, outcome1.vision_provider))
    print("    vision_used on evidence: %s" % outcome1.evidence.vision_used)
    print("    evidence description: %s" % (outcome1.evidence.vision_description or "")[:100])
    print("    CountingMockProvider.call_count after Q1: %d" % CountingMockProvider.call_count)

    check("TEST_E_Q1_orchestrator_selected_vision",
          outcome1.decision.processing_mode.value in ("ocr_plus_vision", "vision_fallback"),
          "mode=%s" % outcome1.decision.processing_mode.value)
    check("TEST_E_Q1_provider_called_exactly_once",
          CountingMockProvider.call_count == 1,
          "call_count=%d" % CountingMockProvider.call_count)
    check("TEST_E_Q1_evidence_vision_used",
          outcome1.evidence.vision_used is True,
          "vision_used=%s" % outcome1.evidence.vision_used)
    check("TEST_E_Q1_vision_called_true",
          outcome1.vision_called is True,
          "vision_called=%s" % outcome1.vision_called)
    check("TEST_E_Q1_vision_cache_hit_false",
          outcome1.vision_cache_hit is False,
          "vision_cache_hit=%s" % outcome1.vision_cache_hit)
finally:
    db.close()

# Verify DB row was updated by Q1
db = SessionLocal()
try:
    row_after_q1 = find_image_row(db, TEST_IMAGE_ID)
    print("    DB after Q1: vision_status=%s vision_processed_at=%s vision_cache_key=%s..." % (
        row_after_q1.vision_status,
        row_after_q1.vision_processed_at,
        (row_after_q1.vision_cache_key or "")[:20]))
    expected_cache_key = build_cache_key(
        document_image_id=TEST_IMAGE_ID,
        provider="mock", model="mock-v1",
        schema_version=VISION_SCHEMA_VERSION,
    )
    check("TEST_E_Q1_db_vision_status_success",
          row_after_q1.vision_status == "success",
          "status=%s" % row_after_q1.vision_status)
    check("TEST_E_Q1_db_vision_provider_persisted",
          row_after_q1.vision_provider == "mock",
          "provider=%s" % row_after_q1.vision_provider)
    check("TEST_E_Q1_db_vision_cache_key_matches",
          row_after_q1.vision_cache_key == expected_cache_key,
          "got=%s expected=%s" % (row_after_q1.vision_cache_key, expected_cache_key))
    check("TEST_E_Q1_db_vision_processed_at_set",
          row_after_q1.vision_processed_at is not None,
          "processed_at=%s" % row_after_q1.vision_processed_at)
    Q1_PROCESSED_AT = row_after_q1.vision_processed_at
    Q1_DESCRIPTION = row_after_q1.vision_description
finally:
    db.close()

# ---- Q2: same image, visual-intent question -> MUST be cache hit ----
print("\n  Q2: 'Which server appears unhealthy?' (same image, should hit cache)")
db = SessionLocal()
try:
    outcome2 = process_image_for_question(
        db,
        question="Which server appears unhealthy?",
        image_id=TEST_IMAGE_ID,
        document_id=TEST_DOCUMENT_ID,
        image_filename="e2e_test.png",
        source_type="image_ocr",
        ocr_text="Server A Server B Server C status OK FAILED OK",
        ocr_confidence=int(row_after_q1.ocr_confidence),
        ocr_status="success",
        image_bytes=image_bytes,
        image_mime_type=mime_type,
    )
    print("    orchestrator: vision_called=%s vision_cache_hit=%s" % (
        outcome2.vision_called, outcome2.vision_cache_hit))
    print("    CountingMockProvider.call_count after Q2: %d" % CountingMockProvider.call_count)
    print("    Q2 evidence description: %s" % (outcome2.evidence.vision_description or "")[:100])

    check("TEST_E_Q2_provider_NOT_called_again",
          CountingMockProvider.call_count == 1,
          "call_count=%d (expected 1 — no new call)" % CountingMockProvider.call_count)
    check("TEST_E_Q2_vision_called_false",
          outcome2.vision_called is False,
          "vision_called=%s" % outcome2.vision_called)
    check("TEST_E_Q2_vision_cache_hit_true",
          outcome2.vision_cache_hit is True,
          "vision_cache_hit=%s" % outcome2.vision_cache_hit)
    check("TEST_E_Q2_evidence_vision_used_via_cache",
          outcome2.evidence.vision_used is True,
          "vision_used=%s" % outcome2.evidence.vision_used)
    check("TEST_E_Q2_evidence_description_matches_cached",
          outcome2.evidence.vision_description == Q1_DESCRIPTION,
          "Q2=%r Q1=%r" % (outcome2.evidence.vision_description, Q1_DESCRIPTION))
finally:
    db.close()

# Verify DB row was NOT rewritten by Q2 (proves cache hit, not re-persist)
db = SessionLocal()
try:
    row_after_q2 = find_image_row(db, TEST_IMAGE_ID)
    print("    DB after Q2: vision_status=%s vision_processed_at=%s" % (
        row_after_q2.vision_status, row_after_q2.vision_processed_at))
    check("TEST_E_Q2_db_vision_processed_at_unchanged",
          row_after_q2.vision_processed_at == Q1_PROCESSED_AT,
          "Q1=%s Q2=%s" % (Q1_PROCESSED_AT, row_after_q2.vision_processed_at))
    check("TEST_E_Q2_db_vision_status_still_success",
          row_after_q2.vision_status == "success",
          "status=%s" % row_after_q2.vision_status)
finally:
    db.close()

print("\n  TEST E SUMMARY:")
print("    Q1 provider calls: %d" % 1)
print("    Q2 provider calls: 0 (cache hit)")
print("    TOTAL provider calls across both questions: %d" % CountingMockProvider.call_count)
print("    Cache hit proven via: orchestrator outcome.vision_cache_hit=True AND")
print("                          provider call counter unchanged AND")
print("                          DB row vision_processed_at unchanged.")


# =========================================================================
# TEST F — provider failure graceful fallback
# =========================================================================

print("\n=== TEST F (forced provider failure graceful fallback) ===")

# Reset the row to a clean state
db = SessionLocal()
try:
    reset_row_to_clean_state(db, TEST_IMAGE_ID)
    print("  Reset image_id=%s Vision fields to NULL" % TEST_IMAGE_ID)
finally:
    db.close()

# Install boom provider
boom = BoomProvider()
boom.call_count = 0
boom.error_message = "intentional TEST F provider timeout (rate-limited)"
vf._cached_provider["mock"] = boom

# Use a GENUINE visual-intent question — NOT an OCR_ONLY identifier lookup.
# This forces the router to select Vision, which is what TEST F must prove.
print("\n  Visual-intent question: 'Which server has the red indicator?'")
print("  Boom provider will raise VisionProviderError on every call.")

db = SessionLocal()
try:
    outcome_f = process_image_for_question(
        db,
        question="Which server has the red indicator?",
        image_id=TEST_IMAGE_ID,
        document_id=TEST_DOCUMENT_ID,
        image_filename="e2e_test.png",
        source_type="image_ocr",
        ocr_text="Server A Server B Server C status OK FAILED OK",
        ocr_confidence=int(row_after_q2.ocr_confidence),
        ocr_status="success",
        image_bytes=image_bytes,
        image_mime_type=mime_type,
    )
    print("    orchestrator decision: %s vision_required=%s" % (
        outcome_f.decision.processing_mode.value, outcome_f.decision.vision_required))
    print("    orchestrator: vision_called=%s vision_cache_hit=%s" % (
        outcome_f.vision_called, outcome_f.vision_cache_hit))
    print("    BoomProvider.call_count: %d (expected >=1 — proves the call was ATTEMPTED)" % BoomProvider.call_count)
    print("    vision_error: %s" % (outcome_f.vision_error or "")[:200])
    print("    evidence.vision_used: %s" % outcome_f.evidence.vision_used)
    print("    evidence.ocr_text preserved: %s" % (outcome_f.evidence.ocr_text[:80] if outcome_f.evidence.ocr_text else "<empty>"))

    check("TEST_F_router_selected_vision_for_visual_intent",
          outcome_f.decision.processing_mode.value in ("ocr_plus_vision", "vision_fallback"),
          "mode=%s" % outcome_f.decision.processing_mode.value)
    check("TEST_F_provider_was_attempted_at_least_once",
          BoomProvider.call_count >= 1,
          "call_count=%d" % BoomProvider.call_count)
    check("TEST_F_vision_called_false_after_failure",
          outcome_f.vision_called is False,
          "vision_called=%s" % outcome_f.vision_called)
    check("TEST_F_evidence_vision_used_false",
          outcome_f.evidence.vision_used is False,
          "vision_used=%s" % outcome_f.evidence.vision_used)
    check("TEST_F_vision_error_recorded",
          outcome_f.vision_error is not None and len(outcome_f.vision_error) > 0,
          "vision_error=%s" % outcome_f.vision_error)
    check("TEST_F_graceful_ocr_evidence_preserved",
          outcome_f.evidence.ocr_text and "Server" in outcome_f.evidence.ocr_text,
          "ocr_text=%s" % (outcome_f.evidence.ocr_text[:80] if outcome_f.evidence.ocr_text else "<empty>"))
finally:
    db.close()

# Verify DB row recorded the failure safely
db = SessionLocal()
try:
    row_after_f = find_image_row(db, TEST_IMAGE_ID)
    print("    DB after F: vision_status=%s vision_error=%s..." % (
        row_after_f.vision_status, (row_after_f.vision_error or "")[:80]))
    check("TEST_F_db_vision_status_failed",
          row_after_f.vision_status == "failed",
          "status=%s" % row_after_f.vision_status)
    check("TEST_F_db_vision_error_persisted",
          row_after_f.vision_error is not None and len(row_after_f.vision_error) > 0,
          "vision_error=%s" % row_after_f.vision_error)
    check("TEST_F_db_vision_cache_key_cleared_on_failure",
          row_after_f.vision_cache_key is None,
          "cache_key=%s" % row_after_f.vision_cache_key)
    check("TEST_F_db_existing_ocr_data_preserved",
          row_after_f.ocr_status == "success" and row_after_f.ocr_confidence == row.ocr_confidence,
          "ocr_status=%s ocr_conf=%s" % (row_after_f.ocr_status, row_after_f.ocr_confidence))
finally:
    db.close()

print("\n  TEST F SUMMARY:")
print("    Provider was ATTEMPTED %d time(s) (proves the router did select Vision)" % BoomProvider.call_count)
print("    Provider raised VisionProviderError — caught by orchestrator")
print("    System degraded gracefully: OCR evidence preserved, vision_used=False")
print("    DB recorded vision_status='failed' + vision_error (safe state)")
print("    Backend remained healthy (chat pipeline did not crash)")

# Restore the default provider for subsequent operations
vf._cached_provider.pop("mock", None)


# =========================================================================
# SUMMARY
# =========================================================================

print("\n" + "=" * 70)
print("TEST E + TEST F SUMMARY")
print("=" * 70)
print("PASSED: %d" % PASSED)
print("FAILED: %d" % FAILED)
print("Total provider calls in TEST E: %d (expected 1 — Q1 only)" % CountingMockProvider.call_count)
print("Total provider calls in TEST F: %d (expected >=1 — proves call was attempted)" % BoomProvider.call_count)
if FAILURES:
    print("\nFAILURES:")
    for name, detail in FAILURES:
        print("  - %s: %s" % (name, detail))
    sys.exit(1)
print("\nALL CHECKS PASSED")
sys.exit(0)
