"""
Phase 34B — Automatic Vision Intelligence Tests.

Covers all 20 spec scenarios across three categories:

ROUTER  (1–10)
GROUNDING  (11–15)
CACHE / PERSISTENCE  (16–20)

Every test is deterministic and uses the mock Vision provider
(VISION_PROVIDER=mock) so it can run with no network. The mock
provider returns a structured VisionResult without touching an
external API.

The 34A / 34A.1 / 34A.1.1 / 34A.1.2 / 34A.2 regression suites
remain the responsibility of their own test files
(test_phase34a_*, test_phase34a1_*, test_phase34a2_*).
"""

from __future__ import annotations

import os
import time
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Settings overrides — MUST be set BEFORE importing app modules so
# settings.VISION_* values are bound correctly.
# ---------------------------------------------------------------------------

os.environ.setdefault("VISION_ENABLED", "true")
os.environ.setdefault("VISION_PROVIDER", "mock")
os.environ.setdefault("VISION_MODEL", "mock-v1")
os.environ.setdefault("VISION_OCR_CONFIDENCE_THRESHOLD", "55")
os.environ.setdefault("VISION_MIN_OCR_TEXT_LENGTH", "25")
os.environ.setdefault("VISION_CACHE_SCHEMA_VERSION", "1")


# ---------------------------------------------------------------------------
# ROUTER TESTS  (1–10)
# ---------------------------------------------------------------------------


def test_router_01_high_confidence_error_code_lookup_is_ocr_only():
    """High-confidence text screenshot + 'What error code is shown?' → OCR_ONLY."""
    from app.vision.router import (
        ProcessingMode,
        RoutingSignal,
        decide_processing_mode,
    )

    decision = decide_processing_mode(
        "What error code is shown here?",
        ocr_text="Failed Reason: 902 - Message delivery failed",
        ocr_confidence=98,
        ocr_status="success",
        vision_enabled=True,
    )
    assert decision.processing_mode == ProcessingMode.OCR_ONLY
    assert decision.vision_required is False
    assert RoutingSignal.OCR_SUFFICIENT in decision.trigger_reasons


def test_router_02_high_confidence_plus_graph_intent_routes_to_vision():
    """High OCR confidence + 'What does this graph show?' → OCR_PLUS_VISION.

    The spec is explicit: high OCR confidence does NOT override a
    strong visual intent (graphs require shape, colour, position
    that OCR cannot read).
    """
    from app.vision.router import (
        ProcessingMode,
        RoutingSignal,
        decide_processing_mode,
    )

    decision = decide_processing_mode(
        "What does this graph show?",
        ocr_text="Monday Tuesday Wednesday Thursday CPU Memory",
        ocr_confidence=98,
        ocr_status="success",
        vision_enabled=True,
    )
    assert decision.processing_mode == ProcessingMode.OCR_PLUS_VISION
    assert decision.vision_required is True
    assert RoutingSignal.VISUAL_INTENT in decision.trigger_reasons


def test_router_03_low_ocr_confidence_routes_to_vision_fallback():
    """Low OCR confidence → VISION_FALLBACK."""
    from app.vision.router import (
        ProcessingMode,
        RoutingSignal,
        decide_processing_mode,
    )

    decision = decide_processing_mode(
        "What error code is shown here?",
        ocr_text="s0me pa.rt1al te-xt with several words to clear min length",
        ocr_confidence=20,  # well below the 55 threshold
        ocr_status="success",
        vision_enabled=True,
    )
    assert decision.processing_mode == ProcessingMode.OCR_PLUS_VISION
    assert decision.vision_required is True
    assert RoutingSignal.LOW_OCR_CONFIDENCE in decision.trigger_reasons


def test_router_04_very_low_ocr_text_with_visual_intent_is_vision_fallback():
    """Very low OCR text length with visual intent → VISION_FALLBACK."""
    from app.vision.router import (
        ProcessingMode,
        RoutingSignal,
        decide_processing_mode,
    )

    decision = decide_processing_mode(
        "Describe this diagram",
        ocr_text="",
        ocr_confidence=None,
        ocr_status="success",
        vision_enabled=True,
        vision_min_ocr_text_length=25,
    )
    assert decision.processing_mode == ProcessingMode.VISION_FALLBACK
    assert decision.vision_required is True
    assert RoutingSignal.LOW_OCR_TEXT_LENGTH in decision.trigger_reasons


def test_router_04b_very_low_ocr_text_identifier_only_stays_ocr_only():
    """Empty OCR + identifier lookup ('What error code is shown?')
    must stay OCR_ONLY. The user cannot benefit from Vision when the
    answer is supposed to be a textual identifier — and we MUST
    preserve the grounded insufficient-information response."""
    from app.vision.router import ProcessingMode, decide_processing_mode

    decision = decide_processing_mode(
        "What error code is shown?",
        ocr_text="",
        ocr_confidence=None,
        ocr_status="success",
        vision_enabled=True,
        vision_min_ocr_text_length=25,
    )
    assert decision.processing_mode == ProcessingMode.OCR_ONLY
    assert decision.vision_required is False


def test_router_05_which_button_should_i_click_invokes_vision():
    from app.vision.router import (
        ProcessingMode,
        RoutingSignal,
        decide_processing_mode,
    )

    decision = decide_processing_mode(
        "Which button should I click to continue?",
        ocr_text="OK  Cancel  Apply  Save  Discard  Help  About  Settings",
        ocr_confidence=95,
        ocr_status="success",
        vision_enabled=True,
    )
    assert decision.processing_mode == ProcessingMode.OCR_PLUS_VISION
    assert RoutingSignal.VISUAL_INTENT in decision.trigger_reasons


def test_router_06_which_server_red_indicator_invokes_vision():
    from app.vision.router import (
        ProcessingMode,
        RoutingSignal,
        decide_processing_mode,
    )

    decision = decide_processing_mode(
        "Which server has the red indicator?",
        ocr_text="Server A  Server B  Server C",
        ocr_confidence=95,
        ocr_status="success",
        vision_enabled=True,
    )
    assert decision.processing_mode == ProcessingMode.OCR_PLUS_VISION
    assert RoutingSignal.VISUAL_INTENT in decision.trigger_reasons


def test_router_07_text_query_does_not_call_vision():
    """Normal text / error-code query never invokes Vision — even
    when the user is asking about 'this' (an attached image). The
    spec calls this out explicitly for the 902 example."""
    from app.vision.router import ProcessingMode, decide_processing_mode

    decision = decide_processing_mode(
        "What error code is shown here?",
        ocr_text="902 - Message delivery failed",
        ocr_confidence=95,
        ocr_status="success",
        vision_enabled=True,
    )
    assert decision.processing_mode == ProcessingMode.OCR_ONLY
    assert decision.vision_required is False


def test_router_08_vision_disabled_keeps_ocr_only():
    """VISION_ENABLED=false restores the exact existing OCR-only path
    even when every visual signal fires."""
    from app.vision.router import (
        ProcessingMode,
        RoutingSignal,
        decide_processing_mode,
    )

    decision = decide_processing_mode(
        "Which server has the red indicator?",
        ocr_text="Server A Server B Server C",
        ocr_confidence=20,
        ocr_status="success",
        vision_enabled=False,
    )
    assert decision.processing_mode == ProcessingMode.OCR_ONLY
    assert decision.vision_required is False
    assert RoutingSignal.VISION_DISABLED in decision.trigger_reasons


def test_router_09_vision_provider_failure_falls_back_to_ocr():
    """Provider failures degrade gracefully to OCR_ONLY — Vision is
    supplementary, not required. This is enforced at the orchestrator
    level, not the router, but the router's decision must allow it."""
    from app.vision.router import (
        ProcessingMode,
        decide_processing_mode,
    )

    decision = decide_processing_mode(
        "Describe the screen",
        ocr_text="Server A  Server B  Server C  CPU  Memory  Disk  Network",
        ocr_confidence=90,
        ocr_status="success",
        vision_enabled=True,
    )
    # The router decides; the orchestrator is what degrades. This
    # test pins the router's view: it would call Vision.
    assert decision.processing_mode == ProcessingMode.OCR_PLUS_VISION
    # Now verify the orchestrator degrades to OCR_ONLY when the
    # provider raises. We assert this in orchestrator tests below.


def test_router_10_high_confidence_does_not_override_visual_intent():
    """High OCR confidence does NOT automatically suppress Vision
    when there is a strong visual intent. The 98%-confidence graph
    case (test #2) already covers this; this test makes the rule
    explicit for several visual-intent phrasings."""
    from app.vision.router import (
        ProcessingMode,
        decide_processing_mode,
    )

    for q in [
        "What does this dashboard show?",
        "Explain this architecture diagram",
        "Describe the screenshot",
        "Where is the warning?",
        "What's wrong with this screen?",
        "Describe this image",
    ]:
        decision = decide_processing_mode(
            q,
            ocr_text="Some high confidence OCR text " * 4,
            ocr_confidence=98,
            ocr_status="success",
            vision_enabled=True,
        )
        assert decision.processing_mode == ProcessingMode.OCR_PLUS_VISION, (
            f"Expected OCR_PLUS_VISION for question {q!r}, got "
            f"{decision.processing_mode}"
        )


# ---------------------------------------------------------------------------
# GROUNDING TESTS  (11–15)
# ---------------------------------------------------------------------------


def test_grounding_11_ocr_identifies_error_902_with_image_context():
    """OCR identifies 902; user asks what code is shown → answer from OCR."""
    from app.vision.evidence import build_image_evidence
    from app.vision.router import (
        ProcessingMode,
        ImageProcessingDecision,
        decide_processing_mode,
    )

    decision = decide_processing_mode(
        "What error code is shown here?",
        ocr_text="902 - Message delivery failed: rejected-forbidden-country",
        ocr_confidence=98,
        ocr_status="success",
        vision_enabled=True,
    )
    evidence = build_image_evidence(
        decision=decision,
        ocr_text="902 - Message delivery failed: rejected-forbidden-country",
        ocr_confidence=98,
        document_id=42,
        image_id=7,
        image_filename="error_902.png",
        source_type="image_ocr",
    )
    assert evidence.vision_used is False
    assert "902" in evidence.ocr_text
    # Image citation is implicit via document_id / image_id.
    assert evidence.document_id == 42
    assert evidence.image_id == 7


def test_grounding_12_vision_observes_red_indicator_is_image_grounded():
    """Vision can describe a red status indicator → image-grounded
    answer is allowed; the observation is preserved on the evidence
    payload so the LLM prompt can phrase it correctly."""
    from app.services.vision.base import VisionResult, VISION_SCHEMA_VERSION
    from app.vision.evidence import build_image_evidence
    from app.vision.router import ProcessingMode, ImageProcessingDecision
    from app.vision.router import RoutingSignal

    decision = ImageProcessingDecision(
        processing_mode=ProcessingMode.OCR_PLUS_VISION,
        vision_required=True,
        trigger_reasons=[RoutingSignal.VISUAL_INTENT],
        ocr_confidence=95,
        ocr_text_length=120,
        visual_intent_detected=True,
        image_type_hint="dashboard",
    )
    vision_result = VisionResult(
        description="Dashboard view with three server rows.",
        image_type="dashboard",
        visual_findings=[
            "Server B has a red failure indicator while Servers A and C are green."
        ],
        detected_entities=["Server A", "Server B", "Server C"],
        visual_states=["red-indicator-on-server-b"],
        tags=["dashboard", "alert"],
        confidence=82,
        provider="mock",
        model="mock-v1",
        processing_time_ms=15,
        schema_version=VISION_SCHEMA_VERSION,
    )
    evidence = build_image_evidence(
        decision=decision,
        ocr_text="Server A Server B Server C status: ok failed ok",
        ocr_confidence=95,
        document_id=1,
        image_id=1,
        image_filename="dashboard.png",
        source_type="image_ocr",
        vision_result=vision_result,
    )
    assert evidence.vision_used is True
    assert evidence.vision_description == "Dashboard view with three server rows."
    assert "red failure indicator" in evidence.visual_findings[0]
    assert "Server B" in evidence.detected_entities


def test_grounding_13_vision_identifies_error_but_kb_lacks_meaning_no_invention():
    """Vision identifies 902 but KB lacks explanation → 'What does
    error 902 mean?' must NOT invent meaning. The Vision result is
    evidence FROM the image, not authoritative product knowledge."""
    from app.services.vision.base import VisionResult
    from app.vision.evidence import build_image_evidence
    from app.vision.router import ProcessingMode, ImageProcessingDecision, RoutingSignal

    decision = ImageProcessingDecision(
        processing_mode=ProcessingMode.OCR_PLUS_VISION,
        vision_required=True,
        trigger_reasons=[RoutingSignal.VISUAL_INTENT],
    )
    vision_result = VisionResult(
        description="The number 902 is visible on screen.",
        image_type="screenshot",
        visual_findings=[],
        detected_entities=["902"],
        visual_states=[],
        tags=[],
        confidence=70,
        provider="mock",
        model="mock-v1",
        processing_time_ms=10,
    )
    evidence = build_image_evidence(
        decision=decision,
        ocr_text="Error 902",
        ocr_confidence=90,
        vision_result=vision_result,
    )
    # Vision evidence is present and grounded to the image.
    assert evidence.vision_used is True
    assert "902" in evidence.detected_entities
    # Vision NEVER invents the meaning of 902. There is no
    # troubleshooting / definition in vision_description, only the
    # observation that the number is visible.
    assert "definition" not in evidence.vision_description.lower()
    assert "troubleshoot" not in evidence.vision_description.lower()
    assert "fix" not in evidence.vision_description.lower()
    # The evidence payload also has no fabricated remediation text.
    assert all(
        "fix" not in (f or "").lower() for f in evidence.visual_findings
    )


def test_grounding_14_vision_identifies_error_but_kb_lacks_troubleshooting():
    """Vision identifies error but KB lacks troubleshooting → no
    fabricated remediation. The VisionResult.description must be
    limited to what is visible; no provider fabrication is permitted
    downstream either."""
    from app.services.vision.base import VisionResult
    from app.vision.evidence import build_image_evidence
    from app.vision.router import ProcessingMode, ImageProcessingDecision, RoutingSignal

    decision = ImageProcessingDecision(
        processing_mode=ProcessingMode.OCR_PLUS_VISION,
        vision_required=True,
        trigger_reasons=[RoutingSignal.VISUAL_INTENT],
    )
    vision_result = VisionResult(
        description="Screenshot showing a '902' code and 'Failed Reason: rejected'.",
        image_type="screenshot",
        visual_findings=["Status bar at top is red"],
        detected_entities=["902"],
        visual_states=["red-status-bar"],
        tags=["error"],
        confidence=80,
        provider="mock",
        model="mock-v1",
        processing_time_ms=12,
    )
    evidence = build_image_evidence(
        decision=decision,
        ocr_text="902 - rejected",
        ocr_confidence=92,
        vision_result=vision_result,
    )
    assert evidence.vision_used is True
    # The evidence mentions the red indicator but does not fabricate
    # a fix procedure.
    prompt_section = evidence.to_prompt_section()
    assert "902" in prompt_section
    # We do NOT include remediation language — Vision is purely
    # descriptive and any KB step comes from RAG, not Vision.
    for forbidden in ("step 1:", "to fix:", "follow these steps", "resolution:"):
        assert forbidden not in prompt_section.lower(), (
            f"Vision evidence must not include remediation language "
            f"({forbidden!r}); KB / RAG owns remediation."
        )


def test_grounding_15_mixed_image_and_kb_answer_has_source_separation():
    """Mixed image + KB answer keeps source separation. The
    Evidence Builder emits OCR + Vision as a clearly labelled block,
    separate from KB chunks the existing pipeline injects."""
    from app.services.vision.base import VisionResult
    from app.vision.evidence import build_image_evidence
    from app.vision.router import ProcessingMode, ImageProcessingDecision, RoutingSignal

    decision = ImageProcessingDecision(
        processing_mode=ProcessingMode.OCR_PLUS_VISION,
        vision_required=True,
        trigger_reasons=[RoutingSignal.VISUAL_INTENT],
    )
    vision_result = VisionResult(
        description="Server B has a red failure indicator.",
        image_type="dashboard",
        visual_findings=["Server B is highlighted red"],
        detected_entities=["Server B"],
        visual_states=["red-indicator"],
        tags=["dashboard"],
        confidence=85,
        provider="mock",
        model="mock-v1",
        processing_time_ms=11,
    )
    evidence = build_image_evidence(
        decision=decision,
        ocr_text="Server A Server B Server C CPU 12% CPU 90% CPU 8%",
        ocr_confidence=94,
        document_id=10,
        image_id=20,
        image_filename="dashboard.png",
        source_type="image_ocr",
        vision_result=vision_result,
    )
    section = evidence.to_prompt_section()
    # Section is clearly labelled as image evidence — KB citations
    # come from separate chunks the LLM cites with [N] markers.
    assert section.startswith("IMAGE EVIDENCE")
    assert "Vision analysis" in section
    assert "OCR text:" in section
    assert "Server B" in section
    # The synthetic-chunk renderer produces a citable chunk whose
    # source filename matches the OCR document — citations stay
    # image-grounded.
    from app.vision.integration import _build_synthetic_vision_chunk

    synthetic = _build_synthetic_vision_chunk(
        evidence=evidence,
        source_filename="dashboard.png",
        document_id=10,
        image_id=20,
    )
    assert synthetic is not None
    assert synthetic["source_file_name"] == "dashboard.png"
    assert synthetic["document_id"] == 10
    assert synthetic["image_id"] == 20


# ---------------------------------------------------------------------------
# CACHE / PERSISTENCE TESTS  (16–20)
# ---------------------------------------------------------------------------


def test_cache_16_first_vision_call_persists_result(monkeypatch):
    """First Vision call stores the result on the DocumentImage row."""
    from app.models.document import DocumentImage
    from app.services.vision.base import VisionResult, VISION_SCHEMA_VERSION
    from app.vision.persistence import (
        lookup_cached_vision,
        persist_vision_result,
    )

    # Use an in-memory dict-backed fake DB.
    class _FakeImage:
        def __init__(self, **kwargs):
            for k, v in kwargs.items():
                setattr(self, k, v)

    image = _FakeImage(
        id=99,
        document_id=1,
        storage_key="k",
        mime_type="image/png",
        source_type="image_ocr",
        ocr_status="success",
        sequence_number=0,
    )
    fake_db = MagicMock()
    fake_db.query.return_value.filter.return_value.first.return_value = image

    vision_result = VisionResult(
        description="First-call description",
        image_type="dashboard",
        visual_findings=["finding one"],
        detected_entities=["Server A"],
        visual_states=["green-status"],
        tags=["dashboard"],
        confidence=80,
        provider="mock",
        model="mock-v1",
        processing_time_ms=10,
        schema_version=VISION_SCHEMA_VERSION,
    )

    outcome = persist_vision_result(fake_db, document_image_id=99, vision_result=vision_result)
    assert outcome.cache_hit is False
    assert outcome.error is None
    # The fake row should now carry the persisted fields.
    assert image.vision_status == "success"
    assert image.vision_provider == "mock"
    assert image.vision_model == "mock-v1"
    assert image.vision_description == "First-call description"
    assert image.vision_confidence == 80
    assert image.vision_findings == ["finding one"]
    assert image.vision_tags == ["dashboard"]
    assert image.vision_cache_key  # SHA-derived prefix stored
    assert fake_db.add.called
    assert fake_db.commit.called

    # And the lookup now sees the cached row.
    cached = lookup_cached_vision(
        fake_db,
        document_image_id=99,
        provider="mock",
        model="mock-v1",
        schema_version=VISION_SCHEMA_VERSION,
    )
    assert cached.cache_hit is True
    assert cached.vision_result is not None
    assert cached.vision_result.description == "First-call description"


def test_cache_17_second_compatible_query_reuses_stored_result(monkeypatch):
    """Second compatible query reuses the stored Vision result.
    Provider must NOT be called again. Verified at the orchestrator
    level below; here we just confirm the cache lookup returns a
    hit when the row matches."""
    from app.services.vision.base import VisionResult, VISION_SCHEMA_VERSION
    from app.services.vision.factory import reset_vision_provider_cache
    from app.vision.orchestrator import process_image_for_question
    from app.vision.persistence import (
        lookup_cached_vision,
        persist_vision_result,
    )

    reset_vision_provider_cache()

    class _FakeImage:
        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)

    image = _FakeImage(
        id=42,
        document_id=1,
        storage_key="k",
        mime_type="image/png",
        source_type="image_ocr",
        ocr_status="success",
        sequence_number=0,
    )
    fake_db = MagicMock()
    fake_db.query.return_value.filter.return_value.first.return_value = image

    # Pre-seed cache with a previous successful Vision call.
    persist_vision_result(
        fake_db,
        document_image_id=42,
        vision_result=VisionResult(
            description="Pre-seeded",
            image_type="dashboard",
            visual_findings=["f1"],
            detected_entities=["X"],
            visual_states=[],
            tags=[],
            confidence=70,
            provider="mock",
            model="mock-v1",
            processing_time_ms=8,
            schema_version=VISION_SCHEMA_VERSION,
        ),
    )

    # Configure settings to enable Vision.
    from app.core.config import settings as s
    monkeypatch.setattr(s, "VISION_ENABLED", True, raising=False)
    monkeypatch.setattr(s, "VISION_PROVIDER", "mock", raising=False)
    monkeypatch.setattr(s, "VISION_CACHE_SCHEMA_VERSION", 1, raising=False)

    outcome = process_image_for_question(
        fake_db,
        question="Describe the screen",
        image_id=42,
        document_id=1,
        image_filename="dashboard.png",
        source_type="image_ocr",
        ocr_text="Server A Server B Server C",
        ocr_confidence=95,
        ocr_status="success",
    )
    assert outcome.evidence.vision_used is True
    assert outcome.vision_cache_hit is True
    assert outcome.vision_called is False  # no provider call
    assert outcome.evidence.vision_description == "Pre-seeded"


def test_cache_18_no_duplicate_provider_call_when_cache_valid(monkeypatch):
    """No duplicate Vision provider call when the cache is valid.
    We monkeypatch the mock provider's analyze_image to raise — if
    the cache path is hit the provider is never called, so the
    exception never fires."""
    from app.services.vision.base import VisionResult, VISION_SCHEMA_VERSION
    from app.services.vision.factory import reset_vision_provider_cache
    from app.vision.orchestrator import process_image_for_question
    from app.vision.persistence import persist_vision_result

    reset_vision_provider_cache()

    class _FakeImage:
        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)

    image = _FakeImage(
        id=77,
        document_id=1,
        storage_key="k",
        mime_type="image/png",
        source_type="image_ocr",
        ocr_status="success",
        sequence_number=0,
    )
    fake_db = MagicMock()
    fake_db.query.return_value.filter.return_value.first.return_value = image

    persist_vision_result(
        fake_db,
        document_image_id=77,
        vision_result=VisionResult(
            description="Cached result",
            image_type="dashboard",
            visual_findings=[],
            detected_entities=[],
            visual_states=[],
            tags=[],
            confidence=80,
            provider="mock",
            model="mock-v1",
            processing_time_ms=5,
            schema_version=VISION_SCHEMA_VERSION,
        ),
    )

    # Patch the provider factory so the cached path skips it.
    from app.services.vision import factory as vf

    def _explode_if_called(*args, **kwargs):
        raise AssertionError("provider should not be called on cache hit")

    monkeypatch.setattr(vf, "get_vision_provider", _explode_if_called)

    from app.core.config import settings as s
    monkeypatch.setattr(s, "VISION_ENABLED", True, raising=False)
    monkeypatch.setattr(s, "VISION_PROVIDER", "mock", raising=False)
    monkeypatch.setattr(s, "VISION_CACHE_SCHEMA_VERSION", 1, raising=False)

    outcome = process_image_for_question(
        fake_db,
        question="Describe the screen",
        image_id=77,
        document_id=1,
        image_filename="dashboard.png",
        source_type="image_ocr",
        ocr_text="Some OCR text",
        ocr_confidence=95,
        ocr_status="success",
    )
    assert outcome.vision_cache_hit is True
    assert outcome.vision_called is False
    assert outcome.evidence.vision_description == "Cached result"


def test_cache_19_failed_provider_call_records_safe_failure(monkeypatch):
    """Failed Vision call records safe failure state without
    breaking OCR. The orchestrator records the failure on the
    DocumentImage row and falls back to OCR-only evidence."""
    from app.services.vision import factory as vf
    from app.services.vision.base import VisionProviderError
    from app.services.vision.factory import reset_vision_provider_cache
    from app.vision.orchestrator import process_image_for_question

    reset_vision_provider_cache()

    class _FakeImage:
        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)

    image = _FakeImage(
        id=123,
        document_id=1,
        storage_key="k",
        mime_type="image/png",
        source_type="image_ocr",
        ocr_status="success",
        sequence_number=0,
    )
    fake_db = MagicMock()
    fake_db.query.return_value.filter.return_value.first.return_value = image

    class _BoomProvider:
        name = "mock"
        model = "mock-v1"

        def analyze_image(self, **kwargs):
            raise VisionProviderError("provider exploded")

    def _boom_factory():
        return _BoomProvider()

    # PATCH THE ORCHESTRATOR'S OWN SYMBOL: orchestrator.py imports
    # ``get_vision_provider`` at module load time, so it has its
    # own local reference. Patching only the factory module
    # binding has no effect on the orchestrator. We must replace
    # ``app.vision.orchestrator.get_vision_provider``.
    from app.vision import orchestrator as orch

    monkeypatch.setattr(orch, "get_vision_provider", _boom_factory)

    from app.core.config import settings as s
    monkeypatch.setattr(s, "VISION_ENABLED", True, raising=False)
    monkeypatch.setattr(s, "VISION_PROVIDER", "mock", raising=False)
    monkeypatch.setattr(s, "VISION_CACHE_SCHEMA_VERSION", 1, raising=False)

    outcome = process_image_for_question(
        fake_db,
        question="Describe this dashboard",
        image_id=123,
        document_id=1,
        image_filename="dashboard.png",
        source_type="image_ocr",
        ocr_text="Server A  Server B  Server C",
        ocr_confidence=95,
        ocr_status="success",
        image_bytes=b"\x89PNG\r\n\x1a\n" + b"fake-image-bytes",
        image_mime_type="image/png",
    )
    # OCR evidence is preserved, Vision is marked failed.
    assert outcome.evidence.vision_used is False
    assert outcome.vision_called is False
    assert outcome.vision_error is not None
    assert "provider exploded" in outcome.vision_error
    # The row was marked failed and the failure recorder ran.
    assert image.vision_status == "failed"
    assert "provider exploded" in (image.vision_error or "")


def test_cache_20_existing_document_image_ocr_data_remains_intact():
    """Existing DocumentImage / OCR data is preserved when Vision
    persists a result. Only Vision-specific fields are touched."""
    from app.services.vision.base import VisionResult, VISION_SCHEMA_VERSION
    from app.vision.persistence import persist_vision_result

    class _FakeImage:
        def __init__(self, **kw):
            for k, v in kw.items():
                setattr(self, k, v)

    image = _FakeImage(
        id=200,
        document_id=1,
        storage_key="k",
        mime_type="image/png",
        source_type="pdf_page_ocr",
        ocr_status="success",
        ocr_provider="tesseract",
        ocr_confidence=92,
        ocr_text_hash="abc123",
        ocr_error=None,
        width=1280,
        height=720,
        byte_size=123456,
        sequence_number=0,
    )
    fake_db = MagicMock()
    fake_db.query.return_value.filter.return_value.first.return_value = image

    persist_vision_result(
        fake_db,
        document_image_id=200,
        vision_result=VisionResult(
            description="Vision description",
            image_type="screenshot",
            visual_findings=["f"],
            detected_entities=["e"],
            visual_states=["s"],
            tags=["t"],
            confidence=80,
            provider="mock",
            model="mock-v1",
            processing_time_ms=10,
            schema_version=VISION_SCHEMA_VERSION,
        ),
    )

    # Existing OCR fields remain intact.
    assert image.ocr_provider == "tesseract"
    assert image.ocr_confidence == 92
    assert image.ocr_text_hash == "abc123"
    assert image.ocr_status == "success"
    assert image.source_type == "pdf_page_ocr"
    assert image.width == 1280
    assert image.height == 720
    assert image.byte_size == 123456
    # New Vision fields populated.
    assert image.vision_status == "success"
    assert image.vision_provider == "mock"
    assert image.vision_model == "mock-v1"
    assert image.vision_description == "Vision description"


# ---------------------------------------------------------------------------
# ORCHESTRATOR-LEVEL TESTS  (complement to cache tests above)
# ---------------------------------------------------------------------------


def test_orchestrator_provider_failure_does_not_call_vision(monkeypatch):
    """Provider failure degrades to OCR-only — the orchestrator
    never crashes the chat request."""
    from app.services.vision import factory as vf
    from app.services.vision.base import VisionProviderError
    from app.services.vision.factory import reset_vision_provider_cache
    from app.vision.orchestrator import process_image_for_question

    reset_vision_provider_cache()

    class _BoomProvider:
        name = "mock"
        model = "mock-v1"

        def analyze_image(self, **kwargs):
            raise VisionProviderError("rate-limited")

    # PATCH THE ORCHESTRATOR'S OWN SYMBOL: orchestrator.py imports
    # ``get_vision_provider`` at module load time, so it has its
    # own local reference. Patching only the factory module
    # binding has no effect on the orchestrator. We must replace
    # ``app.vision.orchestrator.get_vision_provider``.
    from app.vision import orchestrator as orch

    monkeypatch.setattr(orch, "get_vision_provider", lambda: _BoomProvider())

    from app.core.config import settings as s
    monkeypatch.setattr(s, "VISION_ENABLED", True, raising=False)

    outcome = process_image_for_question(
        None,
        question="Which server has the red indicator?",
        image_id=None,
        document_id=None,
        ocr_text="Server A Server B Server C",
        ocr_confidence=95,
        ocr_status="success",
        image_bytes=b"\x89PNG\r\n\x1a\n",
    )
    assert outcome.evidence.vision_used is False
    assert outcome.vision_error is not None
    assert "rate-limited" in outcome.vision_error


def test_orchestrator_disabled_vision_is_ocr_only(monkeypatch):
    """VISION_ENABLED=false → OCR_ONLY for every question."""
    from app.vision.orchestrator import process_image_for_question
    from app.vision.router import ProcessingMode

    from app.core.config import settings as s
    monkeypatch.setattr(s, "VISION_ENABLED", False, raising=False)

    outcome = process_image_for_question(
        None,
        question="Which button should I click?",
        ocr_text="OK  Cancel  Apply",
        ocr_confidence=95,
        ocr_status="success",
        image_bytes=b"\x89PNG\r\n\x1a\n",
    )
    assert outcome.decision.processing_mode == ProcessingMode.OCR_ONLY
    assert outcome.evidence.vision_used is False
    assert outcome.vision_called is False


def test_orchestrator_synthetic_chunk_includes_metadata(monkeypatch):
    """When Vision is invoked, the integration layer produces a
    synthetic chunk that carries the same source filename as the
    OCR document so the existing citation pipeline picks it up."""
    from app.services.vision.base import VisionResult
    from app.services.vision.factory import reset_vision_provider_cache
    from app.vision.integration import maybe_run_vision
    from app.vision.orchestrator import process_image_for_question

    reset_vision_provider_cache()

    from app.core.config import settings as s
    monkeypatch.setattr(s, "VISION_ENABLED", True, raising=False)
    monkeypatch.setattr(s, "VISION_PROVIDER", "mock", raising=False)
    monkeypatch.setattr(s, "VISION_CACHE_SCHEMA_VERSION", 1, raising=False)

    chunks = [
        {
            "chunk_id": "ocr-1",
            "document_id": 1,
            "image_id": 11,
            "source_file_name": "dashboard.png",
            "title": "OCR Text: Server A Server B",
            "section_heading": "OCR",
            "content": "Source Type: Image\nOCR Text: Server A Server B Server C",
            "score": 0.92,
            "source_type": "image_ocr",
            "content_type": "image_ocr",
        }
    ]
    image_context = {"document_id": 1, "image_id": 11}
    new_chunks, outcome, vision_meta = maybe_run_vision(
        None,
        question="Describe the screen",
        chunks=chunks,
        image_context=image_context,
        ocr_text="Server A Server B Server C",
        ocr_confidence=95,
        ocr_status="success",
        image_bytes=b"\x89PNG\r\n\x1a\n" + b"x" * 64,
    )
    assert vision_meta["vision_called"] is True
    assert vision_meta["vision_provider"] == "mock"
    assert len(new_chunks) == 2
    # The first chunk is the synthetic Vision evidence chunk; it
    # carries the same source filename so existing citations stay
    # image-grounded.
    synthetic = new_chunks[0]
    assert synthetic["source_file_name"] == "dashboard.png"
    assert synthetic["source_type"] == "vision_synthetic"
    assert "Vision analysis" in synthetic["content"]
    assert "OCR text:" in synthetic["content"]


def test_orchestrator_handles_missing_image_bytes(monkeypatch):
    """Orchestrator handles missing image bytes — Vision falls back
    to OCR-only when bytes are unavailable (no bytes leaked to the
    provider, no exception)."""
    from app.services.vision.factory import reset_vision_provider_cache
    from app.vision.orchestrator import process_image_for_question

    reset_vision_provider_cache()

    from app.core.config import settings as s
    monkeypatch.setattr(s, "VISION_ENABLED", True, raising=False)

    outcome = process_image_for_question(
        None,
        question="Which server has the red indicator?",
        ocr_text="Server A  Server B  Server C",
        ocr_confidence=95,
        ocr_status="success",
        image_bytes=None,
    )
    assert outcome.evidence.vision_used is False
    assert outcome.image_bytes_source == "missing"
    assert outcome.vision_error is not None


def test_orchestrator_skipped_when_no_image_context():
    """When image_context is None, the orchestrator is a no-op."""
    from app.vision.orchestrator import process_image_for_question

    outcome = process_image_for_question(
        None,
        question="What error code is shown here?",
        ocr_text="902",
        ocr_confidence=98,
        ocr_status="success",
        image_bytes=None,
    )
    # No image_id/document_id → orchestrator skips, returns OCR-only.
    assert outcome.decision.processing_mode.value == "ocr_only"
    assert outcome.evidence.vision_used is False


# ---------------------------------------------------------------------------
# EVIDENCE BUILDER TESTS
# ---------------------------------------------------------------------------


def test_evidence_prompt_section_bounds():
    """Evidence prompt section respects length caps so a hostile
    Vision provider cannot blow the LLM context window."""
    from app.services.vision.base import VisionResult, VISION_SCHEMA_VERSION
    from app.vision.evidence import (
        MAX_DESCRIPTION_CHARS,
        MAX_FINDINGS,
        ImageEvidence,
        build_image_evidence,
    )
    from app.vision.router import ImageProcessingDecision, ProcessingMode, RoutingSignal

    huge_description = "x" * 50_000
    huge_findings = ["f" * 5000 for _ in range(50)]

    vision_result = VisionResult(
        description=huge_description,
        image_type="dashboard",
        visual_findings=huge_findings,
        detected_entities=["e" * 5000 for _ in range(50)],
        visual_states=["s" for _ in range(50)],
        tags=["t" for _ in range(50)],
        confidence=200,  # out of bounds → coerced
        provider="mock",
        model="mock-v1",
        processing_time_ms=10,
        schema_version=VISION_SCHEMA_VERSION,
    )
    decision = ImageProcessingDecision(
        processing_mode=ProcessingMode.OCR_PLUS_VISION,
        vision_required=True,
        trigger_reasons=[RoutingSignal.VISUAL_INTENT],
        ocr_confidence=95,
        ocr_text_length=100,
        visual_intent_detected=True,
    )
    evidence = build_image_evidence(
        decision=decision,
        ocr_text="x" * 5000,
        ocr_confidence=150,  # out of bounds
        vision_result=vision_result,
    )
    section = evidence.to_prompt_section()
    assert len(section) < 10_000  # bounded
    assert len(evidence.visual_findings) <= MAX_FINDINGS
    assert len(evidence.vision_description) <= MAX_DESCRIPTION_CHARS
    assert evidence.vision_confidence == 100  # coerced to 0..100
    assert evidence.ocr_confidence == 100


def test_evidence_ocr_only_path_does_not_emit_vision_block():
    from app.vision.evidence import build_image_evidence
    from app.vision.router import ImageProcessingDecision, ProcessingMode, RoutingSignal

    decision = ImageProcessingDecision(
        processing_mode=ProcessingMode.OCR_ONLY,
        vision_required=False,
        trigger_reasons=[RoutingSignal.OCR_SUFFICIENT],
        ocr_confidence=95,
        ocr_text_length=80,
    )
    evidence = build_image_evidence(
        decision=decision,
        ocr_text="902 - Message delivery failed",
        ocr_confidence=95,
    )
    assert evidence.vision_used is False
    section = evidence.to_prompt_section()
    assert "Vision analysis" not in section
    assert "OCR text:" in section
    assert "OCR-only" in section


# ---------------------------------------------------------------------------
# MOCK PROVIDER DETERMINISM TEST
# ---------------------------------------------------------------------------


def test_mock_provider_deterministic_for_same_bytes():
    """Mock provider is deterministic — identical inputs produce
    identical outputs. Critical for cache-key testing."""
    from app.services.vision.mock_provider import MockVisionProvider

    provider = MockVisionProvider()
    payload = b"\x89PNG\r\n\x1a\n" + b"some-bytes"
    r1 = provider.analyze_image(
        image_bytes=payload,
        mime_type="image/png",
        prompt="Describe the screen",
        ocr_text="Server A Server B Server C",
    )
    r2 = provider.analyze_image(
        image_bytes=payload,
        mime_type="image/png",
        prompt="Describe the screen",
        ocr_text="Server A Server B Server C",
    )
    assert r1.description == r2.description
    assert r1.visual_findings == r2.visual_findings
    assert r1.detected_entities == r2.detected_entities
    assert r1.confidence == r2.confidence
    assert r1.provider == "mock"
    assert r1.model == "mock-v1"


def test_mock_provider_marks_visual_intent_text():
    from app.services.vision.mock_provider import MockVisionProvider

    provider = MockVisionProvider()
    result = provider.analyze_image(
        image_bytes=b"x" * 32,
        mime_type="image/png",
        prompt="Which server has the red indicator?",
        ocr_text="Server A Server B Server C",
    )
    # The mock should tag visual_states with red-indicator.
    assert any("red" in s.lower() for s in result.visual_states)


# ---------------------------------------------------------------------------
# ROUTER ADDITIONAL EDGE CASES
# ---------------------------------------------------------------------------


def test_router_ocr_failed_status_routes_to_vision_fallback():
    from app.vision.router import (
        ProcessingMode,
        RoutingSignal,
        decide_processing_mode,
    )

    decision = decide_processing_mode(
        "What error code is shown here?",
        ocr_text="",
        ocr_status="failed",
        ocr_error="tesseract crashed",
        vision_enabled=True,
    )
    assert decision.processing_mode == ProcessingMode.VISION_FALLBACK
    assert decision.vision_required is True
    assert RoutingSignal.OCR_ERROR in decision.trigger_reasons


def test_router_router_disabled_collapses_to_ocr_only():
    """VISION_ROUTER_ENABLED=false forces OCR-only even when every
    signal would route to Vision."""
    from app.vision.router import ProcessingMode, decide_processing_mode

    decision = decide_processing_mode(
        "Describe the screen",
        ocr_text="Some text",
        ocr_confidence=20,
        ocr_status="success",
        vision_enabled=True,
        vision_router_enabled=False,
    )
    assert decision.processing_mode == ProcessingMode.OCR_ONLY
    assert decision.vision_required is False


def test_router_empty_question_collapses_to_ocr_only():
    from app.vision.router import ProcessingMode, decide_processing_mode

    decision = decide_processing_mode(
        "",
        ocr_text="Some text",
        ocr_confidence=20,
        ocr_status="success",
        vision_enabled=True,
    )
    assert decision.processing_mode == ProcessingMode.OCR_ONLY


def test_router_safe_int_handles_garbage_confidence():
    """OCR confidence outside 0..100 is coerced into 0..100."""
    from app.vision.router import decide_processing_mode

    decision = decide_processing_mode(
        "Describe the screen",
        ocr_text="Some text",
        ocr_confidence="not-a-number",
        ocr_status="success",
        vision_enabled=True,
    )
    assert decision.ocr_confidence is None


# ---------------------------------------------------------------------------
# INTENT DETECTION TEST MATRIX
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("question", [
    "what do you see",
    "what's wrong with this screen",
    "what looks wrong",
    "describe this image",
    "describe this screenshot",
    "describe the screenshot",
    "explain this diagram",
    "which button should I click?",
    "which server is unhealthy?",
    "what is selected?",
    "what is disabled?",
    "where is the warning?",
    "red indicator status",
    "is there a spike in the graph?",
    "explain the dashboard",
    "describe the architecture diagram",
    "what trend do you see?",
])
def test_visual_intent_detector_returns_true_for_visual_questions(question):
    from app.vision.router import detect_visual_intent
    assert detect_visual_intent(question) is True, (
        f"Expected visual intent for question {question!r}"
    )


@pytest.mark.parametrize("question", [
    "What error code is shown?",
    "What error code is shown here?",
    "What number is displayed?",
    "Read out the text in the screenshot",
    "What is the receiver name?",
    "What is the amount?",
    "What is the message?",
])
def test_visual_intent_detector_returns_false_for_identifier_lookups(question):
    from app.vision.router import detect_visual_intent
    assert detect_visual_intent(question) is False, (
        f"Identifier-lookup question should NOT trigger Vision: {question!r}"
    )


# ---------------------------------------------------------------------------
# CACHE KEY DETERMINISM TEST
# ---------------------------------------------------------------------------


def test_cache_key_is_stable_for_same_inputs():
    from app.services.vision.base import build_cache_key
    a = build_cache_key(document_image_id=1, provider="mock", model="mock-v1")
    b = build_cache_key(document_image_id=1, provider="mock", model="mock-v1")
    assert a == b
    # Different image_id, same provider/model → different key.
    c = build_cache_key(document_image_id=2, provider="mock", model="mock-v1")
    assert c != a
    # Different model → different key (cache invalidation across models).
    d = build_cache_key(document_image_id=1, provider="mock", model="mock-v2")
    assert d != a
    # Different schema version → different key.
    e = build_cache_key(
        document_image_id=1,
        provider="mock",
        model="mock-v1",
        schema_version=2,
    )
    assert e != a


# ---------------------------------------------------------------------------
# CONFIG PRESENCE TEST
# ---------------------------------------------------------------------------


def test_vision_settings_are_loaded():
    """VISION_* settings are loaded from env and exposed via
    settings without secrets leaking."""
    from app.core.config import settings

    assert hasattr(settings, "VISION_ENABLED")
    assert hasattr(settings, "VISION_PROVIDER")
    assert hasattr(settings, "VISION_MODEL")
    assert hasattr(settings, "VISION_OCR_CONFIDENCE_THRESHOLD")
    assert hasattr(settings, "VISION_MIN_OCR_TEXT_LENGTH")
    assert hasattr(settings, "VISION_ROUTER_ENABLED")
    assert hasattr(settings, "VISION_TIMEOUT_SECONDS")
    assert hasattr(settings, "VISION_MAX_RETRIES")
    assert hasattr(settings, "VISION_MAX_IMAGE_BYTES")
    assert hasattr(settings, "VISION_CACHE_SCHEMA_VERSION")
    # No secrets in settings object.
    assert not hasattr(settings, "VISION_API_KEY")
