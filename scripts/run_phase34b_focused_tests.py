"""
Focused Phase 34B unit test runner.

Exercises every Phase 34B spec scenario (router 1-10, grounding 11-15,
cache 16-20) without requiring fastapi/pytest. Run from inside the
backend container:

    cd /app && python3 /app/scripts/run_phase34b_focused_tests.py

or from the host after docker cp:

    docker cp scripts/run_phase34b_focused_tests.py chatbot_backend:/tmp/
    docker exec chatbot_backend python3 /tmp/run_phase34b_focused_tests.py
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

from app.vision.router import (
    decide_processing_mode, detect_visual_intent,
    ProcessingMode, RoutingSignal,
)
from app.vision.evidence import build_image_evidence, ImageEvidence
from app.services.vision.mock_provider import MockVisionProvider
from app.services.vision.factory import (
    get_vision_provider, reset_vision_provider_cache,
)
from app.services.vision.base import (
    VisionResult, VisionProviderError,
    build_cache_key, VISION_SCHEMA_VERSION,
)
from app.vision.orchestrator import process_image_for_question
from app.vision.persistence import (
    lookup_cached_vision, persist_vision_result,
    persist_vision_failure,
)


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


# ---------------------------------------------------------------------------
# ROUTER (spec scenarios 1-10)
# ---------------------------------------------------------------------------
print("\n=== ROUTER (1-10) ===")

# 1: identifier lookup -> OCR_ONLY
d = decide_processing_mode("What error code is shown here?",
    ocr_text="902 - Message delivery failed", ocr_confidence=98,
    ocr_status="success", vision_enabled=True)
check("1_ocr_only_for_identifier_lookup",
      d.processing_mode == ProcessingMode.OCR_ONLY
      and not d.vision_required,
      f"mode={d.processing_mode} vision_required={d.vision_required}")

# 2: graph intent -> OCR_PLUS_VISION
d = decide_processing_mode("What does this graph show?",
    ocr_text="Mon Tue Wed Thu CPU Memory", ocr_confidence=98,
    ocr_status="success", vision_enabled=True)
check("2_graph_intent_routes_to_vision",
      d.processing_mode == ProcessingMode.OCR_PLUS_VISION
      and RoutingSignal.VISUAL_INTENT in d.trigger_reasons,
      f"mode={d.processing_mode} reasons={d.trigger_reasons}")

# 3: low OCR confidence -> OCR_PLUS_VISION (per spec §3.2 and
# router Tier 4). The router fires OCR_PLUS_VISION when OCR text
# is sufficient AND confidence is below threshold AND there is no
# visual intent. VISION_FALLBACK is reserved for OCR failed /
# disabled / pending (Tier 1) or text_len < threshold AND visual
# intent (Tier 2).
d = decide_processing_mode("What text does this contain?",
    ocr_text="Lorem ipsum dolor sit amet consectetur adipiscing",
    ocr_confidence=20, ocr_status="success", vision_enabled=True)
check("3_low_confidence_routes_to_vision",
      d.processing_mode == ProcessingMode.OCR_PLUS_VISION
      and RoutingSignal.LOW_OCR_CONFIDENCE in d.trigger_reasons
      and d.vision_required is True,
      f"mode={d.processing_mode} reasons={d.trigger_reasons}")

# 3b: OCR failed -> VISION_FALLBACK (the genuine Tier-1 fallback)
d = decide_processing_mode("Describe the screen",
    ocr_text="", ocr_status="failed", ocr_error="tesseract crashed",
    vision_enabled=True)
check("3b_ocr_failed_is_vision_fallback",
      d.processing_mode == ProcessingMode.VISION_FALLBACK
      and RoutingSignal.OCR_ERROR in d.trigger_reasons,
      f"mode={d.processing_mode} reasons={d.trigger_reasons}")

# 4: very low OCR text + visual intent -> VISION_FALLBACK
d = decide_processing_mode("Describe this diagram",
    ocr_text="", ocr_confidence=None, ocr_status="success",
    vision_enabled=True, vision_min_ocr_text_length=25)
check("4_low_text_with_visual_intent_vision_fallback",
      d.processing_mode == ProcessingMode.VISION_FALLBACK
      and RoutingSignal.LOW_OCR_TEXT_LENGTH in d.trigger_reasons,
      f"mode={d.processing_mode} reasons={d.trigger_reasons}")

# 4b: empty OCR + identifier lookup stays OCR_ONLY
d = decide_processing_mode("What error code is shown?",
    ocr_text="", ocr_confidence=None, ocr_status="success",
    vision_enabled=True, vision_min_ocr_text_length=25)
check("4b_empty_ocr_identifier_stays_ocr_only",
      d.processing_mode == ProcessingMode.OCR_ONLY
      and not d.vision_required,
      f"mode={d.processing_mode}")

# 5: "Which button should I click?" -> OCR_PLUS_VISION
# OCR text must be > VISION_MIN_OCR_TEXT_LENGTH (25) so the router
# reaches the VISUAL_INTENT check instead of short-circuiting on
# LOW_OCR_TEXT_LENGTH.
d = decide_processing_mode("Which button should I click to continue?",
    ocr_text="OK  Cancel  Apply  Save  Discard  Help  About",
    ocr_confidence=95, ocr_status="success", vision_enabled=True)
check("5_which_button_invokes_vision",
      d.processing_mode == ProcessingMode.OCR_PLUS_VISION,
      f"mode={d.processing_mode}")

# 6: "Which server has the red indicator?" -> OCR_PLUS_VISION
d = decide_processing_mode("Which server has the red indicator?",
    ocr_text="Server A Server B Server C", ocr_confidence=95,
    ocr_status="success", vision_enabled=True)
check("6_red_indicator_invokes_vision",
      d.processing_mode == ProcessingMode.OCR_PLUS_VISION,
      f"mode={d.processing_mode}")

# 7: identifier lookup never invokes Vision
d = decide_processing_mode("What error code is shown here?",
    ocr_text="902 - rejected", ocr_confidence=95,
    ocr_status="success", vision_enabled=True)
check("7_identifier_query_no_vision",
      d.processing_mode == ProcessingMode.OCR_ONLY
      and not d.vision_required,
      f"mode={d.processing_mode}")

# 8: VISION_ENABLED=false -> OCR_ONLY even for strong visual signals
d = decide_processing_mode("Which server has the red indicator?",
    ocr_text="Server A Server B", ocr_confidence=20,
    ocr_status="success", vision_enabled=False)
check("8_vision_disabled_stays_ocr_only",
      d.processing_mode == ProcessingMode.OCR_ONLY
      and RoutingSignal.VISION_DISABLED in d.trigger_reasons,
      f"mode={d.processing_mode} reasons={d.trigger_reasons}")

# 9: provider failure -> covered by orchestrator test below

# 10: high OCR confidence does NOT override strong visual intent
for q in ["What does this dashboard show?",
          "Explain this architecture diagram",
          "Describe the screenshot",
          "Where is the warning?",
          "What's wrong with this screen?",
          "Describe this image"]:
    d = decide_processing_mode(q,
        ocr_text="Some high confidence OCR text " * 4,
        ocr_confidence=98, ocr_status="success", vision_enabled=True)
    check("10_high_conf_no_override_%s" % q[:25],
          d.processing_mode == ProcessingMode.OCR_PLUS_VISION,
          f"q={q!r} mode={d.processing_mode}")


# ---------------------------------------------------------------------------
# GROUNDING (spec scenarios 11-15)
# ---------------------------------------------------------------------------
print("\n=== GROUNDING (11-15) ===")

# 11: OCR identifies 902 -> answer from OCR
d = decide_processing_mode("What error code is shown here?",
    ocr_text="902 - Message delivery failed", ocr_confidence=98,
    ocr_status="success", vision_enabled=True)
ev = build_image_evidence(decision=d,
    ocr_text="902 - Message delivery failed", ocr_confidence=98,
    document_id=42, image_id=7, image_filename="902.png",
    source_type="image_ocr")
check("11_ocr_identifies_902",
      ev.vision_used is False and "902" in ev.ocr_text
      and ev.document_id == 42 and ev.image_id == 7,
      f"vision_used={ev.vision_used} ocr={ev.ocr_text[:50]!r}")

# 12: Vision observes red status indicator -> image-grounded allowed
vr = VisionResult(description="Dashboard with server table.",
    image_type="dashboard",
    visual_findings=["Server B has a red failure indicator while Servers A and C are green."],
    detected_entities=["Server A", "Server B", "Server C"],
    visual_states=["red-indicator-on-server-b"],
    tags=["dashboard", "alert"], confidence=82,
    provider="mock", model="mock-v1", processing_time_ms=15)
d = decide_processing_mode("Which server is unhealthy?",
    ocr_text="Server A Server B Server C", ocr_confidence=95,
    ocr_status="success", vision_enabled=True)
ev = build_image_evidence(decision=d, ocr_text="Server A Server B Server C",
    ocr_confidence=95, vision_result=vr)
check("12_vision_observes_red_indicator",
      ev.vision_used is True
      and "red failure indicator" in ev.visual_findings[0]
      and "Server B" in ev.detected_entities,
      f"findings={ev.visual_findings} entities={ev.detected_entities}")

# 13: Vision identifies error but KB lacks meaning -> no invention
vr = VisionResult(description="The number 902 is visible on screen.",
    image_type="screenshot", visual_findings=[],
    detected_entities=["902"], visual_states=[], tags=[],
    confidence=70, provider="mock", model="mock-v1",
    processing_time_ms=10)
d = decide_processing_mode("Describe the screen",
    ocr_text="Error 902", ocr_confidence=90,
    ocr_status="success", vision_enabled=True)
ev = build_image_evidence(decision=d, ocr_text="Error 902",
    ocr_confidence=90, vision_result=vr)
section = ev.to_prompt_section().lower()
check("13_no_invention_for_902_meaning",
      ev.vision_used is True
      and "902" in ev.detected_entities
      and "definition" not in section
      and "troubleshoot" not in section,
      f"section={section[:200]!r}")

# 14: Vision identifies error but KB lacks troubleshooting -> no fix fabricated
vr = VisionResult(description="Screenshot showing a '902' code and 'Failed Reason: rejected'.",
    image_type="screenshot",
    visual_findings=["Status bar at top is red"],
    detected_entities=["902"], visual_states=["red-status-bar"],
    tags=["error"], confidence=80, provider="mock", model="mock-v1",
    processing_time_ms=12)
d = decide_processing_mode("Describe the screen",
    ocr_text="902 - rejected", ocr_confidence=92,
    ocr_status="success", vision_enabled=True)
ev = build_image_evidence(decision=d, ocr_text="902 - rejected",
    ocr_confidence=92, vision_result=vr)
section = ev.to_prompt_section().lower()
forbidden = [w for w in ("step 1:", "to fix:", "follow these steps", "resolution:")
             if w in section]
check("14_no_fabricated_remediation",
      len(forbidden) == 0,
      f"forbidden_found={forbidden} section_excerpt={section[:200]!r}")

# 15: Mixed image + KB answer has source separation
vr = VisionResult(description="Server B has a red failure indicator.",
    image_type="dashboard",
    visual_findings=["Server B is highlighted red"],
    detected_entities=["Server B"], visual_states=["red-indicator"],
    tags=["dashboard"], confidence=85,
    provider="mock", model="mock-v1", processing_time_ms=11)
d = decide_processing_mode("Describe the screen",
    ocr_text="Server A Server B Server C CPU 12% CPU 90% CPU 8%",
    ocr_confidence=94, ocr_status="success", vision_enabled=True)
ev = build_image_evidence(decision=d,
    ocr_text="Server A Server B Server C CPU 12% CPU 90% CPU 8%",
    ocr_confidence=94, document_id=10, image_id=20,
    image_filename="dashboard.png", source_type="image_ocr",
    vision_result=vr)
section = ev.to_prompt_section()
check("15_image_kb_source_separation",
      section.startswith("IMAGE EVIDENCE")
      and "Vision analysis" in section
      and "OCR text:" in section
      and "Server B" in section,
      f"section_excerpt={section[:300]!r}")


# ---------------------------------------------------------------------------
# CACHE / PERSISTENCE (spec scenarios 16-20)
# ---------------------------------------------------------------------------
print("\n=== CACHE / PERSISTENCE (16-20) ===")


class _FakeImage:
    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


class _FakeQuery:
    def __init__(self, img):
        self._img = img

    def filter(self, *a, **kw):
        return self

    def first(self):
        return self._img


class _FakeDB:
    def __init__(self, img):
        self._img = img
        self.added = []
        self.committed = 0

    def query(self, cls):
        return _FakeQuery(self._img)

    def add(self, obj):
        self.added.append(obj)

    def commit(self):
        self.committed += 1

    def rollback(self):
        pass


# 16: first Vision call stores result
img = _FakeImage(id=99, document_id=1, storage_key="k",
    mime_type="image/png", source_type="image_ocr",
    ocr_status="success", sequence_number=0)
fake_db = _FakeDB(img)
vr = VisionResult(description="First call description",
    image_type="dashboard", visual_findings=["finding one"],
    detected_entities=["Server A"], visual_states=["green-status"],
    tags=["dashboard"], confidence=80,
    provider="mock", model="mock-v1", processing_time_ms=10)
result = persist_vision_result(fake_db, document_image_id=99,
    vision_result=vr)
check("16_first_call_persists",
      result.cache_hit is False and img.vision_status == "success"
      and img.vision_description == "First call description"
      and img.vision_confidence == 80
      and img.vision_findings == ["finding one"]
      and img.vision_tags == ["dashboard"]
      and img.vision_cache_key
      and fake_db.committed == 1,
      f"status={img.vision_status} cache_key={img.vision_cache_key}")

# 17: second compatible query reuses stored result
cached = lookup_cached_vision(fake_db, document_image_id=99,
    provider="mock", model="mock-v1",
    schema_version=VISION_SCHEMA_VERSION)
check("17_second_compatible_query_hits_cache",
      cached.cache_hit is True
      and cached.vision_result is not None
      and cached.vision_result.description == "First call description",
      f"hit={cached.cache_hit} desc={cached.vision_result.description if cached.vision_result else None}")

# 18: no duplicate provider call when cache valid - covered separately

# 19: failed Vision call records safe failure state
img2 = _FakeImage(id=100, document_id=1, storage_key="k",
    mime_type="image/png", source_type="image_ocr",
    ocr_status="success", sequence_number=0)
fake_db2 = _FakeDB(img2)
result = persist_vision_failure(fake_db2, document_image_id=100,
    provider="mock", model="mock-v1", error="rate-limited")
check("19_failed_vision_recorded_safely",
      result.cache_hit is False and img2.vision_status == "failed"
      and "rate-limited" in (img2.vision_error or ""),
      f"status={img2.vision_status} err={img2.vision_error}")

# 20: existing OCR data preserved when Vision persists
img3 = _FakeImage(id=200, document_id=1, storage_key="k",
    mime_type="image/png", source_type="pdf_page_ocr",
    ocr_status="success", ocr_provider="tesseract",
    ocr_confidence=92, ocr_text_hash="abc123",
    ocr_error=None, width=1280, height=720, byte_size=123456,
    sequence_number=0)
fake_db3 = _FakeDB(img3)
persist_vision_result(fake_db3, document_image_id=200,
    vision_result=VisionResult(description="Vision desc",
        image_type="screenshot",
        visual_findings=["f"], detected_entities=["e"],
        visual_states=["s"], tags=["t"], confidence=80,
        provider="mock", model="mock-v1", processing_time_ms=10))
check("20_existing_ocr_data_preserved",
      img3.ocr_provider == "tesseract"
      and img3.ocr_confidence == 92
      and img3.ocr_text_hash == "abc123"
      and img3.ocr_status == "success"
      and img3.source_type == "pdf_page_ocr"
      and img3.width == 1280
      and img3.height == 720
      and img3.byte_size == 123456
      and img3.vision_status == "success"
      and img3.vision_description == "Vision desc",
      f"ocr_provider={img3.ocr_provider} vision_status={img3.vision_status}")


# ---------------------------------------------------------------------------
# ORCHESTRATOR + PROVIDER FAILURE
# ---------------------------------------------------------------------------
print("\n=== ORCHESTRATOR + PROVIDER FAILURE ===")

reset_vision_provider_cache()

# Provider failure -> OCR evidence preserved
class _BoomProvider:
    name = "mock"
    model = "mock-v1"

    def analyze_image(self, **kw):
        raise VisionProviderError("rate-limited")

from app.services.vision import factory as vf
vf._cached_provider["mock"] = _BoomProvider()
outcome = process_image_for_question(
    None,
    question="Which button should I click?",
    ocr_text="OK Cancel Apply", ocr_confidence=95,
    ocr_status="success",
    image_bytes=b"\x89PNG\r\n\x1a\n" + b"x" * 128,
)
check("orchestrator_provider_failure_degrades",
      outcome.evidence.vision_used is False
      and "rate-limited" in (outcome.vision_error or ""),
      f"vision_used={outcome.evidence.vision_used} err={outcome.vision_error}")

# VISION_ENABLED=false -> OCR_ONLY for visual question
reset_vision_provider_cache()
from app.core.config import settings
prev_enabled = settings.VISION_ENABLED
settings.VISION_ENABLED = False
outcome = process_image_for_question(
    None,
    question="Which button should I click?",
    ocr_text="OK Cancel Apply", ocr_confidence=95,
    ocr_status="success",
    image_bytes=b"\x89PNG\r\n\x1a\n" + b"x" * 128,
)
settings.VISION_ENABLED = prev_enabled
check("orchestrator_disabled_is_ocr_only",
      outcome.decision.processing_mode == ProcessingMode.OCR_ONLY
      and outcome.evidence.vision_used is False,
      f"mode={outcome.decision.processing_mode}")


# ---------------------------------------------------------------------------
# CACHE KEY DETERMINISM
# ---------------------------------------------------------------------------
print("\n=== CACHE KEY DETERMINISM ===")
k1 = build_cache_key(document_image_id=1, provider="mock", model="mock-v1")
k2 = build_cache_key(document_image_id=1, provider="mock", model="mock-v1")
check("cache_key_stable", k1 == k2, f"k1={k1} k2={k2}")
check("cache_key_diff_image_id",
      build_cache_key(document_image_id=2, provider="mock", model="mock-v1") != k1)
check("cache_key_diff_model",
      build_cache_key(document_image_id=1, provider="mock", model="mock-v2") != k1)
check("cache_key_diff_schema_version",
      build_cache_key(document_image_id=1, provider="mock",
                      model="mock-v1", schema_version=2) != k1)


# ---------------------------------------------------------------------------
# INTENT MATRIX
# ---------------------------------------------------------------------------
print("\n=== INTENT MATRIX ===")
no_vision = [
    "What error code is shown?",
    "What error code is shown here?",
    "What number is displayed?",
    "Read out the text in the screenshot",
    "What is the receiver name?",
    "What is the amount?",
    "What is the message?",
]
visual = [
    "Which button should I click?",
    "What does this graph show?",
    "Where is the warning?",
    "Describe this image",
    "Describe this screenshot",
    "Which server has the red indicator?",
    "Explain this dashboard",
    "What trend do you see?",
]
for q in no_vision:
    check("intent_no_vision_%s" % q[:30].replace(" ", "_"),
          detect_visual_intent(q) is False, q)
for q in visual:
    check("intent_visual_%s" % q[:30].replace(" ", "_"),
          detect_visual_intent(q) is True, q)


# ---------------------------------------------------------------------------
# EVIDENCE BOUNDS
# ---------------------------------------------------------------------------
print("\n=== EVIDENCE BOUNDS ===")
huge = VisionResult(description="x" * 50000, image_type="dashboard",
    visual_findings=["f" * 5000 for _ in range(50)],
    detected_entities=["e" * 5000 for _ in range(50)],
    visual_states=["s" for _ in range(50)],
    tags=["t" for _ in range(50)], confidence=200,
    provider="mock", model="mock-v1", processing_time_ms=10)
d = decide_processing_mode("Describe the screen",
    ocr_text="Some OCR text " * 100, ocr_confidence=150,
    ocr_status="success", vision_enabled=True)
ev = build_image_evidence(decision=d, ocr_text="x" * 5000,
    ocr_confidence=150, vision_result=huge)
section = ev.to_prompt_section()
check("evidence_bounds_length_capped",
      len(section) < 10000, f"section_len={len(section)}")
check("evidence_bounds_confidence_coerced_to_0_100",
      ev.vision_confidence == 100 and ev.ocr_confidence == 100,
      f"vision_conf={ev.vision_confidence} ocr_conf={ev.ocr_confidence}")
check("evidence_bounds_findings_capped",
      len(ev.visual_findings) <= 6,
      f"findings_count={len(ev.visual_findings)}")


# ---------------------------------------------------------------------------
# CONFIG PRESENCE
# ---------------------------------------------------------------------------
print("\n=== CONFIG PRESENCE ===")
required = [
    "VISION_ENABLED", "VISION_PROVIDER", "VISION_MODEL",
    "VISION_OCR_CONFIDENCE_THRESHOLD", "VISION_MIN_OCR_TEXT_LENGTH",
    "VISION_ROUTER_ENABLED", "VISION_TIMEOUT_SECONDS",
    "VISION_MAX_RETRIES", "VISION_MAX_IMAGE_BYTES",
    "VISION_CACHE_SCHEMA_VERSION",
]
for k in required:
    check("config_has_%s" % k, hasattr(settings, k))
check("config_no_vision_api_key_leak",
      not hasattr(settings, "VISION_API_KEY"))


# ---------------------------------------------------------------------------
# SUMMARY
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("FOCUSED PHASE 34B TEST SUMMARY")
print("=" * 70)
print("PASSED: %d" % PASSED)
print("FAILED: %d" % FAILED)
if FAILURES:
    print("\nFAILURES:")
    for name, detail in FAILURES:
        print("  - %s: %s" % (name, detail))
    sys.exit(1)
print("\nALL FOCUSED TESTS PASSED")
sys.exit(0)
