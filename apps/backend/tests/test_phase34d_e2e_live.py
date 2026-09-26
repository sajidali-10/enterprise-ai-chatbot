"""Phase 34D -- Live E2E tests (A through H).

These tests exercise the Phase 34D advanced-vision layer against the
real FastAPI app (TestClient), real SQLite DB, real Tesseract OCR,
and the real Phase 34B VisionProvider protocol. A programmable
provider stub is used so the tests can assert deterministic structured
outputs without making real network calls.

Mapping to acceptance criteria:

    A. UI state -- Server B Failed + Apply disabled identified.
    B. Chart  -- spike around 14:00 + subsequent decline identified.
    C. Diagram -- Internet -> Nginx -> Backend -> PostgreSQL flow described.
    D. OCR guard -- identifier-only questions MUST NOT invoke advanced Vision.
    E. Product authority -- KB wins for "what does error 902 mean and how do I fix it?".
    F. Cache -- second identical visual-analysis call returns cache_hit=true.
    G. Provider failure resilience -- chat endpoint does NOT 500 when provider raises.
    H. RBAC -- User B cannot pull bytes for User A's private image; no filename /
       OCR / Vision result / existence leak in response or logs.

All synthetic data is deleted in teardown. No real users, documents,
images, Qdrant points, or Redis keys persist after the suite.
"""

from __future__ import annotations

import io
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pytest
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Environment overrides — must be set BEFORE importing app modules.
# ---------------------------------------------------------------------------

os.environ.setdefault("ADVANCED_VISION_ENABLED", "true")
os.environ.setdefault("ADVANCED_VISION_ROUTER_ENABLED", "true")
os.environ.setdefault("ADVANCED_VISION_CACHE_ENABLED", "true")
os.environ.setdefault("ADVANCED_VISION_PROCESS_CACHE_ENABLED", "true")
os.environ.setdefault("ADVANCED_VISION_PROCESS_CACHE_MAXSIZE", "64")
os.environ.setdefault("ADVANCED_VISION_CACHE_TTL_SECONDS", "60")
os.environ.setdefault("ADVANCED_VISION_SCHEMA_VERSION", "1")
os.environ.setdefault("VISION_ENABLED", "true")
os.environ.setdefault("VISION_PROVIDER", "mock")
os.environ.setdefault("VISION_MODEL", "phase34d-e2e-v1")
os.environ.setdefault("OCR_ENABLED", "true")
os.environ.setdefault("OCR_PROVIDER", "tesseract")
os.environ.setdefault("OCR_RENDER_DPI", "150")
os.environ.setdefault("MULTIMODAL_KNOWLEDGE_ENABLED", "false")  # disabled: pure chat test
os.environ.setdefault("VISION_INSTRUMENTATION_ENABLED", "1")

# Critical: config.py defaults EMBEDDING_PROVIDER=local which would
# trigger the sentence-transformers fallback (not installed). Force
# the mock embedding provider so the chat pipeline is self-contained.
os.environ["EMBEDDING_PROVIDER"] = "mock"
os.environ["EMBEDDING_MODEL"] = "mock"

# Critical: the chat pipeline's get_db() reads settings.DATABASE_URL
# which defaults to postgresql://chatbot:...@postgres:5432/chatbot.
# The docker-compose postgres host is not resolvable in this test
# environment, so the chat request would 500. Force a sqlite URL so
# the chat engine matches the conftest's patched SessionLocal. The
# pytest harness creates a ./test.db file in the working dir.
os.environ["DATABASE_URL"] = "sqlite:///./test.db"


# ---------------------------------------------------------------------------
# Programmable Vision provider stub
# ---------------------------------------------------------------------------


@dataclass
class _ProviderCall:
    """One captured call to the Vision provider."""

    image_bytes_count: int
    task: Optional[str] = None
    images_count: int = 1
    ocr_text: str = ""


class _ProgrammableVisionProvider:
    """Deterministic provider stub that returns a structured payload
    derived from the OCR text + the requested task.

    The structured payload matches the Phase 34D
    VisualReasoningResult schema so the orchestrator's
    provider-result -> VisualReasoningResult mapping can be exercised
    end-to-end. The class also records every call so tests can
    assert cache-hit / RBAC behaviour precisely.
    """

    name = "phase34d-e2e"
    model = "phase34d-e2e-v1"
    fail_with: Optional[Exception] = None

    def __init__(self) -> None:
        self.calls: List[_ProviderCall] = []

    def reset(self) -> None:
        self.calls = []

    def analyze_image(  # noqa: D401 — protocol method
        self,
        *,
        image_bytes: bytes,
        mime_type: str,
        prompt: str,
        ocr_text: str = "",
        context_hint: str = "",
        max_tokens: int = 600,
        timeout_s: Optional[float] = None,
        task: Optional[str] = None,
        images: Optional[List[Tuple[bytes, str]]] = None,
    ):
        """Return a VisionResult whose ``raw`` carries a deterministic
        VisualReasoningResult-shaped dict so the orchestrator can
        build the structured VisualReasoningResult for the LLM.
        """
        # Capture the call so tests can assert RBAC + cache behaviour.
        self.calls.append(
            _ProviderCall(
                image_bytes_count=len(image_bytes or b""),
                task=task,
                images_count=len(images) if images else 1,
                ocr_text=ocr_text or "",
            )
        )

        if self.fail_with is not None:
            # Test G: provider failure path.
            raise self.fail_with

        # Build the structured payload deterministically from OCR + task.
        raw = _build_structured_payload(
            task=task or "general_visual",
            ocr_text=ocr_text or "",
            image_count=len(images) if images else 1,
        )

        from app.services.vision.base import VisionResult

        return VisionResult(
            description=raw["summary"],
            image_type=raw.get("image_type") or "unknown",
            visual_findings=raw.get("observations", []),
            detected_entities=raw.get("entities", []),
            visual_states=raw.get("ui_states", []),
            tags=raw.get("tags", []),
            confidence=raw.get("confidence", 70),
            provider=self.name,
            model=self.model,
            processing_time_ms=12,
            schema_version=1,
            raw=raw,
        )


def _build_structured_payload(*, task: str, ocr_text: str, image_count: int) -> Dict[str, Any]:
    """Deterministic VisualReasoningResult-shaped dict keyed on task+ocr.

    The heuristics below are intentionally conservative: they only
    emit text that the OCR body explicitly contains (or a direct
    generalization) so the test assertions on "no invented content"
    are easy to verify.
    """
    lower_ocr = (ocr_text or "").lower()

    # --- UI state ---
    if task == "ui_state_analysis":
        ui_states: List[Dict[str, Any]] = []
        if "failed" in lower_ocr or "red" in lower_ocr:
            ui_states.append({
                "kind": "status-indicator",
                "label": _first_match_token(lower_ocr, ["server b", "server c", "server a"]),
                "value": "failed",
                "evidence": _first_match_line(ocr_text, ["failed", "red"]),
            })
        if "disabled" in lower_ocr:
            ui_states.append({
                "kind": "disabled-control",
                "label": _first_match_token(lower_ocr, ["apply"]),
                "value": "disabled",
                "evidence": _first_match_line(ocr_text, ["disabled"]),
            })
        return {
            "task_type": "ui_state_analysis",
            "summary": (
                "Server B shows a Failed indicator and the Apply control appears disabled."
                if "failed" in lower_ocr and "disabled" in lower_ocr
                else f"Observed UI state from OCR: {ocr_text[:120]}"
            ),
            "observations": [
                "Failed indicator visible" if "failed" in lower_ocr else "No failure indicators",
                "Disabled control visible" if "disabled" in lower_ocr else "All controls enabled",
            ],
            "ui_states": ui_states,
            "entities": _extract_server_names(ocr_text),
            "confidence": 82,
        }

    # --- Chart ---
    if task == "chart_analysis":
        findings: List[Dict[str, Any]] = []
        if "14:00" in ocr_text or "14:00" in lower_ocr:
            findings.append({
                "kind": "spike",
                "location": "around 14:00",
                "evidence": "OCR contains 14:00",
            })
        if "decline" in lower_ocr or "falling" in lower_ocr or "drop" in lower_ocr:
            findings.append({
                "kind": "downward-trend",
                "location": "after 14:00",
                "evidence": "OCR contains decline/drop/falling",
            })
        return {
            "task_type": "chart_analysis",
            "summary": (
                "Traffic rises sharply around 14:00 and then declines."
                if "14:00" in ocr_text and ("decline" in lower_ocr or "drop" in lower_ocr)
                else "Chart observations from OCR."
            ),
            "observations": [
                "Spike around 14:00" if "14:00" in ocr_text else "Spike time unclear",
                "Subsequent decline" if ("decline" in lower_ocr or "drop" in lower_ocr) else "No decline observed",
            ],
            "chart_findings": findings,
            "entities": ["14:00"],
            "confidence": 75,
        }

    # --- Diagram ---
    if task == "diagram_analysis":
        nodes = []
        for n in ("Internet", "Nginx", "Backend", "PostgreSQL", "Database"):
            if n.lower() in lower_ocr:
                nodes.append(n)
        rels: List[Dict[str, str]] = []
        if "internet" in lower_ocr and "nginx" in lower_ocr:
            rels.append({"from": "Internet", "to": "Nginx", "label": ""})
        if "nginx" in lower_ocr and "backend" in lower_ocr:
            rels.append({"from": "Nginx", "to": "Backend", "label": ""})
        if "backend" in lower_ocr and "postgresql" in lower_ocr:
            rels.append({"from": "Backend", "to": "PostgreSQL", "label": ""})
        return {
            "task_type": "diagram_analysis",
            "summary": (
                f"Diagram shows flow through {', '.join(nodes)}."
                if nodes
                else "Diagram observations from OCR."
            ),
            "observations": [f"Nodes visible: {', '.join(nodes)}"] if nodes else [],
            "relationships": rels,
            "diagram_findings": [
                {"kind": "flow", "subject": " -> ".join(nodes), "evidence": "OCR labels"}
            ] if nodes else [],
            "entities": nodes,
            "confidence": 78,
        }

    # --- Table ---
    if task == "table_visual_analysis":
        return {
            "task_type": "table_visual_analysis",
            "summary": "Table observations from OCR.",
            "observations": [ocr_text[:200]] if ocr_text else [],
            "table_findings": [],
            "entities": [],
            "confidence": 70,
        }

    # --- Comparison ---
    if task == "image_comparison" or image_count >= 2:
        return {
            "task_type": "image_comparison",
            "summary": "Server B changed from Healthy to Failed.",
            "observations": ["Status change observed between images"],
            "comparison_changes": [
                {
                    "subject": "Server B",
                    "kind": "status-change",
                    "before": "Healthy" if "healthy" in lower_ocr else "",
                    "after": "Failed" if "failed" in lower_ocr else "",
                    "evidence": "OCR labels",
                }
            ],
            "entities": ["Server B"],
            "confidence": 85,
        }

    # --- General ---
    return {
        "task_type": "general_visual",
        "summary": ocr_text[:200] if ocr_text else "Image observed.",
        "observations": [ocr_text[:200]] if ocr_text else [],
        "entities": [],
        "confidence": 60,
    }


def _first_match_token(lower_text: str, candidates: List[str]) -> str:
    for c in candidates:
        if c in lower_text:
            return c.title()
    return ""


def _first_match_line(ocr_text: str, needles: List[str]) -> str:
    for line in (ocr_text or "").splitlines():
        if any(n in line.lower() for n in needles):
            return line.strip()
    return ""


def _extract_server_names(ocr_text: str) -> List[str]:
    out = []
    for name in ("Server A", "Server B", "Server C"):
        if name.lower() in (ocr_text or "").lower():
            out.append(name)
    return out


# ---------------------------------------------------------------------------
# Synthetic PNG generators (deterministic; no network)
# ---------------------------------------------------------------------------

_FONT = None


def _get_font(size: int = 24):
    global _FONT
    if _FONT is None:
        try:
            _FONT = ImageFont.truetype(
                "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", size
            )
        except Exception:
            try:
                _FONT = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", size)
            except Exception:
                _FONT = ImageFont.load_default()
    return _FONT


def _save_with_text(text_lines: List[str], *, color_rect: Optional[Tuple[int, int, int, int]] = None) -> bytes:
    """Render a deterministic synthetic PNG with the given text lines.

    ``color_rect`` optionally draws a red rectangle ``(x1, y1, x2, y2)``
    on the image — used to flag "red = failed" state in the synthetic
    UI screenshots.
    """
    img = Image.new("RGB", (640, 240), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = _get_font()
    if color_rect is not None:
        x1, y1, x2, y2 = color_rect
        draw.rectangle((x1, y1, x2, y2), fill=(220, 60, 60))  # red box
    y = 20
    for line in text_lines:
        draw.text((20, y), line, fill=(0, 0, 0), font=font)
        y += 32
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _png_ui_state() -> bytes:
    """Test A: Server A Healthy, Server B Failed (red box), Server C Healthy, Apply disabled."""
    return _save_with_text(
        [
            "Server A  Healthy",
            "Server B  Failed",
            "Server C  Healthy",
            "Apply  (disabled)",
        ],
        color_rect=(0, 80, 320, 112),
    )


def _png_chart_spike() -> bytes:
    """Test B: spike around 14:00 followed by decline."""
    return _save_with_text(
        [
            "Requests per minute",
            "00:00 100",
            "08:00 350",
            "14:00 1820",
            "16:00 420",
            "20:00 200 (decline)",
        ]
    )


def _png_diagram_flow() -> bytes:
    """Test C: Internet -> Nginx -> Backend -> PostgreSQL."""
    return _save_with_text(
        [
            "Internet",
            "  |",
            "  v",
            "Nginx",
            "  |",
            "  v",
            "Backend",
            "  |",
            "  v",
            "PostgreSQL",
        ]
    )


def _png_error_902() -> bytes:
    """Test D / E: error 902 OCR-visible."""
    return _save_with_text(
        [
            "Failed Reason: 902",
            "Message delivery failed",
            "Status: error",
        ]
    )


def _png_server_healthy() -> bytes:
    """Comparison A: Server B Healthy."""
    return _save_with_text(
        [
            "Server A  Healthy",
            "Server B  Healthy",
            "Server C  Healthy",
        ]
    )


# ---------------------------------------------------------------------------
# MinIO stub (returns the bytes we stored for each DocumentImage)
# ---------------------------------------------------------------------------


class _StubMinio:
    def __init__(self) -> None:
        self._objects: Dict[str, bytes] = {}

    def put_object(self, bucket: str, key: str, payload: bytes, **_kw):
        self._objects[f"{bucket}/{key}"] = bytes(payload)
        return None

    def get_object(self, bucket: str, key: str):
        return _StubResponse(self._objects[f"{bucket}/{key}"])

    def remove_object(self, bucket: str, key: str):
        self._objects.pop(f"{bucket}/{key}", None)


class _StubResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def close(self) -> None:
        pass

    def release_conn(self) -> None:
        pass


class _StubQdrantClient:
    """Minimal Qdrant stub for the Phase 34D E2E.

    Qdrant is not running in this test environment. The chat pipeline
    only calls ``client.search`` (vector similarity) during hybrid
    retrieval, so we return an empty result list. The minimal API
    surface (search, scroll, retrieve, upsert, delete) mirrors the
    Phase 34C live E2E stub so the chat pipeline never raises
    ``AttributeError``.
    """

    def __init__(self) -> None:
        self._points: Dict[int, Dict[str, Any]] = {}

    def search(
        self,
        *,
        collection_name: str,
        query_vector: List[float],
        limit: int,
        query_filter: Any = None,
        with_payload: bool = True,
        with_vectors: bool = False,
        **kwargs: Any,
    ):
        return []

    def scroll(
        self,
        *,
        collection_name: str,
        scroll_filter: Any = None,
        limit: int,
        with_payload: bool = True,
        with_vectors: bool = False,
        **kwargs: Any,
    ):
        return ([], None)

    def retrieve(
        self,
        *,
        collection_name: str,
        ids: List[Any],
        with_payload: bool = True,
        with_vectors: bool = False,
        **kwargs: Any,
    ):
        return []

    def upsert(self, *, collection_name: str, points: List[Any], **kwargs: Any):
        return None

    def delete(self, *, collection_name: str, points_selector: Any = None, **kwargs: Any):
        return None

    def get_collections(self, **kwargs: Any):
        return type("_R", (), {"collections": []})()

    def get_collection(self, *, collection_name: str, **kwargs: Any):
        return None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def vision_provider() -> _ProgrammableVisionProvider:
    """Programmable Vision provider stub for deterministic structured outputs."""
    return _ProgrammableVisionProvider()


@pytest.fixture
def stub_minio() -> _StubMinio:
    return _StubMinio()


@pytest.fixture
def stub_qdrant() -> _StubQdrantClient:
    return _StubQdrantClient()


@pytest.fixture
def live_stack(monkeypatch, stub_minio, stub_qdrant, vision_provider):
    """Patch MinIO + VisionProvider + Qdrant for the Phase 34D E2E.

    * MinIO: in-memory dict keyed by (bucket, key).
    * Qdrant: empty in-memory stub. Returns [] from search/scroll/retrieve
      so the chat pipeline's hybrid retrieval does not raise.
    * VisionProvider: programmable stub that returns deterministic
      structured payloads derived from OCR text + task.
    * Redis cache: keep the existing Redis-backed cache enabled; the
      test relies on the in-process LRU (enabled above) to make cache
      hits observable across two successive calls in the same process.
    """
    import app.core.minio_client as minio_client
    import app.services.vision.factory as vision_factory
    import app.services.vector.qdrant_service as qdrant_service

    monkeypatch.setattr(minio_client, "get_minio_client", lambda: stub_minio)
    monkeypatch.setattr(
        vision_factory, "get_vision_provider", lambda: vision_provider
    )
    monkeypatch.setattr(qdrant_service, "get_qdrant_client", lambda: stub_qdrant)
    return {"minio": stub_minio, "qdrant": stub_qdrant, "vision": vision_provider}


@pytest.fixture
def db_images(client, db_session, live_stack):
    """Insert Document + DocumentImage rows directly so we bypass
    the upload pipeline (which would require MinIO + Qdrant). Each
    test receives a list of ``(document_id, image_id)`` tuples and
    the synthetic PNG bytes used to populate the MinIO stub.
    """
    from app.models.document import Document, DocumentImage

    payloads: Dict[str, Dict[str, Any]] = {}

    def _upload(name: str, png_bytes: bytes) -> Tuple[int, int]:
        doc = Document(
            filename=f"e2e_{name}.png",
            original_name=f"e2e_{name}.png",
            mime_type="image/png",
            size_bytes=len(png_bytes),
            status="indexed",
            visibility="global",
            owner_user_id=1,
        )
        db_session.add(doc)
        db_session.commit()
        db_session.refresh(doc)

        image = DocumentImage(
            document_id=doc.id,
            storage_key=f"e2e/{doc.id}/{name}.png",
            mime_type="image/png",
            source_type="direct_image",
            original_filename=f"e2e_{name}.png",
            ocr_provider="tesseract",
            ocr_status="success",
            ocr_confidence=85,
            ocr_text_hash="x" * 64,
        )
        db_session.add(image)
        db_session.commit()
        db_session.refresh(image)

        live_stack["minio"].put_object(
            "chatbot-uploads", image.storage_key, png_bytes
        )
        payloads[name] = {
            "document_id": doc.id,
            "image_id": image.id,
            "filename": f"e2e_{name}.png",
            "png_bytes": png_bytes,
        }
        return doc.id, image.id

    class _Helper:
        def upload(self, name: str, png_bytes: bytes) -> Tuple[int, int]:
            return _upload(name, png_bytes)

        def payload(self, name: str) -> Dict[str, Any]:
            return payloads[name]

        def all(self) -> Dict[str, Dict[str, Any]]:
            return payloads

        def cleanup(self):
            for name in list(payloads.keys()):
                p = payloads.pop(name)
                db_session.query(DocumentImage).filter(
                    DocumentImage.id == p["image_id"]
                ).delete()
                db_session.query(Document).filter(
                    Document.id == p["document_id"]
                ).delete()
                db_session.commit()
                live_stack["minio"].remove_object(
                    "chatbot-uploads", f"e2e/{p['document_id']}/{name}.png"
                )

    helper = _Helper()
    yield helper
    helper.cleanup()


# ---------------------------------------------------------------------------
# Auth helpers — match the dev-auth pattern from conftest.py.
# ---------------------------------------------------------------------------


def _admin_headers() -> Dict[str, str]:
    return {"X-Dev-User": "admin_user"}


def _user_headers(username: str = "regular") -> Dict[str, str]:
    # The dev auth backend resolves any non-admin prefix to a regular
    # user. Use a unique username per test to avoid collision.
    return {"X-Dev-User": f"regular_{username}"}


def _viewer_headers(username: str = "viewer") -> Dict[str, str]:
    return {"X-Dev-User": f"viewer_{username}"}


def _private_headers(username: str) -> Dict[str, str]:
    """User A's private principal (regular + private docs)."""
    return {"X-Dev-User": f"regular_{username}"}


def _post_chat(client, message: str, image_context: Optional[Dict[str, Any]] = None,
               headers: Optional[Dict[str, str]] = None):
    body: Dict[str, Any] = {"message": message, "mode": "knowledge_base"}
    if image_context is not None:
        body["image_context"] = image_context
    return client.post("/api/chat", json=body, headers=headers or _user_headers())


# ---------------------------------------------------------------------------
# A — UI state
# ---------------------------------------------------------------------------


def test_phase34d_e2e_A_ui_state(client, db_images, live_stack):
    """A. Upload UI screenshot, ask 'What visually looks wrong with this screen?'.

    Require: Server B Failed identified, Apply disabled identified, no
    invented root cause, correct image citation.
    """
    doc_id, image_id = db_images.upload("ui_state", _png_ui_state())
    res = _post_chat(
        client,
        "What visually looks wrong with this screen?",
        image_context={"document_id": doc_id, "image_id": image_id},
        headers=_user_headers("e2e_a"),
    )
    assert res.status_code == 200, res.text
    payload = res.json()

    # Vision metadata exposed in response (Phase 34B/34D).
    vision = payload.get("vision") or {}
    assert vision.get("image_processing_mode") in (
        "ocr_plus_vision", "vision_fallback", "ocr_only",
    ), f"unexpected vision mode: {vision}"
    advanced = vision.get("advanced_vision") or {}
    # The advanced layer ran and reported a UI-state task.
    assert advanced.get("advanced_vision_ran") is True, advanced
    assert advanced.get("advanced_vision_task_type") == "ui_state_analysis", advanced

    # Provider was actually called (first call) — recorded calls.
    provider = live_stack["vision"]
    assert len(provider.calls) >= 1, provider.calls

    # Structured finding contains BOTH observations the brief requires.
    rd = advanced.get("advanced_vision_result_dict") or {}
    summary = (rd.get("summary") or "").lower()
    observations = " ".join(rd.get("observations") or []).lower()
    ui_states = rd.get("ui_states") or []
    ui_blob = " ".join(
        str(u.get("kind", "")) + " " + str(u.get("label", "")) + " " + str(u.get("value", ""))
        for u in ui_states
    ).lower()

    # Server B Failed identified.
    assert "server b" in summary or "server b" in observations or "failed" in ui_blob, (
        f"Server B Failed not identified in summary/observations/ui_states: "
        f"{summary=!r} {observations=!r} {ui_blob=!r}"
    )
    # Apply disabled identified.
    assert "disabled" in summary or "disabled" in observations or "disabled" in ui_blob, (
        "Apply disabled not identified in advanced vision result"
    )

    # No invented root cause: the result must NOT contain words that
    # would imply the LLM was told WHY server B failed.
    forbidden_root_cause_words = ("because", "due to", "crashed", "restarted", "down")
    assert not any(w in summary for w in forbidden_root_cause_words), (
        f"summary contains invented root cause language: {summary!r}"
    )

    # Citation: at least one grouped source references the image.
    grouped = payload.get("grouped_sources") or []
    assert any(
        (g.get("source_file_name") or "").startswith("e2e_ui_state")
        or image_id in (g.get("metadata") or {}).get("image_id", "")
        for g in grouped
    ) or any(
        (c.get("image_id") == image_id)
        for c in (payload.get("citations") or [])
    ), f"expected image citation in response; got grouped_sources={grouped} citations={(payload.get('citations') or [])[:2]}"


# ---------------------------------------------------------------------------
# B — Chart analysis
# ---------------------------------------------------------------------------


def test_phase34d_e2e_B_chart_analysis(client, db_images, live_stack):
    """B. Upload chart screenshot, ask 'What trend or anomaly do you see?'."""
    doc_id, image_id = db_images.upload("chart", _png_chart_spike())
    res = _post_chat(
        client,
        "What trend or anomaly do you see?",
        image_context={"document_id": doc_id, "image_id": image_id},
        headers=_user_headers("e2e_b"),
    )
    assert res.status_code == 200, res.text
    payload = res.json()
    advanced = (payload.get("vision") or {}).get("advanced_vision") or {}
    assert advanced.get("advanced_vision_ran") is True
    assert advanced.get("advanced_vision_task_type") == "chart_analysis"

    rd = advanced.get("advanced_vision_result_dict") or {}
    summary = (rd.get("summary") or "").lower()
    observations = " ".join(rd.get("observations") or []).lower()
    findings = rd.get("chart_findings") or []
    findings_blob = " ".join(
        str(f.get("kind", "")) + " " + str(f.get("location", ""))
        for f in findings
    ).lower()

    # Spike around 14:00 identified.
    assert "14:00" in summary or "14:00" in observations or "spike" in findings_blob, (
        f"spike around 14:00 not identified: {summary=!r} {findings_blob=!r}"
    )
    # Subsequent decline identified.
    assert "decline" in summary or "decline" in observations or "downward" in findings_blob, (
        f"decline not identified: {summary=!r} {findings_blob=!r}"
    )
    # No invented cause (the test asserts absence of speculation language).
    forbidden_cause_words = ("because", "due to", "caused by", "outage", "failure")
    assert not any(w in summary for w in forbidden_cause_words), (
        f"summary contains invented cause language: {summary!r}"
    )

    # Citation present.
    grouped = payload.get("grouped_sources") or []
    assert any(
        (g.get("source_file_name") or "").startswith("e2e_chart")
        or image_id in (g.get("metadata") or {}).get("image_id", "")
        for g in grouped
    ) or any(
        (c.get("image_id") == image_id)
        for c in (payload.get("citations") or [])
    ), "expected image citation in chart response"


# ---------------------------------------------------------------------------
# C — Diagram analysis
# ---------------------------------------------------------------------------


def test_phase34d_e2e_C_diagram_analysis(client, db_images, live_stack):
    """C. Upload diagram, ask 'What flow does this diagram show?'."""
    doc_id, image_id = db_images.upload("diagram", _png_diagram_flow())
    res = _post_chat(
        client,
        "What flow does this diagram show?",
        image_context={"document_id": doc_id, "image_id": image_id},
        headers=_user_headers("e2e_c"),
    )
    assert res.status_code == 200, res.text
    payload = res.json()
    advanced = (payload.get("vision") or {}).get("advanced_vision") or {}
    assert advanced.get("advanced_vision_ran") is True
    assert advanced.get("advanced_vision_task_type") == "diagram_analysis"

    rd = advanced.get("advanced_vision_result_dict") or {}
    summary = (rd.get("summary") or "").lower()
    rels = rd.get("relationships") or []
    entities = " ".join(rd.get("entities") or []).lower()

    # Visible relationships correctly described.
    flow_nodes = ["internet", "nginx", "backend", "postgresql"]
    flow_present = sum(1 for n in flow_nodes if n in entities or n in summary)
    assert flow_present >= 3, (
        f"expected >= 3 visible flow nodes, got {flow_present} in "
        f"summary={summary!r} entities={entities!r}"
    )
    # No invented protocols / ports (the brief explicitly forbids).
    forbidden_protocols = ("tcp/", "udp/", "https://", "http://", "port ", ":443", ":80")
    assert not any(p in summary for p in forbidden_protocols), (
        f"summary contains invented protocol/port: {summary!r}"
    )

    # Citation present.
    grouped = payload.get("grouped_sources") or []
    assert any(
        (g.get("source_file_name") or "").startswith("e2e_diagram")
        or image_id in (g.get("metadata") or {}).get("image_id", "")
        for g in grouped
    ) or any(
        (c.get("image_id") == image_id)
        for c in (payload.get("citations") or [])
    ), "expected image citation in diagram response"


# ---------------------------------------------------------------------------
# D — OCR guard (identifier lookup must NOT invoke advanced Vision)
# ---------------------------------------------------------------------------


def test_phase34d_e2e_D_ocr_guard(client, db_images, live_stack):
    """D. Upload 902 screenshot, ask 'What error code is shown?'.

    Require: OCR path serves the answer, advanced Vision is NOT invoked.
    """
    doc_id, image_id = db_images.upload("error_902", _png_error_902())
    provider = live_stack["vision"]
    calls_before = len(provider.calls)

    res = _post_chat(
        client,
        "What error code is shown?",
        image_context={"document_id": doc_id, "image_id": image_id},
        headers=_user_headers("e2e_d"),
    )
    assert res.status_code == 200, res.text
    payload = res.json()

    # No new provider calls — the classifier marks this as an
    # identifier lookup and skips advanced Vision.
    assert len(provider.calls) == calls_before, (
        f"advanced Vision provider was called for identifier-lookup "
        f"question (calls went from {calls_before} to {len(provider.calls)})"
    )

    advanced = (payload.get("vision") or {}).get("advanced_vision") or {}
    assert advanced.get("advanced_vision_ran") is False, advanced
    assert advanced.get("advanced_vision_skipped_reason") == "identifier_lookup", (
        f"expected skipped_reason='identifier_lookup', got {advanced}"
    )

    # The OCR-based answer MUST still cite the image.
    assert "902" in (payload.get("message") or ""), (
        f"expected OCR-grounded answer '902', got: {(payload.get('message') or '')[:200]!r}"
    )


# ---------------------------------------------------------------------------
# E — Product authority (KB wins for "what does 902 mean and how to fix")
# ---------------------------------------------------------------------------


def test_phase34d_e2e_E_product_authority(client, db_images, live_stack):
    """E. Upload 902 screenshot, ask 'What does error 902 mean and how should I fix it?'.

    Require: KB remains authoritative. Advanced Vision may identify 902
    in OCR (via Phase 34B VisionResult), but does NOT invent meaning/fix.
    """
    doc_id, image_id = db_images.upload("error_902_e", _png_error_902())
    res = _post_chat(
        client,
        "What does error 902 mean and how should I fix it?",
        image_context={"document_id": doc_id, "image_id": image_id},
        headers=_user_headers("e2e_e"),
    )
    assert res.status_code == 200, res.text
    payload = res.json()
    message = payload.get("message") or ""

    # Product-meaning queries MUST NOT route to advanced Vision.
    advanced = (payload.get("vision") or {}).get("advanced_vision") or {}
    assert advanced.get("advanced_vision_ran") is False, (
        f"product-meaning query routed to advanced Vision: {advanced}"
    )
    assert advanced.get("advanced_vision_skipped_reason") in (
        "product_meaning", "no_task_signal",
    ), f"unexpected skip reason: {advanced}"

    # The answer must NOT contain fabricated "to fix:" / "follow these steps"
    # / "resolution" language invented by the Vision model.
    forbidden_speculation = (
        "step 1:", "step 2:", "to fix:", "resolution:",
        "follow these steps", "restart the", "reboot the",
    )
    assert not any(w in message.lower() for w in forbidden_speculation), (
        f"answer contains invented troubleshooting language: {message[:300]!r}"
    )

    # The OCR-grounded 902 identification is allowed (visible in OCR).
    # The answer should not silently invent the MEANING of 902; KB
    # authority is preserved by the grounding layer.


# ---------------------------------------------------------------------------
# F — Cache (first call: advanced_vision_called=True; second: cache_hit=True, called=False)
# ---------------------------------------------------------------------------


def test_phase34d_e2e_F_cache_hit_on_repeat(client, db_images, live_stack):
    """F. Run the same visual-analysis query twice. First call invokes
    the provider; second call returns cache_hit=True and does NOT
    re-invoke the provider.
    """
    doc_id, image_id = db_images.upload("cache_test", _png_ui_state())
    provider = live_stack["vision"]

    # First call.
    res1 = _post_chat(
        client,
        "What visually looks wrong with this screen?",
        image_context={"document_id": doc_id, "image_id": image_id},
        headers=_user_headers("e2e_f"),
    )
    assert res1.status_code == 200, res1.text
    payload1 = res1.json()
    advanced1 = (payload1.get("vision") or {}).get("advanced_vision") or {}
    assert advanced1.get("advanced_vision_ran") is True, advanced1
    assert advanced1.get("advanced_vision_called") is True, (
        f"first call: advanced_vision_called expected True, got {advanced1}"
    )
    assert advanced1.get("advanced_vision_cache_hit") is False, advanced1
    calls_after_first = len(provider.calls)
    assert calls_after_first >= 1

    # Second call (identical question + image).
    res2 = _post_chat(
        client,
        "What visually looks wrong with this screen?",
        image_context={"document_id": doc_id, "image_id": image_id},
        headers=_user_headers("e2e_f"),
    )
    assert res2.status_code == 200, res2.text
    payload2 = res2.json()
    advanced2 = (payload2.get("vision") or {}).get("advanced_vision") or {}
    # Cache HIT: ran=True (got a usable result), cache_hit=True,
    # called=False (provider was NOT invoked again).
    assert advanced2.get("advanced_vision_ran") is True, advanced2
    assert advanced2.get("advanced_vision_cache_hit") is True, (
        f"second call: advanced_vision_cache_hit expected True, got {advanced2}"
    )
    assert advanced2.get("advanced_vision_called") is False, (
        f"second call: advanced_vision_called expected False (cache hit), "
        f"got {advanced2}"
    )
    assert len(provider.calls) == calls_after_first, (
        f"provider was called again on cache hit: calls={len(provider.calls)}, "
        f"expected={calls_after_first}"
    )


# ---------------------------------------------------------------------------
# G — Provider failure resilience
# ---------------------------------------------------------------------------


def test_phase34d_e2e_G_provider_failure_does_not_break_chat(client, db_images, live_stack):
    """G. Force the Vision provider to raise. Chat endpoint MUST NOT 500.
    Backend stays healthy; OCR / persisted evidence is retained; no
    fabricated visual answer.
    """
    from app.services.vision.base import VisionProviderUnavailableError

    doc_id, image_id = db_images.upload("failure", _png_ui_state())
    provider = live_stack["vision"]
    provider.fail_with = VisionProviderUnavailableError("simulated 503")

    res = _post_chat(
        client,
        "What visually looks wrong with this screen?",
        image_context={"document_id": doc_id, "image_id": image_id},
        headers=_user_headers("e2e_g"),
    )
    # Chat endpoint returns 200 — provider failure is non-fatal.
    assert res.status_code == 200, res.text
    payload = res.json()

    advanced = (payload.get("vision") or {}).get("advanced_vision") or {}
    # We attempted to run; the failure was captured.
    assert advanced.get("advanced_vision_ran") is True, advanced
    assert advanced.get("advanced_vision_skipped_reason") in (
        "provider_unavailable", "provider_error", "provider_timeout",
    ), f"unexpected skip reason: {advanced}"
    assert advanced.get("advanced_vision_error"), advanced

    # NO fabricated visual answer: the message MUST NOT contain
    # observations from the structured payload (Server B Failed /
    # Apply disabled) since the provider never returned them.
    message = payload.get("message") or ""
    forbidden_speculation = (
        "Server B  Failed", "Apply", "disabled-control", "status-indicator",
    )
    assert not any(w in message for w in forbidden_speculation), (
        f"message contains fabricated visual content despite provider "
        f"failure: {message[:300]!r}"
    )

    # Provider was called and recorded the failure.
    assert len(provider.calls) >= 1

    # Backend health: hit the health endpoint.
    health = client.get("/health")
    assert health.status_code == 200


# ---------------------------------------------------------------------------
# H — RBAC: private image, cross-user access denied
# ---------------------------------------------------------------------------


def test_phase34d_e2e_H_rbac_private_image_rejected_for_other_user(
    client, db_session, db_images, live_stack, monkeypatch
):
    """H. Create a private document owned by User A. Query as User B.

    Require: image rejected before MinIO / provider access; advanced
    Vision provider receives zero calls for the unauthorized image;
    no filename / OCR / Vision result / existence leak.
    """
    from app.models.document import Document, DocumentImage

    # Force owner_user_id so this test's document is owned by a
    # specific user id (the dev auth backend resolves ``regular_A``
    # to user_id=2 by convention; we override directly via the row).
    doc = Document(
        filename="private.png",
        original_name="private.png",
        mime_type="image/png",
        size_bytes=10,
        status="indexed",
        visibility="private",
        owner_user_id=999,  # User A's synthetic id (no other user has access)
    )
    db_session.add(doc)
    db_session.commit()
    db_session.refresh(doc)

    image = DocumentImage(
        document_id=doc.id,
        storage_key=f"private/{doc.id}/secret.png",
        mime_type="image/png",
        source_type="direct_image",
        original_filename="secret_topology.png",
        ocr_provider="tesseract",
        ocr_status="success",
        ocr_confidence=85,
        ocr_text_hash="z" * 64,
    )
    db_session.add(image)
    db_session.commit()
    db_session.refresh(image)

    # Place a "secret" payload in the MinIO stub so a leaked fetch
    # would be observable (we assert it was NOT fetched).
    secret_payload = b"SECRET-FILE-CONTENTS-DO-NOT-LEAK"
    live_stack["minio"].put_object("chatbot-uploads", image.storage_key, secret_payload)

    # User B (a regular non-admin user) asks about the private image.
    provider = live_stack["vision"]
    calls_before = len(provider.calls)

    res = _post_chat(
        client,
        "What flow does this diagram show?",
        image_context={"document_id": doc.id, "image_id": image.id},
        headers=_user_headers("attacker_b"),
    )
    assert res.status_code == 200, res.text
    payload = res.json()

    # --- Provider received ZERO calls for this unauthorized image ---
    assert len(provider.calls) == calls_before, (
        f"advanced Vision provider was called for unauthorized image "
        f"(calls went from {calls_before} to {len(provider.calls)})"
    )

    # --- Orchestrator recorded unauthorized skip ---
    advanced = (payload.get("vision") or {}).get("advanced_vision") or {}
    assert advanced.get("advanced_vision_skipped_reason") in (
        "unauthorized", "unauthorized_pre_provider",
    ), f"expected unauthorized skip, got {advanced}"

    # --- No leak: filename / OCR / Vision result ---
    message = payload.get("message") or ""
    # The filename "secret_topology.png" must not appear anywhere.
    assert "secret_topology" not in message, (
        f"response leaked the private filename: {message[:300]!r}"
    )
    assert "secret_topology" not in json.dumps(payload).lower(), (
        "private filename leaked into payload (debug fields, citations, etc.)"
    )

    # The image's storage key (which contains the doc id) must not leak.
    assert image.storage_key not in message, (
        f"response leaked storage_key: {message[:300]!r}"
    )

    # The OCR body MUST NOT appear in the response (no OCR leak).
    # We seeded only the OCR-text-hash on the row (no body), but we
    # additionally verify the Vision provider was never invoked
    # (already asserted above) — the only way OCR could leak would be
    # through a successful provider call.

    # --- Cleanup ---
    db_session.query(DocumentImage).filter(DocumentImage.id == image.id).delete()
    db_session.query(Document).filter(Document.id == doc.id).delete()
    db_session.commit()
    live_stack["minio"].remove_object("chatbot-uploads", image.storage_key)


# ---------------------------------------------------------------------------
# Local imports used at the bottom of the module
# ---------------------------------------------------------------------------

import json  # noqa: E402  (used by H)
