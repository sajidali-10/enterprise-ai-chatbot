"""Phase 34D -- Live Docker-Compose-backed E2E tests (A through H).

Runs against the REAL backend stack (PostgreSQL, Redis, Qdrant, MinIO,
real FastAPI/chat path, real JWT auth/RBAC) inside the backend
container.

CRITICAL HARNESS NOTE: this file lives in ``live_e2e/`` (NOT
``tests/``) so that ``tests/conftest.py`` -- which forces the host-side
SQLite engine -- is never loaded. Running the associated app inside the
container therefore uses the REAL ``settings.DATABASE_URL``
(PostgreSQL), real Redis cache, real MinIO, and real Qdrant.

The only test-harness seam is the Vision provider: we inject a
deterministic provider stub (via monkeypatch on the Phase 34B factory)
so A-H assertions are reproducible without requiring a live Vision
model / API key. Every OTHER component (DB, Redis cache, MinIO upload,
Qdrant indexing, JWT auth, RBAC) is the real production implementation.

Environment variables are expected to be set when invoking pytest in
the container (see README / Makefile). The test also sets the Phase 34D
kill-switch ON so the layer is exercised end-to-end.

Required acceptance checks:
    A  UI-state analysis
    B  chart analysis
    C  diagram analysis
    D  OCR guard (identifier lookup does NOT invoke advanced Vision)
    E  KB / product authority
    F  real Redis cache (first call calls provider; second hits cache)
    G  provider failure resilience
    H  RBAC-before-image/provider access
"""

from __future__ import annotations

import io
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Environment overrides -- MUST be set before importing any app module.
# ---------------------------------------------------------------------------

os.environ.setdefault("ADVANCED_VISION_ENABLED", "true")
os.environ.setdefault("ADVANCED_VISION_ROUTER_ENABLED", "true")
os.environ.setdefault("ADVANCED_VISION_CACHE_ENABLED", "true")
os.environ.setdefault("ADVANCED_VISION_CACHE_TTL_SECONDS", "3600")
os.environ.setdefault("ADVANCED_VISION_SCHEMA_VERSION", "1")
os.environ.setdefault("VISION_ENABLED", "true")
os.environ.setdefault("VISION_PROVIDER", "mock")
os.environ.setdefault("VISION_MODEL", "phase34d-live-v1")
os.environ.setdefault("VISION_INSTRUMENTATION_ENABLED", "1")

# Deterministic TEXT LLM: force the project's existing mock text provider so
# answer generation is reproducible through the real /api/chat pipeline. This
# sets ONLY the text-LLM seam (``get_llm_provider`` reads ``LLM_PROVIDER`` at
# call time); the Phase 34D advanced-vision provider is a SEPARATE seam and is
# still patched to the deterministic provider by the ``client`` fixture. The
# mock text provider's RAG branch emits [N] citation markers deterministically.
os.environ["LLM_PROVIDER"] = "mock"

# Do NOT override DATABASE_URL -- the real postgres URL is used.
# Do NOT set EMBEDDING_PROVIDER=local off -- sentence-transformers is
# installed in the container image.

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app  # noqa: E402
from app.core.config import settings  # noqa: E402

# Force Phase 34D + Vision ON for the live run.
settings.ADVANCED_VISION_ENABLED = True
settings.ADVANCED_VISION_ROUTER_ENABLED = True
settings.ADVANCED_VISION_CACHE_ENABLED = True
settings.ADVANCED_VISION_CACHE_TTL_SECONDS = 3600
settings.VISION_ENABLED = True

# The real login/chat endpoints are protected by a rate limiter
# (a DoS guard orthogonal to the auth/RBAC logic being validated).
# Running 8 sequential E2E tests inside one process would otherwise
# exhaust the 60s window and every login/upload/chat call would return
# 429. Disabling the rate limiter for the live E2E session keeps the
# REAL JWT auth, RBAC, permissions, user-creation and profile paths
# intact while removing the flaky guard. This does NOT touch
# DATABASE_URL or any production DB initialization.
settings.RATE_LIMIT_ENABLED = False


# ---------------------------------------------------------------------------
# Programmable deterministic Vision provider stub (test seam only)
# ---------------------------------------------------------------------------


class _ProgrammableVisionProvider:
    """Deterministic provider returning a structured VisualReasoningResult
    derived from the OCR text + requested task, so A-H are reproducible.

    Records every call so F (cache) and H (RBAC) can assert on the
    exact number of provider invocations. The rest of the pipeline
    (cache, RBAC gate, MinIO fetch, chat path, LLM) is the real
    implementation.
    """

    name = "phase34d-live"
    model = "phase34d-live-v1"
    fail_with: Optional[Exception] = None

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def reset(self) -> None:
        self.calls = []

    def analyze_image(  # protocol
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
        self.calls.append({
            "task": task,
            "images_count": len(images) if images else 1,
            "ocr_text_len": len(ocr_text or ""),
        })
        if self.fail_with is not None:
            raise self.fail_with

        raw = _structured_payload(task or "general_visual", ocr_text or "", len(images) if images else 1)

        from app.services.vision.base import VisionResult
        return VisionResult(
            description=raw.get("summary") or "",
            image_type=raw.get("image_type") or "unknown",
            visual_findings=raw.get("observations") or [],
            detected_entities=raw.get("entities") or [],
            visual_states=raw.get("ui_states") or [],
            tags=raw.get("tags") or [],
            confidence=raw.get("confidence", 70),
            provider=self.name,
            model=self.model,
            processing_time_ms=12,
            schema_version=1,
            raw=raw,
        )


def _structured_payload(task: str, ocr_text: str, image_count: int) -> Dict[str, Any]:
    lower = (ocr_text or "").lower()
    if task == "ui_state_analysis":
        ui: List[Dict[str, Any]] = []
        if "failed" in lower or "red" in lower:
            ui.append({"kind": "status-indicator", "label": _token(lower, ["server b", "server c", "server a"]), "value": "failed", "evidence": _line(ocr_text, ["failed", "red"])})
        if "disabled" in lower or "apply" in lower:
            ui.append({"kind": "disabled-control", "label": "Apply", "value": "disabled", "evidence": _line(ocr_text, ["disabled", "apply"])})
        return {"task_type": "ui_state_analysis", "summary": "Server B shows a Failed indicator and Apply appears disabled.", "observations": ["Failed indicator visible", "Disabled control visible"], "ui_states": ui, "entities": _servers(ocr_text), "confidence": 82, "image_type": "ui"}
    if task == "chart_analysis":
        findings: List[Dict[str, Any]] = []
        if "14:00" in ocr_text:
            findings.append({"kind": "spike", "location": "around 14:00", "evidence": "OCR contains 14:00"})
        if "decline" in lower or "drop" in lower or "falling" in lower:
            findings.append({"kind": "downward-trend", "location": "after 14:00", "evidence": "OCR contains decline/drop/falling"})
        return {"task_type": "chart_analysis", "summary": "Traffic rises sharply around 14:00 and then declines.", "observations": ["Spike around 14:00", "Subsequent decline"], "chart_findings": findings, "entities": ["14:00"], "confidence": 75, "image_type": "chart"}
    if task == "diagram_analysis":
        nodes = [n for n in ("Internet", "Nginx", "Backend", "PostgreSQL") if n.lower() in lower]
        return {"task_type": "diagram_analysis", "summary": f"Diagram shows flow through {', '.join(nodes)}.", "observations": [f"Nodes visible: {', '.join(nodes)}"], "relationships": [{"from": "Internet", "to": "Nginx", "label": ""}, {"from": "Nginx", "to": "Backend", "label": ""}, {"from": "Backend", "to": "PostgreSQL", "label": ""}] if len(nodes) >= 4 else [], "diagram_findings": [{"kind": "flow", "subject": " -> ".join(nodes), "evidence": "OCR labels"}], "entities": nodes, "confidence": 78, "image_type": "diagram"}
    if task == "image_comparison" or image_count >= 2:
        return {"task_type": "image_comparison", "summary": "Server B changed from Healthy to Failed.", "observations": ["Status change observed"], "comparison_changes": [{"subject": "Server B", "kind": "status-change", "before": "Healthy", "after": "Failed", "evidence": "OCR labels"}], "entities": ["Server B"], "confidence": 85, "image_type": "ui"}
    return {"task_type": "general_visual", "summary": ocr_text[:200] or "Image observed.", "observations": [ocr_text[:200]] if ocr_text else [], "entities": [], "confidence": 60, "image_type": "unknown"}


def _token(lower_text: str, candidates: List[str]) -> str:
    for c in candidates:
        if c in lower_text:
            return c.title()
    return ""


def _line(ocr_text: str, needles: List[str]) -> str:
    for line in (ocr_text or "").splitlines():
        if any(n in line.lower() for n in needles):
            return line.strip()
    return ""


def _servers(ocr_text: str) -> List[str]:
    return [n for n in ("Server A", "Server B", "Server C") if n.lower() in (ocr_text or "").lower()]


# ---------------------------------------------------------------------------
# Synthetic PNG generators (deterministic; Pillow)
# ---------------------------------------------------------------------------


def _png_layout(
    text_lines: List[str],
    *,
    red_box: Optional[Tuple[int, int, int, int]] = None,
    font_size: int = 22,
) -> bytes:
    from PIL import Image, ImageDraw, ImageFont
    img = Image.new("RGB", (760, 300), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            font_size,
        )
    except Exception:
        font = ImageFont.load_default()
    if red_box:
        x1, y1, x2, y2 = red_box
        draw.rectangle((x1, y1, x2, y2), fill=(220, 60, 60))
    y = 16
    row_h = font_size + 12
    for line in text_lines:
        draw.text((20, y), line, fill=(0, 0, 0), font=font)
        y += row_h
    # Deja Vu outlines letters tightly; add padding rows so OCR can
    # resolve each label as its own line.
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _png_ui_state() -> bytes:
    return _png_layout(
        ["Server A  Healthy", "Server B  Failed", "Server C  Healthy", "Apply  (disabled)"],
        red_box=(0, 80, 320, 112),
        font_size=28,
    )


def _png_chart() -> bytes:
    return _png_layout(
        [
            "Requests per minute",
            "00:00 100",
            "08:00 350",
            "14:00 1820",
            "16:00 420",
            "20:00 200 (decline)",
        ],
        font_size=24,
    )


def _png_diagram() -> bytes:
    # Larger font + one node per line so Tesseract reliably captures ALL
    # four node labels (Internet / Nginx / Backend / PostgreSQL). The
    # prior small-font layout only OCR'd two labels, which starved the
    # diagram task of exact OCR ground truth.
    return _png_layout(
        ["Internet", "Nginx", "Backend", "PostgreSQL", "Flow: down"],
        font_size=34,
    )


def _png_error_902() -> bytes:
    return _png_layout(
        ["Failed Reason: 902", "Message delivery failed", "Status: error"]
    )


# ---------------------------------------------------------------------------
# Real-stack fixtures
# ---------------------------------------------------------------------------

# Unique run marker so every Phase 34D synthetic upload is identifiable by
# this E2E suite and can be deleted via the real lifecycle APIs without
# touching normal user documents.
import uuid as _uuid
_RUN_MARKER = "phase34d_e2e_" + _uuid.uuid4().hex[:8]

# recorded (doc_id) values created by this run — used for final cleanup.
_CREATED_DOC_IDS: List[Tuple[int, str]] = []


def _redis_client():
    import redis as redis_lib
    from app.core.config import settings as _s
    return redis_lib.Redis(
        host=_s.REDIS_HOST,
        port=int(_s.REDIS_PORT),
        socket_connect_timeout=2,
        socket_timeout=2,
        decode_responses=True,
    )


def _clear_phase34d_redis_keys():
    """Targeted Phase 34D test isolation.

    Deletes ONLY keys in the Phase 34D cache namespace (``advanced_vision:*``)
    so a test run starts from a clean cache without flushing the whole
    Redis database (which would clobber unrelated app state).
    """
    try:
        r = _redis_client()
        keys = list(r.keys("advanced_vision:*"))
        if keys:
            r.delete(*keys)
    except Exception:
        # Redis isolation is best-effort — a failure to connect just
        # leaves whatever cache entries exist (the cache degrades to
        # miss on outage anyway).
        return


def _delete_document_via_api(client: TestClient, token: str, doc_id: int):
    """Delete one synthetic document using the REAL lifecycle endpoint.

    ``DELETE /api/documents/{id}`` cascades PostgreSQL chunks/versions,
    Qdrant vectors, image_knowledge, and MinIO objects — the same real
    lifecycle path production uses. Best-effort; never raises.
    """
    try:
        res = client.delete(f"/api/documents/{int(doc_id)}", headers={"Authorization": f"Bearer {token}"})
        # 200/204/404 are acceptable (404 => already gone).
        if res.status_code not in (200, 204, 404):
            logger.debug("phase34d cleanup: delete doc %s status %s", doc_id, res.status_code)
    except Exception:
        # Best-effort cleanup — never break the test on cleanup failure.
        return


def _list_phase34d_stale_doc_ids(db=None) -> List[int]:
    """Return document IDs whose original_name carries this suite's run
    prefix (``phase34d_e2e_``). These are stale Phase 34D synthetic docs
    left behind by any prior/finished run; they are safe to remove via the
    real lifecycle API and never touch normal user documents.
    """
    try:
        from app.db.session import SessionLocal
        from app.models.document import Document
        session = db or SessionLocal()
        try:
            rows = session.query(Document.id).filter(
                Document.original_name.like("phase34d_e2e_%")
            ).all()
            return [int(r[0]) for r in rows]
        finally:
            if db is None:
                session.close()
    except Exception:
        return []


def _cleanup_phase34d_artifacts(client: Optional[TestClient] = None, *, delete_stale: bool = True):
    """Remove only this run's Phase 34D synthetic documents + Redis keys.

    * Every synthetic doc created by this run is deleted via the real
      ``DELETE /api/documents/{id}`` lifecycle endpoint.
    * When ``delete_stale`` is True (setup), also remove stale docs from
      prior runs whose filenames carry the ``phase34d_e2e_`` prefix.
    * Phase 34D Redis cache keys are removed (targeted namespace only).
    """
    _clear_phase34d_redis_keys()
    if client is None:
        return
    try:
        admin_token = _login(client, "phase34d_admin", E2E_PASSWORD)
    except Exception:
        return
    ids = list(_CREATED_DOC_IDS)
    _CREATED_DOC_IDS.clear()
    for doc_id, _token in ids:
        _delete_document_via_api(client, admin_token, doc_id)
    if delete_stale:
        for doc_id in _list_phase34d_stale_doc_ids():
            _delete_document_via_api(client, admin_token, doc_id)


@pytest.fixture(autouse=True)
def _phase34d_e2e_cleanup(client):
    """Per-test isolation + teardown for Phase 34D E2E synthetic artifacts.

    Before each test, remove stale Phase 34D synthetic docs from prior
    runs (via the real DELETE lifecycle API) and clear Phase 34D Redis
    keys. After each test, also clean up so a repeated or interrupted run
    does not accumulate synthetic images again.
    """
    _clear_phase34d_redis_keys()
    # Setup-time stale removal: only this suite's synthetic docs.
    stale = _list_phase34d_stale_doc_ids()
    if stale:
        try:
            admin_token = _login(client, "phase34d_admin", E2E_PASSWORD)
            for doc_id in stale:
                _delete_document_via_api(client, admin_token, doc_id)
        except Exception:
            pass  # best-effort setup cleanup
    yield
    _clear_phase34d_redis_keys()
    _cleanup_phase34d_artifacts(client, delete_stale=False)


@pytest.fixture(autouse=True)
def _phase34d_redis_isolation():
    """Run targeted Phase 34D Redis key cleanup before each test."""
    _clear_phase34d_redis_keys()
    yield
    _clear_phase34d_redis_keys()


@pytest.fixture
def vision_provider() -> _ProgrammableVisionProvider:
    return _ProgrammableVisionProvider()


@pytest.fixture
def client(monkeypatch, vision_provider):
    """Real TestClient (real app + real dependencies), with ONLY the
    Phase 34B Vision factory patched to the deterministic stub.

    Everything else (DB, Redis, MinIO, Qdrant, auth) is the real
    implementation.
    """
    import app.services.vision.factory as vision_factory
    monkeypatch.setattr(vision_factory, "get_vision_provider", lambda: vision_provider)
    # Use the real app as-is. The TestClient talks to the real
    # Postgres, Redis, MinIO, and Qdrant via the container network.
    return TestClient(app)


# ---------------------------------------------------------------------------
# Auth helpers -- real JWT via the real auth/RBAC path
#
# NOTE: we seed the admin + temp users DIRECTLY into the real PostgreSQL
# database (via the app's own SessionLocal + hash_password) because the
# .env bootstrap admin was never created in the running DB (the ``users``
# table has a different sysadmin named ``admin``). This mirrors the Phase
# 34C.1 live-E2E pattern (which seeded phase34c1_e2e_admin / user_a / user_b
# the same way) and keeps authentication 100% on the real JWT path.
# ---------------------------------------------------------------------------

E2E_PASSWORD = "Temp-Pass-123!"


def _seed_user(username: str, email: str, role: str, is_active: bool = True) -> Dict[str, Any]:
    """Idempotently upsert a user row into the real PostgreSQL DB.

    Uses the app's own SessionLocal + hash_password so the stored bcrypt
    hash is valid for the real /api/auth/login flow. Only ever touches the
    row identified by ``username`` (best-effort; never wipes other data).
    """
    from app.db.session import SessionLocal
    from app.security.models import User, UserRole
    from app.security.password import hash_password

    db = SessionLocal()
    try:
        user = db.query(User).filter(User.username == username).first()
        if user is None:
            user = User(
                username=username,
                email=email,
                full_name=username,
                hashed_password=hash_password(E2E_PASSWORD),
                role=getattr(UserRole, role.upper()) if hasattr(UserRole, role.upper()) else role,
                is_active=is_active,
                is_external=False,
            )
            db.add(user)
        else:
            user.is_active = is_active
            # Ensure the known password works regardless of prior state.
            user.hashed_password = hash_password(E2E_PASSWORD)
        db.commit()
        db.refresh(user)
        return {"id": user.id, "username": user.username, "email": user.email, "role": user.role}
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def _login(client: TestClient, username: str, password: str) -> str:
    res = client.post("/api/auth/login", json={"username_or_email": username, "password": password})
    assert res.status_code == 200, f"login failed for {username}: {res.status_code} {res.text[:300]}"
    data = res.json()
    token = data.get("access_token")
    assert token, f"no access_token in login response: {data}"
    return token


def _admin_token(client: TestClient) -> str:
    # Ensure the seeded admin exists BEFORE attempting login.
    _seed_user("phase34d_admin", "phase34d_admin@e2e.local", "admin", is_active=True)
    return _login(client, "phase34d_admin", E2E_PASSWORD)


def _create_temp_user(client: TestClient, admin_token: str, username: str, role: str = "user") -> Dict[str, Any]:
    """Seed a temp user directly into the real PostgreSQL DB."""
    return _seed_user(username, f"{username}@e2e.local", role, is_active=True)


def _upload_image(client: TestClient, token: str, png_bytes: bytes, filename: str, visibility: str = "global") -> Dict[str, Any]:
    """Upload a synthetic PNG via the real /api/documents/upload + OCR path.

    Every upload filename is prefixed with this run's unique ``_RUN_MARKER``
    so the artifact is identifiable by the E2E suite and can be removed via
    the real lifecycle API. The returned ``document_id`` is recorded into
    ``_CREATED_DOC_IDS`` for per-run teardown.
    """
    marked_filename = f"{_RUN_MARKER}_{filename}"
    files = {"file": (marked_filename, io.BytesIO(png_bytes), "image/png")}
    data = {"visibility": visibility}
    res = client.post(
        "/api/documents/upload",
        files=files,
        data=data,
        headers={"Authorization": f"Bearer {token}"},
    )
    assert res.status_code in (200, 201), f"upload failed: {res.status_code} {res.text[:400]}"
    payload = res.json()
    if payload.get("id"):
        _CREATED_DOC_IDS.append((int(payload["id"]), str(token)))
    return payload


def _chat(client: TestClient, token: str, message: str, image_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    body: Dict[str, Any] = {"message": message, "mode": "knowledge_base"}
    if image_context is not None:
        body["image_context"] = image_context
    res = client.post("/api/chat", json=body, headers={"Authorization": f"Bearer {token}"})
    assert res.status_code == 200, f"chat failed: {res.status_code} {res.text[:400]}"
    return res.json()


# ---------------------------------------------------------------------------
# A -- UI state
# ---------------------------------------------------------------------------


def test_live_e2e_A_ui_state(client, vision_provider):
    admin = _admin_token(client)
    user = _create_temp_user(client, admin, "e2e_a_user")
    utoken = _login(client, "e2e_a_user", "Temp-Pass-123!")

    up = _upload_image(client, utoken, _png_ui_state(), "e2e_ui_state.png", visibility="global")
    doc_id = up["id"]
    image_id = up.get("image_id")
    assert image_id, f"upload did not return image_id: {up}"

    payload = _chat(
        client,
        utoken,
        "What visually looks wrong with this screen?",
        {"document_id": doc_id, "image_id": image_id},
    )
    vision = payload.get("vision") or {}
    advanced = vision.get("advanced_vision") or {}
    assert advanced.get("advanced_vision_ran") is True, advanced
    assert advanced.get("advanced_vision_task_type") == "ui_state_analysis", advanced
    assert len(vision_provider.calls) >= 1

    rd = advanced.get("advanced_vision_result_dict") or {}
    summary = (rd.get("summary") or "").lower()
    obs = " ".join(rd.get("observations") or []).lower()
    assert "server b" in summary or "failed" in obs, f"Server B Failed not identified: {rd}"
    assert "disabled" in summary or "disabled" in obs, f"Apply disabled not identified: {rd}"
    forbidden = ("because", "due to", "crashed", "root cause")
    assert not any(w in summary for w in forbidden), f"invented root cause: {summary}"

    # Real image citation.
    #
    # The image's Qdrant chunk source_file_name is the parser-renamed
    # name (e.g. "Screenshot 2026-08-05 …png"), not the upload filename
    # "e2e_ui_state.png". The authoritative signal that the response
    # carried a correct image citation is the presence of the Phase 34D
    # Advanced Vision Evidence excerpt in grouped_sources (the synthetic
    # advanced-vision chunk is the image-grounded citation) OR an
    # explicit image_id match on the citations list.
    grouped = payload.get("grouped_sources") or []
    grouped_blob = json.dumps(grouped)
    cited = bool(
        "[Phase 34D Advanced Vision Evidence]" in grouped_blob
        or any(str(c.get("image_id")) == str(image_id) for c in (payload.get("citations") or []))
    )
    assert cited, f"expected image citation: grouped={grouped}"


# ---------------------------------------------------------------------------
# B -- chart analysis
# ---------------------------------------------------------------------------


def test_live_e2e_B_chart(client, vision_provider):
    admin = _admin_token(client)
    utoken = _login(client, _create_temp_user(client, admin, "e2e_b_user")["username"], "Temp-Pass-123!")
    up = _upload_image(client, utoken, _png_chart(), "e2e_chart.png")
    payload = _chat(client, utoken, "What trend or anomaly do you see?", {"document_id": up["id"], "image_id": up["image_id"]})
    advanced = (payload.get("vision") or {}).get("advanced_vision") or {}
    assert advanced.get("advanced_vision_task_type") == "chart_analysis", advanced
    rd = advanced.get("advanced_vision_result_dict") or {}
    summary = (rd.get("summary") or "").lower()
    obs = " ".join(rd.get("observations") or []).lower()
    assert "14:00" in summary or "spike" in obs, f"spike around 14:00 not identified: {rd}"
    assert "decline" in summary or "decline" in obs, f"decline not identified: {rd}"
    forbidden = ("because", "due to", "caused by", "outage")
    assert not any(w in summary for w in forbidden), f"invented cause: {summary}"


# ---------------------------------------------------------------------------
# C -- diagram analysis
# ---------------------------------------------------------------------------


def test_live_e2e_C_diagram(client, vision_provider):
    admin = _admin_token(client)
    utoken = _login(client, _create_temp_user(client, admin, "e2e_c_user")["username"], "Temp-Pass-123!")
    up = _upload_image(client, utoken, _png_diagram(), "e2e_diagram.png")
    payload = _chat(client, utoken, "What flow does this diagram show?", {"document_id": up["id"], "image_id": up["image_id"]})
    advanced = (payload.get("vision") or {}).get("advanced_vision") or {}
    assert advanced.get("advanced_vision_task_type") == "diagram_analysis", advanced
    rd = advanced.get("advanced_vision_result_dict") or {}
    summary = (rd.get("summary") or "").lower()
    entities = " ".join(rd.get("entities") or []).lower()
    flow = sum(1 for n in ("internet", "nginx", "backend", "postgresql") if n in entities or n in summary)
    assert flow >= 3, f"expected >=3 flow nodes: {rd}"
    forbidden = ("tcp/", "udp/", "https://", ":443", ":80", "port ")
    assert not any(p in summary for p in forbidden), f"invented protocol/port: {summary}"


# ---------------------------------------------------------------------------
# D -- OCR guard: identifier lookup does NOT invoke advanced Vision
# ---------------------------------------------------------------------------


def test_live_e2e_D_ocr_guard(client, vision_provider):
    admin = _admin_token(client)
    utoken = _login(client, _create_temp_user(client, admin, "e2e_d_user")["username"], "Temp-Pass-123!")
    up = _upload_image(client, utoken, _png_error_902(), "e2e_902.png")
    before = len(vision_provider.calls)
    payload = _chat(client, utoken, "What error code is shown?", {"document_id": up["id"], "image_id": up["image_id"]})
    assert len(vision_provider.calls) == before, f"advanced Vision called for identifier lookup: {vision_provider.calls}"
    advanced = (payload.get("vision") or {}).get("advanced_vision") or {}
    assert advanced.get("advanced_vision_ran") is False, advanced
    assert advanced.get("advanced_vision_skipped_reason") == "identifier_lookup", advanced
    assert "902" in (payload.get("message") or ""), f"expected OCR-grounded '902': {payload.get('message')[:200]}"


# ---------------------------------------------------------------------------
# E -- KB / product authority
# ---------------------------------------------------------------------------


def test_live_e2e_E_product_authority(client, vision_provider):
    admin = _admin_token(client)
    utoken = _login(client, _create_temp_user(client, admin, "e2e_e_user")["username"], "Temp-Pass-123!")
    up = _upload_image(client, utoken, _png_error_902(), "e2e_902_e.png")
    payload = _chat(client, utoken, "What does error 902 mean and how should I fix it?", {"document_id": up["id"], "image_id": up["image_id"]})
    advanced = (payload.get("vision") or {}).get("advanced_vision") or {}
    assert advanced.get("advanced_vision_ran") is False, advanced
    assert advanced.get("advanced_vision_skipped_reason") in ("product_meaning", "no_task_signal"), advanced
    message = (payload.get("message") or "").lower()
    forbidden = ("step 1:", "to fix:", "resolution:", "follow these steps")
    assert not any(w in message for w in forbidden), f"invented troubleshooting language: {payload.get('message')[:300]}"


# ---------------------------------------------------------------------------
# F -- real Redis cache: first call calls provider; second hits cache
# ---------------------------------------------------------------------------


def test_live_e2e_F_redis_cache(client, vision_provider):
    admin = _admin_token(client)
    utoken = _login(client, _create_temp_user(client, admin, "e2e_f_user")["username"], "Temp-Pass-123!")
    up = _upload_image(client, utoken, _png_ui_state(), "e2e_cache.png")
    q = "What visually looks wrong with this screen?"
    ctx = {"document_id": up["id"], "image_id": up["image_id"]}

    r1 = _chat(client, utoken, q, ctx)
    a1 = (r1.get("vision") or {}).get("advanced_vision") or {}
    assert a1.get("advanced_vision_ran") is True, a1
    assert len(vision_provider.calls) >= 1
    calls_after_first = len(vision_provider.calls)

    r2 = _chat(client, utoken, q, ctx)
    a2 = (r2.get("vision") or {}).get("advanced_vision") or {}
    assert a2.get("advanced_vision_ran") is True, a2
    assert a2.get("advanced_vision_cache_hit") is True, f"expected cache_hit on second call: {a2}"
    assert len(vision_provider.calls) == calls_after_first, (
        f"provider re-invoked on cache hit: calls={len(vision_provider.calls)}, expected={calls_after_first}"
    )


# ---------------------------------------------------------------------------
# G -- provider failure resilience
# ---------------------------------------------------------------------------


def test_live_e2e_G_provider_failure(client, vision_provider):
    from app.services.vision.base import VisionProviderUnavailableError
    admin = _admin_token(client)
    utoken = _login(client, _create_temp_user(client, admin, "e2e_g_user")["username"], "Temp-Pass-123!")
    up = _upload_image(client, utoken, _png_ui_state(), "e2e_fail.png")
    vision_provider.fail_with = VisionProviderUnavailableError("simulated 503")
    payload = _chat(client, utoken, "What visually looks wrong with this screen?", {"document_id": up["id"], "image_id": up["image_id"]})
    advanced = (payload.get("vision") or {}).get("advanced_vision") or {}
    assert advanced.get("advanced_vision_skipped_reason") in ("provider_unavailable", "provider_error", "provider_timeout"), advanced
    message = payload.get("message") or ""
    forbidden = ("Server B  Failed", "disabled-control", "status-indicator")
    assert not any(w in message for w in forbidden), f"fabricated visual content despite failure: {message[:300]}"
    health = client.get("/health")
    assert health.status_code == 200


# ---------------------------------------------------------------------------
# H -- RBAC: private image owned by User A; User B rejected before provider
# ---------------------------------------------------------------------------


def test_live_e2e_H_rbac(client, vision_provider):
    admin = _admin_token(client)
    user_a = _create_temp_user(client, admin, "e2e_user_a", role="user")
    user_b = _create_temp_user(client, admin, "e2e_user_b", role="user")
    atok = _login(client, user_a["username"], "Temp-Pass-123!")
    btok = _login(client, user_b["username"], "Temp-Pass-123!")

    # User A uploads a PRIVATE image.
    up = _upload_image(client, atok, _png_diagram(), "secret_topology.png", visibility="private")
    doc_id, image_id = up["id"], up["image_id"]

    before = len(vision_provider.calls)
    payload = _chat(client, btok, "What flow does this diagram show?", {"document_id": doc_id, "image_id": image_id})
    assert len(vision_provider.calls) == before, f"provider called for unauthorized image: {vision_provider.calls}"

    advanced = (payload.get("vision") or {}).get("advanced_vision") or {}
    assert advanced.get("advanced_vision_skipped_reason") in ("unauthorized", "unauthorized_pre_provider"), advanced

    # No leak of filename / OCR / existence.
    blob = json.dumps(payload)
    assert "secret_topology" not in blob.lower(), "private filename leaked in payload"
    message = payload.get("message") or ""
    assert "postgresql" not in message.lower() or "diagram" not in message.lower(), (
        f"OCR/existence may have leaked: {message[:300]}"
    )


# ---------------------------------------------------------------------------
# OpenRouter smoke (non-gating)
#
# This test verifies that the configured OpenRouter text-LLM endpoint
# responds successfully (HTTP request succeeds, answer is non-empty). It is
# NOT an acceptance gate for Phase 34D — the A–H suite runs with the
# deterministic mock text provider so an external free-model refusal or
# wording variance cannot mask a correct Phase 34D path. This smoke runs
# only when ``PHASE34D_OPENROUTER_SMOKE=1`` is set.
# ---------------------------------------------------------------------------


def test_phase34d_openrouter_smoke():
    import os as _os
    if _os.environ.get("PHASE34D_OPENROUTER_SMOKE") != "1":
        pytest.skip("OpenRouter smoke disabled (set PHASE34D_OPENROUTER_SMOKE=1 to run)")

    try:
        from app.services.llm import get_llm_provider
        provider = get_llm_provider()
    except Exception as exc:  # pragma: no cover - smoke only
        pytest.skip(f"OpenRouter provider not available: {exc}")

    # This test is only meaningful against the real openrouter provider, not
    # the mock. If the harness forced mock for A-H, temporarily re-read the
    # configured provider directly.
    if getattr(provider, "provider_name", "") == "mock":
        try:
            provider = _configured_openrouter_provider()
        except Exception as exc:  # pragma: no cover - smoke only
            pytest.skip(f"OpenRouter not configured: {exc}")

    resp = provider.chat(
        __import__("app.schemas.chat", fromlist=["ChatRequest"]).ChatRequest(
            message="Reply with the single word: ok"
        )
    )
    answer = str(getattr(resp, "message", "") or "")
    assert answer.strip(), "OpenRouter smoke: answer is empty"


def _configured_openrouter_provider():
    """Build the OpenRouter provider from env (bypassing the mock override).

    Throwaway helper used only by the non-gating smoke test so it can talk to
    the real OpenRouter endpoint without affecting the A–H mock path.
    """
    from app.services.llm.openrouter_provider import OpenRouterProvider
    return OpenRouterProvider()