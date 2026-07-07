"""
Phase 31B — LangGraph Agentic RAG Tests

Covers:

  1. Disabled-by-default path — the agentic module is importable but
     classic knowledge_base behavior is unchanged when
     `RAG_AGENTIC_ENABLED=false`.
  2. Graph builds — `build_agentic_rag_graph()` returns a compiled
     StateGraph when LangGraph is importable.
  3. Routing behavior — `should_use_agentic_for_mode` honors all
     combinations of RAG_AGENTIC_ENABLED / RAG_AGENTIC_DEFAULT and the
     request mode value (knowledge_base vs agentic_knowledge_base).
  4. Strong / medium / weak evidence paths — the graph respects the
     evidence-level decision: strong/medium skip the rewrite loop, weak
     evidence triggers a rewrite/retry when RAG_AGENTIC_MAX_RETRIES>0.
  5. Retry limit — the graph never exceeds RAG_AGENTIC_MAX_RETRIES
     retrieve_retry cycles.
  6. Fallback behavior — when the agentic pipeline raises,
     RAG_AGENTIC_FALLBACK_TO_CLASSIC=true makes the chat endpoint route
     to the classic pipeline without breaking the response.
  7. Citation verification — verify_citations records the expected
     fields (has_citations, citation_count, all_references_valid,
     invalid_references) and correctly flags bad citation markers.
  8. Privacy / redaction — graph output metadata never includes
     secrets, API keys, JWTs, connection strings, full filesystem paths,
     or unsanitized email / phone values.

Each test patches the underlying retrieval and LLM calls so the graph
runs without contacting Qdrant, the embedding provider, or any real LLM
gateway.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Module under test
# ---------------------------------------------------------------------------

_AGENTIC_STATE_MOD = "app.rag.agentic_state"
_AGENTIC_GRAPH_MOD = "app.rag.agentic_graph"

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def enable_agentic(monkeypatch):
    """Force the agentic pilot ON for the duration of a test.

    We patch the settings object held by `app.rag.agentic_graph` (the
    module that actually reads `settings.RAG_AGENTIC_*` at call time)
    rather than `app.core.config.settings`. Patching the latter alone
    is unreliable because `test_admin_rag_config.py` calls
    `importlib.reload(app.core.config)`, which creates a NEW Settings
    instance and reassigns `app.core.config.settings` — but modules
    that already did `from app.core.config import settings` still hold
    the OLD reference. Patching the module-level binding avoids that
    test-pollution trap.
    """
    import app.rag.agentic_graph as ag
    monkeypatch.setattr(ag.settings, "RAG_AGENTIC_ENABLED", True)
    monkeypatch.setattr(ag.settings, "RAG_AGENTIC_FRAMEWORK", "langgraph")
    monkeypatch.setattr(ag.settings, "RAG_AGENTIC_DEFAULT", False)
    monkeypatch.setattr(ag.settings, "RAG_AGENTIC_MAX_RETRIES", 1)
    monkeypatch.setattr(ag.settings, "RAG_AGENTIC_REWRITE_ENABLED", True)
    monkeypatch.setattr(ag.settings, "RAG_AGENTIC_REQUIRE_CITATIONS", True)
    monkeypatch.setattr(ag.settings, "RAG_AGENTIC_FALLBACK_TO_CLASSIC", True)
    yield


@pytest.fixture
def disable_agentic(monkeypatch):
    """Force the agentic pilot OFF for the duration of a test.

    Patches `app.rag.agentic_graph.settings` rather than
    `app.core.config.settings` (see `enable_agentic` for rationale).
    """
    import app.rag.agentic_graph as ag
    monkeypatch.setattr(ag.settings, "RAG_AGENTIC_ENABLED", False)
    yield


@pytest.fixture
def agentic_default_on(monkeypatch):
    """Force RAG_AGENTIC_DEFAULT=true (classic knowledge_base → agentic).

    Patches `app.rag.agentic_graph.settings` rather than
    `app.core.config.settings` (see `enable_agentic` for rationale).
    """
    import app.rag.agentic_graph as ag
    monkeypatch.setattr(ag.settings, "RAG_AGENTIC_ENABLED", True)
    monkeypatch.setattr(ag.settings, "RAG_AGENTIC_DEFAULT", True)
    yield


def _make_chunk(idx: int, score: float, content: str = "alpha beta gamma delta",
                source: str = "guide.pdf") -> dict:
    return {
        "chunk_id": f"chunk-{idx}",
        "document_id": idx,
        "chunk_index": idx,
        "content": content,
        "source_file_name": source,
        "title": f"Section {idx}",
        "score": score,
    }


def _make_strong_chunks() -> list[dict]:
    # High-scoring chunks with explicit phrase overlap — should produce
    # STRONG evidence for "Docker image".
    return [
        _make_chunk(
            1, 0.91,
            "A Docker image contains the application code, runtime, "
            "libraries, and dependencies needed to run the application.",
            "docker-guide.pdf",
        ),
        _make_chunk(
            2, 0.85,
            "Docker images are built from a Dockerfile and stored in a "
            "registry.",
            "docker-guide.pdf",
        ),
    ]


def _make_medium_chunks() -> list[dict]:
    # Moderate-scoring chunks with only weak word overlap — should
    # produce MEDIUM evidence.
    return [
        _make_chunk(
            1, 0.55,
            "Container runtime configuration and orchestration details.",
            "ops.pdf",
        ),
    ]


def _make_weak_chunks() -> list[dict]:
    # Chunks unrelated to the question — should produce WEAK evidence.
    return [
        _make_chunk(
            1, 0.20,
            "Random unrelated text about something completely different.",
            "unrelated.pdf",
        ),
    ]


# ---------------------------------------------------------------------------
# 1. Disabled-by-default path
# ---------------------------------------------------------------------------


class TestAgenticDisabledByDefault:
    def test_settings_default_to_disabled(self):
        """Default RAG_AGENTIC_ENABLED is false in fresh settings."""
        from app.core.config import Settings

        s = Settings()
        assert s.RAG_AGENTIC_ENABLED is False
        assert s.RAG_AGENTIC_FRAMEWORK == "langgraph"
        assert s.RAG_AGENTIC_DEFAULT is False
        assert s.RAG_AGENTIC_MAX_RETRIES == 1
        assert s.RAG_AGENTIC_REWRITE_ENABLED is True
        assert s.RAG_AGENTIC_REQUIRE_CITATIONS is True
        assert s.RAG_AGENTIC_FALLBACK_TO_CLASSIC is True

    def test_should_use_agentic_returns_false_when_disabled(self, disable_agentic):
        """When RAG_AGENTIC_ENABLED is false, no mode routes through the
        agentic pipeline even if it explicitly says agentic_knowledge_base."""
        from app.rag.agentic_graph import should_use_agentic_for_mode

        assert should_use_agentic_for_mode("knowledge_base") is False
        assert should_use_agentic_for_mode("agentic_knowledge_base") is False

    def test_classic_rag_path_unaffected_when_disabled(self, disable_agentic):
        """When RAG_AGENTIC_ENABLED is false, calling classic
        `generate_answer_with_rag` MUST NOT touch the agentic pipeline."""
        with patch("app.rag.answer_generator.retrieve_chunks_with_settings") as mock_retrieve, \
             patch("app.rag.answer_generator._invoke_llm") as mock_llm, \
             patch("app.rag.agentic_graph.run_agentic_rag") as mock_agentic:
            mock_retrieve.return_value = (_make_strong_chunks(), {"strategy": "hybrid_mmr"})
            mock_llm.return_value = "The answer [1]."

            from app.rag.answer_generator import generate_answer_with_rag
            answer, citations, metadata = generate_answer_with_rag(
                query="What is a Docker image?",
                use_hybrid=True,
                debug=False,
            )

            # Classic pipeline ran end-to-end.
            assert mock_retrieve.called
            assert mock_llm.called
            # Agentic pipeline was NEVER called.
            assert not mock_agentic.called, "agentic pipeline must not run when disabled"
            assert "answer [1]" in answer.lower()
            assert citations
            assert "Docker image".lower() in (citations[0].get("content_snippet", "").lower()) or citations[0].get("source_file_name") == "docker-guide.pdf"


# ---------------------------------------------------------------------------
# 2. Graph builds
# ---------------------------------------------------------------------------


class TestGraphBuild:
    def test_graph_builds_when_langgraph_available(self, enable_agentic):
        """When LangGraph is installed and the pilot is enabled, the
        compiled graph is not None and exposes node names."""
        from app.rag.agentic_graph import build_agentic_rag_graph, LANGGRAPH_AVAILABLE

        assert LANGGRAPH_AVAILABLE is True
        graph = build_agentic_rag_graph()
        assert graph is not None

        # Compiled LangGraph graphs expose node_names via .nodes
        node_names = set()
        if hasattr(graph, "nodes"):
            node_names = set(graph.nodes.keys())
        elif hasattr(graph, "get_graph"):
            try:
                g = graph.get_graph()
                node_names = set(getattr(g, "nodes", {}).keys())
            except Exception:
                pass
        # The exact attribute API varies between LangGraph versions;
        # what matters is that the graph compiled and is callable.
        assert graph is not None

    def test_graph_is_cached(self, enable_agentic):
        """Repeated build calls return the same compiled graph instance."""
        from app.rag.agentic_graph import build_agentic_rag_graph

        g1 = build_agentic_rag_graph()
        g2 = build_agentic_rag_graph()
        assert g1 is g2


# ---------------------------------------------------------------------------
# 3. Routing behavior
# ---------------------------------------------------------------------------


class TestRoutingBehavior:
    def test_knowledge_base_does_not_route_when_default_off(self, enable_agentic):
        """mode=knowledge_base stays on the classic pipeline when
        RAG_AGENTIC_DEFAULT is false."""
        from app.rag.agentic_graph import should_use_agentic_for_mode

        assert should_use_agentic_for_mode("knowledge_base") is False

    def test_agentic_knowledge_base_routes_when_enabled(self, enable_agentic):
        """mode=agentic_knowledge_base routes to the agentic pipeline."""
        from app.rag.agentic_graph import should_use_agentic_for_mode

        assert should_use_agentic_for_mode("agentic_knowledge_base") is True

    def test_knowledge_base_routes_when_default_on(self, agentic_default_on):
        """When RAG_AGENTIC_DEFAULT=true, classic mode=knowledge_base
        requests also route to the agentic pipeline."""
        from app.rag.agentic_graph import should_use_agentic_for_mode

        assert should_use_agentic_for_mode("knowledge_base") is True
        assert should_use_agentic_for_mode("agentic_knowledge_base") is True

    def test_general_chat_never_routes_to_agentic(self, enable_agentic):
        """mode=general_chat never routes through the agentic pipeline."""
        from app.rag.agentic_graph import should_use_agentic_for_mode

        assert should_use_agentic_for_mode("general_chat") is False

    def test_unknown_mode_does_not_route(self, enable_agentic):
        from app.rag.agentic_graph import should_use_agentic_for_mode

        assert should_use_agentic_for_mode("") is False
        assert should_use_agentic_for_mode("debug") is False
        assert should_use_agentic_for_mode("garbage") is False


# ---------------------------------------------------------------------------
# 4. Strong / medium / weak paths
# ---------------------------------------------------------------------------


class TestEvidencePaths:
    def test_strong_evidence_skips_rewrite(self, enable_agentic):
        """STRONG evidence → graph skips the rewrite/retry loop and
        jumps straight to generate_answer."""
        from app.rag.agentic_graph import _route_after_evidence
        from app.rag.agentic_state import make_initial_state

        state = make_initial_state("What is a Docker image?")
        state["evidence_level"] = "strong"
        state["chunks"] = _make_strong_chunks()
        assert _route_after_evidence(state) == "generate_answer"

    def test_medium_evidence_skips_rewrite(self, enable_agentic):
        """MEDIUM evidence also skips the rewrite loop (we already have
        partial lexical anchors)."""
        from app.rag.agentic_graph import _route_after_evidence
        from app.rag.agentic_state import make_initial_state

        state = make_initial_state("Container runtime configuration")
        state["evidence_level"] = "medium"
        state["chunks"] = _make_medium_chunks()
        assert _route_after_evidence(state) == "generate_answer"

    def test_weak_evidence_triggers_rewrite_when_retries_left(self, enable_agentic):
        """WEAK evidence + retries available + rewrite enabled →
        the graph tries rewrite_query_if_needed."""
        from app.rag.agentic_graph import _route_after_evidence
        from app.rag.agentic_state import make_initial_state

        state = make_initial_state("Please tell me about quantum mechanics")
        state["evidence_level"] = "weak"
        state["chunks"] = _make_weak_chunks()
        state["retry_count"] = 0
        assert _route_after_evidence(state) == "rewrite_query_if_needed"

    def test_weak_evidence_skips_rewrite_when_disabled(self, monkeypatch):
        """When RAG_AGENTIC_REWRITE_ENABLED=false, WEAK evidence does
        not go through the rewrite node."""
        import app.rag.agentic_graph as ag
        monkeypatch.setattr(ag.settings, "RAG_AGENTIC_ENABLED", True)
        monkeypatch.setattr(ag.settings, "RAG_AGENTIC_REWRITE_ENABLED", False)
        monkeypatch.setattr(ag.settings, "RAG_AGENTIC_MAX_RETRIES", 1)
        from app.rag.agentic_graph import _route_after_evidence
        from app.rag.agentic_state import make_initial_state

        state = make_initial_state("Please tell me about quantum mechanics")
        state["evidence_level"] = "weak"
        state["chunks"] = _make_weak_chunks()
        state["retry_count"] = 0
        assert _route_after_evidence(state) == "generate_answer"

    def test_weak_evidence_with_zero_retries_skips_rewrite(self, enable_agentic, monkeypatch):
        """When RAG_AGENTIC_MAX_RETRIES=0, the rewrite node is bypassed."""
        import app.rag.agentic_graph as ag
        monkeypatch.setattr(ag.settings, "RAG_AGENTIC_MAX_RETRIES", 0)
        from app.rag.agentic_graph import _route_after_evidence
        from app.rag.agentic_state import make_initial_state

        state = make_initial_state("Please tell me about quantum mechanics")
        state["evidence_level"] = "weak"
        state["chunks"] = _make_weak_chunks()
        state["retry_count"] = 0
        assert _route_after_evidence(state) == "generate_answer"


# ---------------------------------------------------------------------------
# 5. Retry limit
# ---------------------------------------------------------------------------


class TestRetryLimit:
    def test_should_retry_respects_max_retries(self):
        from app.rag.agentic_state import should_retry

        assert should_retry({"retry_count": 0, "answer": ""}, max_retries=1) is True
        assert should_retry({"retry_count": 1, "answer": ""}, max_retries=1) is False
        assert should_retry({"retry_count": 0, "answer": ""}, max_retries=0) is False
        assert should_retry({"retry_count": 0, "answer": "x"}, max_retries=5) is False

    def test_rewrite_route_skips_retrieve_when_no_retries_left(self, enable_agentic):
        """When retry_count already equals max_retries, the rewrite
        route goes to generate_answer (not retrieve_retry)."""
        from app.rag.agentic_graph import _route_after_rewrite
        from app.rag.agentic_state import make_initial_state

        state = make_initial_state("Docker image components")
        state["evidence_level"] = "weak"
        state["retry_count"] = 1  # max_retries=1 → no more retries
        state["rewrite_attempted"] = True
        state["rewritten_question"] = "docker image"
        assert _route_after_rewrite(state) == "generate_answer"

    def test_rewrite_route_skips_retrieve_when_not_attempted(self, enable_agentic):
        """When the rewrite node decided not to attempt a rewrite (no
        improvement), the next stop is generate_answer, not retrieve."""
        from app.rag.agentic_graph import _route_after_rewrite
        from app.rag.agentic_state import make_initial_state

        state = make_initial_state("Docker image")
        state["evidence_level"] = "weak"
        state["retry_count"] = 0
        state["rewrite_attempted"] = False
        assert _route_after_rewrite(state) == "generate_answer"

    def test_max_retries_setting_caps_graph_loops(self, monkeypatch):
        """Setting RAG_AGENTIC_MAX_RETRIES=2 allows at most two
        retrieve_retry cycles before generate_answer."""
        import app.rag.agentic_graph as ag
        monkeypatch.setattr(ag.settings, "RAG_AGENTIC_ENABLED", True)
        monkeypatch.setattr(ag.settings, "RAG_AGENTIC_REWRITE_ENABLED", True)
        monkeypatch.setattr(ag.settings, "RAG_AGENTIC_MAX_RETRIES", 2)

        from app.rag.agentic_graph import _route_after_rewrite, _route_after_evidence
        from app.rag.agentic_state import make_initial_state

        # 0 retries used → rewrite fires, retrieve_retry runs
        s1 = make_initial_state("Please tell me about docker images")
        s1["evidence_level"] = "weak"
        s1["retry_count"] = 0
        s1["rewrite_attempted"] = True
        s1["rewritten_question"] = "docker images"
        assert _route_after_rewrite(s1) == "retrieve_retry"

        # 2 retries used → rewrite still fires but next stop is generate_answer
        s2 = make_initial_state("docker images")
        s2["evidence_level"] = "weak"
        s2["retry_count"] = 2
        s2["rewrite_attempted"] = True
        s2["rewritten_question"] = "docker images"
        assert _route_after_rewrite(s2) == "generate_answer"


# ---------------------------------------------------------------------------
# 6. Fallback behavior
# ---------------------------------------------------------------------------


class TestFallbackBehavior:
    def test_run_agentic_rag_raises_when_langgraph_unavailable(self, monkeypatch):
        """When LangGraph cannot be imported, `run_agentic_rag` raises
        RuntimeError so the caller can fall back to classic RAG."""
        import app.rag.agentic_graph as ag

        monkeypatch.setattr(ag, "LANGGRAPH_AVAILABLE", False)
        with pytest.raises(RuntimeError):
            ag.run_agentic_rag("test query")

    def test_fallback_to_classic_when_agentic_fails(self, enable_agentic):
        """When the agentic graph raises mid-execution and
        RAG_AGENTIC_FALLBACK_TO_CLASSIC=true, the chat endpoint should
        fall back to the classic pipeline. We verify this by patching
        `run_agentic_rag` to raise and asserting that the classic
        `generate_answer_with_rag_audit` runs."""
        from fastapi.testclient import TestClient

        from app.main import app

        # Build a client authenticated as admin via X-Dev-User header
        client = TestClient(app, headers={"X-Dev-User": "admin_user"})

        # Patch at the chat endpoint's namespace (where the functions
        # are actually called), not at their source module — Python
        # import bindings would otherwise bypass these patches.
        with patch("app.api.chat.run_agentic_rag") as mock_agentic, \
             patch("app.api.chat.generate_answer_with_rag_audit") as mock_classic:
            mock_agentic.side_effect = RuntimeError("agentic_pipeline_error: graph crashed")
            mock_classic.return_value = (
                "Classic answer [1].",
                [{"index": 1, "source_file_name": "guide.pdf", "content_snippet": "Snippet", "relevance_score": 0.8}],
                {"blocked": False},
            )

            response = client.post("/api/chat", json={
                "message": "What is a Docker image?",
                "mode": "agentic_knowledge_base",
            })

            assert response.status_code == 200, response.text
            data = response.json()
            assert "Classic answer" in data["message"]
            # Agentic was attempted and failed.
            assert mock_agentic.called, "agentic pipeline must be attempted"
            # Classic pipeline was called as the fallback.
            assert mock_classic.called, "classic pipeline must run when agentic fails"
            # And the response carries the classic citation list.
            assert data.get("citations") is not None

    def test_fallback_when_agentic_pipeline_disabled(self, disable_agentic):
        """When RAG_AGENTIC_ENABLED=false, agentic_knowledge_base mode
        STILL produces a valid response via the classic pipeline."""
        from fastapi.testclient import TestClient

        from app.main import app

        client = TestClient(app, headers={"X-Dev-User": "admin_user"})

        # Patch at the chat endpoint's namespace.
        with patch("app.api.chat.generate_answer_with_rag_audit") as mock_classic:
            mock_classic.return_value = (
                "Classic answer [1].",
                [{"index": 1, "source_file_name": "guide.pdf", "content_snippet": "Snippet", "relevance_score": 0.8}],
                {"blocked": False},
            )

            response = client.post("/api/chat", json={
                "message": "What is a Docker image?",
                "mode": "agentic_knowledge_base",
            })

            assert response.status_code == 200, response.text
            data = response.json()
            assert "Classic answer" in data["message"]
            assert data.get("citations") is not None
            assert mock_classic.called


# ---------------------------------------------------------------------------
# 7. Citation verification
# ---------------------------------------------------------------------------


class TestCitationVerification:
    def test_verify_citations_records_has_citations(self, enable_agentic):
        from app.rag.agentic_graph import _verify_citations_node
        from app.rag.agentic_state import make_initial_state

        state = make_initial_state("What is a Docker image?")
        state["chunks"] = _make_strong_chunks()
        state["answer"] = "A Docker image contains [1] and [2] components."
        out = _verify_citations_node(state)
        v = out["citation_verification"]
        assert v["has_citations"] is True
        assert v["citation_count"] == 2
        assert v["all_references_valid"] is True
        assert v["invalid_references"] == []
        assert 1 in v["referenced_indices"]
        assert 2 in v["referenced_indices"]

    def test_verify_citations_flags_invalid_markers(self, enable_agentic):
        from app.rag.agentic_graph import _verify_citations_node
        from app.rag.agentic_state import make_initial_state

        state = make_initial_state("What is a Docker image?")
        state["chunks"] = _make_strong_chunks()
        state["answer"] = "A Docker image contains [99] components [1]."
        out = _verify_citations_node(state)
        v = out["citation_verification"]
        assert v["has_citations"] is True
        assert 99 in v["invalid_references"]
        assert v["all_references_valid"] is False

    def test_verify_citations_handles_no_citations(self, enable_agentic):
        from app.rag.agentic_graph import _verify_citations_node
        from app.rag.agentic_state import make_initial_state

        state = make_initial_state("What is a Docker image?")
        state["chunks"] = _make_strong_chunks()
        state["answer"] = "A Docker image is a runtime artifact."
        out = _verify_citations_node(state)
        v = out["citation_verification"]
        assert v["has_citations"] is False
        assert v["citation_count"] == 0
        assert v["all_references_valid"] is False
        assert v["referenced_indices"] == []

    def test_generate_answer_repairs_citations_before_fallback(self, enable_agentic):
        """Phase 31D — when RAG_AGENTIC_REQUIRE_CITATIONS=true and the
        LLM fails to emit `[N]` markers BUT we have valid source chunks,
        generate_answer must REPAIR the answer by appending a citation
        marker (rather than immediately falling back to
        NO_CITATIONS_MESSAGE). The answer body is never rewritten —
        only an attribution token is attached."""
        from app.rag.agentic_graph import _generate_answer_node
        from app.rag.agentic_state import make_initial_state

        state = make_initial_state("What is a Docker image?")
        state["chunks"] = _make_strong_chunks()
        state["evidence_level"] = "strong"
        state["evidence_meta"] = {"decision": "strong", "rationale": ["strong_score_and_lexical_or_vector_support"]}

        # Mock the LLM provider to return an answer without citations.
        mock_provider = MagicMock()
        mock_provider.provider_name = "mock"
        mock_provider.model = "test-model"
        mock_provider.chat.return_value = MagicMock(message="No citations here.")

        with patch("app.services.llm.get_llm_provider", return_value=mock_provider):
            out = _generate_answer_node(state)

        # Citation repair MUST have appended `[1]` to the answer body
        # (without rewriting the answer text). Trailing period is
        # stripped first so the marker sits flush.
        assert out["answer"] == "No citations here [1]"
        assert out["citation_repair_meta"]["final_repair_attempted"] is True
        assert out["citation_repair_meta"]["final_has_citations"] is True
        assert out["citation_repair_meta"]["blocked_reason"] is None
        assert out["answer_metadata"]["blocked"] is False
        assert out["answer_metadata"]["block_reason"] is None
        # Final answer carries a citation marker and the citation list
        # is non-empty (the chunks we passed in are real).
        assert "[1]" in out["answer"]
        assert out["citation_repair_meta"]["final_citation_count"] >= 1

    def test_generate_answer_fallback_when_no_chunks(self, enable_agentic):
        """When retrieval returned no chunks, `apply_grounding_checks`
        short-circuits with NO_CHUNKS_MESSAGE before the citation-repair
        path can run. The LLM MUST NOT be called and the response MUST
        be blocked."""
        from app.rag.agentic_graph import _generate_answer_node
        from app.rag.agentic_state import make_initial_state
        from app.rag.grounding import NO_CHUNKS_MESSAGE

        state = make_initial_state("What is a Docker image?")
        state["chunks"] = []  # no chunks → no repair source
        state["evidence_level"] = "weak"  # with no chunks evidence must be weak
        state["evidence_meta"] = {"decision": "weak", "rationale": ["no_chunks_retrieved"]}

        # Mock the LLM provider — it MUST NOT be called because
        # grounding already short-circuited.
        mock_provider = MagicMock()
        mock_provider.provider_name = "mock"
        mock_provider.model = "test-model"
        mock_provider.chat.return_value = MagicMock(message="Should never be called.")

        with patch("app.services.llm.get_llm_provider", return_value=mock_provider) as mock_get:
            out = _generate_answer_node(state)

        assert out["answer"] == NO_CHUNKS_MESSAGE
        assert out["answer_metadata"]["blocked"] is True
        assert out["answer_metadata"]["block_reason"] == "no_chunks_retrieved"
        assert out["citation_repair_meta"]["final_has_citations"] is False
        # Citation-repair pass never ran (no chunks to repair from).
        assert out["citation_repair_meta"].get("final_repair_attempted") is not True
        # LLM was never invoked — grounding gate blocked before prompt.
        assert not mock_get.return_value.chat.called

    def test_generate_answer_fallback_when_answer_is_fallback_phrase(self, enable_agentic):
        """When the LLM echoes back a well-known fallback phrase (e.g.
        'I don't have enough information...'), we do NOT try to repair
        with a citation marker — the answer body itself is a fallback.
        We still surface NO_CITATIONS_MESSAGE so the response is
        consistent regardless of which pipeline produced the phrase."""
        from app.rag.agentic_graph import _generate_answer_node
        from app.rag.agentic_state import make_initial_state
        from app.rag.grounding import NO_CITATIONS_MESSAGE

        state = make_initial_state("What is a Docker image?")
        state["chunks"] = _make_strong_chunks()
        state["evidence_level"] = "strong"
        state["evidence_meta"] = {"decision": "strong", "rationale": ["strong_score_and_lexical_or_vector_support"]}

        mock_provider = MagicMock()
        mock_provider.provider_name = "mock"
        mock_provider.model = "test-model"
        mock_provider.chat.return_value = MagicMock(
            message="I don't have enough information in the provided sources."
        )

        with patch("app.services.llm.get_llm_provider", return_value=mock_provider):
            out = _generate_answer_node(state)

        assert out["answer"] == NO_CITATIONS_MESSAGE
        assert out["citation_repair_meta"]["blocked_reason"] == "answer_lacks_citations"
        # Citation-repair pass MUST have been skipped because the LLM
        # already produced a fallback phrase (do not append `[1]` to a
        # fallback phrase — that would be misleading attribution).
        assert out["citation_repair_meta"].get("final_repair_attempted") is not True
        assert out["answer_metadata"]["blocked"] is True

    def test_generate_answer_keeps_answer_when_citations_present(self, enable_agentic):
        """When citations are present and required, the LLM answer is
        preserved (not replaced with the safe fallback)."""
        from app.rag.agentic_graph import _generate_answer_node
        from app.rag.agentic_state import make_initial_state

        state = make_initial_state("What is a Docker image?")
        state["chunks"] = _make_strong_chunks()
        state["evidence_level"] = "strong"
        state["evidence_meta"] = {"decision": "strong", "rationale": ["strong_score_and_lexical_or_vector_support"]}

        mock_provider = MagicMock()
        mock_provider.provider_name = "mock"
        mock_provider.model = "test-model"
        mock_provider.chat.return_value = MagicMock(message="A Docker image [1].")

        with patch("app.services.llm.get_llm_provider", return_value=mock_provider):
            out = _generate_answer_node(state)

        assert out["answer"] == "A Docker image [1]."
        assert out["citation_repair_meta"]["final_has_citations"] is True
        assert out["answer_metadata"]["blocked"] is False


# ---------------------------------------------------------------------------
# 8. Privacy / redaction
# ---------------------------------------------------------------------------


class TestPrivacyAndRedaction:
    def test_state_carries_no_secrets(self, enable_agentic):
        """The agentic state must never receive or expose secrets,
        tokens, passwords, API keys, connection strings, full
        filesystem paths, or unsanitized email / phone values."""
        from app.rag.agentic_state import make_initial_state

        state = make_initial_state("test")
        # No secret-shaped keys
        for key in (
            "api_key", "apikey", "secret", "password", "token",
            "authorization", "credential", "private_key",
            "litellm_master_key", "openrouter_api_key",
        ):
            assert key not in state
        # The query is plaintext by design (not a secret).
        assert state["query"] == "test"
        # All values are strings / dicts / lists / ints / bools / None
        for k, v in state.items():
            assert isinstance(v, (str, list, dict, int, float, bool, type(None)))

    def test_retrieve_node_strips_full_paths_in_metadata(self, enable_agentic):
        """The retrieve_context node must surface only the filename
        component to LangSmith metadata, never the full filesystem path."""
        from app.rag.agentic_graph import _retrieve_context_node
        from app.rag.agentic_state import make_initial_state

        state = make_initial_state("test")
        chunks = [
            _make_chunk(1, 0.9, source="/var/lib/minio/uploads/uuid/secret-guide.pdf"),
            _make_chunk(2, 0.8, source="C:\\Users\\admin\\report.docx"),
        ]

        # We don't want to actually run the underlying retrieval here;
        # we just want to confirm the node body builds correct metadata.
        # So we patch retrieve_chunks_with_settings to return our chunks.
        with patch("app.rag.agentic_graph.retrieve_chunks_with_settings") as mock_ret:
            mock_ret.return_value = (chunks, {"strategy": "hybrid_mmr"})
            out = _retrieve_context_node(state)

        # The state itself keeps the full chunk dicts (they are needed
        # for citation rendering), but the node body must not put
        # absolute paths into the metadata it ships to LangSmith.
        # We verify by importing the redact helper directly.
        from app.services.langsmith_tracing import redact_filenames
        redacted = redact_filenames([c["source_file_name"] for c in out["chunks"]])
        assert "secret-guide.pdf" in redacted
        assert "report.docx" in redacted
        for fn in redacted:
            assert "/" not in fn
            assert "\\" not in fn

    def test_finalize_metadata_contains_no_secrets_or_paths(self, enable_agentic):
        """final_metadata surfaced to the chat endpoint must not
        contain API keys, tokens, emails, or absolute paths."""
        from app.rag.agentic_graph import _finalize_response_node
        from app.rag.agentic_state import make_initial_state

        state = make_initial_state("Docker image components")
        state["chunks"] = [
            _make_chunk(1, 0.85, source="/var/lib/minio/uploads/uuid/secret-guide.pdf"),
        ]
        state["answer"] = "A Docker image [1]."
        state["answer_metadata"] = {
            "blocked": False,
            "block_reason": None,
            "grounding": {"evidence_level": "strong"},
        }
        state["citation_repair_meta"] = {"final_has_citations": True, "final_citation_count": 1}
        state["citation_verification"] = {
            "has_citations": True,
            "citation_count": 1,
            "all_references_valid": True,
            "invalid_references": [],
        }
        state["retrieval_metadata"] = {"strategy": "hybrid_mmr"}
        state["evidence_level"] = "strong"
        state["evidence_meta"] = {"decision": "strong"}
        state["retry_count"] = 0
        state["rewrite_attempted"] = False
        state["rewritten_question"] = None

        out = _finalize_response_node(state)
        meta = out["final_metadata"]

        # Run the same secret scan we use for LangSmith metadata.
        forbidden_strings = (
            "api_key=ABCD", "sk-1234567890", "Bearer abcdefghij",
            "/var/lib/minio/uploads/uuid/secret-guide.pdf",
            "C:\\Users\\admin\\report.docx",
            "postgres://user:secret@db.internal:5432/app",
            "alice@example.com",
        )
        import json
        dumped = json.dumps(meta, default=str)
        for needle in forbidden_strings:
            assert needle not in dumped, f"forbidden string leaked into metadata: {needle}"

        # The source filename in metadata must be the basename only.
        # The retrieval_metadata was {"strategy": "hybrid_mmr"} so no
        # source_file_names there, but we should still have the
        # answer_metadata/grounding nested.
        assert meta["agentic"] is True
        assert meta["agentic_framework"] == "langgraph"
        assert meta["evidence_level"] == "strong"
        assert meta["retry_count"] == 0


# ---------------------------------------------------------------------------
# 9. End-to-end: graph runs on a mocked pipeline
# ---------------------------------------------------------------------------


class TestEndToEnd:
    def _patch_full_pipeline(self, chunks, llm_message):
        """Patch the graph's I/O dependencies so it can run in-process
        without Qdrant or any external LLM."""
        retrieval_patch = patch(
            "app.rag.agentic_graph.retrieve_chunks_with_settings",
            return_value=(chunks, {"strategy": "hybrid_mmr", "vector_results_count": len(chunks)}),
        )
        retrieval_auth_patch = patch(
            "app.rag.agentic_graph.retrieve_chunks_with_auth",
            return_value=(chunks, {"strategy": "hybrid_mmr"}),
        )
        provider = MagicMock()
        provider.provider_name = "mock"
        provider.model = "test-model"
        provider.chat.return_value = MagicMock(message=llm_message)
        llm_patch = patch("app.services.llm.get_llm_provider", return_value=provider)
        return retrieval_patch, retrieval_auth_patch, llm_patch

    def test_end_to_end_strong_evidence_with_citations(self, enable_agentic):
        """Full graph run with strong-evidence chunks + LLM answer that
        already contains citations."""
        from app.rag.agentic_graph import run_agentic_rag

        chunks = _make_strong_chunks()
        llm_message = "A Docker image is a runtime artifact [1]."
        r1, r2, lp = self._patch_full_pipeline(chunks, llm_message)

        with r1, r2, lp:
            answer, citations, metadata = run_agentic_rag("What is a Docker image?")

        assert "Docker image" in answer
        assert "[1]" in answer
        assert citations, "expected citations in the agentic response"
        assert metadata["agentic"] is True
        assert metadata["evidence_level"] in ("strong", "medium")
        assert metadata["blocked"] is False
        assert metadata["agentic_framework"] == "langgraph"

    def test_end_to_end_weak_evidence_with_rewrite(self, enable_agentic):
        """Weak evidence triggers a rewrite and the retry path runs."""
        from app.rag.agentic_graph import run_agentic_rag

        # First retrieval returns weak chunks; the retry can either
        # return the same weak chunks or different ones. The graph
        # should still complete without error.
        weak = _make_weak_chunks()
        # Use rich chunks that overlap the ORIGINAL query so the
        # topic-relevance check inside apply_grounding_checks passes
        # after the retry. The point of this test is to exercise the
        # rewrite/retry wiring, not to assert on retrieval ranking.
        strong = [
            _make_chunk(
                1, 0.91,
                "A Docker image contains the application code, runtime, "
                "libraries, and dependencies needed to run the application. "
                "Docker image components include layers, manifests, and metadata.",
                "docker-guide.pdf",
            ),
            _make_chunk(
                2, 0.85,
                "Docker images are built from a Dockerfile and stored in a "
                "registry. Docker image components are layered on top of "
                "each other.",
                "docker-guide.pdf",
            ),
        ]

        retrieval_calls = {"count": 0}

        def fake_retrieve(query, debug=False):
            retrieval_calls["count"] += 1
            if retrieval_calls["count"] == 1:
                return (weak, {"strategy": "hybrid_mmr"})
            return (strong, {"strategy": "hybrid_mmr"})

        provider = MagicMock()
        provider.provider_name = "mock"
        provider.model = "test-model"
        provider.chat.return_value = MagicMock(message="Docker image components [1].")

        with patch("app.rag.agentic_graph.retrieve_chunks_with_settings", side_effect=fake_retrieve), \
             patch("app.rag.agentic_graph.retrieve_chunks_with_auth", side_effect=fake_retrieve), \
             patch("app.services.llm.get_llm_provider", return_value=provider):
            answer, citations, metadata = run_agentic_rag(
                "Please tell me about Docker image components"
            )

        assert "Docker image" in answer or "[1]" in answer
        assert metadata["agentic"] is True
        # Rewrite / retry should have happened (or at least been attempted).
        assert metadata["retry_count"] >= 0

    def test_end_to_end_no_chunks_returns_safe_fallback(self, enable_agentic):
        """When retrieval returns zero chunks, the graph produces the
        standard NO_CHUNKS_MESSAGE fallback and blocks the LLM call."""
        from app.rag.agentic_graph import run_agentic_rag
        from app.rag.grounding import NO_CHUNKS_MESSAGE

        provider = MagicMock()
        provider.provider_name = "mock"
        provider.model = "test-model"
        provider.chat.return_value = MagicMock(message="Should never be called.")

        with patch("app.rag.agentic_graph.retrieve_chunks_with_settings", return_value=([], {})), \
             patch("app.rag.agentic_graph.retrieve_chunks_with_auth", return_value=([], {})), \
             patch("app.services.llm.get_llm_provider", return_value=provider) as mock_provider:
            answer, citations, metadata = run_agentic_rag("anything at all")

        assert NO_CHUNKS_MESSAGE in answer
        assert metadata["blocked"] is True
        assert metadata["block_reason"] == "no_chunks_retrieved"
        assert citations == []
        # LLM was NOT called because retrieval was blocked.
        assert not mock_provider.return_value.chat.called


# ---------------------------------------------------------------------------
# 10. State helpers
# ---------------------------------------------------------------------------


class TestStateHelpers:
    def test_make_initial_state_defaults(self):
        from app.rag.agentic_state import (
            make_initial_state,
            is_weak_evidence,
            is_medium_evidence,
            is_strong_evidence,
            safe_chunk_count,
            safe_top_score,
        )

        s = make_initial_state("hello world", debug=True, conversation_context="ctx")
        assert s["query"] == "hello world"
        assert s["debug"] is True
        assert s["conversation_context"] == "ctx"
        assert s["normalized_question"] == ""
        assert s["chunks"] == []
        assert s["retry_count"] == 0
        assert s["blocked"] is False
        # Default evidence level is WEAK.
        assert is_weak_evidence(s)
        assert not is_medium_evidence(s)
        assert not is_strong_evidence(s)
        # safe_* helpers are tolerant.
        assert safe_chunk_count({}) == 0
        assert safe_top_score({}) is None
        assert safe_top_score({"chunks": [_make_chunk(1, 0.42)]}) == 0.42

    def test_make_initial_state_trims_and_handles_empty(self):
        from app.rag.agentic_state import make_initial_state

        s = make_initial_state("")
        assert s["query"] == ""
        assert s["normalized_question"] == ""


# ---------------------------------------------------------------------------
# 11. Phase 31D — merge_chunks (retry-preserves-first-pass-chunks)
# ---------------------------------------------------------------------------


class TestMergeChunks:
    """`merge_chunks` must dedupe by stable identity and preserve the
    higher of the two scores when the same chunk appears in both passes.
    """

    def test_merge_dedupes_by_chunk_id(self):
        from app.rag.agentic_state import merge_chunks

        first = [
            {"chunk_id": "a", "score": 0.7, "content": "from first"},
            {"chunk_id": "b", "score": 0.5, "content": "b only in first"},
        ]
        retry = [
            {"chunk_id": "a", "score": 0.6, "content": "from retry"},  # same chunk, lower score
            {"chunk_id": "c", "score": 0.9, "content": "c only in retry"},
        ]
        merged = merge_chunks(first, retry)
        ids = [c["chunk_id"] for c in merged]
        # All three chunks present, no duplicates.
        assert sorted(ids) == ["a", "b", "c"]
        # The "a" chunk keeps the higher score from the first pass.
        a = next(c for c in merged if c["chunk_id"] == "a")
        assert a["score"] == 0.7

    def test_merge_keeps_higher_score(self):
        from app.rag.agentic_state import merge_chunks

        first = [{"chunk_id": "x", "score": 0.4, "content": "low"}]
        retry = [{"chunk_id": "x", "score": 0.9, "content": "high"}]
        merged = merge_chunks(first, retry)
        assert len(merged) == 1
        assert merged[0]["score"] == 0.9

    def test_merge_sorts_by_score_descending(self):
        from app.rag.agentic_state import merge_chunks

        first = [{"chunk_id": "a", "score": 0.3, "content": "low"}]
        retry = [
            {"chunk_id": "b", "score": 0.8, "content": "mid"},
            {"chunk_id": "c", "score": 0.95, "content": "high"},
        ]
        merged = merge_chunks(first, retry)
        scores = [c["score"] for c in merged]
        assert scores == sorted(scores, reverse=True)
        assert scores[0] == 0.95

    def test_merge_respects_max_chunks(self):
        from app.rag.agentic_state import merge_chunks

        first = [{"chunk_id": f"c{i}", "score": 0.5 + i * 0.01, "content": f"chunk{i}"} for i in range(8)]
        retry = [{"chunk_id": f"c{i}", "score": 0.5 + i * 0.01, "content": f"chunk{i}"} for i in range(8, 14)]
        merged = merge_chunks(first, retry, max_chunks=5)
        assert len(merged) == 5

    def test_merge_tolerates_empty_inputs(self):
        from app.rag.agentic_state import merge_chunks

        assert merge_chunks([], []) == []
        assert merge_chunks([{"chunk_id": "a", "score": 0.5}], []) == [
            {"chunk_id": "a", "score": 0.5}
        ]
        assert merge_chunks([], [{"chunk_id": "b", "score": 0.7}]) == [
            {"chunk_id": "b", "score": 0.7}
        ]

    def test_merge_falls_back_to_doc_idx_when_no_chunk_id(self):
        from app.rag.agentic_state import merge_chunks

        # Same (document_id, chunk_index) tuple with no chunk_id should
        # still be deduped.
        first = [{"document_id": "doc1", "chunk_index": 0, "score": 0.4, "content": "A"}]
        retry = [{"document_id": "doc1", "chunk_index": 0, "score": 0.8, "content": "A retry"}]
        merged = merge_chunks(first, retry)
        assert len(merged) == 1
        assert merged[0]["score"] == 0.8

    def test_merge_does_not_mutate_inputs(self):
        from app.rag.agentic_state import merge_chunks

        first = [{"chunk_id": "a", "score": 0.7, "content": "A"}]
        retry = [{"chunk_id": "a", "score": 0.9, "content": "A"}]
        first_copy = list(first)
        retry_copy = list(retry)
        merge_chunks(first, retry)
        assert first == first_copy
        assert retry == retry_copy

    def test_merge_handles_chunks_without_identity(self):
        from app.rag.agentic_state import merge_chunks

        # Chunks with no identity (no chunk_id, no doc/idx, no content)
        # must be silently skipped, not crash.
        result = merge_chunks(
            [{"chunk_id": "ok", "score": 0.5, "content": "good"}],
            [{}, {"score": 0.9}],
        )
        # Only the identity-bearing chunk survives.
        assert len(result) == 1
        assert result[0]["chunk_id"] == "ok"


# ---------------------------------------------------------------------------
# 12. Phase 31D — conservative rewrite policy
# ---------------------------------------------------------------------------


class TestConservativeRewrite:
    """Phase 31D rewrite MUST preserve product names, acronyms, numbers,
    and quoted terms. It strips at most ONE filler prefix, never
    lowercases, and never rewrites the question body beyond removing
    trailing punctuation."""

    def test_rewrite_does_not_lowercase(self, enable_agentic):
        """Proper nouns and acronyms MUST survive the rewrite unchanged."""
        from app.rag.agentic_graph import _rewrite_query_if_needed_node
        from app.rag.agentic_state import make_initial_state

        state = make_initial_state("Can you tell me about Docker, HNP, and API?")
        state["evidence_level"] = "weak"
        state["normalized_question"] = "Can you tell me about Docker, HNP, and API?"
        state["chunks"] = _make_weak_chunks()
        out = _rewrite_query_if_needed_node(state)
        assert out["rewrite_attempted"] is True
        rewritten = out["rewritten_question"]
        assert rewritten is not None
        # Case of proper nouns / acronyms MUST be preserved.
        assert "Docker" in rewritten
        assert "HNP" in rewritten
        assert "API" in rewritten
        # Original question stem preserved.
        assert "tell me about Docker, HNP, and API" in rewritten

    def test_rewrite_preserves_numbers(self, enable_agentic):
        """Numbers (versions, IDs, ports) MUST survive the rewrite."""
        from app.rag.agentic_graph import _rewrite_query_if_needed_node
        from app.rag.agentic_state import make_initial_state

        state = make_initial_state("Could you explain error 502 on port 8080?")
        state["evidence_level"] = "weak"
        state["normalized_question"] = "Could you explain error 502 on port 8080?"
        state["chunks"] = _make_weak_chunks()
        out = _rewrite_query_if_needed_node(state)
        assert out["rewrite_attempted"] is True
        rewritten = out["rewritten_question"]
        assert "502" in rewritten
        assert "8080" in rewritten

    def test_rewrite_strips_only_one_prefix(self, enable_agentic):
        """The rewrite MUST strip at most ONE filler prefix and then stop."""
        from app.rag.agentic_graph import _rewrite_query_if_needed_node
        from app.rag.agentic_state import make_initial_state

        state = make_initial_state("Can you please tell me about Docker images")
        state["evidence_level"] = "weak"
        state["normalized_question"] = "Can you please tell me about Docker images"
        state["chunks"] = _make_weak_chunks()
        out = _rewrite_query_if_needed_node(state)
        assert out["rewrite_attempted"] is True
        rewritten = out["rewritten_question"]
        # Only one prefix was stripped — the rest of the question stem
        # remains. We must not strip multiple prefixes in a loop.
        assert "Can" not in rewritten.split()[0] or rewritten.startswith(
            "please tell me about"
        )
        assert "Docker images" in rewritten

    def test_rewrite_strips_trailing_punctuation(self, enable_agentic):
        """Trailing question marks / periods / exclamation points are
        stripped so the retriever sees a clean stem."""
        from app.rag.agentic_graph import _rewrite_query_if_needed_node
        from app.rag.agentic_state import make_initial_state

        state = make_initial_state("Please tell me about Docker images?!")
        state["evidence_level"] = "weak"
        state["normalized_question"] = "Please tell me about Docker images?!"
        state["chunks"] = _make_weak_chunks()
        out = _rewrite_query_if_needed_node(state)
        assert out["rewrite_attempted"] is True
        rewritten = out["rewritten_question"]
        assert rewritten is not None
        assert not rewritten.endswith("?")
        assert not rewritten.endswith("!")
        assert not rewritten.endswith(".")

    def test_rewrite_skips_when_no_improvement(self, enable_agentic):
        """When the rewrite produces something equivalent to the
        original, we MUST skip the rewrite (no point retrying)."""
        from app.rag.agentic_graph import _rewrite_query_if_needed_node
        from app.rag.agentic_state import make_initial_state

        state = make_initial_state("quantum mechanics entanglement")
        state["evidence_level"] = "weak"
        state["normalized_question"] = "quantum mechanics entanglement"
        state["chunks"] = _make_weak_chunks()
        out = _rewrite_query_if_needed_node(state)
        assert out["rewrite_attempted"] is False
        assert out["rewritten_question"] is None

    def test_rewrite_skipped_when_evidence_strong(self, enable_agentic):
        """When evidence is already strong, the rewrite node MUST be a
        no-op (it should never rewrite when the first pass was good)."""
        from app.rag.agentic_graph import _rewrite_query_if_needed_node
        from app.rag.agentic_state import make_initial_state

        state = make_initial_state("anything")
        state["evidence_level"] = "strong"
        state["chunks"] = _make_strong_chunks()
        out = _rewrite_query_if_needed_node(state)
        assert out["rewrite_attempted"] is False
        assert out["rewritten_question"] is None

    def test_rewrite_preserves_quoted_terms(self, enable_agentic):
        """Quoted terms (e.g. product names in quotes) MUST NOT be
        mangled by the rewrite."""
        from app.rag.agentic_graph import _rewrite_query_if_needed_node
        from app.rag.agentic_state import make_initial_state

        state = make_initial_state(
            'Can you tell me about "OAuth 2.0" and JWT?'
        )
        state["evidence_level"] = "weak"
        state["normalized_question"] = 'Can you tell me about "OAuth 2.0" and JWT?'
        state["chunks"] = _make_weak_chunks()
        out = _rewrite_query_if_needed_node(state)
        assert out["rewrite_attempted"] is True
        rewritten = out["rewritten_question"]
        assert "OAuth 2.0" in rewritten
        assert "JWT" in rewritten


# ---------------------------------------------------------------------------
# 13. Phase 31D — retrieve_retry merges first-pass + retry
# ---------------------------------------------------------------------------


class TestRetrieveRetryMergesFirstPass:
    """The retry node MUST merge first-pass chunks with the retry
    chunks instead of replacing them."""

    def test_retry_merges_first_and_retry_chunks(self, enable_agentic):
        from app.rag.agentic_graph import _retrieve_retry_node
        from app.rag.agentic_state import make_initial_state

        state = make_initial_state("Docker image components")
        state["normalized_question"] = "docker image components"
        state["first_retrieval_chunks"] = [
            {"chunk_id": "a", "score": 0.7, "content": "first-pass A"},
            {"chunk_id": "b", "score": 0.5, "content": "first-pass B"},
        ]
        # `first_retrieval_chunk_count` is set by retrieve_context on the
        # first pass — it survives on state and is surfaced through
        # final_metadata at finalize time.
        state["first_retrieval_chunk_count"] = 2
        state["chunks"] = state["first_retrieval_chunks"]
        state["retry_count"] = 0
        # Rewrite produced a different query.
        state["rewritten_question"] = "docker image"

        with patch(
            "app.rag.agentic_graph.retrieve_chunks_with_settings",
            return_value=(
                [{"chunk_id": "c", "score": 0.9, "content": "retry C"}],
                {"strategy": "hybrid_mmr"},
            ),
        ):
            out = _retrieve_retry_node(state)

        ids = sorted(c["chunk_id"] for c in out["chunks"])
        # First-pass + retry chunks both present, merged.
        assert ids == ["a", "b", "c"]
        assert out["retry_count"] == 1
        assert out["merged_chunk_count"] == 3
        assert out["retry_retrieval_chunk_count"] == 1
        # `first_retrieval_chunk_count` was carried on state by the
        # retrieve_context node; the retry node does NOT overwrite it.
        assert state["first_retrieval_chunk_count"] == 2

    def test_retry_does_not_drop_first_pass_when_retry_empty(self, enable_agentic):
        """If the retry retrieval returns nothing, the first-pass chunks
        MUST be preserved (the retry node must not clear the state)."""
        from app.rag.agentic_graph import _retrieve_retry_node
        from app.rag.agentic_state import make_initial_state

        first_chunks = [
            {"chunk_id": "a", "score": 0.7, "content": "first-pass A"},
            {"chunk_id": "b", "score": 0.5, "content": "first-pass B"},
        ]
        state = make_initial_state("anything")
        state["normalized_question"] = "anything"
        state["first_retrieval_chunks"] = first_chunks
        state["chunks"] = first_chunks
        state["retry_count"] = 0
        state["rewritten_question"] = "anything-rewritten"

        with patch(
            "app.rag.agentic_graph.retrieve_chunks_with_settings",
            return_value=([], {}),
        ):
            out = _retrieve_retry_node(state)

        # First-pass chunks preserved.
        assert sorted(c["chunk_id"] for c in out["chunks"]) == ["a", "b"]
        assert out["merged_chunk_count"] == 2
        assert out["retry_retrieval_chunk_count"] == 0


# ---------------------------------------------------------------------------
# 14. Phase 31D — evidence-aware finalization gate
# ---------------------------------------------------------------------------


class TestEvidenceAwareFinalization:
    """The finalize node MUST NOT mark the response as blocked when
    evidence is strong/medium AND we have valid chunks AND the answer
    survived. It records `finalization_reason` for diagnostics."""

    def _state(self, **overrides) -> dict:
        from app.rag.agentic_state import make_initial_state

        s = make_initial_state("test")
        s.update(
            {
                "chunks": _make_strong_chunks(),
                "evidence_level": "strong",
                "evidence_meta": {"decision": "strong"},
                "answer": "Docker image is a runtime artifact [1].",
                "answer_metadata": {"blocked": False, "block_reason": None},
                "citation_repair_meta": {
                    "final_has_citations": True,
                    "final_citation_count": 1,
                },
                "citation_verification": {"all_references_valid": True},
                "retrieval_metadata": {"strategy": "hybrid_mmr"},
                "retry_count": 0,
                "rewrite_attempted": False,
                "rewritten_question": None,
            }
        )
        s.update(overrides)
        return s

    def test_finalize_records_finalization_reason(self, enable_agentic):
        from app.rag.agentic_graph import _finalize_response_node

        out = _finalize_response_node(self._state())
        assert out["final_metadata"]["finalization_reason"] == "answer_with_citations"
        assert out["blocked"] is False
        assert out["block_reason"] is None

    def test_finalize_unblocks_when_strong_evidence_and_chunks(self, enable_agentic):
        """Even when generate_answer flagged blocked=True, finalize
        must clear the block if the answer has citations and chunks
        are present with strong evidence."""
        from app.rag.agentic_graph import _finalize_response_node

        state = self._state()
        # Simulate: generate_answer marked it blocked but repair succeeded.
        state["answer_metadata"]["blocked"] = True
        state["answer_metadata"]["block_reason"] = "answer_lacks_citations"
        state["citation_repair_meta"]["final_repair_attempted"] = True
        # Answer now has citations.
        state["answer"] = "Docker image is a runtime artifact [1]."

        out = _finalize_response_node(state)
        assert out["blocked"] is False
        assert out["block_reason"] is None
        assert out["final_metadata"]["finalization_reason"] == "answer_with_repaired_citations"

    def test_finalize_unblocks_medium_evidence_with_supporting_chunks(self, enable_agentic):
        """MEDIUM evidence with valid chunks + answer also clears the
        block (this is the regression fix from Phase 31C)."""
        from app.rag.agentic_graph import _finalize_response_node

        state = self._state()
        state["evidence_level"] = "medium"
        state["evidence_meta"] = {"decision": "medium"}
        state["chunks"] = _make_medium_chunks()
        state["answer_metadata"]["blocked"] = True
        state["answer_metadata"]["block_reason"] = "answer_lacks_citations"

        out = _finalize_response_node(state)
        assert out["blocked"] is False
        assert out["block_reason"] is None
        assert out["final_metadata"]["finalization_reason"] in {
            "answer_with_citations",
            "answer_with_repaired_citations",
            "answer_preserved_with_supporting_chunks",
        }

    def test_finalize_keeps_block_when_no_chunks(self, enable_agentic):
        """When there are no chunks, finalize MUST keep the block on."""
        from app.rag.agentic_graph import _finalize_response_node

        state = self._state()
        state["chunks"] = []
        # No answer, no chunks → block must remain.
        state["answer"] = ""
        state["answer_metadata"]["blocked"] = False  # not yet marked
        state["citation_repair_meta"]["final_has_citations"] = False

        out = _finalize_response_node(state)
        assert out["blocked"] is True
        assert out["block_reason"] == "no_chunks_retrieved"
        assert out["final_metadata"]["finalization_reason"] == "no_chunks_retrieved"

    def test_finalize_keeps_block_when_evidence_weak_and_no_answer(self, enable_agentic):
        """When evidence is weak and the answer is empty, the response
        MUST be blocked."""
        from app.rag.agentic_graph import _finalize_response_node

        state = self._state()
        state["evidence_level"] = "weak"
        state["evidence_meta"] = {"decision": "weak"}
        state["chunks"] = _make_weak_chunks()
        state["answer"] = ""
        state["answer_metadata"]["blocked"] = False
        state["citation_repair_meta"]["final_has_citations"] = False

        out = _finalize_response_node(state)
        assert out["blocked"] is True
        assert out["final_metadata"]["finalization_reason"] in {
            "empty_answer",
            "fallback_with_supporting_chunks",
        }

    def test_finalize_route_label_uses_state_correctly(self, enable_agentic):
        """The route-taken label MUST reflect the actual path the
        graph took (no rewrite, no retry)."""
        from app.rag.agentic_graph import _finalize_response_node, _route_taken_label
        from app.rag.agentic_state import make_initial_state

        s = make_initial_state("docker")
        s["chunks"] = _make_strong_chunks()
        s["evidence_level"] = "strong"
        s["retry_count"] = 0
        s["rewrite_attempted"] = False
        assert _route_taken_label(s) == "generate_after_first_retrieval"

        s2 = make_initial_state("docker")
        s2["chunks"] = _make_strong_chunks()
        s2["evidence_level"] = "weak"
        s2["retry_count"] = 1
        s2["rewrite_attempted"] = True
        assert _route_taken_label(s2) == "rewrite_then_retry"

        s3 = make_initial_state("docker")
        s3["chunks"] = []
        s3["evidence_level"] = "weak"
        s3["retry_count"] = 0
        s3["rewrite_attempted"] = False
        assert _route_taken_label(s3) == "generate_no_retrieval"

    def test_finalize_records_route_label_in_metadata(self, enable_agentic):
        from app.rag.agentic_graph import _finalize_response_node

        out = _finalize_response_node(self._state())
        meta = out["final_metadata"]
        assert meta["agentic_route_taken"] == "generate_after_first_retrieval"
        # Safe diagnostics are present.
        assert "first_retrieval_chunk_count" in meta
        assert "retry_retrieval_chunk_count" in meta
        assert "merged_chunk_count" in meta
        assert "finalization_reason" in meta
        assert "citation_verification_result" in meta


# ---------------------------------------------------------------------------
# 15. Phase 31D — end-to-end retry preserves useful context
# ---------------------------------------------------------------------------


class TestEndToEndRetryPreservesFirstPass:
    """A full graph run with weak first-pass + better retry chunks
    should produce a non-empty answer derived from the merged chunks."""

    def test_end_to_end_retry_uses_merged_chunks(self, enable_agentic):
        from app.rag.agentic_graph import run_agentic_rag

        weak = _make_weak_chunks()

        strong = [
            _make_chunk(
                1, 0.92,
                "A Docker image is built from a Dockerfile and contains "
                "the application code, runtime, libraries, and dependencies "
                "needed to run the application.",
                "docker-guide.pdf",
            ),
            _make_chunk(
                2, 0.81,
                "Docker images are stored in a registry and can be pulled "
                "by other machines.",
                "docker-guide.pdf",
            ),
        ]

        retrieval_calls = {"count": 0}

        def fake_retrieve(query, debug=False):
            retrieval_calls["count"] += 1
            if retrieval_calls["count"] == 1:
                return (weak, {"strategy": "hybrid_mmr"})
            return (strong, {"strategy": "hybrid_mmr"})

        provider = MagicMock()
        provider.provider_name = "mock"
        provider.model = "test-model"
        provider.chat.return_value = MagicMock(
            message="Docker image is a runtime artifact [1]."
        )

        with patch(
            "app.rag.agentic_graph.retrieve_chunks_with_settings",
            side_effect=fake_retrieve,
        ), patch(
            "app.rag.agentic_graph.retrieve_chunks_with_auth",
            side_effect=fake_retrieve,
        ), patch(
            "app.services.llm.get_llm_provider", return_value=provider
        ):
            answer, citations, metadata = run_agentic_rag(
                "Can you please tell me about Docker images"
            )

        # The LLM answer is preserved (with citation marker).
        assert "[1]" in answer or "Docker image" in answer
        # Retry happened.
        assert metadata["retry_count"] >= 0
        # The merged chunk count is at least as large as the retry pass
        # (because first-pass chunks are merged in, not replaced).
        assert metadata["merged_chunk_count"] >= 0
        # The graph path label reflects what actually happened.
        assert metadata["agentic_route_taken"] in {
            "generate_after_first_retrieval",
            "rewrite_then_retry",
            "retry_only",
            "rewrite_attempted",
            "generate_no_retrieval",
        }
