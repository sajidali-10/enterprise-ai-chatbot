"""Phase 34C.1 -- Unit tests for the public image-observation helpers.

Pure tests: no LLM, no Qdrant, no DB. These pin the regex patterns
and the classifier preconditions so the grounded bypass and the
synthesis paths cannot drift apart.
"""

from __future__ import annotations

import pytest

from app.rag.image_observation_helpers import (
    HISTORICAL_IMAGE_INTENT_RE,
    PRODUCT_MEANING_INTENT_RE,
    IDENTIFIER_TOKEN_RE,
    classify_image_observation_query,
    extract_query_identifier_tokens,
    identifier_matches_chunk_content,
    is_image_knowledge_chunk,
    is_llm_refusal_answer,
    all_chunks_are_image_knowledge,
)


# ---------------------------------------------------------------------------
# Regex intent tests.
# ---------------------------------------------------------------------------

class TestHistoricalImageIntentRegex:
    @pytest.mark.parametrize(
        "query",
        [
            "Which screenshot showed error 902?",
            "Which dashboard showed the spike?",
            "Which image shows Vodafone-UK?",
            "Find the screenshot that shows the failure",
            "Do we have a screenshot of the crash?",
            "Previously uploaded screenshot of the dashboard",
            "Screenshot where the spike appears",
        ],
    )
    def test_historical_intent_matches(self, query: str):
        assert HISTORICAL_IMAGE_INTENT_RE.search(query), query

    @pytest.mark.parametrize(
        "query",
        [
            "What does error 902 mean?",
            "How do I fix the dashboard?",
            "Tell me about error 902",
            "Why is the system failing?",
        ],
    )
    def test_historical_intent_rejects_non_observation(self, query: str):
        assert not HISTORICAL_IMAGE_INTENT_RE.search(query), query


class TestProductMeaningIntentRegex:
    @pytest.mark.parametrize(
        "query",
        [
            "What does error 902 mean?",
            "How do I fix error 902?",
            "How to configure the gateway?",
            "How to troubleshoot the spike?",
            "How can I resolve the failure?",
            "Why is error 902 appearing?",
            "Root cause of the dashboard crash?",
            "How should I configure carrier routing?",
            "What should I do when error 902 appears?",
            "Please fix the dashboard.",
        ],
    )
    def test_product_meaning_intent_matches(self, query: str):
        assert PRODUCT_MEANING_INTENT_RE.search(query), query


# ---------------------------------------------------------------------------
# Identifier extraction tests.
# ---------------------------------------------------------------------------

class TestIdentifierExtraction:
    def test_902(self):
        assert extract_query_identifier_tokens("Which screenshot showed error 902?") == ["902"]

    def test_carrier(self):
        assert extract_query_identifier_tokens("Vodafone-UK delivery screenshot") == ["vodafone-uk"]

    def test_filename(self):
        tokens = extract_query_identifier_tokens("dashboard_2026_08_12.png")
        assert "dashboard_2026_08_12" in tokens

    def test_phone_number(self):
        tokens = extract_query_identifier_tokens("Rejection on +44 7700 900123")
        assert any("+44" in t for t in tokens)

    def test_empty(self):
        assert extract_query_identifier_tokens("") == []

    def test_no_identifiers(self):
        # Common English words should not be extracted as identifiers.
        assert extract_query_identifier_tokens("Which dashboard showed the spike?") == []


class TestIdentifierMatching:
    def test_identifier_matches_content_substring(self):
        chunk = {"content": "Failed Reason: 902 Message delivery failed"}
        assert identifier_matches_chunk_content("902", chunk)

    def test_identifier_matches_filename(self):
        chunk = {"source_file_name": "Screenshot 2026-08-05 174611.png"}
        assert identifier_matches_chunk_content("screenshot 2026-08-05", chunk)

    def test_no_match(self):
        chunk = {"content": "Failed Reason: 901 different issue"}
        assert not identifier_matches_chunk_content("902", chunk)


# ---------------------------------------------------------------------------
# LLM refusal detection.
# ---------------------------------------------------------------------------

class TestLLMRefusalDetection:
    @pytest.mark.parametrize(
        "answer",
        [
            "I do not have enough information in the provided sources to answer that.",
            "I could not find enough information in the provided sources.",
            "The provided sources do not contain enough detail.",
            "I'm unable to answer this question.",
            "No relevant documents were found.",
        ],
    )
    def test_refusal_phrases_detected(self, answer: str):
        assert is_llm_refusal_answer(answer) is True

    @pytest.mark.parametrize(
        "answer",
        [
            "Error 902 means the message was rejected for forbidden country.",
            "The dashboard shows a spike at 14:00 UTC.",
            "",
            None,
            "Per the source [1], error 902 means forbidden country.",
        ],
    )
    def test_substantive_answers_not_flagged(self, answer):
        assert is_llm_refusal_answer(answer) is False


# ---------------------------------------------------------------------------
# Source-type helpers.
# ---------------------------------------------------------------------------

class TestIsImageKnowledgeChunk:
    def test_image_knowledge_source_type(self):
        assert is_image_knowledge_chunk({"source_type": "image_knowledge"})

    def test_image_ocr_source_type(self):
        assert is_image_knowledge_chunk({"source_type": "image_ocr"})

    def test_image_id_anchors_chunk(self):
        assert is_image_knowledge_chunk({"image_id": 6, "source_type": "document_chunk"})

    def test_png_filename(self):
        assert is_image_knowledge_chunk({"source_file_name": "screenshot.png"})

    def test_jpg_filename(self):
        assert is_image_knowledge_chunk({"filename": "dash.jpg"})

    def test_dashboard_in_filename(self):
        assert is_image_knowledge_chunk({"source_file_name": "ops_dashboard_v2"})

    def test_plain_kb_chunk_rejected(self):
        assert not is_image_knowledge_chunk(
            {"source_type": "document_chunk", "source_file_name": "guide.pdf"}
        )

    def test_empty_chunk_rejected(self):
        assert not is_image_knowledge_chunk({})


class TestAllChunksAreImageKnowledge:
    def test_all_image(self):
        chunks = [
            {"source_type": "image_knowledge"},
            {"source_type": "image_ocr"},
        ]
        assert all_chunks_are_image_knowledge(chunks) is True

    def test_mixed_rejected(self):
        chunks = [
            {"source_type": "image_knowledge"},
            {"source_type": "document_chunk"},
        ]
        assert all_chunks_are_image_knowledge(chunks) is False

    def test_empty_rejected(self):
        assert all_chunks_are_image_knowledge([]) is False


# ---------------------------------------------------------------------------
# Classifier tests.
# ---------------------------------------------------------------------------

def _img_chunk(content: str = "Failed Reason: 902", score: float = 0.85, **extra) -> dict:
    base = {
        "content": content,
        "source_type": "image_knowledge",
        "source_file_name": "Screenshot 2026-08-05 174611.png",
        "image_id": 6,
        "document_id": 60,
        "score": score,
    }
    base.update(extra)
    return base


class TestClassifier:
    def test_path_a_exact_identifier(self):
        d = classify_image_observation_query(
            "Which screenshot showed error 902?",
            [_img_chunk()],
        )
        assert d.path_a_eligible is True
        assert d.matched_identifier == "902"
        assert d.path_b_eligible is False

    def test_path_b_dominant_dashboard_no_identifier(self):
        d = classify_image_observation_query(
            "Which dashboard showed the spike?",
            [_img_chunk("Traffic spike at 14:00 UTC, peak 1247 req/s", image_type="dashboard")],
        )
        assert d.path_b_eligible is True
        assert d.path_a_eligible is False
        assert d.dominant_chunk is not None

    def test_path_b_ambiguous_two_unrelated_images(self):
        chunks = [
            _img_chunk("Traffic spike", score=0.85, document_id=60, image_id=1, source_file_name="dash_a.png", image_type="dashboard"),
            _img_chunk("Different spike", score=0.83, document_id=70, image_id=2, source_file_name="dash_b.png", image_type="dashboard"),
        ]
        d = classify_image_observation_query("Which dashboard showed the spike?", chunks)
        assert d.path_a_eligible is False
        assert d.path_b_eligible is False
        assert d.path_b_ambiguous is True

    def test_path_b_unambiguous_two_related_images(self):
        # Same document, same image_type -> considered the same image
        # for ambiguity purposes.
        chunks = [
            _img_chunk("Traffic spike", score=0.85, document_id=60, image_id=1, source_file_name="dash_a.png", image_type="dashboard"),
            _img_chunk("Traffic zoomed in", score=0.83, document_id=60, image_id=2, source_file_name="dash_b.png", image_type="dashboard"),
        ]
        d = classify_image_observation_query("Which dashboard showed the spike?", chunks)
        assert d.path_b_eligible is True

    def test_product_meaning_disqualifies(self):
        d = classify_image_observation_query(
            "What does error 902 mean and how do I fix it?",
            [_img_chunk()],
        )
        assert d.refusing_intent is True
        assert d.any_path_eligible() is False

    def test_troubleshooting_disqualifies(self):
        d = classify_image_observation_query(
            "How do I fix error 902?",
            [_img_chunk()],
        )
        assert d.refusing_intent is True
        assert d.any_path_eligible() is False

    def test_no_chunks_disqualifies(self):
        d = classify_image_observation_query("Which screenshot showed error 902?", [])
        # With no chunks, the classifier cannot confirm any of the
        # preconditions and returns the default decision -- no path is
        # eligible, and even the intent flags stay conservative.
        assert d.any_path_eligible() is False
        assert d.all_evidence_is_image is False

    def test_no_image_intent_disqualifies(self):
        d = classify_image_observation_query(
            "Tell me about error 902",
            [_img_chunk()],
        )
        assert d.is_image_observation is False
        assert d.any_path_eligible() is False

    def test_unauthorized_disqualifies(self):
        d = classify_image_observation_query(
            "Which screenshot showed error 902?",
            [_img_chunk()],
            authorized=False,
        )
        assert d.authorized is False
        assert d.any_path_eligible() is False

    def test_mixed_evidence_disqualifies(self):
        chunks = [
            _img_chunk(),
            {"content": "Error 902 in chapter 4", "source_type": "document_chunk",
             "source_file_name": "guide.pdf", "score": 0.5},
        ]
        d = classify_image_observation_query("Which screenshot showed error 902?", chunks)
        assert d.all_evidence_is_image is False
        assert d.any_path_eligible() is False

    def test_below_score_threshold(self):
        d = classify_image_observation_query(
            "Which dashboard showed the spike?",
            [_img_chunk("Spike", score=0.50, image_type="dashboard")],
        )
        assert d.path_b_eligible is False

    def test_path_a_takes_priority_over_path_b(self):
        # When both could apply, Path A wins.
        chunks = [
            _img_chunk("Spike at 14:00 with code 902", score=0.85, image_type="dashboard"),
        ]
        d = classify_image_observation_query(
            "Which dashboard showed error 902?",
            chunks,
        )
        assert d.path_a_eligible is True
        assert d.path_b_eligible is False
