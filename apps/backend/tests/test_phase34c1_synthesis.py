"""Phase 34C.1 -- Acceptance tests for deterministic image-observation synthesis.

Eight tests, one per requirement from the plan:

1. exact identifier synthesis                 (Path A)
2. semantic dashboard synthesis without identifier (Path B)
3. ambiguous semantic results do not synthesize
4. product meaning does not synthesize
5. troubleshooting does not synthesize
6. unauthorized image cannot synthesize
7. no image evidence cannot synthesize
8. substantive LLM answer is preserved unchanged

The synthesizer is invoked as post-LLM refusal recovery. The LLM
"answer" in these tests is always a known refusal phrase unless the
test is verifying the "substantive answer preserved" guarantee.
"""

from __future__ import annotations

import pytest

from app.rag.image_observation_synthesis import (
    maybe_synthesize_image_observation_answer,
)


REFUSAL = (
    "I do not have enough information in the provided sources to answer "
    "that question with confidence."
)
SUBSTANTIVE = (
    "Error 902 is described in the Installation Guide: messages are rejected "
    "when the destination country is in the carrier's forbidden list [1]."
)


def _img_chunk(
    content: str = "Failed Reason: 902 Message delivery failed: rejected-forbidden-country",
    *,
    score: float = 0.85,
    document_id: int = 60,
    image_id: int = 6,
    source_file_name: str = "Screenshot 2026-08-05 174611.png",
    image_type: str = "screenshot",
    **extra,
) -> dict:
    base = {
        "content": content,
        "source_type": "image_knowledge",
        "source_file_name": source_file_name,
        "image_id": image_id,
        "document_id": document_id,
        "score": score,
        "image_type": image_type,
    }
    base.update(extra)
    return base


def _kb_chunk(content: str = "Error 902 in chapter 4", *, document_id: int = 1) -> dict:
    return {
        "content": content,
        "source_type": "document_chunk",
        "source_file_name": "Installation_Guide.pdf",
        "document_id": document_id,
        "score": 0.50,
    }


# ---------------------------------------------------------------------------
# 1. Path A — exact identifier synthesis.
# ---------------------------------------------------------------------------

class TestExactIdentifierSynthesis:
    def test_path_a_synthesizes_with_filename_and_ocr(self):
        chunks = [_img_chunk()]
        synth = maybe_synthesize_image_observation_answer(
            query="Which screenshot showed error 902?",
            chunks=chunks,
            answer=REFUSAL,
            accessible_doc_ids={60},
        )
        assert synth is not None, "expected synthesis for exact identifier match"
        assert "Screenshot 2026-08-05 174611.png" in synth.answer
        assert "902" in synth.answer
        assert "[1]" in synth.answer

        # Meta should record which path was taken and which identifier matched.
        assert synth.meta["path"] == "A"
        assert synth.meta["matched_identifier"] == "902"
        assert synth.meta["image_id"] == 6
        assert synth.meta["document_id"] == 60
        assert synth.meta["source_file_name"] == "Screenshot 2026-08-05 174611.png"

        # Citation must be a properly-shaped image_knowledge citation.
        assert len(synth.citations) == 1
        cite = synth.citations[0]
        assert cite["citation_kind"] == "image_knowledge"
        assert cite["image_id"] == 6
        assert cite["document_id"] == 60
        assert cite["source_file_name"] == "Screenshot 2026-08-05 174611.png"
        assert cite["index"] == 1

    def test_path_a_carrier_identifier(self):
        chunks = [
            _img_chunk(
                "Carrier: Vodafone-UK\nTimestamp: 2026-08-20 14:32 UTC",
                source_file_name="vodafone_uk_2026_08_20.png",
            )
        ]
        synth = maybe_synthesize_image_observation_answer(
            query="Which screenshot shows Vodafone-UK delivery?",
            chunks=chunks,
            answer=REFUSAL,
            accessible_doc_ids={60},
        )
        assert synth is not None
        assert "vodafone_uk_2026_08_20.png" in synth.answer
        assert synth.meta["matched_identifier"] == "vodafone-uk"


# ---------------------------------------------------------------------------
# 2. Path B — semantic dashboard synthesis without identifier.
# ---------------------------------------------------------------------------

class TestSemanticDashboardSynthesis:
    def test_path_b_synthesizes_when_dashboard_dominates(self):
        chunks = [
            _img_chunk(
                "Traffic spike at 14:00 UTC, peak 1247 req/s, return-code 200",
                score=0.85,
                image_type="dashboard",
                source_file_name="dashboard_2026_08_12.png",
                document_id=60,
                image_id=11,
            ),
        ]
        synth = maybe_synthesize_image_observation_answer(
            query="Which dashboard showed the spike?",
            chunks=chunks,
            answer=REFUSAL,
            accessible_doc_ids={60},
        )
        assert synth is not None, "expected synthesis when a single dashboard dominates"
        assert "dashboard_2026_08_12.png" in synth.answer
        assert "[1]" in synth.answer
        assert synth.meta["path"] == "B"
        assert synth.meta["matched_identifier"] is None
        assert synth.meta["image_id"] == 11
        assert synth.meta["source_file_name"] == "dashboard_2026_08_12.png"
        # Quote must be non-empty (Path B picks the top OCR line).
        assert synth.meta["quote"]


# ---------------------------------------------------------------------------
# 3. Ambiguous semantic results do not synthesize.
# ---------------------------------------------------------------------------

class TestAmbiguousSemantic:
    def test_two_unrelated_dashboards_close_score_do_not_synthesize(self):
        chunks = [
            _img_chunk(
                "Dashboard A spike 14:00",
                score=0.85,
                image_type="dashboard",
                source_file_name="dash_a.png",
                document_id=60,
                image_id=11,
            ),
            _img_chunk(
                "Dashboard B spike 14:30",
                score=0.83,
                image_type="dashboard",
                source_file_name="dash_b.png",
                document_id=70,
                image_id=22,
            ),
        ]
        synth = maybe_synthesize_image_observation_answer(
            query="Which dashboard showed the spike?",
            chunks=chunks,
            answer=REFUSAL,
            accessible_doc_ids={60, 70},
        )
        assert synth is None, "must refuse synthesis when two unrelated images tie near the top"


# ---------------------------------------------------------------------------
# 4. Product meaning does not synthesize.
# ---------------------------------------------------------------------------

class TestProductMeaningDoesNotSynthesize:
    @pytest.mark.parametrize(
        "query",
        [
            "What does error 902 mean?",
            "What does error 902 mean and how do I fix it?",
            "What is the cause of error 902?",
            "What is the definition of error 902?",
        ],
    )
    def test_product_meaning_no_synthesis(self, query: str):
        chunks = [_img_chunk()]
        synth = maybe_synthesize_image_observation_answer(
            query=query,
            chunks=chunks,
            answer=REFUSAL,
            accessible_doc_ids={60},
        )
        assert synth is None


# ---------------------------------------------------------------------------
# 5. Troubleshooting does not synthesize.
# ---------------------------------------------------------------------------

class TestTroubleshootingDoesNotSynthesize:
    @pytest.mark.parametrize(
        "query",
        [
            "How do I fix error 902?",
            "How to troubleshoot the spike?",
            "How can I resolve error 902 in our messaging pipeline?",
            "Please fix the dashboard issue.",
            "What should I do when error 902 appears?",
            "Root cause of error 902?",
        ],
    )
    def test_troubleshooting_no_synthesis(self, query: str):
        chunks = [_img_chunk()]
        synth = maybe_synthesize_image_observation_answer(
            query=query,
            chunks=chunks,
            answer=REFUSAL,
            accessible_doc_ids={60},
        )
        assert synth is None


# ---------------------------------------------------------------------------
# 6. Unauthorized image cannot synthesize.
# ---------------------------------------------------------------------------

class TestUnauthorizedCannotSynthesize:
    def test_accessible_doc_ids_excludes_chunk_doc(self):
        chunks = [_img_chunk(document_id=60)]
        synth = maybe_synthesize_image_observation_answer(
            query="Which screenshot showed error 902?",
            chunks=chunks,
            answer=REFUSAL,
            accessible_doc_ids={999},  # does NOT include 60
        )
        assert synth is None

    def test_accessible_doc_ids_none_derives_from_chunks(self):
        """accessible_doc_ids=None -> the synthesizer derives the
        authorized set from the chunks' own ``document_id`` values.

        Rationale: chunks have already been RBAC-filtered upstream by
        ``retrieve_chunks_with_auth``, so the union of their
        ``document_id`` values IS exactly the caller's accessible
        set in scope for this candidate list. A caller that forgets
        to pass ``retrieval_metadata['accessible_doc_ids']`` still
        gets a correct (defense-in-depth) auth check instead of a
        fail-closed refusal that would defeat the synthesis safety
        net.
        """
        chunks = [_img_chunk()]  # document_id=60
        synth = maybe_synthesize_image_observation_answer(
            query="Which screenshot showed error 902?",
            chunks=chunks,
            answer=REFUSAL,
            accessible_doc_ids=None,
        )
        # Path A applies -> synthesize.
        assert synth is not None
        assert synth.meta["path"] == "A"

    def test_explicit_empty_accessible_set_refuses_synthesis(self):
        """Passing an EXPLICIT empty set denies synthesis (the strict
        path). This is the only way for a caller to opt out of the
        derivation safety net.
        """
        chunks = [_img_chunk()]  # document_id=60
        synth = maybe_synthesize_image_observation_answer(
            query="Which screenshot showed error 902?",
            chunks=chunks,
            answer=REFUSAL,
            accessible_doc_ids=set(),
        )
        assert synth is None

    def test_authorized_false_refuses_synthesis(self):
        chunks = [_img_chunk()]
        synth = maybe_synthesize_image_observation_answer(
            query="Which screenshot showed error 902?",
            chunks=chunks,
            answer=REFUSAL,
            accessible_doc_ids={60},
            authorized=False,
        )
        assert synth is None


# ---------------------------------------------------------------------------
# 7. No image evidence cannot synthesize.
# ---------------------------------------------------------------------------

class TestNoImageEvidenceCannotSynthesize:
    def test_only_kb_chunks_no_synthesis(self):
        chunks = [_kb_chunk(document_id=1)]
        synth = maybe_synthesize_image_observation_answer(
            query="Which screenshot showed error 902?",
            chunks=chunks,
            answer=REFUSAL,
            accessible_doc_ids={1},
        )
        assert synth is None

    def test_empty_chunks_no_synthesis(self):
        synth = maybe_synthesize_image_observation_answer(
            query="Which screenshot showed error 902?",
            chunks=[],
            answer=REFUSAL,
            accessible_doc_ids=set(),
        )
        assert synth is None

    def test_mixed_image_and_kb_synthesizes_via_image_filter(self):
        """Mixed KB + image chunks DO synthesize.

        The hybrid retriever always returns mixed KB + image_knowledge
        evidence. The synthesizer filters to image-derived chunks
        BEFORE classifying, so the image-only subset still satisfies
        the all-evidence-is-image precondition. KB chunks coexist in
        the retriever response but do not vote on whether a
        synthesized answer is safe.
        """
        chunks = [_img_chunk(), _kb_chunk()]
        synth = maybe_synthesize_image_observation_answer(
            query="Which screenshot showed error 902?",
            chunks=chunks,
            answer=REFUSAL,
            accessible_doc_ids={60, 1},
        )
        assert synth is not None
        assert synth.meta["path"] == "A"
        assert synth.meta["source_file_name"] == "Screenshot 2026-08-05 174611.png"


# ---------------------------------------------------------------------------
# 8. Substantive LLM answer is preserved unchanged.
# ---------------------------------------------------------------------------

class TestSubstantiveAnswerPreserved:
    def test_substantive_answer_bypasses_synthesizer(self):
        chunks = [_img_chunk()]
        synth = maybe_synthesize_image_observation_answer(
            query="Which screenshot showed error 902?",
            chunks=chunks,
            answer=SUBSTANTIVE,  # NOT a refusal phrase
            accessible_doc_ids={60},
        )
        assert synth is None, "must not clobber a substantive LLM answer"

    def test_substantive_answer_preserved_even_with_refusal_path_eligible(self):
        # Same shape as the exact-identifier test, but the LLM
        # already answered -- the synthesizer must yield.
        chunks = [_img_chunk()]
        synth = maybe_synthesize_image_observation_answer(
            query="Which screenshot showed error 902?",
            chunks=chunks,
            answer="The screenshot showing error 902 is Screenshot 2026-08-05 174611.png [1].",
            accessible_doc_ids={60},
        )
        assert synth is None

    def test_empty_answer_treated_as_substantive(self):
        # An empty answer is not a refusal phrase. The synthesizer
        # should NOT run -- the upstream pipeline handles empty
        # answers via its own path.
        chunks = [_img_chunk()]
        synth = maybe_synthesize_image_observation_answer(
            query="Which screenshot showed error 902?",
            chunks=chunks,
            answer="",
            accessible_doc_ids={60},
        )
        assert synth is None
