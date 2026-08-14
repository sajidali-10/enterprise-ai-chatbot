"""
Phase 34A.1 — Deterministic Query Analysis tests.

Covers positive / negative / false-positive cases for identifier
extraction and intent classification.
"""

import pytest

from app.rag.query_analysis import analyze_query, references_uploaded_image


# ---------------------------------------------------------------------------
# Error-code extraction (positive cases)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query,expected_codes",
    [
        ("How do I troubleshoot error 902?", ["902"]),
        ("error 18456 occurred", ["18456"]),
        ("ORA-12541 connection refused", ["ORA-12541"]),
        ("ORA-xxxxx please help", ["ORA-XXXXX"]),
        ("SQLSTATE 08001 error", ["SQLSTATE 08001"]),
        ("HTTP 404", ["HTTP 404"]),
        ("HTTP 500 server error", ["HTTP 500"]),
        ("0x80070005 access denied", ["0X80070005"]),
        ("ERR_CONNECTION_REFUSED", ["ERR_CONNECTION_REFUSED"]),
        ("HL-1053 error", ["HL-1053"]),
        ("code 902", ["902"]),
        ("errno 500", ["500"]),
        ("return code 1053", ["1053"]),
    ],
)
def test_error_code_extraction_positive(query, expected_codes):
    analysis = analyze_query(query)
    for c in expected_codes:
        assert c in analysis.error_codes, (
            f"expected {c!r} in {analysis.error_codes!r} for query {query!r}"
        )


# ---------------------------------------------------------------------------
# Error-code extraction (negative / false-positive cases)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query",
    [
        "the year 2024 was great",          # bare numeric, no context word
        "page 42 of the manual",            # page number
        "we received 1000 messages",        # count
        "session 12 today",                 # session id
        "what's the weather like",          # plain English
        "summarise the document",           # plain English
    ],
)
def test_no_false_positive_codes(query):
    analysis = analyze_query(query)
    assert analysis.error_codes == [], (
        f"unexpected codes {analysis.error_codes!r} for query {query!r}"
    )
    assert analysis.query_type == "general"


# ---------------------------------------------------------------------------
# Technical-term extraction
# ---------------------------------------------------------------------------


def test_technical_term_extraction():
    analysis = analyze_query("Why did rejected-forbidden-country trigger 902?")
    assert "rejected-forbidden-country" in analysis.technical_terms
    assert "902" in analysis.error_codes
    assert analysis.query_type == "error_lookup"


def test_underscore_technical_term():
    analysis = analyze_query("error in message_delivery_failed path")
    assert "message_delivery_failed" in analysis.technical_terms


# ---------------------------------------------------------------------------
# Image intent detection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "query",
    [
        "What error code is shown in the image I uploaded?",
        "What does the screenshot say?",
        "What message is visible in the latest screenshot?",
        "Read the text from the screenshot I just uploaded",
        "What does the image say",
        "What's shown in this picture",
    ],
)
def test_image_intent_positive(query):
    assert references_uploaded_image(query) is True
    analysis = analyze_query(query)
    assert analysis.references_uploaded_image is True


@pytest.mark.parametrize(
    "query,references_image",
    [
        ("What does error 902 mean?", False),
        ("How do I troubleshoot the error shown in the screenshot?", True),
        ("Has this error occurred in our documentation?", False),
    ],
)
def test_troubleshooting_intent_with_image_reference(query, references_image):
    analysis = analyze_query(query)
    # Troubleshooting questions are classified as error_lookup when an
    # identifier is present, general otherwise. Only the screenshot
    # phrasing actually references the user's uploaded image.
    assert analysis.references_uploaded_image is references_image
    assert analysis.query_type in ("error_lookup", "general")


# ---------------------------------------------------------------------------
# Query classification
# ---------------------------------------------------------------------------


def test_query_type_general():
    a = analyze_query("summarise the document for me")
    assert a.query_type == "general"


def test_query_type_error_lookup():
    a = analyze_query("ORA-12541 connection refused")
    assert a.query_type == "error_lookup"


def test_query_type_image_content():
    a = analyze_query("What error code is shown in the image I uploaded?")
    assert a.query_type == "image_content"


def test_query_type_hybrid_image_with_id():
    a = analyze_query(
        "What does error 902 in the image I uploaded mean?"
    )
    assert a.references_uploaded_image is True
    assert "902" in a.error_codes
    # When an identifier is present AND the image is referenced,
    # classify as error_lookup so retrieval broadens to the KB.
    assert a.query_type == "error_lookup"


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_query_safe():
    a = analyze_query("")
    assert a.query_type == "general"
    assert a.error_codes == []
    assert a.technical_terms == []
    assert a.references_uploaded_image is False


def test_none_query_safe():
    a = analyze_query(None)
    assert a.query_type == "general"
    assert a.error_codes == []


def test_to_dict_serialisable():
    a = analyze_query("ORA-12541 connection refused")
    d = a.to_dict()
    assert d["query_type"] == "error_lookup"
    assert "ORA-12541" in d["error_codes"]
    assert isinstance(d["references_uploaded_image"], bool)


def test_case_insensitive_http():
    a = analyze_query("http 500 server error")
    assert "HTTP 500" in a.error_codes


def test_multiple_codes_in_one_query():
    a = analyze_query(
        "I see ORA-12541 and HTTP 500 in the screenshot"
    )
    assert "ORA-12541" in a.error_codes
    assert "HTTP 500" in a.error_codes
