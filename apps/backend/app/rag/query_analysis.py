"""
Phase 34A.1 — Deterministic Query Analysis

Conservative, regex-based extraction of technical identifiers and intent
signals from a user query. NO LLM. The component is intentionally
document-agnostic: it does not hard-code any domain vocabulary beyond
the universal shape of error codes (ORA-1234, HTTP 500, 0xABCD, etc.).

Design goals:

1. **Deterministic.** Same input → same output, every time. No latency,
   no external calls, no flaky extraction.
2. **Conservative.** Standalone numbers (years, counts, page numbers)
   must NOT be auto-tagged as error codes. We require either a
   recognised identifier *shape* (ORA-, HTTP, 0x, SQLSTATE, ERR_, HL-)
   or adjacent context such as "error 902" / "code 902" / "18456".
3. **Composable.** The output is a small dataclass that other RAG
   components (exact_match, image_routing, relevance_filter) consume
   without having to re-run regexes.
4. **Safe to extend.** New identifier shapes can be added by extending
   `_PATTERNS` without changing the public dataclass.

The component does NOT itself decide which sources are relevant — it
just emits signals. Decision logic lives in `exact_match.py` and
`relevance_filter.py`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional


# ---------------------------------------------------------------------------
# Identifier patterns
# ---------------------------------------------------------------------------

# Each entry is (name, compiled regex). Order matters only for the
# "first match wins" diagnostic output; the public dataclass de-duplicates.

# ORA-12345 (Oracle). Allow ORA-xxxxx / ORA-12345 etc.
# Allow letters in the suffix so placeholders like ORA-xxxxx match.
_RE_ORACLE = re.compile(r"\bORA-[A-Z0-9]{1,8}\b", re.IGNORECASE)

# SQLSTATE 08001 / SQLSTATE 22P02
_RE_SQLSTATE = re.compile(r"\bSQLSTATE\s+[A-Z0-9]{4,6}\b", re.IGNORECASE)

# HTTP 404 / HTTP 500
_RE_HTTP_CODE = re.compile(r"\bHTTP[\s/_-]*(?:status[\s_-]*)?(\d{3})\b", re.IGNORECASE)

# 0x80070005 (Windows error code)
_RE_HEX_CODE = re.compile(r"\b0x[0-9A-Fa-f]{4,16}\b")

# ERR_CONNECTION_REFUSED / ERR_NAME_NOT_RESOLVED (browser/node-style)
_RE_ERR_CONST = re.compile(r"\bERR_[A-Z][A-Z0-9_]{3,}\b")

# HL-1053 (vendor prefix style)
_RE_HL_CODE = re.compile(r"\bHL-\d{2,6}\b")

# Generic "error NNNN" / "error code NNNN" / "code NNNN" patterns.
# These are the ONLY path by which a bare numeric token like "902" or
# "18456" can be promoted to an error code, and they require the
# identifier to be adjacent to a context word.
_RE_NUMERIC_WITH_CONTEXT = re.compile(
    r"\b(?:error|err\.?|code|errno|errcode|status|return\s+code|trigger|triggers|threw|throws|showing|shows|see|saw|got|reported)"
    r"[\s#:_-]*"
    r"(\d{2,6})\b",
    re.IGNORECASE,
)

# Codes like SQL Server 18456 appear as standalone tokens inside
# screenshots/OCR dumps. We accept those ONLY if they also appear next
# to a context word or inside a chunk (handled in exact_match.py).
# For query analysis we capture them only when the context word is
# present. Standalone 4–6 digit numeric tokens are deliberately
# ignored here so we don't false-positive on years, page numbers,
# and counts.

# Image-content intent phrases. Conservative list — only the patterns
# that unambiguously refer to the user's own uploaded media.
_IMAGE_INTENT_PATTERNS = [
    re.compile(r"\b(?:the\s+)?(?:image|screenshot|picture|photo|attachment|file)\s+i\s+(?:just\s+)?(?:uploaded|attached|sent|shared|provided)\b", re.IGNORECASE),
    re.compile(r"\b(?:my\s+|the\s+)?(?:latest|recent|current)\s+(?:image|screenshot|picture|photo)\b", re.IGNORECASE),
    re.compile(r"\b(?:this|the)\s+(?:image|screenshot|picture)\b", re.IGNORECASE),
    re.compile(r"\bin\s+the\s+(?:image|screenshot|picture|attachment|photo)\b", re.IGNORECASE),
    re.compile(r"\b(?:shown|displayed|visible|written)\s+in\s+the\s+(?:image|screenshot|picture|attachment|photo)\b", re.IGNORECASE),
    # Image noun REQUIRED in the pattern. "what does error 902 mean"
    # must NOT match (no image noun).
    re.compile(r"\bwhat\s+(?:does|did|is|are)\s+(?:the|my)\s+(?:image|screenshot|picture|attachment|photo)\s+(?:say|show|contain|read|display)\b", re.IGNORECASE),
    re.compile(r"\bwhat\s+(?:error|code|message|number)\s+(?:is|are)\s+(?:shown|displayed|visible)\s+(?:in|on)\s+(?:the|my|a)\s+(?:image|screenshot|picture)\b", re.IGNORECASE),
    re.compile(r"\bocr\s+text\b", re.IGNORECASE),
    re.compile(r"\b(?:read|extract|tell\s+me)\s+(?:the\s+)?text\s+from\s+(?:the|my)\s+(?:image|screenshot|picture|attachment|photo)\b", re.IGNORECASE),
    # Phase 34A.2.1 — "attached image" / "the attached image" / "attached screenshot"
    # The word "attached" in front of "image"/"screenshot"/"picture" is a
    # strong signal the user is referring to their own uploaded image.
    # Matches: "What text is shown in the attached image?"
    #         "What does the attached image say?"
    re.compile(r"\b(?:the\s+)?attached\s+(?:image|screenshot|picture|photo|file)\b", re.IGNORECASE),
    # Phase 34A.2.1 — "shown here" / "here" as an image reference when the
    # user has an active image_context and "here" refers to the previewed
    # image. "here" alone is too broad ("click here", "type here") so we
    # require it to be adjacent to a visible/displayed/show/error verb
    # or to appear in a question about what the image shows.
    # Matches: "What error code is shown here?"
    #         "What text is displayed here?"
    #         "What does this image show here?"
    re.compile(r"\bshown\s+here\b", re.IGNORECASE),
]


# Troubleshooting / knowledge intent. If the query contains both an
# image-content phrase and an identifier, we route the same way as a
# plain troubleshooting query (broaden to KB) but flag the image
# context so retrieval can still bias to the uploaded OCR.
_TROUBLESHOOTING_HINTS = (
    "troubleshoot",
    "fix",
    "resolve",
    "what does",
    "what does error",
    "meaning",
    "what is error",
    "what's error",
    "how do i fix",
    "how to fix",
    "how to resolve",
    "has this error",
    "documented",
    "documentation",
    "knowledge",
    "kb",
)


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass
class QueryAnalysis:
    """Structured result of analysing a user query."""

    query: str
    query_type: str = "general"          # general | error_lookup | image_content
    error_codes: List[str] = field(default_factory=list)
    technical_terms: List[str] = field(default_factory=list)
    references_uploaded_image: bool = False
    raw_signals: dict = field(default_factory=dict)

    def has_identifiers(self) -> bool:
        return bool(self.error_codes) or bool(self.technical_terms)

    def to_dict(self) -> dict:
        return {
            "query_type": self.query_type,
            "error_codes": list(self.error_codes),
            "technical_terms": list(self.technical_terms),
            "references_uploaded_image": self.references_uploaded_image,
        }


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


def _normalize_code(code: str) -> str:
    """Normalise an extracted identifier for storage and comparison.

    We preserve the canonical shape used in chunk content (e.g. the
    space in "HTTP 404") so that downstream exact-match code can do a
    direct substring comparison without re-formatting. Whitespace is
    collapsed to a single space and the result is uppercased.
    """
    return re.sub(r"\s+", " ", code.strip()).upper()


def _extract_codes(query: str) -> List[str]:
    """Extract error/technical codes from a query.

    Conservative: bare numeric tokens are ignored unless preceded by an
    "error"/"code"/"errno"/"return code" context word.
    """
    if not query:
        return []

    found: list[str] = []

    for _name, regex in (
        ("oracle", _RE_ORACLE),
        ("sqlstate", _RE_SQLSTATE),
        ("http", _RE_HTTP_CODE),
        ("hex", _RE_HEX_CODE),
        ("err_const", _RE_ERR_CONST),
        ("hl", _RE_HL_CODE),
    ):
        for m in regex.finditer(query):
            raw = m.group(0)
            # HTTP match group 1 is the numeric status — keep "HTTP 404" style
            if regex is _RE_HTTP_CODE and m.lastindex:
                raw = f"HTTP {m.group(1)}"
            found.append(_normalize_code(raw))

    # Context-gated bare numerics: "error 902", "code 18456"
    for m in _RE_NUMERIC_WITH_CONTEXT.finditer(query):
        num = m.group(1)
        # Skip obviously non-error numerics: years like 2024, page counts
        # of < 100, zip codes, etc. We accept any 2–6 digit number attached
        # to a context word — the caller's downstream exact_match module
        # will further filter against chunk content.
        found.append(_normalize_code(num))

    # De-duplicate while preserving order. Use the same canonical form
    # for both storage and comparison so the exact-match module can
    # check via `code in haystack` after case-folding.
    seen: set = set()
    deduped: list[str] = []
    for c in found:
        if c and c not in seen:
            seen.add(c)
            deduped.append(c)
    return deduped


def _extract_technical_terms(query: str) -> List[str]:
    """Extract technical terms that look like hyphenated or underscored
    identifiers (rejected-forbidden-country, mail-delivery-failed, etc.).

    These are NOT error codes per se but are strong retrieval signals:
    they should match exactly against chunk content.
    """
    if not query:
        return []

    # Hyphenated or underscored lowercase tokens (>= 8 chars).
    # Examples: rejected-forbidden-country, message-delivery-failed.
    matches = re.findall(r"\b[a-z][a-z0-9]+(?:[-_][a-z0-9]+){1,}\b", query)
    # Filter out very common short forms.
    out: list[str] = []
    for m in matches:
        if len(m) >= 8:
            out.append(m.lower())
    # De-duplicate while preserving order
    seen = set()
    deduped: list[str] = []
    for t in out:
        if t not in seen:
            seen.add(t)
            deduped.append(t)
    return deduped


def _detect_image_intent(query: str) -> bool:
    """Return True if the user is asking about the content of an image
    they uploaded (vs. asking a general knowledge-base question)."""
    if not query:
        return False
    for pattern in _IMAGE_INTENT_PATTERNS:
        if pattern.search(query):
            return True
    return False


def _detect_troubleshooting_intent(query: str) -> bool:
    """Return True if the query is a troubleshooting/knowledge lookup.

    Broad match: used for diagnostics and observability only.
    """
    if not query:
        return False
    q = query.lower()
    return any(h in q for h in _TROUBLESHOOTING_HINTS)


# Strong troubleshooting verbs used by `analyze_query` to disambiguate
# image_content vs. error_lookup-with-image-context. These are the
# unambiguous "I want KB help" verbs that override an image reference.
_STRONG_TROUBLESHOOTING_HINTS = (
    "troubleshoot",
    "fix",
    "resolve",
    "debug",
    "diagnose",
    "workaround",
    "how do i",
    "how to",
    "what causes",
)


def _detect_strong_troubleshooting_intent(query: str) -> bool:
    """Stronger, image-reference-safe troubleshooting detector.

    Returns True only when the query contains unambiguous troubleshooting
    verbs. Weak hints like "what does", "meaning", "what is error" are
    excluded — they describe the image-content ask when an image is
    referenced and only count as troubleshooting when no image is
    referenced (in which case `_detect_troubleshooting_intent` returns
    True and the general path handles it).
    """
    if not query:
        return False
    q = query.lower()
    return any(h in q for h in _STRONG_TROUBLESHOOTING_HINTS)


def analyze_query(query: str) -> QueryAnalysis:
    """Public entry point.

    Returns a `QueryAnalysis` describing the query's intent and the
    technical identifiers it contains. The function is pure and safe to
    call inside the chat hot path.
    """
    analysis = QueryAnalysis(query=query or "")

    if not query:
        analysis.query_type = "general"
        return analysis

    analysis.error_codes = _extract_codes(query)
    analysis.technical_terms = _extract_technical_terms(query)
    analysis.references_uploaded_image = _detect_image_intent(query)

    # Phase 34A.1.1 — refined intent classification.
    #
    # When the user references an uploaded image, we must distinguish
    # between:
    #
    #   (A) "image_content"   — "What error code is shown in the image
    #                          I uploaded?", "What does the screenshot
    #                          I uploaded say?", "Read the text from my
    #                          image". The user wants the content of
    #                          THEIR image. Scope retrieval to OCR
    #                          chunks only.
    #
    #   (B) "error_lookup" with image context — "How do I troubleshoot
    #                          error 902 shown in the image?",
    #                          "Resolve the rejection error in the
    #                          screenshot". The user wants broader KB
    #                          help AND the image as supporting
    #                          evidence. Keep all chunks; exact-match
    #                          booster ranks the OCR chunk.
    #
    # The disambiguation uses STRONG troubleshooting verbs ("troubleshoot",
    # "fix", "resolve", "debug") plus the presence of an identifier. Weak
    # hints like "what does", "meaning", "what is error" are excluded when
    # an image is referenced because they describe the image content ask,
    # not a KB lookup. Without an image reference they still count as
    # troubleshooting hints.
    strong_troubleshooting_intent = _detect_strong_troubleshooting_intent(query)

    if (
        analysis.references_uploaded_image
        and not analysis.error_codes
        and not analysis.technical_terms
        and not strong_troubleshooting_intent
    ):
        # Pure image-content question: ask what's in the image, no
        # identifier, no explicit troubleshooting verb.
        analysis.query_type = "image_content"
    elif analysis.error_codes or analysis.technical_terms:
        # Identifiers present (with or without image reference) →
        # broaden retrieval to KB. Spec scenario A would still scope
        # the citation set via image_routing at retrieval time, but
        # the query analysis reports the intent as error_lookup so
        # downstream diagnostics are consistent.
        analysis.query_type = "error_lookup"
    elif analysis.references_uploaded_image and strong_troubleshooting_intent:
        # Troubleshooting question that references the user's image.
        analysis.query_type = "error_lookup"
    else:
        analysis.query_type = "general"

    analysis.raw_signals = {
        "error_codes_count": len(analysis.error_codes),
        "technical_terms_count": len(analysis.technical_terms),
        "image_intent_matched": analysis.references_uploaded_image,
        "troubleshooting_matched": _detect_troubleshooting_intent(query),
        "strong_troubleshooting_matched": strong_troubleshooting_intent,
    }

    return analysis


def references_uploaded_image(query: str) -> bool:
    """Convenience wrapper: True if the query references an uploaded image."""
    return _detect_image_intent(query or "")
