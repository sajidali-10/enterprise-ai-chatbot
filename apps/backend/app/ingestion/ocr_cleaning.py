"""
OCR Text Cleaning (Phase 34A).

Deterministic cleanup of OCR-derived text. Operates on a single
string and returns the cleaned string plus a small stats dict.

The cleaning rules are deliberately conservative — we must not
invent or rewrite content. The goal is to normalize noisy OCR output
into something that is easy to chunk and easy to search, without
changing the semantic content.

Pipeline:
    1. Normalize Unicode (NFKC).
    2. Replace control characters (other than ``\\n``, ``\\t``) with
       spaces.
    3. Normalize line endings to ``\\n``.
    4. Strip trailing whitespace from each line.
    5. Collapse 3+ consecutive blank lines into one blank line.
    6. Repair common OCR line-wrapping artifacts (a hyphen at end of
       line followed by a lowercase letter on the next line is
       joined). This is safe for English-like languages; non-hyphen
       language scripts are unaffected.
    7. Collapse runs of 3+ spaces (not newlines) into a single space.

This module does NOT call any LLM. Sanitization of secrets, keys,
passwords, etc. is performed by
``app.services.langsmith_tracing.redact_text`` and is layered AFTER
this module.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Dict


_HYPHEN_WRAP_RE = re.compile(r"(\w)-\n(\w)")
_MULTI_BLANK = re.compile(r"\n{3,}")
_RUN_SPACES = re.compile(r"[^\S\n]{3,}")
_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


@dataclass
class CleanStats:
    """Statistics about a single cleaning pass."""

    chars_in: int
    chars_out: int
    lines_in: int
    lines_out: int
    hyphens_joined: int
    control_chars_removed: int

    def as_dict(self) -> Dict[str, int]:
        return {
            "chars_in": self.chars_in,
            "chars_out": self.chars_out,
            "lines_in": self.lines_in,
            "lines_out": self.lines_out,
            "hyphens_joined": self.hyphens_joined,
            "control_chars_removed": self.control_chars_removed,
        }


def clean_ocr_text(text: str) -> str:
    """Return deterministically cleaned OCR text. See module docstring."""
    if not text:
        return ""
    return _clean(text)


def clean_ocr_text_with_stats(text: str) -> tuple[str, CleanStats]:
    """Same as :func:`clean_ocr_text` but returns statistics."""
    if not text:
        return "", CleanStats(0, 0, 0, 0, 0, 0)
    return _clean(text, with_stats=True)  # type: ignore[return-value]


def _clean(text: str, with_stats: bool = False):
    chars_in = len(text)
    lines_in = text.count("\n") + 1 if text else 0

    # 1. Unicode normalize.
    out = unicodedata.normalize("NFKC", text)

    # 2. Strip control chars.
    removed_ctrl = sum(1 for c in out if c in "\x00\x01\x02\x03\x04\x05\x06\x07\x08\x0b\x0c\x0e\x0f\x10\x11\x12\x13\x14\x15\x16\x17\x18\x19\x1a\x1b\x1c\x1d\x1e\x1f\x7f")
    out = _CONTROL_CHARS.sub(" ", out)

    # 3. Normalize line endings.
    out = out.replace("\r\n", "\n").replace("\r", "\n")

    # 4. Repair hyphen-wrapped words.
    new_out, n_hyphens = _HYPHEN_WRAP_RE.subn(r"\1\2", out)
    out = new_out

    # 5. Trim trailing whitespace per line.
    out = "\n".join(line.rstrip() for line in out.split("\n"))

    # 6. Collapse multi-blank lines.
    out = _MULTI_BLANK.sub("\n\n", out)

    # 7. Collapse runs of 3+ spaces (or tabs) into one space, but
    #    preserve newlines.
    out = _RUN_SPACES.sub(" ", out)

    # Final trim of leading/trailing whitespace.
    out = out.strip()

    chars_out = len(out)
    lines_out = out.count("\n") + 1 if out else 0

    if not with_stats:
        return out

    stats = CleanStats(
        chars_in=chars_in,
        chars_out=chars_out,
        lines_in=lines_in,
        lines_out=lines_out,
        hyphens_joined=n_hyphens,
        control_chars_removed=removed_ctrl,
    )
    return out, stats
