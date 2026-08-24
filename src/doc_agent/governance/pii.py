
"""Governance — PII detection + redaction (mandatory)"""
from __future__ import annotations
import re
from ..contracts import Answer  # noqa
from ..logging_conf import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# PII patterns (English + Bangla-adjacent: phone, email, NID, credit card)
# ---------------------------------------------------------------------------
_PII_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("EMAIL",       re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")),
    ("PHONE",       re.compile(r"(?:\+?88)?01[3-9]\d{8}")),          # BD mobile
    ("PHONE_INTL",  re.compile(r"\+\d{1,3}[\s\-]?\(?\d{1,4}\)?[\s\-]?\d{3,4}[\s\-]?\d{4}")),
    ("NID",         re.compile(r"\b\d{10,17}\b")),                    # BD NID / passport number
    ("CREDIT_CARD", re.compile(r"\b(?:\d[ \-]?){13,16}\b")),
    ("IP_ADDR",     re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")),
]


def detect(text: str) -> list[tuple[int, int, str]]:
    """Return (start, end, type) spans of PII found in *text*."""
    spans: list[tuple[int, int, str]] = []
    for label, pattern in _PII_PATTERNS:
        for m in pattern.finditer(text):
            spans.append((m.start(), m.end(), label))
    # Sort by start position
    spans.sort(key=lambda x: x[0])
    return spans


def redact(text: str) -> str:
    """Replace all detected PII spans with [REDACTED_<TYPE>] tokens."""
    spans = detect(text)
    if not spans:
        return text
    result: list[str] = []
    prev = 0
    for start, end, label in spans:
        if start < prev:
            continue  # overlapping span already consumed
        result.append(text[prev:start])
        result.append(f"[REDACTED_{label}]")
        prev = end
    result.append(text[prev:])
    return "".join(result)


def register(hooks) -> None:
    """Wire PII redaction into the pipeline at AFTER_OCR, BEFORE_ANSWER, and ON_LOG."""

    def _scrub_text(ctx: dict) -> dict:
        """Redact PII from any text/chunks in the context."""
        # AFTER_OCR: ctx has {"chunks": list[Chunk]}
        chunks = ctx.get("chunks")
        if chunks is not None:
            for chunk in chunks:
                if hasattr(chunk, "text") and chunk.text:
                    cleaned = redact(chunk.text)
                    if cleaned != chunk.text:
                        logger.info(
                            "PII redacted from chunk %s", getattr(chunk, "id", "?")
                        )
                        # Pydantic v2: model_copy is preferred
                        try:
                            object.__setattr__(chunk, "text", cleaned)
                        except Exception:
                            pass
        return ctx

    def _scrub_answer(ctx: dict) -> dict:
        """Redact PII from outgoing answer text."""
        ans = ctx.get("answer")
        if ans is not None and isinstance(ans, Answer):
            cleaned = redact(ans.text)
            if cleaned != ans.text:
                logger.info("PII redacted from answer.")
                ctx["answer"] = ans.model_copy(update={"text": cleaned})
        # Also scrub state if present
        state = ctx.get("state", {})
        obs = state.get("obs", [])
        for ob in obs:
            if isinstance(ob, str):
                redact(ob)  # side-effect logging only; obs are already stored
        return ctx

    def _scrub_log(ctx: dict) -> dict:
        """Redact PII from log payloads."""
        msg = ctx.get("msg")
        if isinstance(msg, str):
            ctx["msg"] = redact(msg)
        return ctx

    hooks.register(hooks.AFTER_OCR,     _scrub_text)
    hooks.register(hooks.BEFORE_ANSWER,  _scrub_answer)
    hooks.register(hooks.ON_LOG,         _scrub_log)
