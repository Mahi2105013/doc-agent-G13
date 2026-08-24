"""LLM — answer post-process / format / abstention"""
from __future__ import annotations
import re
from ..contracts import Answer, Citation  # noqa
from ..logging_conf import get_logger

logger = get_logger(__name__)


def format_answer(raw: str, citations: list) -> Answer:
    """Attach citations, set grounded/confidence, enforce abstention.

    Parameters
    ----------
    raw : str
        The raw text produced by the LLM synthesize call.
    citations : list
        List of Citation objects or dicts already accumulated.
    """
    # Normalise citation list
    normalised: list[Citation] = []
    for c in citations:
        if isinstance(c, Citation):
            normalised.append(c)
        elif isinstance(c, dict):
            try:
                normalised.append(Citation(**c))
            except Exception:
                pass

    # Parse ABSTAIN sentinel
    if raw.strip().startswith("ABSTAIN:"):
        return Answer(text=raw.strip(), citations=[], grounded=False, confidence=0.0)

    # Parse trailing "Confidence: 0.85" line
    confidence = 0.5
    lines = raw.strip().splitlines()
    if lines:
        m = re.search(r"confidence[:\s]+([0-9.]+)", lines[-1], re.IGNORECASE)
        if m:
            try:
                confidence = max(0.0, min(1.0, float(m.group(1))))
                raw = "\n".join(lines[:-1]).strip()
            except ValueError:
                pass

    grounded = len(normalised) > 0
    return Answer(
        text=raw,
        citations=normalised,
        grounded=grounded,
        confidence=confidence,
    )


def register(hooks) -> None:
    """Wire the grounding / abstention gate at BEFORE_ANSWER.

    If the answer text starts with ABSTAIN or has no citations, mark it
    as ungrounded.  This is a safety net; the primary check happens in
    agent.synthesize().
    """
    def _ground(ctx: dict) -> dict:
        state = ctx.get("state", {})
        # Find the pending answer in state if already built, or skip.
        # The hook runs *before* synthesize() returns, so we patch ctx.
        # If an answer is already in ctx, re-check grounding.
        ans = ctx.get("answer")
        if ans is not None and isinstance(ans, Answer):
            if ans.text.strip().startswith("ABSTAIN:"):
                ctx["answer"] = Answer(
                    text=ans.text, citations=[], grounded=False, confidence=0.0
                )
            elif not ans.citations:
                # No citations → downgrade confidence but keep answer.
                ctx["answer"] = Answer(
                    text=ans.text,
                    citations=[],
                    grounded=False,
                    confidence=min(ans.confidence, 0.3),
                )
        return ctx

    hooks.register(hooks.BEFORE_ANSWER, _ground)
