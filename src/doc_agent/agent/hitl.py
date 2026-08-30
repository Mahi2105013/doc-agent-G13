"""HITL — human-in-the-loop review queue (E13)"""
from __future__ import annotations
from ..contracts import ToolResult
from . import hitl_store


def escalate(reason: str, context: dict) -> ToolResult:
    """Queue an agent step for human review and return a pending ToolResult.
    The agent will treat this as a low-confidence answer requiring human approval."""
    item_id = hitl_store.enqueue({"reason": reason, "context": context})
    return ToolResult(
        ok=False,
        payload={"status": "pending_human_review", "item_id": item_id, "reason": reason},
    )


def review_queue() -> list[dict]:
    """Return all pending items for the reviewer UI."""
    return hitl_store.pending()
