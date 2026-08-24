"""Stage 6 — working/episodic memory

Stores every observation the agent receives during a reasoning session and
allows retrieval of the most relevant prior observations for a new query.

Design: token-overlap scoring (works for Bengali + English with no extra deps).
"""
from __future__ import annotations
from collections import deque
from ..contracts import ToolResult  # noqa
from ..logging_conf import get_logger

logger = get_logger(__name__)

# Maximum number of observations to keep in the working window.
_MAX_ITEMS = 64


class Memory:
    """Simple episodic/working memory backed by a bounded deque.

    ``add()`` stores any observation (ToolResult, dict, str, etc.).
    ``recall()`` returns the most relevant stored items for a query using
    token-overlap scoring — no heavy dependencies, works offline.
    """

    def __init__(self) -> None:
        self.items: deque = deque(maxlen=_MAX_ITEMS)

    # ------------------------------------------------------------------
    def add(self, item) -> None:
        """Store a new observation.  ``item`` may be a ToolResult, dict, or str."""
        self.items.append(item)
        logger.debug("Memory.add: total stored items=%d", len(self.items))

    # ------------------------------------------------------------------
    def recall(self, query: str) -> list:
        """Return up to 5 stored items most relevant to *query*.

        Relevance is measured by the number of distinct query tokens that
        appear in the string representation of each item.  Items with no
        overlap are excluded; results are returned best-first.
        """
        if not self.items:
            return []

        query_tokens: set[str] = set(query.lower().split())
        if not query_tokens:
            # No query tokens → return the most recent items.
            return list(self.items)[-5:]

        scored: list[tuple[float, int]] = []
        for i, item in enumerate(self.items):
            item_str = self._to_str(item).lower()
            item_tokens = set(item_str.split())
            overlap = len(query_tokens & item_tokens)
            if overlap > 0:
                # Normalise by query length so precision matters as well as count.
                score = overlap / len(query_tokens)
                scored.append((score, i))

        # Sort descending by score, take top-5.
        scored.sort(key=lambda x: x[0], reverse=True)
        items_list = list(self.items)
        return [items_list[i] for _, i in scored[:5]]

    # ------------------------------------------------------------------
    @staticmethod
    def _to_str(item) -> str:
        """Convert any observation type to a plain string for scoring."""
        if isinstance(item, str):
            return item
        if isinstance(item, ToolResult):
            parts = [str(item.ok)]
            payload = item.payload
            if isinstance(payload, dict):
                # Include chunk texts and keys for token matching.
                for k, v in payload.items():
                    parts.append(str(k))
                    if isinstance(v, str):
                        parts.append(v)
                    elif isinstance(v, list):
                        for el in v:
                            if isinstance(el, dict):
                                parts.append(el.get("text", ""))
                            else:
                                parts.append(str(el))
            return " ".join(parts)
        if isinstance(item, dict):
            return " ".join(str(v) for v in item.values())
        return str(item)
