"""HITL — persistent review queue (survives restarts) — backed by JSON file."""
from __future__ import annotations
import json
import time
import uuid
from pathlib import Path

_STORE_PATH = Path(__file__).resolve().parents[4] / "traces" / "hitl_queue.json"


def _load() -> list[dict]:
    if _STORE_PATH.exists():
        try:
            return json.loads(_STORE_PATH.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def _save(items: list[dict]) -> None:
    _STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    _STORE_PATH.write_text(json.dumps(items, indent=2, default=str), encoding="utf-8")


def enqueue(item: dict) -> str:
    """Persist a pending review item; return its id."""
    items = _load()
    item_id = str(uuid.uuid4())[:8]
    items.append({"id": item_id, "status": "pending", "ts": time.time(), **item})
    _save(items)
    return item_id


def pending() -> list[dict]:
    """Return all items that are still pending human review."""
    return [i for i in _load() if i.get("status") == "pending"]


def resolve(item_id: str, decision: str) -> None:
    """Mark an item as resolved with the given decision ('approve' or 'reject')."""
    items = _load()
    for item in items:
        if item.get("id") == item_id:
            item["status"] = "resolved"
            item["decision"] = decision
            item["resolved_ts"] = time.time()
            break
    _save(items)

