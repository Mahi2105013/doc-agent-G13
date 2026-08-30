"""Data — corpus versioning (which corpus version -> which result)"""
from __future__ import annotations
import hashlib
import json
import time
from pathlib import Path


def snapshot(corpus_dir: str) -> str:
    """Hash the filenames+sizes in corpus_dir and return a deterministic version id.
    Records the snapshot to traces/corpus_versions.jsonl for audit.
    """
    p = Path(corpus_dir)
    if not p.exists():
        return "empty"

    # Build a stable hash from sorted (filename, size) pairs
    entries = sorted(
        (f.name, f.stat().st_size)
        for f in p.rglob("*") if f.is_file()
    )
    digest = hashlib.sha256(
        json.dumps(entries, sort_keys=True).encode()
    ).hexdigest()[:12]
    version_id = f"v_{digest}"

    # Record for audit
    traces_dir = Path(__file__).resolve().parents[4] / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    record = {"version_id": version_id, "corpus_dir": str(corpus_dir),
              "ts": time.time(), "n_files": len(entries)}
    with open(traces_dir / "corpus_versions.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    return version_id
