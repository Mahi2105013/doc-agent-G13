"""MLOps — experiment tracking (E20)
Tries W&B; falls back to local JSON if W&B is unavailable.
"""
from __future__ import annotations
import json
import os
import time
from pathlib import Path

_run: dict = {}
_log_path: Path | None = None


def init_run(cfg: dict, tags: list[str]) -> None:
    """Start a tracking run; log config. Falls back to local JSON if wandb absent."""
    global _run, _log_path

    _run = {
        "cfg": cfg,
        "tags": tags,
        "start_time": time.time(),
        "metrics": [],
    }

    # Try W&B first
    try:
        import wandb  # type: ignore
        wandb.init(
            project=cfg.get("project", "doc-agent-g13"),
            config=cfg,
            tags=tags,
            reinit=True,
        )
        _run["backend"] = "wandb"
        return
    except Exception:
        pass

    # Local JSON fallback
    traces_dir = Path(__file__).resolve().parents[3] / "traces"
    traces_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"run_{int(time.time())}"
    _log_path = traces_dir / f"{run_id}.json"
    _run["run_id"] = run_id
    _run["backend"] = "local_json"
    _persist()


def log(metrics: dict) -> None:
    """Log a metrics dict to the active run."""
    global _run
    if not _run:
        return
    entry = {"t": time.time(), **metrics}
    _run.setdefault("metrics", []).append(entry)

    if _run.get("backend") == "wandb":
        try:
            import wandb  # type: ignore
            wandb.log(metrics)
            return
        except Exception:
            pass

    _persist()


def _persist() -> None:
    if _log_path:
        with open(_log_path, "w", encoding="utf-8") as f:
            json.dump(_run, f, indent=2, default=str)

