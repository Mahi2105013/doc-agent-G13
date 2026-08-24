"""FIXED — structured logging (auditable NFR). Use get_logger(), never print()."""
from __future__ import annotations
import json
import logging
import sys
import time
from pathlib import Path

_step_counter: list[int] = [0]   # mutable list for closure mutation

def get_logger(name: str) -> logging.Logger:
    lg = logging.getLogger(name)
    if not lg.handlers:
        # Force UTF-8 so Bengali text and special chars never cause UnicodeEncodeError
        # on Windows cp1252 consoles.
        try:
            stream = open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)
        except Exception:
            stream = sys.stdout
        h = logging.StreamHandler(stream)
        h.setFormatter(logging.Formatter(
            '{"ts":"%(asctime)s","lvl":"%(levelname)s","mod":"%(name)s","msg":"%(message)s"}'
        ))
        lg.addHandler(h)
        lg.setLevel(logging.INFO)
    return lg

_logger = get_logger(__name__)

# Resolve traces directory relative to the project root (3 levels above src/doc_agent/).
_TRACES_DIR: Path = Path(__file__).resolve().parents[2] / "traces"
_TRACE_FILE: Path = _TRACES_DIR / "run.jsonl"

def _ensure_traces_dir() -> None:
    _TRACES_DIR.mkdir(parents=True, exist_ok=True)

def _append_trace(record: dict) -> None:
    """Append one JSON record to traces/run.jsonl (one line per step)."""
    _ensure_traces_dir()
    try:
        with _TRACE_FILE.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception as exc:
        _logger.warning("Tracing: could not write to %s: %s", _TRACE_FILE, exc)


def register(hooks) -> None:
    """Wire structured tracing at each seam (auditable trail) AND emit traces/run.jsonl.
    Each seam appends a contracts.TraceStep line to traces/run.jsonl so the A3 agentic-feature
    check can read the trajectory (path must depend on observations).
    """

    def _trace_step(ctx: dict) -> dict:
        _step_counter[0] += 1
        state = ctx.get("state", {})
        record = {
            "step":   _step_counter[0],
            "seam":   "on_step",
            "tool":   "decide",
            "args":   {"query": state.get("query", "")[:120]},
            "obs":    {"n_obs": len(state.get("obs", []))},
            "ts":     time.time(),
        }
        _logger.info("TRACE on_step: step=%d query=%r", record["step"], record["args"]["query"])
        _append_trace(record)
        return ctx

    def _trace_tool_call(ctx: dict) -> dict:
        _step_counter[0] += 1
        action = ctx.get("action", {})
        tool_name = action.get("tool", "unknown")
        args = action.get("args", {})
        obs_data = action.get("_obs", {})
        record = {
            "step":  _step_counter[0],
            "seam":  "on_tool_call",
            "tool":  tool_name,
            "args":  {k: str(v)[:200] for k, v in args.items()},
            "obs":   obs_data,
            "ts":    time.time(),
        }
        _logger.info(
            "TRACE on_tool_call: step=%d tool=%s top_score=%s k=%s",
            record["step"], tool_name,
            obs_data.get("top_score", "?"), obs_data.get("k", "?"),
        )
        _append_trace(record)
        return ctx

    def _trace_after_answer(ctx: dict) -> dict:
        _step_counter[0] += 1
        ans = ctx.get("answer")
        record = {
            "step":   _step_counter[0],
            "seam":   "after_answer",
            "tool":   "answer",
            "args":   {},
            "obs": {
                "grounded":    getattr(ans, "grounded", None),
                "confidence":  getattr(ans, "confidence", None),
                "n_citations": len(getattr(ans, "citations", [])),
                "text_len":    len(getattr(ans, "text", "")),
            },
            "ts":  time.time(),
        }
        _logger.info(
            "TRACE after_answer: grounded=%s confidence=%.2f citations=%d",
            record["obs"]["grounded"],
            record["obs"]["confidence"] or 0.0,
            record["obs"]["n_citations"],
        )
        _append_trace(record)
        return ctx

    hooks.register(hooks.ON_STEP,      _trace_step)
    hooks.register(hooks.ON_TOOL_CALL, _trace_tool_call)
    hooks.register(hooks.AFTER_ANSWER, _trace_after_answer)
