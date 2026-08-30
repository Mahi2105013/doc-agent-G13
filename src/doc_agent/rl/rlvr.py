"""Stage 7 — RLVR — verifiable reward on extraction accuracy (E22)"""
from __future__ import annotations


def verifiable_reward(prediction: str, gold: str) -> float:
    """+1 if extraction exactly matches gold (after normalisation), else 0.
    Drives RLVR/GRPO for verifiable fact-extraction tasks."""
    def _norm(s: str) -> str:
        return " ".join(str(s).strip().lower().split())
    return 1.0 if _norm(prediction) == _norm(gold) else 0.0
