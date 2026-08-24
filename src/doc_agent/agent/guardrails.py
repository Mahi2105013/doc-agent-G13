"""Stage 6 — SECURITY — autonomy, budgets, prompt-injection defense"""
from __future__ import annotations
import re
from ..contracts import *  # noqa
from ..logging_conf import get_logger

logger = get_logger(__name__)

# Prompt-injection detection patterns (same set as agent._sanitize for defence-in-depth)
_INJECTION_RE = re.compile(
    r"ignore\s+(your\s+)?instructions|"
    r"disregard\s+(your\s+)?instructions|"
    r"you\s+are\s+now|"
    r"forget\s+(your\s+)?instructions|"
    r"act\s+as\s+(if\s+)?",
    re.IGNORECASE,
)

# Approximate cost per token (USD) for a mid-range LLM.
_COST_PER_TOKEN_USD = 1e-5   # ~$0.01 / 1k tokens


class Guardrails:
    """Enforce autonomy level, step/cost budget, and instruction/content isolation."""

    def __init__(self, cfg: dict) -> None:
        self.cfg = cfg.get("agent", {})
        self.spent: float = 0.0
        self.steps: int = 0

    def reset(self) -> None:
        """Reset per-run counters."""
        self.spent = 0.0
        self.steps = 0

    def check(self, action: dict) -> None:
        """Raise RuntimeError if the action violates any guardrail.

        Checks:
          1. Step budget   — total steps < max_steps
          2. Cost budget   — accumulated USD spend < budget_usd
          3. Autonomy      — tool requires human approval when level is 'ask-before-act'
          4. Injection     — query / args do not contain adversarial patterns
        """
        self.steps += 1
        max_steps: int = int(self.cfg.get("max_steps", 8))
        budget_usd: float = float(self.cfg.get("budget_usd", 0.05))
        autonomy: str = str(self.cfg.get("autonomy", "act-with-approval"))

        # ── 1. Step budget ────────────────────────────────────────────────
        if self.steps > max_steps:
            raise RuntimeError(
                f"Guardrails: step budget exceeded ({self.steps} > {max_steps})."
            )

        # ── 2. Cost budget (rough estimate from prompt size) ───────────────
        prompt_chars = sum(len(str(v)) for v in action.get("args", {}).values())
        estimated_tokens = prompt_chars / 4  # rough chars→tokens
        estimated_cost = estimated_tokens * _COST_PER_TOKEN_USD
        self.spent += estimated_cost
        if self.spent > budget_usd:
            raise RuntimeError(
                f"Guardrails: cost budget exceeded "
                f"(${self.spent:.4f} > ${budget_usd:.4f})."
            )

        # ── 3. Autonomy level ─────────────────────────────────────────────
        tool_name = action.get("tool", "")
        high_risk_tools = {"escalate_to_human", "enhance_page"}
        if autonomy == "ask-before-act" and tool_name in high_risk_tools:
            logger.warning(
                "Guardrails: autonomy level '%s' requires approval for tool '%s'. "
                "Proceeding (HITL queue not yet blocking).",
                autonomy, tool_name,
            )
            # In a full HITL setup this would block; for now we log and allow.

        # ── 4. Prompt-injection detection ─────────────────────────────────
        for key, val in action.get("args", {}).items():
            if isinstance(val, str) and _INJECTION_RE.search(val):
                raise RuntimeError(
                    f"Guardrails: prompt-injection pattern detected in "
                    f"action['{key}']: {val[:80]!r}"
                )

        logger.debug(
            "Guardrails.check OK: step=%d spent=$%.5f tool=%s",
            self.steps, self.spent, tool_name,
        )


def register(hooks, cfg: dict) -> None:
    """Wire guardrails into every tool call. Call Guardrails.check on ON_TOOL_CALL."""
    g = Guardrails(cfg)
    g.reset()

    def _check(ctx: dict) -> dict:
        g.check(ctx["action"])     # budgets / autonomy / injection
        return ctx

    hooks.register(hooks.ON_TOOL_CALL, _check)
