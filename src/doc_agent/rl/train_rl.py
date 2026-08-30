"""Stage 7 — RL/RLVR training loop (E22 bonus)"""
from __future__ import annotations


def train(cfg: dict) -> None:
    """Train the tool-selection policy using cfg['rl']['algo'].
    PPO/GRPO implementations go here. Currently a graceful no-op
    (set cfg['rl']['train_policy'] = true to activate).
    """
    rl_cfg = cfg.get("rl", {})
    algo = rl_cfg.get("algo", "ppo")
    train_policy = rl_cfg.get("train_policy", False)

    if not train_policy:
        print(f"[train_rl] train_policy=false — skipping RL training (algo={algo}).")
        return

    # Dispatch to algorithm-specific trainer
    if algo == "ppo":
        _train_ppo(cfg)
    elif algo == "grpo":
        _train_grpo(cfg)
    else:
        raise ValueError(f"Unknown RL algo: {algo!r}. Choose 'ppo' or 'grpo'.")


def _train_ppo(cfg: dict) -> None:
    """PPO training stub — extend with trl/PPOTrainer when ready."""
    raise RuntimeError(
        "[train_rl] PPO training requires the `trl` library, which is critical for RL. "
        "Please install it with: pip install trl"
    )


def _train_grpo(cfg: dict) -> None:
    """GRPO (Group Relative Policy Optimisation) stub for RLVR."""
    from .rlvr import verifiable_reward  # noqa — ensure importable
    raise RuntimeError(
        "[train_rl] GRPO training requires a policy model and `trl`. "
        "Please install it with: pip install trl"
    )
