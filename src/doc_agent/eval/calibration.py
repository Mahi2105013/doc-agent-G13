"""Stage 9 — confidence calibration (calibrated-confidence NFR, E17)"""
from __future__ import annotations


def temperature_scale(confidences: list[float], labels: list[bool]) -> float:
    """Fit a temperature T on a validation set by minimising NLL.
    Returns the scalar T. Apply via conf_scaled = conf ** (1/T).
    Simple grid search implementation (no torch required).
    """
    import math
    best_t, best_nll = 1.0, float("inf")
    for t in [t / 10 for t in range(1, 51)]:  # 0.1 … 5.0
        nll = 0.0
        for c, y in zip(confidences, labels):
            c_scaled = min(max(c ** (1 / t), 1e-9), 1 - 1e-9)
            nll -= math.log(c_scaled) if y else math.log(1 - c_scaled)
        if nll < best_nll:
            best_nll, best_t = nll, t
    return best_t


def ece(confidences: list[float], correct: list[bool], n_bins: int = 10) -> float:
    """Expected Calibration Error (E17 target: ECE ≤ 0.05).
    Bins predictions by confidence; returns weighted avg |conf - acc|."""
    if not confidences:
        return 0.0
    bin_size = 1.0 / n_bins
    total_ece = 0.0
    n = len(confidences)
    for b in range(n_bins):
        lo = b * bin_size
        hi = lo + bin_size
        indices = [i for i, c in enumerate(confidences) if lo <= c < hi]
        if not indices:
            continue
        bin_conf = sum(confidences[i] for i in indices) / len(indices)
        bin_acc = sum(1 for i in indices if correct[i]) / len(indices)
        total_ece += (len(indices) / n) * abs(bin_conf - bin_acc)
    return total_ece
