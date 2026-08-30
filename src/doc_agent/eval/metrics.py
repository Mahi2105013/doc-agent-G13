"""Stage 9 — metrics (E14)"""
from __future__ import annotations
from collections import Counter
from ..contracts import Answer


def ocr_f1(pred: str, gold: str) -> float:
    """Token-level F1 between predicted OCR text and gold transcription."""
    pred_tokens = pred.split()
    gold_tokens = gold.split()
    if not pred_tokens or not gold_tokens:
        return 0.0
    pred_counts = Counter(pred_tokens)
    gold_counts = Counter(gold_tokens)
    overlap = sum((pred_counts & gold_counts).values())
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def recall_at_k(retrieved: list, gold: list, k: int) -> float:
    """Fraction of gold items found in the top-k retrieved items."""
    if not gold:
        return 1.0
    top_k_ids = {
        (r.id if hasattr(r, "id") else str(r))
        for r in retrieved[:k]
    }
    gold_ids = {(g if isinstance(g, str) else g) for g in gold}
    return len(top_k_ids & gold_ids) / len(gold_ids)


def groundedness(answer: Answer) -> float:
    """1.0 if answer is grounded AND has at least one citation, else 0.0.
    This is the no-hallucination gate required by the precision-first NFR."""
    if answer.grounded and len(answer.citations) >= 1:
        return 1.0
    return 0.0


def citation_accuracy(answer: Answer) -> float:
    """Fraction of citations whose chunk_id is non-empty and span is valid."""
    if not answer.citations:
        return 0.0
    valid = sum(
        1 for c in answer.citations
        if c.chunk_id and len(c.span) == 2 and c.span[0] <= c.span[1]
    )
    return valid / len(answer.citations)


def ece(confidences: list[float], correct: list[bool], n_bins: int = 10) -> float:
    """Expected Calibration Error: average |confidence - accuracy| weighted by bin size."""
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


def subgroup_gap(scores_by_group: dict[str, list[float]]) -> float:
    """Max pairwise difference in mean scores across subgroups (fairness metric)."""
    if len(scores_by_group) < 2:
        return 0.0
    means = {g: sum(v) / len(v) for g, v in scores_by_group.items() if v}
    values = list(means.values())
    return max(values) - min(values)
