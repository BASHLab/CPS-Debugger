"""Metrics for multi-label trace prediction."""

from __future__ import annotations

from typing import Dict, Iterable, List, Sequence

import numpy as np
from sklearn.metrics import f1_score


def threshold_predictions(probs: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (probs >= threshold).astype(np.int8)


def precision_at_k(y_true: np.ndarray, probs: np.ndarray, k: int) -> float:
    """Mean precision@k for multi-label predictions."""
    if y_true.size == 0:
        return 0.0
    n_samples, n_labels = y_true.shape
    if n_labels == 0:
        return 0.0

    k_eff = min(k, n_labels)
    # argpartition keeps this O(n_labels) per sample.
    topk_idx = np.argpartition(-probs, kth=k_eff - 1, axis=1)[:, :k_eff]
    hits = np.take_along_axis(y_true, topk_idx, axis=1).sum(axis=1)
    return float(np.mean(hits / float(k_eff)))


def compute_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    threshold: float = 0.5,
    ks: Sequence[int] = (5, 10),
) -> Dict[str, float]:
    y_pred = threshold_predictions(probs, threshold=threshold)
    out: Dict[str, float] = {
        "micro_f1": float(f1_score(y_true.ravel(), y_pred.ravel(), zero_division=0)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    for k in ks:
        out[f"precision_at_{k}"] = precision_at_k(y_true, probs, k)
    return out


def per_class_f1_top_n(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    label_names: Sequence[str],
    n: int = 10,
) -> List[Dict[str, float]]:
    """Per-class F1 for top-n most frequent labels in ``y_true``."""
    supports = y_true.sum(axis=0)
    top_idx = np.argsort(supports)[::-1][:n]
    rows = []
    for idx in top_idx:
        rows.append(
            {
                "label": label_names[idx],
                "support": int(supports[idx]),
                "f1": float(f1_score(y_true[:, idx], y_pred[:, idx], zero_division=0)),
            }
        )
    return rows

