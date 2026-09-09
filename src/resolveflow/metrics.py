"""Shared evaluation metrics for baselines and retrieval."""

from __future__ import annotations

from collections import Counter
from typing import Any, Iterable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def intent_metrics(
    y_true: list[str], y_pred: list[str], labels: list[str] | None = None
) -> dict[str, Any]:
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0)),
        "weighted_f1": float(
            f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0)
        ),
        "per_intent": _per_intent(y_true, y_pred, labels),
        "confusion_matrix": {
            "labels": labels,
            "matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        },
        "classification_report": classification_report(
            y_true, y_pred, labels=labels, zero_division=0, output_dict=True
        ),
    }


def _per_intent(y_true, y_pred, labels) -> dict[str, dict[str, float]]:
    p, r, f, s = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, zero_division=0
    )
    out = {}
    for i, lab in enumerate(labels):
        out[lab] = {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f[i]),
            "support": int(s[i]),
        }
    return out


def escalation_metrics(
    y_true_esc: list[str],
    y_pred_esc: list[str],
) -> dict[str, Any]:
    """
    Treat escalate as the positive class for P/R/F1.
    auto_handle / uncertain are non-escalate for binary metrics.
    """
    true_pos = [1 if y == "escalate" else 0 for y in y_true_esc]
    pred_pos = [1 if y == "escalate" else 0 for y in y_pred_esc]

    # Auto-handle rate among predictions
    auto = [y == "auto_handle" for y in y_pred_esc]
    auto_rate = float(np.mean(auto)) if auto else 0.0

    # False auto-handle: predicted auto_handle but gold says escalate
    false_auto = [
        yp == "auto_handle" and yt == "escalate"
        for yt, yp in zip(y_true_esc, y_pred_esc, strict=False)
    ]
    n_auto = sum(auto)
    false_auto_rate = (sum(false_auto) / n_auto) if n_auto else 0.0

    p, r, f, _ = precision_recall_fscore_support(
        true_pos, pred_pos, average="binary", zero_division=0
    )
    return {
        "escalate_precision": float(p),
        "escalate_recall": float(r),
        "escalate_f1": float(f),
        "auto_handle_rate": auto_rate,
        "false_auto_handle_rate": false_auto_rate,
        "pred_distribution": dict(Counter(y_pred_esc)),
        "true_distribution": dict(Counter(y_true_esc)),
    }


def recall_at_k(
    retrieved_intent_lists: Iterable[list[str]],
    gold_intents: Iterable[str],
    ks: list[int] | None = None,
) -> dict[str, float]:
    """Intent Recall@K proxy: ≥1 of top-K retrieved cases shares gold intent."""
    ks = ks or [1, 3, 5]
    gold_intents = list(gold_intents)
    retrieved_intent_lists = list(retrieved_intent_lists)
    out = {}
    for k in ks:
        hits = 0
        for gold, retrieved in zip(gold_intents, retrieved_intent_lists, strict=False):
            top = retrieved[:k]
            if gold in top:
                hits += 1
        out[f"recall_at_{k}"] = hits / len(gold_intents) if gold_intents else 0.0
    return out
