"""Evaluation metrics — pure NumPy.

Why these are written from scratch even though sklearn ships them: the goal
of the project is to make every step inspectable. Computing precision /
recall / F1 by hand also makes the failure modes (e.g., divide-by-zero on
empty positive predictions) explicit rather than hidden behind library defaults.
"""
from __future__ import annotations

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 2) -> np.ndarray:
    """Return a (n_classes, n_classes) matrix where cm[t, p] = count of (true=t, pred=p)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of correctly classified samples."""
    return float(np.mean(np.asarray(y_true) == np.asarray(y_pred)))


def precision_recall_f1(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    positive_class: int = 1,
) -> tuple[float, float, float]:
    """Per-class precision, recall, and F1 against `positive_class`.

    precision = TP / (TP + FP) — of those we called positive, what fraction were?
    recall    = TP / (TP + FN) — of the actual positives, what fraction did we catch?
    F1        = 2·P·R / (P + R) — harmonic mean.

    Returns (0.0, 0.0, 0.0) for any metric with a zero denominator rather than
    raising. This matches sklearn's `zero_division=0` default.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    tp = int(((y_pred == positive_class) & (y_true == positive_class)).sum())
    fp = int(((y_pred == positive_class) & (y_true != positive_class)).sum())
    fn = int(((y_pred != positive_class) & (y_true == positive_class)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return precision, recall, f1


def classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str] | None = None,
) -> str:
    """Human-readable per-class metrics summary."""
    classes = np.unique(np.concatenate([np.asarray(y_true), np.asarray(y_pred)]))
    names = class_names or [str(c) for c in classes]

    lines = [f"{'class':<10}{'precision':>12}{'recall':>10}{'F1':>8}{'support':>10}"]
    for c, name in zip(classes, names):
        p, r, f = precision_recall_f1(y_true, y_pred, positive_class=int(c))
        support = int((np.asarray(y_true) == c).sum())
        lines.append(f"{name:<10}{p:>12.4f}{r:>10.4f}{f:>8.4f}{support:>10}")

    acc = accuracy(y_true, y_pred)
    lines.append("")
    lines.append(f"accuracy: {acc:.4f}  ({len(y_true)} samples)")
    return "\n".join(lines)


if __name__ == "__main__":
    # Hand-checked toy example.
    # 5 ham (0) and 3 spam (1). Predictions: 1 spam misclassified, 1 ham misclassified.
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 0, 0, 1, 1])
    # TP = 2, FP = 1, FN = 1, TN = 4
    # precision = 2/3 = 0.667; recall = 2/3 = 0.667; F1 = 0.667; accuracy = 6/8 = 0.75
    print(classification_report(y_true, y_pred, class_names=["ham", "spam"]))
    print()
    print("Confusion matrix:")
    print(confusion_matrix(y_true, y_pred))
