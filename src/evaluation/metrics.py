from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def recall_at_top_fraction(y_true: np.ndarray, y_score: np.ndarray, fraction: float = 0.1) -> float:
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1].")
    top_n = max(1, int(len(y_true) * fraction))
    order = np.argsort(y_score)[::-1][:top_n]
    positives = y_true.sum()
    if positives == 0:
        return 0.0
    return float(y_true[order].sum() / positives)


def binary_metrics(
    y_true: pd.Series,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    y_true_array = y_true.to_numpy()
    y_pred = (y_score >= threshold).astype(int)
    matrix = confusion_matrix(y_true_array, y_pred, labels=[0, 1])
    prob_true, prob_pred = calibration_curve(y_true_array, y_score, n_bins=10, strategy="quantile")

    return {
        "pr_auc": float(average_precision_score(y_true_array, y_score)),
        "roc_auc": float(roc_auc_score(y_true_array, y_score)),
        "f1": float(f1_score(y_true_array, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true_array, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_array, y_pred, zero_division=0)),
        "recall_at_top_10_pct": recall_at_top_fraction(y_true_array, y_score, 0.1),
        "brier_score": float(brier_score_loss(y_true_array, y_score)),
        "threshold": threshold,
        "confusion_matrix": {
            "tn": int(matrix[0, 0]),
            "fp": int(matrix[0, 1]),
            "fn": int(matrix[1, 0]),
            "tp": int(matrix[1, 1]),
        },
        "calibration_curve": [
            {"predicted": float(predicted), "observed": float(observed)}
            for predicted, observed in zip(prob_pred, prob_true, strict=False)
        ],
    }


def slice_metrics(
    frame: pd.DataFrame,
    y_score: np.ndarray,
    slice_column: str,
    min_rows: int = 5_000,
    max_slices: int = 10,
) -> list[dict[str, Any]]:
    if slice_column not in frame.columns:
        return []

    scored = frame[[slice_column, "isReturned"]].copy()
    scored["score"] = y_score
    rows = []
    for value, group in scored.groupby(slice_column, dropna=False):
        if len(group) < min_rows:
            continue
        rows.append(
            {
                "slice_column": slice_column,
                "slice_value": str(value),
                "rows": int(len(group)),
                "return_rate": float(group["isReturned"].mean()),
                "pr_auc": float(average_precision_score(group["isReturned"], group["score"])),
                "roc_auc": float(roc_auc_score(group["isReturned"], group["score"])),
            }
        )
    return sorted(rows, key=lambda row: row["rows"], reverse=True)[:max_slices]
