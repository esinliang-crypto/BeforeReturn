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


def validate_binary_inputs(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score, dtype=float)

    if y_true.size == 0 or y_score.size == 0:
        raise ValueError("y_true and y_score must not be empty.")
    if len(y_true) != len(y_score):
        raise ValueError("y_true and y_score must have the same length.")
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("y_true must contain only 0/1 labels.")
    if np.isnan(y_score).any():
        raise ValueError("y_score must not contain NaN values.")
    if ((y_score < 0) | (y_score > 1)).any():
        raise ValueError("y_score probabilities must be in [0, 1].")
    return y_true.astype(int), y_score


def top_fraction_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    fraction: float = 0.1,
) -> dict[str, float | int]:
    if not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1].")
    y_true, y_score = validate_binary_inputs(y_true, y_score)
    top_n = max(1, int(len(y_true) * fraction))
    top_idx = np.argsort(y_score)[::-1][:top_n]
    top_y_true = y_true[top_idx]
    positives = y_true.sum()
    positive_rate = float(y_true.mean())
    precision_at_top = float(top_y_true.mean())
    recall_at_top = 0.0 if positives == 0 else float(top_y_true.sum() / positives)
    lift_at_top = 0.0 if positive_rate == 0 else precision_at_top / positive_rate
    return {
        "top_n": int(top_n),
        "recall": recall_at_top,
        "precision": precision_at_top,
        "lift": float(lift_at_top),
    }


def recall_at_top_fraction(y_true: np.ndarray, y_score: np.ndarray, fraction: float = 0.1) -> float:
    return float(top_fraction_metrics(y_true, y_score, fraction)["recall"])


def expected_calibration_error(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bins: int = 10,
) -> float:
    if n_bins <= 0:
        raise ValueError("n_bins must be positive.")
    y_true, y_score = validate_binary_inputs(y_true, y_score)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for index in range(n_bins):
        lower = bin_edges[index]
        upper = bin_edges[index + 1]
        if index == n_bins - 1:
            mask = (y_score >= lower) & (y_score <= upper)
        else:
            mask = (y_score >= lower) & (y_score < upper)
        if not mask.any():
            continue
        bin_confidence = float(y_score[mask].mean())
        bin_accuracy = float(y_true[mask].mean())
        ece += float(mask.mean()) * abs(bin_confidence - bin_accuracy)
    return float(ece)


def constant_baseline_brier(y_true: np.ndarray) -> float:
    y_true = np.asarray(y_true)
    if y_true.size == 0:
        raise ValueError("y_true must not be empty.")
    if not np.isin(y_true, [0, 1]).all():
        raise ValueError("y_true must contain only 0/1 labels.")
    positive_rate = float(y_true.mean())
    return positive_rate * (1 - positive_rate)


def overview_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    logistic_pr_auc: float,
    model_version: str,
    evaluation_timestamp: str,
    data_processing_version: str,
) -> dict[str, Any]:
    y_true, y_score = validate_binary_inputs(y_true, y_score)
    positive_rate = float(y_true.mean())
    catboost_pr_auc = float(average_precision_score(y_true, y_score))
    top_10 = top_fraction_metrics(y_true, y_score, 0.1)
    model_brier = float(brier_score_loss(y_true, y_score))
    baseline_brier = constant_baseline_brier(y_true)
    binary = binary_metrics(pd.Series(y_true), y_score)
    brier_skill_score = 0.0 if baseline_brier == 0 else 1 - model_brier / baseline_brier

    return {
        "test_positive_rate": positive_rate,
        "logistic_pr_auc": float(logistic_pr_auc),
        "catboost_pr_auc": catboost_pr_auc,
        "pr_auc_absolute_gain": catboost_pr_auc - positive_rate,
        "pr_auc_relative_gain": (
            0.0 if positive_rate == 0 else (catboost_pr_auc - positive_rate) / positive_rate
        ),
        "recall_at_10": top_10["recall"],
        "precision_at_10": top_10["precision"],
        "lift_at_10": top_10["lift"],
        "top_10_sample_count": top_10["top_n"],
        "brier_score": model_brier,
        "constant_baseline_brier": baseline_brier,
        "brier_skill_score": brier_skill_score,
        "roc_auc": binary["roc_auc"],
        "f1": binary["f1"],
        "precision": binary["precision"],
        "recall": binary["recall"],
        "ece": expected_calibration_error(y_true, y_score),
        "test_sample_count": int(len(y_true)),
        "model_version": model_version,
        "evaluation_timestamp": evaluation_timestamp,
        "data_processing_version": data_processing_version,
    }


def binary_metrics(
    y_true: pd.Series,
    y_score: np.ndarray,
    threshold: float = 0.5,
) -> dict[str, Any]:
    y_true_array, y_score = validate_binary_inputs(y_true.to_numpy(), y_score)
    y_pred = (y_score >= threshold).astype(int)
    matrix = confusion_matrix(y_true_array, y_pred, labels=[0, 1])
    prob_true, prob_pred = calibration_curve(y_true_array, y_score, n_bins=10, strategy="quantile")
    top_10 = top_fraction_metrics(y_true_array, y_score, 0.1)

    return {
        "pr_auc": float(average_precision_score(y_true_array, y_score)),
        "roc_auc": float(roc_auc_score(y_true_array, y_score)),
        "f1": float(f1_score(y_true_array, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true_array, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_array, y_pred, zero_division=0)),
        "recall_at_top_10_pct": top_10["recall"],
        "precision_at_top_10_pct": top_10["precision"],
        "lift_at_top_10_pct": top_10["lift"],
        "brier_score": float(brier_score_loss(y_true_array, y_score)),
        "ece": expected_calibration_error(y_true_array, y_score),
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
