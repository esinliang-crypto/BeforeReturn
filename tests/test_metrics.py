import numpy as np
import pandas as pd
import pytest

from src.evaluation.metrics import (
    binary_metrics,
    constant_baseline_brier,
    overview_metrics,
    recall_at_top_fraction,
    top_fraction_metrics,
)


def test_recall_at_top_fraction() -> None:
    y_true = np.array([1, 0, 1, 0])
    y_score = np.array([0.9, 0.8, 0.1, 0.2])
    assert recall_at_top_fraction(y_true, y_score, 0.5) == 0.5


def test_binary_metrics_contains_required_keys() -> None:
    metrics = binary_metrics(pd.Series([0, 1, 1, 0]), np.array([0.1, 0.9, 0.8, 0.2]))
    assert "pr_auc" in metrics
    assert "recall_at_top_10_pct" in metrics
    assert metrics["confusion_matrix"] == {"tn": 2, "fp": 0, "fn": 0, "tp": 2}


def test_top_fraction_perfect_sorting_selects_best_ranked_examples() -> None:
    y_true = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    y_prob = np.array([0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1])
    metrics = top_fraction_metrics(y_true, y_prob, 0.1)
    assert metrics["top_n"] == 1
    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 0.5
    assert metrics["lift"] == 5.0


def test_top_fraction_constant_prediction_is_deterministic() -> None:
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    y_prob = np.full(10, 0.5)
    metrics = top_fraction_metrics(y_true, y_prob, 0.1)
    assert metrics["top_n"] == 1
    assert metrics["precision"] in {0.0, 1.0}


def test_top_fraction_when_sample_count_not_divisible_by_ten() -> None:
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1])
    y_prob = np.linspace(0, 1, 11)
    metrics = top_fraction_metrics(y_true, y_prob, 0.1)
    assert metrics["top_n"] == 1


def test_top_fraction_with_rare_positive() -> None:
    y_true = np.array([0, 0, 0, 0, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.4, 0.9])
    metrics = top_fraction_metrics(y_true, y_prob, 0.1)
    assert metrics["top_n"] == 1
    assert metrics["recall"] == 1.0
    assert metrics["precision"] == 1.0


def test_top_fraction_selects_at_least_one_sample() -> None:
    metrics = top_fraction_metrics(np.array([1, 0, 0]), np.array([0.8, 0.2, 0.1]), 0.1)
    assert metrics["top_n"] == 1


def test_metrics_reject_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        top_fraction_metrics(np.array([1, 0]), np.array([0.5]))


def test_metrics_reject_empty_inputs() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        top_fraction_metrics(np.array([]), np.array([]))


def test_metrics_reject_probabilities_outside_unit_interval() -> None:
    with pytest.raises(ValueError, match="in \\[0, 1\\]"):
        top_fraction_metrics(np.array([1, 0]), np.array([1.2, 0.1]))


def test_constant_baseline_brier_matches_definition() -> None:
    assert constant_baseline_brier(np.array([1, 1, 0, 0])) == 0.25


def test_overview_metrics_records_model_and_data_versions() -> None:
    metrics = overview_metrics(
        y_true=np.array([1, 0, 1, 0]),
        y_score=np.array([0.9, 0.1, 0.8, 0.2]),
        logistic_pr_auc=0.75,
        model_version="model",
        evaluation_timestamp="timestamp",
        data_processing_version="data",
    )
    assert metrics["model_version"] == "model"
    assert metrics["data_processing_version"] == "data"
