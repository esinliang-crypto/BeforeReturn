import numpy as np
import pandas as pd

from src.evaluation.metrics import binary_metrics, recall_at_top_fraction


def test_recall_at_top_fraction() -> None:
    y_true = np.array([1, 0, 1, 0])
    y_score = np.array([0.9, 0.8, 0.1, 0.2])
    assert recall_at_top_fraction(y_true, y_score, 0.5) == 0.5


def test_binary_metrics_contains_required_keys() -> None:
    metrics = binary_metrics(pd.Series([0, 1, 1, 0]), np.array([0.1, 0.9, 0.8, 0.2]))
    assert "pr_auc" in metrics
    assert "recall_at_top_10_pct" in metrics
    assert metrics["confusion_matrix"] == {"tn": 2, "fp": 0, "fn": 0, "tp": 2}

