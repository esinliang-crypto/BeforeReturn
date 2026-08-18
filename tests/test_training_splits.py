from __future__ import annotations

import numpy as np
import pandas as pd

import src.training.calibration as calibration_module
import src.training.train as train_module
from src.data.dataset import TARGET
from src.training.splits import (
    SPLIT_ROW_ID,
    split_metadata,
    split_train_validation_calibration,
)


def make_frame(rows: int = 100, marker_offset: int = 0) -> pd.DataFrame:
    labels = [0, 1] * (rows // 2)
    return pd.DataFrame(
        {
            "marker": np.arange(marker_offset, marker_offset + rows),
            "category": ["A", "B"] * (rows // 2),
            TARGET: labels,
        }
    )


def split_ids(frame: pd.DataFrame) -> set[int]:
    return set(frame[SPLIT_ROW_ID].astype(int))


def test_train_validation_calibration_splits_are_disjoint_and_complete() -> None:
    frame = make_frame(100)
    split = split_train_validation_calibration(frame, random_seed=7)
    assert split.calibration is not None

    train_ids = split_ids(split.train_fit)
    validation_ids = split_ids(split.validation)
    calibration_ids = split_ids(split.calibration)

    assert train_ids.isdisjoint(validation_ids)
    assert train_ids.isdisjoint(calibration_ids)
    assert validation_ids.isdisjoint(calibration_ids)
    assert train_ids | validation_ids | calibration_ids == set(range(len(frame)))
    assert len(split.train_fit) == 64
    assert len(split.validation) == 16
    assert len(split.calibration) == 20


def test_internal_split_is_reproducible_for_same_random_state() -> None:
    frame = make_frame(100)
    first = split_train_validation_calibration(frame, random_seed=42)
    second = split_train_validation_calibration(frame, random_seed=42)

    assert split_ids(first.train_fit) == split_ids(second.train_fit)
    assert split_ids(first.validation) == split_ids(second.validation)
    assert first.calibration is not None
    assert second.calibration is not None
    assert split_ids(first.calibration) == split_ids(second.calibration)


def test_internal_splits_keep_positive_and_negative_labels() -> None:
    split = split_train_validation_calibration(make_frame(100), random_seed=42)
    assert split.calibration is not None

    for frame in (split.train_fit, split.validation, split.calibration):
        assert set(frame[TARGET]) == {0, 1}


def test_validation_and_calibration_are_different_index_sets() -> None:
    split = split_train_validation_calibration(make_frame(100), random_seed=42)
    assert split.calibration is not None
    assert split_ids(split.validation) != split_ids(split.calibration)


def test_split_metadata_records_rows_seed_and_positive_rates() -> None:
    split = split_train_validation_calibration(make_frame(100), random_seed=42)
    assert split.calibration is not None
    test = make_frame(20, marker_offset=1_000)

    metadata = split_metadata(
        random_seed=42,
        train_fit=split.train_fit,
        validation=split.validation,
        calibration=split.calibration,
        test=test,
    )

    assert metadata["split_seed"] == 42
    assert metadata["train_fit_rows"] == 64
    assert metadata["validation_rows"] == 16
    assert metadata["calibration_rows"] == 20
    assert metadata["test_rows"] == 20
    assert metadata["train_fit_positive_rate"] == 0.5
    assert metadata["validation_positive_rate"] == 0.5
    assert metadata["calibration_positive_rate"] == 0.5
    assert metadata["test_positive_rate"] == 0.5


def test_official_test_is_not_used_as_catboost_fit_or_eval_set(monkeypatch) -> None:
    seen: dict[str, list[int]] = {}

    class FakePool:
        def __init__(self, data, label=None, cat_features=None):
            self.data = data
            self.label = label
            self.cat_features = cat_features

    class FakeCatBoostClassifier:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit(self, train_pool, eval_set=None, use_best_model=False):
            seen["train_markers"] = train_pool.data["marker"].tolist()
            seen["eval_markers"] = eval_set.data["marker"].tolist()
            seen["eval_labels"] = eval_set.label.tolist()
            seen["use_best_model"] = use_best_model

        def predict_proba(self, pool):
            scores = np.linspace(0.1, 0.9, len(pool.data))
            return np.column_stack([1 - scores, scores])

    monkeypatch.setattr(train_module, "Pool", FakePool)
    monkeypatch.setattr(train_module, "CatBoostClassifier", FakeCatBoostClassifier)

    train_module.train_catboost(
        train=make_frame(100),
        test=make_frame(20, marker_offset=1_000),
        features=["marker", "category"],
        categorical=["category"],
        random_seed=42,
    )

    assert max(seen["train_markers"]) < 1_000
    assert max(seen["eval_markers"]) < 1_000
    assert seen["use_best_model"] is True


def test_calibration_labels_only_go_to_calibrator_fit(monkeypatch) -> None:
    seen: dict[str, set[int]] = {}
    official_test = make_frame(20, marker_offset=1_000)

    class FakePool:
        def __init__(self, data, label=None, cat_features=None):
            self.data = data
            self.label = label
            self.cat_features = cat_features

    class FakeCatBoostClassifier:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit(self, train_pool, eval_set=None, use_best_model=False):
            seen["train_fit_ids"] = set(train_pool.data.index)
            seen["validation_ids"] = set(eval_set.data.index)
            seen["validation_label_ids"] = set(eval_set.label.index)

        def predict_proba(self, pool):
            scores = np.linspace(0.1, 0.9, len(pool.data))
            return np.column_stack([1 - scores, scores])

    class FakeIsotonicRegression:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def fit(self, scores, labels):
            seen["calibration_label_ids"] = set(labels.index)

        def transform(self, scores):
            return scores

    monkeypatch.setattr(calibration_module, "Pool", FakePool)
    monkeypatch.setattr(calibration_module, "CatBoostClassifier", FakeCatBoostClassifier)
    monkeypatch.setattr(calibration_module, "IsotonicRegression", FakeIsotonicRegression)
    monkeypatch.setattr(
        calibration_module,
        "load_processed",
        lambda feature_set, split: make_frame(100) if split == "training" else official_test,
    )
    monkeypatch.setattr(calibration_module.joblib, "dump", lambda *args, **kwargs: None)

    def fake_write_text(self, text):
        return len(text)

    monkeypatch.setattr(calibration_module.Path, "write_text", fake_write_text)

    metrics = calibration_module.calibrate_catboost(calibration_module.CalibrationConfig())

    assert seen["validation_ids"].isdisjoint(seen["calibration_label_ids"])
    assert seen["train_fit_ids"].isdisjoint(seen["calibration_label_ids"])
    assert seen["validation_label_ids"] == seen["validation_ids"]
    assert max(seen["validation_ids"]) < 1_000
    assert metrics["train_fit_rows"] == 64
    assert metrics["validation_rows"] == 16
    assert metrics["calibration_rows"] == 20
    assert metrics["test_rows"] == 20
