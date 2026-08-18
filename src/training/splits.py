from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.dataset import TARGET

SPLIT_ROW_ID = "__split_row_id"


@dataclass(frozen=True)
class InternalSplit:
    train_fit: pd.DataFrame
    validation: pd.DataFrame
    metadata: dict[str, Any]
    calibration: pd.DataFrame | None = None


def positive_rate(frame: pd.DataFrame) -> float:
    return float(frame[TARGET].mean())


def split_metadata(
    *,
    random_seed: int,
    train_fit: pd.DataFrame,
    validation: pd.DataFrame,
    test: pd.DataFrame,
    calibration: pd.DataFrame | None = None,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "split_seed": random_seed,
        "train_fit_rows": int(len(train_fit)),
        "validation_rows": int(len(validation)),
        "test_rows": int(len(test)),
        "train_fit_positive_rate": positive_rate(train_fit),
        "validation_positive_rate": positive_rate(validation),
        "test_positive_rate": positive_rate(test),
    }
    if calibration is not None:
        metadata.update(
            {
                "calibration_rows": int(len(calibration)),
                "calibration_positive_rate": positive_rate(calibration),
            }
        )
    return metadata


def split_train_validation(
    frame: pd.DataFrame,
    *,
    validation_size: float = 0.2,
    random_seed: int = 42,
) -> InternalSplit:
    if not 0 < validation_size < 1:
        raise ValueError("validation_size must be in (0, 1).")
    working = frame.reset_index(drop=True).copy()
    working[SPLIT_ROW_ID] = range(len(working))
    train_fit, validation = train_test_split(
        working,
        test_size=validation_size,
        stratify=working[TARGET],
        random_state=random_seed,
    )
    return InternalSplit(
        train_fit=train_fit.copy(),
        validation=validation.copy(),
        metadata={
            "split_seed": random_seed,
            "train_fit_rows": int(len(train_fit)),
            "validation_rows": int(len(validation)),
            "train_fit_positive_rate": positive_rate(train_fit),
            "validation_positive_rate": positive_rate(validation),
        },
    )


def split_train_validation_calibration(
    frame: pd.DataFrame,
    *,
    validation_size: float = 0.16,
    calibration_size: float = 0.2,
    random_seed: int = 42,
) -> InternalSplit:
    if not 0 < validation_size < 1:
        raise ValueError("validation_size must be in (0, 1).")
    if not 0 < calibration_size < 1:
        raise ValueError("calibration_size must be in (0, 1).")
    if validation_size + calibration_size >= 1:
        raise ValueError("validation_size + calibration_size must be less than 1.")

    working = frame.reset_index(drop=True).copy()
    working[SPLIT_ROW_ID] = range(len(working))
    train_and_validation, calibration = train_test_split(
        working,
        test_size=calibration_size,
        stratify=working[TARGET],
        random_state=random_seed,
    )
    relative_validation_size = validation_size / (1 - calibration_size)
    train_fit, validation = train_test_split(
        train_and_validation,
        test_size=relative_validation_size,
        stratify=train_and_validation[TARGET],
        random_state=random_seed,
    )
    return InternalSplit(
        train_fit=train_fit.copy(),
        validation=validation.copy(),
        calibration=calibration.copy(),
        metadata={
            "split_seed": random_seed,
            "train_fit_rows": int(len(train_fit)),
            "validation_rows": int(len(validation)),
            "calibration_rows": int(len(calibration)),
            "train_fit_positive_rate": positive_rate(train_fit),
            "validation_positive_rate": positive_rate(validation),
            "calibration_positive_rate": positive_rate(calibration),
        },
    )
