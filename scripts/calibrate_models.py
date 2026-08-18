from __future__ import annotations

import argparse
import json

from src.data.dataset import FEATURE_SETS
from src.training.calibration import CalibrationConfig, calibrate_catboost


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate BeforeReturn CatBoost models.")
    parser.add_argument("--feature-set", choices=FEATURE_SETS, default="strict_no_leak")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = calibrate_catboost(CalibrationConfig(feature_set=args.feature_set))
    print(
        json.dumps(
            {
                "feature_set": metrics["feature_set"],
                "model_name": metrics["model_name"],
                "pr_auc": metrics["pr_auc"],
                "roc_auc": metrics["roc_auc"],
                "brier_score": metrics["brier_score"],
                "raw_brier_score": metrics["raw_brier_score"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

