from __future__ import annotations

import argparse
import json

from src.data.dataset import FEATURE_SETS
from src.training.train import TrainingConfig, train_model

MODELS = ("logistic_regression", "catboost")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train BeforeReturn models.")
    parser.add_argument("--feature-set", choices=[*FEATURE_SETS, "all"], default="all")
    parser.add_argument("--model", choices=[*MODELS, "all"], default="all")
    parser.add_argument("--train-limit", type=int, default=None)
    parser.add_argument("--test-limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feature_sets = FEATURE_SETS if args.feature_set == "all" else (args.feature_set,)
    models = MODELS if args.model == "all" else (args.model,)
    all_metrics = []

    for feature_set in feature_sets:
        for model_name in models:
            print(f"Training {feature_set}/{model_name}...", flush=True)
            metrics = train_model(
                TrainingConfig(
                    feature_set=feature_set,
                    model_name=model_name,
                    train_limit=args.train_limit,
                    test_limit=args.test_limit,
                )
            )
            all_metrics.append(metrics)
            print(
                json.dumps(
                    {
                        "feature_set": feature_set,
                        "model_name": model_name,
                        "pr_auc": metrics["pr_auc"],
                        "roc_auc": metrics["roc_auc"],
                        "recall_at_top_10_pct": metrics["recall_at_top_10_pct"],
                        "brier_score": metrics["brier_score"],
                    },
                    indent=2,
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
