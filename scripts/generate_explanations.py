from __future__ import annotations

import argparse
from pathlib import Path

from src.inference.explanations import catboost_shap_summary, write_shap_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate BeforeReturn explanation artifacts.")
    parser.add_argument("--feature-set", default="strict_no_leak")
    parser.add_argument(
        "--model-path",
        default="models/strict_no_leak_catboost_calibrated.joblib",
    )
    parser.add_argument("--sample-size", type=int, default=5000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = catboost_shap_summary(
        model_path=Path(args.model_path),
        feature_set=args.feature_set,
        sample_size=args.sample_size,
    )
    output_path = write_shap_summary(summary)
    print(f"Wrote {output_path}")
    for item in summary["top_features"][:10]:
        print(f"{item['feature']}: {item['mean_abs_shap']:.6f}")


if __name__ == "__main__":
    main()

