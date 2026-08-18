from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.data.dataset import (
    FEATURE_SETS,
    SPLITS,
    build_feature_frame,
    leakage_risk_columns,
    write_bundle,
    write_demo_sample,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build BeforeReturn modeling datasets.")
    parser.add_argument(
        "--feature-set",
        choices=[*FEATURE_SETS, "all"],
        default="all",
        help="Feature set to build.",
    )
    parser.add_argument(
        "--split",
        choices=[*SPLITS, "all"],
        default="all",
        help="Raw data split to build.",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=200,
        help="Rows to write into committed CSV samples.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    feature_sets = FEATURE_SETS if args.feature_set == "all" else (args.feature_set,)
    splits = SPLITS if args.split == "all" else (args.split,)
    manifest = []

    for feature_set in feature_sets:
        for split in splits:
            bundle = build_feature_frame(feature_set, split)
            processed_path = write_bundle(bundle)
            sample_path = write_demo_sample(bundle, rows=args.sample_rows)
            manifest.append(
                {
                    "feature_set": feature_set,
                    "split": split,
                    "rows": len(bundle.frame),
                    "complete_metadata_rows": int(bundle.frame["has_complete_metadata"].sum()),
                    "columns": len(bundle.frame.columns),
                    "features": bundle.feature_columns,
                    "categorical_features": bundle.categorical_columns,
                    "leakage_risk_features": leakage_risk_columns(bundle.feature_columns),
                    "processed_path": str(processed_path),
                    "sample_path": str(sample_path),
                }
            )
            print(
                f"Built {feature_set}/{split}: "
                f"{len(bundle.frame):,} rows, {len(bundle.feature_columns)} features"
            )

    manifest_path = Path("data/processed/dataset_manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"Wrote {manifest_path}")


if __name__ == "__main__":
    main()
