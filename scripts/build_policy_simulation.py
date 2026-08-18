from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.inference.policy_simulation import (
    PolicyArtifactConfig,
    build_policy_artifact_from_disk,
    write_policy_artifact,
)
from src.inference.scenarios import load_calibrated_bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build full strict policy simulation artifact.")
    parser.add_argument("--feature-set", default="strict_no_leak")
    parser.add_argument("--model-path", default="models/strict_no_leak_catboost_calibrated.joblib")
    parser.add_argument("--candidate-k", type=int, default=25)
    parser.add_argument("--test-limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=250_000)
    parser.add_argument(
        "--artifact-path",
        default="reports/policy/strict_no_leak_policy_simulation.pkl.gz",
    )
    parser.add_argument(
        "--manifest-path",
        default="reports/policy/strict_no_leak_policy_simulation_manifest.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PolicyArtifactConfig(
        feature_set=args.feature_set,
        candidate_k=args.candidate_k,
        test_limit=args.test_limit,
        batch_size=args.batch_size,
        artifact_path=Path(args.artifact_path),
        manifest_path=Path(args.manifest_path),
    )
    bundle = load_calibrated_bundle(Path(args.model_path))
    artifact, manifest = build_policy_artifact_from_disk(bundle, config)
    write_policy_artifact(
        artifact,
        manifest,
        artifact_path=config.artifact_path,
        manifest_path=config.manifest_path,
    )
    print(f"Wrote {config.artifact_path}")
    print(f"Wrote {config.manifest_path}")
    print(
        json.dumps(
            {
                "artifact_rows": manifest["artifact_rows"],
                "rows_with_peer_candidate": manifest["rows_with_peer_candidate"],
                "generation_seconds": manifest["generation_seconds"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
