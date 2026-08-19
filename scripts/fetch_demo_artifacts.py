from __future__ import annotations

import argparse
import hashlib
import json
import os
import urllib.request
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_url(artifact: dict[str, Any]) -> str:
    env_name = artifact.get("url_env")
    if env_name and os.getenv(env_name):
        return os.environ[env_name]
    return artifact.get("url", "")


def verify_artifact(path: Path, expected_sha256: str, expected_size: int | None) -> None:
    if expected_size is not None and path.stat().st_size != expected_size:
        raise RuntimeError(
            f"Artifact size mismatch for {path}: expected {expected_size}, "
            f"got {path.stat().st_size}."
        )
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"Artifact SHA256 mismatch for {path}: expected {expected_sha256}, "
            f"got {actual_sha256}."
        )


def fetch_artifact(artifact: dict[str, Any]) -> None:
    path = Path(artifact["path"])
    expected_sha256 = artifact["sha256"]
    expected_size = artifact.get("size_bytes")
    if path.exists():
        verify_artifact(path, expected_sha256, expected_size)
        print(f"ok {path}")
        return

    url = artifact_url(artifact)
    if not url:
        env_hint = artifact.get("url_env", "the artifact URL environment variable")
        raise RuntimeError(
            f"Missing required artifact {path}. Configure {env_hint} or set "
            f"`url` in the artifact manifest, then rerun this command."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    print(f"downloading {path}")
    urllib.request.urlretrieve(url, tmp_path)
    verify_artifact(tmp_path, expected_sha256, expected_size)
    tmp_path.replace(path)
    print(f"ok {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and verify BeforeReturn demo artifacts.")
    parser.add_argument("--manifest", default="artifacts/demo-artifacts.json")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text())
    for artifact in manifest["artifacts"]:
        fetch_artifact(artifact)


if __name__ == "__main__":
    main()
