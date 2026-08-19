from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.fetch_demo_artifacts import fetch_artifact, sha256_file, verify_artifact


def test_fetch_artifact_verifies_existing_file(tmp_path, capsys) -> None:
    artifact_path = tmp_path / "artifact.bin"
    artifact_path.write_bytes(b"before-return")
    artifact = {
        "path": str(artifact_path),
        "size_bytes": artifact_path.stat().st_size,
        "sha256": sha256_file(artifact_path),
        "url_env": "BEFORE_RETURN_TEST_ARTIFACT_URL",
        "url": "",
    }

    fetch_artifact(artifact)

    assert f"ok {artifact_path}" in capsys.readouterr().out


def test_fetch_artifact_requires_configured_url_for_missing_file(tmp_path) -> None:
    artifact = {
        "path": str(tmp_path / "missing.bin"),
        "size_bytes": 1,
        "sha256": "0" * 64,
        "url_env": "BEFORE_RETURN_TEST_ARTIFACT_URL",
        "url": "",
    }

    with pytest.raises(RuntimeError, match="Configure BEFORE_RETURN_TEST_ARTIFACT_URL"):
        fetch_artifact(artifact)


def test_artifact_manifest_excludes_raw_and_processed_data() -> None:
    manifest = json.loads(Path("artifacts/demo-artifacts.json").read_text())

    paths = [artifact["path"] for artifact in manifest["artifacts"]]
    assert paths
    assert all("data/raw" not in path for path in paths)
    assert all("data/processed" not in path for path in paths)


def test_verify_artifact_rejects_sha_mismatch(tmp_path) -> None:
    artifact_path = tmp_path / "artifact.bin"
    artifact_path.write_bytes(b"before-return")

    with pytest.raises(RuntimeError, match="SHA256 mismatch"):
        verify_artifact(artifact_path, "0" * 64, artifact_path.stat().st_size)
