from __future__ import annotations

import gzip
import json
from pathlib import Path
from typing import Any

import pandas as pd

RUNTIME_DIR = Path("reports/runtime")
RUNTIME_ARTIFACT_PATH = RUNTIME_DIR / "strict_no_leak_demo_runtime.json.gz"
RUNTIME_SCHEMA_VERSION = 1


def write_runtime_artifact(payload: dict[str, Any], path: Path = RUNTIME_ARTIFACT_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw_file:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw_file, mtime=0) as gzip_file:
            gzip_file.write(
                json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
            )
    return path


def load_runtime_artifact(path: Path = RUNTIME_ARTIFACT_PATH) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Demo runtime artifact is missing: {path}")
    with gzip.open(path, "rt", encoding="utf-8") as file:
        payload = json.load(file)
    if payload.get("artifact_schema_version") != RUNTIME_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported demo runtime artifact schema version: "
            f"{payload.get('artifact_schema_version')}"
        )
    return payload


def runtime_frame(payload: dict[str, Any], key: str) -> pd.DataFrame:
    return pd.DataFrame(payload.get(key, []))
