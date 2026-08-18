from __future__ import annotations

import hashlib
import json
from pathlib import Path

import requests

OSF_FILES_URL = "https://api.osf.io/v2/nodes/c793h/files/osfstorage/"
RAW_DIR = Path("data/raw")
MANIFEST_PATH = RAW_DIR / "manifest.json"


def stream_download(url: str, output_path: Path) -> None:
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        with output_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    response = requests.get(OSF_FILES_URL, timeout=60)
    response.raise_for_status()
    payload = response.json()

    files = []
    for item in payload["data"]:
        attrs = item["attributes"]
        if attrs["kind"] != "file":
            continue

        name = attrs["name"]
        output_path = RAW_DIR / name
        expected_hash = attrs.get("extra", {}).get("hashes", {}).get("sha256")
        download_url = item["links"]["download"]

        if output_path.exists() and expected_hash and sha256(output_path) == expected_hash:
            status = "exists"
        else:
            print(f"Downloading {name}...")
            stream_download(download_url, output_path)
            actual_hash = sha256(output_path)
            if expected_hash and actual_hash != expected_hash:
                raise RuntimeError(
                    f"Checksum mismatch for {name}: expected {expected_hash}, got {actual_hash}"
                )
            status = "downloaded"

        files.append(
            {
                "name": name,
                "size": attrs["size"],
                "sha256": expected_hash,
                "download_url": download_url,
                "status": status,
            }
        )

    MANIFEST_PATH.write_text(json.dumps({"source": OSF_FILES_URL, "files": files}, indent=2))
    print(f"Wrote {MANIFEST_PATH}")


if __name__ == "__main__":
    main()

