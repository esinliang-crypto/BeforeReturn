from __future__ import annotations

from pathlib import Path

import pandas as pd


def describe_frame(path: Path) -> None:
    frame = pd.read_pickle(path)
    print(f"\n=== {path.name} ===")
    print(f"shape: {frame.shape}")
    print("columns:")
    for column, dtype in frame.dtypes.items():
        print(f"  - {column}: {dtype}")


def main() -> None:
    for path in sorted(Path("data/raw").glob("*.p")):
        describe_frame(path)


if __name__ == "__main__":
    main()

