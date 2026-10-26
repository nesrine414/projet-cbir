"""Append the MPEG-7 descriptors to an existing dataset index."""

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILD_SCRIPT = ROOT / "scripts" / "build_index.py"


def main() -> int:
    dataset = sys.argv[1] if len(sys.argv) > 1 else "objects"
    if dataset not in {"objects", "mirflickr"}:
        print("Usage: python scripts/add_mpeg7_to_index.py [objects|mirflickr]", file=sys.stderr)
        return 2
    command = [
        sys.executable,
        str(BUILD_SCRIPT),
        "--dataset",
        dataset,
        "--descriptors",
        "cld",
        "ehd",
        "scd",
        "--append",
    ]
    return subprocess.call(command, cwd=ROOT)


if __name__ == "__main__":
    raise SystemExit(main())
