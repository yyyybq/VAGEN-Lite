#!/usr/bin/env python3
"""Create a deterministic SHA256 inventory for R1 code and artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files = []
    for value in args.paths:
        if value.is_file():
            files.append(value)
        elif value.is_dir():
            files.extend(path for path in value.rglob("*") if path.is_file() and "__pycache__" not in path.parts)
    unique = sorted(set(path.resolve() for path in files), key=str)
    inventory = {
        "files": [
            {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in unique
        ]
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(inventory, indent=2) + "\n")
    print(json.dumps({"files": len(unique), "output": str(args.output)}, indent=2))


if __name__ == "__main__":
    main()
