#!/usr/bin/env python3
"""Have the start weights changed since they were locked?  Run before every job submission.

    python downstream/verify_starts.py lock   [--starts B S RS RD]   # write downstream/starts_lock.json
    python downstream/verify_starts.py verify [--starts B S RS RD]   # exit 1 and list differences if anything changed

Per start: the size of every shard named in model.safetensors.index.json, and the sha256 of the index file and of
config.json.  Reads only metadata and two small files; never changes a path (a changed start is reported, not
replaced).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STARTS = ROOT / "starts.json"
LOCK = ROOT / "starts_lock.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint(path: str) -> dict:
    root = Path(path)
    index = root / "model.safetensors.index.json"
    shards = sorted(set(json.loads(index.read_text())["weight_map"].values()))
    return {"path": str(root), "index_sha256": sha256(index), "config_sha256": sha256(root / "config.json"),
            "shards": {name: (root / name).stat().st_size for name in shards}}


def load_starts(names=None) -> dict:
    starts = {k: v for k, v in json.loads(STARTS.read_text()).items() if not k.startswith("_")}
    if names:
        missing = [n for n in names if n not in starts]
        if missing:
            raise KeyError(f"unknown starts {missing}")
        return {n: starts[n] for n in names}
    return {k: v for k, v in starts.items() if v["role"] == "main"}


def compare(locked: dict, current: dict) -> list:
    problems = []
    for key in ("path", "index_sha256", "config_sha256"):
        if locked.get(key) != current.get(key):
            problems.append(f"{key}: {locked.get(key)} -> {current.get(key)}")
    for name in sorted(set(locked["shards"]) | set(current["shards"])):
        a, b = locked["shards"].get(name), current["shards"].get(name)
        if a != b:
            problems.append(f"shard {name}: size {a} -> {b}")
    return problems


def verify(names=None, lock_path: Path = LOCK) -> dict:
    lock = json.loads(lock_path.read_text())
    report = {}
    for name, entry in load_starts(names).items():
        if name not in lock:
            report[name] = ["not in starts_lock.json"]
            continue
        try:
            report[name] = compare(lock[name], fingerprint(entry["path"]))
        except Exception as exc:  # a missing file is a change too
            report[name] = [f"{type(exc).__name__}: {exc}"]
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["lock", "verify"])
    ap.add_argument("--starts", nargs="*", default=None, help="default: every start with role main")
    ap.add_argument("--lock-file", type=Path, default=LOCK)
    args = ap.parse_args(argv)
    if args.action == "lock":
        lock = json.loads(args.lock_file.read_text()) if args.lock_file.exists() else {}
        for name, entry in load_starts(args.starts).items():
            lock[name] = fingerprint(entry["path"])
        args.lock_file.write_text(json.dumps(lock, indent=1) + "\n")
        print(f"locked {sorted(lock)} -> {args.lock_file}")
        return 0
    report = verify(args.starts, args.lock_file)
    print(json.dumps({"unchanged": all(not v for v in report.values()), "problems": report}, indent=1))
    return 0 if all(not v for v in report.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
