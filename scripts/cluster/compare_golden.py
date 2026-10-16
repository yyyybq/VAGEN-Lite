#!/usr/bin/env python3
"""Compare two golden-output directories written by golden_outputs.sh (cleanup equivalence check).

    python scripts/cluster/compare_golden.py <before> <after>

Every file listed in either SHA256SUMS.txt is compared byte for byte. A file that differs is compared again
with the wall-time fields removed (JSON keys "timings" and "seconds", the CSV column "seconds").
Exit code 0 when every file is identical or differs only in those fields, 1 otherwise. Only stdlib.
"""
from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path

TIME_KEYS = {"timings", "seconds"}


def sums(d: Path) -> dict[str, str]:
    out = {}
    for line in (d / "SHA256SUMS.txt").read_text().splitlines():
        if line.strip():
            digest, name = line.split(maxsplit=1)
            out[name.strip()] = digest
    return out


def strip(x):
    if isinstance(x, dict):
        return {k: strip(v) for k, v in x.items() if k not in TIME_KEYS}
    if isinstance(x, list):
        return [strip(v) for v in x]
    return x


def same_without_time(a: Path, b: Path) -> bool:
    if a.suffix == ".json":
        return strip(json.loads(a.read_text())) == strip(json.loads(b.read_text()))
    if a.suffix == ".jsonl":
        la, lb = a.read_text().splitlines(), b.read_text().splitlines()
        return len(la) == len(lb) and all(strip(json.loads(x)) == strip(json.loads(y)) for x, y in zip(la, lb))
    if a.suffix == ".csv":
        rows = [[{k: v for k, v in r.items() if k not in TIME_KEYS} for r in csv.DictReader(io.StringIO(p.read_text()))] for p in (a, b)]
        return rows[0] == rows[1]
    return False


def commit(d: Path) -> str:
    p = d / "commit.txt"
    return p.read_text().strip()[:8] if p.is_file() else "?"


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    a, b = Path(sys.argv[1]), Path(sys.argv[2])
    sa, sb = sums(a), sums(b)
    rows, bad = [], 0
    for name in sorted(set(sa) | set(sb)):
        if name not in sa or name not in sb:
            status = "only in after" if name in sb else "only in before"
        elif sa[name] == sb[name]:
            status = "identical"
        elif same_without_time(a / name, b / name):
            status = "differs only in wall-time fields"
        else:
            status = "DIFFERS"
        bad += status not in ("identical", "differs only in wall-time fields")
        rows.append((name, status))
    for name, status in rows:
        print(f"{status:34s} {name}")
    n_same = sum(s == "identical" for _, s in rows)
    print(f"{len(rows)} files: {n_same} identical, {len(rows) - n_same - bad} only wall-time, {bad} real differences "
          f"(before {commit(a)}, after {commit(b)})")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
