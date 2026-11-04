"""Read-only: every file the archive plan names must be in the manifests written since START with ok true and the
local size; prints the counts, bytes and any problem.  Usage: arch_check.py START_STAMP (e.g. 20261104T0741)."""
import json
import os
import sys
from pathlib import Path

START = sys.argv[1]
plan = json.loads(Path("/tmp/yb_msgs/arch_plan.json").read_text())
expected = {}
for cell in plan.values():
    for path, _, _ in cell["archive"]:
        p = Path(path)
        for f in ([p] if p.is_file() else [Path(d) / n for d, _, ns in os.walk(p) for n in ns]):
            expected[str(f)] = f.stat().st_size
mans = sorted(m for m in Path("/mnt/umm/users/yinbaiqiao/aoss_archive_manifests").glob("manifest_*.jsonl")
              if m.stem.split("_", 1)[1] >= START)
seen = {}
for m in mans:
    for line in m.read_text().splitlines():
        r = json.loads(line)
        if r["path"] in expected:
            seen[r["path"]] = r
missing = [p for p in expected if p not in seen]
bad = [(p, r.get("error"), r["bytes"], expected[p]) for p, r in seen.items() if not r["ok"] or r["bytes"] != expected[p]]
print("manifests", [m.name for m in mans])
print(f"expected {len(expected)} files {sum(expected.values()) / 1e9:.1f} GB; in manifests {len(seen)}; "
      f"missing {len(missing)}; bad {len(bad)}")
for p in missing[:5]:
    print("  missing", p)
for b in bad[:5]:
    print("  bad", b)
print("ALL_OK" if not missing and not bad else "NOT_OK")
