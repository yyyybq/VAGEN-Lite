#!/usr/bin/env python3
"""Collect the manifest.json files of the fixed layout into configs/{datasets,models,results}.yaml (cleanup, 2026-10).

    python scripts/cluster/collect_manifests.py --root /mnt/umm/users/yinbaiqiao/VAGEN-Lite --out configs

Reads data/releases/*/manifest.json, models/*/manifest.json, models/historical/*/manifest.json,
models/base/*.manifest.json and results/*/manifest.json (written by make_all_manifests.sh) and keeps, per entry,
what a reader needs to find and trust it: name, path behind the fixed link, size, code commit, note, and the
kind's summary (dataset line and split counts; model type, hidden size and weight shards).  The per-file hashes stay
in the manifest itself, whose path is recorded.  Only stdlib + PyYAML.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import yaml

KEEP = ("name", "kind", "path", "link", "size_KB", "created", "code_commit", "source_run", "note")


def entry(manifest: Path) -> dict:
    m = json.loads(manifest.read_text())
    e = {k: m.get(k) for k in KEEP if m.get(k) is not None}
    e["manifest"] = str(manifest)
    if m.get("kind") == "dataset":
        d = m.get("dataset") or {}
        e["jsonl_line_counts"] = d.get("jsonl_line_counts")
        e["splits"] = d.get("splits")
    elif m.get("kind") == "model":
        e.update({k: v for k, v in (m.get("model") or {}).items() if v is not None})
    elif m.get("kind") == "result":
        e["report_files"] = (m.get("result") or {}).get("report_files")
    return e


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    root, out = Path(a.root), Path(a.out)
    def find(pattern: str) -> list:  # glob.glob follows symlinked directories, Path.glob does not
        return [Path(p) for p in sorted(glob.glob(str(root / pattern)))]

    groups = {
        "datasets": find("data/releases/*/manifest.json"),
        "models": find("models/*/manifest.json") + find("models/historical/*/manifest.json")
        + find("models/base/*.manifest.json"),
        "results": find("results/*/manifest.json"),
    }
    for name, files in groups.items():
        body = {"generated_by": "scripts/cluster/collect_manifests.py", "root": str(root),
                name: [entry(f) for f in files]}
        head = (f"# {name}: collected from the manifest.json files of the cluster's fixed layout (cleanup phase 2).\n"
                f"# Do not edit by hand; regenerate with\n"
                f"#   python scripts/cluster/collect_manifests.py --root {root} --out configs\n")
        (out / f"{name}.yaml").write_text(head + yaml.safe_dump(body, allow_unicode=True, sort_keys=False, width=120))
        print(f"{name}: {len(files)} entries -> {out / (name + '.yaml')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
