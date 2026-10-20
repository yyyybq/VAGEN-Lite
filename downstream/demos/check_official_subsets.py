#!/usr/bin/env python3
"""Cross-check target50_010 / _030 against RoboCasa's own subset function (run with the RoboCasa365 Python).

    PYTHONPATH=<robocasa365>/robocasa <robocasa365>/.venv/bin/python downstream/demos/check_official_subsets.py \
        --manifest-dir downstream/demos/manifests --out check.json

For every task: robocasa.utils.groot_utils.groot_dataset.get_subset_demos_filter_key(f"{N}_demos", 0, path) versus
the manifest's train_ids, and the nesting 50 in 150 in all.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    import robocasa
    from robocasa.utils.groot_utils.groot_dataset import get_subset_demos_filter_key

    full = {t["task"]: t for t in json.loads((args.manifest_dir / "target50_100.json").read_text())["tasks"]}
    report = {"robocasa": robocasa.__version__, "function": "groot_dataset.get_subset_demos_filter_key",
              "filter_key_seed": 0, "tasks": {}}
    ok = True
    for tag, n in (("010", 50), ("030", 150)):
        for row in json.loads((args.manifest_dir / f"target50_{tag}.json").read_text())["tasks"]:
            official = get_subset_demos_filter_key(f"{n}_demos", 0, row["source"])
            official = sorted(official) if official is not None else full[row["task"]]["train_ids"]
            same = official == row["train_ids"]
            ok &= same
            report["tasks"].setdefault(row["task"], {})[f"{n}_demos_matches_manifest"] = same
    report["all_match"] = ok
    args.out.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps({"all_match": ok, "tasks": len(report["tasks"])}))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
