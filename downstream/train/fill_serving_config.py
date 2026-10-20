#!/usr/bin/env python3
"""Add the serving keys a run's config.yaml lacks, from its config.full.yaml (for runs trained before
train_entry.mark_for_serving existed, e.g. STEP2-RS-s1).

    python downstream/train/fill_serving_config.py EXP/run

StarVLA's policy server reads run/config.yaml, which holds only the keys training read; config.full.yaml holds the
complete configuration the run was trained with.  Only keys missing under datasets.vla_data, framework.action_model
and framework.qwenvl are added (never changed); the original is kept as config.yaml.before_fill and the added keys
are listed in config.yaml.fill_note.json.
"""
from __future__ import annotations

import argparse
import datetime
import json
import shutil
import sys
from pathlib import Path

SECTIONS = (("datasets", "vla_data"), ("framework", "action_model"), ("framework", "qwenvl"))


def fill(accessed: dict, full: dict) -> list:
    added = []
    for path in SECTIONS:
        src, dst = full, accessed
        for part in path:
            src = src.get(part, {})
            dst = dst.setdefault(part, {})
        for key, value in src.items():
            if key not in dst:
                dst[key] = value
                added.append(".".join(path + (key,)))
    return added


def main(argv=None) -> int:
    import yaml

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", type=Path)
    args = ap.parse_args(argv)
    cfg_path, full_path = args.run / "config.yaml", args.run / "config.full.yaml"
    accessed = yaml.safe_load(cfg_path.read_text())
    added = fill(accessed, yaml.safe_load(full_path.read_text()))
    if not added:
        print("nothing to add")
        return 0
    backup = args.run / "config.yaml.before_fill"
    if not backup.exists():
        shutil.copy2(cfg_path, backup)
    cfg_path.write_text(yaml.safe_dump(accessed, sort_keys=False))
    note = {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "source": str(full_path),
            "added_keys": added, "reason": "policy-server serving contract (training_obs_image_size etc.)"}
    (args.run / "config.yaml.fill_note.json").write_text(json.dumps(note, indent=1) + "\n")
    print(json.dumps(note, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
