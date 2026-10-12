"""Scene-level train / val / test split for a finished run_all output.

Same-source scenes stay together (groups file, required), the R1 training scenes
are pinned to train and the R1 evaluation scenes to test, the rest is assigned by
``release.assign_splits`` (default 80/10/10 by scene count).  Writes the
version-2 manifest plus per-split counts and fails if a group crosses splits or
an R1 evaluation scene lands in train.

    python -m data_gen.interiorgs_v10.make_split --run OUT \
        --groups data_gen/interiorgs_v10/cluster_runs/same_source_groups_20261012.json \
        --r1-frozen exps/vagen_active_spatial/R1-clean-Projective-v0/frozen_v1 \
        --out OUT/split_manifest.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from . import release


def r1_pins(frozen_dir: Path) -> Dict[str, str]:
    """R1 training scenes -> train, R1 evaluation scenes -> test."""
    pins: Dict[str, str] = {}
    for name, split in (("train.jsonl", "train"), ("eval_policy_rows.jsonl", "test")):
        for line in (frozen_dir / name).read_text().splitlines():
            if line.strip():
                sid = json.loads(line).get("scene_id")
                if sid:
                    if pins.get(sid, split) != split:
                        raise ValueError(f"R1 scene {sid} is both a training and an evaluation scene")
                    pins[str(sid)] = split
    return pins


def scene_counts(run: Path) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for p in sorted((run / "scenes").glob("*.json")):
        res = json.loads(p.read_text())
        sid = res["summary"]["scene_id"]
        out[sid] = {"tasks": len(res.get("rows", [])), "reserve": len(res.get("reserve_rows", []) or [])}
    return out


def make_split(run: Path, groups: List[List[str]], pins: Dict[str, str],
               ratios: Optional[Dict[str, float]] = None, seed: str = "interiorgs_v10") -> Dict[str, Any]:
    counts = scene_counts(run)
    manifest = release.assign_splits(sorted(counts), groups=groups, reserved=pins, ratios=ratios, seed=seed)
    lookup = release.split_lookup(manifest)
    aliases = release.groups_to_aliases(groups)
    crossing = []
    by_group = defaultdict(set)
    for sid in counts:
        by_group[aliases.get(sid, sid)].add(lookup[sid])
    for g, splits in sorted(by_group.items()):
        if len(splits) > 1:
            crossing.append(g)
    r1_eval_in_train = sorted(s for s, sp in pins.items() if sp == "test" and lookup.get(s) == "train")
    if crossing or r1_eval_in_train:
        raise RuntimeError(f"split check failed: groups crossing splits={crossing}, "
                           f"R1 evaluation scenes in train={r1_eval_in_train}")
    table = {}
    for sp in release.SPLITS:
        ss = [s for s in counts if lookup[s] == sp]
        table[sp] = {"scenes": len(ss), "scenes_with_tasks": sum(1 for s in ss if counts[s]["tasks"] > 0),
                     "tasks": sum(counts[s]["tasks"] for s in ss), "reserve": sum(counts[s]["reserve"] for s in ss)}
    return {"manifest": manifest, "splits": table,
            "checks": {"groups_crossing_splits": 0, "r1_eval_scenes_in_train": 0,
                       "groups": len(groups), "scenes_in_groups": sum(len(g) for g in groups),
                       "pinned": {sp: sum(1 for v in pins.values() if v == sp) for sp in ("train", "test")}}}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, type=Path, help="run_all output directory")
    ap.add_argument("--groups", required=True, type=Path, help="same-source groups file")
    ap.add_argument("--r1-frozen", required=True, type=Path, help="R1 frozen_v1 directory (train / eval rows)")
    ap.add_argument("--out", required=True, type=Path, help="where to write split_manifest.json")
    args = ap.parse_args(argv)
    rep = make_split(args.run, release.load_groups(args.groups), r1_pins(args.r1_frozen))
    args.out.write_text(json.dumps(rep["manifest"], indent=1))
    args.out.with_name(args.out.stem + "_counts.json").write_text(json.dumps(
        {"splits": rep["splits"], "checks": rep["checks"]}, indent=1))
    print(json.dumps({"splits": rep["splits"], "checks": rep["checks"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
