#!/usr/bin/env python3
"""Build SaPaVe ActiveViewPose dataset splits and ActiveManip-Bench suites.

Official SaPaVe assets are not public. This script remaps InteriorGS
active_spatial JSONL onto the paper protocol and writes train/eval files
the existing active_spatial env can consume unchanged.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_gen.sapave.taxonomy import (  # noqa: E402
    DEFAULT_VAL_SCENES,
    OOD_TEST_SCENES,
    annotate_item,
    avp_eval_bucket,
    keep_for_sapave,
    make_out_of_view_item,
    summarize,
)

DEFAULT_TRAIN_POOL = ROOT / "data_gen/active_spatial_pipeline/output_100scenes/train_100scenes_no_ood.jsonl"
DEFAULT_SCENE_DIR = ROOT / "data_gen/active_spatial_pipeline/output_100scenes"
DEFAULT_OUT = ROOT / "data_gen/sapave/output"


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_ood_scene_pool(scene_dir: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    missing = []
    for scene_id in OOD_TEST_SCENES:
        path = scene_dir / f"train_data_{scene_id}.jsonl"
        if not path.is_file():
            missing.append(scene_id)
            continue
        rows.extend(load_jsonl(path))
    if missing:
        print(f"[warn] missing OOD scene files: {missing}")
    return rows


def stratified_sample(
    rows: Sequence[Dict[str, Any]],
    key: str,
    cap: int,
    rng: random.Random,
) -> List[Dict[str, Any]]:
    if cap <= 0 or len(rows) <= cap:
        return list(rows)
    buckets: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[row.get(key)].append(row)
    keys = sorted(buckets)
    out: List[Dict[str, Any]] = []
    # Round-robin so no visibility/modality is wiped out by a large bucket.
    idxs = {k: 0 for k in keys}
    for k in keys:
        rng.shuffle(buckets[k])
    while len(out) < cap and any(idxs[k] < len(buckets[k]) for k in keys):
        for k in keys:
            i = idxs[k]
            if i < len(buckets[k]):
                out.append(buckets[k][i])
                idxs[k] = i + 1
                if len(out) >= cap:
                    break
    return out


def resolve_val_scenes(train_pool: Sequence[Dict[str, Any]], requested: Sequence[str]) -> List[str]:
    available = {r.get("scene_id") for r in train_pool}
    chosen = [s for s in requested if s in available]
    if len(chosen) >= 3:
        return chosen
    # Fallback: pick moderate-size scenes not in the OOD test list.
    counts: Dict[str, int] = defaultdict(int)
    for row in train_pool:
        sid = row.get("scene_id")
        if sid and sid not in OOD_TEST_SCENES:
            counts[sid] += 1
    ranked = [s for s, n in sorted(counts.items(), key=lambda kv: kv[1]) if 20 <= n <= 200]
    for sid in ranked:
        if sid not in chosen:
            chosen.append(sid)
        if len(chosen) >= 5:
            break
    return chosen


def build(args: argparse.Namespace) -> Dict[str, Any]:
    rng = random.Random(args.seed)
    ood_scene_set = set(OOD_TEST_SCENES)
    train_pool = [
        r for r in load_jsonl(args.train_pool)
        if keep_for_sapave(r) and r.get("scene_id") not in ood_scene_set
    ]
    test_pool = [r for r in load_ood_scene_pool(args.scene_dir) if keep_for_sapave(r)]
    val_scenes = resolve_val_scenes(train_pool, args.val_scenes)
    val_scene_set = set(val_scenes)

    train_rows = []
    val_rows = []
    for row in train_pool:
        if row.get("scene_id") in val_scene_set:
            val_rows.append(annotate_item(row, "val"))
        else:
            train_rows.append(annotate_item(row, "train"))

    test1_rows = []
    test2_rows = []
    for row in test_pool:
        bucket = avp_eval_bucket(row)
        annotated = annotate_item(row, bucket)
        if bucket == "test1":
            test1_rows.append(annotated)
        else:
            test2_rows.append(annotated)

    if args.cap_test1 > 0:
        test1_rows = stratified_sample(test1_rows, "sapave_modality", args.cap_test1, rng)
    if args.cap_test2 > 0:
        test2_rows = stratified_sample(test2_rows, "sapave_modality", args.cap_test2, rng)

    out = Path(args.out_dir)
    files = {
        "activeviewpose_train.jsonl": train_rows,
        "activeviewpose_val.jsonl": val_rows,
        "activeviewpose_test1.jsonl": test1_rows,
        "activeviewpose_test2.jsonl": test2_rows,
        "activeviewpose_test.jsonl": test1_rows + test2_rows,
    }

    annotated_test_pool = [annotate_item(r) for r in test_pool]
    oov_pool = []
    for row in annotated_test_pool:
        if row.get("sapave_visibility") != "unoccluded":
            continue
        synthesized = make_out_of_view_item(row)
        if synthesized is None:
            continue
        synthesized["sapave_split"] = row.get("sapave_explicit_spatial") and "test1" or "test2"
        synthesized["sapave_manip_family"] = row.get("sapave_manip_family")
        synthesized["sapave_modality"] = row.get("sapave_modality")
        synthesized["sapave_explicit_spatial"] = row.get("sapave_explicit_spatial")
        oov_pool.append(synthesized)

    manip_suites: Dict[str, List[Dict[str, Any]]] = {}
    natural_test = annotated_test_pool
    for vis in ("unoccluded", "occluded", "out_of_view"):
        source = oov_pool if vis == "out_of_view" else natural_test
        for family in ("pick_and_place", "articulated"):
            name = f"activemanip_{vis}_{family}.jsonl"
            subset = [
                r for r in source
                if r.get("sapave_visibility") == vis and r.get("sapave_manip_family") == family
            ]
            subset = stratified_sample(subset, "sapave_modality", args.cap_manip, rng)
            for row in subset:
                row["sapave_bench"] = "activemanip"
            manip_suites[name] = subset
            files[name] = subset
    files["activemanip_bench.jsonl"] = [r for rs in manip_suites.values() for r in rs]

    train_scenes = {r.get("scene_id") for r in train_rows}
    eval_scenes = {r.get("scene_id") for r in val_rows + test1_rows + test2_rows}
    leak = sorted(train_scenes & eval_scenes)
    if leak:
        raise RuntimeError(f"train/eval scene leakage: {leak}")

    for name, rows in files.items():
        write_jsonl(out / name, rows)

    manifest = {
        "source": {
            "train_pool": str(args.train_pool),
            "scene_dir": str(args.scene_dir),
            "ood_test_scenes": list(OOD_TEST_SCENES),
            "val_scenes": val_scenes,
            "excluded_task_types": ["delta_control"],
            "seed": args.seed,
        },
        "note": (
            "Official ActiveViewPose-200K / ActiveManip-Bench are not released. "
            "These splits remap InteriorGS active_spatial samples onto the SaPaVe "
            "Train/Val/Test1/Test2 and Unoccluded/Occluded/Out-of-View protocol."
        ),
        "splits": {name: summarize(rows) for name, rows in files.items()},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-pool", type=Path, default=DEFAULT_TRAIN_POOL)
    p.add_argument("--scene-dir", type=Path, default=DEFAULT_SCENE_DIR)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--val-scenes", nargs="*", default=list(DEFAULT_VAL_SCENES))
    p.add_argument("--cap-test1", type=int, default=400)
    p.add_argument("--cap-test2", type=int, default=400)
    p.add_argument("--cap-manip", type=int, default=80)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = build(args)
    print(json.dumps({k: v["n"] if isinstance(v, dict) and "n" in v else v
                      for k, v in manifest["splits"].items()}, indent=2))
    print("val_scenes:", manifest["source"]["val_scenes"])
    print("wrote", args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
