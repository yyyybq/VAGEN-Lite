#!/usr/bin/env python3
"""Episode manifests of the downstream line (which demonstrations each experiment trains on).

    python downstream/demos/subsets.py target50 --out-dir downstream/demos/manifests
        -> target50_100.json (all human demos of the 50 target tasks), target50_030.json, target50_010.json
    python downstream/demos/subsets.py step2 --out downstream/demos/manifests/step2_smoke.json

target50_010 / _030 follow RoboCasa 1.0.1's own definition of its 10% / 30% target subsets
(robocasa/utils/groot_utils/groot_dataset.py:get_subset_demos_filter_key, filter_key "50_demos" / "150_demos",
filter_key_seed 0): the episode_index values of meta/episodes.jsonl in file order, shuffled with Python's
random.seed(0), first N.  One permutation per task, so 50 is a prefix of 150 and both are inside "all".
The manifest records the source metadata hashes; prepare_data.py turns a manifest into a training directory.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_ROOT = Path("/mnt/umm/users/yinbaiqiao/probe_spatial/robocasa365/datasets/v1.0")
TASKS = HERE / "target50_tasks.json"
OFFICIAL_SEED = 0
FRACTIONS = {"010": 50, "030": 150}  # RoboCasa: num_total_demos 500 for target x 0.1 / 0.3
STEP2_TASKS = ("NavigateKitchen", "OpenStandMixerHead", "CloseToasterOvenDoor")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def official_subset(file_order_ids, num_demos: int, seed: int = OFFICIAL_SEED):
    """RoboCasa's get_subset_demos_filter_key on a list of ids (None = all demos)."""
    ids = list(file_order_ids)
    random.Random(seed).shuffle(ids)
    return None if num_demos >= len(ids) else ids[:num_demos]


def task_table(path: Path = None) -> list:
    return json.loads((path or TASKS).read_text())["tasks"]


def episodes(source: Path) -> list:
    return [json.loads(line) for line in (source / "meta/episodes.jsonl").read_text().splitlines() if line.strip()]


def source_of(task: dict, data_root: Path = DATA_ROOT) -> Path:
    """The local LeRobot dir of a target task: <root>/target/<atomic|composite>/<Task>/<date>/lerobot."""
    kind = "atomic" if task["group"] == "atomic_seen" else "composite"
    dates = sorted((data_root / "target" / kind / task["task"]).glob("*/lerobot"))
    if len(dates) != 1:
        raise FileNotFoundError(f"expected one dataset for {task['task']}, found {dates}")
    return dates[0]


def entry(task: dict, source: Path, train_ids, val_ids, eps=None) -> dict:
    eps = eps if eps is not None else episodes(source)
    length = {int(e["episode_index"]): int(e["length"]) for e in eps}
    return {"task": task["task"], "group": task["group"], "horizon": task["horizon"], "source": str(source),
            "source_info_sha256": sha256(source / "meta/info.json"),
            "source_episodes_sha256": sha256(source / "meta/episodes.jsonl"),
            "train_ids": sorted(int(i) for i in train_ids), "val_ids": sorted(int(i) for i in val_ids),
            "frames": {"train": sum(length[int(i)] for i in train_ids), "val": sum(length[int(i)] for i in val_ids)}}


def target50(out_dir: Path, data_root: Path = DATA_ROOT) -> dict:
    tables = {"100": [], "030": [], "010": []}
    for task in task_table():
        source = source_of(task, data_root)
        eps = episodes(source)
        order = [int(e["episode_index"]) for e in eps]
        full = sorted(order)
        tables["100"].append(entry(task, source, full, [], eps))
        for tag, n in FRACTIONS.items():
            sub = official_subset(order, n) or full
            tables[tag].append(entry(task, source, sub, [], eps))
    checks = {}
    for small, big in (("010", "030"), ("030", "100")):
        checks[f"{small}_in_{big}"] = all(set(a["train_ids"]) <= set(b["train_ids"])
                                          for a, b in zip(tables[small], tables[big]))
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for tag, rows in tables.items():
        manifest = {"name": f"target50_{tag}", "data_root": str(data_root),
                    "rule": ("all human demonstrations" if tag == "100" else
                             f"RoboCasa 1.0.1 filter_key {FRACTIONS[tag]}_demos: episodes.jsonl order, "
                             f"random.seed({OFFICIAL_SEED}) shuffle, first {FRACTIONS[tag]}"),
                    "nesting": checks, "tasks": rows,
                    "totals": {"tasks": len(rows), "episodes": sum(len(r["train_ids"]) for r in rows),
                               "frames": sum(r["frames"]["train"] for r in rows)}}
        path = out_dir / f"target50_{tag}.json"
        path.write_text(json.dumps(manifest, indent=1) + "\n")
        summary[tag] = {"path": str(path), **manifest["totals"], "sha256": sha256(path)}
    summary["nesting"] = checks
    return summary


def step2(out: Path, n_train: int = 100, n_val: int = 20, data_root: Path = DATA_ROOT) -> dict:
    """Step-2 smoke: three 450-step atomic tasks, first n_train episodes by index, last n_val held out."""
    rows = []
    for task in task_table():
        if task["task"] not in STEP2_TASKS:
            continue
        source = source_of(task, data_root)
        ids = sorted(int(e["episode_index"]) for e in episodes(source))
        rows.append(entry(task, source, ids[:n_train], ids[-n_val:]))
    manifest = {"name": "step2_smoke", "data_root": str(data_root),
                "rule": f"first {n_train} episodes by episode_index for training, last {n_val} held out",
                "tasks": rows, "totals": {"tasks": len(rows), "episodes": sum(len(r["train_ids"]) for r in rows),
                                          "frames": sum(r["frames"]["train"] for r in rows)}}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=1) + "\n")
    return {"path": str(out), **manifest["totals"], "sha256": sha256(out)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("target50")
    a.add_argument("--out-dir", type=Path, default=HERE / "manifests")
    b = sub.add_parser("step2")
    b.add_argument("--out", type=Path, default=HERE / "manifests/step2_smoke.json")
    args = ap.parse_args(argv)
    result = target50(args.out_dir) if args.cmd == "target50" else step2(args.out)
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
