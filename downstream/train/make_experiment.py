#!/usr/bin/env python3
"""Create one experiment directory: the final StarVLA config plus everything needed to reproduce it.

    python downstream/train/make_experiment.py --exp-id DS-RS-100-s1 --start RS --data RUNS/data/target50_100 \
        --seed 1 --steps 200000 --save-interval 25000

EXP = <runs-root>/<exp-id>/ gets
  config.yaml       recipe.yaml + the per-experiment fields (the only differences between starts: base_vlm, and
                    run paths named after the experiment)
  experiment.json   start (name, path, locked shard sizes / index hash), load dtype bf16, seed, data (prepared
                    dir, split-manifest hash, episode manifest), steps, StarVLA commit + patch sha256 + reverse
                    check, code commit, config sha256
  sources.json      scripts/prepare_starvla_transfer.inspect_checkpoint of the start (for the held-out evaluator)
Refuses to run if the start changed since it was locked, the StarVLA tree is not base + the recorded patch, or the
prepared data no longer matches its manifest.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOWNSTREAM = HERE.parent
WORKTREE = DOWNSTREAM.parent
sys.path.insert(0, str(DOWNSTREAM))
sys.path.insert(0, str(DOWNSTREAM / "demos"))
RUNS_ROOT = Path("/mnt/umm/users/yinbaiqiao/downstream_runs/experiments")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_state() -> dict:
    def git(*a):
        res = subprocess.run(["git", "-C", str(WORKTREE), *a], capture_output=True, text=True)
        return res.stdout.strip() if res.returncode == 0 else None

    dirty = git("status", "--porcelain", "--untracked-files=no", "--", "downstream")
    return {"commit": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty_downstream_files": dirty.splitlines() if dirty else []}


def build_config(recipe: dict, *, exp_dir: Path, exp_id: str, start_path: str, data_dir: Path, seed: int,
                 steps: int, save_interval: int, log_every: int) -> dict:
    import copy

    cfg = copy.deepcopy(recipe)
    split = json.loads((data_dir / "split_manifest.json").read_text())
    cfg.update(run_id="run", run_root_dir=str(exp_dir), seed=int(seed))
    cfg["framework"]["qwenvl"]["base_vlm"] = str(start_path)
    data = cfg["datasets"]["vla_data"]
    data["data_root_dir"] = str(data_dir / "train")
    data["mixture_spec"] = json.loads((data_dir / "mixture_spec.json").read_text())
    trainer = cfg["trainer"]
    trainer.update(max_train_steps=int(steps), save_interval=int(save_interval), logging_frequency=int(log_every))
    cfg["transfer"] = {"source_label": exp_id, "source_manifest": str(exp_dir / "sources.json"),
                       "split_manifest": str(data_dir / "split_manifest.json"),
                       "note": "downstream line; trainer mse_score is a training-batch diagnostic"}
    if [d["name"] + "/lerobot" for d in split["datasets"]] != [m[0] for m in data["mixture_spec"]]:
        raise ValueError("mixture_spec does not match the split manifest")
    return cfg


def main(argv=None) -> int:
    import yaml
    from prepare_data import check as check_data
    from starvla_patch import check as check_patch
    from verify_starts import load_starts, verify

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp-id", required=True)
    ap.add_argument("--start", required=True)
    ap.add_argument("--data", type=Path, required=True, help="directory made by downstream/demos/prepare_data.py")
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--save-interval", type=int, required=True)
    ap.add_argument("--log-every", type=int, default=100)
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    ap.add_argument("--recipe", type=Path, default=HERE / "recipe.yaml")
    args = ap.parse_args(argv)

    start = load_starts([args.start])[args.start]
    problems = verify([args.start])[args.start]
    if problems:
        raise SystemExit(f"start {args.start} changed since it was locked: {problems} (not switching paths)")
    patch = check_patch()
    if not patch["ok"]:
        raise SystemExit(f"StarVLA tree is not base + recorded patch: {patch}")
    data_check = check_data(args.data)
    if not data_check["ok"]:
        raise SystemExit(f"prepared data does not match its manifest: {data_check}")
    exp_dir = args.runs_root / args.exp_id
    exp_dir.mkdir(parents=True, exist_ok=False)
    recipe = yaml.safe_load(args.recipe.read_text())
    cfg = build_config(recipe, exp_dir=exp_dir, exp_id=args.exp_id, start_path=start["path"], data_dir=args.data,
                       seed=args.seed, steps=args.steps, save_interval=args.save_interval, log_every=args.log_every)
    (exp_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    sys.path.insert(0, str(WORKTREE / "scripts"))
    from prepare_starvla_transfer import inspect_checkpoint

    (exp_dir / "sources.json").write_text(json.dumps({args.exp_id: inspect_checkpoint(start["path"])}, indent=1) + "\n")
    split = json.loads((args.data / "split_manifest.json").read_text())
    lock = json.loads((DOWNSTREAM / "starts_lock.json").read_text())[args.start]
    record = {
        "exp_id": args.exp_id, "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "start": {"name": args.start, "desc": start["desc"], "path": start["path"], "lock": lock},
        "load_dtype": "bfloat16", "seed": args.seed,
        "data": {"dir": str(args.data), "split_manifest_sha256": sha256(args.data / "split_manifest.json"),
                 "episode_manifest": split["manifest"], "episode_manifest_sha256": split["manifest_sha256"],
                 "rule": split["rule"], "tasks": len(split["datasets"]),
                 "train_episodes": sum(len(d["train_ids"]) for d in split["datasets"]),
                 "train_frames": sum(d["frames"]["train"] for d in split["datasets"])},
        "steps": args.steps, "save_interval": args.save_interval, "gpus": args.gpus,
        "per_device_batch": cfg["datasets"]["vla_data"]["per_device_batch_size"],
        "global_batch": cfg["datasets"]["vla_data"]["per_device_batch_size"] * args.gpus,
        "recipe": {"path": str(args.recipe), "sha256": sha256(args.recipe)},
        "starvla": {"commit": patch["commit"], "patch": patch["patch"], "patch_sha256": patch["patch_sha256"],
                    "git_apply_reverse_check": patch["reverse_check"]},
        "code": code_state(), "config_sha256": sha256(exp_dir / "config.yaml"),
        "output_dir": str(exp_dir / "run"),
    }
    (exp_dir / "experiment.json").write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({"exp_dir": str(exp_dir), "start": args.start, "seed": args.seed, "steps": args.steps}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
