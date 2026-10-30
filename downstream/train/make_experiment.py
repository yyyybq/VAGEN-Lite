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


def parse_lora(text: str) -> dict:
    """T2: "rank=16,alpha=32,dropout=0.05,lr=1e-4" -> trainer.lora (downstream/train/lora.py via train_entry.py)."""
    if not text:
        return {}
    out = {k.strip(): float(v) for k, v in (part.split("=") for part in text.split(",") if part.strip())}
    missing = {"rank", "alpha", "lr"} - set(out)
    if missing or set(out) - {"rank", "alpha", "dropout", "lr"}:
        raise ValueError(f"--lora needs rank, alpha, lr (and optionally dropout): {text}")
    out["rank"] = int(out["rank"])
    out.setdefault("dropout", 0.0)
    return out


def parse_cotrain(text: str) -> dict:
    """C1: "lambda=0.1,diag_every=10,skip_final_save=1" -> trainer.cotrain (downstream/train/cotrain_entry.py), with
    the text sources of cotrain_text.py written out so the config says what was read."""
    if not text:
        return {}
    out = {}
    for part in text.split(","):
        if not part.strip():
            continue
        k, v = (x.strip() for x in part.split("="))
        out[k] = float(v) if k == "lambda" else int(v)
    unknown = set(out) - {"lambda", "diag_every", "skip_final_save", "max_tokens", "seed"}
    if "lambda" not in out or unknown:
        raise ValueError(f"--cotrain needs lambda (and optionally diag_every, skip_final_save, max_tokens, seed): {text}")
    if "skip_final_save" in out:
        out["skip_final_save"] = bool(out["skip_final_save"])
    sys.path.insert(0, str(HERE))
    import cotrain_text

    out["sources"] = dict(cotrain_text.SOURCES)
    out["mix"] = "even ranks traj, odd ranks qa (1:1)"
    return out


def build_config(recipe: dict, *, exp_dir: Path, exp_id: str, start_path: str, data_dir: Path, seed: int,
                 steps: int, save_interval: int, log_every: int, full_state_interval: int = 0,
                 eval_points=(), freeze_modules: str = "", lora: dict = None, cotrain: dict = None) -> dict:
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
    if freeze_modules:  # X1 (the user's decision 2 of 2026-10-26): StarVLA's own freeze_backbones / LR-group filter
        trainer["freeze_modules"] = str(freeze_modules)
    if lora:  # T2 (the user's decision of 2026-10-27): our own LoRA, train_entry.py freezes everything else itself
        if freeze_modules:
            raise ValueError("--lora freezes the model itself; do not combine it with --freeze-modules")
        trainer["lora"] = dict(lora)
    if cotrain:  # C1 (the user's decision of 2026-10-30): a text loss beside the action loss, cotrain_entry.py
        if lora or freeze_modules:
            raise ValueError("--cotrain trains the whole model: no --lora or --freeze-modules with it")
        trainer["cotrain"] = dict(cotrain)
    if full_state_interval:
        trainer.update(full_state_interval=int(full_state_interval),
                       eval_points=[int(p) for p in eval_points if 0 < int(p) < int(steps)])
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
    ap.add_argument("--full-state-interval", type=int, default=0,
                    help="stage 1: save the whole training state every N steps (newest kept) and run segmented "
                         "(downstream/train/run_segmented.sh); --save-interval must divide N and the eval points")
    ap.add_argument("--eval-points", default="", help="stage 1: comma-separated steps evaluated inside the job "
                                                      "(the last step always is)")
    ap.add_argument("--archive-suffix", default="",
                    help="reuse an existing experiment directory whose run/, job/ and evals/ are gone (renamed): "
                         "its experiment.json, config.yaml and sources.json are renamed with this suffix, never deleted")
    ap.add_argument("--repeat-final", action="store_true",
                    help="the final checkpoint gets a second evaluation, same seeds, as one 8-card job on an idle "
                         "machine (the noise floor; only start B, the user's decision of 2026-10-22)")
    ap.add_argument("--freeze-modules", default="",
                    help="comma-separated module paths StarVLA freezes (trainer.freeze_modules), e.g. "
                         "qwen_vl_interface = the whole VLM, so only the action head trains (X1)")
    ap.add_argument("--lora", default="", help="T2: LoRA on the language model, e.g. rank=16,alpha=32,dropout=0.05,lr=1e-4 "
                                               "(trainer.lora; only LoRA and the action head train)")
    ap.add_argument("--eval-plan", default="", help="recorded in experiment.json: the plan of the in-job evaluation "
                                                    "when it is not eval_d6.json (the job sets EVAL_PLAN; T5)")
    ap.add_argument("--cotrain", default="", help="C1: a text loss beside the action loss, e.g. lambda=0.1,diag_every=10 "
                                                  "(trainer.cotrain; run with downstream/train/cotrain_entry.py)")
    args = ap.parse_args(argv)
    points = [int(p) for p in args.eval_points.split(",") if p.strip()]
    for v in [args.full_state_interval] + points:
        if v and v % args.save_interval:
            raise SystemExit(f"{v} is not a multiple of --save-interval {args.save_interval}")

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
    if args.archive_suffix and exp_dir.is_dir():
        busy = [n for n in ("run", "job", "evals") if (exp_dir / n).exists()]
        if busy:
            raise SystemExit(f"{exp_dir} still has {busy}: rename them first")
        for name in ("experiment.json", "config.yaml", "sources.json"):
            old = exp_dir / name
            if old.exists():
                new = exp_dir / f"{old.stem}{args.archive_suffix}{old.suffix}"
                if new.exists():
                    raise SystemExit(f"{new} exists already")
                old.rename(new)
    else:
        exp_dir.mkdir(parents=True, exist_ok=False)
    recipe = yaml.safe_load(args.recipe.read_text())
    cfg = build_config(recipe, exp_dir=exp_dir, exp_id=args.exp_id, start_path=start["path"], data_dir=args.data,
                       seed=args.seed, steps=args.steps, save_interval=args.save_interval, log_every=args.log_every,
                       full_state_interval=args.full_state_interval, eval_points=points,
                       freeze_modules=args.freeze_modules, lora=parse_lora(args.lora),
                       cotrain=parse_cotrain(args.cotrain))
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
        **({"freeze_modules": args.freeze_modules} if args.freeze_modules else {}),
        **({"lora": parse_lora(args.lora)} if args.lora else {}),
        **({"cotrain": parse_cotrain(args.cotrain)} if args.cotrain else {}),
        **({"eval_plan": args.eval_plan} if args.eval_plan else {}),
        **({"segmented": True, "full_state_interval": args.full_state_interval,
            "eval_points": sorted({p for p in points if 0 < p <= args.steps} | {args.steps}),
            "injob_eval": {"servers": 8, "workers": 96}, "repeat_final": bool(args.repeat_final)}
           if args.full_state_interval else {}),
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
