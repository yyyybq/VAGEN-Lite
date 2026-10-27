"""A StarVLA experiment folder of the model line: the downstream line's recipe and config builder, read only
(VAGEN-Lite-downstream/downstream/train/{recipe.yaml, make_experiment.build_config}), with a start of this line
(model_matrix/downstream/starts.json).  The only difference between the downstream line's experiments and these is
the start, as the downstream protocol has it.

    python -m model_matrix.downstream.make_config --start Q3B --data DATA_DIR --exp-dir EXP --steps 100 \
        --save-interval 100 --log-every 10 --seed 1

Writes EXP/config.yaml and EXP/experiment.json (the start, its path, the data, the recipe's sha256, the commits).

A full run (2026-10-26, the user's approval of DS-Q3B-100-s1): the downstream line's stage-1 form of DS-B-100-s1
(make_experiment --full-state-interval 25000 --eval-points 200000: the whole training state every 25k steps, newest
kept, the final checkpoint evaluated inside the job), run by model_matrix/downstream/run_full.sh:

    python -m model_matrix.downstream.make_config --start Q3B --data .../target50_100 --exp-dir EXP --steps 200000 \
        --save-interval 25000 --log-every 100 --seed 1 --full-state-interval 25000 --eval-points 200000

The record then has the downstream line's fields (segmented, full_state_interval, eval_points, injob_eval,
repeat_final false: only their start B repeats its final evaluation) and the start's lock (the downstream line's
verify_starts.fingerprint: index and config sha256, shard sizes and mtimes, and each shard's first and last 4 MiB
when readable; for the root-only Qwen3-VL snapshot, every file's size, mtime and blob name), which
``--check-start EXP`` compares at the start of a job.
``--qwenvl KEY=VALUE`` (repeatable) sets framework.qwenvl.KEY in the config, e.g. the Cambrian interface's
``cambrian_checkpointing=vision`` (model_matrix/downstream/starvla_cambrian.py).
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

DS_WORKTREE = Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite-downstream")
RECIPE = DS_WORKTREE / "downstream" / "train" / "recipe.yaml"
STARTS = Path(__file__).with_name("starts.json")
ROOT = Path(__file__).resolve().parents[2]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def commit(repo: Path):
    res = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True)
    return res.stdout.strip() if res.returncode == 0 else None


def _verify_starts():
    sys.path.insert(0, str(DS_WORKTREE / "downstream"))
    import verify_starts

    return verify_starts


def stat_lock(path: str) -> dict:
    """Every file of a start folder by what a non-owner can see: size and mtime (through the symlink) and the
    symlink's target -- in a Hugging Face cache snapshot the blob name, the content's sha256 for the weight files."""
    root = Path(path)
    files = {}
    for p in sorted(root.iterdir()):
        st = p.stat()
        files[p.name] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns,
                         "target": os.readlink(p) if p.is_symlink() else None}
    return {"path": str(root), "stat_only": True, "files": files}


def lock(path: str) -> dict:
    """The downstream line's fingerprint of a start: deep when every shard is readable, else with the mtimes; when
    not even the index is readable (the Qwen3-VL snapshot is root-only), stat_lock."""
    vs = _verify_starts()
    try:
        return vs.fingerprint(path, deep=True)
    except PermissionError:
        pass
    try:
        out = vs.fingerprint(path, deep=False)
    except PermissionError:
        return stat_lock(path)
    out["shard_mtime_ns"] = {name: (Path(path) / name).stat().st_mtime_ns for name in out["shards"]}
    return out


def check_start(exp_dir: Path) -> list:
    """What changed in the experiment's start since its lock (an empty list: nothing)."""
    rec = json.loads((Path(exp_dir) / "experiment.json").read_text())
    locked = rec["start"].get("lock")
    if not locked:
        return ["the experiment has no start lock"]
    if locked.get("stat_only"):
        try:
            now = stat_lock(locked["path"])["files"]
        except Exception as exc:
            return [f"{type(exc).__name__}: {exc}"]
        old = locked["files"]
        return [f"{name}: {old.get(name)} -> {now.get(name)}" for name in sorted(set(old) | set(now))
                if old.get(name) != now.get(name)]
    vs = _verify_starts()
    deep = "shard_head_tail_sha256" in locked
    try:
        current = vs.fingerprint(locked["path"], deep=deep)
        if not deep:
            current["shard_mtime_ns"] = {n: (Path(locked["path"]) / n).stat().st_mtime_ns for n in current["shards"]}
    except Exception as exc:  # a missing file is a change too
        return [f"{type(exc).__name__}: {exc}"]
    return vs.compare(locked, current)


def parse_value(text: str):
    """A --qwenvl value: JSON when it parses (true, 1, "x"), else the string."""
    try:
        return json.loads(text)
    except ValueError:
        return text


def build(start: str, data: Path, exp_dir: Path, steps: int, save_interval: int, log_every: int, seed: int,
          recipe_path: Path = RECIPE, full_state_interval: int = 0, eval_points=(), qwenvl=None,
          gpus: int = 8) -> dict:
    import yaml

    sys.path.insert(0, str(DS_WORKTREE / "downstream" / "train"))
    from make_experiment import build_config

    starts = json.loads(STARTS.read_text())
    if start not in starts or start.startswith("_"):
        raise SystemExit(f"unknown start {start}; known: {sorted(k for k in starts if not k.startswith('_'))}")
    for v in [full_state_interval] + list(eval_points):  # make_experiment's rule
        if v and v % save_interval:
            raise SystemExit(f"{v} is not a multiple of --save-interval {save_interval}")
    entry = starts[start]
    recipe = yaml.safe_load(recipe_path.read_text())
    exp_dir.mkdir(parents=True, exist_ok=False)
    kwargs = {"full_state_interval": full_state_interval, "eval_points": list(eval_points)} if full_state_interval else {}
    cfg = build_config(recipe, exp_dir=exp_dir, exp_id=exp_dir.name, start_path=entry["path"], data_dir=data,
                       seed=seed, steps=steps, save_interval=save_interval, log_every=log_every, **kwargs)
    for key, value in (qwenvl or {}).items():
        cfg["framework"]["qwenvl"][key] = value
    (exp_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    record = {"exp_id": exp_dir.name, "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "start": {"name": start, **entry}, "data": str(data), "seed": seed, "steps": steps,
              "save_interval": save_interval, "recipe": {"path": str(recipe_path), "sha256": sha256(recipe_path)},
              "code": {"models": commit(ROOT), "downstream": commit(DS_WORKTREE)}}
    if qwenvl:
        record["qwenvl"] = dict(qwenvl)
    if full_state_interval:  # the downstream line's stage-1 fields (make_experiment), for its full_state.py
        record["start"]["lock"] = lock(entry["path"])
        record["data"] = {"dir": str(data)}
        record.update(segmented=True, full_state_interval=int(full_state_interval),
                      eval_points=sorted({int(p) for p in eval_points if 0 < int(p) <= steps} | {int(steps)}),
                      injob_eval={"servers": 8, "workers": 96}, repeat_final=False, gpus=gpus,
                      per_device_batch=cfg["datasets"]["vla_data"]["per_device_batch_size"],
                      global_batch=cfg["datasets"]["vla_data"]["per_device_batch_size"] * gpus,
                      config_sha256=sha256(exp_dir / "config.yaml"), output_dir=str(exp_dir / "run"))
    (exp_dir / "experiment.json").write_text(json.dumps(record, indent=1))
    return record


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check-start", type=Path, default=None, metavar="EXP",
                    help="compare EXP's start with its lock; exit 3 when it changed")
    ap.add_argument("--start")
    ap.add_argument("--data", type=Path)
    ap.add_argument("--exp-dir", type=Path)
    ap.add_argument("--steps", type=int)
    ap.add_argument("--save-interval", type=int)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--full-state-interval", type=int, default=0)
    ap.add_argument("--eval-points", default="")
    ap.add_argument("--qwenvl", action="append", default=[], metavar="KEY=VALUE")
    a = ap.parse_args()
    if a.check_start:
        problems = check_start(a.check_start)
        print(json.dumps({"exp": str(a.check_start), "problems": problems}, indent=1))
        sys.exit(3 if problems else 0)
    missing = [f for f in ("start", "data", "exp_dir", "steps", "save_interval") if getattr(a, f) is None]
    if missing:
        ap.error("missing " + ", ".join("--" + m.replace("_", "-") for m in missing))
    qwenvl = {}
    for item in a.qwenvl:
        key, sep, value = item.partition("=")
        if not sep or not key:
            ap.error(f"--qwenvl wants KEY=VALUE, not {item!r}")
        qwenvl[key] = parse_value(value)
    points = [int(p) for p in a.eval_points.split(",") if p.strip()]
    print(json.dumps(build(a.start, a.data, a.exp_dir, a.steps, a.save_interval, a.log_every, a.seed,
                           full_state_interval=a.full_state_interval, eval_points=points, qwenvl=qwenvl), indent=1))


if __name__ == "__main__":
    main()
