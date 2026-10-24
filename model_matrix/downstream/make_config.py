"""A StarVLA experiment folder of the model line: the downstream line's recipe and config builder, read only
(VAGEN-Lite-downstream/downstream/train/{recipe.yaml, make_experiment.build_config}), with a start of this line
(model_matrix/downstream/starts.json).  The only difference between the downstream line's experiments and these is
the start, as the downstream protocol has it.

    python -m model_matrix.downstream.make_config --start Q3B --data DATA_DIR --exp-dir EXP --steps 100 \
        --save-interval 100 --log-every 10 --seed 1

Writes EXP/config.yaml and EXP/experiment.json (the start, its path, the data, the recipe's sha256, the commits).
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
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


def build(start: str, data: Path, exp_dir: Path, steps: int, save_interval: int, log_every: int, seed: int,
          recipe_path: Path = RECIPE) -> dict:
    import yaml

    sys.path.insert(0, str(DS_WORKTREE / "downstream" / "train"))
    from make_experiment import build_config

    starts = json.loads(STARTS.read_text())
    if start not in starts or start.startswith("_"):
        raise SystemExit(f"unknown start {start}; known: {sorted(k for k in starts if not k.startswith('_'))}")
    entry = starts[start]
    recipe = yaml.safe_load(recipe_path.read_text())
    exp_dir.mkdir(parents=True, exist_ok=False)
    cfg = build_config(recipe, exp_dir=exp_dir, exp_id=exp_dir.name, start_path=entry["path"], data_dir=data,
                       seed=seed, steps=steps, save_interval=save_interval, log_every=log_every)
    (exp_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    record = {"exp_id": exp_dir.name, "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
              "start": {"name": start, **entry}, "data": str(data), "seed": seed, "steps": steps,
              "save_interval": save_interval, "recipe": {"path": str(recipe_path), "sha256": sha256(recipe_path)},
              "code": {"models": commit(ROOT), "downstream": commit(DS_WORKTREE)}}
    (exp_dir / "experiment.json").write_text(json.dumps(record, indent=1))
    return record


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", required=True)
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--exp-dir", type=Path, required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--save-interval", type=int, required=True)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    print(json.dumps(build(a.start, a.data, a.exp_dir, a.steps, a.save_interval, a.log_every, a.seed), indent=1))


if __name__ == "__main__":
    main()
