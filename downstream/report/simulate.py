"""SIMULATED results in the real directory layout, for the tests and for a preview of the report before real
evaluations exist.  Never mix with real runs: write under a scratch root, and the report marks the page as simulated.

    python -m downstream.report.simulate --root /tmp/ds-report-sim

What it writes (same files as the real pipeline, every number invented): the 36 step-3 experiments
DS-<start>-<ratio>-s<seed> with experiment.json and a loss curve; the D5 runs (100%, seed 1) evaluated every 25,000
steps, all runs at step 200,000; the 950-episode plan of eval_d6.json per evaluation; the DS-B-100-s1 step-25,000
repeat at P = 48 and 96; and example strips (drawn boxes, labelled SIMULATED) for the spatial tasks' first 6 seeds.
Successes come from a logistic model: task difficulty + a per-scene term shared by all starts (so starts are
correlated as in a paired evaluation) + start, data-ratio and step effects.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOWNSTREAM = HERE.parent
sys.path.insert(0, str(DOWNSTREAM / "eval"))
import closed_loop  # noqa: E402

from . import collect  # noqa: E402

STARTS = ("B", "S", "RS", "RD")
EFFECT = {"B": 0.0, "S": 0.10, "RS": 0.30, "RD": 0.20}            # logit shift, spatial tasks
EFFECT_OTHER = {"B": 0.0, "S": 0.05, "RS": 0.05, "RD": 0.0}        # logit shift, other tasks
RATIO_SHIFT = {"010": -1.4, "030": -0.6, "100": 0.0}
COLORS = {"B": (122, 130, 145), "S": (53, 105, 181), "RS": (217, 130, 43), "RD": (47, 154, 106)}


def sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def step_shift(step: int) -> float:
    return -2.2 + 2.2 * (1 - math.exp(-step / 50000))


def write_strip(path: Path, start: str, task: str, seed: int, success: bool, steps: int, size: int = 64):
    from PIL import Image, ImageDraw

    n = 8
    img = Image.new("RGB", (size * n, size + 18), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    r, g, b = COLORS.get(start, (90, 90, 90))
    rng = random.Random(f"{task}{seed}")
    for i in range(n):
        shade = 0.55 + 0.45 * i / (n - 1)
        draw.rectangle([i * size, 0, (i + 1) * size - 2, size - 1], fill=(int(r * shade), int(g * shade), int(b * shade)))
        x = rng.randint(6, size - 18)
        draw.rectangle([i * size + x, 20, i * size + x + 10, 30], fill=(240, 240, 240))
        draw.text((i * size + 3, size + 1), f"t={round(i * steps / (n - 1))}", fill=(40, 40, 40))
    draw.text((4, 4), "SIMULATED", fill=(255, 255, 255))
    draw.rectangle([0, size + 14, img.width, img.height], fill=(34, 139, 34) if success else (200, 40, 40))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, quality=80)


def evaluation(ep_dir: Path, start: str, ratio: str, step: int, train_seed: int, plan, tasks, scene, rng,
               workers=24, slow=1.0, frames=True, flip=0.0):
    rows = []
    flips = random.Random(f"flip{ep_dir.name}")  # separate, so the outcome sequence itself matches the first job's
    for e in plan:
        info = tasks[e["task"]]
        effect = EFFECT[start] if info.klass == "spatial" else EFFECT_OTHER[start]
        p = sigmoid(scene[(e["task"], e["seed"])] + effect + RATIO_SHIFT[ratio] + step_shift(step)
                    + 0.15 * (train_seed - 2))
        ok = rng.random() < p
        if flip and flips.random() < flip:
            ok = not ok
        steps = int(e["max_steps"] * rng.uniform(0.2, 0.85)) if ok else e["max_steps"]
        row = dict(e, success=ok, steps=steps, status="complete", reset_seconds=21.6,
                   wall_seconds=round((21.6 + 0.196 * steps) * slow, 2),
                   init={"instruction": f"(simulated) {e['task']} seed {e['seed']}"})
        if frames and info.klass == "spatial" and e["seed"] < 1006:
            rel = f"frames/{e['task']}_s{e['seed']}.jpg"
            write_strip(ep_dir / rel, start, e["task"], e["seed"], ok, steps)
            row["frames"] = rel
        rows.append(row)
    ep_dir.mkdir(parents=True, exist_ok=True)
    (ep_dir / "episodes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    wall = sum(r["wall_seconds"] for r in rows) / workers
    summary = dict(closed_loop.summarize(rows, workers, wall), stable_counter_regions=True,
                   eval_plan="seen34_spatial50_v2", simulated=True)
    (ep_dir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")


def simulate(root: Path, seed: int = 0, d5_steps=(25000, 50000, 75000, 100000, 125000, 150000, 175000, 200000),
             ratios=("010", "030", "100"), train_seeds=(1, 2, 3), frames: bool = True) -> Path:
    root = Path(root)
    rng = random.Random(seed)
    tasks = collect.task_table()
    plan, _ = closed_loop.eval_plan_episodes(DOWNSTREAM / "eval/eval_d6.json")
    difficulty = {t: rng.uniform(-2.5, 1.0) for t in tasks}
    scene = {(e["task"], e["seed"]): difficulty[e["task"]] + rng.gauss(0, 1.2) for e in plan}
    (root / "SIMULATED").parent.mkdir(parents=True, exist_ok=True)
    (root / "SIMULATED").write_text("every number under this directory is invented (downstream/report/simulate.py)\n")
    for ratio in ratios:
        for start in STARTS:
            for k in train_seeds:
                exp = root / "experiments" / f"DS-{start}-{ratio}-s{k}"
                (exp / "run").mkdir(parents=True, exist_ok=True)
                (exp / "experiment.json").write_text(json.dumps({"steps": 200000, "seed": k, "simulated": True}))
                lrng = random.Random(f"{start}{ratio}{k}")
                offset = {"B": 0.0, "S": 0.012, "RS": 0.013, "RD": 0.015}[start]
                (exp / "run/metrics.jsonl").write_text("".join(
                    json.dumps({"step": s, "action_dit_loss": 0.055 + 0.25 * math.exp(-s / 3000)
                                + offset * math.exp(-s / 8000) + lrng.gauss(0, 0.012) * math.exp(-s / 60000)}) + "\n"
                    for s in range(100, 200001, 100)))
                steps = d5_steps if (ratio == "100" and k == 1) else (200000,)
                for step in steps:
                    evaluation(exp / f"evals/steps_{step}/out/episodes", start, ratio, step, k, plan, tasks, scene,
                               random.Random(f"{start}{ratio}{k}{step}"),
                               frames=frames and ratio == "100" and k == 1 and step in (d5_steps[0], 200000))
    b = root / "experiments/DS-B-100-s1/evals" / f"steps_{d5_steps[0]}_repeat/out"
    for p, slow, flip in ((48, 1.25, 0.01), (96, 1.7, 0.01)):
        evaluation(b / f"episodes_P{p}", "B", "100", d5_steps[0], 1, plan, tasks, scene,
                   random.Random(f"B1001{d5_steps[0]}"), workers=p, slow=slow, frames=False, flip=flip)
    return root


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)
    if (args.root / "experiments").exists() and not (args.root / "SIMULATED").exists():
        raise SystemExit(f"{args.root} has experiments that are not simulated; refusing to write there")
    print(simulate(args.root, args.seed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
