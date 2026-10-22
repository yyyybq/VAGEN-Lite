"""Read what the runs wrote: training losses, evaluations (first and repeat jobs) and their episodes.

Layout (downstream/train/runs.py, downstream/eval/watch.py, downstream/eval/closed_loop.py):
  <runs-root>/experiments/DS-<start>-<ratio>-s<seed>/experiment.json, run/metrics.jsonl
      evals/steps_<N>/out/episodes/{summary.json, episodes.jsonl, plan.json, frames/<task>_s<seed>.jpg}
      evals/steps_<N>_repeat/out/episodes_P<P>/...   (the noise / parallelism test)
Only finished evaluations count (summary.json with per_task).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

DOWNSTREAM = Path(__file__).resolve().parents[1]
EXP_RE = re.compile(r"^DS-(?P<start>[A-Z0-9]+)-(?P<ratio>\d{3})-s(?P<seed>\d+)$")
START_ORDER = ("B", "S", "RS", "RD", "R10S5", "R1")
RATIO_PERCENT = {"010": 10, "030": 30, "100": 100}


@dataclass
class TaskInfo:
    task: str
    group: str        # atomic_seen / composite_seen
    klass: str        # spatial / borderline / other
    horizon: int


@dataclass
class Evaluation:
    exp: str
    start: str
    ratio: str
    seed: int
    step: int
    kind: str          # "first" (the regular evaluation) or "repeat"
    workers: int
    directory: Path
    summary: dict
    episodes: list = field(default_factory=list)

    @property
    def successes(self) -> int:
        return sum(int(e["success"]) for e in self.episodes)


def task_table(downstream: Path = DOWNSTREAM) -> dict:
    tasks = json.loads((downstream / "demos/target50_tasks.json").read_text())["tasks"]
    sp = json.loads((downstream / "demos/spatial_tasks.json").read_text())
    spatial, border = set(sp["spatial"]), set(sp.get("borderline_not_in_group", {}))
    return {t["task"]: TaskInfo(t["task"], t["group"],
                                "spatial" if t["task"] in spatial else "borderline" if t["task"] in border else "other",
                                int(t["horizon"]))
            for t in tasks if t["group"] in ("atomic_seen", "composite_seen")}


def parse_exp(name: str):
    m = EXP_RE.match(name)
    return (m["start"], m["ratio"], int(m["seed"])) if m else None


def read_jsonl(path: Path) -> list:
    if not path.is_file():
        return []
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def losses(exp_dir: Path) -> list:
    """[(step, loss)] in step order."""
    rows = read_jsonl(exp_dir / "run/metrics.jsonl")
    return sorted((int(r["step"]), float(r["action_dit_loss"])) for r in rows if "action_dit_loss" in r)


def _evaluation(exp_dir, start, ratio, seed, step, kind, ep_dir: Path):
    summary_path = ep_dir / "summary.json"
    if not summary_path.is_file():
        return None
    summary = json.loads(summary_path.read_text())
    if "per_task" not in summary:
        return None
    eps = []
    for r in read_jsonl(ep_dir / "episodes.jsonl"):
        if r.get("status") != "complete":
            continue
        frames = ep_dir / r["frames"] if r.get("frames") else None
        eps.append({"task": r["task"], "eval_seed": int(r["seed"]), "success": bool(r["success"]),
                    "steps": int(r["steps"]), "max_steps": int(r["max_steps"]),
                    "wall_seconds": float(r.get("wall_seconds") or 0.0),
                    "instruction": (r.get("init") or {}).get("instruction"),
                    "frames": frames if frames and frames.is_file() else None})
    return Evaluation(exp_dir.name, start, ratio, seed, step, kind, int(summary.get("workers") or 0), ep_dir,
                      summary, eps)


def _sharded(exp_dir, start, ratio, seed, step, kind, evaldir: Path):
    """One evaluation from its one-card shards (shards/s<I>of<K>/out/episodes), None until all K have a result."""
    parts = sorted((evaldir / "shards").glob("s*of*"))
    if not parts:
        return None
    k = int(parts[0].name.split("of")[1])
    pieces = [_evaluation(exp_dir, start, ratio, seed, step, kind, evaldir / "shards" / f"s{i}of{k}" / "out/episodes")
              for i in range(1, k + 1)]
    if any(p is None for p in pieces):
        return None
    eps = [x for p in pieces for x in p.episodes]
    summary = {"per_task": {}, "shards": k, "workers": pieces[0].workers,
               "episode_wall_mean_seconds": sum(x["wall_seconds"] for x in eps) / max(1, len(eps)),
               "wall_seconds": max(p.summary.get("wall_seconds") or 0 for p in pieces)}
    for x in eps:
        t = summary["per_task"].setdefault(x["task"], {"n": 0, "successes": 0})
        t["n"] += 1
        t["successes"] += int(x["success"])
    return Evaluation(exp_dir.name, start, ratio, seed, step, kind, pieces[0].workers, evaldir, summary, eps)


def evaluations(exp_dir: Path) -> list:
    parsed = parse_exp(exp_dir.name)
    if not parsed:
        return []
    start, ratio, seed = parsed
    out = []
    for d in sorted((exp_dir / "evals").glob("steps_*")):
        m = re.match(r"^steps_(\d+)(_repeat)?$", d.name)
        if not m:
            continue
        step = int(m[1])
        if (d / "shards").is_dir():
            e = _sharded(exp_dir, start, ratio, seed, step, "repeat" if m[2] else "first", d)
            if e:
                out.append(e)
            continue
        if m[2]:
            for ep_dir in sorted((d / "out").glob("episodes_P*")) or [d / "out/episodes"]:
                e = _evaluation(exp_dir, start, ratio, seed, step, "repeat", ep_dir)
                if e:
                    out.append(e)
        else:
            e = _evaluation(exp_dir, start, ratio, seed, step, "first", d / "out/episodes")
            if e:
                out.append(e)
    return out


@dataclass
class Collection:
    runs_root: Path
    tasks: dict
    experiments: dict          # exp -> (start, ratio, seed)
    losses: dict               # exp -> [(step, loss)]
    evals: list                # Evaluation, first and repeat

    def first(self):
        return [e for e in self.evals if e.kind == "first"]

    def starts(self):
        present = {e[0] for e in self.experiments.values()}
        return [s for s in START_ORDER if s in present] + sorted(present - set(START_ORDER))


def collect(runs_root: Path, downstream: Path = DOWNSTREAM) -> Collection:
    exps, loss, evs = {}, {}, []
    for exp_dir in sorted((Path(runs_root) / "experiments").glob("DS-*")):
        parsed = parse_exp(exp_dir.name)
        if not parsed or not (exp_dir / "experiment.json").is_file():
            continue
        exps[exp_dir.name] = parsed
        loss[exp_dir.name] = losses(exp_dir)
        evs += evaluations(exp_dir)
    return Collection(Path(runs_root), task_table(downstream), exps, loss, evs)
