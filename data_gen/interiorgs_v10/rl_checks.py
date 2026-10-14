"""Checks of SFT data and RL reward in the RL env itself (round 7, 2026-10-14).

Run with the env config RL actually uses (``TR/rl/train_env.yaml``) and the RL render service, on one card:

    R1_RENDER_URL=http://127.0.0.1:8915/render python -m data_gen.interiorgs_v10.rl_checks run \\
        --env-yaml TR/rl/train_env.yaml --rl-rows TR/rl/train.jsonl --sft SFT_DIR --out OUT
    python -m data_gen.interiorgs_v10.rl_checks report --out OUT

``sft``: SFT conversations replayed in the env from their start: every user turn's image (pixel difference),
text (character by character, the pose line separately), the system prompt, and how a conversation ends.
``shortest``: the task's shortest action sequence in the RL answer format (runs of one action per turn):
success, termination and the reward of every turn with its components.
``baselines``: model-free policies (a random action, keep turning left, keep moving forward), one action per
turn so success is checked after every action, up to the env's turn limit.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np

ACTIONS = ("move_forward", "move_backward", "move_left", "move_right", "turn_left", "turn_right")
POSE = re.compile(r"Current camera pose: \[[^\]]*\]")
POLICIES = ("random", "turn_left", "forward")
TERMINATION_FLAGS = ("auto_terminated", "truncated_max_primitive_steps", "early_terminated_collision",
                     "early_terminated_invalid", "early_terminated_low_info")


def _jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def band(d: int) -> str:
    return "1-3" if d <= 3 else ("4-6" if d <= 6 else "7-8")


def stratified(rows: Sequence[Dict[str, Any]], n: int, seed: int) -> List[Dict[str, Any]]:
    """``n`` rows spread evenly over families and, inside a family, over d* bands (deterministic)."""
    rng = random.Random(seed)
    cells: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for r in sorted(rows, key=lambda r: r["task_id"]):
        cells[(r["task_type"], band(int(r["difficulty"]["d_star"])))].append(r)
    for v in cells.values():
        rng.shuffle(v)
    fams = sorted({k[0] for k in cells})
    quota = {f: n // len(fams) + (1 if i < n % len(fams) else 0) for i, f in enumerate(fams)}
    out: List[Dict[str, Any]] = []
    for f in fams:
        bands = sorted(k for k in cells if k[0] == f)
        take = {b: 0 for b in bands}
        while sum(take.values()) < quota[f] and any(take[b] < len(cells[b]) for b in bands):
            b = min((b for b in bands if take[b] < len(cells[b])), key=lambda b: (take[b], b))
            take[b] += 1
        for b in bands:
            out += cells[b][: take[b]]
    return out


# ---------------------------------------------------------------------------
# the env
# ---------------------------------------------------------------------------
def env_config(env_yaml: Path, render_url: str) -> Dict[str, Any]:
    import yaml

    entry = yaml.safe_load(Path(env_yaml).read_text())["envs"][0]
    cfg = dict(entry["config"], client_url=render_url)
    cfg["_max_turns"] = int(entry.get("max_turns", 12))
    return cfg


def make_env(rows: Sequence[Dict[str, Any]], path: Path, cfg: Dict[str, Any]):
    """An ActiveSpatialEnv over ``rows`` (written, scene by scene, to ``path``); returns (env, ordered rows)."""
    from vagen.envs.active_spatial.env import ActiveSpatialEnv
    from vagen.envs.active_spatial.env_config import ActiveSpatialEnvConfig

    ordered = sorted(rows, key=lambda r: (r["scene_id"], r["task_id"]))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in ordered))
    keys = set(ActiveSpatialEnvConfig.__dataclass_fields__)
    env = ActiveSpatialEnv(ActiveSpatialEnvConfig(**{k: v for k, v in cfg.items() if k in keys},
                                                  jsonl_path=str(path)))
    return env, ordered


def _image(obs: Dict[str, Any]):
    ims = [im for v in (obs.get("multi_modal_data") or {}).values() for im in (v or [])]
    return ims[0] if len(ims) == 1 else None


def pixel_difference(a, b) -> Dict[str, Any]:
    x = np.asarray(a.convert("RGB"), np.int16)
    y = np.asarray(b.convert("RGB"), np.int16)
    if x.shape != y.shape:
        return {"shape_env": list(x.shape), "shape_sft": list(y.shape)}
    d = np.abs(x - y)
    return {"max": int(d.max()), "mean": float(d.mean())}


def step(env, text: str) -> Dict[str, Any]:
    """One env.step with what the checks need: reward, termination and the reward components of the turn."""
    obs, reward, done, info = env.step(text)
    tm = (info.get("metrics") or {}).get("turn_metrics") or {}
    trace = info.get("reward_trace") or {}
    comps = {k: {"applied": v["applied"], "scale": v["scale"], "events": v["events"]}
             for k, v in (trace.get("components") or {}).items() if v.get("events")}
    return {"obs": obs, "reward": float(reward), "done": bool(done), "success": bool(info.get("success")),
            "env_step": int(info.get("env_step", 0)), "collisions": int(tm.get("collision_count", 0) or 0),
            "low_info": bool(info.get("low_info_frame")), "flags": [f for f in TERMINATION_FLAGS if info.get(f)],
            "components": comps,
            "reconstruction_error": (trace.get("accounting") or {}).get("reconstruction_error")}


def _public(rec: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in rec.items() if k != "obs"}


# ---------------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------------
def sft_replay(records: Sequence[Dict[str, Any]], rows_by_id: Dict[str, Dict[str, Any]], workdir: Path,
               cfg: Dict[str, Any], open_image: Callable[[str], Any]) -> List[Dict[str, Any]]:
    """Each SFT conversation replayed from its start; user turns compared with the env's observations."""
    env, ordered = make_env([rows_by_id[r["source_task_id"]] for r in records], Path(workdir) / "sft_tasks.jsonl", cfg)
    by_id = {r["source_task_id"]: r for r in records}
    out = []
    for i, row in enumerate(ordered):
        conv = by_id[row["task_id"]]["conversations"]
        users = [t for t in conv if t["role"] == "user"]
        answers = [t for t in conv if t["role"] == "assistant"]
        res: Dict[str, Any] = {"task_id": row["task_id"], "task_type": row["task_type"],
                               "d_star": int(row["difficulty"]["d_star"]), "ends_with": conv[-1]["role"],
                               "compare": [], "steps": []}
        try:
            obs, _ = env.reset(seed=i)
            res["system_identical"] = env.system_prompt() == conv[0]["content"]
            for k, u in enumerate(users):
                if k > 0:
                    rec = step(env, answers[k - 1]["content"])
                    obs = rec["obs"]
                    res["steps"].append(_public(rec))
                text = obs.get("obs_str", "")
                pe, ps = POSE.search(text), POSE.search(u["content"])
                im = _image(obs)
                res["compare"].append({"turn": k, "text_identical": text == u["content"],
                                       "pose_identical": bool(pe and ps and pe.group(0) == ps.group(0)),
                                       "pixels": pixel_difference(im, open_image(u["image_path"])) if im else None})
            res["final"] = _public(step(env, answers[-1]["content"]))
        except Exception as exc:  # recorded, counted by the report
            res["error"] = f"{type(exc).__name__}: {exc}"
        out.append(res)
    return out


def shortest_replay(rows: Sequence[Dict[str, Any]], workdir: Path, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The certificate answered in the RL format (one run of identical actions per turn)."""
    from .initial_states import chunk_turns

    env, ordered = make_env(rows, Path(workdir) / "shortest_tasks.jsonl", cfg)
    out = []
    for i, row in enumerate(ordered):
        turns = chunk_turns(row["difficulty"]["certificate"])
        res: Dict[str, Any] = {"task_id": row["task_id"], "task_type": row["task_type"],
                               "d_star": int(row["difficulty"]["d_star"]), "n_turns_planned": len(turns), "steps": []}
        try:
            env.reset(seed=i)
            for turn in turns:
                rec = _public(step(env, "<action>" + "|".join(turn) + "|</action>"))
                res["steps"].append(rec)
                if rec["done"]:
                    break
        except Exception as exc:
            res["error"] = f"{type(exc).__name__}: {exc}"
        out.append(res)
    return out


def baselines(rows: Sequence[Dict[str, Any]], workdir: Path, cfg: Dict[str, Any], seed: int,
              policies: Iterable[str] = POLICIES) -> List[Dict[str, Any]]:
    env, ordered = make_env(rows, Path(workdir) / "baseline_tasks.jsonl", cfg)
    max_turns = int(cfg.get("_max_turns", 12))
    out = []
    for i, row in enumerate(ordered):
        for policy in policies:
            rng = random.Random(f"{seed}-{row['task_id']}")
            res: Dict[str, Any] = {"task_id": row["task_id"], "task_type": row["task_type"], "policy": policy,
                                   "d_star": int(row["difficulty"]["d_star"])}
            try:
                env.reset(seed=i)
                total, success, steps = 0.0, False, []
                for _ in range(max_turns):
                    a = {"random": rng.choice(ACTIONS), "turn_left": "turn_left", "forward": "move_forward"}[policy]
                    rec = _public(step(env, f"<action>{a}|</action>"))
                    total += rec["reward"]
                    steps.append(rec)
                    success = success or rec["success"]
                    if rec["done"]:
                        break
                res.update(success=success, total_reward=total, turns=len(steps),
                           ended_by=(steps[-1]["flags"] or ["turn_limit"]) if steps else [])
            except Exception as exc:
                res["error"] = f"{type(exc).__name__}: {exc}"
            out.append(res)
    return out


# ---------------------------------------------------------------------------
# summaries
# ---------------------------------------------------------------------------
def summarize_sft(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    comps = [c for r in results for c in r.get("compare", [])]
    px = [c["pixels"] for c in comps if c.get("pixels") and "max" in c["pixels"]]
    finals = [r.get("final") or {} for r in results]
    return {"conversations": len(results), "errors": sum("error" in r for r in results),
            "system_identical": sum(bool(r.get("system_identical")) for r in results),
            "user_turns": len(comps), "text_identical": sum(c["text_identical"] for c in comps),
            "pose_identical": sum(c["pose_identical"] for c in comps),
            "first_turn_text_identical": sum(c["text_identical"] for c in comps if c["turn"] == 0),
            "pixel_max": max((p["max"] for p in px), default=None),
            "ends_with": dict(Counter(r.get("ends_with") for r in results)),
            "final_success": sum(bool(f.get("success")) for f in finals),
            "final_done": sum(bool(f.get("done")) for f in finals),
            "early_done": sum(any(s["done"] for s in r.get("steps", [])) for r in results)}


def summarize_shortest(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    ok = [r for r in results if r["steps"] and "error" not in r]
    rewards: Counter = Counter()
    for r in ok:
        for k, s in enumerate(r["steps"]):
            rewards[("last" if k == len(r["steps"]) - 1 else "earlier", round(s["reward"], 4))] += 1
    return {"tasks": len(results), "errors": sum("error" in r for r in results),
            "success": sum(r["steps"][-1]["success"] for r in ok),
            "success_at_d_star": sum(r["steps"][-1]["success"] and r["steps"][-1]["env_step"] == r["d_star"] for r in ok),
            "done_on_last_planned_turn": sum(len(r["steps"]) == r["n_turns_planned"] and r["steps"][-1]["done"] for r in ok),
            "collisions": sum(s["collisions"] for r in ok for s in r["steps"]),
            "rewards": {f"{k[0]}:{k[1]}": v for k, v in sorted(rewards.items())},
            "max_reconstruction_error": max((abs(s.get("reconstruction_error") or 0.0) for r in ok for s in r["steps"]),
                                            default=0.0)}


def summarize_baselines(results: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    out = {}
    for policy in sorted({r["policy"] for r in results}):
        rs = [r for r in results if r["policy"] == policy and "error" not in r]
        fam: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in rs:
            fam[r["task_type"]].append(r)
        out[policy] = {"tasks": len(rs), "success": sum(r["success"] for r in rs),
                       "success_rate": round(sum(r["success"] for r in rs) / max(1, len(rs)), 4),
                       "by_family": {f: [sum(x["success"] for x in v), len(v)] for f, v in sorted(fam.items())},
                       "ended_by": dict(Counter("+".join(r["ended_by"]) for r in rs)),
                       "fewer_turns_than_d_star": sum(r["success"] and r["turns"] < r["d_star"] for r in rs)}
    return out


def cmd_run(args: argparse.Namespace) -> None:  # pragma: no cover - needs the render service
    from PIL import Image

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = env_config(Path(args.env_yaml), os.environ["R1_RENDER_URL"])
    rl_rows = _jsonl(Path(args.rl_rows))
    if "sft" in args.parts:
        recs = [r for r in _jsonl(Path(args.sft) / "sft_trajectories.jsonl") if r["split"] == "train"]
        for r in recs:
            r.setdefault("task_id", r["source_task_id"])
            r.setdefault("difficulty", {"d_star": r["d_star"]})
        picked = stratified(recs, args.n_sft, args.seed)
        rows = {r["task_id"]: r for r in _jsonl(Path(args.sft) / "rows.jsonl")}
        res = sft_replay(picked, rows, out, cfg, lambda p: Image.open(p))
        (out / "sft_replay.json").write_text(json.dumps(res, indent=1))
    if "shortest" in args.parts:
        (out / "shortest.json").write_text(json.dumps(shortest_replay(stratified(rl_rows, args.n_shortest, args.seed + 1),
                                                                      out, cfg), indent=1))
    if "baselines" in args.parts:
        (out / "baselines.json").write_text(json.dumps(baselines(stratified(rl_rows, args.n_baselines, args.seed + 2),
                                                                 out, cfg, args.seed), indent=1))


def cmd_report(args: argparse.Namespace) -> None:
    out = Path(args.out)
    report = {}
    for name, fn in (("sft_replay", summarize_sft), ("shortest", summarize_shortest), ("baselines", summarize_baselines)):
        if (out / f"{name}.json").exists():
            report[name] = fn(json.loads((out / f"{name}.json").read_text()))
    (out / "report.json").write_text(json.dumps(report, indent=1))
    print(json.dumps(report, indent=1))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--env-yaml", required=True, help="the env config RL uses (TR/rl/train_env.yaml)")
    r.add_argument("--rl-rows", required=True, help="RL training rows (TR/rl/train.jsonl)")
    r.add_argument("--sft", required=True, help="the SFT folder (sft_trajectories.jsonl, rows.jsonl)")
    r.add_argument("--out", required=True)
    r.add_argument("--parts", default="sft,shortest,baselines")
    r.add_argument("--n-sft", type=int, default=50)
    r.add_argument("--n-shortest", type=int, default=100)
    r.add_argument("--n-baselines", type=int, default=200)
    r.add_argument("--seed", type=int, default=20261014)
    p = sub.add_parser("report")
    p.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    {"run": cmd_run, "report": cmd_report}[args.cmd](args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
