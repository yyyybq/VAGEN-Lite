"""InteriorGS v10 evaluation (plan of 2026-10-14, round 6; run after training).

A fixed subset of the test split's formal tasks, the same env as RL (default v10 success rules,
``r1_legacy_success`` off), one greedy episode per task and model, on one card (render service and vLLM
share it):

    python -m data_gen.interiorgs_v10.evaluation select --run RUN/output --split-manifest RUN/split_manifest.json \\
        --out EVAL                                      # EVAL/tasks.jsonl + EVAL/selection.json
    # per model, a one-card sco job (sco_interiorgs_v10_train.sh eval RUN EVAL NAME MODEL [PROMPT_FORMAT]):
    python -m data_gen.interiorgs_v10.evaluation run --eval EVAL --name NAME --model HF_DIR
    python -m data_gen.interiorgs_v10.evaluation table --eval EVAL     # EVAL/table.md, EVAL/table.json
    # round 7: drop tasks flagged by the rendered-depth check, refill from the same family and d* band
    python -m data_gen.interiorgs_v10.evaluation replace --eval EVAL --flagged depth_check_summary.json \
        --run RUN/output --split-manifest RUN/split_manifest.json --out EVAL2

Subset: ``PER_FAMILY`` tasks per family (all of a family's test tasks when it has fewer), spread evenly over
the d* bands 1-3 / 4-6 / 7-8 (a band short of tasks gives its share to the others), drawn with ``SEED``.
Each results row also holds the trajectory (frames, answers, actions; round 8) for the browsing page.
Metrics per model: success rate (overall, per family, per d* band), actions taken / d* on successes,
collision rate (blocked moves per action) and invalid-answer rate (turns whose answer did not parse).
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PER_FAMILY = 50
SEED = 20261014
BANDS = ((1, 3), (4, 6), (7, 8))
VERSION = "interiorgs_v10_eval_v2"  # v2 (round 8): results rows carry the trajectory and frames


def band_name(d: int) -> str:
    for lo, hi in BANDS:
        if lo <= d <= hi:
            return f"{lo}-{hi}"
    return "other"


def _jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


# ---------------------------------------------------------------------------
# select
# ---------------------------------------------------------------------------
def select_tasks(rows: Sequence[Dict[str, Any]], per_family: int = PER_FAMILY, seed: int = SEED) -> List[Dict[str, Any]]:
    """Stratified by family, then by d* band; deterministic for a seed and a row set (any order)."""
    rng = random.Random(seed)
    by_family: Dict[str, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for r in sorted(rows, key=lambda r: r["task_id"]):
        by_family[r["task_type"]][band_name(int(r["difficulty"]["d_star"]))].append(r)
    picked: List[Dict[str, Any]] = []
    for fam in sorted(by_family):
        bands = {b: list(v) for b, v in by_family[fam].items()}
        for v in bands.values():
            rng.shuffle(v)
        want = min(per_family, sum(len(v) for v in bands.values()))
        take = {b: 0 for b in bands}
        while sum(take.values()) < want:  # one at a time to the band with the fewest picks that still has tasks
            open_bands = [b for b in sorted(bands) if take[b] < len(bands[b])]
            b = min(open_bands, key=lambda b: (take[b], b))
            take[b] += 1
        for b in sorted(bands):
            picked += bands[b][: take[b]]
    return picked


def cmd_select(args: argparse.Namespace) -> None:
    from .release import finalize_task_row, split_lookup

    manifest = json.loads(Path(args.split_manifest).read_text())
    lookup = split_lookup(manifest)
    rows = [finalize_task_row(r, "test") for r in _jsonl(Path(args.run) / "tasks.jsonl")
            if lookup.get(str(r["scene_id"])) == "test"]
    picked = select_tasks(rows, args.per_family, args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tasks.jsonl").write_text("".join(json.dumps(r) + "\n" for r in picked))
    cells = Counter((r["task_type"], band_name(int(r["difficulty"]["d_star"]))) for r in picked)
    avail = Counter((r["task_type"], band_name(int(r["difficulty"]["d_star"]))) for r in rows)
    selection = {"version": VERSION, "seed": args.seed, "per_family": args.per_family, "test_tasks": len(rows),
                 "picked": len(picked), "scenes": len({r["scene_id"] for r in picked}),
                 "cells": {f"{f}|{b}": {"picked": n, "available": avail[(f, b)]} for (f, b), n in sorted(cells.items())},
                 "task_ids": [r["task_id"] for r in picked]}
    (out / "selection.json").write_text(json.dumps(selection, indent=1))
    print(json.dumps({k: v for k, v in selection.items() if k != "task_ids"}, indent=1))


def replace_flagged(picked: Sequence[Dict[str, Any]], flagged: Sequence[str], pool: Sequence[Dict[str, Any]],
                    seed: int = SEED) -> Dict[str, Any]:
    """Each flagged task of ``picked`` is replaced by an unflagged task of ``pool`` (not yet picked) from the same
    family and d* band, else from the same family; a task with no substitute is dropped.  The others stay."""
    flagged = set(flagged)
    cell = lambda r: (r["task_type"], band_name(int(r["difficulty"]["d_star"])))  # noqa: E731
    used = {r["task_id"] for r in picked}
    cand: Dict[Any, List[Dict[str, Any]]] = defaultdict(list)
    for r in sorted(pool, key=lambda r: r["task_id"]):
        if r["task_id"] not in flagged and r["task_id"] not in used:
            cand[cell(r)].append(r)
    rng = random.Random(seed)
    for v in cand.values():
        rng.shuffle(v)
    out, swaps = [], []
    for r in picked:
        if r["task_id"] not in flagged:
            out.append(r)
            continue
        c = cell(r)
        src = cand[c] or next((cand[k] for k in sorted(cand) if k[0] == c[0] and cand[k]), [])
        new = src.pop() if src else None
        if new is not None:
            out.append(new)
        swaps.append({"out": r["task_id"], "in": new["task_id"] if new else None, "cell": list(c)})
    return {"tasks": out, "swaps": swaps}


def cmd_replace(args: argparse.Namespace) -> None:
    from .release import finalize_task_row, split_lookup

    picked = _jsonl(Path(args.eval) / "tasks.jsonl")
    flagged = json.loads(Path(args.flagged).read_text())["flagged_task_ids"]
    lookup = split_lookup(json.loads(Path(args.split_manifest).read_text()))
    pool = [finalize_task_row(r, "test") for r in _jsonl(Path(args.run) / "tasks.jsonl")
            if lookup.get(str(r["scene_id"])) == "test"]
    res = replace_flagged(picked, flagged, pool, args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "tasks.jsonl").write_text("".join(json.dumps(r) + "\n" for r in res["tasks"]))
    summary = {"version": VERSION, "seed": args.seed, "source": str(args.eval), "tasks": len(res["tasks"]),
               "flagged_in_source": len(res["swaps"]), "replaced": sum(s["in"] is not None for s in res["swaps"]),
               "dropped": sum(s["in"] is None for s in res["swaps"]), "swaps": res["swaps"],
               "task_ids": [r["task_id"] for r in res["tasks"]]}
    (out / "selection.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("swaps", "task_ids")}, indent=1))


# ---------------------------------------------------------------------------
# run (one card: the render service and vLLM)
# ---------------------------------------------------------------------------
def env_config(env_template: Path, tasks: Path, gs_root: str, prompt_format: Optional[str]) -> Dict[str, Any]:
    import os

    import yaml

    cfg = dict(yaml.safe_load(env_template.read_text())["envs"][0]["config"])
    cfg.update(jsonl_path=str(tasks.resolve()), gs_root=gs_root, client_url=os.environ["R1_RENDER_URL"],
               r1_legacy_success=False)
    if prompt_format:
        cfg["prompt_format"] = prompt_format
    return cfg


def gate_summary(metric: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The success conditions not met at a pose (``failed_gates``) and the 0..1 progress toward success."""
    if not metric:
        return {"failed_gates": None, "progress": None}
    progress = metric.get("continuous_gate_progress", metric.get("shaping_score"))
    return {"failed_gates": sorted(k for k, v in (metric.get("gates") or {}).items() if not v),
            "progress": None if progress is None else round(float(progress), 4)}


def _images(obs: Dict[str, Any]) -> List[Any]:
    return [im for v in (obs.get("multi_modal_data") or {}).values() for im in (v or [])]


def run_episode(env: Any, policy: Any, index: int, max_turns: int, frames: Optional[Path] = None) -> Dict[str, Any]:
    """One greedy episode; per-turn bookkeeping as in scripts/r1_run_canonical_dev_eval32.py.

    With ``frames`` (round 8) the row also carries the trajectory: per turn the frame the model saw (a JPEG in
    ``frames``, named <index>_<turn>.jpg), the user text, the answer, the parsed actions, blocked moves, reward
    and whether the task was met after the turn; then the frame after the last turn and the env's end reason."""
    obs, info = env.reset(seed=index)
    system = env.system_prompt()
    item = env.current_item or {}
    turns, invalid, attempted, collisions, success, actions_to_success = 0, 0, 0, 0, False, None
    done, trajectory, end_reason = False, [], None
    start = gate_summary(env._calculate_canonical_metric()) if frames is not None else None

    def keep(image: Any, turn: int) -> Optional[str]:
        if frames is None or image is None:
            return None
        frames.mkdir(parents=True, exist_ok=True)
        name = f"{index:03d}_{turn:02d}.jpg"
        image.convert("RGB").save(frames / name, format="JPEG", quality=85)
        return f"{frames.name}/{name}"

    while not done and turns < max_turns:
        images = _images(obs)
        if len(images) != 1:
            raise RuntimeError(f"expected one observation image, got {len(images)}")
        text = policy.generate(system, obs["obs_str"], images[0])
        seen = {"frame": keep(images[0], turns), "user": obs["obs_str"]}
        obs, reward, done, step_info = env.step(text)
        turns += 1
        parsed = list(step_info.get("actions") or [])
        tm = (step_info.get("metrics") or {}).get("turn_metrics") or {}
        attempted += len(parsed)
        blocked = int(tm.get("collision_count", 0) or 0)
        collisions += blocked
        valid = bool(tm.get("action_is_valid", bool(parsed)))
        invalid += int(not valid)
        metric = env._calculate_canonical_metric()
        met = bool(metric and metric.get("success"))
        if met and not success:
            success, actions_to_success = True, int(env._current_step)
        end_reason = step_info.get("termination_reason", end_reason)
        if frames is not None:
            trajectory.append(dict(seen, answer=text, actions=[str(a) for a in parsed], valid=valid,
                                   collisions=blocked, reward=round(float(reward), 4), met=met,
                                   **gate_summary(metric)))
    row = {"index": index, "task_id": item.get("task_id"), "scene_id": item.get("scene_id"),
           "task_type": item.get("task_type"), "d_star": int((item.get("difficulty") or {}).get("d_star", 0)),
           "success": success, "actions_to_success": actions_to_success, "turns": turns,
           "actions": attempted, "collisions": collisions, "invalid_turns": invalid}
    if frames is not None:
        last = _images(obs)
        row.update(task_description=item.get("task_description"), start=start, trajectory=trajectory,
                   final_frame=keep(last[0] if last else None, turns),
                   end_reason=end_reason if done else "max_turns")
    return row


class VllmPolicy:  # pragma: no cover - needs a GPU
    def __init__(self, model: str, gpu_memory_utilization: float = 0.6, max_model_len: int = 8192) -> None:
        from vllm import LLM

        self.llm = LLM(model=model, gpu_memory_utilization=gpu_memory_utilization, max_model_len=max_model_len,
                       enforce_eager=True, enable_prefix_caching=True, limit_mm_per_prompt={"image": 1},
                       trust_remote_code=False)

    def generate(self, system: str, user_text: str, image: Any) -> str:
        import base64
        import io

        from vllm import SamplingParams

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        url = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
        before, _, after = user_text.partition("<image>")
        content = ([{"type": "text", "text": before}] if before else []) + [{"type": "image_url", "image_url": {"url": url}}]
        content += [{"type": "text", "text": after}] if after else []
        params = SamplingParams(temperature=0.0, max_tokens=384, stop=["</action>"], include_stop_str_in_output=True)
        out = self.llm.chat([{"role": "system", "content": system}, {"role": "user", "content": content}],
                            sampling_params=params, use_tqdm=False)
        return out[0].outputs[0].text


def cmd_run(args: argparse.Namespace) -> None:  # pragma: no cover - needs a GPU and the render service
    from vagen.envs.active_spatial.env import ActiveSpatialEnv
    from vagen.envs.active_spatial.env_config import ActiveSpatialEnvConfig

    ev = Path(args.eval)
    out = ev / args.name
    out.mkdir(parents=True, exist_ok=True)
    tasks = ev / "tasks.jsonl"
    n = len(_jsonl(tasks))
    cfg = env_config(Path(args.env_template), tasks, args.gs_root, args.prompt_format)
    keys = set(ActiveSpatialEnvConfig.__dataclass_fields__)
    env = ActiveSpatialEnv(ActiveSpatialEnvConfig(**{k: v for k, v in cfg.items() if k in keys}))
    t0 = time.time()
    policy = VllmPolicy(args.model)
    (out / "model.json").write_text(json.dumps({"name": args.name, "model": args.model, "version": VERSION,
                                                "prompt_format": cfg["prompt_format"], "greedy": True,
                                                "load_seconds": round(time.time() - t0, 1)}, indent=1))
    done = {r["index"] for r in _jsonl(out / "results.jsonl")} if (out / "results.jsonl").exists() else set()
    with open(out / "results.jsonl", "a") as f:
        for i in range(n):
            if i in done:
                continue
            try:
                row = run_episode(env, policy, i, int(args.max_turns), frames=out / "frames")
            except Exception as exc:  # an infrastructure error is recorded, not counted as a failure
                row = {"index": i, "error": f"{type(exc).__name__}: {exc}"}
            f.write(json.dumps(row) + "\n")
            f.flush()
    (out / "done.json").write_text(json.dumps({"episodes": n, "seconds": round(time.time() - t0, 1)}))


# ---------------------------------------------------------------------------
# table
# ---------------------------------------------------------------------------
def metrics(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    ok = [r for r in rows if "error" not in r]
    succ = [r for r in ok if r["success"]]
    rate = lambda rs: round(sum(r["success"] for r in rs) / len(rs), 4) if rs else None  # noqa: E731
    by_family = defaultdict(list)
    by_band = defaultdict(list)
    for r in ok:
        by_family[r["task_type"]].append(r)
        by_band[band_name(r["d_star"])].append(r)
    actions = sum(r["actions"] for r in ok)
    turns = sum(r["turns"] for r in ok)
    return {
        "episodes": len(ok), "errors": len(rows) - len(ok), "success": rate(ok),
        "success_by_family": {k: rate(v) for k, v in sorted(by_family.items())},
        "success_by_band": {k: rate(v) for k, v in sorted(by_band.items())},
        "actions_over_dstar": (round(sum(r["actions_to_success"] / r["d_star"] for r in succ) / len(succ), 3)
                               if succ else None),
        "collision_rate": round(sum(r["collisions"] for r in ok) / actions, 4) if actions else None,
        "invalid_rate": round(sum(r["invalid_turns"] for r in ok) / turns, 4) if turns else None,
    }


def cmd_table(args: argparse.Namespace) -> None:
    ev = Path(args.eval)
    models = {}
    for d in sorted(p for p in ev.iterdir() if (p / "results.jsonl").exists()):
        models[d.name] = metrics(_jsonl(d / "results.jsonl"))
    (ev / "table.json").write_text(json.dumps(models, indent=1))
    families = sorted({f for m in models.values() for f in m["success_by_family"]})
    bands = [f"{lo}-{hi}" for lo, hi in BANDS]
    head = ["模型", "题数", "成功率"] + families + [f"d* {b}" for b in bands] + ["步数/d*", "碰撞率", "不合法率"]
    lines = ["| " + " | ".join(head) + " |", "| " + " | ".join("---" for _ in head) + " |"]
    fmt = lambda v: "—" if v is None else f"{v:.3f}"  # noqa: E731
    for name, m in models.items():
        cells = [name, str(m["episodes"]), fmt(m["success"])] + [fmt(m["success_by_family"].get(f)) for f in families]
        cells += [fmt(m["success_by_band"].get(b)) for b in bands]
        cells += [fmt(m["actions_over_dstar"]), fmt(m["collision_rate"]), fmt(m["invalid_rate"])]
        lines.append("| " + " | ".join(cells) + " |")
    (ev / "table.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--run", required=True)
    s.add_argument("--split-manifest", required=True)
    s.add_argument("--per-family", type=int, default=PER_FAMILY)
    s.add_argument("--seed", type=int, default=SEED)
    s.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--eval", required=True)
    r.add_argument("--name", required=True)
    r.add_argument("--model", required=True, help="a Hugging Face model folder (an RL checkpoint's actor/huggingface)")
    r.add_argument("--gs-root", default="/mnt/umm/users/yinbaiqiao/InteriorGS_ply")
    r.add_argument("--env-template", default=str(Path(__file__).with_name("configs") / "rl_env_v10_S0.yaml"))
    r.add_argument("--prompt-format", default=None, help="the model's own prompt format (R1: its own)")
    r.add_argument("--max-turns", type=int, default=12)
    t = sub.add_parser("table")
    t.add_argument("--eval", required=True)
    x = sub.add_parser("replace")
    x.add_argument("--eval", required=True, help="the evaluation folder to start from (tasks.jsonl)")
    x.add_argument("--flagged", required=True, help="depth_check_summary.json (flagged_task_ids)")
    x.add_argument("--run", required=True)
    x.add_argument("--split-manifest", required=True)
    x.add_argument("--seed", type=int, default=SEED)
    x.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    {"select": cmd_select, "run": cmd_run, "table": cmd_table, "replace": cmd_replace}[args.cmd](args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
