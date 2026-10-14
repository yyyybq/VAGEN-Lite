"""InteriorGS v10 evaluation (plan of 2026-10-14, round 6; run after training).

A fixed subset of the test split's formal tasks, the same env as RL (default v10 success rules,
``r1_legacy_success`` off), one greedy episode per task and model, on one card (render service and vLLM
share it):

    python -m data_gen.interiorgs_v10.evaluation select --run RUN/output --split-manifest RUN/split_manifest.json \\
        --out EVAL                                      # EVAL/tasks.jsonl + EVAL/selection.json
    # per model, a one-card sco job (sco_interiorgs_v10_train.sh eval RUN EVAL NAME MODEL [PROMPT_FORMAT]):
    python -m data_gen.interiorgs_v10.evaluation run --eval EVAL --name NAME --model HF_DIR
    python -m data_gen.interiorgs_v10.evaluation table --eval EVAL     # EVAL/table.md, EVAL/table.json

Subset: ``PER_FAMILY`` tasks per family (all of a family's test tasks when it has fewer), spread evenly over
the d* bands 1-3 / 4-6 / 7-8 (a band short of tasks gives its share to the others), drawn with ``SEED``.
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
VERSION = "interiorgs_v10_eval_v1"


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


def run_episode(env: Any, policy: Any, index: int, max_turns: int) -> Dict[str, Any]:
    """One greedy episode; per-turn bookkeeping as in scripts/r1_run_canonical_dev_eval32.py."""
    obs, info = env.reset(seed=index)
    system = env.system_prompt()
    item = env.current_item or {}
    turns, invalid, attempted, collisions, success, actions_to_success = 0, 0, 0, 0, False, None
    done = False
    while not done and turns < max_turns:
        images = [im for v in (obs.get("multi_modal_data") or {}).values() for im in (v or [])]
        if len(images) != 1:
            raise RuntimeError(f"expected one observation image, got {len(images)}")
        text = policy.generate(system, obs["obs_str"], images[0])
        obs, reward, done, step_info = env.step(text)
        turns += 1
        parsed = list(step_info.get("actions") or [])
        tm = (step_info.get("metrics") or {}).get("turn_metrics") or {}
        attempted += len(parsed)
        collisions += int(tm.get("collision_count", 0) or 0)
        invalid += int(not bool(tm.get("action_is_valid", bool(parsed))))
        metric = env._calculate_canonical_metric()
        if metric and metric.get("success") and not success:
            success, actions_to_success = True, int(env._current_step)
    return {"index": index, "task_id": item.get("task_id"), "scene_id": item.get("scene_id"),
            "task_type": item.get("task_type"), "d_star": int((item.get("difficulty") or {}).get("d_star", 0)),
            "success": success, "actions_to_success": actions_to_success, "turns": turns,
            "actions": attempted, "collisions": collisions, "invalid_turns": invalid}


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
                row = run_episode(env, policy, i, int(args.max_turns))
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
    (ev / "table.md").write_text("\n".join(lines) + "\n")
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
    args = ap.parse_args(argv)
    {"select": cmd_select, "run": cmd_run, "table": cmd_table}[args.cmd](args)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
