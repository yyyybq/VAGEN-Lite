"""Reward line, phase 2 direction 3: statement accuracy of evaluation results, and the env's statement scoring check.
CPU only, no rendering.

score:     each result folder (results.jsonl, model.json) is replayed in the evaluation's env (as reward/fail_end_steps.py,
           the result's prompt format and env switches) with statement_record on; the env scores every recorded
           answer's statement against the frame the model saw (vagen/statement_truth.py).  Per result: readable share,
           accuracy and per-field accuracy, by family and by turn (1, 2, 3, 4+), and the replay check (task met and
           blocked moves as recorded).
selfcheck: the recorded actions of a result without statements (e.g. C0 e8) are given a statement built here from the
           truth at the env's pose before the step (the frame the model saw), once exact and once with "dist" changed;
           the env must score them 1.0 and 0.75 on every turn.

    python -m reward.statement_eval score --result DIR [--result DIR ...] --out FILE.json [--tasks FILE]
        [--env-template YAML] [--gs-root DIR] [--workers 4]
    python -m reward.statement_eval selfcheck --result DIR --out FILE.json [--limit N] [--workers 4]
"""

import argparse
import json
import re
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

TEMPLATE = Path(__file__).resolve().parents[1] / "data_gen" / "interiorgs_v10" / "configs" / "rl_env_v10_S0.yaml"
GS_ROOT = "/mnt/umm/users/yinbaiqiao/InteriorGS_ply"
ACTION = re.compile(r"<action>.*?</action>", re.S)
OTHER_DIST = {"near": "far", "mid": "near", "far": "mid"}


def make_env(tasks_path, template, gs_root, model_json, prompt_format=None):
    import yaml

    from vagen.envs.active_spatial.env import ActiveSpatialEnv
    from vagen.envs.active_spatial.env_config import ActiveSpatialEnvConfig

    model = json.loads(Path(model_json).read_text())
    cfg = dict(yaml.safe_load(Path(template).read_text())["envs"][0]["config"])
    cfg.update(jsonl_path=str(tasks_path), gs_root=gs_root, render_backend=None, r1_legacy_success=False)
    if model.get("prompt_format"):
        cfg["prompt_format"] = model["prompt_format"]
    cfg.update(model.get("env_switches") or {})
    if prompt_format:
        cfg["prompt_format"] = prompt_format
    cfg["statement_record"] = True
    env = ActiveSpatialEnv(ActiveSpatialEnvConfig(**{k: v for k, v in cfg.items()
                                                     if k in ActiveSpatialEnvConfig.__dataclass_fields__}))

    def fake_render(**_kwargs):
        env.last_image_std = 30.0
        env.last_low_info_frame = False
        env.consecutive_low_info_frame_count = 0
        return {"obs_str": "", "multi_modal_data": {}}

    env._render = fake_render
    position = {}
    for i, line in enumerate(l for l in Path(tasks_path).read_text().splitlines() if l.strip()):
        position.setdefault(json.loads(line)["task_id"], i)
    return env, position


def worker(args):
    mode, tasks_path, template, gs_root, model_json, items = args
    from vagen.statement_truth import format_statement, truth

    env, position = make_env(tasks_path, template, gs_root, model_json, "observe" if mode == "selfcheck" else None)
    out = []
    for r in items:
        variants = ("exact", "wrong_dist") if mode == "selfcheck" else ("recorded",)
        for variant in variants:
            try:
                env.reset(seed=position[r["task_id"]])
                assert env.current_item["task_id"] == r["task_id"]
                turns, same = [], True
                for step in r["trajectory"]:
                    answer = step["answer"]
                    if mode == "selfcheck":
                        t = truth(env.current_item, env.view_engine.get_pose(), env._visibility_occluders())
                        if variant == "wrong_dist":
                            t = dict(t, dist=OTHER_DIST[t["dist"]])
                        answer = format_statement(t) + "".join(ACTION.findall(answer))
                    _, _, done, info = env.step(answer)
                    tm = (info.get("metrics") or {}).get("turn_metrics") or {}
                    same &= bool(info.get("success")) == bool(step.get("met")) and \
                        int(tm.get("collision_count", 0) or 0) == int(step.get("collisions", 0) or 0)
                    s = info["statement"]
                    turns.append({k: s[k] for k in ("readable", "accuracy", "where", "face", "dist", "missing")})
                    if done:
                        break
                out.append({"task_id": r["task_id"], "task_type": r["task_type"], "success": bool(r["success"]),
                            "variant": variant, "replay_same": bool(same), "turns": turns})
            except Exception as exc:
                out.append({"task_id": r["task_id"], "variant": variant, "error": f"{type(exc).__name__}: {exc}"})
    return out


def summarize(rows):
    ok = [x for x in rows if "error" not in x]
    turns = [t for x in ok for t in x["turns"]]

    def stats(ts):
        if not ts:
            return None
        return {"turns": len(ts), "readable": round(float(np.mean([t["readable"] for t in ts])), 4),
                "accuracy": round(float(np.mean([t["accuracy"] for t in ts])), 4),
                **{k: round(float(np.mean([t[k] for t in ts])), 4) for k in ("where", "face", "dist", "missing")}}

    by_family = defaultdict(list)
    by_turn = defaultdict(list)
    for x in ok:
        for i, t in enumerate(x["turns"]):
            by_family[x["task_type"]].append(t)
            by_turn[str(i + 1) if i < 3 else "4+"].append(t)
    return {"episodes": len(rows), "errors": len(rows) - len(ok), "replay_same": sum(x["replay_same"] for x in ok),
            "all": stats(turns), "by_family": {k: stats(v) for k, v in sorted(by_family.items())},
            "by_turn": {k: stats(v) for k, v in sorted(by_turn.items())},
            "successful_episodes": stats([t for x in ok if x["success"] for t in x["turns"]]),
            "failed_episodes": stats([t for x in ok if not x["success"] for t in x["turns"]])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("score", "selfcheck"))
    ap.add_argument("--result", action="append", required=True)
    ap.add_argument("--tasks", default=None)
    ap.add_argument("--env-template", default=str(TEMPLATE))
    ap.add_argument("--gs-root", default=GS_ROOT)
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)  # the login node: at most 4 cores a task
    args = ap.parse_args()
    summary = {}
    for res in args.result:
        result = Path(res)
        tasks = Path(args.tasks) if args.tasks else result.parent / "tasks.jsonl"
        rows = [json.loads(l) for l in (result / "results.jsonl").read_text().splitlines() if l.strip()]
        rows = [r for r in rows if "error" not in r and r.get("trajectory")]
        if args.limit:
            rows = rows[:args.limit]
        chunks = [(args.mode, str(tasks), args.env_template, args.gs_root, str(result / "model.json"), rows[k:k + 10])
                  for k in range(0, len(rows), 10)]
        res_rows = []
        with Pool(args.workers) as pool:
            for part in pool.imap_unordered(worker, chunks):
                res_rows += part
        if args.mode == "selfcheck":
            summary[str(result)] = {v: summarize([x for x in res_rows if x.get("variant") == v])
                                    for v in ("exact", "wrong_dist")}
        else:
            summary[str(result)] = summarize(res_rows)
    Path(args.out).write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
