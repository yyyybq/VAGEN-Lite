"""Reward line (the user's item 6 of 2026-11-02): how much the turn that ends an episode inside its action loop misses
of the pose reward, on a run's training records, before the main line is told.

ActiveSpatialEnv.step computes the pose reward (the potential term scale x (gamma phi' - phi), visibility, near-success,
milestones; and the update of prev_potential_score) only ``if not done`` after the action loop.  A turn whose loop sets
done itself (a success seen after an action with success_check_each_action, the primitive-step budget used up, the
consecutive-collision stop) gets none of it, and its info["current_potential_score"] is the previous turn's phi.

Every recorded episode (every EVERY-th update) is replayed without rendering in the run's own env config (its
train_env.yaml), the recorded answers one env.step each; the replayed return is checked against the logged one.  On
such an ending turn the env's own _calculate_pose_reward is then called once (the trace switched off): what the turn
would have received had the loop not ended the episode, with phi before (the stale value) and after.  The corrected
return is the logged return + that amount.  Compared, logged against corrected: the returns by how the episode ended;
the group comparison (verl's GRPO, std n - 1), by group kind (mixed groups are the ones trained on; all-fail ones only
when keep fills them; all-success ones are dropped): the share of episodes whose advantage changes sign and the mean
change;
mixed groups where a failure's return reaches a success's; the all-fail groups whose return spread reaches the keep
threshold (vagen/custom_filter/filter.py keep_by_return_std_filter).

    python -m reward.terminal_turn_reward --run RL_RUN_DIR --env-yaml TRAIN_ENV.yaml --rows ROWS --out OUT
        [--every 4] [--max-update N] [--keep-threshold 0.3318] [--workers 4]

OUT/episodes.jsonl (one line per group), OUT/summary.json.  CPU only; the login node at no more than 4 cores.
"""

import argparse
import json
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from reward.replay_q3 import answers_of

SKIPPED = ("success_mid_turn", "budget_success", "budget_fail", "collision_stop")
EPS = 1e-6


def end_reason(info):
    if info.get("success_mid_turn"):
        return "success_mid_turn"
    if info.get("truncated_max_primitive_steps"):
        return "budget_success" if info.get("success") else "budget_fail"
    if info.get("early_terminated_collision"):
        return "collision_stop"
    if info.get("auto_terminated"):
        return "success_after_turn"
    if info.get("early_terminated_invalid"):
        return "invalid_stop"
    if info.get("early_terminated_low_info"):
        return "low_info_stop"
    return "other_done"


def worker(args):
    chunk_path, env_yaml = args
    import yaml

    from vagen.envs.active_spatial.env import ActiveSpatialEnv
    from vagen.envs.active_spatial.env_config import ActiveSpatialEnvConfig

    groups = [json.loads(l) for l in Path(chunk_path).read_text().splitlines() if l.strip()]
    rows_path = Path(chunk_path).with_suffix(".rows.jsonl")
    rows_path.write_text("".join(json.dumps(g["row"]) + "\n" for g in groups))
    cfg = dict(yaml.safe_load(Path(env_yaml).read_text())["envs"][0]["config"])
    cfg.update(jsonl_path=str(rows_path), render_backend=None)
    env = ActiveSpatialEnv(ActiveSpatialEnvConfig(**{k: v for k, v in cfg.items()
                                                     if k in ActiveSpatialEnvConfig.__dataclass_fields__}))

    def fake_render(**_kwargs):
        env.last_image_std = 30.0
        env.last_low_info_frame = False
        env.consecutive_low_info_frame_count = 0
        return {"obs_str": "", "multi_modal_data": {}}

    env._render = fake_render
    out = []
    for gi, g in enumerate(groups):
        tid = g["row"]["task_id"]
        try:
            eps = []
            for ep in g["episodes"]:
                env.reset(seed=gi)
                assert env.current_item["task_id"] == tid
                total, end, turns = 0.0, "not_done", 0
                for a in ep["answers"]:
                    _, r, done, info = env.step(a)
                    total += float(r)
                    turns += 1
                    if done:
                        end = end_reason(info)
                        break
                rec = {"logged": ep["score"], "replayed": round(total, 6), "success": ep["success"], "turns": turns,
                       "end": end, "missing": 0.0}
                if end in SKIPPED:
                    rec["phi_before"] = float(env.prev_potential_score)
                    env._reward_trace = None
                    rec["missing"] = float(env._calculate_pose_reward())
                    rec["phi_after"] = float(env.prev_potential_score)
                eps.append(rec)
            out.append({"update": g["update"], "task_id": tid, "task_type": g["row"]["task_type"], "episodes": eps})
        except Exception as exc:
            out.append({"update": g["update"], "task_id": tid, "error": f"{type(exc).__name__}: {exc}"})
    return out


def advantages(r):
    r = np.asarray(r, float)
    return (r - r.mean()) / (r.std(ddof=1) + EPS) if len(r) > 1 else np.zeros_like(r)


def summarize(groups, keep_threshold):
    eps = [e for g in groups for e in g["episodes"]]
    q = lambda v, p: round(float(np.quantile(v, p)), 4) if len(v) else None  # noqa: E731
    by_end = defaultdict(list)
    for e in eps:
        by_end[e["end"]].append(e)
    ends = {}
    for k, v in sorted(by_end.items()):
        m = [e["missing"] for e in v]
        ends[k] = {"episodes": len(v), "share": round(len(v) / len(eps), 4), "missing_mean": round(float(np.mean(m)), 4),
                   "missing_abs_mean": round(float(np.mean(np.abs(m))), 4), "missing_p10": q(m, 0.1),
                   "missing_p90": q(m, 0.9)}
        if k in SKIPPED:
            ends[k]["phi_before_mean"] = round(float(np.mean([e["phi_before"] for e in v])), 4)
            ends[k]["phi_after_mean"] = round(float(np.mean([e["phi_after"] for e in v])), 4)
    adv = defaultdict(lambda: {"groups": 0, "episodes_with_nonzero_adv": 0, "sign_flips": 0, "abs_change": []})
    inv_logged = inv_fixed = mixed = 0
    keep = {"all_fail_groups": 0, "above_logged": 0, "above_fixed": 0, "gained": 0, "lost": 0}
    spread_logged, spread_fixed, group_std = [], [], []
    for g in groups:
        lg = [e["logged"] for e in g["episodes"]]
        fx = [e["logged"] + e["missing"] for e in g["episodes"]]
        ok = [e["success"] for e in g["episodes"]]
        if len(lg) > 1:
            group_std.append(float(np.std(lg, ddof=1)))
        kind = "mixed" if any(ok) and not all(ok) else ("all_success" if all(ok) else "all_fail")
        a, b = advantages(lg), advantages(fx)
        st = adv[kind]
        st["groups"] += 1
        for x, y in zip(a, b):
            if abs(x) > EPS:
                st["episodes_with_nonzero_adv"] += 1
                st["sign_flips"] += int(np.sign(x) != np.sign(y))
            st["abs_change"].append(abs(x - y))
        if any(ok) and not all(ok):
            mixed += 1
            inv_logged += int(max(r for r, s in zip(lg, ok) if not s) >= min(r for r, s in zip(lg, ok) if s))
            inv_fixed += int(max(r for r, s in zip(fx, ok) if not s) >= min(r for r, s in zip(fx, ok) if s))
        if not any(ok) and len(lg) > 1:
            sl, sf = float(np.std(lg, ddof=1)), float(np.std(fx, ddof=1))
            spread_logged.append(sl)
            spread_fixed.append(sf)
            keep["all_fail_groups"] += 1
            keep["above_logged"] += int(sl >= keep_threshold)
            keep["above_fixed"] += int(sf >= keep_threshold)
            keep["gained"] += int(sf >= keep_threshold > sl)
            keep["lost"] += int(sl >= keep_threshold > sf)
    keep["threshold"] = keep_threshold
    keep["spread_mean_logged"] = round(float(np.mean(spread_logged)), 4) if spread_logged else None
    keep["spread_mean_fixed"] = round(float(np.mean(spread_fixed)), 4) if spread_fixed else None
    succ = [e for e in eps if e["success"]]
    fail = [e for e in eps if not e["success"]]
    ret = lambda v, fixed: round(float(np.mean([e["logged"] + (e["missing"] if fixed else 0.0) for e in v])), 4) if v else None  # noqa: E731
    return {"groups": len(groups), "episodes": len(eps),
            "replay_matches_logged": [sum(abs(e["replayed"] - e["logged"]) < 1e-4 for e in eps), len(eps)],
            "affected": sum(e["end"] in SKIPPED for e in eps), "by_end": ends,
            "return_mean": {"success_logged": ret(succ, False), "success_fixed": ret(succ, True),
                            "fail_logged": ret(fail, False), "fail_fixed": ret(fail, True)},
            "group_return_std_mean": round(float(np.mean(group_std)), 4) if group_std else None,
            "advantage_by_group_kind": {k: {"groups": v["groups"],
                                            "episodes_with_nonzero_adv": v["episodes_with_nonzero_adv"],
                                            "sign_flips": v["sign_flips"],
                                            "sign_flip_share": (round(v["sign_flips"] / v["episodes_with_nonzero_adv"], 4)
                                                                if v["episodes_with_nonzero_adv"] else None),
                                            "abs_change_mean": round(float(np.mean(v["abs_change"])), 4),
                                            "abs_change_p90": q(v["abs_change"], 0.9)}
                                        for k, v in sorted(adv.items())},
            "mixed_groups": mixed, "mixed_fail_reaches_success_logged": inv_logged,
            "mixed_fail_reaches_success_fixed": inv_fixed, "keep": keep}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="the run's rl_run_<name> folder (rollout_data/)")
    ap.add_argument("--env-yaml", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--max-update", type=int, default=None)
    ap.add_argument("--keep-threshold", type=float, default=0.3318)  # V4's (rw_g, q3d)
    ap.add_argument("--workers", type=int, default=4)  # the login node: at most 4 cores a task
    args = ap.parse_args()
    rows = {}
    for l in Path(args.rows).read_text().splitlines():
        if l.strip():
            r = json.loads(l)
            rows[r["task_id"]] = r
    files = sorted(Path(args.run, "rollout_data").glob("*.jsonl"), key=lambda p: int(p.stem))
    if args.max_update:
        files = [f for f in files if int(f.stem) <= args.max_update]
    files = files[::args.every]
    groups = []
    for f in files:
        by = defaultdict(list)
        for line in f.read_text().splitlines():
            r = json.loads(line)
            by[r["task_id"]].append(r)
        for tid, eps in sorted(by.items()):
            groups.append({"update": int(f.stem), "row": rows[tid],
                           "episodes": [{"answers": answers_of(e["output"]), "score": float(e["score"]),
                                         "success": bool(float(e.get("traj_success") or 0))} for e in eps]})
    out = Path(args.out)
    work = out / "work"
    work.mkdir(parents=True, exist_ok=True)
    chunks = []
    for k in range(0, len(groups), 20):
        p = work / f"chunk_{k:06d}.jsonl"
        p.write_text("".join(json.dumps(g) + "\n" for g in groups[k:k + 20]))
        chunks.append((str(p), args.env_yaml))
    res = []
    with Pool(args.workers) as pool:
        for part in pool.imap_unordered(worker, chunks):
            res += part
    ok = sorted((x for x in res if "error" not in x), key=lambda x: (x["update"], x["task_id"]))
    with open(out / "episodes.jsonl", "w") as f:
        for x in ok:
            f.write(json.dumps(x) + "\n")
    summary = {"run": args.run, "env_yaml": args.env_yaml, "updates": len(files), "every": args.every,
               "errors": len(res) - len(ok), "error_examples": [x["error"] for x in res if "error" in x][:5],
               **summarize(ok, args.keep_threshold)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
