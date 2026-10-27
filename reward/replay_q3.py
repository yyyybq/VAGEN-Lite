"""Reward line: replay a Qwen3 run's training records in the env without rendering, the main line's way
(R12/scripts/reward_replay_r13o.py: the recorded answers, a fake render), with the run's own env config read from
its train_env.yaml.

For every recorded episode of the given tasks (every EVERY-th update):

* coverage: every pose the env scores (each call of _calculate_canonical_metric: the start and each primitive
  action, which is where the optimal progress looks its table up) looked up in TABLES, and in OLD when given: in
  the table or not, the table's d, and whether d is guaranteed exact (d <= depth - the pose's depth from the start);
  poses outside OLD go to OUT/outside_old.jsonl (for the re-search check of step 2a);
* returns (unless --no-returns): the return under O (the run's env + progress_version optimal + TABLES, slack 4,
  the run's own progress coefficient) as the logged return + (replay under O - replay under the run's own config),
  and the check that the own replay equals the logged return; from the all-fail groups, the keep threshold the main
  line's way (sample standard deviation, n - 1, of a group's 8 returns; 80th percentile; floor 1.0), under O and
  under the run's own reward.

    python -m reward.replay_q3 --run RUN_DIR --env-yaml TRAIN_ENV.yaml --rows ROWS --task-ids FILE --tables DIR
        --out OUT [--old-tables DIR] [--every 1] [--workers 12] [--no-returns]

OUT/replays.jsonl (one line per update and task), OUT/summary.json.
"""

import argparse
import json
import re
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

DEPTH = 12


def answers_of(output):
    out = []
    for t in re.split(r"<\|im_start\|>assistant\n", output):
        a = t.split("<|im_end|>")[0].strip()
        if a and not a.startswith("[Observation]"):
            out.append(a)
    return out


def _lookup(table, keys):
    """(found, d, depth_of) arrays for contract keys in a loaded table."""
    tk = table["keys"]
    i = np.searchsorted(tk, keys)
    i = np.minimum(i, len(tk) - 1)
    found = tk[i] == keys
    return found, np.where(found, table["dist"][i], -2), np.where(found, table["depth_of"][i], -1)


def worker(args):
    chunk_path, env_yaml, tables, old_tables, do_returns = args
    import yaml

    from data_gen.interiorgs_v10 import distance_table as DT
    from data_gen.interiorgs_v10.difficulty import _split, contract_keys
    from vagen.envs.active_spatial.env import ActiveSpatialEnv
    from vagen.envs.active_spatial.env_config import ActiveSpatialEnvConfig

    groups = [json.loads(l) for l in Path(chunk_path).read_text().splitlines() if l.strip()]
    rows_path = Path(chunk_path).with_suffix(".rows.jsonl")
    rows_path.write_text("".join(json.dumps(g["row"]) + "\n" for g in groups))
    base = dict(yaml.safe_load(Path(env_yaml).read_text())["envs"][0]["config"])
    base.update(jsonl_path=str(rows_path), render_backend=None)
    fields = set(ActiveSpatialEnvConfig.__dataclass_fields__)
    cfgs = {"O": {**base, "progress_version": "optimal", "distance_tables_dir": tables}}
    if do_returns:
        cfgs["own"] = base
    envs = {}
    for name, cfg in cfgs.items():
        env = ActiveSpatialEnv(ActiveSpatialEnvConfig(**{k: v for k, v in cfg.items() if k in fields}))

        def fake_render(env=env, **_kwargs):
            env.last_image_std = 30.0
            env.last_low_info_frame = False
            env.consecutive_low_info_frame_count = 0
            return {"obs_str": "", "multi_modal_data": {}}

        env._render = fake_render
        envs[name] = env
    seen = []
    env_o = envs["O"]
    orig = env_o._calculate_canonical_metric

    def recorded():
        m = orig()
        if m is not None:
            seen.append(np.asarray(env_o.view_engine.get_pose(), float).copy())
        return m

    env_o._calculate_canonical_metric = recorded
    out, outside = [], []
    for gi, g in enumerate(groups):
        tid = g["row"]["task_id"]
        try:
            new = DT.load(DT.table_path(Path(tables), tid))
            old = DT.load(DT.table_path(Path(old_tables), tid)) if old_tables and \
                DT.table_path(Path(old_tables), tid).is_file() else None
            rets = {name: [] for name in envs}
            cov = defaultdict(int)
            missing = []
            for ep in g["episodes"]:
                for name, env in envs.items():
                    if name == "O":
                        seen.clear()
                    env.reset(seed=gi)
                    assert env.current_item["task_id"] == tid
                    total = 0.0
                    for a in ep["answers"]:
                        _, r, done, _ = env.step(a)
                        total += float(r)
                        if done:
                            break
                    rets[name].append(total)
                poses = np.stack(seen) if seen else np.zeros((0, 4, 4))
                keys = np.asarray([int(contract_keys(*_split(p))[0]) for p in poses], dtype=np.int64)
                found, d, dep = _lookup(new, keys)
                cov["lookups"] += len(keys)
                cov["distinct"] += len(set(keys.tolist()))
                cov["in_new"] += int(found.sum())
                cov["unreached_new"] += int((found & (d == -1)).sum())
                cov["not_guaranteed_exact"] += int((found & (d >= 0) & (d > DEPTH - dep)).sum())
                for p, k, f in zip(poses, keys, found):
                    if not f and len(missing) < 3:
                        missing.append(p.tolist())
                if old is not None:
                    f_old, _, _ = _lookup(old, keys)
                    cov["in_old"] += int(f_old.sum())
                    for p, k, f, dd, de in zip(poses, keys, f_old, d, dep):
                        if not f:
                            outside.append({"task_id": tid, "key": int(k), "pose": p.tolist(), "d_new": int(dd),
                                            "depth_of_new": int(de)})
            rec = {"update": g["update"], "task_id": tid, "task_type": g["row"]["task_type"],
                   "d_star": int(g["row"]["difficulty"]["d_star"]), "success": [e["success"] for e in g["episodes"]],
                   "logged": [e["score"] for e in g["episodes"]], "coverage": dict(cov), "missing_examples": missing}
            if do_returns:
                logged = rec["logged"]
                rec["check"] = [abs(l - o) < 1e-6 for l, o in zip(logged, rets["own"])]
                rec["returns_O"] = [l + (x - o) for l, x, o in zip(logged, rets["O"], rets["own"])]
            out.append(rec)
        except Exception as exc:
            out.append({"update": g["update"], "task_id": tid, "error": f"{type(exc).__name__}: {exc}"})
    return out, outside


def keep_threshold(groups, key):
    """The main line's keep calibration over the all-fail groups: 80th percentile of the 8 returns' sample std."""
    stds = [float(np.std(g[key], ddof=1)) for g in groups if not any(g["success"]) and len(g[key]) > 1]
    if not stds:
        return None
    return {"all_fail_groups": len(stds), "all_fail_tasks": len({g["task_id"] for g in groups if not any(g["success"])}),
            "p50": round(float(np.percentile(stds, 50)), 4), "p80": round(float(np.percentile(stds, 80)), 4),
            "p90": round(float(np.percentile(stds, 90)), 4), "threshold": round(float(np.percentile(stds, 80)), 4),
            "floor": 1.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--env-yaml", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--task-ids", required=True)
    ap.add_argument("--tables", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--old-tables", default=None)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--no-returns", action="store_true")
    args = ap.parse_args()

    ids = {l.strip() for l in Path(args.task_ids).read_text().splitlines() if l.strip()}
    rows = {}
    for l in Path(args.rows).read_text().splitlines():
        if l.strip():
            r = json.loads(l)
            if r["task_id"] in ids:
                rows[r["task_id"]] = r
    files = sorted(Path(args.run, "rollout_data").glob("*.jsonl"), key=lambda p: int(p.stem))[::args.every]
    groups = []
    for f in files:
        by = defaultdict(list)
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if r.get("task_id") in rows:
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
        chunks.append((str(p), args.env_yaml, args.tables, args.old_tables, not args.no_returns))
    res, outside = [], []
    with Pool(args.workers) as pool:
        for part, out_part in pool.imap_unordered(worker, chunks):
            res += part
            outside += out_part
    ok = [x for x in res if "error" not in x]
    with open(out / "replays.jsonl", "w") as f:
        for x in sorted(ok, key=lambda x: (x["update"], x["task_id"])):
            f.write(json.dumps(x) + "\n")
    with open(out / "outside_old.jsonl", "w") as f:
        for x in outside:
            f.write(json.dumps(x) + "\n")
    cov = defaultdict(int)
    by_task = defaultdict(lambda: defaultdict(int))
    for x in ok:
        for k, v in x["coverage"].items():
            cov[k] += v
            by_task[x["task_id"]][k] += v
    summary = {"run": args.run, "env_yaml": args.env_yaml, "tables": args.tables, "old_tables": args.old_tables,
               "updates": len(files), "every": args.every, "tasks_asked": len(ids), "tasks_seen": len(by_task),
               "groups": len(groups), "episodes": sum(len(g["episodes"]) for g in groups), "errors": len(res) - len(ok),
               "error_examples": [x["error"] for x in res if "error" in x][:5], "coverage": dict(cov),
               "coverage_by_task": {t: dict(v) for t, v in sorted(by_task.items())},
               "outside_old_distinct": len({(x["task_id"], x["key"]) for x in outside})}
    if not args.no_returns:
        checks = [c for x in ok for c in x["check"]]
        summary["own_replay_matches_logged"] = [sum(checks), len(checks)]
        summary["keep_under_O"] = keep_threshold(ok, "returns_O")
        summary["keep_under_own"] = keep_threshold(ok, "logged")
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "coverage_by_task"}, indent=1))


if __name__ == "__main__":
    main()
