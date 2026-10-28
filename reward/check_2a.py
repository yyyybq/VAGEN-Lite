"""Reward line, step 2a: the depth-12 tables of the ten check tasks against everything the main line has.

1. Against the old tables (depth min(8, d* + 3)): every old pose is in the new table; the new d is never larger than
   the old one; where the old d is exact (d <= old depth - the pose's depth) the two are equal.
2. Against the goal-side tables (d* + 3 actions back from the old tables' successful states, an upper bound): where
   the bound's path stays within depth 12 (the pose's depth + bound <= 12) the new d is at most the bound.
3. Re-search: poses the q3 runs reached outside the old tables (OUT_DIR of reward.replay_q3, outside_old.jsonl),
   N per task, each the start of a new search of depth min(8, d* + 3) with the same rule (the main line's
   outside_table_probe_r13.py): when it finds d and d <= 12 - the pose's depth, the table must give exactly d; when it
   finds d further out, the table gives d or more (or no success); when it finds nothing, the table's d is beyond
   the search depth (or none).
4. The start's d equals d*, the table reaches depth 12.

    python -m reward.check_2a --rows ROWS --task-ids FILE --new DIR --old DIR --goal DIR --outside FILE [FILE...]
        --gs-root DIR --out OUT.json [--per-task 5] [--workers 4]
"""

import argparse
import contextlib
import io
import json
import random
from multiprocessing import Pool
from pathlib import Path

import numpy as np

DEPTH = 12


def _align(a_keys, b_keys):
    """Indices into b for each key of a, and whether it is there."""
    i = np.minimum(np.searchsorted(b_keys, a_keys), len(b_keys) - 1)
    return i, b_keys[i] == a_keys


def compare_old(new, old):
    i, found = _align(old["keys"], new["keys"])
    d_new, d_old = new["dist"][i], old["dist"]
    old_depth = int(old.get("depth") or old["depth_of"].max())
    larger = found & (d_old >= 0) & ((d_new < 0) | (d_new > d_old))
    exact = found & (d_old >= 0) & (d_old <= old_depth - old["depth_of"])
    return {"old_states": int(len(old["keys"])), "old_missing_in_new": int((~found).sum()),
            "new_larger_than_old": int(larger.sum()), "old_exact": int(exact.sum()),
            "old_exact_not_equal": int((exact & (d_new != d_old)).sum()),
            "old_none_new_found": int((found & (d_old < 0) & (d_new >= 0)).sum())}


def compare_goal(new, goal):
    i, found = _align(goal["keys"], new["keys"])
    d_new, d_goal = new["dist"][i], goal["dist"]
    dep = new["depth_of"][i]
    inside = found & (d_goal >= 0) & (dep.astype(np.int64) + d_goal <= DEPTH)
    return {"goal_states": int(len(goal["keys"])), "goal_in_new": int(found.sum()), "checked": int(inside.sum()),
            "new_above_bound": int((inside & ((d_new < 0) | (d_new > d_goal))).sum()),
            "new_equal_bound": int((inside & (d_new == d_goal)).sum()),
            "new_below_bound": int((inside & (d_new >= 0) & (d_new < d_goal)).sum())}


def research(item):
    row, pose, gs_root = item
    from data_gen.interiorgs_v10 import distance_table as DT
    from data_gen.interiorgs_v10.scene import load_scene

    ds = int(row["difficulty"]["d_star"])
    depth = min(8, ds + 3)
    with contextlib.redirect_stdout(io.StringIO()):
        scene = load_scene(Path(gs_root) / str(row["scene_id"]), str(row["scene_id"]), wall_model="thick")
    probe = dict(row, init_camera=dict(row["init_camera"], extrinsics=pose))
    t = DT.build(scene, probe, depth)
    return {"d": t["d_init"] if t["d_init"] >= 0 else None, "depth": depth, "states": t["states"],
            "seconds": t["seconds"]["total"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--task-ids", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--old", required=True)
    ap.add_argument("--goal", required=True)
    ap.add_argument("--outside", nargs="+", required=True)
    ap.add_argument("--gs-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-task", type=int, default=5)
    ap.add_argument("--workers", type=int, default=4)  # the login node: at most 4 cores a task
    ap.add_argument("--seed", type=int, default=20261027)
    args = ap.parse_args()
    from data_gen.interiorgs_v10 import distance_table as DT

    ids = [l.strip() for l in Path(args.task_ids).read_text().splitlines() if l.strip()]
    rows = {}
    for l in Path(args.rows).read_text().splitlines():
        if l.strip():
            r = json.loads(l)
            if r["task_id"] in ids:
                rows[r["task_id"]] = r
    per_task, tables = {}, {}
    for t in ids:
        new = DT.load(DT.table_path(Path(args.new), t))
        tables[t] = new
        rec = {"task_type": rows[t]["task_type"], "d_star": int(rows[t]["difficulty"]["d_star"]),
               "d_init": int(new["d_init"]), "agree": bool(new["agree"]), "depth_reached": int(new["depth_reached"]),
               "states": int(new["states"])}
        old_p, goal_p = DT.table_path(Path(args.old), t), DT.table_path(Path(args.goal), t)
        rec["old"] = compare_old(new, DT.load(old_p)) if old_p.is_file() else None
        if goal_p.is_file():  # the goal-side files hold keys, dist and meta only (no depth_of)
            with np.load(goal_p) as z:
                rec["goal"] = compare_goal(new, {"keys": z["keys"], "dist": z["dist"]})
        else:
            rec["goal"] = None
        per_task[t] = rec
    # re-search a sample of the poses the q3 runs reached outside the old tables
    outside = {}
    for f in args.outside:
        for l in Path(f).read_text().splitlines():
            if l.strip():
                x = json.loads(l)
                if x["task_id"] in rows:
                    outside.setdefault((x["task_id"], x["key"]), x)
    rng = random.Random(args.seed)
    sample = []
    for t in ids:
        cands = sorted((x for (tid, _), x in outside.items() if tid == t), key=lambda x: x["key"])
        sample += rng.sample(cands, min(args.per_task, len(cands)))
    with Pool(args.workers) as pool:
        found = pool.map(research, [(rows[x["task_id"]], x["pose"], args.gs_root) for x in sample], chunksize=1)
    probes, bad = [], 0
    for x, r in zip(sample, found):
        new = tables[x["task_id"]]
        d = DT.lookup(new, np.asarray(x["pose"]))
        i = int(np.searchsorted(new["keys"], x["key"]))
        dep = int(new["depth_of"][i]) if i < len(new["keys"]) and int(new["keys"][i]) == x["key"] else None
        if dep is None:
            verdict = "pose_not_in_new_table"
        elif r["d"] is not None and r["d"] <= DEPTH - dep:
            verdict = "equal" if d == r["d"] else "MISMATCH"
        elif r["d"] is not None:
            verdict = "ok_beyond" if d is None or d >= r["d"] else "MISMATCH"
        else:
            verdict = "ok_not_found" if d is None or d > r["depth"] else "MISMATCH"
        bad += verdict in ("MISMATCH", "pose_not_in_new_table")
        probes.append({"task_id": x["task_id"], "key": x["key"], "depth_of_new": dep, "d_table": d,
                       "d_research": r["d"], "research_depth": r["depth"], "verdict": verdict,
                       "seconds": r["seconds"]})
    old_bad = sum((v["old"] or {}).get("old_missing_in_new", 0) + (v["old"] or {}).get("new_larger_than_old", 0)
                  + (v["old"] or {}).get("old_exact_not_equal", 0) for v in per_task.values())
    goal_bad = sum((v["goal"] or {}).get("new_above_bound", 0) for v in per_task.values())
    report = {"tasks": len(ids), "agree": sum(v["agree"] for v in per_task.values()),
              "depth_12": sum(v["depth_reached"] == DEPTH for v in per_task.values()),
              "old_violations": old_bad, "goal_violations": goal_bad,
              "research": {"probed": len(probes), "outside_old_distinct": len(outside), "bad": bad,
                           "verdicts": {k: sum(p["verdict"] == k for p in probes)
                                        for k in sorted({p["verdict"] for p in probes})}},
              "pass": old_bad == 0 and goal_bad == 0 and bad == 0
                      and all(v["agree"] and v["depth_reached"] == DEPTH for v in per_task.values()),
              "per_task": per_task, "probes": probes}
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k not in ("per_task", "probes")}, indent=1))


if __name__ == "__main__":
    main()
