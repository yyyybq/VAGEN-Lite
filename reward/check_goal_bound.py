"""Reward line, step 2a: the states where the new depth-12 table's d is above the main line's goal-side bound (with
the bound's path inside depth 12, so the table should have found it).  The goal-side tables count actions backwards
from the old tables' successful states and take a move's inverse to undo it; near walls (collision checks) and at
the 1 cm / 1 deg merge of poses that is not always exact, so the bound itself can be wrong.

For each such state: its pose as the table has it (the representative pose of its key, from a rebuild of the
task's depth-12 graph), and a fresh search from that pose with the env's rule, as deep as the bound.  If it finds
no success within the bound, the bound is wrong there and the table is right; if it does, the table is wrong.

    python -m reward.check_goal_bound --rows ROWS --task-ids FILE --new DIR --goal DIR --gs-root DIR --out OUT.json
        [--workers 12]
"""

import argparse
import contextlib
import io
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

DEPTH = 12


def violations(args):
    row, new_dir, goal_dir, gs_root = args
    from data_gen.interiorgs_v10 import distance_table as DT
    from data_gen.interiorgs_v10.difficulty import FastGeometry, to_c2w
    from data_gen.interiorgs_v10.scene import load_scene

    t = row["task_id"]
    new = DT.load(DT.table_path(Path(new_dir), t))
    with np.load(DT.table_path(Path(goal_dir), t)) as z:
        gk, gd = z["keys"], z["dist"]
    i = np.minimum(np.searchsorted(new["keys"], gk), len(new["keys"]) - 1)
    found = new["keys"][i] == gk
    d_new, dep = new["dist"][i], new["depth_of"][i]
    bad = found & (gd >= 0) & (dep.astype(np.int64) + gd <= DEPTH) & ((d_new < 0) | (d_new > gd))
    if not bad.any():
        return []
    with contextlib.redirect_stdout(io.StringIO()):
        scene = load_scene(Path(gs_root) / str(row["scene_id"]), str(row["scene_id"]), wall_model="thick")
    keys, P, R, depth_of, _ = DT.reachable_graph(FastGeometry.for_scene(scene),
                                                  np.asarray(row["init_camera"]["extrinsics"], float), DEPTH, 6_000_000)
    order = np.argsort(keys, kind="stable")
    c2w = to_c2w(P, R)[order]
    assert np.array_equal(keys[order], new["keys"])
    out = []
    for j in np.flatnonzero(bad):
        k = int(gk[j])
        pos = int(np.searchsorted(new["keys"], k))
        out.append({"task_id": t, "key": k, "pose": c2w[pos].tolist(), "depth_of": int(dep[j]), "d_new": int(d_new[j]),
                    "d_goal": int(gd[j])})
    return out


def research(item):
    row, v, gs_root = item
    from data_gen.interiorgs_v10 import distance_table as DT
    from data_gen.interiorgs_v10.scene import load_scene

    with contextlib.redirect_stdout(io.StringIO()):
        scene = load_scene(Path(gs_root) / str(row["scene_id"]), str(row["scene_id"]), wall_model="thick")
    probe = dict(row, init_camera=dict(row["init_camera"], extrinsics=v["pose"]))
    t = DT.build(scene, probe, v["d_goal"])
    d = t["d_init"] if t["d_init"] >= 0 else None
    return dict({k: x for k, x in v.items() if k != "pose"}, d_research=d,
                verdict="bound_wrong" if d is None or d > v["d_goal"] else ("TABLE_WRONG" if d < v["d_new"] or v["d_new"] < 0
                                                                             else "bound_wrong"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--task-ids", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--goal", required=True)
    ap.add_argument("--gs-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    ids = [l.strip() for l in Path(args.task_ids).read_text().splitlines() if l.strip()]
    rows = {}
    for l in Path(args.rows).read_text().splitlines():
        if l.strip():
            r = json.loads(l)
            if r["task_id"] in ids:
                rows[r["task_id"]] = r
    with Pool(args.workers) as pool:
        found = [v for part in pool.map(violations, [(rows[t], args.new, args.goal, args.gs_root) for t in ids])
                 for v in part]
        res = pool.map(research, [(rows[v["task_id"]], v, args.gs_root) for v in found], chunksize=1)
    verdicts = {k: sum(r["verdict"] == k for r in res) for k in sorted({r["verdict"] for r in res})}
    report = {"violations": len(found), "verdicts": verdicts, "table_wrong": verdicts.get("TABLE_WRONG", 0),
              "states": res}
    Path(args.out).write_text(json.dumps(report, indent=1))
    print(json.dumps({k: v for k, v in report.items() if k != "states"}, indent=1))


if __name__ == "__main__":
    main()
