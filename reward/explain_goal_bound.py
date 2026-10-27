"""Reward line, step 2a: why a fresh search from a table state's own pose can be shorter than the table (the cases
check_goal_bound.py marks TABLE_WRONG).  Hypothesis: poses are merged at 1 cm / 1 deg (contract keys); a table
state stands for the first exact pose that reached its key, and the success rule or a collision check can differ
between two poses of one key, so a path through exact poses (the fresh search) can succeed where the merged graph
does not.

For each case: the fresh search's graph from the state's pose, its successful states nearest to the state, and for
each of them the key's entry in the depth-12 table (is the key there, its d, and whether the table's representative
pose of that key succeeds by the env's rule).  "same key, other outcome": the fresh pose succeeds, the table's pose of
the same key does not.

    python -m reward.explain_goal_bound --rows ROWS --cases 2a_goal_bound.json --new DIR --gs-root DIR --out OUT.json
"""

import argparse
import contextlib
import io
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

DEPTH = 12


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", required=True)
    ap.add_argument("--cases", required=True)
    ap.add_argument("--new", required=True)
    ap.add_argument("--gs-root", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    from data_gen.interiorgs_v10 import distance_table as DT
    from data_gen.interiorgs_v10.difficulty import FastGeometry, to_c2w
    from data_gen.interiorgs_v10.release import env_success_oracle
    from data_gen.interiorgs_v10.scene import load_scene

    cases = [s for s in json.loads(Path(args.cases).read_text())["states"] if s["verdict"] == "TABLE_WRONG"]
    by_task = defaultdict(list)
    for s in cases:
        by_task[s["task_id"]].append(s)
    rows = {}
    for l in Path(args.rows).read_text().splitlines():
        if l.strip():
            r = json.loads(l)
            if r["task_id"] in by_task:
                rows[r["task_id"]] = r
    out, tally = [], Counter()
    for t, items in by_task.items():
        row = rows[t]
        with contextlib.redirect_stdout(io.StringIO()):
            scene = load_scene(Path(args.gs_root) / str(row["scene_id"]), str(row["scene_id"]), wall_model="thick")
        geom = FastGeometry.for_scene(scene)
        oracle = env_success_oracle(row, scene)
        new = DT.load(DT.table_path(Path(args.new), t))
        keys, P, R, _, _ = DT.reachable_graph(geom, np.asarray(row["init_camera"]["extrinsics"], float), DEPTH, 6_000_000)
        order = np.argsort(keys, kind="stable")
        rep = to_c2w(P, R)[order]
        for s in items:
            # the state's pose as the table has it
            pos = int(np.searchsorted(new["keys"], s["key"]))
            fk, fP, fR, fdep, _ = DT.reachable_graph(geom, rep[pos], s["d_research"], 2_000_000)
            fc2w = to_c2w(fP, fR)
            last = np.flatnonzero(fdep == s["d_research"])
            hits = [i for i in last if bool(oracle(fc2w[i]))]
            notes = []
            for i in hits[:20]:
                k = int(fk[i])
                j = int(np.searchsorted(new["keys"], k))
                there = j < len(new["keys"]) and int(new["keys"][j]) == k
                rep_ok = bool(oracle(rep[j])) if there else None
                if not there:
                    kind = "key not in table"
                elif rep_ok:
                    kind = "table pose succeeds too"
                else:
                    kind = "same key, other outcome"
                notes.append({"key": k, "in_table": there, "d_table": int(new["dist"][j]) if there else None,
                              "table_pose_succeeds": rep_ok, "kind": kind,
                              "offset_cm": round(float(np.linalg.norm(fc2w[i][:3, 3] - rep[j][:3, 3]) * 100), 3)
                              if there else None})
            kinds = Counter(n["kind"] for n in notes)
            tally.update({"cases": 1, **{f"cases_with_{k}": 1 for k in kinds}})
            out.append({**{k: v for k, v in s.items() if k != "pose"}, "fresh_successes_at_d": len(hits),
                        "checked": len(notes), "kinds": dict(kinds), "examples": notes[:3]})
    Path(args.out).write_text(json.dumps({"tally": dict(tally), "cases": out}, indent=1))
    print(json.dumps(dict(tally), indent=1))


if __name__ == "__main__":
    main()
