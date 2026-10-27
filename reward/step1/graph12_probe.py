"""Reward line, step 1 (writes nothing): the depth-12 state graph of a few reserve tasks, without judging every state.

For each task: states per level (distance_table.reachable_graph with a high cap), seconds, peak memory, the prefilter's
pass share, and the success oracle's cost on a random sample of prefilter-passing states, from which the CPU hours of
judging every prefilter-passing state follow.  Picks one reserve training task per (family, d*) in WANT.

    PYTHONDONTWRITEBYTECODE=1 python -I reward/step1/graph12_probe.py [DEPTH]

Run on the 2026-10-27 login node with code d530b546 (qwen3-main); results in
docs/reward/reports/STEP1_ASSETS_20261027.md section 3.
"""
import contextlib
import io
import json
import random
import resource
import sys
import time
from pathlib import Path

CODE = str(Path(__file__).resolve().parents[2])  # this worktree
sys.path.insert(0, CODE)
import numpy as np  # noqa: E402

from data_gen.interiorgs_v10.difficulty import FastGeometry, to_c2w  # noqa: E402
from data_gen.interiorgs_v10.distance_table import may_succeed, reachable_graph  # noqa: E402
from data_gen.interiorgs_v10.release import env_success_oracle  # noqa: E402
from data_gen.interiorgs_v10.scene import load_scene  # noqa: E402

ROWS = "/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/rl_rows/rows.jsonl"
GS = "/mnt/umm/users/yinbaiqiao/InteriorGS_ply"
DEPTH = int(sys.argv[1]) if len(sys.argv) > 1 else 12
WANT = [("projective_relations", 1), ("equidistance", 3), ("occlusion_alignment", 5), ("absolute_positioning", 7)]


def fam(r):
    return r.get("family") or r.get("task_family") or r.get("task_type")


rows = [json.loads(l) for l in open(ROWS) if l.strip()]
rows = [r for r in rows if r.get("reserve") and r.get("split") == "train"]
print("reserve train rows", len(rows), "families", sorted({fam(r) for r in rows}), flush=True)
picked = []
for f, d in WANT:
    cand = sorted((r for r in rows if fam(r) == f), key=lambda r: (abs(int(r["difficulty"]["d_star"]) - d), r["task_id"]))
    if cand:
        picked.append(cand[0])
for row in picked:
    t0 = time.perf_counter()
    with contextlib.redirect_stdout(io.StringIO()):
        scene = load_scene(Path(GS) / str(row["scene_id"]), str(row["scene_id"]), wall_model="thick")
    geom = FastGeometry.for_scene(scene)
    t1 = time.perf_counter()
    keys, P, R, depth_of, edges = reachable_graph(geom, np.asarray(row["init_camera"]["extrinsics"], float), DEPTH,
                                                  max_states=60_000_000)
    t2 = time.perf_counter()
    per_level = np.bincount(depth_of.astype(int)).tolist()
    c2w = to_c2w(P, R)
    passed = may_succeed(row, c2w)
    t3 = time.perf_counter()
    oracle = env_success_oracle(row, scene)
    idx = np.flatnonzero(passed)
    rng = random.Random(0)
    sample = rng.sample(list(idx), min(1500, len(idx)))
    t4 = time.perf_counter()
    hits = sum(bool(oracle(c2w[i])) for i in sample)
    t5 = time.perf_counter()
    ms = 1000 * (t5 - t4) / max(1, len(sample))
    rss_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6
    out = {"task_id": row["task_id"], "family": fam(row), "d_star": row["difficulty"]["d_star"], "depth": DEPTH,
           "states": int(len(keys)), "edges": int(len(edges)), "per_level": per_level,
           "growth_last4": [round(per_level[i] / per_level[i - 1], 2) for i in range(len(per_level) - 4, len(per_level))],
           "prefilter_share": round(float(passed.mean()), 3), "sample_success_share": round(hits / max(1, len(sample)), 3),
           "ms_per_judged_state": round(ms, 2),
           "est_cpu_hours_judging": round(float(passed.sum()) * ms / 1000 / 3600, 2),
           "seconds": {"scene": round(t1 - t0, 1), "graph": round(t2 - t1, 1), "prefilter": round(t3 - t2, 1)},
           "peak_rss_gb_process": round(rss_gb, 2)}
    print(json.dumps(out), flush=True)
    del keys, P, R, depth_of, edges, c2w, passed
