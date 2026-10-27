"""Reward line, step 1 (read-only): how big depth-12 tables would be, from the main line's depth-min(8, d*+3) tables.

Reads R12/analysis/r13_distance/{reserve/summary.jsonl, probe_c8_one.json, probe_c8_full.log, reserve_goal/*.npz}.
Prints statistics by table depth, the per-level growth of the main line's depth-8 probes, an extrapolation to depth 12
(superseded by the direct measurement of graph12_probe.py: growth slows far faster past level 8), bytes a state on
disk and the load time of a table.  Writes nothing.

    PYTHONDONTWRITEBYTECODE=1 python -I reward/step1/table_stats.py
"""
import glob
import json
import os
import statistics as st
import sys
import time

import numpy as np

D = "/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r12_20261019/analysis/r13_distance"


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * (len(xs) - 1) + 0.5))]


rows = [json.loads(l) for l in open(f"{D}/reserve/summary.jsonl")]
print("reserve summary rows", len(rows), "agree", sum(1 for r in rows if r.get("agree")),
      "errors", sum(1 for r in rows if "error" in r))
by_depth = {}
for r in rows:
    by_depth.setdefault(r["depth"], []).append(r)
print("depth | n | states med / p90 / max | edges/state | prefilter share | success s med | ms per judged state med | sum s")
for d in sorted(by_depth):
    rs = by_depth[d]
    states = [r["states"] for r in rs]
    eps = [r["edges"] / r["states"] for r in rs]
    share = [r["prefilter_pass"] / r["states"] for r in rs]
    ss = [r["seconds"]["success"] for r in rs]
    ms = [1000 * r["seconds"]["success"] / max(1, r["prefilter_pass"]) for r in rs if r["prefilter_pass"] > 200]
    tot = sum(r["seconds"]["total"] for r in rs)
    print(f"{d} | {len(rs)} | {int(st.median(states))} / {q(states, .9)} / {max(states)} | {st.median(eps):.2f} | "
          f"{st.median(share):.2f} | {st.median(ss):.1f} | {st.median(ms):.2f} | {tot:.0f}")
print("total CPU hours (sum of task seconds)", round(sum(r["seconds"]["total"] for r in rows) / 3600, 1))

# per-level growth from the probes that record states_per_depth
probes = []
try:
    probes += json.load(open(f"{D}/probe_c8_one.json"))
except Exception as exc:
    print("probe_c8_one", exc)
raw = open(f"{D}/probe_c8_full.log", "rb").read().replace(b"\x00", b"").decode("utf-8", "replace")
for line in raw.splitlines():
    line = line.strip()
    if line.startswith("{"):
        try:
            probes.append(json.loads(line))
        except Exception:
            pass
print("\nprobes with states_per_depth:", len(probes))
growth = {}
for p in probes:
    spd = p.get("states_per_depth")
    if not spd:
        continue
    g = [spd[i] / spd[i - 1] for i in range(1, len(spd)) if spd[i - 1] > 0]
    for i, x in enumerate(g, start=1):
        growth.setdefault(i, []).append(x)
    ms = 1000 * p.get("seconds_prefiltered_success", 0) / max(1, p.get("prefilter_pass", 1))
    print(f"  {p['family']:22s} d*={p['d_star']} states={p['states']} last levels={spd[-4:]} "
          f"growth={[round(x, 2) for x in g[-4:]]} prefilter={p['prefilter_pass'] / p['states']:.2f} ms/judged={ms:.2f}")
print("median growth by level:", {k: round(st.median(v), 2) for k, v in sorted(growth.items())})

# projection: states at depth 12 from a depth-8 total, two growth scenarios for levels 9..12
g8 = st.median(growth.get(8, [3.0]))
scen = {"slowing (-0.2 a level)": [g8 - 0.2 * k for k in range(1, 5)], "constant": [g8] * 4}
d8 = [r["states"] for r in by_depth.get(8, [])]
print(f"\ndepth-8 tasks: {len(d8)}, total states median {int(st.median(d8))}, p90 {q(d8, .9)}")
for name, gs in scen.items():
    for label, s8 in (("median", st.median(d8)), ("p90", q(d8, .9))):
        last = s8 * (1 - 1 / g8)  # states on level 8
        total, lvl, through11 = s8, last, None
        for k, g in enumerate(gs, start=9):
            lvl *= g
            total += lvl
            if k == 11:
                through11 = total
        print(f"  {name:24s} {label:6s}: depth-12 total {total / 1e6:.2f} M states (through level 11 {through11 / 1e6:.2f} M)")

# table size and load time: existing tables and the goal-side tables (median 0.69 M states)
sizes = []
for f in sorted(glob.glob(f"{D}/reserve/*.npz"))[:400]:
    sizes.append(os.path.getsize(f))
tot_states = sum(r["states"] for r in rows)
tot_bytes = sum(os.path.getsize(f) for f in glob.glob(f"{D}/reserve/*.npz"))
print(f"\nreserve tables: {tot_bytes / 1e9:.2f} GB for {tot_states / 1e6:.1f} M states = {tot_bytes / tot_states:.2f} bytes a state")
files = sorted(glob.glob(f"{D}/reserve_goal/*.npz"))[:: max(1, len(glob.glob(f'{D}/reserve_goal/*.npz')) // 25)][:25]
n_states, secs, nbytes = 0, 0.0, 0
for f in files:
    t = time.perf_counter()
    with np.load(f) as z:
        k = z["keys"]
        d = z["dist"]
        meta = json.loads(str(z["meta"]))
    secs += time.perf_counter() - t
    n_states += len(k)
    nbytes += os.path.getsize(f)
print(f"goal-side load: {len(files)} tables, {n_states / 1e6:.2f} M states, {nbytes / 1e6:.1f} MB, "
      f"{secs:.2f} s -> {secs / max(1, n_states) * 1e6:.3f} s per M states, {nbytes / n_states:.2f} bytes a state")
print("python", sys.version.split()[0], "numpy", np.__version__)
