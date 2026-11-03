"""Card-hours of the matrix v3 jobs, from the worker logs: start = the timestamp in the log's name, end = its mtime.

Cards per job come from the job folder's prepared.json ("gpus"); rl-* logs use job_rl, renderer-* use job_renderer.
A render service's worker log stops after start-up, so a service ends where the run's next service starts (a pool
move) or, for the last one, at logs/renderer_exit.json's ended_utc; the time it held its card while its trainer
queued counts.
Usage: python m3_cardhours.py ROOT OUT_JSON
"""
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone

root, out = sys.argv[1], sys.argv[2]
pat = re.compile(r"^(?P<kind>[a-z_]+)-(?P<t>\d{8}T\d{6}Z)-(?P<job>pt-[0-9a-f]+)-worker-0\.log$")


def gpus_for(logdir, kind):
    base = os.path.dirname(logdir)
    cands = []
    if kind == "rl":
        cands.append(os.path.join(base, "job_rl", "prepared.json"))
    elif kind == "renderer":
        cands.append(os.path.join(base, "job_renderer", "prepared.json"))
    cands += [os.path.join(base, "prepared.json"), os.path.join(base, "job", "prepared.json")]
    for sub in sorted(os.listdir(base)):
        p = os.path.join(base, sub, "prepared.json")
        if sub.startswith("job") and os.path.exists(p):
            cands.append(p)
    for p in cands:
        if os.path.exists(p):
            return json.load(open(p))["gpus"], p
    return None, None


rows = []
for dp, dn, fn in os.walk(root):
    for f in fn:
        m = pat.match(f)
        if not m:
            continue
        path = os.path.join(dp, f)
        t0 = datetime.strptime(m["t"], "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc).timestamp()
        t1 = os.path.getmtime(path)
        g, src = gpus_for(dp, m["kind"])
        rows.append({"log": os.path.relpath(path, root), "kind": m["kind"], "gpus": g, "gpus_from": src,
                     "start": t0, "hours": (t1 - t0) / 3600})

services = defaultdict(list)
for r in rows:
    if r["kind"] == "renderer":
        services[os.path.dirname(os.path.join(root, r["log"]))].append(r)
for logdir, rs in services.items():
    rs.sort(key=lambda r: r["start"])
    ended = json.load(open(os.path.join(logdir, "renderer_exit.json")))["ended_utc"]
    last = datetime.strptime(ended, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()
    for i, r in enumerate(rs):
        r["hours"] = ((rs[i + 1]["start"] if i + 1 < len(rs) else last) - r["start"]) / 3600

tot = defaultdict(float)
for r in rows:
    rel = r["log"]
    top = rel.split(os.sep)[0]
    if "/sft_" in "/" + rel and r["kind"] != "eval" and "eval_runs" not in rel:
        what = "sft_train"
    elif "eval_runs" in rel:
        what = "eval"
    elif "probe_runs" in rel:
        what = "probe"
    elif r["kind"] == "rl":
        what = "rl_trainer"
    elif r["kind"] == "renderer":
        what = "rl_renderer"
    else:
        what = r["kind"]
    group = top.split("_")[1] if top.startswith("m3_") else top
    r["what"], r["group"] = what, group
    if r["gpus"] is not None:
        tot[(group, what)] += r["gpus"] * r["hours"]

missing = [r["log"] for r in rows if r["gpus"] is None]
summary = {f"{g}/{w}": round(v, 1) for (g, w), v in sorted(tot.items())}
by_group = defaultdict(float)
for (g, w), v in tot.items():
    by_group[g] += v
json.dump({"rows": rows, "summary": summary, "by_group": {k: round(v, 1) for k, v in by_group.items()},
           "total": round(sum(tot.values()), 1), "missing_gpus": missing}, open(out, "w"), indent=1)
print(json.dumps(summary, indent=1))
print("by group", {k: round(v, 1) for k, v in sorted(by_group.items())})
print("total", round(sum(tot.values()), 1), "logs", len(rows), "missing gpus", len(missing), missing[:5])
print("kinds", sorted({r["kind"] for r in rows}))
