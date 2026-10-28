"""Reward line, step 2b: submit the depth-12 table build as one 8-card sco job, or as two shards when h800 or
zoetrope has two idle machines (the user's decision 1 of 2026-10-27).  Each job builds its share of the 3,000
training tasks (replacing over-limit ones from its share of the 300 backups, --fill), then its share of the 705
evaluation tasks (no replacement), with the prefilter step 2a validated.

Before submitting:
* downstream_runs/PAUSE_SUBMISSIONS must not exist (the handoff's rule for every line);
* the quota is read and recorded (the user's rule 7: before every submission);
* each pool's idle machines are estimated the downstream line's way: the pool's machines (sco aec2 clusters
  describe) minus the machines running any job of this workspace minus the jobs waiting in the pool.  Other
  workspaces' jobs are not seen, so the estimate can be too high (a job may then wait).

The job (as root in its container) gives the files back to the owner of reward_runs every 30 minutes and at exit
(the exit trap also ends that loop, so nothing outlives the build), and picks its worker count from the container's
CPU quota and memory limit (3 GB a worker; 1.6-3.0 GB measured at depth 12).

    python -m reward.submit_2b --prefilter strict|loose [--dry] [--force-shards 1|2] [--pool h800|zoetrope]

Writes RR/tables/jobs/<b3000_all | b3000_iof2>/job/{launch.sh, prepared.json, submission.json} and
RR/tables/jobs/submit_2b_<utc>.json (quota, idle estimates, decision, job ids).
"""

import argparse
import json
import os
import re
import subprocess
import time
from pathlib import Path

RR = Path("/mnt/umm/users/yinbaiqiao/reward_runs")
W = Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite-reward")
E = "/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial"
PAUSE = Path("/mnt/umm/users/yinbaiqiao/downstream_runs/PAUSE_SUBMISSIONS")
VOLUME = "019ec9f9-6d12-7d49-aad4-864b15c9eb06"
PREFIX = "rw"

LAUNCH = r"""#!/usr/bin/env bash
# Reward line step 2b, {label}: depth-12 tables of the 3,000 training tasks (then the 705 evaluation tasks).
set -u
W={W}
RR={RR}
PY=/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin/python
ROWS={E}/interiorgs_v10_r11_20261016/rl_rows/rows.jsonl
GS=/mnt/umm/users/yinbaiqiao/InteriorGS_ply
JOB={job}
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cd "$W" || exit 1
mkdir -p "$RR/tables/b3000" "$RR/tables/eval705"
OWNER=$(stat -c %u:%g "$RR")
fix_owner() {{ chown -R "$OWNER" "$RR/tables/b3000" "$RR/tables/eval705" "$JOB" 2>/dev/null; }}
( while true; do sleep 1800; fix_owner; done ) &
LOOP=$!
trap 'kill $LOOP 2>/dev/null; fix_owner' EXIT
Q=$(cut -d' ' -f1 /sys/fs/cgroup/cpu.max 2>/dev/null || echo max)
P=$(cut -d' ' -f2 /sys/fs/cgroup/cpu.max 2>/dev/null || echo 100000)
if [ "$Q" = max ]; then CPUS=$(getconf _NPROCESSORS_ONLN); else CPUS=$((Q / P)); fi
M=$(cat /sys/fs/cgroup/memory.max 2>/dev/null || echo max)
if [ "$M" = max ]; then M=$(awk '/MemTotal/ {{print $2 * 1024}}' /proc/meminfo); fi
BY_MEM=$((M / (3 * 1024 * 1024 * 1024)))
WORKERS=$((CPUS - 2))
[ "$BY_MEM" -lt "$WORKERS" ] && WORKERS=$BY_MEM
echo "{{\"utc\": \"$(date -u +%FT%TZ)\", \"host\": \"$(hostname)\", \"cpus\": $CPUS, \"memory_bytes\": $M, \"workers\": $WORKERS}}" > "$JOB/job_env.json"
$PY -m reward.distance12 build --rows "$ROWS" --task-ids "$RR/tasks/b3000.txt" \
    --backup-ids "$RR/tasks/b3000_backup300.txt" --fill {shard_arg} --out "$RR/tables/b3000" --gs-root "$GS" \
    --prefilter {prefilter} --workers "$WORKERS" > "$JOB/build_b3000.log" 2>&1
echo "$(date -u +%FT%TZ)" > "$JOB/b3000_done"
fix_owner
$PY -m reward.distance12 build --rows "$RR/tasks/eval705_rows.jsonl" --task-ids "$RR/tasks/eval705.txt" {shard_arg} \
    --out "$RR/tables/eval705" --gs-root "$GS" --prefilter {prefilter} --workers "$WORKERS" \
    > "$JOB/build_eval705.log" 2>&1
echo "$(date -u +%FT%TZ)" > "$JOB/eval705_done"
fix_owner
"""


def sco(*args):
    res = subprocess.run(["/mnt/umm/users/yinbaiqiao/.sco/bin/sco", *args], capture_output=True, text=True)
    return res.returncode, res.stdout


def quota():
    rc, out = sco("afs", "dir-quota", "list", "-i", VOLUME)
    line = next((l for l in out.splitlines() if "/users/yinbaiqiao" in l), None)
    if rc != 0 or line is None:
        return {"ok": False, "raw": out[-500:]}
    sizes = re.findall(r"(\d+(?:\.\d+)?)\s*(TB|GB|TiB|GiB)", line)
    return {"ok": True, "line": " ".join(line.split()), "sizes": [" ".join(s) for s in sizes]}


def pool_nodes(pool):
    rc, out = sco("aec2", "clusters", "describe", "--name", pool, "-o", "json")
    try:
        return len(json.loads(out)["properties"]["acn_uids"]) if rc == 0 else None
    except (ValueError, KeyError):
        return None


def idle(pool, job_sco):
    n, v = pool_nodes(pool), job_sco.pool_view(pool)
    if n is None or v is None:
        return None
    return {"machines": n, "busy": len(v["hosts"]), "waiting": len(v["waiting"]),
            "idle": max(0, n - len(v["hosts"]) - len(v["waiting"]))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefilter", choices=("strict", "loose"), required=True)
    ap.add_argument("--dry", action="store_true")
    ap.add_argument("--force-shards", type=int, choices=(1, 2), default=None)
    ap.add_argument("--pool", choices=("h800", "zoetrope"), default=None)
    args = ap.parse_args()
    os.environ["VAGEN_LITE_JOB_PREFIX"] = PREFIX
    os.environ["VAGEN_LITE_WORKTREE"] = str(W)
    from data_gen.interiorgs_v10 import job_sco

    utc = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    rec = {"utc": utc, "prefilter": args.prefilter, "pause_file": PAUSE.exists(), "quota": quota()}
    if rec["pause_file"]:
        print(json.dumps(rec, indent=1))
        raise SystemExit("PAUSE_SUBMISSIONS exists: nothing submitted")
    rec["idle"] = {p: idle(p, job_sco) for p in ("h800", "zoetrope")}
    two = [p for p in ("h800", "zoetrope") if (rec["idle"].get(p) or {}).get("idle", 0) >= 2]
    shards = args.force_shards or (2 if two else 1)
    pool = args.pool or (two[0] if shards == 2 and two else job_sco.pick_pool())
    rec.update(shards=shards, pool=pool, jobs=[])
    for i in range(shards):
        label = "b3000_all" if shards == 1 else f"b3000_{i}of{shards}"
        run = RR / "tables" / "jobs" / label
        job = run / "job"
        job.mkdir(parents=True, exist_ok=True)
        shard_arg = "" if shards == 1 else f"--shard {i}/{shards}"
        (job / "launch.sh").write_text(LAUNCH.format(label=label, W=W, RR=RR, E=E, job=job, shard_arg=shard_arg,
                                                     prefilter=args.prefilter))
        (job / "launch.sh").chmod(0o755)
        (job / "prepared.json").write_text(json.dumps({"utc": job_sco.now(), "gpus": 8, "spec": job_sco.SPECS[8],
                                                       "pool": pool, "mode_args": ["distance12", label]}, indent=1))
        job_id = None if args.dry else job_sco.submit(run)
        rec["jobs"].append({"label": label, "run": str(run), "job_id": job_id})
    out = RR / "tables" / "jobs" / f"submit_2b_{utc}.json"
    out.write_text(json.dumps(rec, indent=1))
    print(json.dumps(rec, indent=1))


if __name__ == "__main__":
    main()
