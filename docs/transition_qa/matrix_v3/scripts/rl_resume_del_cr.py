"""The resume-only part of Cr's four RL runs' last checkpoints (the user's 甲, 2026-11-05, as 12.51 did for B' and
D_noP0): the model / optimizer / extra-state shards of the last checkpoint's actor folder, 24 files a checkpoint; the
huggingface folder and the small files stay.  The last checkpoint is 250 for the two seed-1 runs (stopped at 270 and
286), 300 for seed 2 S0 and 200 for seed 2 V4 (stopped at 223).  Checks: each chain is done, that step is the run's last
checkpoint (also in latest_checkpointed_iteration.txt), its huggingface folder is complete (config.json and every shard
its index names), no live sco job (every page of the job list) is the run's trainer or render service, or names the
actor folder other than its huggingface folder; live jobs that read the huggingface weights are listed and kept.
Record: exps/matrix_v3/rl_resume_deleted_20261105.tsv.

    python rl_resume_del_cr.py              # list and check, nothing deleted
    python rl_resume_del_cr.py --delete
"""
import json
import re
import sys
import time
from pathlib import Path

W = Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix")
M3 = W / "exps" / "matrix_v3"
sys.path.insert(0, str(W))
from data_gen.interiorgs_v10 import job_sco  # noqa: E402

RUNS = {"m3_cr_s20261019": {"m3crs1s0": 250, "m3crs1d": 250}, "m3_cr_s20261021": {"m3crs2s0": 300, "m3crs2d": 200}}
OUT = M3 / "rl_resume_deleted_20261105.tsv"
RESUME = re.compile(r"(model|optim|extra_state)_world_size_\d+_rank_\d+\.pt$")
FINISHED = ("SUCCEEDED", "FAILED", "STOPPED", "DELETED", "COMPLETED")


def actor(chain: str, run: str, step: int) -> Path:
    return M3 / chain / f"rl_run_{run}" / "checkpoints" / f"global_step_{step}" / "actor"


def hf_complete(hf: Path) -> bool:
    shards = set(json.loads((hf / "model.safetensors.index.json").read_text())["weight_map"].values())
    return (hf / "config.json").is_file() and all((hf / s).is_file() for s in shards)


jobs = job_sco.list_jobs()
assert jobs is not None, "a page of the job list could not be read"
live = [j for j in jobs if str(j.get("state")) not in FINISHED]
files, problems, readers = [], [], []
for chain, runs in RUNS.items():
    if json.loads((M3 / chain / "progress.json").read_text()).get("state") != "done":
        problems.append(f"{chain} not done")
    for run, step in runs.items():
        ck = M3 / chain / f"rl_run_{run}" / "checkpoints"
        steps = sorted(int(p.name.rsplit("_", 1)[1]) for p in ck.glob("global_step_*"))
        latest = (ck / "latest_checkpointed_iteration.txt").read_text().strip()
        if steps[-1] != step or latest != str(step):
            problems.append(f"{run}: last checkpoint {steps[-1]} (latest file {latest}), expected {step}")
        a = actor(chain, run, step)
        if not hf_complete(a / "huggingface"):
            problems.append(f"{run}: huggingface folder incomplete")
        fs = sorted(p for p in a.iterdir() if RESUME.match(p.name))
        if len(fs) != 24:
            problems.append(f"{run}: {len(fs)} resume files, not 24")
        files += fs
        bad = re.compile(rf"rl-run-{run}\b|rl_run_{run}/checkpoints/global_step_{step}/actor(?!/huggingface)")
        for j in live:
            text = json.dumps(j, ensure_ascii=False)
            if bad.search(text):
                problems.append(f"live job {j.get('name')} ({j.get('display_name')}) names {run}'s trainer or actor folder")
            elif f"rl_run_{run}/" in text:
                readers.append(f"{j.get('name')} ({j.get('display_name')}, {j.get('state')}): {run}'s huggingface weights")
nbytes = sum(p.stat().st_size for p in files)
print(f"{len(files)} files in {sum(len(r) for r in RUNS.values())} checkpoints, {nbytes / 2**30:.1f} GiB "
      f"({nbytes / 1e9:.1f} GB); jobs listed {len(jobs)}, live {len(live)}; problems: {problems}")
for r in readers:
    print("  live job, kept reading:", r)
for chain, runs in RUNS.items():
    for run, step in runs.items():
        print(f"  {actor(chain, run, step)}")
if "--delete" in sys.argv:
    assert not problems, problems
    new = not OUT.exists()
    with open(OUT, "a") as out:
        if new:
            out.write("path\tbytes\tdeleted_utc\n")
        for p in files:
            b = p.stat().st_size
            p.unlink()
            out.write(f"{p}\t{b}\t{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}\n")
    left = [p for p in files if p.exists()]
    hf_ok = all(hf_complete(actor(c, r, s) / "huggingface") for c, rs in RUNS.items() for r, s in rs.items())
    print(f"deleted {len(files) - len(left)}, left {len(left)}, huggingface folders complete: {hf_ok}; record {OUT}")
