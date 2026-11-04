"""The resume-only part of the six finished RL runs' last checkpoints (B' and D_noP0; the user's 甲, 2026-11-04, as A.4
did on 11-03): the model / optimizer / extra-state shards of global_step_300/actor, 24 files a checkpoint; the
huggingface folder and the small files stay.  Checks: each chain is done, update 300 is each run's last checkpoint,
its huggingface folder is complete (config.json and every shard its index names), no live sco job is the run's
trainer or render service or names a file to delete.  Record: exps/matrix_v3/rl_resume_deleted_20261104.tsv.

    python rl_resume_del.py JOBS_JSON             # list and check, nothing deleted
    python rl_resume_del.py JOBS_JSON --delete
"""
import json
import re
import sys
import time
from pathlib import Path

M3 = Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix/exps/matrix_v3")
RUNS = {"m3_b_s20261019": ["m3bs1s0", "m3bs1d"], "m3_dn_s20261019": ["m3dns1s0", "m3dns1d"],
        "m3_dn_s20261021": ["m3dns2s0", "m3dns2d"]}
OUT = M3 / "rl_resume_deleted_20261104.tsv"
RESUME = re.compile(r"(model|optim|extra_state)_world_size_\d+_rank_\d+\.pt$")
jobs = json.loads(Path(sys.argv[1]).read_text())
live = [j for j in jobs if str(j.get("state")) not in ("SUCCEEDED", "FAILED", "STOPPED", "DELETED", "COMPLETED")]
files, problems = [], []
for chain, runs in RUNS.items():
    if json.loads((M3 / chain / "progress.json").read_text()).get("state") != "done":
        problems.append(f"{chain} not done")
    for run in runs:
        ck = M3 / chain / f"rl_run_{run}" / "checkpoints"
        steps = sorted(int(p.name.rsplit("_", 1)[1]) for p in ck.glob("global_step_*"))
        if steps[-1] != 300:
            problems.append(f"{run}: last checkpoint {steps[-1]}")
        actor = ck / "global_step_300" / "actor"
        hf = actor / "huggingface"
        shards = set(json.loads((hf / "model.safetensors.index.json").read_text())["weight_map"].values())
        if not (hf / "config.json").is_file() or not all((hf / s).is_file() for s in shards):
            problems.append(f"{run}: huggingface folder incomplete")
        fs = sorted(p for p in actor.iterdir() if RESUME.match(p.name))
        if len(fs) != 24:
            problems.append(f"{run}: {len(fs)} resume files, not 24")
        files += fs
        pat = re.compile(rf"rl-run-{run}\b|rl_run_{run}/checkpoints/global_step_300/actor/(model|optim|extra_state)_")
        for j in live:
            if pat.search(json.dumps(j)):
                problems.append(f"live job {j.get('name')} ({j.get('display_name')}) names {run}")
nbytes = sum(p.stat().st_size for p in files)
print(f"{len(files)} files in {sum(len(r) for r in RUNS.values())} checkpoints, {nbytes / 2**30:.1f} GiB "
      f"({nbytes / 1e9:.1f} GB); live jobs checked {len(live)}; problems: {problems}")
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
    hf_ok = all((M3 / c / f"rl_run_{r}" / "checkpoints" / "global_step_300" / "actor" / "huggingface" / "config.json").is_file()
                for c, rs in RUNS.items() for r in rs)
    print(f"deleted {len(files) - len(left)}, left {len(left)}, huggingface folders intact: {hf_ok}; record {OUT}")
