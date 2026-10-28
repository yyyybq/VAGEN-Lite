"""Reward line: run one of this line's modules (reward.<module>) as a one-card sco job.  The user's login-node rule of
2026-10-28 (as corrected the same day): data making, batch rendering and batch scoring go to sco; replays, analyses
and tests may run on the login node at no more than 4 cores a task (processes x threads, after checking the cores in
use), and a task that needs more than 4 cores for long comes here.  Modelled on
data_gen/interiorgs_qa/sco.py, which only runs interiorgs_qa modules: the job settings (image, mount, card spec, sco)
are v10's, imported; ``prepare`` freezes the code (data_gen/interiorgs_v10, vagen, reward) into JOB/source.tar.gz,
so editing the worktree later cannot change a queued job.  The job has 14 vCPUs; the launch script caps every
library's threads at 1 (the modules use process pools of at most 12), and gives JOB and the module's output folder
back to their owner at exit (jobs run as root).

    python -m reward.sco_job prepare --job JOB [--chown DIR ...] -- <module> <args>
    python -m reward.sco_job submit --job JOB --name NAME [--pool h800|zoetrope]
    python -m reward.sco_job status --job JOB
    python -m reward.sco_job delete --job JOB        (after the results are read: the user's rule, delete the job)

Job names are rw-<NAME>.  On the login node itself this module only writes files and calls sco.
"""

from __future__ import annotations

import argparse
import json
import re
import shlex
import subprocess
import tarfile
from pathlib import Path
from typing import List

from data_gen.interiorgs_v10.job_sco import pick_pool
from data_gen.interiorgs_v10.pilot_sco import FINISHED, IMAGE, MOUNT, SCO, WORKER_SPEC, describe, now

ROOT = Path(__file__).resolve().parents[1]
PY = "/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin/python"
FROZEN = ("data_gen/__init__.py", "data_gen/interiorgs_v10", "vagen", "reward", "ply_gaussian_loader.py")


def prepare(job: Path, chown: List[str], module_args: List[str]) -> None:
    job.mkdir(parents=True, exist_ok=True)
    archive = job / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for rel in FROZEN:
            if (ROOT / rel).exists():
                tar.add(ROOT / rel, arcname=rel, filter=lambda ti: None if "__pycache__" in ti.name else ti)
    module, rest = module_args[0], module_args[1:]
    cmd = " ".join(shlex.quote(a) for a in rest)
    owned = " ".join(shlex.quote(d) for d in [str(job), *chown])
    launch = job / "launch.sh"
    launch.write_text(f"""#!/usr/bin/env bash
set -euo pipefail
JOB={job}
OWNER=$(stat -c %u:%g "$JOB/source.tar.gz")
trap 'chown -R "$OWNER" {owned} 2>/dev/null || true' EXIT
SRC=$JOB/src
mkdir -p "$SRC" "$JOB/logs"
exec > >(tee -a "$JOB/logs/run-$(date -u +%Y%m%dT%H%M%SZ)-$(hostname).log") 2>&1
cd "$SRC" && tar -xzf {archive}
export PYTHONPATH="$SRC" PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 NO_PROXY='*' no_proxy='*'
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
echo "{{\\"utc\\": \\"$(date -u +%FT%TZ)\\", \\"host\\": \\"$(hostname)\\", \\"cpu_max\\": \\"$(cat /sys/fs/cgroup/cpu.max 2>/dev/null)\\"}}" > "$JOB/job_env.json"
{PY} -m reward.{module} {cmd}
date -u +%FT%TZ > "$JOB/done"
""")
    launch.chmod(0o755)
    (job / "prepared.json").write_text(json.dumps({"utc": now(), "module": module, "args": rest,
                                                   "worker_spec": WORKER_SPEC, "frozen": list(FROZEN)}, indent=1))
    print(json.dumps({"launch": str(launch)}))


def submit(job: Path, name: str, pool: str = None) -> None:
    receipt = job / "submission.json"
    if receipt.exists():
        print(receipt.read_text())
        return
    pool = pool or pick_pool()
    command = [SCO, "acp", "jobs", "create", "--workspace-name=aigc", f"--aec2-name={pool}",
               f"--job-name=rw-{name}"[:60], "--priority=HIGHEST", "--quota-type=reserved",
               f"--container-image-url={IMAGE}", f"--storage-mount={MOUNT}", "--training-framework=pytorch",
               "--worker-nodes=1", f"--worker-spec={WORKER_SPEC}", f"--command=bash {job}/launch.sh"]
    res = subprocess.run(command, capture_output=True, text=True)
    (job / "submit.log").write_text(res.stdout + "\n" + res.stderr)
    ids = set(re.findall(r"pt-[a-z0-9]+", res.stdout))
    if res.returncode != 0 or len(ids) != 1:
        raise SystemExit("submission uncertain; read submit.log and the job list before any retry")
    receipt.write_text(json.dumps({"utc": now(), "job_id": ids.pop(), "pool": pool, "command": command}, indent=1))
    print(receipt.read_text())


def status(job: Path) -> None:
    job_id = json.loads((job / "submission.json").read_text())["job_id"]
    d = describe(job_id)
    print(json.dumps({"job_id": job_id, "state": d.get("state"), "finished": d.get("state") in FINISHED,
                      "done_marker": (job / "done").exists()}))


def delete(job: Path) -> None:
    receipt = job / "submission.json"
    job_id = json.loads(receipt.read_text())["job_id"]
    state = describe(job_id).get("state")
    if state not in FINISHED | {"NOT_FOUND"}:
        subprocess.run([SCO, "acp", "jobs", "stop", "--workspace-name=aigc", job_id], capture_output=True, text=True)
    rc = subprocess.run([SCO, "acp", "jobs", "delete", "--workspace-name=aigc", job_id], capture_output=True,
                        text=True).returncode if state != "NOT_FOUND" else 0
    if rc == 0:
        receipt.rename(receipt.with_name(f"submission_{job_id}_deleted.json"))
    print(json.dumps({"job_id": job_id, "state_before": state, "delete_rc": rc}))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=("prepare", "submit", "status", "delete"))
    ap.add_argument("--job", required=True, type=Path)
    ap.add_argument("--name", default=None)
    ap.add_argument("--pool", choices=("h800", "zoetrope"), default=None)
    ap.add_argument("--chown", nargs="*", default=[])
    ap.add_argument("module_args", nargs="*")
    args = ap.parse_args()
    job = args.job.resolve()
    if args.action == "prepare":
        prepare(job, args.chown, args.module_args)
    elif args.action == "submit":
        submit(job, args.name or job.name, args.pool)
    elif args.action == "status":
        status(job)
    else:
        delete(job)


if __name__ == "__main__":
    main()
