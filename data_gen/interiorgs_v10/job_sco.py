"""One sco job (zoetrope) running a mode of examples/train/active_spatial/sco_interiorgs_v10_train.sh.

    python data_gen/interiorgs_v10/job_sco.py prepare --run RUN --gpus 1 -- sft-trial LF_DIR 30
    python data_gen/interiorgs_v10/job_sco.py submit  --run RUN
    python data_gen/interiorgs_v10/job_sco.py status  --run RUN
    python data_gen/interiorgs_v10/job_sco.py delete  --run RUN    # after it ends (or to cancel)

A formal PPO run is two jobs on one RUN (the render service writes RUN/renderer_endpoint.txt, the trainer reads
it): give each its own --job-dir, e.g. job_renderer and job_rl.

Trials use one card (the user's rule for every job but the formal training); the formal SFT and PPO
trainers use eight, the PPO render service one.  The pool is zoetrope unless a job is prepared with another one
(--pool, the training chain's plan "pool"; round 11: h800 when zoetrope is full, the user's decision).
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import shlex
import subprocess
from pathlib import Path
from typing import List, Optional

WORKTREE = "/mnt/umm/users/yinbaiqiao/VAGEN-Lite"  # the cluster checkout, the only code directory since round 6
SCRIPT = "examples/train/active_spatial/sco_interiorgs_v10_train.sh"
SCO = "/mnt/umm/users/yinbaiqiao/.sco/bin/sco"
IMAGE = "registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617"
MOUNT = "019ec9f9-6d12-7d49-aad4-864b15c9eb06:/mnt/umm"
POOL = "zoetrope"
POOLS = ("zoetrope", "h800")  # same H800 cards, specs, image and storage
SPECS = {1: "N4lS.Iq.I80.1", 8: "N4lS.Iq.I80.8"}  # H800 cards, 14 vCPU each
FINISHED = {"SUCCEEDED", "FAILED", "STOPPED", "CANCELLED"}


def now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def prepare(run: Path, gpus: int, mode_args: List[str], job_dir: str = "job", pool: Optional[str] = None) -> Path:
    pool = pool or POOL
    if pool not in POOLS:
        raise ValueError(f"pool must be one of {POOLS}")
    if gpus not in SPECS:
        raise ValueError(f"gpus must be one of {sorted(SPECS)}")
    if not mode_args:
        raise ValueError("give the mode and its arguments after --")
    job = run / job_dir
    job.mkdir(parents=True, exist_ok=True)
    launch = job / "launch.sh"
    args = " ".join(shlex.quote(a) for a in mode_args[1:])
    launch.write_text(f"#!/usr/bin/env bash\nexec bash {WORKTREE}/{SCRIPT} {shlex.quote(mode_args[0])} {run} {args}\n")
    launch.chmod(0o755)
    (job / "prepared.json").write_text(json.dumps({"utc": now(), "gpus": gpus, "spec": SPECS[gpus], "pool": pool,
                                                   "mode_args": mode_args}, indent=1))
    return launch


def submit(run: Path, job_dir: str = "job") -> str:
    job = run / job_dir
    receipt = job / "submission.json"
    if receipt.exists():
        return json.loads(receipt.read_text())["job_id"]
    prepared = json.loads((job / "prepared.json").read_text())
    name = f"interiorgs-v10-{prepared['mode_args'][0]}-{run.name}".replace("_", "-")[-60:]
    command = [SCO, "acp", "jobs", "create", "--workspace-name=aigc", f"--aec2-name={prepared.get('pool', POOL)}", f"--job-name={name}",
               "--priority=HIGHEST", "--quota-type=reserved", f"--container-image-url={IMAGE}",
               f"--storage-mount={MOUNT}", "--training-framework=pytorch", "--worker-nodes=1",
               f"--worker-spec={prepared['spec']}", f"--command=bash {job}/launch.sh"]
    res = subprocess.run(command, capture_output=True, text=True)
    (job / "submit.log").write_text(res.stdout + "\n" + res.stderr)
    ids = set(re.findall(r"pt-[a-z0-9]+", res.stdout))
    if res.returncode != 0 or len(ids) != 1:
        raise SystemExit("submission uncertain; read job/submit.log and the job list before any retry")
    job_id = ids.pop()
    receipt.write_text(json.dumps({"utc": now(), "job_id": job_id, "command": command}, indent=1))
    return job_id


def describe(job_id: str) -> dict:
    res = subprocess.run([SCO, "acp", "jobs", "describe", "--workspace-name=aigc", "-o", "json", job_id],
                         capture_output=True, text=True)
    return json.loads(res.stdout) if res.returncode == 0 else {"state": "NOT_FOUND"}


def status(run: Path, job_dir: str = "job") -> dict:
    job_id = json.loads((run / job_dir / "submission.json").read_text())["job_id"]
    d = describe(job_id)
    return {"job_id": job_id, "state": d.get("state"), "finished": d.get("state") in FINISHED}


def delete(run: Path, job_dir: str = "job") -> dict:
    receipt = run / job_dir / "submission.json"
    job_id = json.loads(receipt.read_text())["job_id"]
    state = describe(job_id).get("state")
    rc = 0
    if state != "NOT_FOUND":
        if state not in FINISHED | {"SUSPENDED"}:
            subprocess.run([SCO, "acp", "jobs", "stop", "--workspace-name=aigc", job_id], capture_output=True, text=True)
        rc = subprocess.run([SCO, "acp", "jobs", "delete", "--workspace-name=aigc", job_id], capture_output=True,
                            text=True).returncode
    if rc == 0:
        receipt.rename(receipt.with_name(f"submission_{job_id}_deleted.json"))
    return {"job_id": job_id, "state_before": state, "delete_rc": rc}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=("prepare", "submit", "status", "delete"))
    ap.add_argument("--run", required=True, type=Path)
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--job-dir", default="job", help="folder under RUN for this job")
    ap.add_argument("--pool", choices=POOLS, default=None, help=f"resource pool (default {POOL})")
    ap.add_argument("mode_args", nargs="*")
    args = ap.parse_args()
    run = args.run.resolve()
    if args.action == "prepare":
        print(prepare(run, args.gpus, args.mode_args, args.job_dir, args.pool))
    elif args.action == "submit":
        print(submit(run, args.job_dir))
    elif args.action == "status":
        print(json.dumps(status(run, args.job_dir)))
    else:
        print(json.dumps(delete(run, args.job_dir)))


if __name__ == "__main__":
    main()
