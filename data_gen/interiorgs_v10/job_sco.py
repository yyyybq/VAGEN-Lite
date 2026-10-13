"""One sco job (zoetrope) running a mode of examples/train/active_spatial/sco_interiorgs_v10_train.sh.

    python data_gen/interiorgs_v10/job_sco.py prepare --run RUN --gpus 1 -- sft-trial LF_DIR 30
    python data_gen/interiorgs_v10/job_sco.py submit  --run RUN
    python data_gen/interiorgs_v10/job_sco.py status  --run RUN
    python data_gen/interiorgs_v10/job_sco.py delete  --run RUN    # after it ends (or to cancel)

Trials use one card (the user's rule for every job but the formal training); the formal SFT and PPO
trainers use eight, the PPO render service one.
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import shlex
import subprocess
from pathlib import Path
from typing import List

WORKTREE = "/mnt/umm/users/yinbaiqiao/VAGEN-Lite-r4"
SCRIPT = "examples/train/active_spatial/sco_interiorgs_v10_train.sh"
SCO = "/mnt/umm/users/yinbaiqiao/.sco/bin/sco"
IMAGE = "registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617"
MOUNT = "019ec9f9-6d12-7d49-aad4-864b15c9eb06:/mnt/umm"
POOL = "zoetrope"
SPECS = {1: "N4lS.Iq.I80.1", 8: "N4lS.Iq.I80.8"}  # H800 cards, 14 vCPU each
FINISHED = {"SUCCEEDED", "FAILED", "STOPPED", "CANCELLED"}


def now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def prepare(run: Path, gpus: int, mode_args: List[str]) -> Path:
    if gpus not in SPECS:
        raise ValueError(f"gpus must be one of {sorted(SPECS)}")
    if not mode_args:
        raise ValueError("give the mode and its arguments after --")
    job = run / "job"
    job.mkdir(parents=True, exist_ok=True)
    launch = job / "launch.sh"
    args = " ".join(shlex.quote(a) for a in mode_args[1:])
    launch.write_text(f"#!/usr/bin/env bash\nexec bash {WORKTREE}/{SCRIPT} {shlex.quote(mode_args[0])} {run} {args}\n")
    launch.chmod(0o755)
    (job / "prepared.json").write_text(json.dumps({"utc": now(), "gpus": gpus, "spec": SPECS[gpus], "pool": POOL,
                                                   "mode_args": mode_args}, indent=1))
    return launch


def submit(run: Path) -> str:
    job = run / "job"
    receipt = job / "submission.json"
    if receipt.exists():
        return json.loads(receipt.read_text())["job_id"]
    prepared = json.loads((job / "prepared.json").read_text())
    name = f"interiorgs-v10-{prepared['mode_args'][0]}-{run.name}".replace("_", "-")[-60:]
    command = [SCO, "acp", "jobs", "create", "--workspace-name=aigc", f"--aec2-name={POOL}", f"--job-name={name}",
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


def status(run: Path) -> dict:
    job_id = json.loads((run / "job" / "submission.json").read_text())["job_id"]
    d = describe(job_id)
    return {"job_id": job_id, "state": d.get("state"), "finished": d.get("state") in FINISHED}


def delete(run: Path) -> dict:
    receipt = run / "job" / "submission.json"
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
    ap.add_argument("mode_args", nargs="*")
    args = ap.parse_args()
    run = args.run.resolve()
    if args.action == "prepare":
        print(prepare(run, args.gpus, args.mode_args))
    elif args.action == "submit":
        print(submit(run))
    elif args.action == "status":
        print(json.dumps(status(run)))
    else:
        print(json.dumps(delete(run)))


if __name__ == "__main__":
    main()
