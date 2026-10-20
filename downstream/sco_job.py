#!/usr/bin/env python3
"""sco jobs for the downstream line (StarVLA / RLinf / RoboCasa365): prepare, submit, status, delete.

    python downstream/sco_job.py prepare --run RUN_DIR --gpus 1 --pool h800 --name dsload -- bash PAYLOAD.sh ARGS
    python downstream/sco_job.py submit  --run RUN_DIR
    python downstream/sco_job.py status  --run RUN_DIR
    python downstream/sco_job.py delete  --run RUN_DIR     # after it ends, or to cancel

RUN_DIR/job/ holds launch.sh (what the worker runs), prepared.json, submission.json (the job id) and worker.log.
The payload runs as the shared-filesystem owner (examples/train/active_spatial/sco_run_as_artifact_owner.sh), so
outputs are not root-owned.  Same image, mount and card specs as the training line's job_sco.py.
"""
from __future__ import annotations

import argparse
import datetime
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path
from typing import List

WORKTREE = Path(__file__).resolve().parents[1]
SCO = "/mnt/umm/users/yinbaiqiao/.sco/bin/sco"
IMAGE = "registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617"
MOUNT = "019ec9f9-6d12-7d49-aad4-864b15c9eb06:/mnt/umm"
POOLS = ("h800", "zoetrope")  # the same H800-80G cards; h800 usually starts sooner for one-card jobs
SPECS = {1: "N4lS.Iq.I80.1", 8: "N4lS.Iq.I80.8"}
FINISHED = {"SUCCEEDED", "FAILED", "STOPPED", "CANCELLED"}
RUN_AS_OWNER = "examples/train/active_spatial/sco_run_as_artifact_owner.sh"


def now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def launch_script(job: Path, payload: List[str], worktree: Path = WORKTREE) -> str:
    cmd = " ".join(shlex.quote(a) for a in payload)
    log = job / "worker.log"
    return (
        "#!/usr/bin/env bash\n"
        "set -uo pipefail\n"
        f"cd {shlex.quote(str(worktree))}\n"
        f"exec > >(tee -a {shlex.quote(str(log))}) 2>&1\n"
        'echo "[launch] host=$(hostname) date=$(date -u +%FT%TZ) uid=$(id -u)"\n'
        "nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true\n"
        f"bash {shlex.quote(str(worktree / RUN_AS_OWNER))} {cmd}\n"
        "rc=$?\n"
        'echo "[launch] rc=$rc date=$(date -u +%FT%TZ)"\n'
        f"echo $rc > {shlex.quote(str(job / 'exit_code'))}\n"
    )


def prepare(run: Path, gpus: int, pool: str, name: str, payload: List[str]) -> Path:
    if gpus not in SPECS:
        raise ValueError(f"gpus must be one of {sorted(SPECS)}")
    if pool not in POOLS:
        raise ValueError(f"pool must be one of {POOLS}")
    if not re.fullmatch(r"[a-z0-9-]{3,40}", name):
        raise ValueError("name: 3-40 characters a-z, 0-9, -")
    if not payload:
        raise ValueError("give the payload command after --")
    job = run / "job"
    if (job / "submission.json").exists():
        raise FileExistsError(f"{job} was already submitted; use a new run directory")
    job.mkdir(parents=True, exist_ok=True)
    launch = job / "launch.sh"
    launch.write_text(launch_script(job, payload))
    launch.chmod(0o755)
    (job / "prepared.json").write_text(json.dumps({"utc": now(), "gpus": gpus, "spec": SPECS[gpus], "pool": pool,
                                                   "name": name, "payload": payload, "worktree": str(WORKTREE),
                                                   "code": code_state(WORKTREE)}, indent=1) + "\n")
    return launch


def code_state(worktree: Path) -> dict:
    """Commit and dirty files of the worktree at prepare time (the job image has no git)."""
    def git(*a):
        res = subprocess.run(["git", "-C", str(worktree), *a], capture_output=True, text=True)
        return res.stdout.strip() if res.returncode == 0 else None

    dirty = git("status", "--porcelain", "--untracked-files=no", "--", "downstream", "docs/downstream")
    return {"commit": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty_downstream_files": dirty.splitlines() if dirty else []}


def create_command(job: Path, prepared: dict) -> List[str]:
    return [SCO, "acp", "jobs", "create", "--workspace-name=aigc", f"--aec2-name={prepared['pool']}",
            f"--job-name=downstream-{prepared['name']}", "--priority=HIGHEST", "--quota-type=reserved",
            f"--container-image-url={IMAGE}", f"--storage-mount={MOUNT}", "--training-framework=pytorch",
            "--worker-nodes=1", f"--worker-spec={prepared['spec']}", f"--command=bash {job}/launch.sh"]


def submit(run: Path) -> str:
    job = run / "job"
    receipt = job / "submission.json"
    if receipt.exists():
        return json.loads(receipt.read_text())["job_id"]
    prepared = json.loads((job / "prepared.json").read_text())
    command = create_command(job, prepared)
    res = subprocess.run(command, capture_output=True, text=True)
    (job / "submit.log").write_text(res.stdout + "\n" + res.stderr)
    ids = set(re.findall(r"pt-[a-z0-9]+", res.stdout))
    if res.returncode != 0 or len(ids) != 1:
        raise SystemExit("submission uncertain; read job/submit.log and the job list before any retry")
    job_id = ids.pop()
    receipt.write_text(json.dumps({"utc": now(), "job_id": job_id, "command": command}, indent=1) + "\n")
    return job_id


def status(run: Path) -> dict:
    job_id = json.loads((run / "job/submission.json").read_text())["job_id"]
    res = subprocess.run([SCO, "acp", "jobs", "describe", "--workspace-name=aigc", "-o", "json", job_id],
                         capture_output=True, text=True)
    d = json.loads(res.stdout) if res.returncode == 0 else {"state": "NOT_FOUND"}
    exit_code = run / "job/exit_code"
    return {"job_id": job_id, "state": d.get("state"), "finished": d.get("state") in FINISHED,
            "create_time": d.get("create_time"), "start_time": d.get("start_time"),
            "exit_code": exit_code.read_text().strip() if exit_code.exists() else None}


def delete(run: Path) -> str:
    job_id = json.loads((run / "job/submission.json").read_text())["job_id"]
    st = status(run)
    if not st["finished"]:
        subprocess.run([SCO, "acp", "jobs", "stop", "--workspace-name=aigc", job_id], capture_output=True, text=True)
    res = subprocess.run([SCO, "acp", "jobs", "delete", "--workspace-name=aigc", job_id], capture_output=True, text=True)
    return f"{job_id}: delete rc={res.returncode} {res.stdout.strip()} {res.stderr.strip()}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["prepare", "submit", "status", "delete"])
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--pool", default="h800")
    ap.add_argument("--name", default="job")
    ap.add_argument("payload", nargs=argparse.REMAINDER)
    args = ap.parse_args(argv)
    payload = args.payload[1:] if args.payload[:1] == ["--"] else args.payload
    if args.action == "prepare":
        print(prepare(args.run, args.gpus, args.pool, args.name, payload))
    elif args.action == "submit":
        print(submit(args.run))
    elif args.action == "status":
        print(json.dumps(status(args.run)))
    else:
        print(delete(args.run))
    return 0


if __name__ == "__main__":
    sys.exit(main())
