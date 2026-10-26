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

Round 12 batch: ``pick_pool()`` returns the pool with the fewest jobs waiting for cards (the workspace's job list,
every user's jobs); ``status`` also gives the job's creation and start times, so the chain can tell how long it
queued.  ``pool_view(pool)`` gives the workspace's running jobs of a pool by machine (every page of the job list
and each job's workers) and the jobs waiting there, for the chain's renderer_repack.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import List, Optional, Tuple

from vagen import config as _config

DEFAULT_WORKTREE = _config.path("repo")  # the cluster checkout, the code directory since round 6
# Round 12 freeze (the user's section 4 of 2026-10-22): VAGEN_LITE_WORKTREE names another code directory (a copy of the
# checkout the code reorganization does not touch); its jobs start from there and the job script gets the same
# variable.  Unset: the checkout above, and every launch script as before.
WORKTREE = os.environ.get("VAGEN_LITE_WORKTREE") or DEFAULT_WORKTREE
SCRIPT = "examples/train/active_spatial/sco_interiorgs_v10_train.sh"
SCO = _config.cluster("sco.bin")
IMAGE = _config.cluster("sco.image")
MOUNT = _config.cluster("sco.storage_mount")
POOL = _config.cluster("sco.default_pool")
WORKSPACE = _config.cluster("sco.workspace")
POOLS = tuple(_config.cluster("sco.pools"))  # same H800 cards, specs, image and storage
# round 12 batch (the user's decision of 2026-10-20): a pool for the RL render services only (4 machines, same cards,
# zone and network); a job is prepared there only when asked, pick_pool never chooses it
RENDER_POOLS = tuple(_config.cluster("sco.render_pools"))
SPECS = {int(k): v for k, v in _config.cluster("sco.worker_specs").items()}  # H800 cards, 14 vCPU each
FINISHED = {"SUCCEEDED", "FAILED", "STOPPED", "CANCELLED"}
# created, no cards yet; sco spells it QUEUEING (round 12 batch: without it a job in that state counted as not waiting)
WAITING = {"STARTING", "PENDING", "QUEUING", "QUEUEING", "QUEUED", "WAITING", "CREATING"}


def now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def prepare(run: Path, gpus: int, mode_args: List[str], job_dir: str = "job", pool: Optional[str] = None) -> Path:
    pool = pool or POOL
    if pool not in POOLS + RENDER_POOLS:
        raise ValueError(f"pool must be one of {POOLS + RENDER_POOLS}")
    if gpus not in SPECS:
        raise ValueError(f"gpus must be one of {sorted(SPECS)}")
    if not mode_args:
        raise ValueError("give the mode and its arguments after --")
    job = run / job_dir
    job.mkdir(parents=True, exist_ok=True)
    launch = job / "launch.sh"
    args = " ".join(shlex.quote(a) for a in mode_args[1:])
    env = "" if WORKTREE == DEFAULT_WORKTREE else f"env VAGEN_LITE_WORKTREE={shlex.quote(WORKTREE)} "
    launch.write_text(f"#!/usr/bin/env bash\nexec {env}bash {WORKTREE}/{SCRIPT} {shlex.quote(mode_args[0])} {run} {args}\n")
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
    command = [SCO, "acp", "jobs", "create", f"--workspace-name={WORKSPACE}", f"--aec2-name={prepared.get('pool', POOL)}", f"--job-name={name}",
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
    res = subprocess.run([SCO, "acp", "jobs", "describe", f"--workspace-name={WORKSPACE}", "-o", "json", job_id],
                         capture_output=True, text=True)
    return json.loads(res.stdout) if res.returncode == 0 else {"state": "NOT_FOUND"}


def status(run: Path, job_dir: str = "job") -> dict:
    job_id = json.loads((run / job_dir / "submission.json").read_text())["job_id"]
    d = describe(job_id)
    return {"job_id": job_id, "state": d.get("state"), "finished": d.get("state") in FINISHED,
            "create_time": d.get("create_time"), "start_time": d.get("start_time")}


def waiting_by_pool(jobs: List[dict], pools=POOLS) -> dict:
    """{pool: jobs created but not yet given cards} of a job list (sco acp jobs list)."""
    out = {p: 0 for p in pools}
    for j in jobs:
        pool = j.get("resource_pool")
        name = pool.get("name") if isinstance(pool, dict) else pool
        if name in out and str(j.get("state")) in WAITING:
            out[name] += 1
    return out


def pick_pool(pools=POOLS) -> str:
    """The pool with the fewest waiting jobs right now (ties and a failed listing: the first of ``pools``)."""
    res = subprocess.run([SCO, "acp", "jobs", "list", f"--workspace-name={WORKSPACE}", "--page-size", "500", "-o", "json"],
                         capture_output=True, text=True)
    try:
        jobs = json.loads(res.stdout) if res.returncode == 0 else None
    except ValueError:
        jobs = None
    if not isinstance(jobs, list):
        return pools[0]
    waiting = waiting_by_pool(jobs, pools)
    return min(pools, key=lambda p: (waiting[p], list(pools).index(p)))


def list_jobs(max_pages: int = 100) -> Optional[List[dict]]:
    """Every job of the workspace (sco lists 500 at most per page, newest first; jobs started weeks ago still run on
    later pages), None when a page cannot be read."""
    jobs: dict = {}
    for page in range(1, max_pages + 1):
        res = subprocess.run([SCO, "acp", "jobs", "list", f"--workspace-name={WORKSPACE}", "--page-size", "500",
                              "--page-token", str(page), "-o", "json"], capture_output=True, text=True)
        try:
            batch = json.loads(res.stdout) if res.returncode == 0 else None
        except ValueError:
            batch = None
        if not isinstance(batch, list):
            return None
        jobs.update({str(j.get("name")): j for j in batch})
        if len(batch) < 500:
            return list(jobs.values())
    return None


def worker_hosts(text: str) -> List[Tuple[int, str]]:
    """(cards, machine IP) per worker row of ``sco acp jobs get-workers`` (none for a job not placed yet)."""
    return [(int(c), h) for c, h in re.findall(r"加速芯片:(\d+)[^|]*\|\s*(\d+\.\d+\.\d+\.\d+)\s*\|", text)]


def job_hosts(job_id: str) -> List[Tuple[int, str]]:
    res = subprocess.run([SCO, "acp", "jobs", "get-workers", f"--workspace-name={WORKSPACE}", job_id], capture_output=True,
                         text=True)
    return worker_hosts(res.stdout) if res.returncode == 0 else []


def pool_view(pool: str) -> Optional[dict]:
    """Round 12 batch (the training chain's renderer_repack): the workspace's jobs in a pool by machine,
    {"hosts": {machine IP: [{"job_id", "name", "cards"}]}, "waiting": [display names of the jobs waiting for cards]}.
    Other workspaces' jobs on the same machines are not seen.  None when the job list or a running job's machine
    cannot be read (nothing is decided on a partial view)."""
    jobs = list_jobs()
    if jobs is None:
        return None
    hosts: dict = {}
    waiting = []
    for j in jobs:
        p = j.get("resource_pool")
        if (p.get("name") if isinstance(p, dict) else p) != pool:
            continue
        state, name = str(j.get("state")), str(j.get("display_name") or "")
        if state in WAITING:
            waiting.append(name)
        elif state == "RUNNING":
            rows = job_hosts(str(j.get("name")))
            if not rows:
                return None
            for cards, host in rows:
                hosts.setdefault(host, []).append({"job_id": str(j.get("name")), "name": name, "cards": cards})
    return {"hosts": hosts, "waiting": waiting}


def delete(run: Path, job_dir: str = "job") -> dict:
    receipt = run / job_dir / "submission.json"
    job_id = json.loads(receipt.read_text())["job_id"]
    state = describe(job_id).get("state")
    rc = 0
    if state != "NOT_FOUND":
        if state not in FINISHED | {"SUSPENDED"}:
            subprocess.run([SCO, "acp", "jobs", "stop", f"--workspace-name={WORKSPACE}", job_id], capture_output=True, text=True)
        rc = subprocess.run([SCO, "acp", "jobs", "delete", f"--workspace-name={WORKSPACE}", job_id], capture_output=True,
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
