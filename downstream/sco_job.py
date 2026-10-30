#!/usr/bin/env python3
"""sco jobs for the downstream line (StarVLA / RLinf / RoboCasa365): prepare, submit, status, delete.

    python downstream/sco_job.py prepare --run RUN_DIR --gpus 1 --pool h800 --name dsload -- bash PAYLOAD.sh ARGS
    python downstream/sco_job.py submit  --run RUN_DIR
    python downstream/sco_job.py status  --run RUN_DIR
    python downstream/sco_job.py delete  --run RUN_DIR     # after it ends, or to cancel

RUN_DIR/job/ holds launch.sh (what the worker runs), prepared.json, submission.json (the job id) and worker.log.
The payload runs as the shared-filesystem owner (examples/train/active_spatial/sco_run_as_artifact_owner.sh), so
outputs are not root-owned.  Same image, mount and card specs as the training line's job_sco.py.
prepare(..., as_root=True) (2026-10-30, only for jobs that must read the root-only Qwen3-VL snapshot in
/mnt/umm/shared_model, as the model line's jobs do) runs the payload as the container's root instead; such a
payload gives what it writes back to the owner itself (run_qa_batch.sh does).
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


def launch_script(job: Path, payload: List[str], worktree: Path = WORKTREE, as_root: bool = False) -> str:
    cmd = " ".join(shlex.quote(a) for a in payload)
    log = job / "worker.log"
    run = cmd if as_root else f"bash {shlex.quote(str(worktree / RUN_AS_OWNER))} {cmd}"
    return (
        "#!/usr/bin/env bash\n"
        "set -uo pipefail\n"
        f"cd {shlex.quote(str(worktree))}\n"
        f"exec > >(tee -a {shlex.quote(str(log))}) 2>&1\n"
        'echo "[launch] host=$(hostname) date=$(date -u +%FT%TZ) uid=$(id -u)"\n'
        "nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader || true\n"
        f"{run}\n"
        "rc=$?\n"
        'echo "[launch] rc=$rc date=$(date -u +%FT%TZ)"\n'
        f"echo $rc > {shlex.quote(str(job / 'exit_code'))}\n"
    )


def prepare(run: Path, gpus: int, pool: str, name: str, payload: List[str], as_root: bool = False) -> Path:
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
    launch.write_text(launch_script(job, payload, as_root=as_root))
    launch.chmod(0o755)
    rec = {"utc": now(), "gpus": gpus, "spec": SPECS[gpus], "pool": pool, "name": name, "payload": payload,
           "worktree": str(WORKTREE), "code": code_state(WORKTREE)}
    if as_root:
        rec["as_root"] = True
    (job / "prepared.json").write_text(json.dumps(rec, indent=1) + "\n")
    return launch


def code_state(worktree: Path) -> dict:
    """Commit and dirty files of the worktree at prepare time (the job image has no git)."""
    def git(*a):
        res = subprocess.run(["git", "-C", str(worktree), *a], capture_output=True, text=True)
        return res.stdout.strip() if res.returncode == 0 else None

    dirty = git("status", "--porcelain", "--untracked-files=no", "--", "downstream", "docs/downstream",
                ":(exclude)docs/downstream/MATRIX.md")  # rewritten by the watcher every pass
    return {"commit": git("rev-parse", "HEAD"), "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty_downstream_files": dirty.splitlines() if dirty else []}


def create_command(job: Path, prepared: dict) -> List[str]:
    return [SCO, "acp", "jobs", "create", "--workspace-name=aigc", f"--aec2-name={prepared['pool']}",
            f"--job-name=downstream-{prepared['name']}", "--priority=HIGHEST", "--quota-type=reserved",
            f"--container-image-url={IMAGE}", f"--storage-mount={MOUNT}", "--training-framework=pytorch",
            "--worker-nodes=1", f"--worker-spec={prepared['spec']}", f"--command=bash {job}/launch.sh"]


def waiting_by_pool(jobs, pools=POOLS) -> dict:
    """{pool: jobs created but not yet given cards} of a `sco acp jobs list` result (every user's jobs)."""
    out = {p: 0 for p in pools}
    for j in jobs:
        pool = j.get("resource_pool")
        name = pool.get("name") if isinstance(pool, dict) else pool
        if name in out and j.get("state") in ("STARTING", "QUEUEING", "PENDING", "QUEUED", "WAITING", "CREATING"):
            out[name] += 1
    return out


def pick_pool(pools=POOLS) -> str:
    """The pool with the fewest waiting jobs right now (a failed listing: the first of `pools`)."""
    res = subprocess.run([SCO, "acp", "jobs", "list", "--workspace-name=aigc", "--page-size", "500", "-o", "json"],
                         capture_output=True, text=True)
    try:
        waiting = waiting_by_pool(json.loads(res.stdout), pools)
    except Exception:
        return pools[0]
    return min(pools, key=lambda p: (waiting[p], list(pools).index(p)))


WAITING_STATES = ("STARTING", "QUEUEING", "PENDING", "QUEUED", "WAITING", "CREATING")
CARDS_PER_MACHINE = 8


def list_jobs(max_pages: int = 100):
    """Every job of the workspace (sco lists at most 500 per page, newest first), None when a page cannot be read."""
    jobs = {}
    for page in range(1, max_pages + 1):
        res = subprocess.run([SCO, "acp", "jobs", "list", "--workspace-name=aigc", "--page-size", "500",
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


def worker_hosts(text: str):
    """[(cards, machine IP)] per worker row of `sco acp jobs get-workers` (the training line's job_sco.py)."""
    return [(int(c), h) for c, h in re.findall(r"加速芯片:(\d+)[^|]*\|\s*(\d+\.\d+\.\d+\.\d+)\s*\|", text)]


def job_hosts(job_id: str):
    res = subprocess.run([SCO, "acp", "jobs", "get-workers", "--workspace-name=aigc", job_id], capture_output=True,
                         text=True)
    return worker_hosts(res.stdout) if res.returncode == 0 else []


def pool_view(pool: str, jobs=None, hosts_of=job_hosts):
    """The workspace's running jobs of a pool by machine, {"hosts": {IP: [{"job_id", "name", "cards"}]}}, None when
    the list or a running job's machine cannot be read (decide nothing on a partial view).  Jobs of other workspaces
    on the same machines are not seen.  Same method as the training line's renderer_repack (job_sco.pool_view)."""
    jobs = list_jobs() if jobs is None else jobs
    if jobs is None:
        return None
    hosts = {}
    for j in jobs:
        p = j.get("resource_pool")
        if (p.get("name") if isinstance(p, dict) else p) != pool or j.get("state") != "RUNNING":
            continue
        rows = hosts_of(str(j.get("name")))
        if not rows:
            return None
        for cards, host in rows:
            hosts.setdefault(host, []).append({"job_id": str(j.get("name")), "name": str(j.get("display_name") or ""),
                                               "cards": int(cards)})
    return {"hosts": hosts}


def job_state(job_id: str):
    """sco's state of a job by id (None when it cannot be read)."""
    res = subprocess.run([SCO, "acp", "jobs", "describe", "--workspace-name=aigc", "-o", "json", job_id],
                         capture_output=True, text=True)
    try:
        return json.loads(res.stdout).get("state") if res.returncode == 0 else None
    except ValueError:
        return None


def busy_room(view, own_prefix: str = "downstream-ev") -> dict:
    """{machine: free cards} of machines that already run some job other than our own evaluation shards and still
    have free cards: the only places a one-card evaluation may start (the user's rule of 2026-10-22: never take an
    idle machine, or 8-card jobs cannot get one)."""
    room = {}
    for host, jobs in (view or {}).get("hosts", {}).items():
        used = sum(j["cards"] for j in jobs)
        others = [j for j in jobs if not j["name"].startswith(own_prefix)]
        if others and used < CARDS_PER_MACHINE:
            room[host] = CARDS_PER_MACHINE - used
    return room


def alone_on_machine(view, job_id: str, own_prefix: str = "downstream-ev"):
    """True if the machine running job_id has nothing but our evaluation shards on it (it took an idle machine),
    False if it shares the machine with another job, None if the job is not in the view."""
    for jobs in (view or {}).get("hosts", {}).values():
        if any(j["job_id"] == job_id for j in jobs):
            return all(j["name"].startswith(own_prefix) for j in jobs)
    return None


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
    argv = list(sys.argv[1:] if argv is None else argv)
    payload = argv[argv.index("--") + 1:] if "--" in argv else []  # everything after -- is the payload
    argv = argv[:argv.index("--")] if "--" in argv else argv
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["prepare", "submit", "status", "delete"])
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--pool", default="h800")
    ap.add_argument("--name", default="job")
    args = ap.parse_args(argv)
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
