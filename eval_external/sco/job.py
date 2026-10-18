"""One-card sco jobs for eval_external.

    python -m eval_external.sco.job submit --run RUN_DIR --name step1-vsi-base7b -- python -m eval_external.qa_transfer.run_vlmevalkit ...
    python -m eval_external.sco.job status --run RUN_DIR
    python -m eval_external.sco.job delete --run RUN_DIR      # after it ends (or to cancel)
    python -m eval_external.sco.job active                    # our jobs that are not finished

Rules (decided 2026-10-18): pool h800, one H800 per job, at most MAX_ACTIVE of our jobs not finished at once
(the round-11 training chain runs on zoetrope).  Every job is recorded in ROOT/sco_jobs.jsonl.  The job runs as
root in the container; launch.sh hands RUN_DIR and LMUData back to the user when it ends.  The command runs
with the eval-external conda env, offline Hugging Face (everything is downloaded on the login node first) and
the worktree as working directory.
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import shlex
import subprocess
from pathlib import Path

WORKTREE = "/mnt/umm/users/yinbaiqiao/VAGEN-Lite-eval"
ROOT = Path("/mnt/umm/users/yinbaiqiao/eval_external_assets")
ENV_BIN = "/mnt/umm/users/yinbaiqiao/.conda/envs/eval-external/bin"
HF_HOME = str(ROOT / "hf_home")  # hub/ links to the shared ~/.cache/huggingface/hub (tools/login_env.sh)
SCO = "/mnt/umm/users/yinbaiqiao/.sco/bin/sco"
IMAGE = "registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617"   # same image as the training jobs
MOUNT = "019ec9f9-6d12-7d49-aad4-864b15c9eb06:/mnt/umm"
POOL = "h800"
SPEC = "N4lS.Iq.I80.1"  # one H800, 14 vCPU
MAX_ACTIVE = 4
OWNER = "20325:20325"  # yinbaiqiao
FINISHED = {"SUCCEEDED", "FAILED", "STOPPED", "CANCELLED", "NOT_FOUND"}
LEDGER = ROOT / "sco_jobs.jsonl"


def now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def launch_script(run: Path, command: list[str], env: dict[str, str]) -> str:
    exports = {
        "PATH": f"{ENV_BIN}:$PATH",
        "HF_HOME": HF_HOME,
        "HF_HUB_OFFLINE": "1",
        "HF_DATASETS_OFFLINE": "1",
        "LMUData": str(ROOT / "LMUData"),
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONNOUSERSITE": "1",
        "PYTHONUNBUFFERED": "1",
    }
    exports.update(env)
    lines = ["#!/usr/bin/env bash", f"# eval_external job, written {now()}", "set -o pipefail"]
    lines += [f"export {k}={v if k == 'PATH' else shlex.quote(v)}" for k, v in exports.items()]
    job = run / "job"
    lines += [
        f"cd {WORKTREE}",
        f"nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv > {job}/gpu.txt 2>&1",
        f"{' '.join(shlex.quote(c) for c in command)} > {job}/run.log 2>&1",
        "rc=$?",
        f"echo $rc > {job}/exit_code",
        f"chown -R {OWNER} {run} {ROOT / 'LMUData'} 2>/dev/null",
        f"find {HF_HOME} -mindepth 1 -maxdepth 1 ! -name hub -exec chown -R {OWNER} {{}} + 2>/dev/null",
        "exit $rc",
    ]
    return "\n".join(lines) + "\n"


def describe(job_id: str) -> dict:
    res = subprocess.run([SCO, "acp", "jobs", "describe", "--workspace-name=aigc", "-o", "json", job_id],
                         capture_output=True, text=True)
    return json.loads(res.stdout) if res.returncode == 0 and res.stdout.strip() else {"state": "NOT_FOUND"}


def ledger() -> list[dict]:
    if not LEDGER.exists():
        return []
    return [json.loads(line) for line in LEDGER.read_text().splitlines() if line.strip()]


def active() -> list[dict]:
    out = []
    for rec in ledger():
        if rec.get("deleted"):
            continue
        state = describe(rec["job_id"]).get("state")
        if state not in FINISHED:
            out.append({**rec, "state": state})
    return out


def submit(run: Path, name: str, command: list[str], env: dict[str, str], force_limit: bool = False) -> str:
    job = run / "job"
    receipt = job / "submission.json"
    if receipt.exists():
        return json.loads(receipt.read_text())["job_id"]
    running = active()
    if len(running) >= MAX_ACTIVE and not force_limit:
        raise SystemExit(f"{len(running)} eval_external jobs not finished (limit {MAX_ACTIVE}): "
                         + ", ".join(r["name"] for r in running))
    job.mkdir(parents=True, exist_ok=True)
    (job / "launch.sh").write_text(launch_script(run, command, env))
    (job / "launch.sh").chmod(0o755)
    job_name = ("evalext-" + re.sub(r"[^a-z0-9-]+", "-", name.lower())).strip("-")[:60]
    cmd = [SCO, "acp", "jobs", "create", "--workspace-name=aigc", f"--aec2-name={POOL}", f"--job-name={job_name}",
           "--priority=HIGHEST", "--quota-type=reserved", f"--container-image-url={IMAGE}",
           f"--storage-mount={MOUNT}", "--training-framework=pytorch", "--worker-nodes=1",
           f"--worker-spec={SPEC}", f"--command=bash {job}/launch.sh"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    (job / "submit.log").write_text(res.stdout + "\n" + res.stderr)
    ids = set(re.findall(r"pt-[a-z0-9]+", res.stdout))
    if res.returncode != 0 or len(ids) != 1:
        raise SystemExit(f"submission uncertain; read {job}/submit.log and the job list before any retry")
    job_id = ids.pop()
    rec = {"utc": now(), "job_id": job_id, "name": job_name, "run": str(run), "pool": POOL, "spec": SPEC,
           "command": command}
    receipt.write_text(json.dumps(rec, indent=1) + "\n")
    with open(LEDGER, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return job_id


def status(run: Path) -> dict:
    rec = json.loads((run / "job" / "submission.json").read_text())
    d = describe(rec["job_id"])
    exit_file = run / "job" / "exit_code"
    return {"job_id": rec["job_id"], "name": rec["name"], "state": d.get("state"),
            "exit_code": exit_file.read_text().strip() if exit_file.exists() else None}


def delete(run: Path) -> dict:
    receipt = run / "job" / "submission.json"
    rec = json.loads(receipt.read_text())
    state = describe(rec["job_id"]).get("state")
    rc = 0
    if state != "NOT_FOUND":
        if state not in FINISHED | {"SUSPENDED"}:
            subprocess.run([SCO, "acp", "jobs", "stop", "--workspace-name=aigc", rec["job_id"]], capture_output=True)
        rc = subprocess.run([SCO, "acp", "jobs", "delete", "--workspace-name=aigc", rec["job_id"]],
                            capture_output=True).returncode
    if rc == 0:
        receipt.rename(receipt.with_name(f"submission_{rec['job_id']}_deleted.json"))
        rows = ledger()
        for r in rows:
            if r["job_id"] == rec["job_id"]:
                r["deleted"] = now()
        LEDGER.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return {"job_id": rec["job_id"], "state_before": state, "delete_rc": rc}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=("submit", "status", "delete", "active"))
    ap.add_argument("--run", type=Path)
    ap.add_argument("--name")
    ap.add_argument("--env", action="append", default=[], help="KEY=VALUE exported in the job")
    ap.add_argument("--force-limit", action="store_true", help="submit even with MAX_ACTIVE jobs running")
    ap.add_argument("command", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    if args.action == "active":
        print(json.dumps(active(), indent=1))
        return
    run = args.run.resolve()
    if args.action == "submit":
        command = args.command[1:] if args.command[:1] == ["--"] else args.command
        if not command or not args.name:
            raise SystemExit("submit needs --name and a command after --")
        env = dict(kv.split("=", 1) for kv in args.env)
        print(submit(run, args.name, command, env, args.force_limit))
    elif args.action == "status":
        print(json.dumps(status(run)))
    else:
        print(json.dumps(delete(run)))


if __name__ == "__main__":
    main()
