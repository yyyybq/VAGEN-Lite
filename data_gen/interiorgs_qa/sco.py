"""Submit / inspect / delete a one-GPU SCO job (zoetrope pool) that runs a data_gen.interiorgs_qa module.

The job settings (image, mount, pool, card spec, CLI) are v10's (data_gen/interiorgs_v10/pilot_sco.py),
imported, not copied.  ``prepare`` freezes the code into JOB/source.tar.gz and writes JOB/launch.sh.

    python -m data_gen.interiorgs_qa.sco prepare --job JOB -- pipeline render --plan PLAN --out OUT
    python -m data_gen.interiorgs_qa.sco submit --job JOB
    python -m data_gen.interiorgs_qa.sco status --job JOB
    python -m data_gen.interiorgs_qa.sco delete --job JOB
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shlex
import subprocess
import tarfile
from pathlib import Path
from typing import List

from data_gen.interiorgs_v10.pilot_sco import (FINISHED, IMAGE, MOUNT, POOL, RUNTIME_ENV, SCO, WORKER_SPEC,
                                              describe, now)

ROOT = Path(__file__).resolve().parents[2]
GS_ROOT = "/mnt/umm/users/yinbaiqiao/InteriorGS_ply"


def prepare(job: Path, module_args: List[str]) -> None:
    job.mkdir(parents=True, exist_ok=True)
    archive = job / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for rel in ("data_gen/__init__.py", "data_gen/interiorgs_v10", "data_gen/interiorgs_qa", "vagen",
                    "ply_gaussian_loader.py"):
            if (ROOT / rel).exists():
                tar.add(ROOT / rel, arcname=rel,
                        filter=lambda ti: None if "__pycache__" in ti.name or "figures_cabinet" in ti.name else ti)
    module, rest = module_args[0], module_args[1:]
    cmd = " ".join(shlex.quote(a) for a in rest)
    launch = job / "launch.sh"
    launch.write_text(f"""#!/usr/bin/env bash
set -euo pipefail
JOB={job}
OWNER=$(stat -c %u:%g "$JOB/source.tar.gz")
trap 'chown -R "$OWNER" "$JOB" 2>/dev/null || true; chown -R "$OWNER" {ROOT}/exps/interiorgs_qa_* 2>/dev/null || true' EXIT
SRC=$JOB/src
mkdir -p "$SRC" "$JOB/logs"
exec > >(tee -a "$JOB/logs/run-$(date -u +%Y%m%dT%H%M%SZ)-$(hostname).log") 2>&1
cd "$SRC" && tar -xzf {archive}
RUNTIME_ENV={RUNTIME_ENV}
export PYTHONPATH="$SRC" PYTHONDONTWRITEBYTECODE=1 PATH="$RUNTIME_ENV/bin:$PATH" CUDA_VISIBLE_DEVICES=0
export VAGEN_GSPLAT_PREBUILT=1 TORCH_EXTENSIONS_DIR=/mnt/umm/users/yinbaiqiao/.cache/torch_extensions_jumpbox_renderer
export CC=$RUNTIME_ENV/bin/x86_64-conda-linux-gnu-gcc CXX=$RUNTIME_ENV/bin/x86_64-conda-linux-gnu-g++
export CUDA_HOME=$RUNTIME_ENV TORCH_CUDA_ARCH_LIST=9.0 OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 NO_PROXY='*' no_proxy='*'
export CPATH="$RUNTIME_ENV/targets/x86_64-linux/include:${{CPATH:-}}"
export LIBRARY_PATH="$RUNTIME_ENV/targets/x86_64-linux/lib:${{LIBRARY_PATH:-}}"
nvidia-smi --query-gpu=name,memory.total --format=csv || true

$RUNTIME_ENV/bin/python -m data_gen.interiorgs_qa.{module} {cmd} --gs-root {GS_ROOT}
echo JOB_DONE
""")
    launch.chmod(0o755)
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
    (job / "SHA256SUMS").write_text(f"{digest(archive)}  source.tar.gz\n{digest(launch)}  launch.sh\n")
    (job / "prepared.json").write_text(json.dumps({"utc": now(), "module": module, "args": rest, "pool": POOL,
                                                   "worker_spec": WORKER_SPEC}, indent=1))
    print(json.dumps({"launch": str(launch)}))


def submit(job: Path, name: str) -> None:
    receipt = job / "submission.json"
    if receipt.exists():
        print(json.loads(receipt.read_text())["job_id"])
        return
    subprocess.run(["sha256sum", "-c", "SHA256SUMS"], cwd=job, check=True)
    command = [SCO, "acp", "jobs", "create", "--workspace-name=aigc", f"--aec2-name={POOL}",
               f"--job-name={name}", "--priority=HIGHEST", "--quota-type=reserved",
               f"--container-image-url={IMAGE}", f"--storage-mount={MOUNT}", "--training-framework=pytorch",
               "--worker-nodes=1", f"--worker-spec={WORKER_SPEC}", f"--command=bash {job}/launch.sh"]
    (job / "submission_intent.json").write_text(json.dumps({"utc": now(), "command": command}, indent=1))
    res = subprocess.run(command, capture_output=True, text=True)
    (job / "submit.log").write_text(res.stdout + "\n" + res.stderr)
    ids = set(re.findall(r"pt-[a-z0-9]+", res.stdout))
    if res.returncode != 0 or len(ids) != 1:
        raise SystemExit("submission uncertain; read job/submit.log and the job list before any retry")
    job_id = ids.pop()
    receipt.write_text(json.dumps({"utc": now(), "job_id": job_id, "command": command}, indent=1))
    print(job_id)


def status(job: Path) -> None:
    job_id = json.loads((job / "submission.json").read_text())["job_id"]
    d = describe(job_id)
    print(json.dumps({"job_id": job_id, "state": d.get("state"), "start_time": d.get("start_time"),
                      "complete_time": d.get("complete_time"), "finished": d.get("state") in FINISHED}))


def delete(job: Path) -> None:
    receipt = job / "submission.json"
    job_id = json.loads(receipt.read_text())["job_id"]
    state = describe(job_id).get("state")
    if state not in FINISHED | {"SUSPENDED"}:
        subprocess.run([SCO, "acp", "jobs", "stop", "--workspace-name=aigc", job_id], capture_output=True, text=True)
    res = subprocess.run([SCO, "acp", "jobs", "delete", "--workspace-name=aigc", job_id], capture_output=True, text=True)
    (job / "delete.log").write_text(res.stdout + "\n" + res.stderr)
    if res.returncode == 0:
        receipt.rename(receipt.with_name(f"submission_{job_id}_deleted.json"))
    print(json.dumps({"job_id": job_id, "state_before": state, "delete_rc": res.returncode}))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=("prepare", "submit", "status", "delete"))
    ap.add_argument("--job", required=True, type=Path)
    ap.add_argument("--name", default=None, help="job name (submit)")
    import sys
    argv = sys.argv[1:]
    module_args = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[:argv.index("--")] if "--" in argv else argv
    args = ap.parse_args(argv)
    args.module_args = module_args
    job = args.job.resolve()
    if args.action == "prepare":
        margs = [a for a in args.module_args if a != "--"]
        prepare(job, margs)
    elif args.action == "submit":
        submit(job, args.name or ("interiorgs-qa-" + job.name.replace("_", "-")[-40:]))
    elif args.action == "status":
        status(job)
    else:
        delete(job)


if __name__ == "__main__":
    main()
