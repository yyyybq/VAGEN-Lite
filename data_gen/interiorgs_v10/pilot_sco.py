"""Submit / inspect / stop the pilot render job: one sco job, one GPU, zoetrope pool.

    python data_gen/interiorgs_v10/pilot_sco.py prepare --pilot PILOT --gs-root SCENES_WITH_PLY
    python data_gen/interiorgs_v10/pilot_sco.py submit  --pilot PILOT
    python data_gen/interiorgs_v10/pilot_sco.py status  --pilot PILOT
    python data_gen/interiorgs_v10/pilot_sco.py stop    --pilot PILOT     # only if it hangs
    python data_gen/interiorgs_v10/pilot_sco.py delete  --pilot PILOT     # remove a stopped/suspended job

On this cluster a stopped job is SUSPENDED and could be resumed by the scheduler; ``delete`` removes it so
nothing is left hanging.

``prepare`` freezes the code (data_gen/interiorgs_v10, vagen, ply_gaussian_loader.py) into
PILOT/job/source.tar.gz and writes
launch.sh, which unpacks it, sets up the renderer exactly as the QA production job did, runs
``pilot render`` and hands every output back to the submitting user (sco jobs run as root).
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import re
import subprocess
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ENV = "/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite"
SCO = "/mnt/umm/users/yinbaiqiao/.sco/bin/sco"
IMAGE = "registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617"
MOUNT = "019ec9f9-6d12-7d49-aad4-864b15c9eb06:/mnt/umm"
POOL = "zoetrope"
WORKER_SPEC = "N4lS.Iq.I80.1"  # one H800 (14 vCPU)
FINISHED = {"SUCCEEDED", "FAILED", "STOPPED", "CANCELLED"}


def now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def prepare(pilot: Path, gs_root: str, command: str = "pilot render") -> None:
    """``command``: the data_gen.interiorgs_v10 module and subcommand the job runs with ``--pilot --gs-root``
    (default the frame renderer; ``depth_check run`` for the rendered-depth check)."""
    job = pilot / "job"
    job.mkdir(parents=True, exist_ok=True)
    archive = job / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        # ply_gaussian_loader.py: the renderer imports it top-level; the repo copy decodes packed PLYs ~60x
        # faster than the older one installed in the conda env
        for rel in ("data_gen/interiorgs_v10", "vagen", "ply_gaussian_loader.py"):
            tar.add(ROOT / rel, arcname=rel,
                    filter=lambda ti: None if "__pycache__" in ti.name or "figures_cabinet" in ti.name else ti)
    launch = job / "launch.sh"
    launch.write_text(f"""#!/usr/bin/env bash
set -euo pipefail
PILOT={pilot}
OWNER=$(stat -c %u:%g "$PILOT/frames.jsonl")
trap 'chown -R "$OWNER" "$PILOT" 2>/dev/null || true' EXIT
SRC=$PILOT/job/src
mkdir -p "$SRC" "$PILOT/job/logs"
exec > >(tee -a "$PILOT/job/logs/render-$(date -u +%Y%m%dT%H%M%SZ)-$(hostname).log") 2>&1
cd "$SRC" && tar -xzf {archive}
RUNTIME_ENV={RUNTIME_ENV}
export PYTHONPATH="$SRC" PYTHONDONTWRITEBYTECODE=1 PATH="$RUNTIME_ENV/bin:$PATH" CUDA_VISIBLE_DEVICES=0
export VAGEN_GSPLAT_PREBUILT=1 TORCH_EXTENSIONS_DIR=/mnt/umm/users/yinbaiqiao/.cache/torch_extensions_jumpbox_renderer
export CC=$RUNTIME_ENV/bin/x86_64-conda-linux-gnu-gcc CXX=$RUNTIME_ENV/bin/x86_64-conda-linux-gnu-g++
export CUDA_HOME=$RUNTIME_ENV TORCH_CUDA_ARCH_LIST=9.0 OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1 NO_PROXY='*' no_proxy='*'
export CPATH="$RUNTIME_ENV/targets/x86_64-linux/include:${{CPATH:-}}"
export LIBRARY_PATH="$RUNTIME_ENV/targets/x86_64-linux/lib:${{LIBRARY_PATH:-}}"
nvidia-smi --query-gpu=name,memory.total --format=csv || true
$RUNTIME_ENV/bin/python -m data_gen.interiorgs_v10.{command} --pilot "$PILOT" --gs-root {gs_root}
""")
    launch.chmod(0o755)
    digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()  # noqa: E731
    (job / "SHA256SUMS").write_text(f"{digest(archive)}  source.tar.gz\n{digest(launch)}  launch.sh\n")
    (job / "prepared.json").write_text(json.dumps({"utc": now(), "gs_root": gs_root, "pool": POOL,
                                                   "worker_spec": WORKER_SPEC}, indent=1))
    print(json.dumps({"launch": str(launch)}))


def submit(pilot: Path) -> None:
    job = pilot / "job"
    receipt = job / "submission.json"
    if receipt.exists():
        print(json.loads(receipt.read_text())["job_id"])
        return
    subprocess.run(["sha256sum", "-c", "SHA256SUMS"], cwd=job, check=True)
    name = "interiorgs-v10-pilot-render-" + pilot.name.replace("_", "-")[-40:]
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


def describe(job_id: str) -> dict:
    res = subprocess.run([SCO, "acp", "jobs", "describe", "--workspace-name=aigc", "-o", "json", job_id],
                         capture_output=True, text=True, check=True)
    return json.loads(res.stdout)


def status(pilot: Path) -> None:
    job_id = json.loads((pilot / "job" / "submission.json").read_text())["job_id"]
    d = describe(job_id)
    print(json.dumps({"job_id": job_id, "state": d.get("state"), "start_time": d.get("start_time"),
                      "complete_time": d.get("complete_time"), "finished": d.get("state") in FINISHED}))


def stop(pilot: Path) -> None:
    job_id = json.loads((pilot / "job" / "submission.json").read_text())["job_id"]
    state = describe(job_id).get("state")
    if state in FINISHED:
        print(json.dumps({"job_id": job_id, "state": state, "action": "none"}))
        return
    res = subprocess.run([SCO, "acp", "jobs", "stop", "--workspace-name=aigc", job_id], capture_output=True, text=True)
    (pilot / "job" / "stop.log").write_text(res.stdout + "\n" + res.stderr)
    print(json.dumps({"job_id": job_id, "state_before": state, "stop_rc": res.returncode}))


def delete(pilot: Path) -> None:
    receipt = pilot / "job" / "submission.json"
    job_id = json.loads(receipt.read_text())["job_id"]
    state = describe(job_id).get("state")
    if state not in FINISHED | {"SUSPENDED"}:
        subprocess.run([SCO, "acp", "jobs", "stop", "--workspace-name=aigc", job_id], capture_output=True, text=True)
    res = subprocess.run([SCO, "acp", "jobs", "delete", "--workspace-name=aigc", job_id], capture_output=True, text=True)
    (pilot / "job" / "delete.log").write_text(res.stdout + "\n" + res.stderr)
    if res.returncode == 0:  # a later submit creates a new job
        receipt.rename(receipt.with_name(f"submission_{job_id}_deleted.json"))
    print(json.dumps({"job_id": job_id, "state_before": state, "delete_rc": res.returncode}))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=("prepare", "submit", "status", "stop", "delete"))
    ap.add_argument("--pilot", required=True, type=Path)
    ap.add_argument("--gs-root", default=None)
    ap.add_argument("--command", default="pilot render",
                    help="module and subcommand the job runs (default: render the frames of frames.jsonl)")
    args = ap.parse_args()
    if args.action == "prepare":
        if not args.gs_root:
            raise SystemExit("--gs-root is required for prepare")
        prepare(args.pilot.resolve(), args.gs_root, args.command)
    elif args.action == "submit":
        submit(args.pilot.resolve())
    elif args.action == "status":
        status(args.pilot.resolve())
    elif args.action == "stop":
        stop(args.pilot.resolve())
    else:
        delete(args.pilot.resolve())


if __name__ == "__main__":
    main()
