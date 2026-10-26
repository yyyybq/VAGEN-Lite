#!/usr/bin/env python3
"""Prepare / submit / inspect a cluster job that runs the v10 generator on all scenes.

Mirrors scripts/r1_submit_sft_generation.py (same sco CLI, workspace, image,
mount and worker spec).  The generator is CPU-only (no renderer needed); the
worker's CPUs are used through a process pool.  Rendering + RGB/VLM review of
the produced tasks is a separate GPU stage (see README).

    python data_gen/interiorgs_v10/sco_interiorgs_v10.py prepare --tag 20261011
    python data_gen/interiorgs_v10/sco_interiorgs_v10.py submit  --tag 20261011
    python data_gen/interiorgs_v10/sco_interiorgs_v10.py status  --tag 20261011

Nothing is submitted by `prepare`.  `submit` refuses to run twice for one tag.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path

from vagen import config as _config

ROOT = Path(__file__).resolve().parents[2]
# Metadata mirror of all 1,000 scenes (fetch_interiorgs_metadata --source aoss). The older
# /mnt/umm/users/yinbaiqiao/InteriorGS holds only 100 scenes and is not readable by this user.
SCENES_ROOT = Path(_config.path("scenes_meta"))
ENV_PY = Path(_config.path("conda_env")) / "bin" / "python"
SCO = _config.cluster("sco.bin")
IMAGE = _config.cluster("sco.image")
MOUNT = _config.cluster("sco.storage_mount")
WORKER_SPEC = _config.cluster("sco.worker_specs")[1]
POOL = _config.cluster("sco.default_pool")
WORKSPACE = _config.cluster("sco.workspace")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_dir(tag: str) -> Path:
    return ROOT / "exps/vagen_active_spatial" / f"interiorgs_v10_gen_{tag}"


def prepare(tag: str, workers: int, target: int, scenes_root: Path = SCENES_ROOT) -> None:
    run = run_dir(tag)
    package = run / "package"
    if package.exists():
        raise SystemExit(f"refusing existing package: {package}")
    package.mkdir(parents=True)
    archive = package / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        # whole vagen package: vagen.envs.active_spatial/__init__ imports vagen.envs.gym_image_env & co.
        for rel in ("data_gen/interiorgs_v10", "data_gen/active_spatial_pipeline", "vagen"):
            src = ROOT / rel
            if src.exists():
                tar.add(src, arcname=rel, filter=lambda ti: None if "__pycache__" in ti.name else ti)
    launch = package / "launch.sh"
    launch.write_text(f"""#!/usr/bin/env bash
set -euo pipefail
RUN={run}
SRC=$RUN/src
OUT=$RUN/output
# the job runs as root; hand the run directory back to the submitting user however the job ends
OWNER=$(stat -c %u:%g "$RUN/package")
trap 'chown -R "$OWNER" "$RUN" 2>/dev/null || true' EXIT
mkdir -p "$SRC" "$OUT" "$RUN/logs"
exec > >(tee -a "$RUN/logs/worker-$(date -u +%Y%m%dT%H%M%SZ)-$(hostname).log") 2>&1
cd "$SRC" && tar -xzf {archive}
# count CPUs before exporting OMP_NUM_THREADS: GNU nproc honours it and would return 1. Inside the
# job nproc reports the whole host (128) while the worker spec grants a CFS quota (14 vCPU for I80.1).
NPROC=$(nproc)
if [[ -r /sys/fs/cgroup/cpu.max ]]; then read -r Q P < /sys/fs/cgroup/cpu.max
elif [[ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ]]; then
  Q=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us); P=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us); fi
if [[ ${{Q:-max}} =~ ^[0-9]+$ ]] && (( Q > 0 )); then QC=$(( (Q + P - 1) / P )); (( QC < NPROC )) && NPROC=$QC; fi
W={workers}; [[ $W -gt 0 ]] || W=$(( NPROC > 1 ? NPROC - 1 : 1 ))
echo "nproc=$(nproc) cpu_quota=${{Q:-none}}/${{P:-none}} cpus=$NPROC workers=$W"
export PYTHONPATH="$SRC" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NO_PROXY='*' no_proxy='*'
# fail fast when the packaged source cannot load a scene (otherwise every scene just records an error)
{ENV_PY} -c "from pathlib import Path; from data_gen.interiorgs_v10.scene import load_scene; \\
d = sorted(Path('{scenes_root}').glob('*/labels.json'))[0].parent; s = load_scene(d, d.name); \\
print('preflight ok', d.name, len(s.objects), s.convention_v2.get('status'))"
{ENV_PY} -m data_gen.interiorgs_v10.coordinates --scenes-root {scenes_root} --out "$OUT/calibration_report.json" || true
{ENV_PY} -m data_gen.interiorgs_v10.objects --scenes-root {scenes_root} --out "$OUT/object_stats.json" || true
{ENV_PY} -m data_gen.interiorgs_v10.run_all --scenes-root {scenes_root} --out "$OUT" --workers "$W" \\
    --target-per-scene {target} --expected-scenes 1000
{ENV_PY} -c "import json, sys; s = json.load(open('$OUT/summary.json')); \\
print('scenes_with_tasks', s['scenes_with_tasks'], 'tasks_total', s['tasks_total']); \\
sys.exit(0 if s['scenes_with_tasks'] > 0 else 1)"
""")
    launch.chmod(0o755)
    (package / "SHA256SUMS").write_text(f"{digest(archive)}  source.tar.gz\n{digest(launch)}  launch.sh\n")
    (run / "control").mkdir(exist_ok=True)
    (run / "control" / "prepared.json").write_text(json.dumps({"utc": now(), "tag": tag, "workers": workers,
                                                               "target_per_scene": target,
                                                               "scenes_root": str(scenes_root)}, indent=2))
    print(json.dumps({"run": str(run), "launch": str(launch)}, indent=2))


def submit(tag: str) -> None:
    run = run_dir(tag)
    package = run / "package"
    control = run / "control"
    receipt = control / "submission.json"
    if receipt.exists():
        print(json.loads(receipt.read_text())["job_id"])
        return
    subprocess.run(["sha256sum", "-c", "SHA256SUMS"], cwd=package, check=True)
    job_name = f"interiorgs-v10-gen-{tag}"
    command = [SCO, "acp", "jobs", "create", f"--workspace-name={WORKSPACE}", f"--aec2-name={POOL}",
               f"--job-name={job_name}", "--priority=HIGHEST", "--quota-type=reserved",
               f"--container-image-url={IMAGE}", f"--storage-mount={MOUNT}", "--training-framework=pytorch",
               "--worker-nodes=1", f"--worker-spec={WORKER_SPEC}", f"--command=bash {package}/launch.sh"]
    intent = {"utc": now(), "command": command}
    (control / "submission_intent.json").write_text(json.dumps(intent, indent=2))
    result = subprocess.run(command, capture_output=True, text=True)
    (control / "submit.log").write_text(result.stdout + "\n" + result.stderr)
    ids = set(re.findall(r"pt-[a-z0-9]+", result.stdout))
    if result.returncode != 0 or len(ids) != 1:
        raise SystemExit("submission response uncertain; inspect control/submit.log before any retry")
    job_id = ids.pop()
    receipt.write_text(json.dumps({**intent, "job_id": job_id, "state": "SUBMITTED"}, indent=2))
    print(job_id)


def status(tag: str) -> None:
    receipt = run_dir(tag) / "control" / "submission.json"
    job_id = json.loads(receipt.read_text())["job_id"]
    result = subprocess.run([SCO, "acp", "jobs", "describe", f"--workspace-name={WORKSPACE}", "-o", "json", job_id],
                            capture_output=True, text=True, check=True)
    job = json.loads(result.stdout)
    print(json.dumps({"job_id": job_id, "state": job.get("state"), "start_time": job.get("start_time"),
                      "complete_time": job.get("complete_time")}, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=("prepare", "submit", "status"))
    ap.add_argument("--tag", required=True)
    ap.add_argument("--workers", type=int, default=0, help="0 = nproc - 1")
    ap.add_argument("--target-per-scene", type=int, default=14)
    ap.add_argument("--scenes-root", type=Path, default=SCENES_ROOT)
    args = ap.parse_args()
    if args.action == "prepare":
        if not any(args.scenes_root.glob("*/labels.json")):
            raise SystemExit(f"no <scene>/labels.json under {args.scenes_root}")
        prepare(args.tag, args.workers, args.target_per_scene, args.scenes_root)
    elif args.action == "submit":
        submit(args.tag)
    else:
        status(args.tag)


if __name__ == "__main__":
    main()
