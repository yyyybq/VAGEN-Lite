"""Run a v10 generation as shards: some on this node, the rest as sco single-card jobs (zoetrope).

The generator is CPU-only; a single-card job is the smallest spec the pool offers (14 vCPU), so each shard
job uses its CPUs through ``run_all --workers`` and leaves the card idle.  All shards write into one
output directory; ``run_all`` skips scenes already done, so a final local pass with every shard done only
assembles the outputs.

    python data_gen/interiorgs_v10/shard_sco.py prepare --out OUT --config CFG --num-shards 5 --job-shards 1,2,3,4
    python data_gen/interiorgs_v10/shard_sco.py submit  --out OUT
    python data_gen/interiorgs_v10/shard_sco.py status  --out OUT
    python data_gen/interiorgs_v10/shard_sco.py delete  --out OUT    # after the jobs end (or to cancel)

``prepare`` freezes the code into OUT/job/source.tar.gz and writes one launch script per job shard; the
jobs run as root and hand every output back to the submitting user.
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
from typing import Dict, List

from vagen import config as _config

ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ENV = _config.path("conda_env")
SCO = _config.cluster("sco.bin")
IMAGE = _config.cluster("sco.image")
MOUNT = _config.cluster("sco.storage_mount")
POOL = _config.cluster("sco.default_pool")
WORKSPACE = _config.cluster("sco.workspace")
WORKER_SPEC = _config.cluster("sco.worker_specs")[1]  # one H800, 14 vCPU
SCENES_ROOT = _config.path("scenes_meta")
FINISHED = {"SUCCEEDED", "FAILED", "STOPPED", "CANCELLED"}


def now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def prepare(out: Path, config: str, num_shards: int, job_shards: List[int], workers: int = 13,
            scenes_root: str = SCENES_ROOT, job_dir: str = "job") -> Dict[str, str]:
    """``job_dir``: a second set of jobs into the same OUT/output gets its own folder (e.g. re-splitting a shard
    that runs too slowly elsewhere: shards 0, 5, 10, 15 of 20 are exactly shard 0 of 5)."""
    if not job_shards or any(not 0 <= s < num_shards for s in job_shards):
        raise ValueError(f"job shards {job_shards} must lie in 0..{num_shards - 1}")
    job = out / job_dir
    job.mkdir(parents=True, exist_ok=True)
    archive = job / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for rel in ("data_gen/interiorgs_v10", "vagen"):
            tar.add(ROOT / rel, arcname=rel,
                    filter=lambda ti: None if "__pycache__" in ti.name or "figures_cabinet" in ti.name else ti)
    scripts = {}
    for shard in job_shards:
        launch = job / f"launch_shard{shard}.sh"
        launch.write_text(f"""#!/usr/bin/env bash
set -uo pipefail
OUT={out}
OWNER=$(stat -c %u:%g "{archive}")
trap 'chown -R "$OWNER" "$OUT" 2>/dev/null || true' EXIT
SRC={job}/src_shard{shard}
mkdir -p "$SRC" "{job}/logs"
exec > >(tee -a "{job}/logs/shard{shard}-$(date -u +%Y%m%dT%H%M%SZ)-$(hostname).log") 2>&1
cd "$SRC" && tar -xzf {archive}
export PYTHONPATH="$SRC" PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1
{RUNTIME_ENV}/bin/python -m data_gen.interiorgs_v10.run_all --scenes-root {scenes_root} --out "$OUT/output" \\
    --workers {workers} --shard {shard} --num-shards {num_shards} --expected-scenes 1000 --config {config} \\
    --diagnostics
""")
        launch.chmod(0o755)
        scripts[str(shard)] = str(launch)
    sums = [f"{hashlib.sha256(Path(p).read_bytes()).hexdigest()}  {Path(p).name}" for p in [archive, *scripts.values()]]
    (job / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    (job / "prepared.json").write_text(json.dumps({"utc": now(), "config": config, "num_shards": num_shards,
                                                   "job_shards": job_shards, "workers": workers,
                                                   "pool": POOL, "worker_spec": WORKER_SPEC}, indent=1))
    return scripts


def submit(out: Path, job_dir: str = "job") -> Dict[str, str]:
    job = out / job_dir
    receipt = job / "submission.json"
    if receipt.exists():
        return json.loads(receipt.read_text())["jobs"]
    subprocess.run(["sha256sum", "-c", "SHA256SUMS"], cwd=job, check=True)
    prepared = json.loads((job / "prepared.json").read_text())
    jobs: Dict[str, str] = {}
    for shard in prepared["job_shards"]:
        name = f"interiorgs-v10-gen-{out.name}-{job_dir}-s{shard}".replace("_", "-")[-60:]
        command = [SCO, "acp", "jobs", "create", f"--workspace-name={WORKSPACE}", f"--aec2-name={POOL}",
                   f"--job-name={name}", "--priority=HIGHEST", "--quota-type=reserved",
                   f"--container-image-url={IMAGE}", f"--storage-mount={MOUNT}", "--training-framework=pytorch",
                   "--worker-nodes=1", f"--worker-spec={WORKER_SPEC}",
                   f"--command=bash {job}/launch_shard{shard}.sh"]
        res = subprocess.run(command, capture_output=True, text=True)
        (job / f"submit_shard{shard}.log").write_text(res.stdout + "\n" + res.stderr)
        ids = set(re.findall(r"pt-[a-z0-9]+", res.stdout))
        if res.returncode != 0 or len(ids) != 1:
            receipt.write_text(json.dumps({"utc": now(), "jobs": jobs, "failed_shard": shard}, indent=1))
            raise SystemExit(f"submission of shard {shard} uncertain; read job/submit_shard{shard}.log")
        jobs[str(shard)] = ids.pop()
    receipt.write_text(json.dumps({"utc": now(), "jobs": jobs}, indent=1))
    return jobs


def describe(job_id: str) -> dict:
    res = subprocess.run([SCO, "acp", "jobs", "describe", f"--workspace-name={WORKSPACE}", "-o", "json", job_id],
                         capture_output=True, text=True)
    if res.returncode != 0:
        return {"state": "NOT_FOUND"}
    return json.loads(res.stdout)


def status(out: Path, job_dir: str = "job") -> Dict[str, str]:
    jobs = json.loads((out / job_dir / "submission.json").read_text())["jobs"]
    return {shard: f"{jid}:{describe(jid).get('state')}" for shard, jid in jobs.items()}


def delete(out: Path, job_dir: str = "job") -> Dict[str, int]:
    receipt = out / job_dir / "submission.json"
    jobs = json.loads(receipt.read_text())["jobs"]
    codes = {}
    for shard, jid in jobs.items():
        state = describe(jid).get("state")
        if state == "NOT_FOUND":
            codes[shard] = 0
            continue
        if state not in FINISHED | {"SUSPENDED"}:
            subprocess.run([SCO, "acp", "jobs", "stop", f"--workspace-name={WORKSPACE}", jid], capture_output=True, text=True)
        res = subprocess.run([SCO, "acp", "jobs", "delete", f"--workspace-name={WORKSPACE}", jid], capture_output=True,
                             text=True)
        codes[shard] = res.returncode
    if all(c == 0 for c in codes.values()):
        receipt.rename(receipt.with_name("submission_deleted.json"))
    return codes


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=("prepare", "submit", "status", "delete"))
    ap.add_argument("--out", required=True, type=Path, help="run directory (outputs go to OUT/output)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--num-shards", type=int, default=0)
    ap.add_argument("--job-shards", default="", help="comma-separated shards to run as sco jobs")
    ap.add_argument("--workers", type=int, default=13)
    ap.add_argument("--job-dir", default="job", help="folder under OUT for this set of jobs")
    args = ap.parse_args()
    out = args.out.resolve()
    if args.action == "prepare":
        if not args.config or args.num_shards <= 0:
            raise SystemExit("prepare needs --config and --num-shards")
        shards = [int(s) for s in args.job_shards.split(",") if s.strip()]
        print(json.dumps(prepare(out, args.config, args.num_shards, shards, args.workers,
                                 job_dir=args.job_dir), indent=1))
    elif args.action == "submit":
        print(json.dumps(submit(out, args.job_dir)))
    elif args.action == "status":
        print(json.dumps(status(out, args.job_dir)))
    else:
        print(json.dumps(delete(out, args.job_dir)))


if __name__ == "__main__":
    main()
