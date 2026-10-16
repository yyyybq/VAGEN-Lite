#!/usr/bin/env python3
"""Prepare, submit, and inspect the immutable Zoetrope R1 SFT generation job."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile
import tempfile


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "exps/vagen_active_spatial/R1-gate-aligned-sft-v2-20261008"
PACKAGE = RUN / "package"
CONTROL = RUN / "control"
BASE_ARCHIVE = (
    ROOT
    / "exps/vagen_active_spatial/R1-gate-aligned-pilot8-v2-20261005"
    / "package_v3/source.tar.gz"
)
SOURCE = ROOT / "exps/vagen_active_spatial/R1-clean-Projective-v0/frozen_v1/train.jsonl"
AUDIT = SOURCE.parent / "audit_only.jsonl"
ENV_YAML = (
    ROOT
    / "exps/vagen_active_spatial/R1-gate-aligned-pilot8-v2-20261005"
    / "frozen/train.yaml"
)
SCO = "/mnt/umm/users/yinbaiqiao/.sco/bin/sco"
IMAGE = "registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617"
MOUNT = "019ec9f9-6d12-7d49-aad4-864b15c9eb06:/mnt/umm"
JOB_NAME = "R1-gate-aligned-SFT210-v2"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def source_overlays() -> list[Path]:
    files = list((ROOT / "data_gen/active_spatial_sft").glob("*.py"))
    files.extend((ROOT / "vagen/envs/active_spatial").rglob("*.py"))
    files.append(ROOT / "examples/train/active_spatial/sco_r1_sft_generation.sh")
    return sorted(set(files))


def prepare() -> None:
    if RUN.exists():
        raise FileExistsError(f"immutable run already exists: {RUN}")
    if not BASE_ARCHIVE.is_file():
        raise FileNotFoundError(BASE_ARCHIVE)
    rows = [json.loads(line) for line in SOURCE.open() if line.strip()]
    audits = [json.loads(line) for line in AUDIT.open() if line.strip()]
    assert len(rows) == len(audits) == 210
    assert {row["task_id"] for row in rows} == {
        row["policy_task_id"] for row in audits
    }

    stage = Path(tempfile.mkdtemp(prefix=".r1-sft-", dir=RUN.parent))
    package = stage / "package"
    package.mkdir()
    with tarfile.open(BASE_ARCHIVE) as archive:
        contents = {
            member.name: archive.extractfile(member).read()
            for member in archive.getmembers()
            if member.isfile()
        }
    overlay_hashes = {}
    for path in source_overlays():
        relative = path.relative_to(ROOT).as_posix()
        value = path.read_bytes()
        if path.suffix == ".py":
            compile(value, relative, "exec")
        contents[relative] = value
        overlay_hashes[relative] = hashlib.sha256(value).hexdigest()

    archive_path = package / "source.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for name, value in sorted(contents.items()):
            member = tarfile.TarInfo(name)
            member.uid = member.gid = 20325
            member.uname = member.gname = "yinbaiqiao"
            member.mtime = 0
            member.mode = 0o755 if name.endswith(".sh") else 0o644
            member.size = len(value)
            archive.addfile(member, io.BytesIO(value))

    owner = ROOT / "examples/train/active_spatial/sco_run_as_artifact_owner.sh"
    (package / "owner.sh").write_bytes(owner.read_bytes())
    launch = f"""#!/usr/bin/env bash
set -euo pipefail
PACKAGE={RUN}/package
if [[ $(id -u) == 0 ]]; then
  exec bash "${{PACKAGE}}/owner.sh" bash "$0" owner
fi
[[ $(id -u) == 20325 && $(id -g) == 20325 ]] || exit 3
umask 022
cd "${{PACKAGE}}"
sha256sum -c SHA256SUMS
WORK=$(mktemp -d /tmp/r1_sft_launch.XXXXXXXX)
tar -xzf source.tar.gz -C "${{WORK}}"
cd "${{WORK}}"
exec timeout --signal=TERM --kill-after=120s 12h \
  bash examples/train/active_spatial/sco_r1_sft_generation.sh
"""
    (package / "launch.sh").write_text(launch)
    save(
        package / "provenance.json",
        {
            "role": "r1_gate_aligned_sft_generation_and_visualization",
            "created_utc": now(),
            "base_archive": str(BASE_ARCHIVE),
            "base_archive_sha256": digest(BASE_ARCHIVE),
            "source_sha256": digest(SOURCE),
            "audit_sha256": digest(AUDIT),
            "environment_yaml_sha256": digest(ENV_YAML),
            "overlays": overlay_hashes,
            "rows": 210,
            "render_backend": "local",
            "renderer_gpus": 1,
            "beam_width": 16,
            "all_primitive_frames": True,
            "qwen_exports": ["with_think_parquet", "no_think_parquet"],
        },
    )
    (package / "SHA256SUMS").write_text(
        "".join(
            f"{digest(path)}  {path.name}\n"
            for path in sorted(package.iterdir())
            if path.is_file() and path.name != "SHA256SUMS"
        )
    )
    package_sha256 = digest(archive_path)
    os.rename(stage, RUN)
    print(json.dumps({"run": str(RUN), "package_sha256": package_sha256}, indent=2))


def run_sco(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run([SCO, *arguments], capture_output=True, text=True)


def exact_named_jobs() -> list[dict]:
    result = run_sco(
        ["acp", "jobs", "list", "--workspace-name=aigc", "--page-size=500", "-o", "json"]
    )
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return [job for job in json.loads(result.stdout) if job.get("display_name") == JOB_NAME]


def submit() -> None:
    if not PACKAGE.is_dir():
        raise FileNotFoundError("run prepare before submit")
    subprocess.run(["sha256sum", "-c", "SHA256SUMS"], cwd=PACKAGE, check=True)
    CONTROL.mkdir(exist_ok=True)
    receipt = CONTROL / "submission.json"
    intent_path = CONTROL / "submission_intent.json"
    if receipt.exists():
        print(json.loads(receipt.read_text())["job_id"])
        return
    existing = exact_named_jobs()
    if existing:
        save(CONTROL / "existing_jobs.json", {"utc": now(), "jobs": existing})
        raise RuntimeError("exact-name SCO job already exists; inspect existing_jobs.json")
    if intent_path.exists():
        raise RuntimeError("prior submission intent has no receipt; inspect SCO before retry")
    command = [
        SCO,
        "acp", "jobs", "create",
        "--workspace-name=aigc",
        "--aec2-name=zoetrope",
        f"--job-name={JOB_NAME}",
        "--priority=HIGHEST",
        "--quota-type=reserved",
        f"--container-image-url={IMAGE}",
        f"--storage-mount={MOUNT}",
        "--training-framework=pytorch",
        "--worker-nodes=1",
        "--worker-spec=N4lS.Iq.I80.1",
        f"--command=bash {PACKAGE}/launch.sh",
    ]
    intent = {
        "utc": now(),
        "command": command,
        "package_sha256": digest(PACKAGE / "source.tar.gz"),
        "expected_output": str(RUN / "output"),
    }
    save(intent_path, intent)
    result = subprocess.run(command, capture_output=True, text=True)
    (CONTROL / "submit.log").write_text(result.stdout + "\n" + result.stderr)
    identifiers = set(re.findall(r"pt-[a-z0-9]+", result.stdout))
    if result.returncode != 0 or len(identifiers) != 1:
        raise RuntimeError("submission response uncertain; do not automatically resubmit")
    job_id = identifiers.pop()
    save(receipt, {**intent, "job_id": job_id, "state": "SUBMITTED"})
    print(job_id)


def status() -> None:
    receipt = CONTROL / "submission.json"
    if not receipt.is_file():
        raise FileNotFoundError("no submitted job receipt")
    job_id = json.loads(receipt.read_text())["job_id"]
    result = run_sco(
        ["acp", "jobs", "describe", "--workspace-name=aigc", "-o", "json", job_id]
    )
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    job = json.loads(result.stdout)
    save(CONTROL / "job_status.json", job)
    print(json.dumps({
        "job_id": job_id,
        "state": job.get("state"),
        "pool": (job.get("resource_pool") or {}).get("name"),
        "start_time": job.get("start_time"),
        "complete_time": job.get("complete_time"),
    }, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "submit", "status"))
    args = parser.parse_args()
    lock_parent = RUN if RUN.exists() else RUN.parent
    lock_parent.mkdir(parents=True, exist_ok=True)
    lock_path = lock_parent / ("control.lock" if RUN.exists() else ".r1_sft_prepare.lock")
    with lock_path.open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        {"prepare": prepare, "submit": submit, "status": status}[args.action]()


if __name__ == "__main__":
    main()
