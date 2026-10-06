#!/usr/bin/env python3
"""Idempotent Zoetrope submission controller for the gate-aligned pilot."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess
from datetime import datetime, timezone


SCO = "/mnt/umm/users/yinbaiqiao/.sco/bin/sco"
RUN = Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/R1-gate-aligned-pilot8-v2-20261005")
CONTROL = RUN / "control"
IMAGE = "registry.cn-fz-01.fjscms.com/ccr_fj2/wc-dev:260617"
MOUNT = "019ec9f9-6d12-7d49-aad4-864b15c9eb06:/mnt/umm"
TERMINAL = {"FAILED", "SUCCEEDED", "STOPPED", "DELETED", "CANCELED", "CANCELLED"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def run_sco(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run([SCO, *arguments], capture_output=True, text=True)


def exact_named_jobs(display_name: str) -> list[dict]:
    result = run_sco(
        ["acp", "jobs", "list", "--workspace-name=aigc", "--page-size=500", "-o", "json"]
    )
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return [job for job in json.loads(result.stdout) if job.get("display_name") == display_name]


def verify_local_inputs(package_name: str = "package_v3") -> None:
    replay = json.loads((RUN / "gate_aligned_reward_replay.json").read_text())
    if replay["status"] != "PASS" or not all(replay["checks"].values()):
        raise RuntimeError("offline replay gate is not PASS")
    subprocess.run(["sha256sum", "-c", "SHA256SUMS"], cwd=RUN / "frozen", check=True)
    subprocess.run(["sha256sum", "-c", "SHA256SUMS"], cwd=RUN / package_name, check=True)


def create_job(*, display_name: str, spec: str, launch: Path, role: str) -> str:
    receipt = CONTROL / f"{role}_submission.json"
    intent_path = CONTROL / f"{role}_submission_intent.json"
    if receipt.exists():
        value = json.loads(receipt.read_text())
        print(value["job_id"])
        return value["job_id"]
    existing = exact_named_jobs(display_name)
    if existing:
        save(
            CONTROL / f"{role}_existing_job.json",
            {"utc": now(), "jobs": existing},
        )
        raise RuntimeError(f"exact-name SCO job already exists; inspect {role}_existing_job.json")
    if intent_path.exists():
        raise RuntimeError(f"prior {role} intent has no receipt; inspect SCO before retry")
    command = [
        SCO,
        "acp",
        "jobs",
        "create",
        "--workspace-name=aigc",
        "--aec2-name=zoetrope",
        f"--job-name={display_name}",
        "--priority=HIGHEST",
        "--quota-type=reserved",
        f"--container-image-url={IMAGE}",
        f"--storage-mount={MOUNT}",
        "--training-framework=pytorch",
        "--worker-nodes=1",
        f"--worker-spec={spec}",
        f"--command=bash {launch}",
    ]
    intent = {
        "utc": now(),
        "role": role,
        "command": command,
        "package_sha256": hashlib.sha256((launch.parent / "source.tar.gz").read_bytes()).hexdigest(),
        "offline_replay_sha256": hashlib.sha256((RUN / "gate_aligned_reward_replay.json").read_bytes()).hexdigest(),
    }
    save(intent_path, intent)
    result = subprocess.run(command, capture_output=True, text=True)
    (CONTROL / f"{role}_submit.log").write_text(result.stdout + "\n" + result.stderr)
    identifiers = set(re.findall(r"pt-[a-z0-9]+", result.stdout))
    if result.returncode != 0 or len(identifiers) != 1:
        raise RuntimeError("submission response uncertain; do not automatically resubmit")
    job_id = identifiers.pop()
    save(receipt, {**intent, "job_id": job_id, "state": "SUBMITTED"})
    print(job_id)
    return job_id


def describe(job_id: str) -> dict:
    result = run_sco(
        ["acp", "jobs", "describe", "--workspace-name=aigc", "-o", "json", job_id]
    )
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return json.loads(result.stdout)


def submit_renderer() -> None:
    verify_local_inputs()
    create_job(
        display_name="R1-gate-aligned-pilot8-v2-renderer-r3",
        spec="N4lS.Iq.I80.1",
        launch=RUN / "package_v3/launch_renderer.sh",
        role="renderer_r3",
    )


def submit_training(renderer_job: str) -> None:
    verify_local_inputs()
    renderer = describe(renderer_job)
    save(CONTROL / "renderer_job.json", renderer)
    if renderer["resource_pool"]["name"] != "zoetrope":
        raise RuntimeError("renderer is not in zoetrope")
    if renderer["state"] in TERMINAL:
        raise RuntimeError(f"renderer already terminated: {renderer['state']}")
    gate_path = RUN / "runtime_preflight.json"
    endpoint_path = RUN / "renderer_endpoint.txt"
    if not gate_path.exists() or not endpoint_path.exists():
        raise RuntimeError("renderer endpoint/runtime preflight is not ready")
    gate = json.loads(gate_path.read_text())
    if gate.get("status") != "PASS" or gate.get("completed") != 12:
        raise RuntimeError("runtime preflight is not PASS 12/12")
    expected = hashlib.sha256((RUN / "frozen/train.jsonl").read_bytes()).hexdigest()
    if gate.get("manifest_sha256") != expected:
        raise RuntimeError("runtime preflight is for a different frozen manifest")
    create_job(
        display_name="R1-gate-aligned-pilot8-v2-train",
        spec="N4lS.Iq.I80.8",
        launch=RUN / "package_v3/launch_training.sh",
        role="training",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("renderer", "training"))
    parser.add_argument("--renderer-job")
    args = parser.parse_args()
    CONTROL.mkdir(exist_ok=True)
    with (CONTROL / f"{args.mode}.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.mode == "renderer":
            submit_renderer()
        else:
            if not args.renderer_job:
                parser.error("--renderer-job is required for training")
            submit_training(args.renderer_job)


if __name__ == "__main__":
    main()
