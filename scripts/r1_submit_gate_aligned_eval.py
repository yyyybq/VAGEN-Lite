#!/usr/bin/env python3
"""Idempotent Zoetrope submission controller for fixed-32 pilot evaluation."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import re
import subprocess


SCO = "/mnt/umm/users/yinbaiqiao/.sco/bin/sco"
RUN = Path(
    "/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/"
    "R1-gate-aligned-pilot8-v2-20261005"
)
PACKAGE = RUN / "eval_package_v4"
PREFLIGHT_ROOT = RUN / "eval_preflight_v2"
RENDER_ENDPOINT = RUN / "eval_renderer_endpoint_v2.txt"
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


def describe(job_id: str) -> dict:
    result = run_sco(
        ["acp", "jobs", "describe", "--workspace-name=aigc", "-o", "json", job_id]
    )
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return json.loads(result.stdout)


def exact_named_jobs(display_name: str) -> list[dict]:
    result = run_sco(
        ["acp", "jobs", "list", "--workspace-name=aigc", "--page-size=500", "-o", "json"]
    )
    if result.returncode:
        raise RuntimeError(result.stdout + result.stderr)
    return [job for job in json.loads(result.stdout) if job.get("display_name") == display_name]


def verify_training_gate() -> None:
    gate_path = RUN / "pilot_training_gate.json"
    if not gate_path.is_file():
        raise RuntimeError("pilot training gate does not exist")
    gate = json.loads(gate_path.read_text())
    if gate.get("status") != "PASS" or gate.get("updates") != 8:
        raise RuntimeError("pilot training gate is not PASS at update 8")
    for step in (2, 4, 8):
        checkpoint = RUN / f"pilot/checkpoints/global_step_{step}"
        model = checkpoint / "actor/huggingface/model.safetensors.index.json"
        if not (checkpoint / "COMPLETE").is_file() or not model.is_file():
            raise RuntimeError(f"step {step} checkpoint is not atomically complete")
        if not (RUN / f"step{step}_SHA256SUMS").is_file():
            raise RuntimeError(f"step {step} model hashes are missing")


def verify_local_inputs() -> None:
    replay = json.loads((RUN / "gate_aligned_reward_replay.json").read_text())
    if replay.get("status") != "PASS" or not all(replay["checks"].values()):
        raise RuntimeError("offline reward replay gate is not PASS")
    verify_training_gate()
    subprocess.run(["sha256sum", "-c", "SHA256SUMS"], cwd=RUN / "frozen", check=True)
    subprocess.run(["sha256sum", "-c", "SHA256SUMS"], cwd=PACKAGE, check=True)


def verify_preflight() -> None:
    for key in ("base", "step2", "step4", "step8"):
        marker = PREFLIGHT_ROOT / key / "preflight.json"
        if not marker.is_file():
            raise RuntimeError(f"{key} vLLM preflight is missing")
        report = json.loads(marker.read_text())
        if (
            report.get("status") != "PASS"
            or not report.get("sampled_multimodal_inference")
            or not report.get("task_context_check", {}).get("task_present")
        ):
            raise RuntimeError(f"{key} vLLM preflight is not PASS")


def create_job(*, display_name: str, spec: str, launch: Path, role: str) -> str:
    receipt = CONTROL / f"{role}_submission.json"
    intent_path = CONTROL / f"{role}_submission_intent.json"
    if receipt.exists():
        value = json.loads(receipt.read_text())
        print(value["job_id"])
        return value["job_id"]
    existing = exact_named_jobs(display_name)
    if existing:
        save(CONTROL / f"{role}_existing_job.json", {"utc": now(), "jobs": existing})
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
        "package_sha256": hashlib.sha256((PACKAGE / "source.tar.gz").read_bytes()).hexdigest(),
        "training_gate_sha256": hashlib.sha256((RUN / "pilot_training_gate.json").read_bytes()).hexdigest(),
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


def submit_preflight() -> None:
    verify_local_inputs()
    create_job(
        display_name="R1-gate-aligned-pilot8-v2-eval2-preflight",
        spec="N4lS.Iq.I80.1",
        launch=PACKAGE / "launch_preflight.sh",
        role="eval2_preflight",
    )


def submit_renderer(preflight_job: str) -> None:
    verify_local_inputs()
    preflight = describe(preflight_job)
    save(CONTROL / "eval2_preflight_job.json", preflight)
    if preflight["resource_pool"]["name"] != "zoetrope":
        raise RuntimeError("evaluation preflight is not in zoetrope")
    if preflight["state"] != "SUCCEEDED":
        raise RuntimeError(f"evaluation preflight has not succeeded: {preflight['state']}")
    verify_preflight()
    create_job(
        display_name="R1-gate-aligned-pilot8-v2-eval2-renderer",
        spec="N4lS.Iq.I80.1",
        launch=PACKAGE / "launch_renderer.sh",
        role="eval2_renderer",
    )


def submit_evaluation(renderer_job: str) -> None:
    verify_local_inputs()
    verify_preflight()
    renderer = describe(renderer_job)
    save(CONTROL / "eval2_renderer_job.json", renderer)
    if renderer["resource_pool"]["name"] != "zoetrope":
        raise RuntimeError("evaluation renderer is not in zoetrope")
    if renderer["state"] in TERMINAL:
        raise RuntimeError(f"evaluation renderer already terminated: {renderer['state']}")
    if not RENDER_ENDPOINT.is_file():
        raise RuntimeError("evaluation renderer endpoint is not ready")
    create_job(
        display_name="R1-gate-aligned-pilot8-v2-fixed32-eval2",
        spec="N4lS.Iq.I80.1",
        launch=PACKAGE / "launch_evaluation.sh",
        role="eval2_run",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "renderer", "evaluation"))
    parser.add_argument("--preflight-job")
    parser.add_argument("--renderer-job")
    args = parser.parse_args()
    CONTROL.mkdir(exist_ok=True)
    with (CONTROL / f"eval2_{args.mode}.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.mode == "preflight":
            submit_preflight()
        elif args.mode == "renderer":
            if not args.preflight_job:
                parser.error("--preflight-job is required for renderer")
            submit_renderer(args.preflight_job)
        else:
            if not args.renderer_job:
                parser.error("--renderer-job is required for evaluation")
            submit_evaluation(args.renderer_job)


if __name__ == "__main__":
    main()
