#!/usr/bin/env python3
"""Fail-closed static preflight for the submitted S0/S1/S5 pilot package."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def _load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--package", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    run = Path(args.run).resolve()
    package = Path(args.package).resolve()
    frozen = run / "frozen"

    diagnosis = _load(run / "diagnostic" / "r5_production_backward.json")
    config_report = _load(run / "config_diff_report.json")
    preflight = _load(frozen / "preflight_report.json")
    task_context = _load(frozen / "r1_task_context_acceptance.json")
    runtime = _load(frozen / "r1_runtime_210.json")
    gate_pilot = _load(frozen / "r1_gate_aligned_training.json")
    paired = _load(frozen / "r1_paired_dev32.json")
    checks = {
        "real_r5_backward_pass": diagnosis.get("status") == "PASS",
        "real_r5_no_optimizer_step": diagnosis.get("safety", {}).get("optimizer_step_called") is False,
        "real_r5_no_checkpoint_write": diagnosis.get("safety", {}).get("checkpoint_written") is False,
        "real_r5_actor_parameter_identity": diagnosis.get("actor", {}).get("parameters_unchanged") is True,
        "real_r5_critic_parameter_identity": diagnosis.get("critic", {}).get("parameters_unchanged") is True,
        "real_r5_actor_forward_exact": diagnosis.get("actor", {}).get("old_logprob_parity", {}).get("max_abs_error") == 0.0,
        "real_r5_critic_forward_exact": diagnosis.get("critic", {}).get("saved_value_parity", {}).get("max_abs_error") == 0.0,
        "real_r5_historical_gae_exact": diagnosis.get("historical_no_concat_gae_max_abs_error") == 0.0,
        "reward_only_config_diff_pass": config_report.get("status") == "PASS" and all(config_report.get("checks", {}).values()),
        "split_scorer_preflight_pass": preflight.get("status") == "PASS",
        "r1_task_context_pass": task_context.get("status") == "PASS",
        "r1_runtime_210_pass": runtime.get("status") == "PASS",
        "r1_gate_aligned_training_pass": gate_pilot.get("status") == "PASS",
        "r1_paired_dev32_pass": paired.get("status") == "PASS" and paired.get("expanded_training_allowed") is True,
    }
    checksum = subprocess.run(
        ["sha256sum", "-c", "SHA256SUMS"], cwd=frozen, text=True, capture_output=True, check=False
    )
    package_checksum = subprocess.run(
        ["sha256sum", "-c", "SHA256SUMS"], cwd=package, text=True, capture_output=True, check=False
    )
    checks["frozen_checksums_pass"] = checksum.returncode == 0
    checks["package_checksums_pass"] = package_checksum.returncode == 0
    checks["renderer_and_training_launchers_distinct"] = (package / "launch_renderer.sh").read_bytes() != (package / "launch_training.sh").read_bytes()
    status = "PASS" if all(checks.values()) else "BLOCKED"
    report = {
        "schema_version": "active_spatial_dense_score_final_static_preflight_v1",
        "status": status,
        "checks": checks,
        "resource_topology": {
            "cluster": "zoetrope",
            "renderer": {"nodes": 1, "worker_spec": "N4lS.Iq.I80.1", "embedded_in_training": False},
            "training": {"nodes": 1, "worker_spec": "N4lS.Iq.I80.8", "branches": ["S0", "S1", "S5"], "serial": True},
        },
        "artifacts": {
            "diagnosis": {"path": str(run / "diagnostic" / "r5_production_backward.json"), "sha256": _sha(run / "diagnostic" / "r5_production_backward.json")},
            "config_diff": {"path": str(run / "config_diff_report.json"), "sha256": _sha(run / "config_diff_report.json")},
            "frozen_checksums": checksum.stdout,
            "package_checksums": package_checksum.stdout,
        },
        "submission_allowed": status == "PASS",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
