#!/usr/bin/env python3
"""Fail-closed static preflight for the parallel S0/S1/S5 pilot package."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


BRANCHES = ("S0", "S1", "S5")


def load(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--renderer-benchmark", required=True, type=Path)
    args = parser.parse_args()
    run = args.run.resolve()
    package = args.package.resolve()
    frozen = run / "frozen"
    diagnosis = load(run / "diagnostic" / "r5_production_backward.json")
    config_report = load(run / "config_diff_report.json")
    preflight = load(frozen / "preflight_report.json")
    task_context = load(frozen / "r1_task_context_acceptance.json")
    runtime = load(frozen / "r1_runtime_210.json")
    gate_pilot = load(frozen / "r1_gate_aligned_training.json")
    paired = load(frozen / "r1_paired_dev32.json")
    renderer_benchmark = load(args.renderer_benchmark.resolve())
    entry = (package / "overlay_source" / "examples/train/active_spatial/sco_dense_score_reward_only_parallel.sh").read_text(encoding="utf-8")
    checks = {
        "real_r5_backward_pass": diagnosis.get("status") == "PASS",
        "real_r5_no_optimizer_step": diagnosis.get("safety", {}).get("optimizer_step_called") is False,
        "real_r5_parameter_identity": diagnosis.get("actor", {}).get("parameters_unchanged") is True and diagnosis.get("critic", {}).get("parameters_unchanged") is True,
        "reward_only_config_diff_pass": config_report.get("status") == "PASS" and all(config_report.get("checks", {}).values()),
        "split_scorer_preflight_pass": preflight.get("status") == "PASS",
        "r1_task_context_pass": task_context.get("status") == "PASS",
        "r1_runtime_210_pass": runtime.get("status") == "PASS",
        "r1_gate_aligned_training_pass": gate_pilot.get("status") == "PASS",
        "r1_paired_dev32_pass": paired.get("status") == "PASS" and paired.get("expanded_training_allowed") is True,
        "three_branch_configs": all((frozen / "configs" / f"{branch}.yaml").is_file() for branch in BRANCHES),
        "three_independent_renderer_launchers": all((package / f"launch_renderer_{branch}.sh").is_file() for branch in BRANCHES),
        "three_independent_training_launchers": all((package / f"launch_training_{branch}.sh").is_file() for branch in BRANCHES),
        "renderer_benchmark_pass": renderer_benchmark.get("status") == "PASS"
        and renderer_benchmark.get("training_submission_allowed") is True,
        "renderer_benchmark_selected_1gpu_12workers": renderer_benchmark.get("selected_topology")
        == {"gpu_count": 1, "max_workers": 12, "max_inflight": 12},
        "renderer_1gpu_12workers": "--max-workers 12 --max-inflight 12" in entry and "--gpus 0 " in entry,
        "renderer_jit_cache_preseeded": "VALIDATED_GSPLAT_CACHE" in entry and "gsplat_cuda.so" in entry,
        "renderer_real_smoke_gate": "real_render_count" in entry and "real_render_smoke" in entry,
        "branch_timeout_repaired": " 7d " in entry and " 30h " not in entry,
        "branch_isolation": "BRANCH_ROOT=${RUN}/branches/${BRANCH}" in entry,
    }
    frozen_checksum = subprocess.run(["sha256sum", "-c", "SHA256SUMS"], cwd=frozen, text=True, capture_output=True)
    package_checksum = subprocess.run(["sha256sum", "-c", "SHA256SUMS"], cwd=package, text=True, capture_output=True)
    checks["frozen_checksums_pass"] = frozen_checksum.returncode == 0
    checks["package_checksums_pass"] = package_checksum.returncode == 0
    status = "PASS" if all(checks.values()) else "BLOCKED"
    report = {
        "schema_version": "active_spatial_dense_score_parallel_preflight_v1",
        "status": status,
        "submission_allowed": status == "PASS",
        "checks": checks,
        "resource_topology": {
            "cluster": "zoetrope",
            "branches": list(BRANCHES),
            "execution": "parallel",
            "per_branch": {
                "renderer": {"nodes": 1, "worker_spec": "N4lS.Iq.I80.1", "max_workers": 12, "max_inflight": 12},
                "training": {"nodes": 1, "worker_spec": "N4lS.Iq.I80.8", "agent_loop_workers": 48},
            },
            "total_h800": 27,
        },
        "timeout": {"per_branch": "7d", "package": "8d", "supersedes": "30h"},
        "artifacts": {
            "diagnosis": {"path": str(run / "diagnostic" / "r5_production_backward.json"), "sha256": sha(run / "diagnostic" / "r5_production_backward.json")},
            "config_diff": {"path": str(run / "config_diff_report.json"), "sha256": sha(run / "config_diff_report.json")},
            "renderer_benchmark": {"path": str(args.renderer_benchmark.resolve()), "sha256": sha(args.renderer_benchmark.resolve())},
            "frozen_checksums": frozen_checksum.stdout,
            "package_checksums": package_checksum.stdout,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
