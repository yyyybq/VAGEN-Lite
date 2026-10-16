#!/usr/bin/env python3
"""Materialize and print a guarded pilot command; never execute it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex

import yaml


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight-report", required=True, type=Path)
    parser.add_argument("--variant", required=True, choices=("S0", "S1", "S5"))
    parser.add_argument("--id-manifest-name", default="val_id")
    parser.add_argument("--ood-manifest-name", default="validation_proxy")
    parser.add_argument("--experiment-name", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--emit-command", action="store_true")
    args = parser.parse_args()

    if not args.emit_command:
        print("NOT_RUN: pass --emit-command to materialize and print (but not execute) the pilot command")
        return 3

    report_path = args.preflight_report.resolve()
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "PASS":
        print(f"BLOCKED: preflight status={report.get('status')} report={report_path}")
        return 2
    if report.get("config_gate", {}).get("status") != "PASS" or report.get("data_gate") != "PASS":
        print("BLOCKED: config_gate and data_gate must both PASS")
        return 2

    resolved_path = Path(report["config_gate"]["resolved_paths"][args.variant])
    resolved = json.loads(resolved_path.read_text(encoding="utf-8"))
    manifests = resolved["explicit_manifests"]
    if args.id_manifest_name not in manifests or args.ood_manifest_name not in manifests:
        print("BLOCKED: selected explicit ID/OOD manifest names are absent")
        return 2
    manifest_reports = report["manifests"]
    id_count = manifest_reports[args.id_manifest_name]["rows"]
    ood_count = manifest_reports[args.ood_manifest_name]["rows"]
    train_count = manifest_reports["train"]["rows"]

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    env_path = output_dir / f"{args.variant}.env.yaml"
    env_cfg = dict(resolved["environment"])
    env_cfg["jsonl_path"] = manifests["train"]
    env_payload = {
        "env1": {
            "env_name": "active_spatial",
            "train_size": train_count,
            "test_size": 0,
            "env_config": env_cfg,
        }
    }
    env_path.write_text(yaml.safe_dump(env_payload, sort_keys=False), encoding="utf-8")

    repo = Path(__file__).resolve().parents[1]
    command_env = {
        "DENSE_SCORE_PREFLIGHT_REPORT": str(report_path),
        "DENSE_SCORE_VARIANT": args.variant,
        "DENSE_SCORE_ENV_CONFIG": str(env_path),
        "DENSE_SCORE_EXPERIMENT_NAME": args.experiment_name,
        "DENSE_SCORE_ID_MANIFEST": manifests[args.id_manifest_name],
        "DENSE_SCORE_ID_COUNT": str(id_count),
        "DENSE_SCORE_OOD_MANIFEST": manifests[args.ood_manifest_name],
        "DENSE_SCORE_OOD_COUNT": str(ood_count),
    }
    assignments = " ".join(f"{key}={shlex.quote(value)}" for key, value in command_env.items())
    launcher = repo / "examples/train/active_spatial/run_experiment.sh"
    experiment = repo / "examples/train/active_spatial/experiments/dense_score_reward_only_pilot.sh"
    print("NOT_RUN: command draft follows; this tool never executes it")
    print(f"{assignments} bash {shlex.quote(str(launcher))} {shlex.quote(str(experiment))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
