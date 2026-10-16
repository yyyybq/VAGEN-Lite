#!/usr/bin/env python3
"""Materialize and compare the three reward-only pilot configs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf, open_dict


VARIANTS = {
    "S0": {
        "enable_potential_shaping_reward": False,
        "enable_near_success_reward": False,
        "enable_visibility_shaping_reward": False,
        "near_success_bonus": 0.0,
        "potential_field_gamma": 0.95,
    },
    "S1": {
        "enable_potential_shaping_reward": True,
        "enable_near_success_reward": False,
        "enable_visibility_shaping_reward": False,
        "near_success_bonus": 0.0,
        "potential_field_gamma": 0.95,
    },
    "S5": {
        "enable_potential_shaping_reward": True,
        "enable_near_success_reward": True,
        "enable_visibility_shaping_reward": True,
        "near_success_bonus": 0.5,
        "potential_field_gamma": 0.99,
    },
}

COMMON_REWARD = {
    "enable_potential_field": True,
    "format_reward": 0.0,
    "invalid_format_penalty": -0.1,
    "success_reward": 5.0,
}

BOOKKEEPING_PATHS = {
    "training.data.train_files",
    "training.trainer.experiment_name",
    "training.trainer.default_local_dir",
    "training.trainer.rollout_data_dir",
    "training.trainer.validation_data_dir",
}
REWARD_PATHS = {
    f"environment.envs.0.config.{key}" for key in set(COMMON_REWARD) | set(next(iter(VARIANTS.values())))
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            result.update(_flatten(item, f"{prefix}.{key}" if prefix else str(key)))
        return result
    if isinstance(value, list):
        result = {}
        for index, item in enumerate(value):
            result.update(_flatten(item, f"{prefix}.{index}"))
        return result
    return {prefix: value}


def _diff(left: dict[str, Any], right: dict[str, Any]) -> dict[str, dict[str, Any]]:
    left_flat, right_flat = _flatten(left), _flatten(right)
    return {
        key: {"left": left_flat.get(key), "right": right_flat.get(key)}
        for key in sorted(set(left_flat) | set(right_flat))
        if left_flat.get(key) != right_flat.get(key)
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--baseline-env", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--agent-loop-workers", type=int, default=None)
    args = parser.parse_args()
    baseline_path = Path(args.baseline).resolve()
    baseline_env_path = Path(args.baseline_env).resolve()
    output_dir = Path(args.output_dir).resolve()
    run_root = Path(args.run_root).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rendered: dict[str, dict[str, Any]] = {}
    files: dict[str, dict[str, str]] = {}
    for name, reward in VARIANTS.items():
        config = OmegaConf.load(baseline_path)
        environment = OmegaConf.load(baseline_env_path)
        with open_dict(config):
            env = environment.envs[0].config
            for key, value in COMMON_REWARD.items():
                env[key] = value
            for key, value in reward.items():
                env[key] = value
            config.trainer.experiment_name = f"active_spatial_dense_score_{name}_pilot150"
            branch_root = run_root / name
            config.trainer.default_local_dir = str(branch_root / "checkpoints")
            config.trainer.rollout_data_dir = str(branch_root / "rollout_data")
            config.trainer.validation_data_dir = str(branch_root / "validation")
            branch_env_path = output_dir / f"{name}.train.yaml"
            config.data.train_files = str(branch_env_path)
            config.trainer.resume_mode = "disable"
            config.trainer.resume_from_path = None
            config.trainer.total_training_steps = 700
            config.trainer.critic_warmup = 60
            config.trainer.save_freq = 50
            config.trainer.test_freq = 50
            config.trainer.val_before_train = True
            if args.agent_loop_workers is not None:
                if args.agent_loop_workers < 1:
                    raise ValueError("--agent-loop-workers must be positive")
                config.actor_rollout_ref.rollout.agent.num_workers = args.agent_loop_workers
        output = output_dir / f"{name}.yaml"
        OmegaConf.save(environment, branch_env_path, resolve=False)
        OmegaConf.save(config, output, resolve=False)
        resolved_training = OmegaConf.to_container(config, resolve=True)
        # The renderer URL is intentionally late-bound on the worker.  Preserve
        # that interpolation while comparing every concrete reward field.
        resolved_environment = OmegaConf.to_container(environment, resolve=False)
        assert isinstance(resolved_training, dict) and isinstance(resolved_environment, dict)
        rendered[name] = {"training": resolved_training, "environment": resolved_environment}
        files[name] = {
            "training_path": str(output),
            "training_sha256": _sha256(output),
            "environment_path": str(branch_env_path),
            "environment_sha256": _sha256(branch_env_path),
        }

    pairwise = {}
    for left, right in (("S0", "S1"), ("S1", "S5"), ("S0", "S5")):
        differences = _diff(rendered[left], rendered[right])
        unexpected = sorted(set(differences) - REWARD_PATHS - BOOKKEEPING_PATHS)
        pairwise[f"{left}_vs_{right}"] = {
            "differences": differences,
            "unexpected_differences": unexpected,
            "status": "PASS" if not unexpected else "BLOCKED",
        }
    s0_s1_reward = {
        key for key in pairwise["S0_vs_S1"]["differences"] if key in REWARD_PATHS
    }
    expected_single_variable = {"environment.envs.0.config.enable_potential_shaping_reward"}
    checks = {
        "all_pairwise_diffs_reward_or_bookkeeping_only": all(not item["unexpected_differences"] for item in pairwise.values()),
        "s0_s1_only_potential_application_switch": s0_s1_reward == expected_single_variable,
        "ppo_gamma_is_0_95": all(float(item["training"]["algorithm"]["gamma"]) == 0.95 for item in rendered.values()),
        "s1_potential_gamma_matches_ppo": float(rendered["S1"]["environment"]["envs"][0]["config"]["potential_field_gamma"]) == 0.95,
        "fresh_start_all_branches": all(item["training"]["trainer"]["resume_mode"] == "disable" for item in rendered.values()),
        "production_horizon_retained": all(int(item["training"]["trainer"]["total_training_steps"]) == 700 for item in rendered.values()),
        "critic_warmup_retained": all(int(item["training"]["trainer"]["critic_warmup"]) == 60 for item in rendered.values()),
        "agent_loop_workers_common": len({int(item["training"]["actor_rollout_ref"]["rollout"]["agent"]["num_workers"]) for item in rendered.values()}) == 1,
        "milestones_50_100_150": all({50, 100, 150}.issubset(set(item["training"]["trainer"]["ckpt_milestones"])) for item in rendered.values()),
    }
    report = {
        "schema_version": "active_spatial_dense_score_pilot_configs_v1",
        "status": "PASS" if all(checks.values()) else "BLOCKED",
        "baseline": {"path": str(baseline_path), "sha256": _sha256(baseline_path)},
        "baseline_environment": {"path": str(baseline_env_path), "sha256": _sha256(baseline_env_path)},
        "files": files,
        "checks": checks,
        "allowed_difference_paths": sorted(REWARD_PATHS | BOOKKEEPING_PATHS),
        "pairwise": pairwise,
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
