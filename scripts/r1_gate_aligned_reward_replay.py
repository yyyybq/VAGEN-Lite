#!/usr/bin/env python3
"""Offline replay gate for the R1 gate-aligned projective potential.

This re-scores the immutable 32-task certificate set and every recorded pose
from the Base/step-1 paired evaluation.  It does not render, load a model, or
modify either historical run.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from vagen.envs.active_spatial.canonical_task_metrics import (
    GATE_ALIGNED_SHAPING_SCORE_VERSION,
    score_canonical_task,
)
from vagen.envs.active_spatial.prompt import system_prompt
from vagen.envs.active_spatial.utils import ViewManipulator


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def rankdata(values: list[float]) -> np.ndarray:
    values_np = np.asarray(values, dtype=float)
    order = np.argsort(values_np, kind="mergesort")
    ranks = np.empty(len(values_np), dtype=float)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values_np[order[end]] == values_np[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0
        start = end
    return ranks


def spearman(left: list[float], right: list[float]) -> float:
    left_rank, right_rank = rankdata(left), rankdata(right)
    if np.std(left_rank) == 0.0 or np.std(right_rank) == 0.0:
        return 0.0
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def evaluate(root: Path, matrix_path: Path) -> dict[str, Any]:
    frozen = root / "frozen"
    policy_rows = load_jsonl(frozen / "policy_input_rows.jsonl")
    certificates = load_jsonl(frozen / "development_regression_manifest.jsonl")
    by_fingerprint = {
        row["source_identity"]["episode_fingerprint"]: row for row in policy_rows
    }

    certificate_rows = []
    for record in certificates:
        item = by_fingerprint[record["episode_fingerprint"]]
        engine = ViewManipulator(step_translation=0.3, step_rotation_deg=20.0)
        engine.reset(np.asarray(record["initial_pose_c2w"], dtype=float))
        metrics = [score_canonical_task(item, engine.get_pose())]
        for action in record["certificate_actions_audit_only"]:
            engine.step(action)
            metrics.append(score_canonical_task(item, engine.get_pose()))
        terminal = np.asarray(record["terminal_pose_c2w_audit_only"], dtype=float)
        certificate_rows.append(
            {
                "episode_fingerprint": record["episode_fingerprint"],
                "pose_max_abs_error": float(np.max(np.abs(engine.get_pose() - terminal))),
                "expected_first_success_step": record["difficulty"]["first_success_step"],
                "replayed_first_success_step": next(
                    (index for index, metric in enumerate(metrics) if metric["success"]), None
                ),
                "terminal_success": bool(metrics[-1]["success"]),
                "terminal_legacy_score": metrics[-1]["score"],
                "terminal_shaping_score": metrics[-1]["shaping_score"],
            }
        )

    model_reports: dict[str, Any] = {}
    for model in ("base", "step1"):
        states = []
        gate_mismatches = 0
        max_legacy_score_error = 0.0
        episode_paths = sorted(
            (root / "paired_eval_v2" / model / "episodes").glob(
                "*/attempt_00/episode.json"
            )
        )
        for episode_path in episode_paths:
            episode = json.loads(episode_path.read_text())
            item = by_fingerprint[episode["episode_fingerprint"]]
            recorded_and_poses = [
                (
                    episode["initial_canonical_metric"],
                    np.asarray(item["init_camera"]["extrinsics"], dtype=float),
                )
            ]
            recorded_and_poses.extend(
                (turn["canonical_metric"], np.asarray(turn["post_pose_c2w"], dtype=float))
                for turn in episode["turns"]
            )
            for recorded, pose in recorded_and_poses:
                metric = score_canonical_task(item, pose)
                states.append(metric)
                gate_mismatches += int(
                    recorded["gates"] != metric["gates"]
                    or bool(recorded["success"]) != bool(metric["success"])
                )
                max_legacy_score_error = max(
                    max_legacy_score_error, abs(float(recorded["score"]) - metric["score"])
                )

        gate_counts = [sum(metric["gates"].values()) for metric in states]
        legacy_scores = [metric["score"] for metric in states]
        shaping_scores = [metric["shaping_score"] for metric in states]
        grouped: dict[int, list[float]] = defaultdict(list)
        for count, value in zip(gate_counts, shaping_scores):
            grouped[count].append(value)
        means_by_satisfied_gates = {
            str(count): float(np.mean(values)) for count, values in sorted(grouped.items())
        }
        mean_values = list(means_by_satisfied_gates.values())
        off_frame = [metric for metric in states if not metric["gates"]["inside_frame"]]
        legacy_high_off_frame = [metric for metric in off_frame if metric["score"] >= 0.65]
        model_reports[model] = {
            "episodes": len(episode_paths),
            "states": len(states),
            "canonical_gate_mismatches": gate_mismatches,
            "max_legacy_score_reproduction_error": max_legacy_score_error,
            "off_frame_states": len(off_frame),
            "legacy_score_ge_0_65_off_frame_states": len(legacy_high_off_frame),
            "new_score_ge_0_5_off_frame_states": sum(
                metric["shaping_score"] >= 0.5 for metric in off_frame
            ),
            "max_new_score_for_legacy_high_off_frame": max(
                (metric["shaping_score"] for metric in legacy_high_off_frame), default=0.0
            ),
            "max_failed_state_shaping_score": max(
                (metric["shaping_score"] for metric in states if not metric["success"]),
                default=0.0,
            ),
            "legacy_score_gate_count_spearman": spearman(legacy_scores, gate_counts),
            "shaping_score_gate_count_spearman": spearman(shaping_scores, gate_counts),
            "mean_shaping_score_by_satisfied_gate_count": means_by_satisfied_gates,
            "mean_is_monotonic_with_satisfied_gate_count": all(
                later + 1e-12 >= earlier
                for earlier, later in zip(mean_values, mean_values[1:])
            ),
            "shaping_score_version": GATE_ALIGNED_SHAPING_SCORE_VERSION,
        }

    matrix = yaml.safe_load(matrix_path.read_text())
    s1 = matrix["variants"]["S1"]["reward_overrides"]
    prompt = system_prompt(
        task_type="projective_relations",
        action_space="strafe",
        format_reward=s1["format_reward"],
        invalid_format_penalty=-0.1,
    )

    checks = {
        "exactly_32_unique_tasks": len(certificates) == len(by_fingerprint) == 32,
        "certificate_action_replay_matches_terminal_pose": max(
            row["pose_max_abs_error"] for row in certificate_rows
        )
        <= 1e-9,
        "certificate_first_success_step_preserved": all(
            row["expected_first_success_step"] == row["replayed_first_success_step"]
            for row in certificate_rows
        ),
        "all_certificate_terminals_pass_canonical_gate": all(
            row["terminal_success"] for row in certificate_rows
        ),
        "canonical_successes_outrank_all_historical_failures": (
            min(row["terminal_shaping_score"] for row in certificate_rows)
            > max(
                report["max_failed_state_shaping_score"]
                for report in model_reports.values()
            )
        ),
        "historical_gate_and_legacy_score_reproduced": all(
            report["canonical_gate_mismatches"] == 0
            and report["max_legacy_score_reproduction_error"] <= 1e-12
            for report in model_reports.values()
        ),
        "off_frame_cannot_receive_high_new_potential": all(
            report["new_score_ge_0_5_off_frame_states"] == 0
            for report in model_reports.values()
        ),
        "new_reward_ranking_is_closer_to_gate_count": all(
            report["shaping_score_gate_count_spearman"]
            > report["legacy_score_gate_count_spearman"]
            and report["mean_is_monotonic_with_satisfied_gate_count"]
            for report in model_reports.values()
        ),
        "s1_reward_controls_match_requested_recipe": (
            s1["enable_potential_shaping_reward"] is True
            and s1["enable_near_success_reward"] is False
            and s1["enable_visibility_shaping_reward"] is False
            and float(s1["near_success_bonus"]) == 0.0
            and float(s1["format_reward"]) == 0.0
            and float(s1["potential_field_gamma"]) == 0.95
        ),
        "projective_prompt_disambiguates_turns": (
            "do not map directly to turn_left/turn_right" in prompt
            and "first try lateral translation" in prompt
        ),
    }
    return {
        "schema_version": "r1_gate_aligned_reward_replay_v1",
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "certificate_summary": {
            "tasks": len(certificate_rows),
            "terminal_successes": sum(row["terminal_success"] for row in certificate_rows),
            "max_pose_error": max(row["pose_max_abs_error"] for row in certificate_rows),
            "terminal_shaping_score_min": min(
                row["terminal_shaping_score"] for row in certificate_rows
            ),
            "terminal_shaping_score_max": max(
                row["terminal_shaping_score"] for row in certificate_rows
            ),
        },
        "models": model_reports,
        "certificate_rows": certificate_rows,
        "inputs": {
            "run_root": str(root),
            "ablation_matrix": str(matrix_path),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--ablation-matrix", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(args.run_root, args.ablation_matrix)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": report["status"], "checks": report["checks"]}, indent=2))
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
