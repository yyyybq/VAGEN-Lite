#!/usr/bin/env python3
"""Compare fixed-32 Base/step-2/4/8 results against expansion gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


MODEL_KEYS = ("base", "step2", "step4", "step8")


def episodes(root: Path, model_key: str) -> dict[str, dict[str, Any]]:
    result = {}
    paths = sorted((root / model_key / "episodes").glob("*/attempt_00/episode.json"))
    if len(paths) != 32:
        raise RuntimeError(f"{model_key}: expected 32 episodes, found {len(paths)}")
    for path in paths:
        record = json.loads(path.read_text())
        if record["status"] != "complete":
            raise RuntimeError(f"{path}: incomplete episode")
        result[record["episode_fingerprint"]] = record
    return result


def metrics(records: dict[str, dict[str, Any]]) -> dict[str, Any]:
    invalid_turns = sum(int(record["invalid_turns"]) for record in records.values())
    model_turns = sum(int(record["model_turns"]) for record in records.values())
    collision_attempts = sum(int(record["collision_attempts"]) for record in records.values())
    collision_episodes = sum(int(record["collision_attempts"]) > 0 for record in records.values())
    high_score_off_frame = 0
    for record in records.values():
        state_metrics = [record["initial_canonical_metric"]]
        state_metrics.extend(turn["canonical_metric"] for turn in record["turns"])
        high_score_off_frame += sum(
            float(metric.get("shaping_score", metric["score"])) >= 0.5
            and not metric["gates"]["inside_frame"]
            for metric in state_metrics
        )
    return {
        "successes": sum(bool(record["success"]) for record in records.values()),
        "invalid_turns": invalid_turns,
        "model_turns": model_turns,
        "invalid_turn_rate": invalid_turns / max(model_turns, 1),
        "collision_attempts": collision_attempts,
        "collision_attempt_rate": collision_attempts / max(model_turns, 1),
        "episodes_with_collision": collision_episodes,
        "high_shaping_score_off_frame_states": high_score_off_frame,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--eval-root", default="paired_eval")
    args = parser.parse_args()
    eval_root = args.run / args.eval_root
    all_records = {key: episodes(eval_root, key) for key in MODEL_KEYS}
    base_ids = set(all_records["base"])
    if any(set(records) != base_ids for records in all_records.values()):
        raise RuntimeError("evaluation task identities are not paired")
    summaries = {key: metrics(records) for key, records in all_records.items()}
    base = all_records["base"]
    base_summary = summaries["base"]
    comparisons = {}
    for key in MODEL_KEYS[1:]:
        records = all_records[key]
        gained = sorted(
            fingerprint for fingerprint in base_ids
            if records[fingerprint]["success"] and not base[fingerprint]["success"]
        )
        lost = sorted(
            fingerprint for fingerprint in base_ids
            if base[fingerprint]["success"] and not records[fingerprint]["success"]
        )
        summary = summaries[key]
        checks = {
            "canonical_success_appears": summary["successes"] > 0,
            "no_high_score_inside_frame_failure": summary["high_shaping_score_off_frame_states"] == 0,
            "paired_success_net_growth": len(gained) > len(lost),
            "invalid_turn_not_worse": summary["invalid_turn_rate"] <= base_summary["invalid_turn_rate"] + 1e-12,
            "collision_attempt_rate_not_worse": (
                summary["collision_attempt_rate"]
                <= base_summary["collision_attempt_rate"] + 1e-12
            ),
        }
        comparisons[key] = {
            "summary": summary,
            "success_delta": summary["successes"] - base_summary["successes"],
            "gained": gained,
            "lost": lost,
            "checks": checks,
            "eligible_for_expansion": all(checks.values()),
        }
    replay = json.loads((args.run / "gate_aligned_reward_replay.json").read_text())
    report = {
        "schema_version": "r1_gate_aligned_fixed32_comparison_v1",
        "status": "PASS",
        "tasks": len(base_ids),
        "base": base_summary,
        "checkpoints": comparisons,
        "reward_ordering_replay_passed": bool(
            replay["checks"]["new_reward_ranking_is_closer_to_gate_count"]
        ),
        "expanded_training_allowed": any(
            value["eligible_for_expansion"] for value in comparisons.values()
        ) and bool(replay["checks"]["new_reward_ranking_is_closer_to_gate_count"]),
    }
    output = eval_root / "paired_report.json"
    output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
