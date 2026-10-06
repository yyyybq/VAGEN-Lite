#!/usr/bin/env python3
"""Check whether archived rollouts are sufficient for fixed-batch gradients.

This is intentionally read-only and does not deserialize pickle/torch files.
It never reconstructs missing reward terms from an aggregate score.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any


REQUIRED_GROUPS = {
    "exact_model_input_tokens": {"input_ids", "prompt_ids"},
    "generated_token_ids": {"response_ids", "generated_token_ids"},
    "turn_token_correspondence": {"turn_idx", "response_mask", "group_idx", "traj_idx"},
    "reward_components": {"reward_trace", "reward_components", "potential_reward"},
    "behavior_old_logprob": {"old_log_probs", "old_logprob", "response_logprobs"},
    "critic_value": {"values", "value"},
    "optimization_mask": {"response_mask", "attention_mask", "loss_mask"},
    "termination": {"terminated"},
    "truncation": {"truncated"},
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout-dir", required=True, type=Path)
    parser.add_argument("--actor-checkpoint", required=True, type=Path)
    parser.add_argument("--critic-checkpoint", required=True, type=Path)
    parser.add_argument("--resolved-config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    rollout_files = sorted(args.rollout_dir.glob("*.jsonl"))
    union_keys: set[str] = set()
    intersection_keys: set[str] | None = None
    key_counts: Counter[str] = Counter()
    row_count = 0
    parse_errors = []
    total_reward_only_rows = 0
    for path in rollout_files:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except Exception as exc:
                    if len(parse_errors) < 20:
                        parse_errors.append({"path": str(path), "line": line_number, "error": str(exc)})
                    continue
                keys = set(row) if isinstance(row, dict) else set()
                union_keys.update(keys)
                intersection_keys = keys if intersection_keys is None else intersection_keys & keys
                key_counts.update(keys)
                row_count += 1
                if "score" in keys and not keys.intersection(REQUIRED_GROUPS["reward_components"]):
                    total_reward_only_rows += 1

    fields = {}
    missing = []
    for group, aliases in REQUIRED_GROUPS.items():
        present = sorted(union_keys & aliases)
        status = "PASS" if present else "MISSING"
        fields[group] = {"status": status, "accepted_aliases": sorted(aliases), "present": present}
        if not present:
            missing.append(group)

    actor_files = sorted(path for path in args.actor_checkpoint.rglob("*") if path.is_file())
    critic_files = sorted(path for path in args.critic_checkpoint.rglob("*") if path.is_file())
    checkpoint = {
        "actor": {
            "path": str(args.actor_checkpoint.resolve()),
            "files": len(actor_files),
            "safetensor_shards": sum(path.suffix == ".safetensors" for path in actor_files),
            "status": "PASS" if actor_files else "MISSING",
        },
        "critic": {
            "path": str(args.critic_checkpoint.resolve()),
            "files": len(critic_files),
            "status": "PASS" if critic_files else "MISSING",
        },
        "resolved_config": {
            "path": str(args.resolved_config.resolve()),
            "status": "PASS" if args.resolved_config.is_file() else "MISSING",
            "sha256": sha256_file(args.resolved_config) if args.resolved_config.is_file() else None,
        },
        "rollout_to_checkpoint_binding": {
            "status": "MISSING",
            "reason": "rollout rows contain step but no checkpoint/config content identity",
        },
    }
    if checkpoint["actor"]["status"] != "PASS":
        missing.append("actor_checkpoint")
    if checkpoint["critic"]["status"] != "PASS":
        missing.append("critic_checkpoint")
    if checkpoint["resolved_config"]["status"] != "PASS":
        missing.append("resolved_config")
    missing.append("rollout_to_checkpoint_binding")

    status = "PASS_REAL_BATCH" if not missing and not parse_errors else "BLOCKED_REAL_BATCH"
    report: dict[str, Any] = {
        "schema_version": "active_spatial_real_batch_preflight_v1",
        "status": status,
        "gradient_backward_run": False,
        "optimizer_step_run": False,
        "checkpoint_modified": False,
        "rollouts": {
            "directory": str(args.rollout_dir.resolve()),
            "files": len(rollout_files),
            "rows": row_count,
            "parse_errors_first_20": parse_errors,
            "union_keys": sorted(union_keys),
            "intersection_keys": sorted(intersection_keys or set()),
            "key_row_counts": dict(sorted(key_counts.items())),
            "rows_with_total_score_but_no_reward_components": total_reward_only_rows,
            "text_input_output_present": "input" in union_keys and "output" in union_keys,
            "text_is_not_exact_multimodal_batch": True,
        },
        "required_fields": fields,
        "checkpoint_and_config": checkpoint,
        "missing_requirements": sorted(set(missing)),
        "decision": (
            "No reward components, old logprobs, values, masks, token/turn mapping, or termination semantics may be guessed from total score logs."
            if status == "BLOCKED_REAL_BATCH"
            else "Materials are sufficient for a separate explicitly authorized no-step backward diagnostic."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "output": str(args.output.resolve()), "missing": report["missing_requirements"]}))
    return 0 if status == "PASS_REAL_BATCH" else 2


if __name__ == "__main__":
    raise SystemExit(main())
