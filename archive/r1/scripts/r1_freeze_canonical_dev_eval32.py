#!/usr/bin/env python3
"""Freeze the 32-source R1 canonical development-regression evaluation set.

The policy JSONL intentionally excludes certificates, terminal poses, planner
paths, and evidence locations.  Those fields remain in the separate audit
manifest and are never passed to the environment's policy observation.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


VERSION = "r1_canonical_dev_eval32_manifest_v1"
CANONICAL_METRIC = "canonical_spatial_task_h1_v1"
CAMERA_VERSION = "canonical_h1_from_frozen_candidate_intrinsics_and_pose"
FORBIDDEN_POLICY_KEYS = {
    "actions",
    "certificate",
    "certificate_actions",
    "reachability_construction",
    "terminal_pose_c2w",
    "planner",
    "planner_output",
    "evidence",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    tmp.replace(path)


def read_indexed_jsonl(path: Path, index: int) -> dict[str, Any]:
    with path.open() as handle:
        for current, line in enumerate(handle):
            if current == index:
                return json.loads(line)
    raise IndexError(f"missing JSONL record {index} in {path}")


def runtime_record(episode: dict[str, Any]) -> dict[str, Any]:
    ref = episode["evidence"]["runtime"]
    path = Path(ref["path"])
    if sha256(path) != ref["sha256"]:
        raise ValueError(f"runtime evidence SHA mismatch: {path}")
    payload = json.loads(path.read_text())
    rows = payload.get("results", payload if isinstance(payload, list) else [])
    return rows[int(ref["record_index"])]


def candidate_row(episode: dict[str, Any]) -> dict[str, Any]:
    ref = episode["evidence"]["candidate"]
    path = Path(ref["path"])
    if sha256(path) != ref["sha256"]:
        raise ValueError(f"candidate evidence SHA mismatch: {path}")
    return read_indexed_jsonl(path, int(ref["record_index"]))


def make_policy_row(record: dict[str, Any], episode: dict[str, Any]) -> dict[str, Any]:
    row = copy.deepcopy(candidate_row(episode))
    for key in FORBIDDEN_POLICY_KEYS:
        row.pop(key, None)
    # These fields select the frozen R1 runtime backend.  They do not expose a
    # terminal state or expert action to the policy.
    row["canonical_task_metric_version"] = CANONICAL_METRIC
    row["camera_model_version"] = CAMERA_VERSION
    runtime = runtime_record(episode)
    convention = runtime.get("collision_convention")
    if convention:
        row["collision_convention"] = {
            "version": convention["version"],
            "structure_y_sign": convention["structure_y_sign"],
        }
    row["task_id"] = f"r1_devreg_{record['split']}_{int(record['source_row_index']):06d}"
    row["source_identity"] = {
        "split": record["split"],
        "source_row_index": int(record["source_row_index"]),
        "episode_fingerprint": episode["episode_fingerprint"],
        "experiment_role": "development_regression",
    }
    forbidden = FORBIDDEN_POLICY_KEYS.intersection(row)
    if forbidden:
        raise ValueError(f"audit-only keys leaked into policy row: {sorted(forbidden)}")
    if row["init_camera"]["extrinsics"] != episode["initial_pose_c2w"]:
        raise ValueError(f"initial pose mismatch for {record['source_key']}")
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_text())
    episode_by_fp = {row["episode_fingerprint"]: row for row in inventory["episodes"]}
    selected: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for record in inventory["records"]:
        if record["split"] == "train":
            continue
        fingerprints = sorted(record["episode_fingerprints"])
        if not fingerprints:
            raise ValueError(f"no episode for {record['source_key']}")
        selected.append((record, episode_by_fp[fingerprints[0]]))
    selected.sort(key=lambda pair: (pair[0]["split"], int(pair[0]["source_row_index"])))
    if len(selected) != 32:
        raise ValueError(f"expected 32 evaluation-only sources, got {len(selected)}")
    if len({record["source_key"] for record, _ in selected}) != 32:
        raise ValueError("duplicate source identity")

    policy_rows = [make_policy_row(record, episode) for record, episode in selected]
    audit_rows = []
    for eval_index, (record, episode) in enumerate(selected):
        source_path = Path(record["source_manifest"]["path"])
        if sha256(source_path) != record["source_manifest"]["sha256"]:
            raise ValueError(f"source manifest SHA mismatch: {source_path}")
        audit_rows.append({
            "eval_index": eval_index,
            "experiment_role": "development_regression",
            "source_key": record["source_key"],
            "split": record["split"],
            "source_row_index": int(record["source_row_index"]),
            "source_manifest": record["source_manifest"],
            "scene_id": record["scene_id"],
            "object_pair": record["object_pair"],
            "relation": record["relation"],
            "task_type": record["task_type"],
            "episode_fingerprint": episode["episode_fingerprint"],
            "episode_selection_rule": "lexicographically_smallest_episode_fingerprint",
            "initial_pose_c2w": episode["initial_pose_c2w"],
            "terminal_pose_c2w_audit_only": episode["terminal_pose_c2w"],
            "certificate_actions_audit_only": episode["actions"],
            "difficulty": episode["difficulty"],
            "evidence_audit_only": episode["evidence"],
            "versions": episode["versions"],
            "policy_task_id": policy_rows[eval_index]["task_id"],
        })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    policy_path = args.output_dir / "policy_input_rows.jsonl"
    audit_path = args.output_dir / "development_regression_manifest.jsonl"
    atomic_jsonl(policy_path, policy_rows)
    atomic_jsonl(audit_path, audit_rows)
    config = {
        "version": "r1_canonical_dev_eval32_protocol_v1",
        "manifest_version": VERSION,
        "experiment_role": "development_regression",
        "generalization_claim": "none_development_exposed_scenes",
        "inventory": {"path": str(args.inventory.resolve()), "sha256": sha256(args.inventory)},
        "policy_input": {"path": str(policy_path.resolve()), "sha256": sha256(policy_path)},
        "audit_manifest": {"path": str(audit_path.resolve()), "sha256": sha256(audit_path)},
        "episode_selection": "one per non-train source; lexicographically smallest episode fingerprint",
        "paired_seed_base": 2026091500,
        "environment": {
            "camera": CAMERA_VERSION,
            "canonical_metric": CANONICAL_METRIC,
            "render_resolution": [256, 256],
            "action_space": "strafe",
            "step_translation_m": 0.3,
            "step_rotation_deg": 20.0,
            "max_actions_per_model_turn": 5,
            "max_primitive_actions": 12,
            "max_model_turns": 12,
            "explicit_done": False,
            "auto_termination": True,
            "collision": "frozen_per_scene_in_policy_rows",
            "distance_to_sample_target_in_observation": False,
        },
        "policy_contract": {
            "history": "no_concat_system_plus_current_observation_only",
            "audit_only_fields_not_visible": [
                "certificate actions", "terminal pose", "planner output", "evidence paths"
            ],
            "prompt_format": "free_think",
            "image_placeholder": "<image>",
        },
        "decoding": {
            "recipe_source": "v46 hydra actor_rollout_ref.rollout.val_kwargs",
            "temperature": 0.8,
            "top_p": 0.95,
            "do_sample": True,
            "n": 1,
            "max_tokens_per_turn": 384,
            "stop": ["</action>"],
            "include_stop_string": True,
            "paired_per_turn_seed": "paired_seed_base + eval_index*100 + turn_index",
        },
        "counts": {
            "sources": len(selected),
            "splits": dict(sorted(Counter(r["split"] for r, _ in selected).items())),
            "scenes": dict(sorted(Counter(r["scene_id"] for r, _ in selected).items())),
        },
    }
    atomic_json(args.output_dir / "eval_protocol.json", config)
    sums = []
    for path in sorted(args.output_dir.iterdir()):
        if path.is_file() and path.name != "SHA256SUMS":
            sums.append(f"{sha256(path)}  {path.name}")
    (args.output_dir / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    print(json.dumps(config["counts"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
