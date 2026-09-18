#!/usr/bin/env python3
"""Merge independently verified R1 expansion rows into a clean corpus.

No row is accepted from generator status alone.  Projective and FOV expansion
rows need matching independent runtime and official-RGB PASS records.  The
60-source local-action protocol is used only as an exclusion set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from r1_audit_clean_training_inventory import (
    category_pair,
    episode_fingerprint,
    materialize_train_row,
    read_json,
    read_jsonl,
    relation,
    require,
    sha256,
    source_key,
    summarize,
    write_json,
    write_jsonl,
    atomic_text,
)


VERSION = "r1_clean_training_corpus_v1"
TARGET_TASKS = {"projective_relations", "fov_inclusion"}


def load_base(root: Path) -> list[dict[str, Any]]:
    inventory = read_jsonl(root / "verified_train_source_inventory.jsonl")
    rows = read_jsonl(root / "verified_train_manifest.jsonl")
    require(len(inventory) == len(rows), "base inventory/manifest cardinality mismatch")
    return [{**record, "row": row} for record, row in zip(inventory, rows)]


def load_projective(root: Path) -> list[dict[str, Any]]:
    validation = root / "projective_medium_v2" / "independent_validation"
    candidates = {row["task_id"]: row for row in read_jsonl(validation / "candidate_rows.jsonl")}
    reaches = {row["task_id"]: row for row in read_jsonl(validation / "reachability_manifest.jsonl")}
    runtime = {row["task_id"]: row for row in read_json(validation / "runtime_replay.json")["results"]}
    rgb_path = root / "projective_medium_v2" / "official_observability_v1" / "observability_manifest.jsonl"
    rgb = {row["task_id"]: row for row in read_jsonl(rgb_path)}
    selected = []
    for task_id in sorted(candidates):
        replay = runtime.get(task_id)
        observation = rgb.get(task_id)
        reach = reaches.get(task_id)
        if not replay or replay.get("status") != "pass" or not observation or not observation.get("passed"):
            continue
        require(reach is not None and reach.get("status") == "reachable", f"missing Projective path {task_id}")
        require(reach.get("lower_bound_complete") is True and int(reach.get("certified_lower_bound")) == 4,
                f"Projective lower bound incomplete {task_id}")
        raw = candidates[task_id]
        index = int(reach["source_row_index"])
        fp = episode_fingerprint(raw)
        row = materialize_train_row(raw, replay["collision_convention"], source_key("train", index), fp)
        selected.append({
            "source_key": source_key("train", index), "source_row_index": index, "split": "train",
            "scene_id": str(row["scene_id"]), "task_id": task_id, "episode_fingerprint": fp,
            "evidence_class": "expansion_projective_medium_runtime_rgb_verified",
            "difficulty": {"certificate_upper_bound": int(reach["first_success_step"]),
                           "first_success_step": int(replay["steps"]),
                           "certified_lower_bound": 4, "lower_bound_complete": True},
            "row": row,
            "evidence": {"candidate": str(validation / "candidate_rows.jsonl"),
                         "reachability": str(validation / "reachability_manifest.jsonl"),
                         "runtime": str(validation / "runtime_replay.json"),
                         "official_rgb": str(rgb_path), "contact_sheet": observation.get("contact_sheet")},
        })
    return selected


def load_fov(root: Path) -> list[dict[str, Any]]:
    train = root / "fov_min4" / "train"
    candidate_path = train / "trainable.jsonl"
    reach_path = train / "reachability_manifest.jsonl"
    runtime_path = train / "runtime_replay.json"
    rgb_path = root / "fov_min4" / "official_observability_v1" / "observability_manifest.jsonl"
    candidates = {row["task_id"]: row for row in read_jsonl(candidate_path) if row.get("task_type") == "fov_inclusion"}
    reaches = {row["task_id"]: row for row in read_jsonl(reach_path) if row.get("task_id") in candidates}
    runtime = {row["task_id"]: row for row in read_json(runtime_path)["results"]}
    rgb = {row["task_id"]: row for row in read_jsonl(rgb_path)}
    selected = []
    for task_id in sorted(candidates):
        replay = runtime.get(task_id)
        observation = rgb.get(task_id)
        reach = reaches.get(task_id)
        if not replay or replay.get("status") != "pass" or not observation or not observation.get("passed"):
            continue
        require(reach is not None and reach.get("status") == "reachable", f"missing FOV path {task_id}")
        require(replay.get("lower_bound_complete") is True and replay.get("no_success_through_depth_3") is True,
                f"FOV lower bound incomplete {task_id}")
        raw = candidates[task_id]
        index = int(replay["source_row_index"])
        fp = episode_fingerprint(raw)
        row = materialize_train_row(raw, replay["collision_convention"], source_key("train", index), fp)
        selected.append({
            "source_key": source_key("train", index), "source_row_index": index, "split": "train",
            "scene_id": str(row["scene_id"]), "task_id": task_id, "episode_fingerprint": fp,
            "evidence_class": "expansion_fov_min4_runtime_rgb_verified",
            "difficulty": {"certificate_upper_bound": int(replay["steps"]),
                           "first_success_step": int(replay["steps"]),
                           "certified_lower_bound": 4, "lower_bound_complete": True},
            "row": row,
            "evidence": {"candidate": str(candidate_path), "reachability": str(reach_path),
                         "runtime": str(runtime_path), "official_rgb": str(rgb_path),
                         "contact_sheet": observation.get("contact_sheet")},
        })
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-audit", type=Path, required=True)
    parser.add_argument("--expansion", type=Path, required=True)
    parser.add_argument("--old-v46-train", type=Path, required=True)
    parser.add_argument("--canonical-eval", type=Path, required=True)
    parser.add_argument("--local-action-parents", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    records = load_base(args.base_audit) + load_projective(args.expansion) + load_fov(args.expansion)
    keys = [row["source_key"] for row in records]
    require(len(keys) == len(set(keys)), "duplicate source identity in merged repaired corpus")
    records.sort(key=lambda row: (row["split"], row["source_row_index"], row["episode_fingerprint"]))
    train_sources = set(keys)
    train_scenes = {row["scene_id"] for row in records}
    canonical_eval = read_jsonl(args.canonical_eval)
    local_eval = read_jsonl(args.local_action_parents)
    eval_sources = {str(row["source_key"]) for row in canonical_eval}
    local_sources = {str(row["source_key"]) for row in local_eval}
    local_scenes = {str(row["scene_id"]) for row in local_eval}
    isolation = {
        "version": VERSION,
        "canonical_development_eval_source_overlap": sorted(train_sources & eval_sources),
        "permanent_local_action_source_overlap": sorted(train_sources & local_sources),
        "permanent_local_action_scene_overlap": sorted(train_scenes & local_scenes),
    }
    require(not isolation["canonical_development_eval_source_overlap"], "canonical eval source leak")
    require(not isolation["permanent_local_action_source_overlap"], "local-action source leak")
    require(not isolation["permanent_local_action_scene_overlap"], "local-action scene leak")

    target_summary = summarize(records)
    largest_pair = max(target_summary["category_pairs"].values())
    gate = {
        "criteria_version": "r1_canonical_eval_train_pilot_design_20260914",
        "required": {"unique_train_sources": 200, "scenes": 20, "left_sources": 80,
                     "right_sources": 80, "step4_episodes": 40, "step5_episodes": 40,
                     "step6_episodes": 40, "max_category_pair_fraction": 0.35},
        "observed": {"unique_train_sources": target_summary["sources"],
                     "scenes": len(target_summary["scenes"]),
                     "left_sources": target_summary["relations"].get("left", 0),
                     "right_sources": target_summary["relations"].get("right", 0),
                     "step4_episodes": target_summary["first_success_steps"].get(4, 0),
                     "step5_episodes": target_summary["first_success_steps"].get(5, 0),
                     "step6_episodes": target_summary["first_success_steps"].get(6, 0),
                     "max_category_pair_fraction": largest_pair / len(records)},
    }
    observed, required = gate["observed"], gate["required"]
    gate["passed"] = all((observed[key] >= value if key != "max_category_pair_fraction" else observed[key] <= value)
                         for key, value in required.items())

    old_rows = read_jsonl(args.old_v46_train)
    old_counts = Counter(row.get("task_type") for row in old_rows)
    repaired_counts = Counter(row["row"].get("task_type") for row in records)
    non_target = [row for row in old_rows if row.get("task_type") not in TARGET_TASKS]
    full_rows = non_target + [row["row"] for row in records]
    full_counts = Counter(row.get("task_type") for row in full_rows)
    comparability = {
        "old_v46_rows": len(old_rows), "clean_rows": len(full_rows),
        "unchanged_noncanonical_rows": len(non_target),
        "old_task_counts": dict(sorted(old_counts.items())),
        "clean_task_counts": dict(sorted(full_counts.items())),
        "verified_repaired_task_counts": dict(sorted(repaired_counts.items())),
        "target_task_retention_fraction": {
            task: repaired_counts[task] / old_counts[task] for task in sorted(TARGET_TASKS)
        },
        "warning": "Minimal-pilot PASS does not prove old-v46 task-mixture parity; no duplicated rows or reweighting are used.",
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    inventory = [{key: value for key, value in row.items() if key != "row"} for row in records]
    write_jsonl(args.output_dir / "verified_repaired_source_inventory.jsonl", inventory)
    write_jsonl(args.output_dir / "repaired_canonical_train_manifest.jsonl", [row["row"] for row in records])
    write_jsonl(args.output_dir / "clean_training_manifest.jsonl", full_rows)
    write_json(args.output_dir / "distribution_summary.json", target_summary)
    write_json(args.output_dir / "train_eval_isolation.json", isolation)
    write_json(args.output_dir / "minimal_pilot_readiness_gate.json", gate)
    write_json(args.output_dir / "old_v46_comparability.json", comparability)
    write_json(args.output_dir / "provenance.json", {
        "version": VERSION,
        "inputs": {"base_audit": str(args.base_audit.resolve()),
                   "expansion": str(args.expansion.resolve()),
                   "old_v46_train": {"path": str(args.old_v46_train.resolve()), "sha256": sha256(args.old_v46_train)},
                   "canonical_eval": {"path": str(args.canonical_eval.resolve()), "sha256": sha256(args.canonical_eval)},
                   "local_action_parents": {"path": str(args.local_action_parents.resolve()), "sha256": sha256(args.local_action_parents)}},
        "selection": "one independently runtime/RGB-verified episode per original train source",
        "non_target_policy": "copied byte-semantic rows from old v46 train manifest; original target rows excluded",
        "local_action_policy": "permanent evaluation-only exclusion",
    })
    files = sorted(path for path in args.output_dir.iterdir() if path.is_file() and path.name != "SHA256SUMS")
    atomic_text(args.output_dir / "SHA256SUMS", "".join(f"{sha256(path)}  {path.name}\n" for path in files))
    print(json.dumps({"target_summary": target_summary, "gate": gate, "comparability": comparability}, indent=2))


if __name__ == "__main__":
    main()
