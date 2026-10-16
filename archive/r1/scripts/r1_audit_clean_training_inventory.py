#!/usr/bin/env python3
"""Freeze and audit the existing verified R1 canonical train inventory.

This tool is intentionally evidence-only.  It does not generate candidates,
render images, or infer PASS from a task id.  Projective path-first episodes
must have matching independent runtime and official RGB PASS records.  Medium
episodes must come from the frozen verified inventory and retain its evidence
references.  The independent local-action diagnostic is loaded only as an
evaluation exclusion set and can never contribute a training row.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


VERSION = "r1_clean_training_inventory_audit_v1"
CANONICAL_METRIC = "canonical_spatial_task_h1_v1"
CAMERA_VERSION = "canonical_h1_from_frozen_candidate_intrinsics_and_pose"
FORBIDDEN_TRAIN_KEYS = {
    "actions", "certificate", "certificate_actions", "reachability_construction",
    "terminal_pose_c2w", "planner", "planner_output", "evidence",
}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_json(path: Path, payload: Any) -> None:
    atomic_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    atomic_text(path, "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def category_pair(row: dict[str, Any]) -> tuple[str, ...]:
    objects = row.get("target_object", {}).get("objects") or []
    return tuple(sorted(str(obj.get("label") or "unknown") for obj in objects))


def relation(row: dict[str, Any]) -> str:
    return str(row.get("target_region", {}).get("params", {}).get("relation") or "unknown")


def source_key(split: str, index: int) -> str:
    return f"{split}:{index}"


def episode_fingerprint(row: dict[str, Any]) -> str:
    payload = {
        "scene_id": row.get("scene_id"),
        "task_type": row.get("task_type"),
        "task_id": row.get("task_id"),
        "initial": row.get("init_camera", {}).get("extrinsics"),
        "target": row.get("sample_target"),
        "pair": [
            (obj.get("id"), obj.get("label"))
            for obj in row.get("target_object", {}).get("objects") or []
        ],
        "relation": relation(row),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def materialize_train_row(
    raw: dict[str, Any], convention: dict[str, Any], source: str, episode_fp: str
) -> dict[str, Any]:
    """Add runtime selectors while keeping expert/audit data out of training."""
    row = copy.deepcopy(raw)
    existing_metric = row.get("canonical_task_metric_version")
    existing_camera = row.get("camera_model_version")
    require(existing_metric in (None, CANONICAL_METRIC), f"conflicting metric selector {existing_metric}")
    require(existing_camera in (None, CAMERA_VERSION, "canonical_camera_h1_resize_v1"),
            f"conflicting camera selector {existing_camera}")
    for key in FORBIDDEN_TRAIN_KEYS:
        row.pop(key, None)
    row["canonical_task_metric_version"] = CANONICAL_METRIC
    row["camera_model_version"] = CAMERA_VERSION
    require(convention.get("version") == "interiorgs_structure_label_alignment_v1",
            f"wrong collision convention {convention}")
    require(convention.get("status", "frozen") == "frozen", f"unfrozen collision convention {convention}")
    row["collision_convention"] = {
        "version": convention["version"],
        "structure_y_sign": float(convention["structure_y_sign"]),
    }
    row["source_identity"] = {
        "split": "train",
        "source_key": source,
        "episode_fingerprint": episode_fp,
        "experiment_role": "clean_retraining_candidate",
    }
    require(not FORBIDDEN_TRAIN_KEYS.intersection(row), "expert/audit field leaked into train row")
    return row


def load_path_first(root: Path) -> list[dict[str, Any]]:
    aggregate = read_json(root / "canary_aggregate.json")
    merged = root / "merged_v2_unique_key"
    candidates = {row["task_id"]: row for row in read_jsonl(merged / "certificates/candidate_rows.jsonl")}
    reaches = {row["task_id"]: row for row in read_jsonl(merged / "certificates/reachability_manifest.jsonl")}
    runtime = {row["task_id"]: row for row in read_json(merged / "runtime_replay.json")["results"]}
    rgb = {row["task_id"]: row for row in read_jsonl(merged / "official_observability_v1/observability_manifest.jsonl")}
    selected = []
    for accounting in aggregate["rows"]:
        if accounting.get("split") != "train" or accounting.get("final_status") != "same_pair_accepted":
            continue
        task_id = str(accounting["task_id"])
        require(task_id in candidates, f"missing candidate {task_id}")
        require(task_id in reaches, f"missing reachability {task_id}")
        require(task_id in runtime, f"missing runtime {task_id}")
        require(task_id in rgb, f"missing RGB {task_id}")
        replay = runtime[task_id]
        observation = rgb[task_id]
        reach = reaches[task_id]
        require(replay.get("status") == "pass" and replay.get("initial_success") is False and replay.get("final_success") is True,
                f"runtime did not pass {task_id}")
        require(bool(observation.get("passed")), f"RGB did not pass {task_id}")
        require(reach.get("status") == "reachable", f"certificate not reachable {task_id}")
        raw_row = candidates[task_id]
        index = int(accounting["source_row_index"])
        fingerprint = episode_fingerprint(raw_row)
        row = materialize_train_row(
            raw_row, replay["collision_convention"], source_key("train", index), fingerprint
        )
        selected.append({
            "source_key": source_key("train", index),
            "source_row_index": index,
            "split": "train",
            "scene_id": str(row["scene_id"]),
            "task_id": task_id,
            "episode_fingerprint": fingerprint,
            "evidence_class": "path_first_runtime_rgb_verified",
            "difficulty": {
                "certificate_upper_bound": int(reach["steps"]),
                "first_success_step": int(replay["steps"]),
                "certified_lower_bound": None,
                "lower_bound_complete": False,
            },
            "row": row,
            "evidence": {
                "candidate": str(merged / "certificates/candidate_rows.jsonl"),
                "reachability": str(merged / "certificates/reachability_manifest.jsonl"),
                "runtime": str(merged / "runtime_replay.json"),
                "official_rgb": str(merged / "official_observability_v1/observability_manifest.jsonl"),
                "contact_sheet": observation.get("contact_sheet"),
            },
        })
    require(len(selected) == 119, f"unexpected path-first train PASS count {len(selected)}")
    return selected


def load_medium(path: Path) -> list[dict[str, Any]]:
    payload = read_json(path)
    selected = []
    for episode in payload["episodes"]:
        if episode.get("split") != "train":
            continue
        validation = episode.get("validation") or {}
        require(validation.get("runtime", {}).get("status") == "pass", f"medium runtime failed {episode['task_id']}")
        require(validation.get("runtime", {}).get("initial_success") is False, f"medium initial success {episode['task_id']}")
        require(validation.get("runtime", {}).get("final_success") is True, f"medium final failure {episode['task_id']}")
        require(validation.get("official_rgb", {}).get("passed") is True, f"medium RGB failed {episode['task_id']}")
        difficulty = episode["difficulty"]
        require(difficulty.get("lower_bound_complete") is True and int(difficulty.get("certified_lower_bound")) == 4,
                f"medium lower bound incomplete {episode['task_id']}")
        for evidence_ref in episode["evidence"].values():
            if not isinstance(evidence_ref, dict) or "path" not in evidence_ref:
                continue
            evidence_path = Path(evidence_ref["path"])
            require(sha256(evidence_path) == evidence_ref["sha256"],
                    f"Medium evidence SHA mismatch {evidence_path}")
        candidate_ref = episode["evidence"]["candidate"]
        candidate_path = Path(candidate_ref["path"])
        rows = read_jsonl(candidate_path)
        raw_row = rows[int(candidate_ref["record_index"])]
        require(raw_row.get("task_id") == episode.get("task_id"), f"medium candidate index mismatch {episode['task_id']}")
        runtime_ref = episode["evidence"]["runtime"]
        runtime_payload = read_json(Path(runtime_ref["path"]))
        runtime_rows = runtime_payload.get("results", runtime_payload if isinstance(runtime_payload, list) else [])
        runtime = runtime_rows[int(runtime_ref["record_index"])]
        row = materialize_train_row(
            raw_row, runtime["collision_convention"], str(episode["source_key"]),
            str(episode["episode_fingerprint"]),
        )
        selected.append({
            "source_key": str(episode["source_key"]),
            "source_row_index": int(episode["source_row_index"]),
            "split": "train",
            "scene_id": str(episode["scene_id"]),
            "task_id": str(episode["task_id"]),
            "episode_fingerprint": str(episode["episode_fingerprint"]),
            "evidence_class": "medium_lower_bound_runtime_rgb_verified",
            "difficulty": difficulty,
            "row": row,
            "evidence": episode["evidence"],
        })
    require(len({row["source_key"] for row in selected}) == 64, "unexpected Medium train source count")
    return selected


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "sources": len({row["source_key"] for row in records}),
        "episodes": len(records),
        "scenes": dict(sorted(Counter(row["scene_id"] for row in records).items())),
        "tasks": dict(sorted(Counter(row["row"].get("task_type") for row in records).items())),
        "relations": dict(sorted(Counter(relation(row["row"]) for row in records).items())),
        "category_pairs": dict(sorted(Counter("--".join(category_pair(row["row"])) for row in records).items(), key=lambda value: (-value[1], value[0]))),
        "certificate_upper_bounds": dict(sorted(Counter(row["difficulty"].get("certificate_upper_bound") for row in records).items(), key=lambda value: str(value[0]))),
        "first_success_steps": dict(sorted(Counter(row["difficulty"].get("first_success_step") for row in records).items(), key=lambda value: str(value[0]))),
        "evidence_classes": dict(sorted(Counter(row["evidence_class"] for row in records).items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--medium-inventory", type=Path, required=True)
    parser.add_argument("--path-first-root", type=Path, required=True)
    parser.add_argument("--canonical-eval-manifest", type=Path, required=True)
    parser.add_argument("--local-action-parents", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    path_first = load_path_first(args.path_first_root)
    medium = load_medium(args.medium_inventory)
    # Prefer the strongest certified episode for a source.  Multiple Medium
    # episodes are resolved by immutable episode fingerprint, never difficulty.
    per_source: dict[str, list[dict[str, Any]]] = {}
    for record in path_first + medium:
        per_source.setdefault(record["source_key"], []).append(record)
    selected = []
    for key in sorted(per_source):
        options = per_source[key]
        options.sort(key=lambda row: (
            0 if row["evidence_class"].startswith("medium_") else 1,
            row["episode_fingerprint"],
        ))
        selected.append(options[0])
    require(len(selected) == 141, f"unexpected verified source union {len(selected)}")

    canonical_eval = read_jsonl(args.canonical_eval_manifest)
    local_eval = read_jsonl(args.local_action_parents)
    train_sources = {row["source_key"] for row in selected}
    train_scenes = {row["scene_id"] for row in selected}
    canonical_eval_sources = {str(row.get("source_key")) for row in canonical_eval}
    canonical_eval_scenes = {str(row.get("scene_id")) for row in canonical_eval}
    local_eval_sources = {str(row.get("source_key")) for row in local_eval}
    local_eval_scenes = {str(row.get("scene_id")) for row in local_eval}
    isolation = {
        "version": VERSION,
        "canonical_development_eval": {
            "sources": len(canonical_eval_sources),
            "scenes": len(canonical_eval_scenes),
            "source_overlap": sorted(train_sources & canonical_eval_sources),
            "scene_overlap": sorted(train_scenes & canonical_eval_scenes),
        },
        "independent_local_action_evaluation": {
            "sources": len(local_eval_sources),
            "scenes": len(local_eval_scenes),
            "source_overlap": sorted(train_sources & local_eval_sources),
            "scene_overlap": sorted(train_scenes & local_eval_scenes),
            "policy": "permanent_evaluation_only_exclusion",
        },
    }
    require(not isolation["canonical_development_eval"]["source_overlap"], "canonical eval source leaked into train")
    require(not isolation["independent_local_action_evaluation"]["source_overlap"], "local-action source leaked into train")
    require(not isolation["independent_local_action_evaluation"]["scene_overlap"], "local-action scene leaked into train")

    summary = summarize(selected)
    largest_pair = max(summary["category_pairs"].values())
    gate = {
        "criteria_version": "r1_canonical_eval_train_pilot_design_20260914",
        "required": {
            "unique_train_sources": 200,
            "scenes": 20,
            "left_sources": 80,
            "right_sources": 80,
            "step4_episodes": 40,
            "step5_episodes": 40,
            "step6_episodes": 40,
            "max_category_pair_fraction": 0.35,
        },
        "observed": {
            "unique_train_sources": summary["sources"],
            "scenes": len(summary["scenes"]),
            "left_sources": summary["relations"].get("left", 0),
            "right_sources": summary["relations"].get("right", 0),
            "step4_episodes": summary["first_success_steps"].get(4, 0),
            "step5_episodes": summary["first_success_steps"].get(5, 0),
            "step6_episodes": summary["first_success_steps"].get(6, 0),
            "max_category_pair_fraction": largest_pair / len(selected),
        },
    }
    gate["passed"] = (
        gate["observed"]["unique_train_sources"] >= 200
        and gate["observed"]["scenes"] >= 20
        and gate["observed"]["left_sources"] >= 80
        and gate["observed"]["right_sources"] >= 80
        and gate["observed"]["step4_episodes"] >= 40
        and gate["observed"]["step5_episodes"] >= 40
        and gate["observed"]["step6_episodes"] >= 40
        and gate["observed"]["max_category_pair_fraction"] <= 0.35
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    inventory_rows = [{key: value for key, value in row.items() if key != "row"} for row in selected]
    write_jsonl(args.output_dir / "verified_train_source_inventory.jsonl", inventory_rows)
    write_jsonl(args.output_dir / "verified_train_manifest.jsonl", [row["row"] for row in selected])
    write_json(args.output_dir / "distribution_summary.json", summary)
    write_json(args.output_dir / "train_eval_isolation.json", isolation)
    write_json(args.output_dir / "training_readiness_gate.json", gate)
    provenance = {
        "version": VERSION,
        "inputs": {
            "medium_inventory": {"path": str(args.medium_inventory), "sha256": sha256(args.medium_inventory)},
            "path_first_aggregate": {"path": str(args.path_first_root / 'canary_aggregate.json'), "sha256": sha256(args.path_first_root / 'canary_aggregate.json')},
            "canonical_eval_manifest": {"path": str(args.canonical_eval_manifest), "sha256": sha256(args.canonical_eval_manifest)},
            "local_action_parents": {"path": str(args.local_action_parents), "sha256": sha256(args.local_action_parents)},
        },
        "selection": "one episode per source; certified Medium first, then episode fingerprint",
        "local_action_policy": "evaluation_only; never a candidate input",
    }
    write_json(args.output_dir / "provenance.json", provenance)
    files = sorted(path for path in args.output_dir.iterdir() if path.is_file() and path.name != "SHA256SUMS")
    atomic_text(args.output_dir / "SHA256SUMS", "".join(f"{sha256(path)}  {path.name}\n" for path in files))
    print(json.dumps({"summary": summary, "isolation": isolation, "gate": gate}, indent=2))


if __name__ == "__main__":
    main()
