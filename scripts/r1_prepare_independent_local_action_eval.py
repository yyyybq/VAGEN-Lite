#!/usr/bin/env python3
"""Construct and freeze exact d*=1/2 local states on R1-unseen scenes.

The source scope and all budgets are frozen before this script runs.  Formal
distance is computed using runtime-legal actions/collision only.  RGB and layout
quality are applied afterwards when selecting auditable policy observations;
they never remove states from the distance proof.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from r1_action_graph import forward_validated_predecessors
from r1_canonical_tasks import canonical_projective, score_observation
from r1_prepare_canonical_dev_eval32_floor_diagnostics import (
    apply_legal, exact_distance_le2,
)
from r1_projective_observability import (
    PROJECTIVE_OBSERVABILITY_VERSION, evaluate_projective_path,
    save_path_contact_sheet,
)
from r1_projective_path_first_prototype import (
    success_region_points, target_pose, target_shortlist,
)
from r1_reachability_audit import render_task, state_key
from r1_repair_pipeline import (
    SceneConstraints, pair_midpoint, projective_initial_geometry_discernible,
)
from vagen.envs.active_spatial.render.unified_renderer import UnifiedRenderGS


VERSION = "r1_independent_local_action_freeze_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open() if line.strip()]


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for value in values:
            handle.write(json.dumps(value, sort_keys=True) + "\n")
    temporary.replace(path)


def path_from_actions(
    item: dict[str, Any], pose: np.ndarray, actions: list[str], detector: Any,
) -> list[dict[str, Any]]:
    metric = canonical_projective(score_observation(item, pose))
    result = [{"action": None, "c2w": pose.tolist(), "canonical_metric": metric}]
    current = pose.copy()
    for action in actions:
        current, transition = apply_legal(current, action, detector)
        if not transition["legal"]:
            raise RuntimeError("exact-distance witness unexpectedly collides")
        metric = canonical_projective(score_observation(item, current))
        result.append({"action": action, "c2w": current.tolist(), "canonical_metric": metric})
    if not result[-1]["canonical_metric"]["success"]:
        raise RuntimeError("exact-distance witness does not terminate in canonical success")
    return result


def local_candidates(
    item: dict[str, Any], constraints: SceneConstraints,
    seed_cap: int, state_cap: int, reverse_depth_cap: int,
) -> tuple[dict[int, list[dict[str, Any]]], dict[str, Any]]:
    if reverse_depth_cap != 2:
        raise ValueError("independent local-state v1 implements the frozen reverse depth 2")
    scene = str(item["scene_id"])
    layout, _ = constraints.scene(scene)
    detector = constraints._collision_cache.get(scene)
    if detector is None or detector.structure_convention_status != "frozen":
        return {1: [], 2: []}, {"status": "collision_convention_unavailable"}
    midpoint = pair_midpoint(item)
    room_index = constraints.room_index(layout, midpoint[:2])
    if room_index is None:
        initial = np.asarray(item["init_camera"]["extrinsics"], dtype=float)
        room_index = constraints.room_index(layout, initial[:2, 3])
    points = success_region_points(item, constraints, room_index)
    targets = []
    target_counts = Counter()
    for point in points:
        layout_check = constraints.validate(
            item, point, initial_room_index=room_index, check_pair_distance=False,
        )
        if not layout_check["success"]:
            target_counts["layout_reject"] += 1
            continue
        for yaw in (0.0, -5.0, 5.0, -10.0, 10.0, -15.0, 15.0):
            target_counts["tested"] += 1
            pose = target_pose(item, point, yaw)
            metric = canonical_projective(score_observation(item, pose))
            if metric["success"]:
                target_counts["canonical_success"] += 1
                rank = (
                    float(np.linalg.norm(point[:2] - np.asarray(item["sample_target"], dtype=float)[:2])),
                    abs(yaw), state_key(pose),
                )
                targets.append((rank, point, pose, metric, layout_check))
    targets.sort(key=lambda row: row[0])
    targets = target_shortlist(targets, limit=seed_cap)

    candidates: dict[int, list[dict[str, Any]]] = {1: [], 2: []}
    seen: dict[int, set[tuple[float, ...]]] = {1: set(), 2: set()}
    rejection = Counter()
    for seed_index, (_, _, target, target_metric, target_layout) in enumerate(targets):
        first = forward_validated_predecessors(target, detector)
        proposals: list[tuple[int, np.ndarray]] = []
        for edge in first:
            if edge["rejection_reason"] is not None:
                rejection[str(edge["rejection_reason"])] += 1
                continue
            proposals.append((1, np.asarray(edge["predecessor"], dtype=float)))
            for second in forward_validated_predecessors(edge["predecessor"], detector):
                if second["rejection_reason"] is not None:
                    rejection[str(second["rejection_reason"])] += 1
                    continue
                proposals.append((2, np.asarray(second["predecessor"], dtype=float)))
        proposals.sort(key=lambda row: (row[0], state_key(row[1])))
        for reverse_depth, pose in proposals:
            proof = exact_distance_le2(item, pose, detector)
            distance = proof.get("exact_distance")
            if distance not in (1, 2):
                rejection["not_exact_d1_or_d2"] += 1
                continue
            distance = int(distance)
            if len(candidates[distance]) >= state_cap:
                rejection[f"d{distance}_candidate_cap_reached"] += 1
                continue
            key = state_key(pose)
            if key in seen[distance]:
                rejection["duplicate_state"] += 1
                continue
            seen[distance].add(key)
            metric = canonical_projective(score_observation(item, pose))
            layout_check = constraints.validate(
                item, pose[:3, 3], initial_room_index=room_index, check_pair_distance=False,
            )
            if not layout_check["success"]:
                rejection["initial_layout_reject"] += 1
                continue
            if not projective_initial_geometry_discernible(metric):
                rejection["initial_projection_reject"] += 1
                continue
            actions = sorted(tuple(path) for path in proof["paths"])[0]
            witness = path_from_actions(item, pose, list(actions), detector)
            candidates[distance].append({
                "seed_index": seed_index,
                "reverse_depth": reverse_depth,
                "pose_c2w": pose.tolist(),
                "state_key": list(key),
                "metric": metric,
                "layout": layout_check,
                "proof": proof,
                "witness_actions": list(actions),
                "witness_path": witness,
                "terminal_metric": target_metric,
                "terminal_layout": target_layout,
            })
        if all(len(candidates[d]) >= state_cap for d in (1, 2)):
            break
    return candidates, {
        "status": "complete",
        "success_seeds": len(targets),
        "target_counts": dict(target_counts),
        "rejections": dict(rejection),
        "candidate_counts": {str(key): len(value) for key, value in candidates.items()},
        "collision_convention": detector.convention_record(),
    }


async def render_candidate(
    item: dict[str, Any], candidate: dict[str, Any], detector: Any,
    renderer_url: str, output_dir: Path,
) -> dict[str, Any]:
    path = candidate["witness_path"]
    renderer = UnifiedRenderGS(
        render_backend="http", client_url=renderer_url, scene_id=str(item["scene_id"]),
    )
    images = await renderer.render_tasks([
        render_task(item, np.asarray(entry["c2w"], dtype=float)) for entry in path
    ])
    audit = evaluate_projective_path(item, path, images, detector)
    audit.update(save_path_contact_sheet(images, path, audit, output_dir))
    return audit


def round_robin(records: list[dict[str, Any]], scenes: list[str]) -> list[dict[str, Any]]:
    by_scene: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        by_scene[record["scene_id"]].append(record)
    for scene in scenes:
        by_scene[scene].sort(key=lambda row: (row["selection_digest"], row["source_key"]))
    result = []
    rank = 0
    while True:
        progress = False
        for scene in scenes:
            if rank < len(by_scene[scene]):
                result.append(by_scene[scene][rank])
                progress = True
        if not progress:
            break
        rank += 1
    return result


async def run(args: argparse.Namespace) -> None:
    scope = json.loads(args.scope.read_text())
    if not scope.get("selection_is_frozen_before_geometry_or_model_results"):
        raise RuntimeError("scope was not frozen before model results")
    budgets = scope["state_construction_budget"]
    seed_cap = int(budgets["success_seed_cap_per_source"])
    state_cap = int(budgets["candidate_state_cap_per_distance_per_source"])
    reverse_depth_cap = int(budgets["reverse_predecessor_max_depth"])
    parent_target = int(scope["selection_rule"]["parent_target"])
    source_rows = {}
    for split, info in scope["sources"].items():
        path = Path(info["path"])
        if sha256(path) != info["sha256"]:
            raise RuntimeError(f"source hash mismatch: {split}")
        source_rows[split] = read_jsonl(path)
    constraints = SceneConstraints(args.gs_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    generation_path = args.output_dir / "candidate_generation_checkpoint.jsonl"
    generated = read_jsonl(generation_path) if generation_path.exists() else []
    generated_keys = {row["source_key"] for row in generated}
    for position, record in enumerate(round_robin(scope["records"], scope["scenes"]), 1):
        if record["source_key"] in generated_keys:
            continue
        item = source_rows[record["split"]][int(record["source_row_index"])]
        try:
            candidates, details = local_candidates(
                item, constraints, seed_cap, state_cap, reverse_depth_cap,
            )
            row = {**record, "status": "geometry_complete", "details": details, "candidates": candidates}
        except Exception as error:
            row = {**record, "status": "implementation_error", "error": repr(error), "candidates": {"1": [], "2": []}}
        generated.append(row)
        atomic_jsonl(generation_path, generated)
        print(json.dumps({
            "phase": "geometry", "position": position, "source_key": record["source_key"],
            "scene_id": record["scene_id"], "status": row["status"],
            "d1": len(row.get("candidates", {}).get(1, row.get("candidates", {}).get("1", []))),
            "d2": len(row.get("candidates", {}).get(2, row.get("candidates", {}).get("2", []))),
        }), flush=True)

    # Model-free RGB qualification.  Try dual-distance parents first, then any
    # remaining single-distance parent, always in the pre-frozen round-robin order.
    qualification_path = args.output_dir / "rgb_qualification_checkpoint.jsonl"
    qualified = read_jsonl(qualification_path) if qualification_path.exists() else []
    qualified_by_key = {row["source_key"]: row for row in qualified}
    ordered = round_robin(generated, scope["scenes"])
    tiers = [
        [row for row in ordered if all(row.get("candidates", {}).get(str(d), row.get("candidates", {}).get(d, [])) for d in (1, 2))],
        [row for row in ordered if any(row.get("candidates", {}).get(str(d), row.get("candidates", {}).get(d, [])) for d in (1, 2))],
    ]
    selected: list[dict[str, Any]] = []
    selected_keys: set[str] = set()
    for tier_index, tier in enumerate(tiers):
        for row in tier:
            if len(selected) >= parent_target:
                break
            key = row["source_key"]
            if key in selected_keys:
                continue
            existing = qualified_by_key.get(key)
            if existing is None:
                item = source_rows[row["split"]][int(row["source_row_index"])]
                scene = str(row["scene_id"])
                constraints.scene(scene)
                detector = constraints._collision_cache[scene]
                state_results = {}
                for distance in (1, 2):
                    candidates = row.get("candidates", {}).get(str(distance), row.get("candidates", {}).get(distance, []))
                    attempts = []
                    accepted = None
                    for candidate_index, candidate in enumerate(candidates[:state_cap]):
                        destination = args.output_dir / "rgb_evidence" / key.replace(":", "_") / f"d{distance}_candidate{candidate_index}"
                        try:
                            audit = await render_candidate(item, candidate, detector, args.renderer_url, destination)
                        except Exception as error:
                            audit = {"passed": False, "reasons": ["renderer_error"], "renderer_error": repr(error)}
                        attempts.append({"candidate_index": candidate_index, "audit": audit})
                        if audit.get("passed"):
                            accepted = {"candidate_index": candidate_index, "candidate": candidate, "audit": audit}
                            break
                    state_results[str(distance)] = {"accepted": accepted, "attempts": attempts}
                existing = {
                    "source_key": key, "split": row["split"],
                    "source_row_index": row["source_row_index"], "scene_id": scene,
                    "selection_digest": row["selection_digest"],
                    "r1_development_seen": row["r1_development_seen"],
                    "v46_training_scene_seen": row["v46_training_scene_seen"],
                    "v46_training_exact_source_seen": row["v46_training_exact_source_seen"],
                    "tier_attempted": tier_index, "states": state_results,
                    "accepted_distances": [d for d in (1, 2) if state_results[str(d)]["accepted"] is not None],
                }
                qualified.append(existing)
                qualified_by_key[key] = existing
                atomic_jsonl(qualification_path, qualified)
                print(json.dumps({
                    "phase": "rgb", "source_key": key, "scene_id": scene,
                    "accepted_distances": existing["accepted_distances"],
                }), flush=True)
            needed = (len(existing["accepted_distances"]) == 2) if tier_index == 0 else bool(existing["accepted_distances"])
            if needed:
                selected.append(existing)
                selected_keys.add(key)
        if len(selected) >= parent_target:
            break
    selected = selected[:parent_target]
    scene_count = len({row["scene_id"] for row in selected})
    if len(selected) != parent_target or scene_count < 10:
        raise RuntimeError(f"independent parent gate failed: parents={len(selected)}, scenes={scene_count}")

    policy_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for parent_index, selected_row in enumerate(selected):
        item = source_rows[selected_row["split"]][int(selected_row["source_row_index"])]
        for distance in sorted(selected_row["accepted_distances"]):
            accepted = selected_row["states"][str(distance)]["accepted"]
            candidate = accepted["candidate"]
            audit = accepted["audit"]
            local_index = len(policy_rows)
            derived = copy.deepcopy(item)
            derived["task_id"] = f"r1_independent_local_{parent_index:03d}_d{distance}"
            derived["init_camera"] = dict(derived["init_camera"])
            derived["init_camera"]["extrinsics"] = candidate["pose_c2w"]
            derived["source_identity"] = {
                "experiment_role": "independent_scene_local_action_diagnostic",
                "source_key": selected_row["source_key"],
                "split": selected_row["split"],
                "source_row_index": int(selected_row["source_row_index"]),
                "parent_eval_index": parent_index,
                "local_state_index": local_index,
                "exact_distance": distance,
            }
            policy_rows.append(derived)
            frame_path = Path(audit["frame_images"][0])
            audit_rows.append({
                "version": VERSION,
                "local_state_index": local_index,
                "parent_eval_index": parent_index,
                "source_key": selected_row["source_key"],
                "split": selected_row["split"],
                "source_row_index": int(selected_row["source_row_index"]),
                "scene_id": selected_row["scene_id"],
                "object_pair": [f"{obj.get('id')}:{obj.get('label')}" for obj in item.get("target_object", {}).get("objects", [])],
                "relation": item.get("target_region", {}).get("params", {}).get("relation"),
                "r1_development_seen": False,
                "v46_training_scene_seen": selected_row["v46_training_scene_seen"],
                "v46_training_exact_source_seen": selected_row["v46_training_exact_source_seen"],
                "pose_c2w": candidate["pose_c2w"],
                "exact_distance": distance,
                "optimal_first_actions_audit_only": candidate["proof"]["optimal_first_actions"],
                "optimal_paths_audit_only": candidate["proof"]["paths"],
                "enumeration_audit_only": candidate["proof"],
                "prior_rgb_audit_only": {"path": str(frame_path), "sha256": sha256(frame_path)},
                "official_observability_audit_only": audit,
                "policy_oracle_exclusion": [
                    "exact_distance", "optimal_first_actions", "optimal_paths", "terminal_pose",
                    "certificate_actions", "planner_output", "3d_geometry", "evidence_paths",
                ],
            })

    frozen_dir = args.output_dir / "frozen_input"
    policy_path = frozen_dir / "local_policy_rows.jsonl"
    audit_path = frozen_dir / "local_state_audit.jsonl"
    selected_path = frozen_dir / "parent_sources.jsonl"
    atomic_jsonl(policy_path, policy_rows)
    atomic_jsonl(audit_path, audit_rows)
    atomic_jsonl(selected_path, selected)
    counts = {
        "parent_sources": len(selected), "scenes": scene_count, "local_states": len(policy_rows),
        "d1": sum(row["exact_distance"] == 1 for row in audit_rows),
        "d2": sum(row["exact_distance"] == 2 for row in audit_rows),
        "scene": dict(sorted(Counter(row["scene_id"] for row in selected).items())),
        "split": dict(sorted(Counter(row["split"] for row in selected).items())),
    }
    protocol = {
        "version": VERSION,
        "scope": {"path": str(args.scope.resolve()), "sha256": sha256(args.scope)},
        "analysis_plan_frozen_before_policy_inference": scope[
            "analysis_plan_frozen_before_policy_inference"
        ],
        "selection": scope["selection_rule"], "budgets": budgets,
        "counts": counts,
        "distance_proof": "complete_formal_runtime_enumeration_to_any_canonical_success_through_depth_2_without_rgb_quality_filter",
        "policy_contract": {
            "history": "no_concat_system_plus_current_observation_only",
            "prompt_format": "free_think",
            "audit_only_fields_not_visible": [
                "distance", "optimal actions", "certificate", "terminal pose", "planner", "3D truth",
            ],
        },
        "decoding": {
            "temperature": 0.8, "top_p": 0.95, "max_tokens_per_turn": 384,
            "n": 1, "stop": ["</action>"], "do_sample": True,
            "paired_seed": "2026091700 + local_state_index",
        },
        "runtime": {
            "camera": "canonical_h1_from_frozen_candidate_intrinsics_and_pose",
            "canonical_metric": "canonical_spatial_task_h1_v1", "action_space": "strafe",
            "step_translation_m": 0.3, "step_rotation_deg": 20.0,
            "render_resolution": [256, 256], "collision": "frozen_per_scene",
        },
        "observability": PROJECTIVE_OBSERVABILITY_VERSION,
        "local_decision_seed_base": 2026091700,
        "policy_input": {"path": str(policy_path.resolve()), "sha256": sha256(policy_path)},
        "audit_manifest": {"path": str(audit_path.resolve()), "sha256": sha256(audit_path)},
        "parent_manifest": {"path": str(selected_path.resolve()), "sha256": sha256(selected_path)},
    }
    atomic_json(frozen_dir / "independent_local_action_protocol.json", protocol)
    atomic_json(args.output_dir / "prepare_summary.json", {
        "version": VERSION, "counts": counts,
        "attempted_sources": len(generated), "rgb_qualified_sources": len(qualified),
        "geometry_status": dict(Counter(row["status"] for row in generated)),
        "policy_input_sha256": protocol["policy_input"]["sha256"],
        "audit_manifest_sha256": protocol["audit_manifest"]["sha256"],
    })
    print(json.dumps({"complete": True, "counts": counts}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", type=Path, required=True)
    parser.add_argument("--gs-root", type=Path, required=True)
    parser.add_argument("--renderer-url", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
