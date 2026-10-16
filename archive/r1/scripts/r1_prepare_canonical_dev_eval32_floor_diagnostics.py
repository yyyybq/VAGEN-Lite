#!/usr/bin/env python3
"""Freeze local d*=1/2 states and audit the valid 64-episode batch baseline.

This preparation is policy-free and renderer-free.  It uses only the frozen
32-source manifest, formal Active Spatial transitions/collision, canonical H1
metrics, and the already validated policy ledgers.  Oracle fields are written
to a separate audit file and are never added to policy input rows.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from r1_action_graph import ACTIONS, TRANSLATION_ACTIONS, forward_transition
from r1_canonical_tasks import canonical_projective, score_observation
from r1_reachability_audit import state_key
from vagen.envs.active_spatial.collision_detector import create_collision_detector


VERSION = "r1_canonical_dev_eval32_floor_diagnostic_freeze_v1"
INVERSE = {
    "move_forward": "move_backward", "move_backward": "move_forward",
    "move_left": "move_right", "move_right": "move_left",
    "turn_left": "turn_right", "turn_right": "turn_left",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def metric(item: dict[str, Any], pose: np.ndarray) -> dict[str, Any]:
    return canonical_projective(score_observation(item, pose))


class DetectorCache:
    def __init__(self, gs_root: Path):
        self.gs_root = gs_root
        self.cache: dict[tuple[str, float], Any] = {}

    def get(self, item: dict[str, Any]) -> Any:
        scene = str(item["scene_id"])
        sign = float(item["collision_convention"]["structure_y_sign"])
        key = (scene, sign)
        if key not in self.cache:
            detector = create_collision_detector({
                "camera_radius": 0.15,
                "floor_height": 0.3,
                "ceiling_height": 2.5,
                "safety_margin": 0.05,
                "enable_object_collision": True,
                "enable_boundary_collision": True,
                "structure_y_sign_overrides": {scene: sign},
            })
            if not detector.load_scene_from_gs_root(str(self.gs_root), scene):
                raise RuntimeError(f"failed to load collision assets for {scene}")
            actual = detector.convention_record()
            if actual.get("status") != "frozen" or float(actual.get("structure_y_sign")) != sign:
                raise RuntimeError(f"collision convention mismatch for {scene}: {actual}")
            self.cache[key] = detector
        return self.cache[key]


def apply_legal(pose: np.ndarray, action: str, detector: Any) -> tuple[np.ndarray, dict[str, Any]]:
    candidate = forward_transition(pose, action)
    collision = None
    if action in TRANSLATION_ACTIONS:
        collision = detector.check_collision(candidate[:3, 3], previous_position=pose[:3, 3])
        if collision.has_collision:
            return pose.copy(), {
                "legal": False, "collision": True, "collision_type": collision.collision_type,
            }
    return candidate, {"legal": True, "collision": False, "collision_type": None}


def exact_distance_le2(item: dict[str, Any], pose: np.ndarray, detector: Any) -> dict[str, Any]:
    """Enumerate every legal sequence through depth two and all optimal first actions."""
    initial = metric(item, pose)
    if initial["success"]:
        return {"exact_distance": 0, "optimal_first_actions": [], "paths": [[]]}
    first_states: list[tuple[str, np.ndarray]] = []
    collision_rejections = Counter()
    one_step: list[list[str]] = []
    for action in ACTIONS:
        nxt, transition = apply_legal(pose, action, detector)
        if not transition["legal"]:
            collision_rejections[transition["collision_type"] or "collision"] += 1
            continue
        first_states.append((action, nxt))
        if metric(item, nxt)["success"]:
            one_step.append([action])
    if one_step:
        return {
            "exact_distance": 1,
            "optimal_first_actions": sorted({path[0] for path in one_step}),
            "paths": one_step,
            "legal_one_step_states": len(first_states),
            "collision_rejections": dict(collision_rejections),
        }
    two_step: list[list[str]] = []
    for first, first_pose in first_states:
        for second in ACTIONS:
            nxt, transition = apply_legal(first_pose, second, detector)
            if not transition["legal"]:
                collision_rejections[transition["collision_type"] or "collision"] += 1
                continue
            if metric(item, nxt)["success"]:
                two_step.append([first, second])
    return {
        "exact_distance": 2 if two_step else None,
        "optimal_first_actions": sorted({path[0] for path in two_step}),
        "paths": two_step,
        "legal_one_step_states": len(first_states),
        "collision_rejections": dict(collision_rejections),
        "complete_through_depth": 2,
    }


def resolve(path: str, repository_root: Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else repository_root / value


def evidence_record(audit: dict[str, Any], repository_root: Path) -> tuple[Path, dict[str, Any]]:
    evidence = audit["evidence_audit_only"]["observability"]
    path = resolve(evidence["path"], repository_root)
    if sha256(path) != evidence["sha256"]:
        raise RuntimeError(f"observability evidence hash mismatch: {path}")
    record = read_jsonl(path)[int(evidence["record_index"])]
    if int(record["source_row_index"]) != int(audit["source_row_index"]):
        raise RuntimeError("observability record points to another source")
    return path, record


def build_local_states(
    rows: list[dict[str, Any]], audits: list[dict[str, Any]], detectors: DetectorCache,
    repository_root: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    policy_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    for eval_index, (item, audit) in enumerate(zip(rows, audits)):
        detector = detectors.get(item)
        pose = np.asarray(audit["initial_pose_c2w"], dtype=float)
        actions = [str(value) for value in audit["certificate_actions_audit_only"]]
        _, evidence = evidence_record(audit, repository_root)
        if [frame.get("action") for frame in evidence["frames"]][1:] != actions:
            raise RuntimeError(f"evidence/certificate action mismatch for {audit['source_key']}")
        poses = [pose.copy()]
        first_success = None
        for step, action in enumerate(actions, 1):
            pose, transition = apply_legal(pose, action, detector)
            if not transition["legal"]:
                raise RuntimeError(f"certified path collides at {audit['source_key']} step {step}")
            poses.append(pose.copy())
            if metric(item, pose)["success"] and first_success is None:
                first_success = step
        if first_success != int(audit["difficulty"]["first_success_step"]):
            raise RuntimeError(f"certificate first-success mismatch for {audit['source_key']}")
        if len(evidence["frame_images"]) != len(poses):
            raise RuntimeError(f"evidence frame count mismatch for {audit['source_key']}")
        candidates: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for path_step, state in enumerate(poses[:-1]):
            exact = exact_distance_le2(item, state, detector)
            distance = exact["exact_distance"]
            if distance in (1, 2):
                candidates[int(distance)].append({"path_step": path_step, "pose": state, "proof": exact})
        for distance in (1, 2):
            if not candidates[distance]:
                continue
            # Prefer the latest certified-path state, with a stable state-key tie-break.
            selected = sorted(candidates[distance], key=lambda value: (-value["path_step"], state_key(value["pose"])))[0]
            local_index = len(policy_rows)
            derived = copy.deepcopy(item)
            derived["task_id"] = f"r1_floor_local_d{distance}_{eval_index:03d}"
            derived["init_camera"]["extrinsics"] = selected["pose"].tolist()
            derived["source_identity"] = {
                "experiment_role": "development_diagnostic_derived_state",
                "source_row_index": int(audit["source_row_index"]),
                "split": audit["split"],
                "episode_fingerprint": audit["episode_fingerprint"],
                "parent_eval_index": eval_index,
                "local_state_index": local_index,
                "exact_distance": distance,
            }
            policy_rows.append(derived)
            rgb_path = resolve(evidence["frame_images"][selected["path_step"]], repository_root)
            if not rgb_path.is_file():
                raise RuntimeError(f"missing frozen derived-state RGB: {rgb_path}")
            audit_rows.append({
                "version": VERSION,
                "local_state_index": local_index,
                "parent_eval_index": eval_index,
                "source_key": audit["source_key"],
                "split": audit["split"],
                "source_row_index": int(audit["source_row_index"]),
                "scene_id": audit["scene_id"],
                "episode_fingerprint": audit["episode_fingerprint"],
                "path_step": selected["path_step"],
                "pose_c2w": selected["pose"].tolist(),
                "exact_distance": distance,
                "optimal_first_actions_audit_only": selected["proof"]["optimal_first_actions"],
                "optimal_paths_audit_only": selected["proof"]["paths"],
                "enumeration_audit_only": selected["proof"],
                "prior_rgb_audit_only": {"path": str(rgb_path), "sha256": sha256(rgb_path)},
                "policy_oracle_exclusion": [
                    "exact_distance", "optimal_first_actions", "optimal_paths", "certificate_actions",
                    "terminal_pose", "planner_output", "3d_geometry",
                ],
            })
    return policy_rows, audit_rows


def analyse_baseline(
    model_key: str, ledger_path: Path, rows: list[dict[str, Any]], detectors: DetectorCache,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    ledger = json.loads(ledger_path.read_text())
    episode_rows = []
    aggregate = Counter()
    for key in sorted(ledger["episodes"]):
        episode = ledger["episodes"][key]
        index = int(episode["eval_index"])
        item = rows[index]
        detector = detectors.get(item)
        pose = np.asarray(item["init_camera"]["extrinsics"], dtype=float)
        previous_action = None
        first_out_of_frame = None
        first_success = None
        primitive_rows = []
        prior_metric = metric(item, pose)
        for turn in episode["turns"]:
            executed = [str(value) for value in turn.get("parsed_actions", [])][:int(turn.get("executed_action_count", 0))]
            if len(executed) > 1:
                aggregate["multi_action_turns"] += 1
            batch_had_out = False
            batch_had_success = False
            for batch_offset, action in enumerate(executed):
                pre = pose.copy()
                pose, transition = apply_legal(pose, action, detector)
                current = metric(item, pose)
                step = len(primitive_rows) + 1
                out = not bool(current["gates"]["inside_frame"])
                if out and first_out_of_frame is None:
                    first_out_of_frame = step
                if current["success"] and first_success is None:
                    first_success = step
                margin_before = prior_metric.get("relation_margin_px")
                margin_after = current.get("relation_margin_px")
                visibility_keys = ("in_front", "visible", "min_area", "inside_frame")
                degraded = [name for name in visibility_keys if prior_metric["gates"].get(name) and not current["gates"].get(name)]
                improved_margin_degraded = (
                    margin_before is not None and margin_after is not None and margin_after > margin_before and bool(degraded)
                )
                repeated = previous_action == action
                inverse = previous_action is not None and INVERSE.get(previous_action) == action
                primitive_rows.append({
                    "primitive_step": step, "turn_index": int(turn["turn_index"]),
                    "batch_offset": batch_offset, "batch_size": len(executed), "action": action,
                    "pre_pose_c2w": pre.tolist(), "post_pose_c2w": pose.tolist(),
                    "transition": transition, "canonical_metric": current,
                    "visibility_degraded_gates": degraded,
                    "relation_margin_improved_while_visibility_degraded": improved_margin_degraded,
                    "repeat_of_previous": repeated, "inverse_of_previous": inverse,
                })
                aggregate["primitive_actions"] += 1
                aggregate["turn_actions" if action.startswith("turn_") else "translation_actions"] += 1
                aggregate["collision_attempts"] += int(not transition["legal"])
                aggregate["repeated_adjacent_actions"] += int(repeated)
                aggregate["inverse_adjacent_actions"] += int(inverse)
                aggregate["margin_improve_visibility_degrade_steps"] += int(improved_margin_degraded)
                batch_had_out = batch_had_out or out
                batch_had_success = batch_had_success or bool(current["success"])
                prior_metric = current
                previous_action = action
            expected = np.asarray(turn["post_pose_c2w"], dtype=float)
            error = float(np.max(np.abs(pose - expected)))
            if error > 1e-7:
                raise RuntimeError(f"baseline reconstruction mismatch {model_key} {key} turn {turn['turn_index']}: {error}")
            if len(executed) > 1 and batch_had_out:
                aggregate["multi_action_turns_with_out_of_frame"] += 1
            if len(executed) > 1 and batch_had_success:
                aggregate["multi_action_turns_with_success_before_batch_end"] += 1
        aggregate["episodes"] += 1
        aggregate["episodes_ever_out_of_frame"] += int(first_out_of_frame is not None)
        aggregate["episodes_ever_success_midprimitive"] += int(first_success is not None)
        episode_rows.append({
            "model_key": model_key, "eval_index": index, "source_key": episode["source_key"],
            "first_out_of_frame_step": first_out_of_frame, "first_success_step_reconstructed": first_success,
            "turns": len(episode["turns"]), "primitive_actions": len(primitive_rows),
            "primitive_trace": primitive_rows,
        })
    summary = {"model_key": model_key, "ledger": str(ledger_path), "ledger_sha256": sha256(ledger_path), **dict(aggregate)}
    summary["turn_fraction"] = aggregate["turn_actions"] / aggregate["primitive_actions"]
    return summary, episode_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--valid-eval-root", type=Path, required=True)
    parser.add_argument("--gs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    repository_root = next(parent for parent in args.protocol.resolve().parents if (parent / "vagen").is_dir())
    policy_path = Path(protocol["policy_input"]["path"])
    audit_path = Path(protocol["audit_manifest"]["path"])
    if sha256(policy_path) != protocol["policy_input"]["sha256"] or sha256(audit_path) != protocol["audit_manifest"]["sha256"]:
        raise RuntimeError("frozen input hash mismatch")
    rows, audits = read_jsonl(policy_path), read_jsonl(audit_path)
    if len(rows) != 32 or len(audits) != 32:
        raise RuntimeError("expected the frozen 32-source manifest")
    detectors = DetectorCache(args.gs_root)
    local_policy, local_audit = build_local_states(rows, audits, detectors, repository_root)
    if len(local_policy) > 64:
        raise RuntimeError("local diagnostic exceeds frozen 64-state cap")
    frozen = args.output_dir / "frozen_input"
    write_jsonl(frozen / "local_policy_rows.jsonl", local_policy)
    write_jsonl(frozen / "local_state_audit.jsonl", local_audit)
    baseline_summaries, baseline_rows = [], []
    for model_key in ("pretrained", "v46_step250"):
        summary, details = analyse_baseline(
            model_key, args.valid_eval_root / model_key / "episode_ledger.json", rows, detectors,
        )
        baseline_summaries.append(summary); baseline_rows.extend(details)
    write_json(args.output_dir / "baseline_readonly_summary.json", {
        "version": VERSION, "models": baseline_summaries,
        "interpretation": "descriptive_only_not_causal_evidence_for_batch_execution_failure",
    })
    write_jsonl(args.output_dir / "baseline_readonly_per_episode.jsonl", baseline_rows)
    diagnostic_protocol = {
        "version": VERSION,
        "parent_protocol": {"path": str(args.protocol.resolve()), "sha256": sha256(args.protocol)},
        "valid_baseline_root": str(args.valid_eval_root.resolve()),
        "policy_input": {"path": str((frozen / "local_policy_rows.jsonl").resolve()), "sha256": sha256(frozen / "local_policy_rows.jsonl")},
        "audit_manifest": {"path": str((frozen / "local_state_audit.jsonl").resolve()), "sha256": sha256(frozen / "local_state_audit.jsonl")},
        "counts": {
            "parent_sources": 32, "local_states": len(local_policy),
            "d1": sum(row["exact_distance"] == 1 for row in local_audit),
            "d2": sum(row["exact_distance"] == 2 for row in local_audit),
        },
        "selection": "at_most_one_state_per_parent_source_per_exact_distance; latest_certificate_path_step_then_state_key",
        "distance_proof": "complete_formal_runtime_enumeration_to_any_canonical_success_through_depth_2_without_rgb_quality_filter",
        "local_decision_seed_base": 2026091600,
        "replan_seed_base": int(protocol["paired_seed_base"]),
        "runtime": protocol["environment"], "decoding": protocol["decoding"], "policy_contract": protocol["policy_contract"],
    }
    write_json(frozen / "floor_diagnostic_protocol.json", diagnostic_protocol)
    hashes = []
    for path in sorted(p for p in args.output_dir.rglob("*") if p.is_file() and p.name != "SHA256SUMS"):
        hashes.append(f"{sha256(path)}  {path.relative_to(args.output_dir)}")
    (args.output_dir / "SHA256SUMS").write_text("\n".join(hashes) + "\n")
    print(json.dumps({"counts": diagnostic_protocol["counts"], "baseline": baseline_summaries}, indent=2))


if __name__ == "__main__":
    main()
