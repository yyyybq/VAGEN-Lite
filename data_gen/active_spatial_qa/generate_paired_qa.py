#!/usr/bin/env python3
"""Generate a versioned paired Active Spatial Yes/No bank.

This is deliberately a thin adapter around the existing Active JSONL and
success implementation.  It is resumable by sample_id and records failures.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from .contract import QA_SCHEMA_VERSION, ACTIVE_TASK_TYPES, build_task_contract, evaluate_state
from .image_contract_v2 import VERSION as IMAGE_CONTRACT_VERSION, pose_from_forward, pose_legality, classify_scene_pose


def _load_scene_context(scene_root: str | None, scene_id: str):
    if not scene_root:
        return None
    from vagen.envs.active_spatial.collision_detector import CollisionDetector
    detector = CollisionDetector(floor_height=0.3, ceiling_height=2.5)
    scene_path = Path(scene_root) / scene_id
    if not (scene_path/'labels.json').is_file() or not (scene_path/'structure.json').is_file():
        return None
    if not detector.load_scene(scene_path, scene_id=scene_id):
        return None
    # Reuse the detector's convention-selected room profiles, not raw structure
    # coordinates with an independently guessed Y sign.
    from data_gen.active_spatial_pipeline.layout_quality import LayoutGeometry
    layout = LayoutGeometry(room_polys=[p.tolist() for p in detector.room_profiles],
                            wall_segments=[(a.tolist(), b.tolist()) for a, b in detector.wall_segments])
    return detector, layout


def _pose_from_forward(position: Any, forward: Any) -> np.ndarray:
    return pose_from_forward(position, forward)


def _parent_id(item: dict[str, Any]) -> str:
    raw = json.dumps({k: item.get(k) for k in (
        "scene_id", "task_type", "object_label", "preset", "distance",
        "target_region", "target_object", "task_description",
    )}, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha1(raw.encode()).hexdigest()[:16]


def _question(item: dict[str, Any]) -> str:
    task = str(item.get("task_type") or "")
    label = str(item.get("object_label") or "the target object")
    desc = str(item.get("task_description") or "the specified spatial goal")
    suffix = {
        "absolute_positioning": "including the requested distance, facing, and visibility conditions",
        "delta_control": "including the reference-relative target, facing, and visibility conditions",
        "equidistance": "including equal distances, midpoint facing, and visibility of both objects",
        "projective_relations": "including the requested relation, in-frame visibility, and canonical margin gates",
        "centering": "including the requested centering relation and required visibility",
        "occlusion_alignment": "including the occlusion/alignment and visibility conditions",
        "fov_inclusion": "including full-bounding-box inclusion, safe center margins, and visibility",
        "size_distance_invariance": "including equal apparent size and required visibility",
        "apparent_size_ordering": "including the requested size ratio and required visibility",
        "screen_occupancy": "including the requested screen fraction, facing, and visibility conditions",
    }.get(task, "including every condition used by the Active success predicate")
    return f"Does the current view satisfy this Active Spatial goal for {label}: {desc}, {suffix}? Answer Yes or No."


def _state_variants(item: dict[str, Any]) -> list[tuple[str, np.ndarray, str, str]]:
    region = item.get("target_region") or {}
    pos = region.get("sample_point")
    if pos is None:
        pos = item.get("sample_target")
    fwd = region.get("sample_forward")
    if fwd is None:
        fwd = (item.get("camera_params") or {}).get("forward")
    if not (pos and fwd):
        raise ValueError("missing sample target pose")
    base = _pose_from_forward(pos, fwd)
    # Geometry legality is a prerequisite for every emitted state. The
    # original Active pipeline repairs/rejects samples against room polygons;
    # this adapter must never manufacture an outdoor camera and call it a
    # controlled negative. A collision detector is injected by production
    # callers when scene assets are available; without it, status remains
    # coordinate-unconfirmed and is recorded rather than guessed.
    # Positive variants are valid region samples when the task generator has
    # explicit samples; otherwise the canonical sample is retained once.
    variants: list[tuple[str, np.ndarray, str, str]] = [("positive_target", base, "target_sample", "positive_candidate")]
    params = region.get("params") or {}
    if region.get("type") == "circle" and params.get("center") and params.get("radius"):
        c = list(params["center"]) + [float(pos[2])]
        object_center = np.asarray(params.get("object_center", c), dtype=float)
        for angle in (math.pi / 2, math.pi):
            p2 = [c[0] + float(params["radius"]) * math.cos(angle), c[1] + float(params["radius"]) * math.sin(angle), c[2]]
            # Keep every positive candidate aligned to the same facing/visibility
            # condition used by Active, rather than reusing the original yaw.
            toward = object_center - np.asarray(p2, dtype=float)
            variants.append((f"positive_region_{len(variants)}", _pose_from_forward(p2, toward), "region_sample", "positive_candidate"))
    # Controlled failures: position displacement and orientation reversal.
    p = np.asarray(pos, dtype=float)
    variants.append(("negative_position_offset", _pose_from_forward(p + np.array([1.25, 0.0, 0.0]), fwd), "controlled_offset", "negative_candidate"))
    variants.append(("negative_orientation_flip", _pose_from_forward(p, -np.asarray(fwd, dtype=float)), "controlled_flip", "negative_candidate"))
    return variants


def _public_observation(item: dict[str, Any], state_id: str, pose: np.ndarray, state: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build model-visible input without leaking oracle state.

    A generated pose is not allowed to reuse an image rendered at another pose.
    State-aware pipeline records may provide an image explicitly; otherwise the
    image field is left null and the sample is marked unobservable by the caller.
    """
    state = state or {}
    image_path = state.get("image_path") or state.get("image")
    if image_path is None and state.get("image_paths"):
        image_path = state["image_paths"][0]
    return {
        "image_path": image_path,
        "state_id": state_id,
        "observation_config": state.get("observation_config", "matched_history" if item.get("task_type") == "delta_control" else "single_image"),
        "history": state.get("history", item.get("observation_history", [])),
        "camera_pose_available": False,
    }


def _state_variants_with_metadata(item: dict[str, Any]):
    """Yield (state_id, pose, source, provenance, metadata) for existing states."""
    states = item.get("states") or item.get("observation_states")
    if states:
        for i, state in enumerate(states):
            if not isinstance(state, dict):
                continue
            raw_pose = state.get("c2w")
            if raw_pose is None:
                raw_pose = state.get("camera_pose")
            if raw_pose is None:
                raw_pose = state.get("pose")
            if raw_pose is None and state.get("position") is not None and state.get("forward") is not None:
                pose = _pose_from_forward(state["position"], state["forward"])
            elif raw_pose is not None:
                pose = np.asarray(raw_pose, dtype=float)
            else:
                continue
            if pose.shape != (4, 4):
                continue
            yield (str(state.get("state_id") or f"state_{i}"), pose,
                   str(state.get("source") or "existing_state"),
                   str(state.get("provenance") or "pipeline_state"), state)
        return
    for state_id, pose, source, provenance in _state_variants(item):
        yield state_id, pose, source, provenance, {}


def generate(args: argparse.Namespace) -> dict[str, Any]:
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / "manifest.jsonl"
    done_ids = set(); parent_splits: dict[str, str] = {}; split_conflicts = 0
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            if json.loads(line).get('image_contract_version') != IMAGE_CONTRACT_VERSION:
                raise ValueError('legacy pose bank cannot be resumed with v2; use a separate versioned output directory')
            try:
                old = json.loads(line); done_ids.add(old["sample_id"])
                parent_splits.setdefault(str(old.get("parent_goal_id")), str(old.get("split")))
            except Exception: pass
    split_scenes = {s.strip() for s in args.split_scenes.split(",") if s.strip()} if args.split_scenes else None
    contexts = {}
    written = 0; errors = 0; by_task: dict[str, int] = {}; label_counts = {"Yes": 0, "No": 0}
    with open(args.input, encoding="utf-8") as src, manifest.open("a", encoding="utf-8") as dst:
        for line_no, line in enumerate(src):
            if args.limit and written >= args.limit: break
            if not line.strip(): continue
            try:
                item = json.loads(line)
                task = str(item.get("task_type") or "")
                if task not in ACTIVE_TASK_TYPES or (args.task_type and task != args.task_type): continue
                scene = str(item.get("scene_id") or "")
                if split_scenes is not None and scene not in split_scenes: continue
                parent = _parent_id(item)
                split = args.split or ("test" if split_scenes and scene in split_scenes else "train")
                if parent in parent_splits and parent_splits[parent] != split:
                    split_conflicts += 1
                    continue
                parent_splits[parent] = split
                for state_id, pose, source, provenance, state_meta in _state_variants_with_metadata(item):
                    sid = hashlib.sha1(f"{parent}:{state_id}:{split}".encode()).hexdigest()[:20]
                    if sid in done_ids: continue
                    try:
                        pose_arr = np.asarray(pose, dtype=float)
                        if pose_arr.shape != (4, 4) or not np.isfinite(pose_arr).all():
                            raise ValueError('invalid_pose_matrix')
                        geometry = pose_legality(pose_arr[:3, 3], pose_arr[:3, 2], min_height=0.3, max_height=2.5)
                        if geometry.get("valid"):
                            scene_root = getattr(args, 'scene_root', '')
                            if scene_root:
                                if scene not in contexts:
                                    contexts[scene] = _load_scene_context(scene_root, scene)
                                context = contexts[scene]
                                geometry = classify_scene_pose(pose_arr[:3, 3], context[0], context[1]) if context else {"status": "coordinate_unconfirmed", "reasons": ["scene_context_unavailable"]}
                            else:
                                geometry = {"status": "coordinate_unconfirmed", "reasons": ["scene_root_not_provided"]}
                        result = evaluate_state(item, pose)
                        answer = ("Yes" if result["success"] else "No") if geometry.get('status') == 'legal_indoor' else None
                        # Keep the predicate diagnostics outside model-visible fields.
                        record = {
                            "sample_id": sid, "parent_goal_id": parent, "scene_id": scene,
                            "task_type": task, "split": split,
                            "public_observation": _public_observation(item, state_id, pose, state_meta),
                            "question": _question(item), "private_answer": answer,
                            "reference_state_id": (state_meta.get("reference_state_id") or ("initial" if task == "delta_control" else None)), "state_id": state_id,
                            "state_pose_c2w": np.asarray(pose, dtype=float).tolist(),
                            "image_contract_version": IMAGE_CONTRACT_VERSION,
                            "source": source, "provenance": provenance,
                            "success_predicate_version": result["predicate_version"],
                            "runtime_backend": ("canonical_task_metrics" if result["predicate_version"] == "canonical_spatial_task_h1_v1" else "spatial_potential_field"),
                            "success_threshold": result.get("threshold", "canonical_gates"),
                            "complete_goal": {
                                "task_description": item.get("task_description"),
                                "target_object": item.get("target_object"),
                                "target_region": item.get("target_region"),
                                "task_params": item.get("task_params", {}),
                                "preset": item.get("preset"), "distance": item.get("distance"),
                            },
                            "source_manifest": str(args.input),
                            "source_task_metric_version": item.get("canonical_task_metric_version"),
                            "camera": {
                                "intrinsics": (item.get("init_camera") or {}).get("intrinsics"),
                                "resolution": [512, 512],
                                "scene_resource_version": item.get("scene_resource_version") or item.get("renderer_version"),
                            },
                            "label_validity": ("valid" if geometry.get("status") == "legal_indoor" else "invalid_illegal_camera"),
                            "observability_validity": ("valid" if _public_observation(item, state_id, pose, state_meta).get("image_path") else "invalid_missing_render"),
                            "render_status": ("provided" if _public_observation(item, state_id, pose, state_meta).get("image_path") else "not_requested"),
                            "geometry_legality": geometry,
                            "scene_pose_legality": geometry,
                            "supervision_contract_status": ('UNVERIFIED_angular_pixel_conflict' if task == 'screen_occupancy' else 'shared_runtime_contract'),
                            "qa_sft_allowed": False if task == 'screen_occupancy' else geometry.get('status') == 'legal_indoor',
                            "branch_validation": {"state_id": state_id, "validated_before_scoring": True,
                                                   "failure_reasons": geometry.get("reasons", [])},
                            "parent_source_line": line_no,
                            "_audit": {"score": result.get("score"), "predicate": result},
                        }
                        dst.write(json.dumps(record, ensure_ascii=True, default=str) + "\n"); dst.flush()
                        done_ids.add(sid); written += 1; by_task[task] = by_task.get(task, 0) + 1
                        if answer in label_counts:
                            label_counts[answer] += 1
                    except Exception as exc:
                        errors += 1
                        with (out / "errors.jsonl").open("a", encoding="utf-8") as ef:
                            ef.write(json.dumps({"sample_id": sid, "line": line_no, "error": str(exc), "traceback": traceback.format_exc()}) + "\n")
            except Exception as exc:
                errors += 1
                with (out / "errors.jsonl").open("a", encoding="utf-8") as ef:
                    ef.write(json.dumps({"sample_id": None, "line": line_no, "error": str(exc), "traceback": traceback.format_exc()}) + "\n")
    summary = {"schema_version": QA_SCHEMA_VERSION, "input": str(args.input), "samples_written": written, "errors": errors, "by_task": by_task, "labels": label_counts, "split": args.split, "split_conflicts": split_conflicts}
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / "task_contract.json").write_text(json.dumps(build_task_contract(), indent=2) + "\n")
    # Human-checkable contact sheet. Images are linked only when they were
    # actually provided by the upstream state record.
    rows = []
    for line in manifest.read_text().splitlines():
        try:
            rec = json.loads(line); obs = rec.get("public_observation", {}); img = obs.get("image_path")
            media = f'<img src="{img}" width="180">' if img else '<em>no rendered image</em>'
            rows.append(f'<tr><td>{rec.get("task_type")}</td><td>{rec.get("private_answer")}</td><td>{rec.get("state_id")}</td><td>{rec.get("source")}/{rec.get("provenance")}</td><td>{media}</td></tr>')
        except Exception:
            continue
    (out / "contact_sheet.html").write_text("<html><body><table border=1><tr><th>task</th><th>label</th><th>state</th><th>source</th><th>observation</th></tr>" + "".join(rows) + "</table></body></html>\n")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--split", default="train")
    ap.add_argument("--split-scenes", default="")
    ap.add_argument("--task-type", default="")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--scene-root", default="")
    print(json.dumps(generate(ap.parse_args()), indent=2))


if __name__ == "__main__": main()
