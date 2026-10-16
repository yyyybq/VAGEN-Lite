#!/usr/bin/env python3
"""R1 projective relation and v46 scorer audit.

This script is intentionally read-only with respect to training data and model
checkpoints. It writes small audit artifacts under an output directory.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vagen.envs.active_spatial.spatial_potential_field import SpatialPotentialField
from vagen.envs.active_spatial.visual_bbox_metrics import (
    _camera_pose_from_forward,
    compute_visual_bbox_metrics,
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def scaled_k(k: Any, sx: float, sy: float) -> list[list[float]]:
    out = np.asarray(k, dtype=np.float64).copy()
    if out.shape == (4, 4):
        out = out[:3, :3]
    out[0, 0] *= sx
    out[0, 2] *= sx
    out[1, 1] *= sy
    out[1, 2] *= sy
    return out.tolist()


def params_for(item: dict[str, Any], k: Any, width: int, height: int, pose: np.ndarray) -> dict[str, Any]:
    params = dict(item.get("task_params") or {})
    params.update(
        {
            "_target_object": item.get("target_object"),
            "_camera_intrinsics": k,
            "_image_width": width,
            "_image_height": height,
            "_fov_horizontal": 90.0,
            "_fov_vertical": 90.0,
            "_camera_pose_c2w": pose.tolist(),
        }
    )
    return params


def projective_visual_stats(items: list[dict[str, Any]]) -> dict[str, Any]:
    modes = {
        "native_640x480": lambda item: (item["init_camera"]["intrinsics"], 640, 480),
        "eff_256x256_scaled_K": lambda item: (
            scaled_k(item["init_camera"]["intrinsics"], 256.0 / 640.0, 256.0 / 480.0),
            256,
            256,
        ),
        "env_256x256_unscaled_K": lambda item: (item["init_camera"]["intrinsics"], 256, 256),
    }
    out: dict[str, Any] = {}
    for mode, setup in modes.items():
        counter: Counter[str] = Counter()
        margins: list[float] = []
        scores: list[float] = []
        examples: list[dict[str, Any]] = []
        for idx, item in enumerate(items):
            if item.get("task_type") != "projective_relations":
                continue
            pos = np.asarray(item["sample_target"], dtype=np.float64)
            fwd = np.asarray(item["camera_params"]["forward"], dtype=np.float64)
            pose = _camera_pose_from_forward(pos, fwd)
            k, width, height = setup(item)
            vm = compute_visual_bbox_metrics(
                pos,
                fwd,
                "projective_relations",
                params_for(item, k, width, height, pose),
                item["target_region"],
            )
            counter["n"] += 1
            counter["available"] += int(bool(vm.get("available")))
            counter["visible"] += int(float(vm.get("visibility_score", 0.0) or 0.0) > 0.0)
            counter["relation_true"] += int(bool(vm.get("visual_relation_satisfied")))
            counter["success_065"] += int(float(vm.get("visual_score", 0.0) or 0.0) >= 0.65)
            if vm.get("visual_relation_margin_px") is not None:
                margins.append(float(vm["visual_relation_margin_px"]))
            if vm.get("visual_score") is not None:
                scores.append(float(vm["visual_score"]))
            if len(examples) < 5:
                examples.append(
                    {
                        "idx": idx,
                        "relation": item["target_region"]["params"].get("relation"),
                        "margin_px": vm.get("visual_relation_margin_px"),
                        "relation_true": vm.get("visual_relation_satisfied"),
                        "visual_score": vm.get("visual_score"),
                    }
                )
        out[mode] = {
            "counts": dict(counter),
            "margin_min": min(margins) if margins else None,
            "margin_max": max(margins) if margins else None,
            "margin_mean": statistics.mean(margins) if margins else None,
            "score_max": max(scores) if scores else None,
            "score_mean": statistics.mean(scores) if scores else None,
            "examples": examples,
        }
    return out


def dataset_normal_stats(items: list[dict[str, Any]]) -> dict[str, Any]:
    counts: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    for idx, item in enumerate(items):
        if item.get("task_type") != "projective_relations":
            continue
        params = item["target_region"]["params"]
        a = np.asarray(params["object_a_center"], dtype=np.float64)
        b = np.asarray(params["object_b_center"], dtype=np.float64)
        ab = b[:2] - a[:2]
        ab = ab / max(np.linalg.norm(ab), 1e-12)
        normal = np.asarray(params["normal"], dtype=np.float64)
        normal = normal / max(np.linalg.norm(normal), 1e-12)
        cw = np.array([ab[1], -ab[0]], dtype=np.float64)
        ccw = np.array([-ab[1], ab[0]], dtype=np.float64)
        relation = params.get("relation")
        expected = cw if relation == "left" else ccw
        legacy_expected = ccw if relation == "left" else cw
        current_match = bool(np.allclose(normal, expected, atol=1e-6))
        legacy_match = bool(np.allclose(normal, legacy_expected, atol=1e-6))
        counts["n"] += 1
        counts["current_generator_normal_match"] += int(current_match)
        counts["opposite_legacy_normal_match"] += int(legacy_match)
        if len(examples) < 5:
            examples.append(
                {
                    "idx": idx,
                    "relation": relation,
                    "normal": normal.tolist(),
                    "current_expected": expected.tolist(),
                    "opposite_expected": legacy_expected.tolist(),
                    "current_match": current_match,
                    "opposite_match": legacy_match,
                }
            )
    return {"counts": dict(counts), "examples": examples}


def deterministic_cases() -> list[dict[str, Any]]:
    k = [[100.0, 0.0, 128.0], [0.0, 100.0, 128.0], [0.0, 0.0, 1.0]]
    width = height = 256
    objects = [
        {
            "id": "A",
            "label": "A",
            "bbox_min": [-0.1, -0.1, -0.1],
            "bbox_max": [0.1, 0.1, 0.1],
            "center": [0.0, 0.0, 0.0],
        },
        {
            "id": "B",
            "label": "B",
            "bbox_min": [0.9, -0.1, -0.1],
            "bbox_max": [1.1, 0.1, 0.1],
            "center": [1.0, 0.0, 0.0],
        },
    ]
    cases = [
        ("A_left_B_right", [0.5, -3.0, 0.0], [0.0, 1.0, 0.0], "left", True),
        ("B_left_A_right", [0.5, 3.0, 0.0], [0.0, -1.0, 0.0], "right", True),
        ("yaw_reversed_behind", [0.5, -3.0, 0.0], [0.0, -1.0, 0.0], "left", False),
    ]
    out: list[dict[str, Any]] = []
    for name, cam, fwd, relation, expected in cases:
        cam_arr = np.asarray(cam, dtype=np.float64)
        fwd_arr = np.asarray(fwd, dtype=np.float64)
        pose = _camera_pose_from_forward(cam_arr, fwd_arr)
        target_region = {
            "params": {
                "object_a_center": [0.0, 0.0, 0.0],
                "object_b_center": [1.0, 0.0, 0.0],
                "relation": relation,
            }
        }
        task_params = {
            "_target_object": {"objects": objects},
            "_camera_intrinsics": k,
            "_image_width": width,
            "_image_height": height,
            "_camera_pose_c2w": pose.tolist(),
        }
        vm = compute_visual_bbox_metrics(cam_arr, fwd_arr, "projective_relations", task_params, target_region)
        passed = bool(vm.get("visual_relation_satisfied")) == expected
        out.append(
            {
                "name": name,
                "relation": relation,
                "expected_relation_satisfied": expected,
                "actual_relation_satisfied": bool(vm.get("visual_relation_satisfied")),
                "margin_px": vm.get("visual_relation_margin_px"),
                "objects": vm.get("objects"),
                "passed": passed,
            }
        )
        if not passed:
            raise AssertionError(f"deterministic projective case failed: {name}: {vm}")
    return out


def v46_initial_score_match(items: list[dict[str, Any]], rollout_path: Path) -> dict[str, Any]:
    fields = {
        "legacy": SpatialPotentialField(
            position_weight=0.7,
            orientation_weight=0.3,
            fov_horizontal=90.0,
            fov_vertical=90.0,
            use_visual_bbox_scoring=False,
        ),
        "visual": SpatialPotentialField(
            position_weight=0.7,
            orientation_weight=0.3,
            fov_horizontal=90.0,
            fov_vertical=90.0,
            use_visual_bbox_scoring=True,
        ),
    }
    diffs: dict[str, list[float]] = defaultdict(list)
    by_task: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    with rollout_path.open() as f:
        for line in f:
            record = json.loads(line)
            idx = int(str(record["task_id"]).split("/")[-1])
            item = items[idx]
            pose = np.asarray(item["init_camera"]["extrinsics"], dtype=np.float64)
            pos = pose[:3, 3]
            fwd = pose[:3, 2]
            observed = float(record["initial_score"])
            for name, field in fields.items():
                result = field.compute_score(
                    pos,
                    fwd,
                    item["task_type"],
                    params_for(item, item["init_camera"]["intrinsics"], 256, 256, pose),
                    item["target_region"],
                )
                diff = abs(float(result.total_score) - observed)
                diffs[name].append(diff)
                by_task[item["task_type"]][name].append(diff)
    summary: dict[str, Any] = {}
    for name, values in diffs.items():
        summary[name] = {
            "n": len(values),
            "mean_abs": statistics.mean(values),
            "median_abs": statistics.median(values),
            "max_abs": max(values),
            "exact_match_1e-6": sum(v < 1e-6 for v in values),
        }
    summary["by_task_median_abs"] = {
        task: {name: statistics.median(vals) for name, vals in task_values.items()}
        for task, task_values in by_task.items()
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("exps/vagen_active_spatial/v46_baseline_qwen25vl_7b/train_filtered.jsonl"))
    parser.add_argument("--rollout", type=Path, default=Path("exps/vagen_active_spatial/v46_baseline_qwen25vl_7b/rollout_data/250.jsonl"))
    parser.add_argument("--out-dir", type=Path, default=Path("exps/vagen_active_spatial/r1_observation_aligned_reward_audit"))
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    items = load_jsonl(args.data)
    report = {
        "deterministic_cases": deterministic_cases(),
        "dataset_normal_stats": dataset_normal_stats(items),
        "projective_visual_stats": projective_visual_stats(items),
        "v46_initial_score_match_step250": v46_initial_score_match(items, args.rollout),
    }
    out_json = args.out_dir / "r1_projective_convention_audit.json"
    out_md = args.out_dir / "r1_projective_convention_audit.md"
    out_json.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    lines = ["# R1 Projective Convention Audit", ""]
    lines.append("## Deterministic Cases")
    for case in report["deterministic_cases"]:
        lines.append(
            f"- {case['name']}: expected={case['expected_relation_satisfied']} "
            f"actual={case['actual_relation_satisfied']} margin={case['margin_px']} pass={case['passed']}"
        )
    lines.append("")
    lines.append("## Dataset Normal Sign")
    lines.append(f"```json\n{json.dumps(report['dataset_normal_stats']['counts'], indent=2)}\n```")
    lines.append("")
    lines.append("## Projective Visual Stats")
    for mode, stats in report["projective_visual_stats"].items():
        lines.append(f"### {mode}")
        lines.append(f"```json\n{json.dumps(stats['counts'], indent=2)}\n```")
        lines.append(f"margin_mean={stats['margin_mean']} margin_min={stats['margin_min']} margin_max={stats['margin_max']}")
    lines.append("")
    lines.append("## v46 Step-250 Initial Score Match")
    lines.append(f"```json\n{json.dumps(report['v46_initial_score_match_step250'], indent=2)}\n```")
    out_md.write_text("\n".join(lines))
    print(out_md)


if __name__ == "__main__":
    main()
