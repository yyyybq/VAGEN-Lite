#!/usr/bin/env python3
"""
Oracle/discrete-planner audit for Active Spatial JSONL data.

For each item, this script reports:
  1. score at the initial pose,
  2. score at the generator's sample_point/sample_forward oracle pose,
  3. best score found by a privileged discrete planner within N actions.

The planner reuses ViewManipulator and SpatialPotentialField so pose updates and
scoring match the training environment as closely as possible without rendering.
It is collisionless by default; use it as a geometry/action-space upper-bound
audit, not as a rendered-visibility proof.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vagen.envs.active_spatial.spatial_potential_field import ScoreResult, create_potential_field
from vagen.envs.active_spatial.region_metrics import compute_region_metrics, region_heuristic
from vagen.envs.active_spatial.utils import ACTION_SPACE_PRESETS, ViewManipulator


@dataclass
class AuditConfig:
    action_space: str = "strafe"
    step_translation: float = 0.3
    step_rotation_deg: float = 20.0
    max_steps: int = 20
    success_score_threshold: float = 0.85
    success_require_both: bool = False
    success_position_threshold: float = 0.7
    success_orientation_threshold: float = 0.7
    require_all_targets_in_fov: bool = False
    position_weight: float = 0.7
    orientation_weight: float = 0.3
    max_distance: float = 5.0
    image_width: int = 512
    image_height: int = 512
    fov_horizontal: float = 90.0
    fov_vertical: float = 90.0
    use_visual_bbox_scoring: bool = True
    beam_size: int = 4096
    max_states: int = 200000
    position_quant: float = 1e-4
    rotation_round_digits: int = 5
    heuristic_weight: float = 0.10
    repair_derived_fields: bool = True


@dataclass
class PlannerState:
    pose: np.ndarray
    actions: Tuple[str, ...]
    score: ScoreResult
    priority: float


def native(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, dict):
        return {str(k): native(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [native(v) for v in value]
    return value


def load_yaml_or_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    try:
        import yaml  # type: ignore
    except ImportError as exc:
        raise RuntimeError("PyYAML is required for YAML env config files") from exc
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def find_env_config(data: Any) -> Dict[str, Any]:
    if not isinstance(data, dict):
        return {}
    if isinstance(data.get("env_config"), dict):
        return dict(data["env_config"])
    defaults = data.get("defaults")
    if isinstance(defaults, dict) and isinstance(defaults.get("env"), dict):
        out = dict(defaults["env"])
        if "max_steps_per_episode" in defaults:
            out.setdefault("max_steps_per_episode", defaults["max_steps_per_episode"])
        return out
    for value in data.values():
        found = find_env_config(value)
        if found:
            return found
    return {}


def bool_arg(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "on"}


def apply_env_overrides(cfg: AuditConfig, env: Dict[str, Any], use_env_max_steps: bool) -> None:
    mapping = {
        "action_space": "action_space",
        "step_translation": "step_translation",
        "step_rotation_deg": "step_rotation_deg",
        "success_score_threshold": "success_score_threshold",
        "success_require_both": "success_require_both",
        "success_position_threshold": "success_position_threshold",
        "success_orientation_threshold": "success_orientation_threshold",
        "potential_field_position_weight": "position_weight",
        "potential_field_orientation_weight": "orientation_weight",
        "max_distance": "max_distance",
        "image_width": "image_width",
        "image_height": "image_height",
        "fov_horizontal": "fov_horizontal",
        "fov_vertical": "fov_vertical",
        "use_visual_bbox_scoring": "use_visual_bbox_scoring",
    }
    for src, dst in mapping.items():
        if src not in env:
            continue
        if dst in {"success_require_both", "use_visual_bbox_scoring"}:
            setattr(cfg, dst, bool_arg(env[src]))
        else:
            setattr(cfg, dst, type(getattr(cfg, dst))(env[src]))

    if use_env_max_steps:
        if "max_episode_steps" in env:
            cfg.max_steps = int(env["max_episode_steps"])
        elif "max_steps_per_episode" in env:
            cfg.max_steps = int(env["max_steps_per_episode"])


def scoring_task_params(item: Dict[str, Any], cfg: AuditConfig, pose: Optional[np.ndarray] = None) -> Dict[str, Any]:
    params = dict(item.get("task_params", {}) or {})
    params["_target_object"] = item.get("target_object")
    init_cam = item.get("init_camera", {}) if isinstance(item.get("init_camera"), dict) else {}
    K = init_cam.get("intrinsics")
    if K is not None:
        params["_camera_intrinsics"] = K
    params["_image_width"] = int(cfg.image_width)
    params["_image_height"] = int(cfg.image_height)
    params["_fov_horizontal"] = float(cfg.fov_horizontal)
    params["_fov_vertical"] = float(cfg.fov_vertical)
    if pose is not None:
        params["_camera_pose_c2w"] = np.asarray(pose, dtype=np.float64).tolist()
    return params


def score_pose(field: Any, item: Dict[str, Any], pose: np.ndarray, cfg: AuditConfig) -> ScoreResult:
    pos = pose[:3, 3]
    forward = pose[:3, 2]
    return field.compute_score(
        camera_position=pos,
        camera_forward=forward,
        task_type=item.get("task_type", "absolute_positioning"),
        task_params=scoring_task_params(item, cfg, pose),
        target_region=item.get("target_region", {}),
    )


def repair_item_for_audit(item: Dict[str, Any], cfg: AuditConfig) -> Tuple[Dict[str, Any], List[str]]:
    if not cfg.repair_derived_fields:
        return item, []

    task_type = item.get("task_type")
    if task_type != "centering":
        return item, []

    region = item.get("target_region", {})
    params = region.get("params", {}) if isinstance(region, dict) else {}
    missing_b = "object_b_center" not in params
    missing_c = "object_c_center" not in params
    if not (missing_b or missing_c):
        return item, []

    target_object = item.get("target_object", {})
    objects = target_object.get("objects", []) if isinstance(target_object, dict) else []
    if not (isinstance(objects, list) and len(objects) >= 3):
        return item, []

    repaired = copy.deepcopy(item)
    repaired_region = repaired.get("target_region", {})
    repaired_params = repaired_region.get("params", {}) if isinstance(repaired_region, dict) else {}
    repairs: List[str] = []
    if missing_b and isinstance(objects[1], dict) and "center" in objects[1]:
        repaired_params["object_b_center"] = objects[1]["center"]
        repairs.append("centering.object_b_center")
    if missing_c and isinstance(objects[2], dict) and "center" in objects[2]:
        repaired_params["object_c_center"] = objects[2]["center"]
        repairs.append("centering.object_c_center")
    repaired_region["params"] = repaired_params
    repaired["target_region"] = repaired_region
    return repaired, repairs


def score_sample_pose(field: Any, item: Dict[str, Any], cfg: AuditConfig) -> Optional[ScoreResult]:
    region = item.get("target_region", {})
    if not isinstance(region, dict):
        return None
    sample_point = region.get("sample_point") or item.get("sample_target")
    sample_forward = region.get("sample_forward")
    if sample_point is None or sample_forward is None:
        return None
    return field.compute_score(
        camera_position=np.array(sample_point, dtype=np.float64),
        camera_forward=np.array(sample_forward, dtype=np.float64),
        task_type=item.get("task_type", "absolute_positioning"),
        task_params=scoring_task_params(item, cfg),
        target_region=region,
    )


def score_summary(score: Optional[ScoreResult]) -> Dict[str, Any]:
    if score is None:
        return {
            "total": None,
            "position": None,
            "orientation": None,
            "all_targets_in_fov": None,
        }
    details = getattr(score, "details", {}) or {}
    return {
        "total": float(score.total_score),
        "position": float(score.position_score),
        "orientation": float(score.orientation_score),
        "all_targets_in_fov": details.get("all_targets_in_fov"),
        "fov_score": details.get("fov_score"),
        "region_score": details.get("region_score"),
        "distance_to_region": details.get("distance_to_region"),
        "sample_target_distance": details.get("sample_target_distance"),
        "sample_target_is_auxiliary": details.get("sample_target_is_auxiliary"),
        "visual_bbox_overrode_score": details.get("visual_bbox_overrode_score"),
        "visual_bbox_available": (details.get("visual_bbox_metrics") or {}).get("available"),
        "visual_score": details.get("visual_score"),
    }


def success_hit(score: ScoreResult, cfg: AuditConfig) -> bool:
    if cfg.success_require_both:
        ok = (
            float(score.position_score) >= cfg.success_position_threshold
            and float(score.orientation_score) >= cfg.success_orientation_threshold
        )
    else:
        ok = float(score.total_score) >= cfg.success_score_threshold

    if ok and cfg.require_all_targets_in_fov:
        ok = bool((getattr(score, "details", {}) or {}).get("all_targets_in_fov", False))
    return ok


def pose_key(pose: np.ndarray, cfg: AuditConfig) -> Tuple[Any, ...]:
    pos = np.round(pose[:3, 3] / cfg.position_quant).astype(np.int64)
    rot = tuple(np.round(pose[:3, :3].reshape(-1), cfg.rotation_round_digits))
    return tuple(pos.tolist()) + rot


def region_distance_heuristic(item: Dict[str, Any], pose: np.ndarray, cfg: AuditConfig) -> float:
    region = item.get("target_region", {})
    if not isinstance(region, dict):
        return 0.0
    metrics = compute_region_metrics(
        camera_position=pose[:3, 3],
        camera_forward=pose[:3, 2],
        task_type=item.get("task_type", "absolute_positioning"),
        task_params=scoring_task_params(item, cfg, pose),
        target_region=region,
    )
    return region_heuristic(metrics)


def state_priority(item: Dict[str, Any], score: ScoreResult, pose: np.ndarray, cfg: AuditConfig) -> float:
    return float(score.total_score) + cfg.heuristic_weight * region_distance_heuristic(item, pose, cfg)


def planner_actions(cfg: AuditConfig) -> List[str]:
    preset = ACTION_SPACE_PRESETS.get(cfg.action_space)
    if preset is None:
        raise ValueError(f"Unknown action_space={cfg.action_space!r}. Known: {sorted(ACTION_SPACE_PRESETS)}")
    ordered = [
        "move_forward",
        "move_backward",
        "move_left",
        "move_right",
        "turn_left",
        "turn_right",
        "look_up",
        "look_down",
    ]
    return [a for a in ordered if a in preset and a != "done"]


def run_planner(field: Any, item: Dict[str, Any], cfg: AuditConfig) -> Dict[str, Any]:
    init_cam = item.get("init_camera", {})
    init_pose = np.array(init_cam.get("extrinsics", np.eye(4)), dtype=np.float64)
    if init_pose.shape != (4, 4):
        return {
            "planner_status": "invalid_pose",
            "planner_answer": "invalid",
            "planner_solved": False,
            "search_truncated": False,
            "search_mode": "beam" if cfg.beam_size > 0 else "exhaustive_quantized",
            "error": f"init extrinsics shape is {init_pose.shape}",
        }

    actions = planner_actions(cfg)
    manipulator = ViewManipulator(
        step_translation=cfg.step_translation,
        step_rotation_deg=cfg.step_rotation_deg,
    )

    init_score = score_pose(field, item, init_pose, cfg)
    best_state = PlannerState(
        pose=init_pose,
        actions=tuple(),
        score=init_score,
        priority=state_priority(item, init_score, init_pose, cfg),
    )
    if success_hit(init_score, cfg):
        return planner_result(best_state, init_score, "solved", False, 0, 1, 1, cfg)

    frontier: List[PlannerState] = [best_state]
    seen: Dict[Tuple[Any, ...], float] = {pose_key(init_pose, cfg): float(init_score.total_score)}
    expanded = 0
    generated = 1
    truncated = False
    state_limit_hit = False

    for depth in range(1, cfg.max_steps + 1):
        candidates: List[PlannerState] = []
        for state in frontier:
            expanded += 1
            for action in actions:
                manipulator.reset(state.pose)
                next_pose = manipulator.step(action)
                next_score = score_pose(field, item, next_pose, cfg)
                key = pose_key(next_pose, cfg)
                old = seen.get(key)
                if old is not None and old >= float(next_score.total_score) - 1e-9:
                    continue
                seen[key] = float(next_score.total_score)
                generated += 1

                next_state = PlannerState(
                    pose=next_pose,
                    actions=state.actions + (action,),
                    score=next_score,
                    priority=state_priority(item, next_score, next_pose, cfg),
                )
                if next_score.total_score > best_state.score.total_score:
                    best_state = next_state
                if success_hit(next_score, cfg):
                    return planner_result(
                        next_state,
                        init_score,
                        "solved",
                        truncated,
                        expanded,
                        generated,
                        len(seen),
                        cfg,
                    )
                candidates.append(next_state)

                if len(seen) >= cfg.max_states:
                    truncated = True
                    state_limit_hit = True
                    break
            if state_limit_hit:
                break
        if state_limit_hit or not candidates:
            break
        candidates.sort(key=lambda s: s.priority, reverse=True)
        if cfg.beam_size > 0 and len(candidates) > cfg.beam_size:
            # Beam pruning means the search is no longer exhaustive, but it
            # should keep exploring the retained frontier.  It is not a reason
            # to stop at this depth.
            truncated = True
            frontier = candidates[: cfg.beam_size]
        else:
            frontier = candidates

    status = "truncated" if truncated else "not_found"
    return planner_result(best_state, init_score, status, truncated, expanded, generated, len(seen), cfg)


def planner_result(
    best_state: PlannerState,
    init_score: ScoreResult,
    status: str,
    truncated: bool,
    expanded: int,
    generated: int,
    unique_states: int,
    cfg: AuditConfig,
) -> Dict[str, Any]:
    if status == "solved":
        answer = "solvable"
    elif truncated:
        answer = "not_proven_search_truncated"
    elif status == "invalid_pose":
        answer = "invalid"
    else:
        answer = "not_solvable_within_budget"

    return {
        "planner_status": status,
        "planner_answer": answer,
        "planner_solved": status == "solved",
        "search_truncated": bool(truncated),
        "search_mode": "beam" if cfg.beam_size > 0 else "exhaustive_quantized",
        "initial_score": score_summary(init_score),
        "best_score": score_summary(best_state.score),
        "best_step": len(best_state.actions),
        "best_actions": list(best_state.actions),
        "expanded_states": int(expanded),
        "generated_states": int(generated),
        "unique_states": int(unique_states),
        "max_steps": cfg.max_steps,
        "action_space": cfg.action_space,
        "step_translation": cfg.step_translation,
        "step_rotation_deg": cfg.step_rotation_deg,
    }


def iter_jsonl(path: Path) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            yield line_no, json.loads(line)


def expand_input_paths(paths: Sequence[str]) -> List[Path]:
    out: List[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            out.extend(sorted(path.glob("*.jsonl")))
        else:
            out.append(path)
    return out


def item_id(item: Dict[str, Any], line_no: int) -> str:
    scene = item.get("scene_id", "unknown_scene")
    task = item.get("task_type", "unknown_task")
    preset = item.get("preset", "unknown_preset")
    return f"{scene}:{line_no}:{task}:{preset}"


def audit_item(
    field: Any,
    item: Dict[str, Any],
    source: Path,
    line_no: int,
    cfg: AuditConfig,
) -> Dict[str, Any]:
    repaired_item, repairs = repair_item_for_audit(item, cfg)
    oracle_score = score_sample_pose(field, repaired_item, cfg)
    planner = run_planner(field, repaired_item, cfg)
    return {
        "id": item_id(item, line_no),
        "source": str(source),
        "line_no": line_no,
        "scene_id": item.get("scene_id"),
        "task_type": item.get("task_type"),
        "preset": item.get("preset"),
        "object_label": item.get("object_label"),
        "repairs": repairs,
        "oracle_sample_score": score_summary(oracle_score),
        **planner,
    }


def update_summary(summary: Dict[str, Any], row: Dict[str, Any]) -> None:
    task = row.get("task_type") or "unknown"
    summary["overall"]["count"] += 1
    summary["overall"]["solved"] += int(bool(row.get("planner_solved")))
    summary["overall"]["truncated"] += int(bool(row.get("search_truncated")))
    summary["overall"]["status_counts"][row.get("planner_status", "unknown")] += 1

    task_summary = summary["by_task"][task]
    task_summary["count"] += 1
    task_summary["solved"] += int(bool(row.get("planner_solved")))
    task_summary["truncated"] += int(bool(row.get("search_truncated")))
    task_summary["status_counts"][row.get("planner_status", "unknown")] += 1

    for key, src in (
        ("initial_total_sum", row.get("initial_score", {}).get("total")),
        ("oracle_total_sum", row.get("oracle_sample_score", {}).get("total")),
        ("best_total_sum", row.get("best_score", {}).get("total")),
    ):
        if src is not None:
            task_summary[key] += float(src)
            summary["overall"][key] += float(src)


def finalize_summary(summary: Dict[str, Any], cfg: AuditConfig) -> Dict[str, Any]:
    def finalize_block(block: Dict[str, Any]) -> Dict[str, Any]:
        count = max(1, int(block.get("count", 0)))
        out = dict(block)
        out["solve_rate"] = float(block.get("solved", 0)) / count
        out["truncation_rate"] = float(block.get("truncated", 0)) / count
        out["avg_initial_total"] = float(block.get("initial_total_sum", 0.0)) / count
        out["avg_oracle_total"] = float(block.get("oracle_total_sum", 0.0)) / count
        out["avg_best_total"] = float(block.get("best_total_sum", 0.0)) / count
        out["status_counts"] = dict(out.get("status_counts", {}))
        return out

    return {
        "config": native(cfg.__dict__),
        "overall": finalize_block(summary["overall"]),
        "by_task": {task: finalize_block(block) for task, block in sorted(summary["by_task"].items())},
    }


def print_summary(summary: Dict[str, Any]) -> None:
    overall = summary["overall"]
    print(
        f"Overall: {overall['solved']}/{overall['count']} solved "
        f"({overall['solve_rate'] * 100:.1f}%), "
        f"avg_best={overall['avg_best_total']:.3f}, "
        f"truncated={overall['truncated']}"
    )
    print("")
    print("By task:")
    print("task_type,count,solved,solve_rate,avg_initial,avg_oracle,avg_best,truncated")
    for task, block in summary["by_task"].items():
        print(
            f"{task},{block['count']},{block['solved']},"
            f"{block['solve_rate']:.3f},"
            f"{block['avg_initial_total']:.3f},"
            f"{block['avg_oracle_total']:.3f},"
            f"{block['avg_best_total']:.3f},"
            f"{block['truncated']}"
        )


def write_summary_csv(path: Path, summary: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "task_type",
                "count",
                "solved",
                "solve_rate",
                "avg_initial_total",
                "avg_oracle_total",
                "avg_best_total",
                "truncated",
            ],
        )
        writer.writeheader()
        for task, block in summary["by_task"].items():
            writer.writerow({
                "task_type": task,
                "count": block["count"],
                "solved": block["solved"],
                "solve_rate": f"{block['solve_rate']:.6f}",
                "avg_initial_total": f"{block['avg_initial_total']:.6f}",
                "avg_oracle_total": f"{block['avg_oracle_total']:.6f}",
                "avg_best_total": f"{block['avg_best_total']:.6f}",
                "truncated": block["truncated"],
            })


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit Active Spatial data with an oracle discrete planner.")
    parser.add_argument("--input", nargs="+", required=True, help="Input JSONL file(s), or directories containing JSONL files.")
    parser.add_argument("--output", help="Per-item JSONL output path.")
    parser.add_argument("--summary-output", help="Summary JSON output path.")
    parser.add_argument("--summary-csv", help="Summary CSV output path.")
    parser.add_argument("--env-config", help="Optional train/eval env YAML/JSON to load action/reward settings from.")
    parser.add_argument("--use-env-max-steps", action="store_true", help="Use max_episode_steps from --env-config if present.")
    parser.add_argument("--max-items", type=int, default=None, help="Optional cap across all inputs.")
    parser.add_argument("--include-task-types", nargs="*", default=None)
    parser.add_argument("--exclude-task-types", nargs="*", default=None)

    parser.add_argument("--action-space", choices=sorted(ACTION_SPACE_PRESETS), default=None)
    parser.add_argument("--step-translation", type=float, default=None)
    parser.add_argument("--step-rotation-deg", type=float, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--success-score-threshold", type=float, default=None)
    parser.add_argument("--success-require-both", action="store_true")
    parser.add_argument("--success-position-threshold", type=float, default=None)
    parser.add_argument("--success-orientation-threshold", type=float, default=None)
    parser.add_argument("--require-all-targets-in-fov", action="store_true")

    parser.add_argument("--beam-size", type=int, default=None, help="0 disables beam pruning until max-states.")
    parser.add_argument("--exact", action="store_true", help="Alias for --beam-size 0; exhaustive up to --max-states.")
    parser.add_argument("--max-states", type=int, default=None)
    parser.add_argument("--position-quant", type=float, default=None)
    parser.add_argument("--rotation-round-digits", type=int, default=None)
    parser.add_argument("--heuristic-weight", type=float, default=None)
    parser.add_argument("--no-repair-derived-fields", action="store_true", help="Disable in-memory fixes for legacy JSONL fields such as centering B/C centers.")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    cfg = AuditConfig()

    if args.env_config:
        env = find_env_config(load_yaml_or_json(Path(args.env_config)))
        apply_env_overrides(cfg, env, use_env_max_steps=args.use_env_max_steps)

    cli_overrides = {
        "action_space": args.action_space,
        "step_translation": args.step_translation,
        "step_rotation_deg": args.step_rotation_deg,
        "max_steps": args.max_steps,
        "success_score_threshold": args.success_score_threshold,
        "success_position_threshold": args.success_position_threshold,
        "success_orientation_threshold": args.success_orientation_threshold,
        "beam_size": args.beam_size,
        "max_states": args.max_states,
        "position_quant": args.position_quant,
        "rotation_round_digits": args.rotation_round_digits,
        "heuristic_weight": args.heuristic_weight,
    }
    for key, value in cli_overrides.items():
        if value is not None:
            setattr(cfg, key, value)
    if args.exact:
        cfg.beam_size = 0
    if args.success_require_both:
        cfg.success_require_both = True
    if args.require_all_targets_in_fov:
        cfg.require_all_targets_in_fov = True
    if args.no_repair_derived_fields:
        cfg.repair_derived_fields = False

    include = set(args.include_task_types or [])
    exclude = set(args.exclude_task_types or [])

    field = create_potential_field({
        "position_weight": cfg.position_weight,
        "orientation_weight": cfg.orientation_weight,
        "max_distance": cfg.max_distance,
        "fov_horizontal": cfg.fov_horizontal,
        "fov_vertical": cfg.fov_vertical,
        "use_visual_bbox_scoring": cfg.use_visual_bbox_scoring,
    })

    out_f = None
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_f = out_path.open("w", encoding="utf-8")

    summary: Dict[str, Any] = {
        "overall": defaultdict(float, {"status_counts": Counter(), "count": 0, "solved": 0, "truncated": 0}),
        "by_task": defaultdict(lambda: defaultdict(float, {"status_counts": Counter(), "count": 0, "solved": 0, "truncated": 0})),
    }

    processed = 0
    try:
        for path in expand_input_paths(args.input):
            for line_no, item in iter_jsonl(path):
                task_type = str(item.get("task_type", "unknown"))
                if include and task_type not in include:
                    continue
                if exclude and task_type in exclude:
                    continue

                row = audit_item(field, item, path, line_no, cfg)
                update_summary(summary, row)
                processed += 1

                if out_f is not None:
                    out_f.write(json.dumps(native(row), ensure_ascii=False) + "\n")

                if args.max_items is not None and processed >= args.max_items:
                    break
            if args.max_items is not None and processed >= args.max_items:
                break
    finally:
        if out_f is not None:
            out_f.close()

    final_summary = finalize_summary(summary, cfg)
    print_summary(final_summary)

    if args.summary_output:
        path = Path(args.summary_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(native(final_summary), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.summary_csv:
        write_summary_csv(Path(args.summary_csv), final_summary)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
