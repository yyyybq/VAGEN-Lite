#!/usr/bin/env python3
"""Generate, validate, convert, and visualize SFT data with an R1 env YAML.

The environment YAML is the protocol authority.  This prevents an SFT run from
quietly reverting to the historical 30-degree/legacy-action/visual-score setup
after the R1 reward and success checks have changed.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict

import yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
for path in (ROOT, HERE.parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from active_spatial_sft.config import SFTGenerationConfig
from active_spatial_sft.convert_to_qwen25vl_sft import convert_jsonl
from active_spatial_sft.sft_generator import SFTDataGenerator
from active_spatial_sft.visualize_score_guidance import visualize


ENV_PATTERN = re.compile(r"^\$\{oc\.env:([A-Za-z_][A-Za-z0-9_]*)\}$")


def _resolve_env(value: Any) -> Any:
    if isinstance(value, str):
        match = ENV_PATTERN.match(value)
        if match:
            name = match.group(1)
            if name not in os.environ:
                raise ValueError(f"environment YAML requires unset variable {name}")
            return os.environ[name]
    if isinstance(value, dict):
        return {key: _resolve_env(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_resolve_env(item) for item in value]
    return value


def load_env_config(path: Path, env_index: int = 0) -> Dict[str, Any]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    envs = document.get("envs") if isinstance(document, dict) else None
    if not isinstance(envs, list) or not (0 <= env_index < len(envs)):
        raise ValueError(f"{path} does not contain envs[{env_index}]")
    config = envs[env_index].get("config")
    if not isinstance(config, dict):
        raise ValueError(f"{path}: envs[{env_index}].config is missing")
    return config


def validate_r1_rows(path: Path) -> Dict[str, Any]:
    from vagen.envs.active_spatial.canonical_task_metrics import (
        CANONICAL_TASK_METRIC_VERSION,
        SUPPORTED_TASK_TYPES,
        uses_canonical_backend,
    )

    count = 0
    task_types: Dict[str, int] = {}
    scenes = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not uses_canonical_backend(row):
                raise ValueError(
                    f"row {line_number} is not {CANONICAL_TASK_METRIC_VERSION}; "
                    "use run_sft_generation.py for legacy data"
                )
            task_type = str(row.get("task_type"))
            if task_type not in SUPPORTED_TASK_TYPES:
                raise ValueError(f"unsupported canonical task type at row {line_number}: {task_type}")
            task_types[task_type] = task_types.get(task_type, 0) + 1
            scenes.add(str(row.get("scene_id")))
            count += 1
    if count == 0:
        raise ValueError(f"empty R1 JSONL: {path}")
    return {"rows": count, "scenes": len(scenes), "task_types": task_types}


def validate_audit_alignment(source: Path, audit: Path) -> Dict[str, int]:
    source_ids = {
        str(json.loads(line).get("task_id") or "")
        for line in source.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    audit_ids = []
    for line in audit.read_text(encoding="utf-8").splitlines():
        if line.strip():
            audit_ids.append(str(json.loads(line).get("policy_task_id") or ""))
    if "" in source_ids or "" in audit_ids:
        raise ValueError("source/audit alignment requires non-empty task identifiers")
    if len(audit_ids) != len(set(audit_ids)):
        raise ValueError(f"duplicate policy_task_id values in {audit}")
    missing = sorted(source_ids - set(audit_ids))
    if missing:
        raise ValueError(f"audit manifest misses {len(missing)} source task IDs; first={missing[0]}")
    return {
        "source_task_ids": len(source_ids),
        "audit_records": len(audit_ids),
        "matched_source_task_ids": len(source_ids),
    }


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the R1-aligned Active Spatial SFT data pipeline.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--env-yaml", required=True, help="Authoritative ActiveSpatial env YAML.")
    parser.add_argument("--env-index", type=int, default=0)
    parser.add_argument(
        "--jsonl-path",
        default="",
        help="Override the YAML dataset while retaining its reward/camera/action protocol.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-name", default="sft_data")
    parser.add_argument("--client-url", default="", help="Override client_url / R1_RENDER_URL.")
    parser.add_argument("--gs-root", default="", help="Override the scene asset root.")
    parser.add_argument("--render-backend", choices=["local", "client", "http"], default="")
    parser.add_argument("--gpu-device", type=int, default=None)
    parser.add_argument("--max-items", type=int, default=-1)
    parser.add_argument("--start-idx", type=int, default=0)
    parser.add_argument("--end-idx", type=int, default=-1)
    parser.add_argument("--beam-width", type=int, default=16)
    parser.add_argument("--min-improvement", type=float, default=0.001)
    parser.add_argument("--plateau-tolerance", type=int, default=5)
    parser.add_argument(
        "--guided-search",
        action="store_true",
        help=(
            "Use hidden target geometry before score fine-tuning. By default R1 uses "
            "score-only beam search so the visualization tests the reward signal itself."
        ),
    )
    parser.add_argument("--allow-failed", action="store_true")
    parser.add_argument("--partial-success-min-score", type=float, default=0.0)
    parser.add_argument("--image-format", choices=("png", "jpg"), default="jpg", help="Use PNG for pixel-exact replay evidence")
    parser.add_argument("--no-think", action="store_true")
    parser.add_argument("--also-no-think", action="store_true")
    parser.add_argument("--skip-convert", action="store_true")
    parser.add_argument(
        "--qwen-format", choices=["jsonl", "parquet"], default="jsonl",
        help="QwenVL/LLaMA-Factory export container.",
    )
    parser.add_argument("--skip-visualize", action="store_true")
    parser.add_argument("--max-dashboards", type=int, default=50)
    parser.add_argument(
        "--audit-jsonl",
        default="auto",
        help="R1 audit-only certificate manifest; 'auto' uses a sibling audit_only.jsonl.",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser.parse_args()


def write_dataset_info(output_dir: Path, primary_name: str, no_think_name: str = "") -> Path:
    def entry(file_name: str) -> Dict[str, Any]:
        return {
            "file_name": file_name,
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
                "system_tag": "system",
            },
        }

    document = {"active_spatial_r1_sft": entry(primary_name)}
    if no_think_name:
        document["active_spatial_r1_sft_no_think"] = entry(no_think_name)
    target = output_dir / "dataset_info.json"
    with target.open("x", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return target


def main() -> None:
    args = _args()
    env_yaml = Path(args.env_yaml).resolve()
    raw_env = load_env_config(env_yaml, args.env_index)
    if args.client_url:
        raw_env["client_url"] = args.client_url
    if args.gs_root:
        raw_env["gs_root"] = args.gs_root
    if args.render_backend:
        raw_env["render_backend"] = args.render_backend
    if args.gpu_device is not None:
        raw_env["gpu_device"] = args.gpu_device
    if (
        raw_env.get("render_backend") == "local"
        and ENV_PATTERN.match(str(raw_env.get("client_url", "")))
    ):
        raw_env["client_url"] = ""
    env = _resolve_env(raw_env)

    jsonl_path = Path(args.jsonl_path or env.get("jsonl_path", "")).resolve()
    if not jsonl_path.is_file():
        raise FileNotFoundError(f"R1 source JSONL does not exist: {jsonl_path}")
    row_summary = validate_r1_rows(jsonl_path)
    if args.audit_jsonl == "auto":
        candidate = jsonl_path.parent / "audit_only.jsonl"
        audit_jsonl = candidate if candidate.is_file() else None
    elif args.audit_jsonl:
        audit_jsonl = Path(args.audit_jsonl).resolve()
        if not audit_jsonl.is_file():
            raise FileNotFoundError(f"R1 audit manifest does not exist: {audit_jsonl}")
    else:
        audit_jsonl = None
    if audit_jsonl:
        row_summary["audit_alignment"] = validate_audit_alignment(jsonl_path, audit_jsonl)

    required = ("step_translation", "step_rotation_deg", "action_space", "max_episode_steps")
    missing = [key for key in required if key not in env]
    if missing:
        raise ValueError(f"R1 env YAML is missing protocol fields: {missing}")

    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"R1 SFT output directory must be new or empty: {output_dir}"
        )
    cfg = SFTGenerationConfig(
        jsonl_path=str(jsonl_path),
        gs_root=str(Path(env.get("gs_root", "")).resolve()) if env.get("gs_root") else "",
        output_dir=str(output_dir),
        output_name=args.output_name,
        render_backend=str(env.get("render_backend", "http")),
        client_url=str(env.get("client_url", "")),
        client_origin=env.get("client_origin"),
        gpu_device=env.get("gpu_device"),
        image_width=int(env.get("image_width", 256)),
        image_height=int(env.get("image_height", 256)),
        step_translation=float(env["step_translation"]),
        step_rotation_deg=float(env["step_rotation_deg"]),
        action_space=str(env["action_space"]),
        enable_explicit_done=bool(env.get("enable_explicit_done", False)),
        max_episode_steps=int(env["max_episode_steps"]),
        success_threshold=float(env.get("success_score_threshold", 0.65)),
        max_total_actions=int(env["max_episode_steps"]),
        max_actions_per_turn=int(env.get("max_actions_per_step", 5)),
        min_improvement=args.min_improvement,
        plateau_tolerance=args.plateau_tolerance,
        beam_width=args.beam_width,
        use_guided_search=args.guided_search,
        position_weight=float(env.get("potential_field_position_weight", 0.7)),
        orientation_weight=float(env.get("potential_field_orientation_weight", 0.3)),
        max_distance=float(env.get("max_distance", 5.0)),
        enable_collision_detection=bool(env.get("enable_collision_detection", True)),
        collision_camera_radius=float(env.get("collision_camera_radius", 0.15)),
        collision_floor_height=float(env.get("collision_floor_height", 0.3)),
        collision_ceiling_height=float(env.get("collision_ceiling_height", 2.5)),
        collision_safety_margin=float(env.get("collision_safety_margin", 0.05)),
        max_items=args.max_items,
        start_idx=args.start_idx,
        end_idx=args.end_idx,
        min_trajectory_steps=1,
        max_trajectory_steps=int(env.get("turn_budget", env["max_episode_steps"])),
        only_successful=not args.allow_failed,
        partial_success_min_score=args.partial_success_min_score,
        prompt_format=(
            "no_think" if args.no_think else str(env.get("prompt_format", "free_think"))
        ),
        add_think=not args.no_think,
        save_primitive_images=True,
        image_format=args.image_format,
        include_score_in_think=False,
        verbose=args.verbose,
        runtime_env_overrides=env,
        generation_profile="r1_gate_aligned_from_env_yaml_v1",
        env_config_source=str(env_yaml),
        audit_jsonl_path=str(audit_jsonl) if audit_jsonl else "",
    )

    print("[R1 SFT] source rows:", json.dumps(row_summary, sort_keys=True))
    print(
        "[R1 SFT] protocol:",
        json.dumps(
            {
                "action_space": cfg.action_space,
                "rotation_deg": cfg.step_rotation_deg,
                "primitive_budget": cfg.max_episode_steps,
                "score_search": "geometry_guided" if cfg.use_guided_search else "score_only_beam",
                "render_backend": cfg.render_backend,
            },
            sort_keys=True,
        ),
    )
    stats = SFTDataGenerator(cfg).run()
    stats_path = output_dir / f"{args.output_name}_stats.json"
    with stats_path.open("x", encoding="utf-8") as handle:
        json.dump(stats, handle, indent=2, sort_keys=True)
        handle.write("\n")

    sft_jsonl = output_dir / f"{args.output_name}.jsonl"
    outputs: Dict[str, Any] = {"sft_jsonl": str(sft_jsonl), "stats": stats}
    if not args.skip_convert:
        suffix = ".parquet" if args.qwen_format == "parquet" else ".jsonl"
        qwen = output_dir / f"qwen_vl_sft{suffix}"
        total, converted = convert_jsonl(
            sft_jsonl,
            qwen,
            output_dir,
            strip_think=args.no_think,
            strict=True,
            to_parquet=args.qwen_format == "parquet",
        )
        outputs["qwen_vl"] = {"path": str(qwen), "records": converted, "source_records": total}
        no_think_name = ""
        if args.also_no_think and not args.no_think:
            qwen_no_think = output_dir / f"qwen_vl_sft_no_think{suffix}"
            _, converted_no_think = convert_jsonl(
                sft_jsonl,
                qwen_no_think,
                output_dir,
                strip_think=True,
                strict=True,
                to_parquet=args.qwen_format == "parquet",
            )
            no_think_name = qwen_no_think.name
            outputs["qwen_vl_no_think"] = {
                "path": str(qwen_no_think),
                "records": converted_no_think,
            }
        dataset_info = write_dataset_info(output_dir, qwen.name, no_think_name)
        outputs["llamafactory_dataset_info"] = str(dataset_info)
    if not args.skip_visualize:
        visual_dir = output_dir / "visualization"
        outputs["visualization"] = visualize(sft_jsonl, visual_dir, args.max_dashboards)

    with (output_dir / "pipeline_summary.json").open("x", encoding="utf-8") as handle:
        json.dump(outputs, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(json.dumps(outputs, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
