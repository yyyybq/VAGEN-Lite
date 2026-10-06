#!/usr/bin/env python3
"""Read-only data/config gate for Active Spatial reward-only ablations.

The script never launches training and never invents a split.  Every manifest is
named explicitly on the command line, hashed, loaded in full, and compared by
stable identity plus task-content fingerprints.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, fields
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Dict, Iterable, Mapping

import numpy as np
import yaml

from vagen.envs.active_spatial.canonical_camera import CANONICAL_CAMERA_H1_RESIZE_V1
from vagen.envs.active_spatial.canonical_task_metrics import (
    CANONICAL_TASK_METRIC_VERSION,
    SUPPORTED_TASK_TYPES,
    score_canonical_task,
    uses_canonical_backend,
)
from vagen.envs.active_spatial.env_config import ActiveSpatialEnvConfig
from vagen.envs.active_spatial.spatial_potential_field import create_potential_field


REWARD_ONLY_KEYS = frozenset(
    {
        "enable_potential_shaping_reward",
        "enable_near_success_reward",
        "enable_visibility_shaping_reward",
        "near_success_bonus",
        "format_reward",
        "potential_field_gamma",
    }
)
CONTENT_FIELDS = (
    "scene_id",
    "task_type",
    "object_label",
    "preset",
    "task_description",
    "task_params",
    "target_object",
    "target_region",
    "sample_target",
    "target_position",
    "target_orientation",
    "camera_params",
    "init_camera",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected JSON object")
            rows.append(value)
    return rows


def semantic_fingerprint(row: Mapping[str, Any]) -> str:
    return canonical_hash({key: row.get(key) for key in CONTENT_FIELDS if key in row})


def stable_identity(row: Mapping[str, Any]) -> tuple[str, str]:
    if row.get("task_id") not in (None, ""):
        return f"task_id:{row['task_id']}", "task_id"
    if row.get("source_manifest_sha256") and row.get("source_index") is not None:
        return (
            f"lineage:{row['source_manifest_sha256']}:{row['source_index']}",
            "source_manifest_sha256+source_index",
        )
    return f"semantic:{semantic_fingerprint(row)}", "semantic_fingerprint_fallback"


def split_role(name: str) -> str:
    if name == "train":
        return "train"
    if name in {"id", "val_id", "id_test"}:
        return "id"
    if name == "validation_proxy" or name == "ood" or name.startswith("ood_"):
        return "ood"
    return "unknown"


def gate_status(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"status": "BLOCKED", "reason": "missing", "path": str(path)}
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix.lower() == ".json":
        try:
            payload = json.loads(text)
            candidates = [payload.get(key) for key in ("status", "gate", "result")]
            explicit = next((str(v).upper() for v in candidates if v is not None), None)
            if explicit:
                return {"status": "PASS" if explicit == "PASS" else "BLOCKED", "declared": explicit, "path": str(path)}
        except json.JSONDecodeError:
            pass
    matches = list(re.finditer(r"\b(PASS|BLOCKED)\b", text, flags=re.IGNORECASE))
    declared = matches[-1].group(1).upper() if matches else "UNKNOWN"
    return {
        "status": "PASS" if declared == "PASS" else "BLOCKED",
        "declared": declared,
        "path": str(path),
        "sha256": sha256_file(path),
    }


def initial_score_stats(rows: Iterable[dict[str, Any]], env_cfg: Mapping[str, Any]) -> dict[str, Any]:
    field = create_potential_field(
        {
            "position_weight": env_cfg["potential_field_position_weight"],
            "orientation_weight": env_cfg["potential_field_orientation_weight"],
            "max_distance": env_cfg["max_distance"],
            "fov_horizontal": env_cfg["fov_horizontal"],
            "fov_vertical": env_cfg["fov_vertical"],
            "use_visual_bbox_scoring": env_cfg["use_visual_bbox_scoring"],
        }
    )
    by_family: Counter[str] = Counter()
    evaluated = success = failed = 0
    errors: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        family = str(row.get("task_type", "unknown"))
        try:
            pose = np.asarray(row["init_camera"]["extrinsics"], dtype=np.float64)
            if uses_canonical_backend(row):
                metric = score_canonical_task(row, pose)
                hit = bool(metric["success"])
            else:
                params = dict(row.get("task_params") or {})
                params.update(
                    {
                        "_target_object": row.get("target_object"),
                        "_camera_intrinsics": row["init_camera"].get("intrinsics"),
                        "_image_width": int(env_cfg["image_width"]),
                        "_image_height": int(env_cfg["image_height"]),
                        "_fov_horizontal": float(env_cfg["fov_horizontal"]),
                        "_fov_vertical": float(env_cfg["fov_vertical"]),
                        "_camera_pose_c2w": pose.tolist(),
                    }
                )
                result = field.compute_score(
                    camera_position=pose[:3, 3],
                    camera_forward=pose[:3, 2],
                    task_type=family,
                    task_params=params,
                    target_region=row.get("target_region") or {},
                )
                if env_cfg.get("success_require_both", False):
                    hit = (
                        float(result.position_score) >= float(env_cfg["success_position_threshold"])
                        and float(result.orientation_score) >= float(env_cfg["success_orientation_threshold"])
                    )
                else:
                    hit = float(result.total_score) >= float(env_cfg["success_score_threshold"])
            evaluated += 1
            if hit:
                success += 1
                by_family[family] += 1
        except Exception as exc:  # report; never silently drop rows
            failed += 1
            if len(errors) < 20:
                errors.append({"index": index, "task_type": family, "error": f"{type(exc).__name__}: {exc}"})
    return {
        "status": "PASS" if failed == 0 else "BLOCKED",
        "evaluated": evaluated,
        "failed_to_evaluate": failed,
        "initial_success": success,
        "initial_success_rate": success / evaluated if evaluated else None,
        "initial_success_by_task_family": dict(sorted(by_family.items())),
        "errors_first_20": errors,
    }


def manifest_report(path: Path, env_cfg: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, set[str]]]:
    rows = load_jsonl(path)
    identities = []
    identity_methods: Counter[str] = Counter()
    semantic = []
    content = []
    for row in rows:
        identity, method = stable_identity(row)
        identities.append(identity)
        identity_methods[method] += 1
        semantic.append(semantic_fingerprint(row))
        content.append(canonical_hash(row))

    visual_rows = [row for row in rows if row.get("task_type") in SUPPORTED_TASK_TYPES]
    canonical_metric_ok = sum(
        row.get("canonical_task_metric_version") == CANONICAL_TASK_METRIC_VERSION for row in visual_rows
    )
    camera_ok = sum(row.get("camera_model_version") == CANONICAL_CAMERA_H1_RESIZE_V1 for row in visual_rows)
    collision_ok = sum(
        isinstance(row.get("collision_convention"), dict)
        and row["collision_convention"].get("status") == "frozen"
        for row in visual_rows
    )
    duplicates = {
        "stable_identity": len(identities) - len(set(identities)),
        "semantic_fingerprint": len(semantic) - len(set(semantic)),
        "full_content_fingerprint": len(content) - len(set(content)),
    }
    report = {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "rows": len(rows),
        "task_family_distribution": dict(sorted(Counter(str(r.get("task_type", "unknown")) for r in rows).items())),
        "scene_count": len({str(r.get("scene_id", "unknown")) for r in rows}),
        "stable_identity_methods": dict(sorted(identity_methods.items())),
        "duplicates_within_split": duplicates,
        "canonical_gate": {
            "visual_rows": len(visual_rows),
            "canonical_metric_version_ok": canonical_metric_ok,
            "camera_model_version_ok": camera_ok,
            "frozen_collision_convention_ok": collision_ok,
            "status": (
                "PASS"
                if canonical_metric_ok == camera_ok == collision_ok == len(visual_rows)
                else "BLOCKED"
            ),
        },
        "initial_success": initial_score_stats(rows, env_cfg),
    }
    sets = {
        "identity": set(identities),
        "semantic": set(semantic),
        "content": set(content),
        "scene": {str(r.get("scene_id", "unknown")) for r in rows},
    }
    return report, sets


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            out.update(flatten(child, path))
        return out
    if isinstance(value, list):
        return {prefix: value}
    return {prefix: value}


def diff_values(left: Mapping[str, Any], right: Mapping[str, Any]) -> list[dict[str, Any]]:
    flat_left, flat_right = flatten(left), flatten(right)
    changes = []
    for key in sorted(set(flat_left) | set(flat_right)):
        if flat_left.get(key) != flat_right.get(key):
            changes.append({"path": key, "left": flat_left.get(key), "right": flat_right.get(key)})
    return changes


def parse_manifest(spec: str, root: Path) -> tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(f"manifest must be NAME=PATH, got {spec!r}")
    name, raw_path = spec.split("=", 1)
    path = Path(raw_path)
    if not path.is_absolute():
        path = root / path
    return name.strip(), path.resolve()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--manifest", action="append", default=[], help="explicit NAME=PATH; repeat")
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[1]
    matrix_path = args.matrix if args.matrix.is_absolute() else repo / args.matrix
    matrix = yaml.safe_load(matrix_path.read_text(encoding="utf-8"))
    manifests = dict(parse_manifest(spec, repo) for spec in args.manifest)
    roles_present = {split_role(name) for name in manifests}
    missing_roles = sorted(set(matrix["required_split_roles"]) - roles_present)

    baseline_train_path = repo / matrix["baseline_train_resolved_config"]
    baseline_env_path = repo / matrix["baseline_env_config"]
    training_cfg = yaml.safe_load(baseline_train_path.read_text(encoding="utf-8"))
    env_wrapper = yaml.safe_load(baseline_env_path.read_text(encoding="utf-8"))
    if isinstance(env_wrapper.get("envs"), list):
        first_env = env_wrapper["envs"][0]
        raw_env_cfg = dict(first_env.get("config") or {})
    else:
        first_env = env_wrapper[next(iter(env_wrapper))]
        raw_env_cfg = dict(first_env.get("env_config") or {})
    misplaced_split_keys = {key: raw_env_cfg.pop(key) for key in ("train_size", "test_size") if key in raw_env_cfg}
    known_fields = {entry.name for entry in fields(ActiveSpatialEnvConfig)}
    unknown_env_keys = sorted(set(raw_env_cfg) - known_fields)
    typed_env_cfg = asdict(ActiveSpatialEnvConfig(**{k: v for k, v in raw_env_cfg.items() if k in known_fields}))
    if "train" in manifests:
        typed_env_cfg["jsonl_path"] = str(manifests["train"])

    output_dir = args.output_dir if args.output_dir.is_absolute() else repo / args.output_dir
    resolved_dir = output_dir / "resolved_configs"
    resolved_dir.mkdir(parents=True, exist_ok=True)

    variants: dict[str, dict[str, Any]] = {}
    for name, spec in matrix["variants"].items():
        env_cfg = dict(typed_env_cfg)
        env_cfg.update(spec["reward_overrides"])
        resolved = {
            "schema_version": matrix["schema_version"],
            "variant": name,
            "label": spec["label"],
            "interpretation": spec["interpretation"],
            "training": training_cfg,
            "environment": env_cfg,
            "explicit_manifests": {key: str(value) for key, value in manifests.items()},
        }
        variants[name] = resolved
        (resolved_dir / f"{name}.resolved.json").write_text(
            json.dumps(resolved, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    pairwise = {}
    for left, right in (("S0", "S1"), ("S0", "S5"), ("S1", "S5")):
        changes = diff_values(variants[left], variants[right])
        pairwise[f"{left}_vs_{right}"] = changes
    controlled_s1_s0 = [
        change for change in pairwise["S0_vs_S1"]
        if change["path"] not in {"variant", "label", "interpretation"}
    ]
    config_gate = {
        "status": "PASS"
        if {c["path"] for c in controlled_s1_s0} == {"environment.enable_potential_shaping_reward"}
        and all(
            c["path"].split(".")[-1] in REWARD_ONLY_KEYS
            for changes in pairwise.values()
            for c in changes
            if c["path"] not in {"variant", "label", "interpretation"}
        )
        else "BLOCKED",
        "s1_minus_s0_only_potential_switch": controlled_s1_s0,
        "pairwise_full_resolved_diff": pairwise,
        "resolved_paths": {name: str((resolved_dir / f"{name}.resolved.json").resolve()) for name in variants},
        "legacy_reference_verified": {
            "potential_field_gamma": raw_env_cfg.get("potential_field_gamma"),
            "near_success_bonus": raw_env_cfg.get("near_success_bonus"),
            "visibility_reward_scale": typed_env_cfg.get("visibility_reward_scale"),
            "success_reward": raw_env_cfg.get("success_reward"),
        },
    }

    manifest_reports: dict[str, Any] = {}
    manifest_sets: dict[str, dict[str, set[str]]] = {}
    missing_files = []
    for name, path in manifests.items():
        if not path.is_file():
            missing_files.append(str(path))
            continue
        report, sets = manifest_report(path, variants["S0"]["environment"])
        manifest_reports[name] = report
        manifest_sets[name] = sets

    overlaps = []
    names = sorted(manifest_sets)
    forbidden_pairs = {frozenset(pair) for pair in matrix["scene_overlap_policy"].get("forbidden_pairs", [])}
    forbidden_fingerprint_roles = {
        frozenset(pair)
        for pair in matrix.get("fingerprint_overlap_policy", {}).get("forbidden_role_pairs", [])
    }
    overlap_gate = "PASS"
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            intersections = {
                key: sorted(manifest_sets[left][key] & manifest_sets[right][key])
                for key in ("identity", "semantic", "content", "scene")
            }
            scene_forbidden = (
                frozenset((left, right)) in forbidden_pairs
            )
            fingerprint_forbidden = (
                frozenset((split_role(left), split_role(right))) in forbidden_fingerprint_roles
            )
            violations = []
            for key in ("identity", "semantic", "content"):
                if fingerprint_forbidden and intersections[key]:
                    violations.append(f"{key}_overlap")
            if scene_forbidden and intersections["scene"]:
                violations.append("scene_overlap_forbidden_by_split_definition")
            if violations:
                overlap_gate = "BLOCKED"
            overlaps.append(
                {
                    "pair": [left, right],
                    "scene_policy": "forbid" if scene_forbidden else "allow",
                    "fingerprint_policy": "forbid" if fingerprint_forbidden else "allow",
                    "counts": {key: len(value) for key, value in intersections.items()},
                    "examples_first_20": {key: value[:20] for key, value in intersections.items() if value},
                    "violations": violations,
                }
            )

    external_gates = {
        name: gate_status(repo / path) for name, path in matrix.get("required_gates", {}).items()
    }
    data_gate = "PASS"
    if missing_roles or missing_files or overlap_gate != "PASS":
        data_gate = "BLOCKED"
    if any(report["canonical_gate"]["status"] != "PASS" for report in manifest_reports.values()):
        data_gate = "BLOCKED"
    if any(report["initial_success"]["status"] != "PASS" for report in manifest_reports.values()):
        data_gate = "BLOCKED"
    if any(report["initial_success"]["initial_success"] != 0 for report in manifest_reports.values()):
        data_gate = "BLOCKED"
    if any(gate["status"] != "PASS" for gate in external_gates.values()):
        data_gate = "BLOCKED"

    overall = "PASS" if data_gate == config_gate["status"] == "PASS" else "BLOCKED"
    report = {
        "schema_version": "active_spatial_dense_score_preflight_v1",
        "status": overall,
        "training_launched": False,
        "matrix": {"path": str(matrix_path.resolve()), "sha256": sha256_file(matrix_path)},
        "source_config_audit": {
            "baseline_train_resolved_config": str(baseline_train_path.resolve()),
            "baseline_train_resolved_sha256": sha256_file(baseline_train_path),
            "baseline_env_config": str(baseline_env_path.resolve()),
            "baseline_env_sha256": sha256_file(baseline_env_path),
            "misplaced_implicit_split_keys_removed": misplaced_split_keys,
            "unknown_env_keys": unknown_env_keys,
        },
        "required_split_roles_missing": missing_roles,
        "missing_manifest_files": missing_files,
        "manifests": manifest_reports,
        "overlaps": overlaps,
        "overlap_gate": overlap_gate,
        "external_gates": external_gates,
        "config_gate": config_gate,
        "data_gate": data_gate,
        "blockers": [
            item
            for item, active in (
                ("explicit train/ID/OOD role missing", bool(missing_roles)),
                ("manifest file missing", bool(missing_files)),
                ("identity/content/forbidden-scene overlap", overlap_gate != "PASS"),
                ("canonical/camera/collision data gate incomplete", any(r["canonical_gate"]["status"] != "PASS" for r in manifest_reports.values())),
                ("initial-success rows present or unevaluable", any(r["initial_success"]["status"] != "PASS" or r["initial_success"]["initial_success"] != 0 for r in manifest_reports.values())),
                ("required external canonical/data gate blocked", any(g["status"] != "PASS" for g in external_gates.values())),
                ("resolved-config reward-only invariant failed", config_gate["status"] != "PASS"),
            )
            if active
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "preflight_report.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": overall, "report": str(report_path.resolve()), "blockers": report["blockers"]}, ensure_ascii=False))
    return 0 if overall == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
