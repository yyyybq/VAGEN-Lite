#!/usr/bin/env python3
"""Freeze and verify the small R1 train-only PPO diagnostic cohort.

This is deliberately *not* the formal train/ID/OOD gate.  It creates a
policy-facing JSONL containing only the fields the environment needs and a
separate audit ledger containing certificate/evidence metadata.  The selection
does not inspect certificate actions, terminal poses, policy completions, or
reward outcomes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


VERSION = "active_spatial_diagnostic_real_batch_gate_v1"
R1_CAMERA = "canonical_h1_from_frozen_candidate_intrinsics_and_pose"
R1_METRIC = "canonical_spatial_task_h1_v1"
R1_COLLISION = "interiorgs_structure_label_alignment_v1"
R1_RGB_MAX_MAE = 0.5
R1_RGB_MAX_P99 = 2.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_jsonl_at(path: Path, index: int) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle):
            if line_number == index:
                return json.loads(line)
    raise RuntimeError(f"{path} has no record {index}")


def policy_row(candidate: dict[str, Any], episode: dict[str, Any]) -> dict[str, Any]:
    """Materialize only runtime/prompt fields; audit-only fields remain separate."""
    row = {key: value for key, value in candidate.items() if key != "reachability_construction"}
    # The target region's sample pose is not needed by the canonical scorer.
    row.pop("sample_target", None)
    row["target_region"] = dict(row.get("target_region") or {})
    row["target_region"].pop("sample_point", None)
    row["target_region"].pop("sample_forward", None)
    row["camera_model_version"] = R1_CAMERA
    row["canonical_task_metric_version"] = R1_METRIC
    row["collision_convention"] = {"version": R1_COLLISION, "structure_y_sign": 1.0}
    row["source_identity"] = {
        "episode_fingerprint": episode["episode_fingerprint"],
        "source_key": episode["source_key"],
        "source_row_index": episode["source_row_index"],
        "split": "train",
        "experiment_role": "diagnostic_real_batch_train_only",
    }
    return row


def choose(episodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deterministic 4-left/4-right coverage selection, independent of certificates."""
    by_source: dict[str, dict[str, Any]] = {}
    for episode in episodes:
        previous = by_source.get(episode["source_key"])
        if previous is None or episode["episode_fingerprint"] < previous["episode_fingerprint"]:
            by_source[episode["source_key"]] = episode
    buckets: dict[str, list[dict[str, Any]]] = {"left": [], "right": []}
    for episode in by_source.values():
        relation = str(episode["relation"])
        if relation in buckets:
            buckets[relation].append(episode)
    selected: list[dict[str, Any]] = []
    # Greedy lexicographic max-coverage: first avoid a used scene and unordered
    # label pair, then use a stable hash as the only tie-breaker.
    used_scenes: set[str] = set()
    used_pairs: set[tuple[str, str]] = set()
    for relation in ("left", "right"):
        ranked = sorted(
            buckets[relation],
            key=lambda item: stable({"rule": VERSION, "source": item["source_key"], "episode": item["episode_fingerprint"]}),
        )
        for _ in range(4):
            choices = [item for item in ranked if item not in selected]
            if not choices:
                raise RuntimeError(f"not enough unique {relation} train sources")
            candidate = min(
                choices,
                key=lambda item: (
                    int(item["scene_id"] in used_scenes),
                    int(tuple(sorted(item["object_pair"])) in used_pairs),
                    stable({"rule": VERSION, "source": item["source_key"], "episode": item["episode_fingerprint"]}),
                ),
            )
            selected.append(candidate)
            used_scenes.add(candidate["scene_id"])
            used_pairs.add(tuple(sorted(candidate["object_pair"])))
    return sorted(selected, key=lambda item: item["source_key"])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--validation", type=Path, required=True)
    parser.add_argument("--constraint-scope", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--model-sha256s", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    inventory = json.loads(args.inventory.read_text())
    validation = json.loads(args.validation.read_text())
    scope = json.loads(args.constraint_scope.read_text())
    if sha256(args.inventory) != validation.get("inventory_sha256"):
        raise RuntimeError("inventory SHA does not match validation ledger")
    episodes = []
    errors: list[str] = []
    for episode in inventory["episodes"]:
        if episode.get("split") != "train":
            continue
        candidate_ref = episode["evidence"]["candidate"]
        candidate_path = Path(candidate_ref["path"])
        try:
            candidate = read_jsonl_at(candidate_path, int(candidate_ref["record_index"]))
            relation = str((candidate.get("target_region") or {}).get("params", {}).get("relation", ""))
            if relation not in {"left", "right"}:
                raise RuntimeError(f"unsupported relation {relation!r}")
            episode = dict(episode)
            episode["relation"] = relation
            episode["candidate"] = candidate
        except Exception as exc:
            errors.append(f"{episode.get('source_key')}: candidate: {exc}")
            continue
        runtime = episode.get("validation", {}).get("runtime", {})
        rgb = episode.get("validation", {}).get("official_rgb", {})
        if not (runtime.get("status") == "pass" and runtime.get("initial_success") is False and rgb.get("passed") is True):
            errors.append(f"{episode['source_key']}: inventory runtime/RGB/initial-success check failed")
            continue
        for kind, evidence in episode["evidence"].items():
            if not isinstance(evidence, dict) or "path" not in evidence:
                continue
            evidence_path = Path(evidence["path"])
            if not evidence_path.is_file() or sha256(evidence_path) != evidence["sha256"]:
                errors.append(f"{episode['source_key']}: {kind} evidence hash mismatch")
        for image in rgb.get("frame_images") or []:
            image_path = args.repo_root / image
            if not image_path.is_file():
                errors.append(f"{episode['source_key']}: missing official RGB {image_path}")
        episodes.append(episode)
    selected = choose(episodes) if not errors else []
    if len(selected) != 8 or len({item["source_key"] for item in selected}) != 8:
        errors.append("selection is not exactly eight unique train sources")
    if any(item["split"] != "train" for item in selected):
        errors.append("non-train source selected")
    expected_components = scope["component_sha256"]
    component_paths = {
        "canonical_camera.py": args.repo_root / "vagen/envs/active_spatial/canonical_camera.py",
        "canonical_task_metrics.py": args.repo_root / "vagen/envs/active_spatial/canonical_task_metrics.py",
        "collision_detector.py": args.repo_root / "vagen/envs/active_spatial/collision_detector.py",
    }
    component_report = {name: {"actual": sha256(path), "expected": expected_components[name]} for name, path in component_paths.items()}
    if any(row["actual"] != row["expected"] for row in component_report.values()):
        errors.append("frozen canonical runtime component SHA mismatch")
    if not args.model_dir.is_dir() or not args.model_sha256s.is_file():
        errors.append("pretrained model or SHA256 manifest missing")
    rows = [policy_row(item["candidate"], item) for item in selected]
    audit = [{
        "source_key": item["source_key"], "source_row_index": item["source_row_index"],
        "episode_fingerprint": item["episode_fingerprint"], "scene_id": item["scene_id"],
        "relation": item["relation"], "object_pair": item["object_pair"], "difficulty": item["difficulty"],
        "validation": item["validation"], "evidence_audit_only": item["evidence"],
    } for item in selected]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    policy_path, audit_path = args.output_dir / "policy_input_rows.jsonl", args.output_dir / "audit_manifest.jsonl"
    policy_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    audit_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in audit), encoding="utf-8")
    report = {
        "version": VERSION,
        "status": "PASS" if not errors else "BLOCKED",
        "formal_training_gate": "BLOCKED_SEPARATE_NOT_EVALUATED_HERE",
        "diagnostic_scope": "train-only eight-source one-no-update-batch; Projective relations only",
        "selection_rule": "per-source smallest episode fingerprint; deterministic 4-left/4-right greedy scene/category coverage; no certificate action/terminal pose/policy outcome input",
        "inventory": {"path": str(args.inventory), "sha256": sha256(args.inventory), "validation": str(args.validation)},
        "policy_input": {"path": str(policy_path), "sha256": sha256(policy_path), "audit_only_fields_removed": ["reachability_construction", "sample_target", "target_region.sample_point", "target_region.sample_forward"]},
        "audit_manifest": {"path": str(audit_path), "sha256": sha256(audit_path)},
        "selected": audit,
        "selection_counts": {"sources": len({item["source_key"] for item in selected}), "scenes": dict(Counter(item["scene_id"] for item in selected)), "relations": dict(Counter(item["relation"] for item in selected)), "task_families": {"projective_relations": len(selected)}},
        "runtime_contract": {"camera": R1_CAMERA, "canonical_metric": R1_METRIC, "collision": R1_COLLISION, "rgb_thresholds": {"mae": R1_RGB_MAX_MAE, "p99_abs": R1_RGB_MAX_P99}, "primitive_cap": 12, "history": "no_concat_system_plus_current_observation_only"},
        "component_hashes": component_report,
        "model": {"path": str(args.model_dir), "sha256_manifest": str(args.model_sha256s), "manifest_sha256": sha256(args.model_sha256s) if args.model_sha256s.is_file() else None},
        "errors": errors,
    }
    report_path = args.output_dir / "diagnostic_readiness.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "report": str(report_path), "selected": len(selected), "errors": errors}))
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
