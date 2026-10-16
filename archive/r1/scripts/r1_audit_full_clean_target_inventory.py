#!/usr/bin/env python3
"""Deduplicate canonical target inventory while keeping FOV quarantined.

This is a reporting/freeze tool: it never upgrades a FOV row to train-ready,
and it excludes *both* source and scene overlap with frozen evaluation sets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


VERSION = "r1_full_clean_target_inventory_v1_20260928"
TARGETS = ("projective_relations", "fov_inclusion")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    temporary.replace(path)


def pair(row: dict[str, Any]) -> str:
    objects = row.get("target_object", {}).get("objects") or []
    return "--".join(sorted(str(obj.get("label") or "unknown") for obj in objects))


def relation(row: dict[str, Any]) -> str:
    return str(row.get("target_region", {}).get("params", {}).get("relation") or "unknown")


def runtime_steps(root: Path, task: str) -> dict[int, int | None]:
    """Merge source-indexed independent replay results from scene shards."""
    name = "projective_medium_v2/shards" if task == "projective_relations" else "fov_min4/shards"
    relative = "independent_validation/runtime_replay.json" if task == "projective_relations" else "train/runtime_replay.json"
    result: dict[int, int | None] = {}
    for path in sorted((root / name).glob("*/" + relative)):
        payload = read_json(path)
        for row in payload.get("results", []):
            if row.get("source_row_index") is not None:
                result[int(row["source_row_index"])] = row.get("steps")
    return result


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counter = lambda key: dict(sorted(Counter(str(row.get(key) or "unknown") for row in rows).items()))
    steps = Counter(row.get("first_success_step") for row in rows)
    pairs = Counter(row.get("category_pair") or "unknown" for row in rows)
    return {
        "sources": len({row["source_key"] for row in rows}), "episodes": len(rows),
        "scenes": counter("scene_id"), "relations": counter("relation"),
        "category_pairs": dict(sorted(pairs.items(), key=lambda item: (-item[1], item[0]))),
        "first_success_steps": dict(sorted(((str(key), value) for key, value in steps.items()), key=lambda item: item[0])),
        "max_category_pair_fraction": (max(pairs.values()) / len(rows)) if rows else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-inventory", type=Path, required=True)
    parser.add_argument("--base-manifest", type=Path, required=True)
    parser.add_argument("--salvage-root", type=Path, required=True)
    parser.add_argument("--old-v46-train", type=Path, required=True)
    parser.add_argument("--canonical-eval", type=Path, required=True)
    parser.add_argument("--local-action-parents", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    base_inventory, base_manifest = read_jsonl(args.base_inventory), read_jsonl(args.base_manifest)
    if len(base_inventory) != len(base_manifest):
        raise RuntimeError("base inventory/manifest length mismatch")
    old = read_jsonl(args.old_v46_train)
    dev, local = read_jsonl(args.canonical_eval), read_jsonl(args.local_action_parents)
    eval_sources = {str(row.get("source_key")) for row in dev + local}
    eval_scenes = {str(row.get("scene_id")) for row in dev + local}
    p_steps = runtime_steps(args.salvage_root, "projective_relations")
    f_steps = runtime_steps(args.salvage_root, "fov_inclusion")

    candidates: list[dict[str, Any]] = []
    for meta, raw in zip(base_inventory, base_manifest):
        task = str(raw.get("task_type"))
        if task not in TARGETS:
            continue
        source = str(meta["source_key"])
        candidates.append({
            "source_key": source, "source_row_index": int(meta["source_row_index"]), "scene_id": str(raw["scene_id"]),
            "task_type": task, "task_id": raw.get("task_id"), "relation": relation(raw), "category_pair": pair(raw),
            "first_success_step": (meta.get("difficulty") or {}).get("first_success_step"),
            "evidence_class": str(meta.get("evidence_class")), "origin": "preexisting_verified_inventory",
            "fov_quarantined": task == "fov_inclusion", "rank": 2,
        })

    matrix = read_jsonl(args.salvage_root / "salvage_matrix" / "old_v46_target_salvage_matrix.jsonl")
    for row in matrix:
        status, task = row.get("final_status"), row.get("task_type")
        if task not in TARGETS or status not in {"repairable_accepted_runtime_rgb", "repairable_provisional_rgb_v2"}:
            continue
        index = int(row["source_row_index"])
        candidates.append({
            "source_key": str(row["source_key"]), "source_row_index": index, "scene_id": str(row["scene_id"]),
            "task_type": str(task), "task_id": row.get("candidate_task_id"), "relation": str(row.get("relation") or "unknown"),
            "category_pair": "--".join(row.get("category_pair") or ["unknown"]),
            "first_success_step": (p_steps if task == "projective_relations" else f_steps).get(index),
            "evidence_class": "salvage_runtime_official_rgb_verified" if task == "projective_relations" else "salvage_fov_v2_provisional_runtime_rgb",
            "origin": "old_v46_target_salvage", "fov_quarantined": task == "fov_inclusion", "rank": 0,
        })

    # A source is the accounting unit; prefer a salvage episode when it has a
    # matching current replay, then stable task-id tie-break.  Never add rows.
    by_source: dict[str, list[dict[str, Any]]] = {}
    for row in candidates:
        by_source.setdefault(row["source_key"], []).append(row)
    selected, excluded = [], []
    for source, options in sorted(by_source.items()):
        options.sort(key=lambda row: (row["rank"], str(row.get("task_id") or "")))
        row = options[0]
        reasons = []
        if source in eval_sources:
            reasons.append("frozen_eval_source_overlap")
        if row["scene_id"] in eval_scenes:
            reasons.append("frozen_eval_scene_overlap")
        if reasons:
            excluded.append({**row, "exclusion_reasons": reasons})
        else:
            selected.append(row)

    projective = [row for row in selected if row["task_type"] == "projective_relations"]
    fov = [row for row in selected if row["task_type"] == "fov_inclusion"]
    if any(row["fov_quarantined"] for row in projective) or any(not row["fov_quarantined"] for row in fov):
        raise RuntimeError("FOV quarantine classification failure")
    old_counts = Counter(row.get("task_type") for row in old)
    summary = {
        "version": VERSION,
        "policy": {"source_dedup": "one evidence-ranked episode per original source", "fov": "all rows provisional until human-calibrated deterministic gate", "local_action": "permanent evaluation-only exclusion"},
        "candidate_records_before_source_dedup": len(candidates), "source_records_after_dedup": len(selected) + len(excluded),
        "excluded": {"rows": len(excluded), "reasons": dict(sorted(Counter(reason for row in excluded for reason in row["exclusion_reasons"]).items()))},
        "projective_train_ready": summarize(projective), "fov_provisional_not_train_ready": summarize(fov),
        "old_v46_targets": {"projective_relations": old_counts["projective_relations"], "fov_inclusion": old_counts["fov_inclusion"]},
        "deficit": {"projective_relations": max(0, old_counts["projective_relations"] - len(projective)), "fov_inclusion_train_ready": old_counts["fov_inclusion"], "fov_inclusion_provisional_gap": max(0, old_counts["fov_inclusion"] - len(fov))},
        "inputs": {str(path): sha256(path) for path in (args.base_inventory, args.base_manifest, args.old_v46_train, args.canonical_eval, args.local_action_parents)},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "projective_train_ready_deduplicated.jsonl", projective)
    write_jsonl(args.output_dir / "fov_provisional_deduplicated.jsonl", fov)
    write_jsonl(args.output_dir / "evaluation_excluded_deduplicated.jsonl", excluded)
    write_json(args.output_dir / "inventory_summary.json", summary)
    paths = sorted(path for path in args.output_dir.iterdir() if path.is_file() and path.name != "SHA256SUMS")
    (args.output_dir / "SHA256SUMS").write_text("".join(f"{sha256(path)}  {path.name}\n" for path in paths))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
