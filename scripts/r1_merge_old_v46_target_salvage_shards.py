#!/usr/bin/env python3
"""Close row accounting for scene-staged old-v46 target salvage.

This is bookkeeping only: it never infers a PASS from a generator result.  A
historical target row is ``CLEAN_REUSABLE`` only if it already has current
canonical/runtime/RGB evidence (none do in the frozen v46 manifest).  Repaired
FOV v2 RGB passes remain provisional until the calibration has a human-labelled
negative set, so they cannot leak into a clean-training manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


VERSION = "r1_old_v46_target_salvage_merge_v1_20260919"


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


def records_for_scene(root: Path, filename: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for scene_dir in sorted(path for path in root.iterdir() if path.is_dir()):
        path = scene_dir / filename
        if not path.is_file():
            continue
        if path.suffix == ".json":
            payload = read_json(path)
            records.extend(payload.get("results", payload.get("rows", [])))
        else:
            records.extend(read_jsonl(path))
    return records


def unique_by_source(rows: list[dict[str, Any]], field: str = "source_row_index") -> dict[int, dict[str, Any]]:
    out: dict[int, dict[str, Any]] = {}
    for row in rows:
        if row.get(field) is None:
            continue
        key = int(row[field])
        if key in out:
            raise RuntimeError(f"duplicate source row {key} in merged shard evidence")
        out[key] = row
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", type=Path, required=True)
    parser.add_argument("--projective-shards", type=Path, required=True)
    parser.add_argument("--fov-shards", type=Path, required=True)
    parser.add_argument("--fov-calibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    all_rows = read_jsonl(args.scope / "all_target_source_inventory.jsonl")
    eligible = {int(row["source_row_index"]): row for row in read_jsonl(args.scope / "eligible_target_source_inventory.jsonl")}
    excluded = {int(row["source_row_index"]): row for row in read_jsonl(args.scope / "evaluation_excluded_target_source_inventory.jsonl")}
    if set(eligible) & set(excluded) or set(eligible) | set(excluded) != {int(row["source_row_index"]) for row in all_rows}:
        raise RuntimeError("scope identities do not close")

    # Projective scene output: selector, materialized candidate/reachability,
    # independent runtime, then official RGB.  Every map is source keyed.
    p_selectors = records_for_scene(args.projective_shards, "selector_results.json")
    p_candidates = records_for_scene(args.projective_shards, "independent_validation/candidate_rows.jsonl")
    p_reaches = records_for_scene(args.projective_shards, "independent_validation/reachability_manifest.jsonl")
    p_runtime = records_for_scene(args.projective_shards, "independent_validation/runtime_replay.json")
    p_rgb = records_for_scene(args.projective_shards, "official_observability_v1/observability_manifest.jsonl")
    p_selector = unique_by_source(p_selectors)
    p_reach_by_source = unique_by_source(p_reaches)
    p_candidate_by_task = {str(row["task_id"]): row for row in p_candidates}
    p_runtime_by_source = unique_by_source(p_runtime)
    p_rgb_by_source = unique_by_source(p_rgb)

    # FOV scene output uses regeneration accounting, current independent replay,
    # and separate v2 official RGB audit.
    f_accounting = records_for_scene(args.fov_shards, "train/accounting.jsonl")
    f_candidates = records_for_scene(args.fov_shards, "train/trainable.jsonl")
    f_runtime = records_for_scene(args.fov_shards, "train/runtime_replay.json")
    f_rgb = records_for_scene(args.fov_shards, "official_observability_v2/observability_manifest.jsonl")
    f_accounting_by_source = unique_by_source(f_accounting)
    f_candidate_by_task = {str(row["task_id"]): row for row in f_candidates}
    f_runtime_by_source = unique_by_source(f_runtime)
    f_rgb_by_source = unique_by_source(f_rgb)
    calibration = read_json(args.fov_calibration)
    fov_training_eligible = calibration.get("training_gate_status") == "ELIGIBLE"

    matrix: list[dict[str, Any]] = []
    p_accepted: list[dict[str, Any]] = []
    f_provisional: list[dict[str, Any]] = []
    for source in sorted(all_rows, key=lambda row: int(row["source_row_index"])):
        index = int(source["source_row_index"])
        record: dict[str, Any] = {**source, "final_salvage_class": None, "final_status": None}
        if index in excluded:
            record.update({"final_salvage_class": "INVALID_OR_DROP", "final_status": "evaluation_isolation_excluded"})
        elif source["task_type"] == "projective_relations":
            selector = p_selector.get(index)
            reach = p_reach_by_source.get(index)
            candidate = p_candidate_by_task.get(str(reach.get("task_id"))) if reach else None
            runtime, rgb = p_runtime_by_source.get(index), p_rgb_by_source.get(index)
            record.update({"selector_status": selector.get("status") if selector else "missing_selector_result",
                           "candidate_task_id": candidate.get("task_id") if candidate else None,
                           "runtime_status": runtime.get("status") if runtime else None,
                           "rgb_status": rgb.get("passed") if rgb else None})
            if selector is None:
                record.update({"final_salvage_class": "REPAIRABLE", "final_status": "implementation_or_shard_missing"})
            elif selector.get("status") == "difficulty_unverified":
                record.update({"final_salvage_class": "REPAIRABLE", "final_status": "difficulty_unverified"})
            elif candidate is None:
                record.update({"final_salvage_class": "REPAIRABLE", "final_status": str(selector.get("status"))})
            elif runtime and runtime.get("status") == "pass" and rgb and rgb.get("passed") is True:
                record.update({"final_salvage_class": "REPAIRABLE", "final_status": "repairable_accepted_runtime_rgb"})
                p_accepted.append(candidate)
            else:
                record.update({"final_salvage_class": "INVALID_OR_DROP", "final_status": "repair_failed_runtime_or_rgb"})
        else:
            accounting = f_accounting_by_source.get(index)
            candidate = f_candidate_by_task.get(str(accounting.get("new_task_id"))) if accounting else None
            runtime, rgb = f_runtime_by_source.get(index), f_rgb_by_source.get(index)
            record.update({"regeneration_status": accounting.get("status") if accounting else "missing_regeneration_result",
                           "candidate_task_id": candidate.get("task_id") if candidate else None,
                           "runtime_status": runtime.get("status") if runtime else None,
                           "rgb_v2_status": rgb.get("passed") if rgb else None})
            if accounting is None:
                record.update({"final_salvage_class": "REPAIRABLE", "final_status": "implementation_or_shard_missing"})
            elif accounting.get("status") == "unverified":
                record.update({"final_salvage_class": "REPAIRABLE", "final_status": "reachability_unverified"})
            elif candidate is None:
                record.update({"final_salvage_class": "INVALID_OR_DROP", "final_status": str(accounting.get("status"))})
            elif runtime and runtime.get("status") == "pass" and rgb and rgb.get("passed") is True:
                status = "repairable_accepted_runtime_rgb" if fov_training_eligible else "repairable_provisional_rgb_v2"
                record.update({"final_salvage_class": "REPAIRABLE", "final_status": status,
                               "training_eligible": fov_training_eligible})
                f_provisional.append(candidate)
            else:
                record.update({"final_salvage_class": "INVALID_OR_DROP", "final_status": "repair_failed_runtime_or_rgb"})
        matrix.append(record)
    if len(matrix) != len(all_rows) or len({row["source_row_index"] for row in matrix}) != len(matrix):
        raise RuntimeError("merged salvage accounting is not closed")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "old_v46_target_salvage_matrix.jsonl", matrix)
    write_jsonl(args.output_dir / "projective_repairable_runtime_rgb_accepted.jsonl", p_accepted)
    write_jsonl(args.output_dir / "fov_repairable_rgb_v2_provisional.jsonl", f_provisional)
    summary = {
        "version": VERSION, "rows": len(matrix),
        "accounting": {"closed": True, "expected": len(all_rows), "actual": len(matrix)},
        "classes": dict(sorted(Counter(row["final_salvage_class"] for row in matrix).items())),
        "status_by_task": {task: dict(sorted(Counter(row["final_status"] for row in matrix if row["task_type"] == task).items()))
                           for task in ("projective_relations", "fov_inclusion")},
        "accepted": {"projective_runtime_rgb": len(p_accepted), "fov_v2_provisional": len(f_provisional),
                     "fov_training_eligible": fov_training_eligible},
        "inputs": {"scope": str(args.scope), "projective_shards": str(args.projective_shards),
                   "fov_shards": str(args.fov_shards), "fov_calibration": {"path": str(args.fov_calibration), "sha256": sha256(args.fov_calibration)}},
    }
    write_json(args.output_dir / "salvage_summary.json", summary)
    hashes = []
    for path in sorted(args.output_dir.iterdir()):
        if path.is_file() and path.name != "SHA256SUMS":
            hashes.append(f"{sha256(path)}  {path.name}")
    (args.output_dir / "SHA256SUMS").write_text("\n".join(hashes) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
