#!/usr/bin/env python3
"""Freeze old-v46 target-row salvage without contaminating frozen evaluation.

Historical v46 target rows have neither the canonical-runtime metadata nor
action certificates required for ``CLEAN_REUSABLE`` status.  This tool freezes
their identity and separates them into an eligible same-source repair set and
an explicit evaluation-isolation exclusion set; it never changes a source row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


VERSION = "r1_old_v46_target_salvage_scope_v1_20260919"
TARGET_TASKS = {"projective_relations", "fov_inclusion"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    temporary.replace(path)


def source_identity(index: int) -> str:
    return f"train:{index}"


def category_pair(row: dict[str, Any]) -> list[str]:
    objects = row.get("target_object", {}).get("objects") or []
    return sorted(str(obj.get("label") or "unknown") for obj in objects)


def relation(row: dict[str, Any]) -> str:
    return str(row.get("target_region", {}).get("params", {}).get("relation") or "unknown")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-v46-train", type=Path, required=True)
    parser.add_argument("--canonical-development-eval", type=Path, required=True)
    parser.add_argument("--local-action-eval-parents", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    old_rows = read_jsonl(args.old_v46_train)
    canonical_eval = read_jsonl(args.canonical_development_eval)
    local_eval = read_jsonl(args.local_action_eval_parents)
    eval_sets = {
        "canonical_development_eval": canonical_eval,
        "permanent_local_action_eval": local_eval,
    }
    eval_sources = {name: {str(row.get("source_key")) for row in rows} for name, rows in eval_sets.items()}
    eval_scenes = {name: {str(row.get("scene_id")) for row in rows} for name, rows in eval_sets.items()}
    all_eval_sources = set().union(*eval_sources.values())
    all_eval_scenes = set().union(*eval_scenes.values())

    eligible: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    inventory: list[dict[str, Any]] = []
    eligible_asset_rows: list[dict[str, Any]] = []
    for index, row in enumerate(old_rows):
        task_type = str(row.get("task_type"))
        if task_type not in TARGET_TASKS:
            continue
        source = source_identity(index)
        scene = str(row.get("scene_id"))
        reasons = []
        if source in all_eval_sources:
            reasons.append("frozen_evaluation_source_overlap")
        if scene in all_eval_scenes:
            reasons.append("frozen_evaluation_scene_overlap")
        record = {
            "split": "train", "source_row_index": index, "source_key": source,
            "historical_task_id": row.get("task_id"), "scene_id": scene, "task_type": task_type,
            "relation": relation(row), "category_pair": category_pair(row),
            "historical_row_sha256": hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest(),
            "historical_canonical_evidence": "absent_unversioned_v46_target_row",
            "initial_salvage_class": "REPAIRABLE",
            "evaluation_isolation_reasons": reasons,
        }
        inventory.append(record)
        if reasons:
            excluded.append({**record, "initial_salvage_class": "INVALID_OR_DROP_EVALUATION_EXCLUDED"})
        else:
            eligible.append(record)
            # This compact manifest is solely an AOSS-ledger input.  The
            # generators keep indexing the immutable full old-v46 manifest so
            # their source_row_index remains a stable historical identity.
            eligible_asset_rows.append(row)

    projective = [record for record in eligible if record["task_type"] == "projective_relations"]
    fov = [record for record in eligible if record["task_type"] == "fov_inclusion"]
    expected = {(record["source_key"], record["task_type"]) for record in inventory}
    closed = len(expected) == len(inventory) and len(eligible) + len(excluded) == len(inventory)
    if not closed:
        raise RuntimeError("salvage scope accounting is not closed")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "all_target_source_inventory.jsonl", inventory)
    write_jsonl(args.output_dir / "eligible_target_source_inventory.jsonl", eligible)
    write_jsonl(args.output_dir / "evaluation_excluded_target_source_inventory.jsonl", excluded)
    write_jsonl(args.output_dir / "eligible_target_rows_for_asset_ledger.jsonl", eligible_asset_rows)
    write_json(args.output_dir / "projective_medium_selection.json", {
        "version": VERSION,
        "records": [{**record, "requested_bucket": "medium"} for record in projective],
        "policy": "frozen_projective_difficulty_conditioned_initial_selector_v2_shortcut_aware_same_pair_only",
    })
    write_json(args.output_dir / "fov_source_index_selection.json", {
        "train": [int(record["source_row_index"]) for record in fov],
    })
    write_json(args.output_dir / "sources_train_only.json", {"train": str(args.old_v46_train.resolve())})
    write_json(args.output_dir / "asset_ledger_sources.json", {
        "train": str((args.output_dir / "eligible_target_rows_for_asset_ledger.jsonl").resolve()),
    })
    summary = {
        "version": VERSION,
        "inputs": {
            "old_v46_train": {"path": str(args.old_v46_train.resolve()), "sha256": sha256(args.old_v46_train)},
            "canonical_development_eval": {"path": str(args.canonical_development_eval.resolve()), "sha256": sha256(args.canonical_development_eval)},
            "local_action_eval_parents": {"path": str(args.local_action_eval_parents.resolve()), "sha256": sha256(args.local_action_eval_parents)},
        },
        "isolation_policy": {
            "source_overlap": "exclude", "scene_overlap": "exclude",
            "local_action": "permanent_evaluation_only_never_training_or_tuning",
        },
        "old_target_counts": dict(sorted(Counter(str(row.get("task_type")) for row in old_rows if str(row.get("task_type")) in TARGET_TASKS).items())),
        "scope_counts": {
            "all_target_rows": len(inventory), "eligible": len(eligible), "evaluation_excluded": len(excluded),
            "eligible_projective": len(projective), "eligible_fov": len(fov),
            "eligible_scenes": len({record["scene_id"] for record in eligible}),
            "all_target_scenes": len({record["scene_id"] for record in inventory}),
        },
        "evaluation_exclusions": {
            "by_reason": dict(sorted(Counter(reason for record in excluded for reason in record["evaluation_isolation_reasons"]).items())),
            "by_task": dict(sorted(Counter(record["task_type"] for record in excluded).items())),
            "scenes": sorted({record["scene_id"] for record in excluded}),
        },
        "accounting": {"expected": len(expected), "actual": len(inventory), "closed": closed},
    }
    write_json(args.output_dir / "scope_summary.json", summary)
    checksums = []
    for path in sorted(args.output_dir.iterdir()):
        if path.is_file() and path.name != "SHA256SUMS":
            checksums.append(f"{sha256(path)}  {path.name}")
    (args.output_dir / "SHA256SUMS").write_text("\n".join(checksums) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
