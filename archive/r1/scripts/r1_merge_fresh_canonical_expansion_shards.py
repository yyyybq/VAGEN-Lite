#!/usr/bin/env python3
"""Close accounting for fresh, non-v46 target expansion shards.

FOV-v2 results deliberately remain ``FOV_PROVISIONAL``.  This merger is
bookkeeping only: it never upgrades a provisional FOV result to train-ready.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


VERSION = "r1_fresh_canonical_expansion_merge_v1_20260928"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    tmp.replace(path)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def shard_records(root: Path, relative: str, *, payload_key: str | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not root.is_dir():
        return rows
    for scene in sorted(path for path in root.iterdir() if path.is_dir()):
        path = scene / relative
        if not path.is_file():
            continue
        if path.suffix == ".json":
            payload = read_json(path)
            rows.extend(payload.get(payload_key or "results", []))
        else:
            rows.extend(read_jsonl(path))
    return rows


def indexed(rows: list[dict[str, Any]], label: str) -> dict[int, dict[str, Any]]:
    result: dict[int, dict[str, Any]] = {}
    for row in rows:
        if row.get("source_row_index") is None:
            continue
        index = int(row["source_row_index"])
        if index in result:
            raise RuntimeError(f"duplicate {label} source_row_index={index}")
        result[index] = row
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope", type=Path, required=True)
    parser.add_argument("--projective-shards", type=Path, required=True)
    parser.add_argument("--fov-shards", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    sources = read_jsonl(args.scope / "fresh_sources.jsonl")
    if len({row["task_id"] for row in sources}) != len(sources):
        raise RuntimeError("scope has duplicate fresh task IDs")

    p_selector = indexed(shard_records(args.projective_shards, "selector_results.json"), "projective selector")
    p_reach = indexed(shard_records(args.projective_shards, "independent_validation/reachability_manifest.jsonl"), "projective reachability")
    p_runtime = indexed(shard_records(args.projective_shards, "independent_validation/runtime_replay.json"), "projective runtime")
    p_rgb = indexed(shard_records(args.projective_shards, "official_observability_v1/observability_manifest.jsonl"), "projective RGB")
    p_candidates = {
        str(row["task_id"]): row
        for row in shard_records(args.projective_shards, "independent_validation/candidate_rows.jsonl")
    }
    f_account = indexed(shard_records(args.fov_shards, "train/accounting.jsonl"), "FOV accounting")
    f_runtime = indexed(shard_records(args.fov_shards, "train/runtime_replay.json"), "FOV runtime")
    f_rgb = indexed(shard_records(args.fov_shards, "official_observability_v2/observability_manifest.jsonl"), "FOV RGB")
    f_candidates = {
        str(row["task_id"]): row
        for row in shard_records(args.fov_shards, "train/trainable.jsonl")
    }

    matrix: list[dict[str, Any]] = []
    projective_accepted: list[dict[str, Any]] = []
    fov_provisional: list[dict[str, Any]] = []
    for index, source in enumerate(sources):
        base = {"source_row_index": index, "source_task_id": source["task_id"], "scene_id": source["scene_id"],
                "task_type": source["task_type"], "fresh_task_lineage": source.get("fresh_task_lineage")}
        if source["task_type"] == "projective_relations":
            selector, reach, runtime, rgb = p_selector.get(index), p_reach.get(index), p_runtime.get(index), p_rgb.get(index)
            candidate = p_candidates.get(str(reach.get("task_id"))) if reach else None
            if candidate and runtime and runtime.get("status") == "pass" and rgb and rgb.get("passed") is True:
                status = "PROJECTIVE_TRAIN_READY"
                projective_accepted.append(candidate)
            elif selector is None:
                status = "implementation_or_shard_missing"
            else:
                status = str(selector.get("status") or "projective_not_accepted")
            matrix.append({**base, "final_class": status, "selector_status": selector.get("status") if selector else None,
                           "candidate_task_id": candidate.get("task_id") if candidate else None,
                           "runtime_status": runtime.get("status") if runtime else None,
                           "rgb_status": rgb.get("passed") if rgb else None})
        elif source["task_type"] == "fov_inclusion":
            accounting, runtime, rgb = f_account.get(index), f_runtime.get(index), f_rgb.get(index)
            candidate = f_candidates.get(str(accounting.get("new_task_id"))) if accounting else None
            if candidate and runtime and runtime.get("status") == "pass" and rgb and rgb.get("passed") is True:
                status = "FOV_PROVISIONAL_RUNTIME_RGB_V2"
                fov_provisional.append(candidate)
            elif accounting is None:
                status = "implementation_or_shard_missing"
            else:
                status = str(accounting.get("status") or "fov_not_accepted")
            matrix.append({**base, "final_class": status, "regeneration_status": accounting.get("status") if accounting else None,
                           "candidate_task_id": candidate.get("task_id") if candidate else None,
                           "runtime_status": runtime.get("status") if runtime else None,
                           "rgb_v2_status": rgb.get("passed") if rgb else None,
                           "training_eligible": False})
        else:
            raise RuntimeError(f"unexpected task type {source['task_type']}")

    if len(matrix) != len(sources) or {row["source_row_index"] for row in matrix} != set(range(len(sources))):
        raise RuntimeError("fresh expansion accounting is not closed")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.output_dir / "fresh_expansion_matrix.jsonl", matrix)
    write_jsonl(args.output_dir / "projective_train_ready_runtime_rgb.jsonl", projective_accepted)
    write_jsonl(args.output_dir / "fov_provisional_runtime_rgb_v2.jsonl", fov_provisional)
    summary = {
        "version": VERSION, "source_rows": len(sources),
        "accounting": {"closed": True, "expected": len(sources), "actual": len(matrix)},
        "classes_by_task": {task: dict(sorted(Counter(row["final_class"] for row in matrix if row["task_type"] == task).items()))
                            for task in ("projective_relations", "fov_inclusion")},
        "accepted": {"projective_train_ready_runtime_rgb": len(projective_accepted),
                     "fov_provisional_runtime_rgb_v2": len(fov_provisional), "fov_training_eligible": False},
        "scope": {"path": str(args.scope), "fresh_sources_sha256": digest(args.scope / "fresh_sources.jsonl")},
    }
    write_json(args.output_dir / "summary.json", summary)
    checksum_lines = [f"{digest(path)}  {path.name}" for path in sorted(args.output_dir.iterdir()) if path.is_file() and path.name != "SHA256SUMS"]
    (args.output_dir / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
