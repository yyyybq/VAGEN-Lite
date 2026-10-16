#!/usr/bin/env python3
"""Freeze the independently runtime/RGB-validated Projective Medium inventory.

The inventory is evidence-preserving: every episode points back to immutable
candidate, reachability, runtime, and official-render records by file SHA256
and record index.  Sources and episodes are counted separately because one
source may have been independently validated under both frozen selectors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


VERSION = "r1_verified_projective_medium_inventory_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def source_key(split: str, index: int) -> str:
    return f"{split}:{int(index)}"


def task_source_key(task_id: str, fallback_index: int, split_hint: str | None = None) -> str:
    marker = "projective_path_first_proto_"
    if task_id.startswith(marker) and task_id.endswith("_medium"):
        core = task_id[len(marker):-len("_medium")]
        split, index = core.rsplit("_", 1)
        return source_key(split, int(index))
    if split_hint is None:
        raise ValueError(f"cannot recover split from task_id={task_id!r}")
    return source_key(split_hint, fallback_index)


def stable_fingerprint(candidate: dict[str, Any], reachability: dict[str, Any], key: str) -> str:
    objects = candidate.get("target_object", {}).get("objects", [])
    payload = {
        # Episode identity remains source-aware even when two manifests contain
        # byte-identical task geometry.  Split isolation is part of lineage.
        "source_key": key,
        "scene_id": candidate.get("scene_id"),
        "object_ids": [str(obj.get("id")) for obj in objects],
        "initial_c2w": candidate.get("init_camera", {}).get("extrinsics"),
        "terminal_c2w": (reachability.get("path") or [{}])[-1].get("c2w"),
        "actions": reachability.get("actions"),
        "first_success_step": reachability.get("first_success_step"),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def indexed(rows: list[dict[str, Any]], field: str = "task_id") -> dict[str, tuple[int, dict[str, Any]]]:
    result: dict[str, tuple[int, dict[str, Any]]] = {}
    for index, row in enumerate(rows):
        key = str(row[field])
        if key in result:
            raise ValueError(f"duplicate {field}: {key}")
        result[key] = (index, row)
    return result


def load_bundle(name: str, paths: dict[str, Path]) -> list[dict[str, Any]]:
    candidates = read_jsonl(paths["candidate"])
    reachability = read_jsonl(paths["reachability"])
    runtime_payload = json.loads(paths["runtime"].read_text())
    runtime = runtime_payload["results"]
    observability = read_jsonl(paths["observability"])
    by_candidate = indexed(candidates)
    by_reachability = indexed(reachability)
    by_runtime = indexed(runtime)
    by_observability = indexed(observability)
    evidence_files = {
        role: {"path": str(path.resolve()), "sha256": sha256(path)}
        for role, path in paths.items()
    }
    episodes = []
    for task_id, (obs_index, obs) in by_observability.items():
        if not obs.get("passed"):
            continue
        missing = [role for role, table in (
            ("candidate", by_candidate), ("reachability", by_reachability), ("runtime", by_runtime)
        ) if task_id not in table]
        if missing:
            raise ValueError(f"{name}:{task_id}: missing evidence {missing}")
        candidate_index, candidate = by_candidate[task_id]
        reach_index, reach = by_reachability[task_id]
        runtime_index, replay = by_runtime[task_id]
        if replay.get("status") != "pass" or not reach.get("lower_bound_complete"):
            raise ValueError(f"{name}:{task_id}: RGB pass lacks certified runtime evidence")
        split = str(reach["split"])
        row_index = int(reach["source_row_index"])
        key = task_source_key(task_id, row_index, split)
        objects = candidate.get("target_object", {}).get("objects", [])
        episode = {
            "episode_fingerprint": stable_fingerprint(candidate, reach, key),
            "source_key": key,
            "split": split,
            "source_row_index": row_index,
            "scene_id": str(candidate["scene_id"]),
            "task_id": task_id,
            "object_pair": [f"{obj.get('id')}:{obj.get('label')}" for obj in objects],
            "initial_pose_c2w": candidate["init_camera"]["extrinsics"],
            "terminal_pose_c2w": reach["path"][-1]["c2w"],
            "actions": reach["actions"],
            "difficulty": {
                "certified_lower_bound": reach["certified_lower_bound"],
                "certificate_upper_bound": reach["found_path_length_upper_bound"],
                "first_success_step": reach["first_success_step"],
                "lower_bound_complete": reach["lower_bound_complete"],
            },
            "versions": {
                "generator": candidate.get("generator_version"),
                "materialization": reach.get("version"),
                "runtime_replay": runtime_payload.get("version"),
                "canonical_metric": reach["path"][-1]["canonical_metric"].get("metric_version"),
                "observability": obs.get("version"),
                "collision_convention": replay.get("collision_convention", {}).get("version"),
                "camera": "canonical_h1_from_frozen_candidate_intrinsics_and_pose",
            },
            "validation": {
                "runtime": {"status": replay["status"], "steps": replay["steps"],
                            "initial_success": replay["initial_success"], "final_success": replay["final_success"]},
                "official_rgb": {"passed": obs["passed"], "reasons": obs.get("reasons", []),
                                 "contact_sheet": obs.get("contact_sheet"), "frame_images": obs.get("frame_images")},
            },
            "evidence": {
                "bundle": name,
                "candidate": {**evidence_files["candidate"], "record_index": candidate_index},
                "reachability": {**evidence_files["reachability"], "record_index": reach_index},
                "runtime": {**evidence_files["runtime"], "record_index": runtime_index},
                "observability": {**evidence_files["observability"], "record_index": obs_index},
            },
        }
        episodes.append(episode)
    return episodes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frozen-source-inventory", type=Path, required=True)
    parser.add_argument("--v2-candidate", type=Path, required=True)
    parser.add_argument("--v2-reachability", type=Path, required=True)
    parser.add_argument("--v2-runtime", type=Path, required=True)
    parser.add_argument("--v2-observability", type=Path, required=True)
    parser.add_argument("--v3-candidate", type=Path, required=True)
    parser.add_argument("--v3-reachability", type=Path, required=True)
    parser.add_argument("--v3-runtime", type=Path, required=True)
    parser.add_argument("--v3-observability", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    frozen = json.loads(args.frozen_source_inventory.read_text())
    sources = {source_key(row["split"], row["source_row_index"]): row for row in frozen["records"]}
    bundles = {}
    for name in ("v2", "v3"):
        paths = {role: getattr(args, f"{name}_{role}") for role in
                 ("candidate", "reachability", "runtime", "observability")}
        bundles[name] = load_bundle(name, paths)
    all_episodes = bundles["v2"] + bundles["v3"]
    episode_by_fingerprint: dict[str, dict[str, Any]] = {}
    for episode in all_episodes:
        fingerprint = episode["episode_fingerprint"]
        if fingerprint in episode_by_fingerprint:
            existing = episode_by_fingerprint[fingerprint]
            existing.setdefault("duplicate_validation_evidence", []).append(episode["evidence"])
        else:
            episode_by_fingerprint[fingerprint] = episode
    episodes = sorted(episode_by_fingerprint.values(), key=lambda row: (row["split"], row["source_row_index"], row["episode_fingerprint"]))
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in episodes:
        by_source[episode["source_key"]].append(episode)
    if set(by_source) - set(sources):
        raise ValueError(f"validated sources absent from frozen inventory: {sorted(set(by_source)-set(sources))}")
    records = []
    for key in sorted(by_source):
        source = sources[key]
        records.append({
            "source_key": key,
            "split": source["split"],
            "source_row_index": int(source["source_row_index"]),
            "scene_id": source["scene_id"],
            "task_type": source.get("task_type"),
            "source_task_id": source.get("task_id"),
            "object_pair": source.get("object_pair"),
            "relation": source.get("relation"),
            "source_manifest": frozen["sources"][source["split"]],
            "episode_fingerprints": [episode["episode_fingerprint"] for episode in by_source[key]],
            "episode_count": len(by_source[key]),
            "generator_origins": sorted({episode["evidence"]["bundle"] for episode in by_source[key]}),
        })
    split_counts = Counter(row["split"] for row in records)
    split_episode_counts = Counter(episode["split"] for episode in episodes)
    scene_counts = Counter(row["scene_id"] for row in records)
    payload = {
        "version": VERSION,
        "source_inventory": {"path": str(args.frozen_source_inventory.resolve()),
                             "sha256": sha256(args.frozen_source_inventory)},
        "accounting": {
            "v2_rgb_pass_records": len(bundles["v2"]),
            "v3_rgb_pass_records": len(bundles["v3"]),
            "validated_unique_sources": len(records),
            "validated_unique_episodes": len(episodes),
            "deduplicated_cross_bundle_episodes": len(all_episodes) - len(episodes),
            "split_unique_sources": dict(sorted(split_counts.items())),
            "split_unique_episodes": dict(sorted(split_episode_counts.items())),
            "scene_unique_sources": dict(sorted(scene_counts.items())),
            "train_only_inventory_rule": "only split=train is eligible for a future training pool; validation/test/OOD remain isolated",
        },
        "records": records,
        "episodes": episodes,
    }
    write_json(args.output_dir / "verified_medium_inventory.json", payload)
    sums = []
    for path in sorted(args.output_dir.glob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            sums.append(f"{sha256(path)}  {path.name}")
    (args.output_dir / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    print(json.dumps(payload["accounting"], sort_keys=True))


if __name__ == "__main__":
    main()
