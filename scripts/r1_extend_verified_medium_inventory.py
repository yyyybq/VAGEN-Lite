#!/usr/bin/env python3
"""Extend the frozen verified-Medium inventory with independently validated bundles."""
from __future__ import annotations

import argparse
import copy
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from r1_freeze_verified_medium_inventory import load_bundle, read_jsonl, sha256, source_key, write_json


VERSION = "r1_verified_projective_medium_inventory_v2_extended_roles"
COMPONENTS = {
    "camera": "vagen/envs/active_spatial/canonical_camera.py",
    "canonical_metric": "vagen/envs/active_spatial/canonical_task_metrics.py",
    "collision": "vagen/envs/active_spatial/collision_detector.py",
    "action_graph": "scripts/r1_action_graph.py",
    "observability": "scripts/r1_projective_observability.py",
}


def _record_from_source(row: dict[str, Any], split: str, index: int, manifest: dict[str, Any]) -> dict[str, Any]:
    objects = row.get("target_object", {}).get("objects", [])
    params = row.get("target_region", {}).get("params", {})
    relation = params.get("relation") or params.get("relation_type") or row.get("relation")
    return {
        "source_key": source_key(split, index), "split": split, "source_row_index": index,
        "scene_id": str(row.get("scene_id")), "task_type": row.get("task_type"),
        "source_task_id": row.get("task_id"),
        "object_pair": [f"{obj.get('id')}:{obj.get('label')}" for obj in objects[:2]],
        "relation": relation, "source_manifest": manifest,
        "episode_fingerprints": [], "episode_count": 0, "generator_origins": [],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-inventory", type=Path, required=True)
    parser.add_argument("--source-inventory", type=Path, required=True)
    parser.add_argument("--bundle-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    base = json.loads(args.base_inventory.read_text())
    source_inventory = json.loads(args.source_inventory.read_text())
    source_map = source_inventory.get("sources", source_inventory)
    source_rows = {
        split: read_jsonl(Path(value["path"] if isinstance(value, dict) else value))
        for split, value in source_map.items()
    }
    config = json.loads(args.bundle_config.read_text())
    episodes = [copy.deepcopy(row) for row in base["episodes"]]
    for episode in episodes:
        episode.setdefault("experiment_roles", ["development_canary10_verified_medium"])
    bundle_counts = {}
    for bundle in config["bundles"]:
        paths = {role: Path(bundle[role]) for role in ("candidate", "reachability", "runtime", "observability")}
        loaded = load_bundle(str(bundle["name"]), paths)
        for episode in loaded:
            episode["experiment_roles"] = [str(bundle["experiment_role"])]
        episodes.extend(loaded)
        bundle_counts[str(bundle["name"])] = len(loaded)
    by_fingerprint: dict[str, dict[str, Any]] = {}
    for episode in episodes:
        fingerprint = str(episode["episode_fingerprint"])
        if fingerprint not in by_fingerprint:
            by_fingerprint[fingerprint] = episode
            continue
        existing = by_fingerprint[fingerprint]
        existing["experiment_roles"] = sorted(set(existing.get("experiment_roles", [])) | set(episode.get("experiment_roles", [])))
        duplicate = copy.deepcopy(episode["evidence"])
        duplicate["experiment_roles"] = episode.get("experiment_roles", [])
        existing.setdefault("duplicate_validation_evidence", []).append(duplicate)
    unique_episodes = sorted(by_fingerprint.values(), key=lambda row: (row["split"], row["source_row_index"], row["episode_fingerprint"]))
    by_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for episode in unique_episodes:
        by_source[str(episode["source_key"])].append(episode)
    base_records = {str(row["source_key"]): copy.deepcopy(row) for row in base["records"]}
    records = []
    for key in sorted(by_source):
        split, raw_index = key.rsplit(":", 1)
        index = int(raw_index)
        if key in base_records:
            record = base_records[key]
        else:
            if split not in source_rows or index >= len(source_rows[split]):
                raise ValueError(f"missing source manifest row for {key}")
            record = _record_from_source(source_rows[split][index], split, index, source_map[split])
        source_episodes = by_source[key]
        record["episode_fingerprints"] = [row["episode_fingerprint"] for row in source_episodes]
        record["episode_count"] = len(source_episodes)
        record["generator_origins"] = sorted({str(row["evidence"]["bundle"]) for row in source_episodes})
        record["experiment_roles"] = sorted({role for row in source_episodes for role in row.get("experiment_roles", [])})
        record["development_scene_usage"] = sorted({
            "independent_selector_validation_20260914" if "independent" in role
            else "original_canary10_development"
            for role in record["experiment_roles"]
        })
        records.append(record)
    repo = Path(__file__).resolve().parents[1]
    component_fingerprints = {
        name: {"path": relative, "sha256": sha256(repo / relative)}
        for name, relative in COMPONENTS.items()
    }
    split_sources = Counter(row["split"] for row in records)
    split_episodes = Counter(row["split"] for row in unique_episodes)
    role_sources: dict[str, set[str]] = defaultdict(set)
    for record in records:
        for role in record["experiment_roles"]:
            role_sources[role].add(record["source_key"])
    payload = {
        "version": VERSION,
        "base_inventory": {"path": str(args.base_inventory.resolve()), "sha256": sha256(args.base_inventory)},
        "source_inventory": {"path": str(args.source_inventory.resolve()), "sha256": sha256(args.source_inventory)},
        "bundle_config": {"path": str(args.bundle_config.resolve()), "sha256": sha256(args.bundle_config)},
        "component_fingerprints": component_fingerprints,
        "accounting": {
            "bundle_rgb_pass_records": bundle_counts,
            "input_episode_records_before_fingerprint_dedup": len(episodes),
            "validated_unique_sources": len(records),
            "validated_unique_episodes": len(unique_episodes),
            "deduplicated_episode_records": len(episodes) - len(unique_episodes),
            "split_unique_sources": dict(sorted(split_sources.items())),
            "split_unique_episodes": dict(sorted(split_episodes.items())),
            "role_unique_sources": {key: len(value) for key, value in sorted(role_sources.items())},
            "train_only_unique_sources": split_sources.get("train", 0),
            "evaluation_unique_sources": len(records) - split_sources.get("train", 0),
            "train_eval_isolation": "Only original split=train identities are train-eligible; every validation/test/OOD identity remains evaluation-only.",
        },
        "records": records,
        "episodes": unique_episodes,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "verified_medium_inventory.json", payload)
    sums = []
    for path in sorted(args.output_dir.glob("*")):
        if path.is_file() and path.name != "SHA256SUMS":
            sums.append(f"{sha256(path)}  {path.name}")
    (args.output_dir / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    print(json.dumps(payload["accounting"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
