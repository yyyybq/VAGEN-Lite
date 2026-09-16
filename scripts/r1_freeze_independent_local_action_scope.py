#!/usr/bin/env python3
"""Freeze an R1-unseen scene/source scope for independent local-action evaluation.

This stage is deliberately model-free and geometry-free.  It fixes the scene
universe, source ordering, source-manifest hashes, and all downstream search
budgets before local states or policy outputs are observed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


VERSION = "r1_independent_local_action_scope_v1"
R1_DEVELOPMENT_SCENES = {
    "0011_840866", "0012_840878", "0014_841007", "0015_840888",
    "0016_840873", "0019_840447", "0020_840256", "0059_839917",
    "0226_840298", "0229_840306", "0240_840881", "0265_840795",
    "0267_840790", "0270_840784", "0276_840780", "0300_840573",
    "0314_840535", "0328_840489", "0349_840373", "0361_840315",
    "0367_840260",
}
AMBIGUOUS_COLLISION_SCENES = {
    "0059_839917", "0265_840795", "0270_840784", "0314_840535",
    "0328_840489", "0349_840373",
}
NONTRAIN_SPLITS = (
    "id_test", "ood_category", "ood_geometry", "ood_instance",
    "ood_scene", "ood_template", "val_id", "validation_proxy",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open() if line.strip()]


def pair(row: dict[str, Any]) -> list[str]:
    objects = row.get("target_object", {}).get("objects") or []
    return [f"{obj.get('id')}:{obj.get('label')}" for obj in objects]


def source_signature(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        str(row.get("scene_id")), tuple(pair(row)),
        str(row.get("target_region", {}).get("params", {}).get("relation")),
        str(row.get("task_description")),
    )


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scene-scope", type=int, default=18)
    parser.add_argument("--source-cap-per-scene", type=int, default=20)
    parser.add_argument("--minimum-nontrain-projective", type=int, default=12)
    parser.add_argument("--parent-target", type=int, default=60)
    args = parser.parse_args()

    ledger = json.loads(args.ledger.read_text())
    source_info = ledger["source_manifests"]
    train_path = Path(source_info["train"]["source"])
    if sha256(train_path) != source_info["train"]["source_sha256"]:
        raise RuntimeError("v46 train manifest hash mismatch")
    train_rows = rows(train_path)
    train_scenes = {str(row.get("scene_id")) for row in train_rows}
    train_signatures = {
        source_signature(row) for row in train_rows
        if row.get("task_type") == "projective_relations"
    }
    ledger_by_scene = {str(row["scene_id"]): row for row in ledger["scenes"]}

    candidates: dict[str, list[dict[str, Any]]] = defaultdict(list)
    frozen_sources: dict[str, dict[str, Any]] = {}
    for split in NONTRAIN_SPLITS:
        info = source_info[split]
        path = Path(info["source"])
        actual = sha256(path)
        if actual != info["source_sha256"]:
            raise RuntimeError(f"source manifest hash mismatch for {split}")
        frozen_sources[split] = {
            "path": str(path.resolve()), "sha256": actual, "rows": int(info["rows"]),
        }
        for index, row in enumerate(rows(path)):
            if row.get("task_type") != "projective_relations":
                continue
            scene = str(row.get("scene_id") or "")
            if scene in R1_DEVELOPMENT_SCENES or scene in AMBIGUOUS_COLLISION_SCENES:
                continue
            digest = hashlib.sha256(
                f"r1-independent-local-action-v1:{split}:{index}".encode()
            ).hexdigest()
            candidates[scene].append({
                "split": split,
                "source_row_index": index,
                "source_key": f"{split}:{index}",
                "scene_id": scene,
                "object_pair": pair(row),
                "relation": row.get("target_region", {}).get("params", {}).get("relation"),
                "selection_digest": digest,
                "r1_development_seen": False,
                "v46_training_scene_seen": scene in train_scenes,
                "v46_training_exact_source_seen": source_signature(row) in train_signatures,
            })

    eligible = []
    for scene, values in candidates.items():
        entry = ledger_by_scene.get(scene, {})
        if (
            len(values) >= args.minimum_nontrain_projective
            and entry.get("asset_validation", {}).get("status") == "valid"
            and entry.get("download_status") == "ready"
        ):
            eligible.append(scene)
    scenes = sorted(eligible)[: args.scene_scope]
    if len(scenes) != args.scene_scope:
        raise RuntimeError(f"only {len(scenes)} eligible R1-unseen scenes")

    records = []
    for scene in scenes:
        ordered = sorted(
            candidates[scene], key=lambda row: (row["selection_digest"], row["source_key"])
        )
        records.extend(ordered[: args.source_cap_per_scene])

    counts = Counter(record["scene_id"] for record in records)
    payload = {
        "version": VERSION,
        "selection_is_frozen_before_geometry_or_model_results": True,
        "selection_rule": {
            "scene_rule": (
                "lexicographically first asset-valid, collision-unambiguous scenes with the "
                "minimum number of non-train Projective rows after excluding every scene named "
                "in the versioned R1 reports through 2026-09-16"
            ),
            "source_rule": (
                "within scene SHA256(r1-independent-local-action-v1:split:source_row_index)"
            ),
            "nontrain_splits": list(NONTRAIN_SPLITS),
            "scene_scope": args.scene_scope,
            "source_cap_per_scene": args.source_cap_per_scene,
            "minimum_nontrain_projective": args.minimum_nontrain_projective,
            "parent_target": args.parent_target,
        },
        "state_construction_budget": {
            "success_seed_cap_per_source": 12,
            "candidate_state_cap_per_distance_per_source": 4,
            "formal_distance_complete_through_depth": 2,
            "target_yaw_offsets_deg": [0, -5, 5, -10, 10, -15, 15],
            "final_parent_selection": (
                "round-robin by scene and frozen source order; dual-distance parents first, "
                "then single-distance parents; stop at 60; require at least 10 scenes"
            ),
            "model_results_used_for_selection": False,
        },
        "analysis_plan_frozen_before_policy_inference": {
            "primary_unit": "parent source; derived d*=1/2 states are grouped",
            "local_low_hit_replication": {
                "minimum_parent_sources": 60,
                "minimum_r1_unseen_scenes": 10,
                "maximum_hit_rate_per_model_per_distance": 0.25,
                "require_each_model_overall_below_uniform_six_action_expectation": True,
                "note": (
                    "Descriptive development diagnostic, not a significance test or an "
                    "independent v46 scene-OOD evaluation."
                ),
            },
            "action_bias_attribution": (
                "Compare each model with all six fixed-action baselines and the uniform "
                "six-action expectation; report the gap descriptively without causal claims."
            ),
            "no_result_conditioned_prompt_or_selection_changes": True,
        },
        "prior_use": {
            "r1_development_scene_definition": (
                "scene explicitly named in a versioned R1 report through 2026-09-16"
            ),
            "excluded_r1_development_scenes": sorted(R1_DEVELOPMENT_SCENES),
            "selected_scenes_r1_development_seen": {scene: False for scene in scenes},
            "v46_training_seen_is_reported_separately": True,
        },
        "collision": {
            "version": "interiorgs_structure_label_alignment_v1",
            "excluded_ambiguous_scenes": sorted(AMBIGUOUS_COLLISION_SCENES),
        },
        "ledger": {"path": str(args.ledger.resolve()), "sha256": sha256(args.ledger)},
        "sources": frozen_sources,
        "scenes": scenes,
        "records": records,
        "counts": {
            "records": len(records),
            "scene": dict(sorted(counts.items())),
            "v46_training_scene_seen": dict(sorted(Counter(
                str(record["v46_training_scene_seen"]) for record in records
            ).items())),
            "v46_training_exact_source_seen": dict(sorted(Counter(
                str(record["v46_training_exact_source_seen"]) for record in records
            ).items())),
        },
    }
    atomic_json(args.output, payload)
    print(json.dumps({"scenes": scenes, "records": len(records), "counts": payload["counts"]}, indent=2))


if __name__ == "__main__":
    main()
