#!/usr/bin/env python3
"""Versioned, cardinality-accounted R1 regeneration for all formal manifests."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from gen_ood_splits import GEOM_THRESHOLDS, TEMPLATE_REWRITERS
from r1_reachability_audit import search_one
from r1_repair_pipeline import (
    FOV_GENERATOR_VERSION,
    PROJECTIVE_GENERATOR_VERSION,
    SceneConstraints,
    repair_fov,
    repair_projective,
    projective_difficulty_profile,
)
from r1_donor_allocator import AtomicDonorLedger
from vagen.envs.active_spatial.collision_detector import create_collision_detector


TARGET_TASKS = frozenset(("projective_relations", "fov_inclusion"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def atomic_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def pair_key(row: dict[str, Any]) -> tuple[str, ...]:
    objects = row.get("target_object", {}).get("objects") or []
    identity = tuple(f"{obj.get('id')}:{obj.get('label')}" for obj in objects)
    relation = str(row.get("target_region", {}).get("params", {}).get("relation") or "")
    return (str(row.get("scene_id") or ""), *identity, relation)


def row_fingerprint(row: dict[str, Any]) -> str:
    payload = {
        "scene_id": row.get("scene_id"),
        "task_type": row.get("task_type"),
        "pair": pair_key(row),
        "sample_target": row.get("sample_target"),
        "init": row.get("init_camera", {}).get("extrinsics"),
        "description": row.get("task_description"),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def bbox_measure(row: dict[str, Any]) -> float:
    values = []
    for obj in row.get("target_object", {}).get("objects") or []:
        lo = np.asarray(obj.get("bbox_min") or [0, 0, 0], dtype=float)
        hi = np.asarray(obj.get("bbox_max") or [0, 0, 0], dtype=float)
        extent = np.maximum(hi - lo, 1e-6)
        values.append(float(np.prod(extent)))
    return sum(values)


def yaw(row: dict[str, Any]) -> float:
    forward = np.asarray(row.get("camera_params", {}).get("forward") or [1, 0, 0], dtype=float)
    return math.atan2(float(forward[1]), float(forward[0]))


def angle_delta(a: float, b: float) -> float:
    return abs((a - b + math.pi) % (2 * math.pi) - math.pi)


def split_constraint(
    split: str,
    source: dict[str, Any],
    candidate: dict[str, Any],
    train_scenes: set[str],
    train_labels: set[str],
) -> tuple[bool, str]:
    if candidate.get("scene_id") != source.get("scene_id"):
        return False, "scene_changed"
    if candidate.get("task_type") != source.get("task_type"):
        return False, "task_type_changed"
    if source.get("task_type") == "projective_relations":
        old_relation = source.get("target_region", {}).get("params", {}).get("relation")
        new_relation = candidate.get("target_region", {}).get("params", {}).get("relation")
        if new_relation != old_relation:
            return False, "projective_relation_changed"
    label = str(candidate.get("object_label") or "")
    if split == "ood_scene" and str(candidate.get("scene_id")) in train_scenes:
        return False, "ood_scene_leak"
    if split == "ood_instance" and label not in train_labels:
        return False, "ood_instance_category_leak"
    if split == "ood_category" and label in train_labels:
        return False, "ood_category_leak"
    if split == "ood_template":
        if not candidate.get("task_description_original"):
            return False, "template_original_missing"
        if candidate.get("task_description") == candidate.get("task_description_original"):
            return False, "template_not_rewritten"
    if split == "ood_geometry":
        low, high = GEOM_THRESHOLDS[str(candidate["task_type"])]
        distance = float(candidate.get("distance", 0.0))
        if low <= distance <= high:
            return False, "geometry_returned_to_train_band"
    return True, "passed"


def prepare_replacement(donor: dict[str, Any], split: str) -> dict[str, Any]:
    candidate = copy.deepcopy(donor)
    if split == "ood_template":
        original = str(candidate.get("task_description") or "")
        rewrite = TEMPLATE_REWRITERS.get(str(candidate.get("task_type")))
        candidate["task_description_original"] = original
        candidate["task_description"] = rewrite(original) if rewrite else original
    return candidate


def replacement_candidates(
    source: dict[str, Any],
    split: str,
    universe: list[tuple[str, int, dict[str, Any]]],
    forbidden_pairs: set[tuple[str, ...]],
    used_pairs: set[tuple[str, ...]],
    train_scenes: set[str],
    train_labels: set[str],
) -> list[tuple[str, int, dict[str, Any]]]:
    result = []
    for donor_split, donor_index, donor in universe:
        key = pair_key(donor)
        if key in forbidden_pairs or key in used_pairs:
            continue
        if donor.get("scene_id") != source.get("scene_id"):
            continue
        if donor.get("task_type") != source.get("task_type"):
            continue
        candidate = prepare_replacement(donor, split)
        valid, _ = split_constraint(split, source, candidate, train_scenes, train_labels)
        if not valid:
            continue
        category_penalty = 0 if donor.get("object_label") == source.get("object_label") else 1
        rank = (
            category_penalty,
            abs(bbox_measure(donor) - bbox_measure(source)),
            abs(float(donor.get("distance", 0.0)) - float(source.get("distance", 0.0))),
            angle_delta(yaw(donor), yaw(source)),
            donor_split,
            donor_index,
        )
        result.append((rank, donor_split, donor_index, candidate))
    result.sort(key=lambda entry: entry[0])
    return [(entry[1], entry[2], entry[3]) for entry in result]


def repair_one(index: int, row: dict[str, Any], constraints: SceneConstraints):
    if row.get("task_type") == "fov_inclusion":
        return repair_fov(index, row, constraints)
    return repair_projective(index, row, constraints)


def repair_one_cached(
    index: int,
    row: dict[str, Any],
    constraints: SceneConstraints,
    cache: dict[str, tuple[dict[str, Any] | None, dict[str, Any]]],
):
    """Reuse deterministic pose search while retaining per-source lineage IDs."""
    key = row_fingerprint(row)
    if key not in cache:
        cache[key] = repair_one(index, row, constraints)
    repaired, details = copy.deepcopy(cache[key])
    details["source_row_index"] = index
    if repaired is None:
        return None, details
    if repaired.get("task_type") == "fov_inclusion":
        task_id = f"fov_canonical_h1_v2_{index:06d}"
    else:
        task_id = f"projective_canonical_h1_v5_{index:06d}"
    repaired["task_id"] = task_id
    repaired.setdefault("repair_lineage", {})["source_row_index"] = index
    details["new_task_id"] = task_id
    return repaired, details


def select_final_tier(
    split: str, index: int, source: dict[str, Any], details: dict[str, Any]
) -> tuple[bool, str]:
    if split == "validation_proxy":
        return True, "validation_proxy_high_value"
    digest = hashlib.sha256(
        f"{split}:{index}:{source.get('scene_id')}:{source.get('task_type')}".encode()
    ).digest()
    return (digest[0] % 10 == 0), "deterministic_ten_percent_representative"


def audit_reachability_tiered(
    repaired: dict[str, Any],
    detector: Any,
    gs_root: Path,
    max_steps: int,
    expansion_tiers: list[int],
    final_tier_expansions: int,
    run_final_tier: bool,
    final_tier_reason: str,
) -> dict[str, Any]:
    scene_id = str(repaired["scene_id"])
    if not detector.load_scene_from_gs_root(str(gs_root), scene_id):
        return {"status": "reachability_unverified_asset_load", "reachability_verified": False}
    kind = "fov" if repaired.get("task_type") == "fov_inclusion" else "projective"
    tiers = list(expansion_tiers)
    if run_final_tier and final_tier_expansions not in tiers:
        tiers.append(final_tier_expansions)
    tier_rows = []
    result = None
    for budget in tiers:
        result = search_one(
            repaired,
            detector,
            kind=kind,
            max_steps=max_steps,
            max_expansions=budget,
            step_translation=0.3,
            step_rotation_deg=20.0,
        )
        tier_rows.append(
            {
                "max_expansions": budget,
                "status": result.get("status"),
                "expansions": result.get("expansions", 0),
                "visited_states": result.get("visited_states"),
                "steps": result.get("steps"),
            }
        )
        if result.get("status") != "reachability_unverified_expansion_cap":
            break
    assert result is not None
    result["reachability_tiers"] = tier_rows
    result["final_250k_selected"] = bool(run_final_tier)
    result["final_250k_selection_reason"] = final_tier_reason
    return result


def tier_summary(rows: list[dict[str, Any]], budgets: list[int]) -> dict[str, Any]:
    resolved_at: Counter[int] = Counter()
    attempted: Counter[int] = Counter()
    costs: Counter[int] = Counter()
    for row in rows:
        for tier in row.get("reachability_tiers") or []:
            budget = int(tier["max_expansions"])
            attempted[budget] += 1
            costs[budget] += int(tier.get("expansions") or 0)
            if tier.get("status") != "reachability_unverified_expansion_cap":
                resolved_at[budget] += 1
                break
    remaining = len(rows)
    result = {}
    for budget in budgets:
        remaining -= resolved_at[budget]
        result[str(budget)] = {
            "attempted": attempted[budget],
            "newly_resolved": resolved_at[budget],
            "total_expansions": costs[budget],
            "average_expansions": costs[budget] / attempted[budget] if attempted[budget] else None,
            "planner_budget_unresolved_after_tier": remaining,
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--gs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--scenes", help="optional comma-separated canary scene IDs")
    parser.add_argument("--splits", help="optional comma-separated manifest split names")
    parser.add_argument("--index-shard-count", type=int, default=1)
    parser.add_argument("--index-shard-id", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument(
        "--reachability-tiers",
        default="2000,25000",
        help="comma-separated quick and escalation budgets; exhaustion remains unverified",
    )
    parser.add_argument("--final-tier-expansions", type=int, default=250000)
    parser.add_argument(
        "--disable-final-tier",
        action="store_true",
        help="keep this run at the configured quick/escalation tiers",
    )
    parser.add_argument("--collision-convention-overrides", type=Path)
    parser.add_argument("--max-replacement-candidates", type=int, default=64)
    parser.add_argument(
        "--donor-ledger",
        type=Path,
        help="optional cross-shard atomic donor reservation ledger",
    )
    parser.add_argument(
        "--source-index-selection",
        type=Path,
        help="optional JSON object mapping split names to exact source indices",
    )
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    expansion_tiers = [int(value) for value in args.reachability_tiers.split(",") if value]
    if not expansion_tiers or expansion_tiers != sorted(set(expansion_tiers)):
        raise ValueError("reachability tiers must be unique and increasing")
    if args.index_shard_count < 1 or not 0 <= args.index_shard_id < args.index_shard_count:
        raise ValueError("invalid index shard")
    collision_overrides = {}
    if args.collision_convention_overrides:
        collision_overrides = json.loads(
            args.collision_convention_overrides.read_text()
        ).get("structure_y_sign_overrides", {})

    raw_sources = json.loads(args.sources.read_text())
    sources = {
        split: (path if (path := Path(value)).is_absolute() else Path.cwd() / path)
        for split, value in raw_sources.items()
    }
    rows_by_split = {split: read_jsonl(path) for split, path in sources.items()}
    requested_splits = {value for value in (args.splits or "").split(",") if value}
    unknown_splits = requested_splits - set(rows_by_split)
    if unknown_splits:
        raise ValueError(f"unknown splits: {sorted(unknown_splits)}")
    train_rows = rows_by_split["train"]
    train_scenes = {str(row.get("scene_id")) for row in train_rows}
    train_labels = {str(row.get("object_label")) for row in train_rows}
    requested_scenes = {value for value in (args.scenes or "").split(",") if value}
    selected_indices = {}
    if args.source_index_selection:
        selection_payload = json.loads(args.source_index_selection.read_text())
        if "records" in selection_payload:
            selected_indices = defaultdict(set)
            for record in selection_payload["records"]:
                selected_indices[record["split"]].add(int(record["source_row_index"]))
        else:
            selected_indices = {
                split: {int(index) for index in indices}
                for split, indices in selection_payload.items()
            }
    universe = [
        (split, index, row)
        for split, rows in rows_by_split.items()
        for index, row in enumerate(rows)
        if row.get("task_type") in TARGET_TASKS
    ]
    constraints = SceneConstraints(
        args.gs_root, structure_y_sign_overrides=collision_overrides
    )
    detector = create_collision_detector(
        {
            "camera_radius": 0.15,
            "floor_height": 0.3,
            "ceiling_height": 2.5,
            "safety_margin": 0.05,
            "enable_object_collision": True,
            "enable_boundary_collision": True,
            "structure_y_sign_overrides": collision_overrides,
        }
    )
    global_summary = {}
    repair_cache: dict[str, tuple[dict[str, Any] | None, dict[str, Any]]] = {}
    donor_ledger = AtomicDonorLedger(args.donor_ledger) if args.donor_ledger else None

    for split, source_rows in rows_by_split.items():
        if requested_splits and split not in requested_splits:
            continue
        split_dir = args.output_dir / split
        in_scope = {
            index
            for index, row in enumerate(source_rows)
            if row.get("task_type") in TARGET_TASKS
            and (not requested_scenes or str(row.get("scene_id")) in requested_scenes)
            and (not selected_indices or index in selected_indices.get(split, set()))
        }
        in_scope = {
            index for position, index in enumerate(sorted(in_scope))
            if position % args.index_shard_count == args.index_shard_id
        }
        forbidden_pairs = {pair_key(source_rows[index]) for index in in_scope}
        used_replacement_pairs: set[tuple[str, ...]] = set()
        accepted_by_index: dict[int, dict[str, Any]] = {}
        candidate_by_index: dict[int, dict[str, Any]] = {}
        mappings: list[dict[str, Any]] = []
        accounting: list[dict[str, Any]] = []
        failures: list[dict[str, Any]] = []
        unverified: list[dict[str, Any]] = []
        reachability_rows: list[dict[str, Any]] = []

        completed_summary = split_dir / "summary.json"
        if args.resume and completed_summary.is_file():
            summary = json.loads(completed_summary.read_text())
            if summary.get("source_target_rows_in_scope") != len(in_scope):
                raise RuntimeError(f"resume scope mismatch for completed split {split}")
            global_summary[split] = summary
            print(json.dumps({"split": split, "resume": "completed_split_skipped"}), flush=True)
            continue

        checkpoint_paths = {
            "accounting": split_dir / "accounting_checkpoint.jsonl",
            "repaired": split_dir / "repaired_checkpoint.jsonl",
            "mapping": split_dir / "mapping_checkpoint.jsonl",
            "reachability": split_dir / "reachability_checkpoint.jsonl",
            "candidates": split_dir / "candidate_checkpoint.jsonl",
        }
        if args.resume and all(path.is_file() for path in checkpoint_paths.values()):
            accounting = read_jsonl(checkpoint_paths["accounting"])
            wrappers = read_jsonl(checkpoint_paths["repaired"])
            accepted_by_index = {
                int(wrapper["source_row_index"]): wrapper["row"] for wrapper in wrappers
            }
            candidate_by_index = {
                int(wrapper["source_row_index"]): wrapper["row"]
                for wrapper in read_jsonl(checkpoint_paths["candidates"])
            }
            mappings = read_jsonl(checkpoint_paths["mapping"])
            reachability_rows = read_jsonl(checkpoint_paths["reachability"])
            failures = [row for row in accounting if row.get("status") == "hard_failure"]
            unverified = [row for row in accounting if row.get("status") == "unverified"]
            for row in accounting:
                new_pair = (row.get("replacement_lineage") or {}).get("new_pair")
                if new_pair:
                    used_replacement_pairs.add(tuple(new_pair))
                    if donor_ledger:
                        donor_ledger.seed(
                            f"{split}:{row['source_row_index']}", tuple(new_pair)
                        )
            if any(int(row["source_row_index"]) not in in_scope for row in accounting):
                raise RuntimeError(f"resume checkpoint scope mismatch for split {split}")
            print(
                json.dumps(
                    {
                        "split": split,
                        "resume": "checkpoint_loaded",
                        "accounted": len(accounting),
                        "in_scope": len(in_scope),
                    }
                ),
                flush=True,
            )

        processed_indices = {int(row["source_row_index"]) for row in accounting}

        for position, index in enumerate(sorted(in_scope), start=1):
            if index in processed_indices:
                continue
            source = source_rows[index]
            print(
                json.dumps(
                    {
                        "split": split,
                        "position": position,
                        "in_scope": len(in_scope),
                        "source_row_index": index,
                        "task_type": source.get("task_type"),
                        "stage": "repair_start",
                    }
                ),
                flush=True,
            )
            repaired, details = repair_one_cached(index, source, constraints, repair_cache)
            repair_status = "strict_same_pair_repair"
            lineage = None
            reserved_donor_key = None
            constraint_reason = "not_run"
            if repaired is not None:
                valid, constraint_reason = split_constraint(
                    split, source, repaired, train_scenes, train_labels
                )
                if not valid:
                    details = {**details, "failure": constraint_reason}
                    repaired = None

            if repaired is None:
                original_failure = details
                for donor_split, donor_index, candidate in replacement_candidates(
                    source,
                    split,
                    universe,
                    forbidden_pairs,
                    used_replacement_pairs,
                    train_scenes,
                    train_labels,
                )[: args.max_replacement_candidates]:
                    replacement, replacement_details = repair_one_cached(
                        index, candidate, constraints, repair_cache
                    )
                    if replacement is None:
                        continue
                    valid, constraint_reason = split_constraint(
                        split, source, replacement, train_scenes, train_labels
                    )
                    if not valid:
                        continue
                    if donor_ledger:
                        reserved = donor_ledger.reserve(
                            f"{split}:{index}", [pair_key(candidate)]
                        )
                        if reserved is None:
                            continue
                        reserved_donor_key = reserved
                    repaired = replacement
                    details = replacement_details
                    repair_status = "count_matched_replacement"
                    used_replacement_pairs.add(pair_key(candidate))
                    lineage = {
                        "old_source_row_index": index,
                        "old_task_id": source.get("task_id"),
                        "old_pair": pair_key(source),
                        "donor_split": donor_split,
                        "donor_source_row_index": donor_index,
                        "new_pair": pair_key(candidate),
                        "reason": original_failure.get("failure"),
                        "old_source_difficulty": original_failure.get("source_difficulty"),
                        "donor_source_difficulty": replacement_details.get("source_difficulty"),
                        "new_repaired_difficulty": replacement_details.get("repaired_difficulty"),
                    }
                    replacement["repair_lineage"] = {
                        **replacement.get("repair_lineage", {}),
                        "replacement": lineage,
                    }
                    details["replacement_lineage"] = lineage
                    break

            base = {
                "split": split,
                "source_row_index": index,
                "scene_id": source.get("scene_id"),
                "task_type": source.get("task_type"),
                "old_task_id": source.get("task_id"),
                "source_fingerprint": row_fingerprint(source),
            }
            if repaired is None:
                record = {
                    **base,
                    "status": "hard_failure",
                    "failure": details.get("failure"),
                    "failure_taxonomy": details.get(
                        "failure_taxonomy", "source object pair semantically infeasible"
                    ),
                    "repair": details,
                }
                accounting.append(record)
                failures.append(record)
            else:
                candidate_by_index[index] = repaired
                final_selected, final_reason = select_final_tier(split, index, source, details)
                if args.disable_final_tier:
                    final_selected, final_reason = False, "final_tier_disabled_for_small_sample"
                reachability = audit_reachability_tiered(
                    repaired,
                    detector,
                    args.gs_root,
                    args.max_steps,
                    expansion_tiers,
                    args.final_tier_expansions,
                    final_selected,
                    final_reason,
                )
                reachability["source_row_index"] = index
                reachability_rows.append(reachability)
                if reachability.get("status") == "reachable":
                    if details.get("repaired_difficulty") is not None:
                        steps = int(reachability.get("steps") or 0)
                        details["repaired_difficulty"]["planner_steps_found"] = steps
                        details["repaired_difficulty"]["actual_planner_step_bucket"] = sum(
                            steps >= edge for edge in (3, 6, 9, 12)
                        )
                    accepted_by_index[index] = repaired
                    mapping = {
                        **details,
                        "split": split,
                        "r1_status": repair_status,
                        "split_constraint": constraint_reason,
                        "reachability_status": reachability["status"],
                        "found_path_length_upper_bound": reachability.get("steps"),
                        "no_solution_depth_lower_bound": reachability.get(
                            "no_solution_depth_lower_bound"
                        ),
                    }
                    mappings.append(mapping)
                    accounting.append(
                        {
                            **base,
                            "status": repair_status,
                            "new_task_id": repaired.get("task_id"),
                            "replacement_lineage": lineage,
                            "reachability_status": reachability["status"],
                            "found_path_length_upper_bound": reachability.get("steps"),
                            "reachability_tiers": reachability.get("reachability_tiers"),
                            "source_difficulty": details.get("source_difficulty"),
                            "repaired_difficulty": details.get("repaired_difficulty"),
                        }
                    )
                elif reachability.get("reachability_verified"):
                    if donor_ledger and reserved_donor_key is not None:
                        donor_ledger.release(f"{split}:{index}", reserved_donor_key)
                    record = {
                        **base,
                        "status": "hard_failure",
                        "repair_status_before_reachability": repair_status,
                        "failure": reachability.get("status"),
                        "failure_taxonomy": "12-step unreachable",
                        "reachability": reachability,
                    }
                    accounting.append(record)
                    failures.append(record)
                else:
                    if donor_ledger and reserved_donor_key is not None:
                        donor_ledger.release(f"{split}:{index}", reserved_donor_key)
                    record = {
                        **base,
                        "status": "unverified",
                        "repair_status_before_reachability": repair_status,
                        "failure": reachability.get("status"),
                        "failure_taxonomy": (
                            "planner budget unverified"
                            if reachability.get("status") == "reachability_unverified_expansion_cap"
                            else "layout/collision infeasible"
                        ),
                        "reachability": reachability,
                    }
                    accounting.append(record)
                    unverified.append(record)

            print(
                json.dumps(
                    {
                        "split": split,
                        "position": position,
                        "source_row_index": index,
                        "status": accounting[-1]["status"],
                        "path_upper_bound": accounting[-1].get(
                            "found_path_length_upper_bound"
                        ),
                        "repair_cache_entries": len(repair_cache),
                    }
                ),
                flush=True,
            )

            if position % 10 == 0 or position == len(in_scope):
                atomic_jsonl(split_dir / "accounting_checkpoint.jsonl", accounting)
                atomic_jsonl(
                    split_dir / "repaired_checkpoint.jsonl",
                    [
                        {"source_row_index": source_index, "row": row}
                        for source_index, row in sorted(accepted_by_index.items())
                    ],
                )
                atomic_jsonl(split_dir / "mapping_checkpoint.jsonl", mappings)
                atomic_jsonl(split_dir / "reachability_checkpoint.jsonl", reachability_rows)
                atomic_jsonl(
                    split_dir / "candidate_checkpoint.jsonl",
                    [
                        {"source_row_index": source_index, "row": row}
                        for source_index, row in sorted(candidate_by_index.items())
                    ],
                )
                print(
                    json.dumps(
                        {
                            "split": split,
                            "processed": position,
                            "in_scope": len(in_scope),
                            "counts": dict(Counter(row["status"] for row in accounting)),
                        }
                    ),
                    flush=True,
                )

        if requested_scenes:
            output_rows = [accepted_by_index[index] for index in sorted(accepted_by_index)]
            output_semantics = "canary_target_rows_only"
        else:
            output_rows = []
            for index, row in enumerate(source_rows):
                if row.get("task_type") not in TARGET_TASKS:
                    output_rows.append(row)
                elif index in accepted_by_index:
                    output_rows.append(accepted_by_index[index])
            output_semantics = "full_manifest_non_targets_unchanged_failed_targets_excluded"

        output_path = split_dir / "trainable.jsonl"
        atomic_jsonl(output_path, output_rows)
        atomic_jsonl(split_dir / "mapping.jsonl", mappings)
        atomic_jsonl(split_dir / "accounting.jsonl", accounting)
        atomic_jsonl(split_dir / "failure_manifest.jsonl", failures)
        atomic_jsonl(split_dir / "unverified_manifest.jsonl", unverified)
        atomic_jsonl(split_dir / "reachability_manifest.jsonl", reachability_rows)
        atomic_jsonl(
            split_dir / "candidate_manifest.jsonl",
            [
                {"source_row_index": source_index, "row": row}
                for source_index, row in sorted(candidate_by_index.items())
            ],
        )
        counts = Counter(row["status"] for row in accounting)
        path_lengths = [
            int(row["found_path_length_upper_bound"])
            for row in accounting
            if row.get("found_path_length_upper_bound") is not None
        ]
        tier_budgets = expansion_tiers + (
            [args.final_tier_expansions]
            if args.final_tier_expansions not in expansion_tiers
            else []
        )
        summary = {
            "split": split,
            "source": str(sources[split]),
            "source_sha256": sha256(sources[split]),
            "source_rows": len(source_rows),
            "source_target_rows_in_scope": len(in_scope),
            "accounted_target_rows": len(accounting),
            "accounting_closed": len(in_scope) == len(accounting),
            "status_counts": dict(counts),
            "same_pair_repair": counts.get("strict_same_pair_repair", 0),
            "replacement": counts.get("count_matched_replacement", 0),
            "hard_failure": counts.get("hard_failure", 0),
            "unverified": counts.get("unverified", 0),
            "reachable_rate": (
                (counts.get("strict_same_pair_repair", 0) + counts.get("count_matched_replacement", 0))
                / len(in_scope)
                if in_scope
                else None
            ),
            "found_path_length_upper_bound": {
                "min": min(path_lengths) if path_lengths else None,
                "max": max(path_lengths) if path_lengths else None,
                "distribution": dict(Counter(path_lengths)),
            },
            "reachability_max_steps": args.max_steps,
            "reachability_expansion_tiers": expansion_tiers,
            "reachability_final_tier_expansions": args.final_tier_expansions,
            "reachability_tier_results": tier_summary(reachability_rows, tier_budgets),
            "output": str(output_path),
            "output_sha256": sha256(output_path),
            "output_rows": len(output_rows),
            "output_semantics": output_semantics,
            "projective_generator_version": PROJECTIVE_GENERATOR_VERSION,
            "fov_generator_version": FOV_GENERATOR_VERSION,
            "replacement_policy": "same scene/task/relation; preserve formal OOD predicate; no pair already present in current manifest; one use per donor pair",
            "donor_ledger": str(args.donor_ledger) if args.donor_ledger else None,
            "source_index_selection": (
                sorted(selected_indices.get(split, set())) if selected_indices else None
            ),
            "scene_filter": sorted(requested_scenes) if requested_scenes else None,
        }
        atomic_json(split_dir / "summary.json", summary)
        global_summary[split] = summary

    totals = Counter()
    for summary in global_summary.values():
        totals.update(summary["status_counts"])
    final = {
        "splits": global_summary,
        "totals": dict(totals),
        "all_accounting_closed": all(
            summary["accounting_closed"] for summary in global_summary.values()
        ),
        "scene_filter": sorted(requested_scenes) if requested_scenes else None,
        "split_filter": sorted(requested_splits) if requested_splits else None,
        "index_shard": {"count": args.index_shard_count, "id": args.index_shard_id},
        "donor_ledger": str(args.donor_ledger) if args.donor_ledger else None,
        "source_index_selection": str(args.source_index_selection) if args.source_index_selection else None,
        "collision_height_bounds": [0.3, 2.5],
        "search_algorithm": "bounded_best_first_not_shortest",
    }
    atomic_json(args.output_dir / "summary.json", final)
    print(json.dumps({"totals": final["totals"], "all_accounting_closed": final["all_accounting_closed"]}, indent=2))


if __name__ == "__main__":
    main()
