#!/usr/bin/env python3
"""Compare source and repaired difficulty for accepted R1 canary rows."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r1_canonical_tasks import canonical_fov, canonical_projective, score_observation  # noqa: E402
from r1_full_regeneration import atomic_json, read_jsonl  # noqa: E402
from r1_real_render_validation import pose_from_item_target  # noqa: E402


ACCEPTED = frozenset(("strict_same_pair_repair", "count_matched_replacement"))


def describe(values: list[float]) -> dict[str, Any]:
    finite = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not finite:
        return {"count": 0, "min": None, "p10": None, "median": None, "mean": None, "p90": None, "max": None}

    def percentile(fraction: float) -> float:
        position = fraction * (len(finite) - 1)
        low = int(math.floor(position))
        high = int(math.ceil(position))
        if low == high:
            return finite[low]
        return finite[low] * (high - position) + finite[high] * (position - low)

    return {
        "count": len(finite),
        "min": finite[0],
        "p10": percentile(0.10),
        "median": statistics.median(finite),
        "mean": statistics.fmean(finite),
        "p90": percentile(0.90),
        "max": finite[-1],
    }


def yaw_delta_degrees(initial: np.ndarray, target: np.ndarray) -> float:
    first = math.atan2(float(initial[1, 2]), float(initial[0, 2]))
    second = math.atan2(float(target[1, 2]), float(target[0, 2]))
    return math.degrees(abs((second - first + math.pi) % (2 * math.pi) - math.pi))


def observation(item: dict[str, Any], pose: np.ndarray) -> tuple[dict[str, Any], list[float]]:
    result = score_observation(item, pose)
    metric = canonical_fov(result) if item.get("task_type") == "fov_inclusion" else canonical_projective(result)
    areas = [
        float(obj.get("area_ratio", 0.0) or 0.0)
        for obj in result["visual_metrics"].get("objects", [])
    ]
    return metric, areas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--aggregate-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    raw_sources = json.loads(args.sources.read_text())
    sources = {
        split: (path if (path := Path(value)).is_absolute() else Path.cwd() / path)
        for split, value in raw_sources.items()
    }
    values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    category_counts: dict[str, Counter[str]] = defaultdict(Counter)
    scene_counts: dict[str, Counter[str]] = defaultdict(Counter)
    status_counts: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    metric_failures: list[dict[str, Any]] = []
    bucket_matches: dict[str, Counter[str]] = defaultdict(Counter)
    paired_deltas: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))

    for split, source_path in sources.items():
        source_rows = read_jsonl(source_path)
        split_dir = args.aggregate_dir / split
        repaired_rows = read_jsonl(split_dir / "trainable.jsonl")
        repaired_by_index = {
            int(row["repair_lineage"]["source_row_index"]): row for row in repaired_rows
        }
        mappings = {
            int(row["source_row_index"]): row for row in read_jsonl(split_dir / "mapping.jsonl")
        }
        for accounting in read_jsonl(split_dir / "accounting.jsonl"):
            status = str(accounting["status"])
            status_counts[status] += 1
            if status not in ACCEPTED:
                continue
            index = int(accounting["source_row_index"])
            old = source_rows[index]
            new = repaired_by_index[index]
            mapping = mappings[index]
            task = str(old["task_type"])
            group = "fov" if task == "fov_inclusion" else "projective"
            task_counts[group] += 1
            category_counts[f"source_{group}"][str(old.get("object_label"))] += 1
            category_counts[f"repaired_{group}"][str(new.get("object_label"))] += 1
            scene_counts[group][str(new.get("scene_id"))] += 1

            try:
                old_initial = np.asarray(old["init_camera"]["extrinsics"], dtype=float)
                old_target = pose_from_item_target(old)
                new_initial = np.asarray(new["init_camera"]["extrinsics"], dtype=float)
                new_target = pose_from_item_target(new)
                old_initial_metric, old_initial_areas = observation(old, old_initial)
                old_target_metric, old_target_areas = observation(old, old_target)
                new_initial_metric, new_initial_areas = observation(new, new_initial)
                new_target_metric, new_target_areas = observation(new, new_target)
            except Exception as error:
                metric_failures.append({"split": split, "source_row_index": index, "error": repr(error)})
                continue

            bucket = values[group]
            bucket["old_translation"].append(float(np.linalg.norm(old_target[:3, 3] - old_initial[:3, 3])))
            bucket["new_translation"].append(float(np.linalg.norm(new_target[:3, 3] - new_initial[:3, 3])))
            bucket["old_yaw_delta_degrees"].append(yaw_delta_degrees(old_initial, old_target))
            bucket["new_yaw_delta_degrees"].append(yaw_delta_degrees(new_initial, new_target))
            bucket["old_initial_score"].append(float(old_initial_metric["score"]))
            bucket["new_initial_score"].append(float(new_initial_metric["score"]))
            bucket["old_target_score"].append(float(old_target_metric["score"]))
            bucket["new_target_score"].append(float(new_target_metric["score"]))
            bucket["old_initial_bbox_area_ratio_min"].append(min(old_initial_areas) if old_initial_areas else 0.0)
            bucket["new_initial_bbox_area_ratio_min"].append(min(new_initial_areas) if new_initial_areas else 0.0)
            bucket["old_target_bbox_area_ratio_min"].append(min(old_target_areas) if old_target_areas else 0.0)
            bucket["new_target_bbox_area_ratio_min"].append(min(new_target_areas) if new_target_areas else 0.0)
            bucket["planner_path_upper_bound"].append(float(mapping["found_path_length_upper_bound"]))
            retry = mapping.get("retry", {})
            bucket["repair_attempt_count"].append(
                float(retry.get("joint_states_evaluated", retry.get("attempt_count", 0)))
            )
            if group == "projective":
                old_profile = mapping.get("source_difficulty") or {}
                new_profile = mapping.get("repaired_difficulty") or {}
                for name in (
                    "translation_m", "yaw_deg", "bbox_area_ratio",
                    "relation_margin_px", "planner_step_proxy",
                ):
                    if name in old_profile and name in new_profile:
                        paired_deltas[group][name].append(
                            float(new_profile[name]) - float(old_profile[name])
                        )
                        matched = old_profile.get("buckets", {}).get(name) == new_profile.get("buckets", {}).get(name)
                        bucket_matches[name]["matched" if matched else "mismatched"] += 1
            if group == "projective":
                for prefix, metric in (
                    ("old_initial", old_initial_metric), ("new_initial", new_initial_metric),
                    ("old_target", old_target_metric), ("new_target", new_target_metric),
                ):
                    margin = metric.get("relation_margin_px")
                    if margin is not None:
                        bucket[f"{prefix}_relation_margin_px"].append(float(margin))
            else:
                for prefix, metric in (
                    ("old_initial", old_initial_metric), ("new_initial", new_initial_metric),
                    ("old_target", old_target_metric), ("new_target", new_target_metric),
                ):
                    bucket[f"{prefix}_inside_frame_fraction_min"].append(
                        float(metric.get("inside_frame_fraction_min", 0.0))
                    )

    audit = {
        "accounting_status_counts": dict(status_counts),
        "accepted_task_counts": dict(task_counts),
        "replacement_rate_of_source_rows": status_counts["count_matched_replacement"] / max(sum(status_counts.values()), 1),
        "replacement_rate_of_accepted_rows": status_counts["count_matched_replacement"] / max(sum(task_counts.values()), 1),
        "difficulty": {
            group: {metric: describe(series) for metric, series in sorted(metrics.items())}
            for group, metrics in sorted(values.items())
        },
        "paired_difficulty_delta_new_minus_old": {
            group: {metric: describe(series) for metric, series in sorted(metrics.items())}
            for group, metrics in sorted(paired_deltas.items())
        },
        "projective_bucket_match_counts": {
            name: dict(counts) for name, counts in sorted(bucket_matches.items())
        },
        "scene_distribution": {group: dict(counts) for group, counts in sorted(scene_counts.items())},
        "category_distribution": {group: dict(counts) for group, counts in sorted(category_counts.items())},
        "metric_failures": metric_failures,
        "interpretation": (
            "Accepted-row distributions only. They cannot establish whole-source difficulty parity because "
            "hard-failure and unverified rows are excluded; replacement rows change the object pair by design."
        ),
    }
    atomic_json(args.output, audit)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
