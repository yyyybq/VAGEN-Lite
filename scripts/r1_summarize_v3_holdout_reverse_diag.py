#!/usr/bin/env python3
"""Close accounting for the independent v3 A/B and reverse-cap diagnostic."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


VERSION = "r1_v3_holdout_reverse_diag_summary_v1"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open() if line.strip()]


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def key(row: dict[str, Any]) -> str:
    return f"{row['split']}:{int(row['source_row_index'])}"


def task_key(task_id: str, index: int | None = None) -> str:
    marker = "projective_path_first_proto_"
    core = task_id[len(marker):] if task_id.startswith(marker) else task_id
    if core.endswith("_medium"):
        core = core[:-len("_medium")]
    split, parsed = core.rsplit("_", 1)
    return f"{split}:{int(parsed if parsed.isdigit() else index)}"


def evidence_passes(root: Path) -> tuple[dict[str, dict], dict[str, dict]]:
    replay = json.loads((root / "independent_validation/runtime_replay.json").read_text())
    runtime = {task_key(row["task_id"], row.get("source_row_index")): row for row in replay["results"]}
    rgb_rows = read_jsonl(root / "official_rgb/observability_manifest.jsonl")
    rgb = {task_key(row["task_id"], row.get("source_row_index")): row for row in rgb_rows}
    return runtime, rgb


def old_path_first(old_root: Path) -> dict[str, Any]:
    reach = read_jsonl(old_root / "certificates/reachability_manifest.jsonl")
    obs = read_jsonl(old_root / "official_observability_v1/observability_manifest.jsonl")
    obs_pass = {task_key(row["task_id"], row.get("source_row_index")) for row in obs if row.get("passed")}
    rows = []
    for row in reach:
        source = task_key(row["task_id"], row.get("source_row_index"))
        rows.append({"source_key": source, "first_success_step": row.get("first_success_step", row.get("steps")),
                     "rgb_pass": source in obs_pass})
    generated_steps = Counter(str(row["first_success_step"]) for row in rows)
    accepted = [row for row in rows if row["rgb_pass"]]
    accepted_steps = Counter(str(row["first_success_step"]) for row in accepted)
    return {
        "certificate_rows": len(rows), "rgb_pass_rows": len(accepted),
        "certificate_first_success_distribution": dict(sorted(generated_steps.items(), key=lambda item: int(item[0]))),
        "rgb_pass_first_success_distribution": dict(sorted(accepted_steps.items(), key=lambda item: int(item[0]))),
        "verified_one_step_certificate_rows": generated_steps["1"],
        "verified_one_step_rgb_pass_rows": accepted_steps["1"],
        "claim_183": "unknown_unsupported_by_formal_first_success_artifacts",
        "accepted_rows": accepted,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--old-path-first-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output_root
    inventory = json.loads((root / "verified_medium75_inventory/verified_medium_inventory.json").read_text())
    holdout = json.loads((root / "holdout128_frozen_selection.json").read_text())
    v2 = json.loads((root / "holdout_v2_screen/selector_results.json").read_text())
    ab = json.loads((root / "holdout_v3_ab_frozen_selection.json").read_text())
    v3 = json.loads((root / "holdout_v3_ab/selector_results.json").read_text())
    reverse = json.loads((root / "reverse_cap_diagnostic/reverse_cap_diagnostic.json").read_text())
    v2_runtime, v2_rgb = evidence_passes(root / "holdout_v2_screen")
    v3_runtime, v3_rgb = evidence_passes(root / "holdout_v3_ab")
    v2_rows = {key(row): row for row in v2["results"]}
    v3_rows = {key(row): row for row in v3["results"]}
    ab_kind = {key(row): row["ab_kind"] for row in ab["records"]}
    comparisons = []
    for source in sorted(v3_rows):
        old, new = v2_rows[source], v3_rows[source]
        runtime = v3_runtime.get(source)
        rgb = v3_rgb.get(source)
        final = bool(new.get("status") == "difficulty_certified_candidate"
                     and runtime and runtime.get("status") == "pass" and rgb and rgb.get("passed"))
        comparisons.append({
            "source_key": source, "scene_id": new["scene_id"], "ab_kind": ab_kind[source],
            "v2_status": old["status"], "v3_status": new["status"],
            "v2_elapsed_seconds": old.get("elapsed_seconds"), "v3_elapsed_seconds": new.get("elapsed_seconds"),
            "v3_first_success_step": new.get("certificate_upper_bound"),
            "v3_runtime": runtime.get("status") if runtime else None,
            "v3_rgb": rgb.get("passed") if rgb else None,
            "v3_rgb_reasons": rgb.get("reasons") if rgb else None,
            "final_v3_pass": final,
        })
    shortcut_rows = [row for row in comparisons if row["ab_kind"] == "v2_shortcut_rejected"]
    control_rows = [row for row in comparisons if row["ab_kind"] == "v2_positive_control"]
    recovered = [row for row in shortcut_rows if row["final_v3_pass"]]
    controls_preserved = [row for row in control_rows if row["final_v3_pass"]]
    v2_elapsed_screen = sum(float(row.get("elapsed_seconds") or 0) for row in v2["results"])
    v2_elapsed_ab = sum(float(v2_rows[row["source_key"]].get("elapsed_seconds") or 0) for row in comparisons)
    v3_elapsed_ab = sum(float(row.get("v3_elapsed_seconds") or 0) for row in comparisons)
    old = old_path_first(args.old_path_first_root)
    verified_sources = {row["source_key"] for row in inventory["records"]}
    old_one_step = {row["source_key"] for row in old.pop("accepted_rows") if row["first_success_step"] == 1}
    reverse_failure_rows = [row for row in reverse["results"] if row["kind"] == "reverse_cap"]
    reverse_controls = [row for row in reverse["results"] if row["kind"] == "known_positive_control"]
    summary = {
        "version": VERSION,
        "verified_inventory": {
            **inventory["accounting"],
            "verified_sources_overlapping_old_one_step_rgb_pass": len(verified_sources & old_one_step),
        },
        "historical_one_step_audit": old,
        "independent_screen": {
            "source_count": holdout["record_count"], "scenes": holdout["scenes"],
            "scene_denominators": holdout["counts"]["scene"],
            "v2_status_counts": v2["status_counts"],
            "v2_runtime_pass": sum(row.get("status") == "pass" for row in v2_runtime.values()),
            "v2_rgb_pass": sum(row.get("passed") for row in v2_rgb.values()),
            "v2_rgb_reject": sum(not row.get("passed") for row in v2_rgb.values()),
        },
        "v3_ab": {
            "shortcut_screen_count": ab["shortcut_available_in_128_screen"],
            "shortcut_selected": ab["shortcut_selected"],
            "positive_controls": ab["positive_control_selected"],
            "v3_status_counts": v3["status_counts"],
            "shortcut_final_recovered": len(recovered),
            "shortcut_recovery_scenes": dict(sorted(Counter(row["scene_id"] for row in recovered).items())),
            "positive_controls_final_preserved": len(controls_preserved),
            "positive_controls_generator_refound": sum(row["v3_status"] == "difficulty_certified_candidate" for row in control_rows),
            "comparison_rows": comparisons,
        },
        "reverse_cap": {
            "failure_sample_count": len(reverse_failure_rows), "control_count": len(reverse_controls),
            "primary_class_counts": dict(Counter(row["primary_class"] for row in reverse_failure_rows)),
            "evidence_tag_counts": dict(Counter(tag for row in reverse_failure_rows for tag in row["diagnostic_classes"])),
            "failure_rows": [{
                "source_key": f"{row['split']}:{row['source_row_index']}", "scene_id": row["scene_id"],
                "primary_class": row["primary_class"], "tags": row["diagnostic_classes"],
                "success_seeds": row["success_seed_count"],
                "states_at_depth_4_to_6": row["aggregate"]["states_at_depth_4_to_6"],
                "eligible_initials_at_depth_4_to_6": row["aggregate"]["eligible_initials_at_depth_4_to_6"],
                "cross_seed_duplicate_fraction": row["cross_seed_overlap"]["duplicate_visit_fraction"],
                "elapsed_seconds": row["elapsed_seconds"],
            } for row in reverse_failure_rows],
            "controls": [{"source_key": f"{row['split']}:{row['source_row_index']}",
                          "scene_id": row["scene_id"], "primary_class": row["primary_class"],
                          "eligible_initials_at_depth_4_to_6": row["aggregate"]["eligible_initials_at_depth_4_to_6"]}
                         for row in reverse_controls],
        },
        "cost": {
            "v2_screen_sum_per_source_elapsed_seconds_not_parallel_wall": v2_elapsed_screen,
            "v2_ab_subset_sum_elapsed_seconds": v2_elapsed_ab,
            "v3_ab_subset_sum_elapsed_seconds": v3_elapsed_ab,
            "v3_vs_v2_same_subset_elapsed_ratio": v3_elapsed_ab / v2_elapsed_ab if v2_elapsed_ab else None,
            "sequential_v2_then_v3_sum_elapsed_seconds": v2_elapsed_screen + v3_elapsed_ab,
            "resource_accounting_note": "per-source elapsed sums are not SCO parallel wall time; see job timestamps/environment separately",
        },
        "predeclared_v3_cost_tolerance": holdout["frozen_config"]["cost_tolerance_predeclared"],
        "gate_evidence": {
            "recovery_at_least_three_independent_scenes": len({row["scene_id"] for row in recovered}) >= 3,
            "all_recovered_runtime_rgb_pass": all(row["final_v3_pass"] for row in recovered),
            "all_positive_controls_preserved": len(controls_preserved) == len(control_rows),
            "v3_cost_within_tolerance": (v3_elapsed_ab / v2_elapsed_ab <= holdout["frozen_config"]["cost_tolerance_predeclared"]["v3_generation_elapsed_ratio_vs_v2_max"]
                                         if v2_elapsed_ab else False),
        },
    }
    atomic_json(args.output, summary)
    print(json.dumps({"screen": summary["independent_screen"], "v3_ab": {key: value for key, value in summary["v3_ab"].items() if key != "comparison_rows"},
                      "reverse_classes": summary["reverse_cap"]["primary_class_counts"],
                      "gate_evidence": summary["gate_evidence"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
