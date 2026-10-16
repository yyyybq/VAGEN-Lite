#!/usr/bin/env python3
"""Summarize independent local-action decisions and frozen no-inference baselines."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ACTIONS = (
    "move_forward", "move_backward", "move_left", "move_right",
    "turn_left", "turn_right",
)
VERSION = "r1_independent_local_action_summary_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open() if line.strip()]


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def load_states(path: Path, section: str) -> dict[int, dict[str, Any]]:
    payload = json.loads(path.read_text())
    data = payload[section]
    return {int(key): value for key, value in data.items()}


def baseline(audits: list[dict[str, Any]]) -> dict[str, Any]:
    def one(rows: list[dict[str, Any]]) -> dict[str, Any]:
        denominator = len(rows)
        fixed = {
            action: sum(action in row["optimal_first_actions_audit_only"] for row in rows)
            for action in ACTIONS
        }
        membership = Counter()
        fractional = Counter()
        for row in rows:
            optimal = row["optimal_first_actions_audit_only"]
            for action in optimal:
                membership[action] += 1
                fractional[action] += 1.0 / len(optimal)
        expected_hits = sum(len(row["optimal_first_actions_audit_only"]) / 6.0 for row in rows)
        return {
            "states": denominator,
            "uniform_six_action_expected_hits": expected_hits,
            "uniform_six_action_expected_hit_rate": expected_hits / denominator if denominator else None,
            "fixed_action_hits": fixed,
            "fixed_action_hit_rates": {
                key: value / denominator if denominator else None for key, value in fixed.items()
            },
            "oracle_optimal_membership_counts": dict(sorted(membership.items())),
            "oracle_optimal_fractional_distribution": dict(sorted(fractional.items())),
        }
    output = {"overall": one(audits)}
    for distance in (1, 2):
        output[f"d{distance}"] = one([row for row in audits if row["exact_distance"] == distance])
    return output


def delta(after: Any, before: Any) -> float | None:
    if after is None or before is None:
        return None
    return float(after) - float(before)


def projection_change(row: dict[str, Any]) -> dict[str, Any]:
    before_projection = row.get("projection_before") or {}
    after_projection = row.get("projection_after") or {}
    before_canonical = row.get("canonical_before") or {}
    after_canonical = row.get("canonical_after") or {}
    before_gates = before_canonical.get("gates") or {}
    after_gates = after_canonical.get("gates") or {}
    before_inside = before_projection.get("inside_frame_fraction_min")
    after_inside = after_projection.get("inside_frame_fraction_min")
    return {
        "relation_before": before_projection.get("relation_satisfied", before_gates.get("relation")),
        "relation_after": after_projection.get("relation_satisfied", after_gates.get("relation")),
        "relation_margin_before": before_projection.get("relation_margin_px", before_canonical.get("relation_margin_px")),
        "relation_margin_after": after_projection.get("relation_margin_px", after_canonical.get("relation_margin_px")),
        "relation_margin_delta": delta(
            after_projection.get("relation_margin_px", after_canonical.get("relation_margin_px")),
            before_projection.get("relation_margin_px", before_canonical.get("relation_margin_px")),
        ),
        "inside_gate_before": before_gates.get("inside_frame"),
        "inside_gate_after": after_gates.get("inside_frame"),
        "inside_fraction_min_before": before_inside,
        "inside_fraction_min_after": after_inside,
        "inside_fraction_min_delta": delta(after_inside, before_inside),
        "collision": bool(row.get("collision_attempts", 0)),
    }


def model_stats(rows: dict[int, dict[str, Any]], audits: list[dict[str, Any]]) -> dict[str, Any]:
    complete = [rows[index] for index in range(len(audits)) if rows.get(index, {}).get("status") == "complete"]
    def one(selected: list[dict[str, Any]]) -> dict[str, Any]:
        actions = Counter(str(row.get("first_parsed_action")) for row in selected)
        return {
            "states": len(selected),
            "hits": sum(bool(row.get("optimal_first_action_hit")) for row in selected),
            "hit_rate": (
                sum(bool(row.get("optimal_first_action_hit")) for row in selected) / len(selected)
                if selected else None
            ),
            "invalid_or_empty": sum(row.get("first_parsed_action") not in ACTIONS for row in selected),
            "collisions": sum(bool(row.get("collision_attempts", 0)) for row in selected),
            "action_distribution": dict(sorted(actions.items())),
            "relation_improved": sum(
                (projection_change(row)["relation_margin_delta"] or 0.0) > 0 for row in selected
            ),
            "relation_worsened": sum(
                (projection_change(row)["relation_margin_delta"] or 0.0) < 0 for row in selected
            ),
            "inside_gate_lost": sum(
                projection_change(row)["inside_gate_before"] is True
                and projection_change(row)["inside_gate_after"] is False for row in selected
            ),
            "inside_gate_gained": sum(
                projection_change(row)["inside_gate_before"] is False
                and projection_change(row)["inside_gate_after"] is True for row in selected
            ),
        }
    output = {"overall": one(complete)}
    for distance in (1, 2):
        output[f"d{distance}"] = one([
            row for row in complete if int(row["exact_distance_audit_only"]) == distance
        ])
    scenes = sorted({str(row["scene_id"]) for row in complete})
    output["by_scene"] = {
        scene: one([row for row in complete if str(row["scene_id"]) == scene]) for scene in scenes
    }
    parents: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in complete:
        parents[str(row["source_key"])].append(row)
    output["by_parent_source"] = {
        key: one(value) for key, value in sorted(parents.items())
    }
    output["infrastructure_errors"] = len(audits) - len(complete)
    return output


def paired(
    audits: list[dict[str, Any]], first: dict[int, dict[str, Any]], second: dict[int, dict[str, Any]],
    first_name: str, second_name: str,
) -> list[dict[str, Any]]:
    output = []
    for index, audit in enumerate(audits):
        a = first.get(index, {})
        b = second.get(index, {})
        hit_a = bool(a.get("optimal_first_action_hit")) if a.get("status") == "complete" else None
        hit_b = bool(b.get("optimal_first_action_hit")) if b.get("status") == "complete" else None
        if hit_a is None or hit_b is None:
            outcome = "infrastructure_incomplete"
        elif hit_a and hit_b:
            outcome = "both_hit"
        elif hit_a:
            outcome = f"{first_name}_only"
        elif hit_b:
            outcome = f"{second_name}_only"
        else:
            outcome = "both_miss"
        output.append({
            "local_state_index": index, "source_key": audit["source_key"],
            "scene_id": audit["scene_id"], "exact_distance": audit["exact_distance"],
            "optimal_first_actions": audit["optimal_first_actions_audit_only"],
            first_name: {
                "action": a.get("first_parsed_action"), "hit": hit_a,
                "change": projection_change(a) if a.get("status") == "complete" else None,
            },
            second_name: {
                "action": b.get("first_parsed_action"), "hit": hit_b,
                "change": projection_change(b) if b.get("status") == "complete" else None,
            },
            "outcome": outcome,
        })
    return output


def pair_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def one(values: list[dict[str, Any]]) -> dict[str, Any]:
        return {"states": len(values), "outcomes": dict(sorted(Counter(row["outcome"] for row in values).items()))}
    result = {"overall": one(rows)}
    for distance in (1, 2):
        result[f"d{distance}"] = one([row for row in rows if row["exact_distance"] == distance])
    parents: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        parents[row["source_key"]].append(row)
    result["parent_source_grouped"] = {
        key: one(value) for key, value in sorted(parents.items())
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--pretrained-ledger", type=Path, required=True)
    parser.add_argument("--v46-ledger", type=Path, required=True)
    parser.add_argument("--dev-audit", type=Path, required=True)
    parser.add_argument("--dev-pretrained-ledger", type=Path, required=True)
    parser.add_argument("--dev-v46-ledger", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    audit_path = Path(protocol["audit_manifest"]["path"])
    if sha256(audit_path) != protocol["audit_manifest"]["sha256"]:
        raise RuntimeError("independent audit SHA256 mismatch")
    independent = jsonl(audit_path)
    dev = jsonl(args.dev_audit)
    independent_pretrained = load_states(args.pretrained_ledger, "states")
    independent_v46 = load_states(args.v46_ledger, "states")
    dev_pretrained = load_states(args.dev_pretrained_ledger, "local")
    dev_v46 = load_states(args.dev_v46_ledger, "local")
    independent_pair = paired(independent, independent_pretrained, independent_v46, "pretrained", "v46_step250")
    dev_pair = paired(dev, dev_pretrained, dev_v46, "pretrained", "v46_step250")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, values in (("independent_paired.jsonl", independent_pair), ("development_paired.jsonl", dev_pair)):
        path = args.output_dir / name
        path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in values))
    analysis_plan = protocol["analysis_plan_frozen_before_policy_inference"]
    independent_stats = {
        "no_inference_baselines": baseline(independent),
        "pretrained": model_stats(independent_pretrained, independent),
        "v46_step250": model_stats(independent_v46, independent),
        "paired": pair_summary(independent_pair),
    }
    development_stats = {
        "no_inference_baselines": baseline(dev),
        "pretrained": model_stats(dev_pretrained, dev),
        "v46_step250": model_stats(dev_v46, dev),
        "paired": pair_summary(dev_pair),
        "inside_fraction_limitation": (
            "The frozen development runner saved the canonical inside-frame gate but not "
            "numeric inside-frame fractions; independent rows save both."
        ),
    }
    maximum = analysis_plan["local_low_hit_replication"]["maximum_hit_rate_per_model_per_distance"]
    criterion = {
        "parent_sources": len({row["source_key"] for row in independent}),
        "r1_unseen_scenes": len({row["scene_id"] for row in independent}),
        "all_model_distance_hit_rates_at_or_below_frozen_maximum": all(
            independent_stats[model][f"d{distance}"]["hit_rate"] is not None
            and independent_stats[model][f"d{distance}"]["hit_rate"] <= maximum
            for model in ("pretrained", "v46_step250") for distance in (1, 2)
        ),
        "each_model_overall_below_uniform_expectation": all(
            independent_stats[model]["overall"]["hit_rate"]
            < independent_stats["no_inference_baselines"]["overall"]["uniform_six_action_expected_hit_rate"]
            for model in ("pretrained", "v46_step250")
        ),
        "zero_infrastructure_errors": all(
            independent_stats[model]["infrastructure_errors"] == 0
            for model in ("pretrained", "v46_step250")
        ),
    }
    criterion["low_hit_replication_pass"] = (
        criterion["parent_sources"] >= analysis_plan["local_low_hit_replication"]["minimum_parent_sources"]
        and criterion["r1_unseen_scenes"] >= analysis_plan["local_low_hit_replication"]["minimum_r1_unseen_scenes"]
        and criterion["all_model_distance_hit_rates_at_or_below_frozen_maximum"]
        and criterion["each_model_overall_below_uniform_expectation"]
        and criterion["zero_infrastructure_errors"]
    )
    summary = {
        "version": VERSION, "frozen_analysis_plan": analysis_plan,
        "independent": independent_stats, "development": development_stats,
        "predeclared_replication_criterion": criterion,
        "provenance": {
            "protocol": {"path": str(args.protocol), "sha256": sha256(args.protocol)},
            "inputs": {
                str(path): sha256(path) for path in (
                    args.pretrained_ledger, args.v46_ledger, args.dev_audit,
                    args.dev_pretrained_ledger, args.dev_v46_ledger,
                )
            },
        },
    }
    atomic_json(args.output_dir / "summary.json", summary)
    print(json.dumps(criterion, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
