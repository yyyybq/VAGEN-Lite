#!/usr/bin/env python3
"""Build paired statistics for the two frozen R1 canonical dev evaluations."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


VERSION = "r1_canonical_dev_eval32_paired_summary_v2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def load_ledger(path: Path) -> dict[int, dict[str, Any]]:
    payload = json.loads(path.read_text())
    rows = {int(key): value for key, value in payload["episodes"].items()}
    if set(rows) != set(range(32)):
        raise ValueError(f"ledger is not closed 32/32: {path}, keys={sorted(rows)}")
    return rows


def category(row: dict[str, Any]) -> str:
    labels = []
    for value in row.get("object_pair", []):
        labels.append(str(value).split(":", 1)[-1])
    return "+".join(labels) or "unknown"


def grouped(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    counts: dict[str, Counter] = defaultdict(Counter)
    for row in rows:
        value = category(row) if key == "category" else str(row.get(key, "unknown"))
        counts[value]["n"] += 1
        counts[value]["pretrained_success"] += int(row["pretrained_success"])
        counts[value]["v46_success"] += int(row["v46_success"])
    return {name: dict(value) for name, value in sorted(counts.items())}


def failure_class(row: dict[str, Any]) -> str:
    if row.get("status") != "complete":
        return "infrastructure_error"
    if row.get("success"):
        return "success"
    reason = str(row.get("done_reason") or "")
    if "collision" in reason:
        return "collision_termination"
    if "invalid" in reason:
        return "invalid_action_termination"
    if "low_information" in reason:
        return "low_information_termination"
    if "max_" in reason:
        return "timeout_or_truncation"
    if int(row.get("invalid_turns", 0)):
        return "action_parse_or_selection"
    return "observation_or_action_selection"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pretrained-ledger", type=Path, required=True)
    parser.add_argument("--v46-ledger", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    manifest = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    if len(manifest) != 32:
        raise ValueError("manifest must contain 32 rows")
    pretrained = load_ledger(args.pretrained_ledger)
    v46 = load_ledger(args.v46_ledger)
    paired = []
    for index, audit in enumerate(manifest):
        p, v = pretrained[index], v46[index]
        for name, row in (("pretrained", p), ("v46", v)):
            if row.get("episode_fingerprint") != audit["episode_fingerprint"]:
                raise ValueError(f"{name} fingerprint mismatch at {index}")
            if int(row.get("paired_seed")) != 2026091500 + index * 100:
                raise ValueError(f"{name} seed mismatch at {index}")
        p_initial_input = p["turns"][0]["input"]
        v_initial_input = v["turns"][0]["input"]
        lower_bound = int(audit["difficulty"]["certified_lower_bound"])
        paired.append({
            "eval_index": index,
            "source_key": audit["source_key"],
            "split": audit["split"],
            "scene_id": audit["scene_id"],
            "category": category(audit),
            "object_pair": audit["object_pair"],
            "relation": audit["relation"],
            "difficulty": audit["difficulty"],
            "episode_fingerprint": audit["episode_fingerprint"],
            "paired_seed": p["paired_seed"],
            "initial_image_sha_match": p_initial_input["image_sha256"] == v_initial_input["image_sha256"],
            "initial_prompt_fingerprint_match": (
                p_initial_input["messages_fingerprint"] == v_initial_input["messages_fingerprint"]
            ),
            "pretrained_success": bool(p.get("success")),
            "v46_success": bool(v.get("success")),
            "pretrained_first_success_step": p.get("first_success_step"),
            "v46_first_success_step": v.get("first_success_step"),
            "pretrained_lower_bound_consistent": (
                not p.get("success") or int(p["first_success_step"]) >= lower_bound
            ),
            "v46_lower_bound_consistent": (
                not v.get("success") or int(v["first_success_step"]) >= lower_bound
            ),
            "pretrained_done_reason": p.get("done_reason"),
            "v46_done_reason": v.get("done_reason"),
            "pretrained_collisions": p.get("collision_attempts", 0),
            "v46_collisions": v.get("collision_attempts", 0),
            "pretrained_invalid_turns": p.get("invalid_turns", 0),
            "v46_invalid_turns": v.get("invalid_turns", 0),
            "pretrained_actions": p.get("actions", []),
            "v46_actions": v.get("actions", []),
            "pretrained_failure_class": failure_class(p),
            "v46_failure_class": failure_class(v),
        })
    p_success = sum(row["pretrained_success"] for row in paired)
    v_success = sum(row["v46_success"] for row in paired)
    outcomes = Counter()
    for row in paired:
        outcomes[
            "both_success" if row["pretrained_success"] and row["v46_success"] else
            "pretrained_only" if row["pretrained_success"] else
            "v46_only" if row["v46_success"] else "both_fail"
        ] += 1
    summary = {
        "version": VERSION,
        "episodes_per_model": 32,
        "pretrained": {"success": p_success, "rate": p_success / 32},
        "v46_step250": {"success": v_success, "rate": v_success / 32},
        "paired_outcomes": dict(outcomes),
        "paired_input_checks": {
            "initial_image_sha_match": sum(row["initial_image_sha_match"] for row in paired),
            "initial_prompt_fingerprint_match": sum(row["initial_prompt_fingerprint_match"] for row in paired),
            "pretrained_lower_bound_consistent": sum(row["pretrained_lower_bound_consistent"] for row in paired),
            "v46_lower_bound_consistent": sum(row["v46_lower_bound_consistent"] for row in paired),
        },
        "by_split": grouped(paired, "split"),
        "by_scene": grouped(paired, "scene_id"),
        "by_category": grouped(paired, "category"),
        "failure_classes": {
            "pretrained": dict(Counter(row["pretrained_failure_class"] for row in paired)),
            "v46": dict(Counter(row["v46_failure_class"] for row in paired)),
        },
        "action_distribution": {
            "pretrained": dict(Counter(a for row in paired for a in row["pretrained_actions"])),
            "v46": dict(Counter(a for row in paired for a in row["v46_actions"])),
        },
        "collision_attempts": {
            "pretrained": sum(row["pretrained_collisions"] for row in paired),
            "v46": sum(row["v46_collisions"] for row in paired),
        },
        "invalid_turns": {
            "pretrained": sum(row["pretrained_invalid_turns"] for row in paired),
            "v46": sum(row["v46_invalid_turns"] for row in paired),
        },
        "first_success_step_distribution": {
            "pretrained": dict(Counter(
                str(row["pretrained_first_success_step"]) for row in paired if row["pretrained_success"]
            )),
            "v46": dict(Counter(
                str(row["v46_first_success_step"]) for row in paired if row["v46_success"]
            )),
        },
        "inputs": {
            "manifest": {"path": str(args.manifest.resolve()), "sha256": sha256(args.manifest)},
            "pretrained_ledger": {"path": str(args.pretrained_ledger.resolve()), "sha256": sha256(args.pretrained_ledger)},
            "v46_ledger": {"path": str(args.v46_ledger.resolve()), "sha256": sha256(args.v46_ledger)},
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paired_path = args.output_dir / "paired_results.jsonl"
    paired_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in paired))
    atomic_json(args.output_dir / "paired_summary.json", summary)
    sums = []
    for path in sorted(args.output_dir.iterdir()):
        if path.is_file() and path.name != "SHA256SUMS":
            sums.append(f"{sha256(path)}  {path.name}")
    (args.output_dir / "SHA256SUMS").write_text("\n".join(sums) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
