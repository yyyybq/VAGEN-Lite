#!/usr/bin/env python3
"""Paired fixed-manifest comparison for the controlled U1 Phase 4B run."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "exps/vagen_active_spatial/protocol_only_prepost"
RATE_FIELDS = (
    "strict_action_tag_rate",
    "tool_call_rate",
    "fallback_parse_rate",
    "invalid_or_empty_rate",
    "unknown_action_rate",
    "multiple_action_rate",
    "action_executable_rate",
    "env_exception_rate",
    "success_rate",
)
DISPLAY_FIELDS = (
    "strict_action_tag_rate",
    "tool_call_rate",
    "fallback_parse_rate",
    "invalid_or_empty_rate",
    "unknown_action_rate",
    "action_executable_rate",
    "success_rate",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def metric(row: dict[str, Any], name: str) -> float:
    parser = row.get("parser") or {}
    flags = row.get("flags") or {}
    fmt = str(row.get("format_class"))
    mapping: dict[str, Callable[[], bool]] = {
        "strict_action_tag_rate": lambda: fmt == "action_only",
        "tool_call_rate": lambda: "tool_call" in fmt,
        "fallback_parse_rate": lambda: bool(parser.get("fallback_parse")),
        "invalid_or_empty_rate": lambda: bool(flags.get("invalid_or_empty")),
        "unknown_action_rate": lambda: bool(parser.get("unknown_action_name")),
        "multiple_action_rate": lambda: bool(parser.get("multiple_action_tag")),
        "action_executable_rate": lambda: bool(parser.get("action_executable")),
        "env_exception_rate": lambda: bool(flags.get("env_exception")),
        "success_rate": lambda: bool(flags.get("success")),
    }
    return float(mapping[name]())


def percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("empty percentile input")
    position = (len(sorted_values) - 1) * q
    lo = int(position)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = position - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def paired_bootstrap(
    base: list[dict[str, Any]], trained: list[dict[str, Any]], name: str, *, reps: int, seed: int
) -> dict[str, float]:
    deltas = [metric(post, name) - metric(pre, name) for pre, post in zip(base, trained, strict=True)]
    observed = sum(deltas) / len(deltas)
    rng = random.Random(seed)
    boots = []
    n = len(deltas)
    for _ in range(reps):
        boots.append(sum(deltas[rng.randrange(n)] for _ in range(n)) / n)
    boots.sort()
    return {
        "base_rate": sum(metric(row, name) for row in base) / n,
        "trained_rate": sum(metric(row, name) for row in trained) / n,
        "paired_delta": observed,
        "ci95_low": percentile(boots, 0.025),
        "ci95_high": percentile(boots, 0.975),
        "bootstrap_repetitions": reps,
    }


def paired_rows(base_path: Path, post_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    base = read_jsonl(base_path)
    post = read_jsonl(post_path)
    base_ids = [row["sample_id"] for row in base]
    post_ids = [row["sample_id"] for row in post]
    if len(base) != 64 or post_ids != base_ids:
        raise RuntimeError(
            f"paired manifest mismatch: base_n={len(base)} post_n={len(post)} "
            f"same_order={post_ids == base_ids}"
        )
    return base, post


def task_rates(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tasks = sorted({str(row["task_type"]) for row in rows})
    result = {}
    for task in tasks:
        selected = [row for row in rows if str(row["task_type"]) == task]
        result[task] = {
            "n": len(selected),
            **{name: sum(metric(row, name) for row in selected) / len(selected) for name in DISPLAY_FIELDS},
        }
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--steps", type=int, nargs="+", default=[8, 16, 24, 32])
    ap.add_argument("--bootstrap-reps", type=int, default=20000)
    args = ap.parse_args()

    base_report = read_json(args.root / "base_eval/aggregate_report.json")
    manifest_hash = base_report["manifest_sha256"]
    base_raw = args.root / "base_eval/raw_rollouts.jsonl"
    base_rows = read_jsonl(base_raw)
    comparisons = []
    table = [{"checkpoint": "base", **{name: base_report["metrics"][name] for name in DISPLAY_FIELDS}}]

    for step in args.steps:
        eval_dir = args.root / f"eval_step_{step}"
        report = read_json(eval_dir / "aggregate_report.json")
        if report["manifest_sha256"] != manifest_hash:
            raise RuntimeError(f"step {step} manifest hash differs from base")
        pre, post = paired_rows(base_raw, eval_dir / "raw_rollouts.jsonl")
        paired = {
            name: paired_bootstrap(pre, post, name, reps=args.bootstrap_reps, seed=20260829 + step * 100 + i)
            for i, name in enumerate(RATE_FIELDS)
        }
        gate_checks = {
            "strict_gain_at_least_15pp": paired["strict_action_tag_rate"]["paired_delta"] >= 0.15,
            "strict_ci_supports_improvement": paired["strict_action_tag_rate"]["ci95_low"] > 0.0,
            "tool_call_drop_at_least_15pp": paired["tool_call_rate"]["paired_delta"] <= -0.15,
            "tool_call_ci_supports_improvement": paired["tool_call_rate"]["ci95_high"] < 0.0,
            "fallback_decreased": paired["fallback_parse_rate"]["paired_delta"] < 0.0,
            "invalid_not_worse_over_5pp": paired["invalid_or_empty_rate"]["paired_delta"] <= 0.05,
            "unknown_not_worse": paired["unknown_action_rate"]["paired_delta"] <= 0.0,
            "multiple_not_worse": paired["multiple_action_rate"]["paired_delta"] <= 0.0,
            "executable_not_lower": paired["action_executable_rate"]["paired_delta"] >= 0.0,
            "env_exception_near_zero": paired["env_exception_rate"]["trained_rate"] <= 0.01,
            "success_not_worse_over_5pp": paired["success_rate"]["paired_delta"] >= -0.05,
        }
        comparisons.append(
            {
                "checkpoint": f"step_{step}",
                "model_identity": report["model_identity"],
                "paired_bootstrap": paired,
                "per_task": task_rates(post),
                "statistical_protocol_gate": "PASS" if all(gate_checks.values()) else "FAIL",
                "gate_checks": gate_checks,
            }
        )
        table.append({"checkpoint": f"step {step}", **{name: report["metrics"][name] for name in DISPLAY_FIELDS}})

    passing = [entry for entry in comparisons if entry["statistical_protocol_gate"] == "PASS"]
    best = max(
        comparisons,
        key=lambda entry: (
            entry["paired_bootstrap"]["strict_action_tag_rate"]["trained_rate"],
            -entry["paired_bootstrap"]["tool_call_rate"]["trained_rate"],
        ),
    )
    output = {
        "status": "PASS" if passing else "FAIL",
        "scope": "paired protocol statistics only; training-health and checkpoint-reload gates are separate",
        "manifest_sha256": manifest_hash,
        "n": 64,
        "bootstrap_method": "paired nonparametric bootstrap over fixed sample IDs",
        "bootstrap_repetitions": args.bootstrap_reps,
        "base_model_identity": base_report["model_identity"],
        "best_format_checkpoint": best["checkpoint"],
        "passing_checkpoints": [entry["checkpoint"] for entry in passing],
        "table": table,
        "comparisons": comparisons,
    }
    out_path = args.root / "prepost_comparison.json"
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0 if passing else 2


if __name__ == "__main__":
    raise SystemExit(main())
