#!/usr/bin/env python3
"""Diagnostic Medium selector that preserves v2 budgets but collects only 4--6 step initials.

This is deliberately a separate version: v2 remains the frozen baseline.  The
band avoids spending the fixed candidate cap on 1--3 step reverse states that
cannot meet the requested Medium certificate upper bound.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import r1_projective_medium_selector_v2 as v2
from r1_projective_difficulty_conditioned import collect_reverse_candidates

VERSION = "projective_difficulty_conditioned_medium_selector_v3_depth_banded"


def one(item, rec, constraints, cfg):
    original = v2.collect_reverse_candidates

    def depth_banded(*args, **kwargs):
        kwargs["max_steps"] = 6
        kwargs["min_candidate_steps"] = 4
        return collect_reverse_candidates(*args, **kwargs)

    v2.collect_reverse_candidates = depth_banded
    try:
        result = v2.one(item, rec, constraints, cfg)
    finally:
        v2.collect_reverse_candidates = original
    result["version"] = VERSION
    result["candidate_collection"] = {
        "reverse_max_steps": 6,
        "min_candidate_steps": 4,
        "comparison_budget_change": "none",
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--gs-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed-cap", type=int, default=12)
    parser.add_argument("--per-seed-expansions", type=int, default=512)
    parser.add_argument("--candidate-cap-per-seed", type=int, default=48)
    parser.add_argument("--lower-expansions", type=int, default=100000)
    parser.add_argument("--source-keys", help="comma-separated split:index keys")
    args = parser.parse_args()
    selection = json.loads(args.selection.read_text())
    raw_sources = json.loads(args.sources.read_text())
    source_map = raw_sources.get("sources", raw_sources)
    sources = {key: v2.read_jsonl(Path(value["path"] if isinstance(value, dict) else value))
               for key, value in source_map.items()}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg = type("Cfg", (), {
        "seed_cap": args.seed_cap,
        "per_seed_expansions": args.per_seed_expansions,
        "candidate_cap": args.candidate_cap_per_seed,
        "lower_expansions": args.lower_expansions,
    })()
    constraints = v2.SceneConstraints(args.gs_root)
    requested = {(value.rsplit(":", 1)[0], int(value.rsplit(":", 1)[1]))
                 for value in (args.source_keys or "").split(",") if value}
    rows = []
    for rec in selection["records"]:
        if requested and (rec["split"], int(rec["source_row_index"])) not in requested:
            continue
        try:
            result = one(sources[rec["split"]][int(rec["source_row_index"])], rec, constraints, cfg)
        except Exception as error:
            result = {"version": VERSION, "split": rec["split"],
                      "source_row_index": rec["source_row_index"], "scene_id": rec["scene_id"],
                      "requested_bucket": "medium", "status": "implementation_error", "error": repr(error)}
        rows.append(result)
        v2.write_json(args.output_dir / "checkpoint.json", {"version": VERSION, "results": rows})
        print(json.dumps({key: result.get(key) for key in
                          ("split", "source_row_index", "status", "eligible_medium_candidates", "elapsed_seconds")}),
              flush=True)
    v2.write_json(args.output_dir / "selector_results.json", {
        "version": VERSION, "selection": str(args.selection), "same_pair_only": True,
        "budgets": {"seed_cap": args.seed_cap, "per_seed_expansions": args.per_seed_expansions,
                    "candidate_cap_per_seed": args.candidate_cap_per_seed,
                    "lower_expansions": args.lower_expansions},
        "candidate_collection": {"reverse_max_steps": 6, "min_candidate_steps": 4,
                                 "comparison_budget_change": "none"},
        "results": rows, "status_counts": dict(Counter(row["status"] for row in rows)),
    })


if __name__ == "__main__":
    main()
