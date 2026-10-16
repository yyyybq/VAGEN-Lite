#!/usr/bin/env python3
"""Summarize versioned projective regeneration and verify source preservation."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open() if line.strip()]


def canonical(value: dict) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def describe(values: list[float]) -> dict[str, float | None]:
    return {
        "min": min(values) if values else None,
        "mean": statistics.mean(values) if values else None,
        "median": statistics.median(values) if values else None,
        "max": max(values) if values else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--sources-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    sources = json.loads(args.sources_json.read_text())
    report: dict[str, object] = {"generator_version": "projective_canonical_h1_v3", "splits": {}}
    for name, source_name in sources.items():
        source = Path(source_name)
        output = args.input_dir / f"{name}_projective_canonical_h1_v3.jsonl"
        mapping = args.input_dir / f"{name}_mapping.jsonl"
        failures = args.input_dir / f"{name}_failures.jsonl"
        source_rows, output_rows, mappings, failed = rows(source), rows(output), rows(mapping), rows(failures)
        unchanged_nonprojective = [canonical(a) == canonical(b) for a, b in zip(source_rows, output_rows) if a.get("task_type") != "projective_relations"]
        old_distance, new_distance, old_margin, new_margin, initial_success = [], [], [], [], 0
        for entry in mappings:
            old = entry.get("old_sample_target") or []
            new = entry.get("new_sample_target") or []
            if len(old) >= 2 and len(new) >= 2:
                old_distance.append(float((old[0] ** 2 + old[1] ** 2) ** 0.5))
                new_distance.append(float((new[0] ** 2 + new[1] ** 2) ** 0.5))
            old_metric, new_metric = entry.get("old_metric", {}), entry.get("new_metric", {})
            if old_metric.get("relation_margin_px") is not None:
                old_margin.append(float(old_metric["relation_margin_px"]))
            if new_metric.get("relation_margin_px") is not None:
                new_margin.append(float(new_metric["relation_margin_px"]))
            initial_success += int(bool(entry.get("new_initial_metric", {}).get("success")))
        report["splits"][name] = {
            "source": str(source), "output": str(output),
            "source_sha256": sha256(source), "output_sha256": sha256(output),
            "source_rows": len(source_rows), "output_rows": len(output_rows),
            "projective_rows": sum(r.get("task_type") == "projective_relations" for r in source_rows),
            "repaired": len(mappings), "unresolved_retained": len(failed),
            "new_target_success": sum(bool(m.get("new_metric", {}).get("success")) for m in mappings),
            "new_initial_success": initial_success,
            "nonprojective_semantic_preservation": all(unchanged_nonprojective),
            "old_target_relation_margin_px": describe(old_margin),
            "new_target_relation_margin_px": describe(new_margin),
            "old_sample_target_xy_norm": describe(old_distance),
            "new_sample_target_xy_norm": describe(new_distance),
            "mapping_sha256": sha256(mapping), "failure_sha256": sha256(failures),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
