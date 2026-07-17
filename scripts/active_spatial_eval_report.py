#!/usr/bin/env python3
"""
Build a consolidated report from Active Spatial navigation sweep and EASI probe results.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def load_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    with path.open("r") as f:
        return json.load(f)


def as_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def pct(value: Any, digits: int = 1) -> str:
    f = as_float(value)
    return "—" if f is None else f"{f * 100:.{digits}f}%"


def num(value: Any, digits: int = 3) -> str:
    f = as_float(value)
    return "—" if f is None else f"{f:.{digits}f}"


def key_for(row: Dict[str, str]) -> str:
    return f"{row.get('experiment')}@{row.get('step')}"


def nav_completion(rows: List[Dict[str, str]]) -> str:
    total = len(rows)
    ok = sum(1 for r in rows if r.get("status") == "ok")
    missing = total - ok
    return f"{ok}/{total} complete" + (f", {missing} missing" if missing else "")


def best_rows_by_suite(rows: List[Dict[str, str]]) -> Dict[str, Dict[str, str]]:
    best: Dict[str, Dict[str, str]] = {}
    for row in rows:
        if row.get("status") != "ok":
            continue
        suite = row.get("suite", "")
        score = as_float(row.get("test_success_rate"))
        if score is None:
            continue
        old = best.get(suite)
        old_score = as_float(old.get("test_success_rate")) if old else None
        if old is None or old_score is None or score > old_score:
            best[suite] = row
    return best


def suite_pivot(rows: List[Dict[str, str]]) -> Dict[str, Dict[str, Dict[str, str]]]:
    pivot: Dict[str, Dict[str, Dict[str, str]]] = defaultdict(dict)
    for row in rows:
        pivot[key_for(row)][row.get("suite", "")] = row
    return pivot


def generalization_rows(rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    pivot = suite_pivot(rows)
    out = []
    for ckpt, suites in pivot.items():
        id_row = suites.get("id_test")
        id_score = as_float(id_row.get("test_success_rate")) if id_row else None
        for suite_name in (
            "ood_scene",
            "ood_object",
            "ood_category",
            "ood_template",
            "ood_task_mix",
            "ood_geometry",
        ):
            srow = suites.get(suite_name)
            suite_score = as_float(srow.get("test_success_rate")) if srow else None
            if id_score is None or suite_score is None:
                continue
            out.append({
                "checkpoint": ckpt,
                "suite": suite_name,
                "id_success": id_score,
                "suite_success": suite_score,
                "gap": suite_score - id_score,
            })
    return out


def val_test_gap_rows(rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    out = []
    for row in rows:
        if row.get("status") != "ok":
            continue
        test = as_float(row.get("test_success_rate"))
        val = as_float(row.get("val/overall_success_mean"))
        if test is None or val is None:
            continue
        out.append({
            "checkpoint": key_for(row),
            "suite": row.get("suite"),
            "val": val,
            "test": test,
            "gap": test - val,
        })
    return out


def load_result(row: Dict[str, str]) -> Optional[Dict[str, Any]]:
    path = row.get("result_path")
    if not path:
        return None
    return load_json(Path(path))


def per_task_rows(rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        if row.get("status") != "ok":
            continue
        result = load_result(row)
        if not result:
            continue
        by_task = result.get("by_task_type", {}) or {}
        for task, metrics in by_task.items():
            if not isinstance(metrics, dict):
                continue
            n = metrics.get("num_episodes", 0)
            if not n:
                continue
            out.append({
                "checkpoint": key_for(row),
                "suite": row.get("suite"),
                "task": task,
                "n": n,
                "success": metrics.get("success_rate"),
                "final_score": metrics.get("mean_final_score"),
                "improvement": metrics.get("mean_score_improvement"),
                "spl": metrics.get("spl"),
            })
    return out


def per_category_rows(rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        if row.get("status") != "ok":
            continue
        result = load_result(row)
        if not result:
            continue
        by_cat = result.get("by_category", {}) or {}
        for cat, metrics in by_cat.items():
            if not isinstance(metrics, dict):
                continue
            n = metrics.get("num_episodes", 0)
            if not n:
                continue
            out.append({
                "checkpoint": key_for(row),
                "suite": row.get("suite"),
                "category": cat,
                "n": n,
                "success": metrics.get("success_rate"),
                "final_score": metrics.get("mean_final_score"),
                "improvement": metrics.get("mean_score_improvement"),
            })
    return out


def stability_rows(rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    grouped: Dict[tuple, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("status") == "ok":
            grouped[(row.get("experiment"), row.get("suite"))].append(row)
    out: List[Dict[str, Any]] = []
    for (exp, suite), group in grouped.items():
        sorted_group = sorted(group, key=lambda r: int(r.get("step") or 0))
        if len(sorted_group) < 2:
            continue
        best = max(sorted_group, key=lambda r: as_float(r.get("test_success_rate")) or -1.0)
        latest = sorted_group[-1]
        out.append({
            "experiment": exp,
            "suite": suite,
            "best_step": best.get("step"),
            "best_success": best.get("test_success_rate"),
            "latest_step": latest.get("step"),
            "latest_success": latest.get("test_success_rate"),
            "latest_minus_best": (as_float(latest.get("test_success_rate")) or 0.0)
                                  - (as_float(best.get("test_success_rate")) or 0.0),
        })
    return out


def render_probe_summary(probe: Optional[Dict[str, Any]]) -> List[str]:
    if not probe:
        return ["No EASI probe summary found."]
    benchmarks = probe.get("benchmarks", [])
    results = probe.get("results", {})
    lines = []
    lines.append(f"Benchmarks: {', '.join(benchmarks) if benchmarks else '—'}")
    lines.append("")
    lines.append("| checkpoint | " + " | ".join(benchmarks) + " |")
    lines.append("| --- | " + " | ".join(["---:"] * len(benchmarks)) + " |")
    for ckpt, bench_map in results.items():
        vals = []
        for bench in benchmarks:
            metrics = bench_map.get(bench)
            if not metrics:
                vals.append("—")
                continue
            primary = None
            for key, value in metrics.items():
                if isinstance(value, (float, int)) and "stderr" not in key:
                    primary = float(value)
                    break
            vals.append(pct(primary))
        lines.append(f"| {ckpt} | " + " | ".join(vals) + " |")
    return lines


def build_report(nav_rows: List[Dict[str, str]], probe: Optional[Dict[str, Any]]) -> str:
    lines: List[str] = []
    lines.append("# Active Spatial Full Evaluation Report")
    lines.append("")
    lines.append("## Completion")
    lines.append("")
    lines.append(f"- Navigation sweep: {nav_completion(nav_rows)}")
    if probe:
        n_ckpt = len(probe.get("checkpoints", []))
        n_bench = len(probe.get("benchmarks", []))
        lines.append(f"- EASI probe: {n_ckpt} checkpoints x {n_bench} benchmarks in summary")
    else:
        lines.append("- EASI probe: no summary found")
    lines.append("")

    lines.append("## Best Navigation Checkpoints By Suite")
    lines.append("")
    best = best_rows_by_suite(nav_rows)
    if best:
        lines.append("| suite | checkpoint | success | val_mean | test-val | final_score | spl |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
        for suite, row in sorted(best.items()):
            lines.append(
                f"| {suite} | {key_for(row)} | {pct(row.get('test_success_rate'))} | "
                f"{pct(row.get('val/overall_success_mean'))} | {pct(row.get('test_minus_val_success_mean'))} | "
                f"{num(row.get('test_mean_final_score'))} | {num(row.get('test_spl'))} |"
            )
    else:
        lines.append("No completed navigation results yet.")
    lines.append("")

    lines.append("## Validation-Test Alignment")
    lines.append("")
    gaps = val_test_gap_rows(nav_rows)
    if gaps:
        lines.append("| checkpoint | suite | val_mean | test | test-val |")
        lines.append("| --- | --- | ---: | ---: | ---: |")
        for row in sorted(gaps, key=lambda r: (r["checkpoint"], r["suite"])):
            lines.append(
                f"| {row['checkpoint']} | {row['suite']} | {pct(row['val'])} | "
                f"{pct(row['test'])} | {pct(row['gap'])} |"
            )
    else:
        lines.append("No rows with both validation and test metrics yet.")
    lines.append("")

    lines.append("## Generalization Gaps")
    lines.append("")
    gen_rows = generalization_rows(nav_rows)
    if gen_rows:
        lines.append("| checkpoint | suite | id_test | suite | suite-id |")
        lines.append("| --- | --- | ---: | ---: | ---: |")
        for row in sorted(gen_rows, key=lambda r: (r["checkpoint"], r["suite"])):
            lines.append(
                f"| {row['checkpoint']} | {row['suite']} | {pct(row['id_success'])} | "
                f"{pct(row['suite_success'])} | {pct(row['gap'])} |"
            )
    else:
        lines.append("Need completed `id_test` plus OOD suites to compute gaps.")
    lines.append("")

    lines.append("## Stability Across Checkpoints")
    lines.append("")
    st_rows = stability_rows(nav_rows)
    if st_rows:
        lines.append("| experiment | suite | best_step | best | latest_step | latest | latest-best |")
        lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
        for row in sorted(st_rows, key=lambda r: (r["experiment"], r["suite"])):
            lines.append(
                f"| {row['experiment']} | {row['suite']} | {row['best_step']} | {pct(row['best_success'])} | "
                f"{row['latest_step']} | {pct(row['latest_success'])} | {pct(row['latest_minus_best'])} |"
            )
    else:
        lines.append("Need at least two completed checkpoints per experiment/suite.")
    lines.append("")

    lines.append("## Per-Category Navigation Metrics")
    lines.append("")
    cat_rows = per_category_rows(nav_rows)
    if cat_rows:
        lines.append("| checkpoint | suite | category | n | success | final_score | improvement |")
        lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: |")
        for row in sorted(cat_rows, key=lambda r: (r["checkpoint"], r["suite"], r["category"])):
            lines.append(
                f"| {row['checkpoint']} | {row['suite']} | {row['category']} | {row['n']} | "
                f"{pct(row['success'])} | {num(row['final_score'])} | {num(row['improvement'])} |"
            )
    else:
        lines.append("No completed per-category result files yet.")
    lines.append("")

    lines.append("## Per-Task Navigation Metrics")
    lines.append("")
    task_rows = per_task_rows(nav_rows)
    if task_rows:
        lines.append("| checkpoint | suite | task | n | success | final_score | improvement | spl |")
        lines.append("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |")
        for row in sorted(task_rows, key=lambda r: (r["checkpoint"], r["suite"], r["task"])):
            lines.append(
                f"| {row['checkpoint']} | {row['suite']} | {row['task']} | {row['n']} | "
                f"{pct(row['success'])} | {num(row['final_score'])} | {num(row['improvement'])} | {num(row['spl'])} |"
            )
    else:
        lines.append("No completed per-task result files yet.")
    lines.append("")

    lines.append("## EASI / Static Spatial QA Probe")
    lines.append("")
    lines.extend(render_probe_summary(probe))
    lines.append("")

    lines.append("## Analysis Checklist")
    lines.append("")
    lines.append("- If `best-val` is not best on `id_test`, validation is not reliable for checkpoint selection.")
    lines.append("- If `id_test` is high but OOD suites drop sharply, inspect shortcut/farming and scene/object leakage.")
    lines.append("- If `ood_geometry` is much lower than `id_test`, inspect metric scaling and stopping errors.")
    lines.append("- If `apparent_size_ordering` is weak in per-task metrics, inspect depth/scale shortcuts separately from equal-size tasks.")
    lines.append("- If EASI drops while navigation rises, RL may be over-specializing the backbone.")
    lines.append("- If latest is worse than best-val on most suites, report stability and early-stopping behavior explicitly.")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build Active Spatial full evaluation report.")
    parser.add_argument("--nav-summary", required=True, help="Path to navigation summary.csv.")
    parser.add_argument("--probe-summary", default=None, help="Path to EASI probe_summary.json.")
    parser.add_argument("--out", required=True, help="Output Markdown report path.")
    args = parser.parse_args()

    nav_rows = load_csv(Path(args.nav_summary))
    probe = load_json(Path(args.probe_summary)) if args.probe_summary else None
    report = build_report(nav_rows, probe)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report)
    print(f"Wrote report: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
