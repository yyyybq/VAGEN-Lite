#!/usr/bin/env python3
"""
probe_table.py — Live summary table for EASI probe results

Reads all *_results.json files from easi_probe_results/ and renders
a Markdown-formatted comparison table. Run at any time — shows results
as they come in.

Usage:
    python scripts/probe_table.py                     # print table to stdout
    python scripts/probe_table.py --out docs/probe_results.md  # also write MD file
    python scripts/probe_table.py --watch             # refresh every 60s
    python scripts/probe_table.py --benchmarks mindcube_tiny,sparbench_tiny  # subset
"""

import argparse
import glob
import json
import os
import time
from pathlib import Path

# ─────────────────────── Configuration ───────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent
REGISTRY_PATH = SCRIPT_DIR / "probe_registry.yaml"
OUTPUT_DIR = SCRIPT_DIR.parent / "easi_probe_results"

# Display names for benchmarks (shorter for table columns)
BENCH_SHORT = {
    "mindcube_tiny":  "MindCube",
    "sparbench_tiny": "SPAR",
    "mmsi_bench":     "MMSI",
    "viewspatial":    "ViewSpatial",
    "embspatial":     "EmbSpatial",
    "blink":          "BLINK",
    "cv_bench":       "CV-Bench",
    "spatial457":     "Spatial457",
    "refspatial":     "RefSpatial",
}

# Primary metric key per benchmark (lmms-eval uses key,aggregation format)
BENCH_PRIMARY_METRIC = {
    "mindcube_tiny":  "overall_accuracy",
    "sparbench_tiny": "accuracy",
    "mmsi_bench":     "accuracy",
    "viewspatial":    "accuracy",
    "embspatial":     "accuracy",
    "blink":          "accuracy",
    "cv_bench":       "accuracy",
    "spatial457":     "accuracy",
    "refspatial":     "accuracy",
}

# ─────────────────────── Helpers ─────────────────────────────────────────────

import yaml


def load_registry(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def find_result_json(output_dir: Path, ckpt: str, bench: str) -> Path | None:
    pattern = str(output_dir / ckpt / bench / "**" / "*_results.json")
    candidates = glob.glob(pattern, recursive=True)
    if not candidates:
        return None
    return Path(max(candidates, key=os.path.getmtime))


def extract_primary(result_json: Path, bench: str) -> float | None:
    """Return primary metric (0-1 range) from a lmms-eval results.json."""
    try:
        data = json.loads(result_json.read_text())
        results = data.get("results", {})
        # Find the task result (key may have suffix)
        task_result = None
        if bench in results:
            task_result = results[bench]
        else:
            for k, v in results.items():
                if k.startswith(bench) or bench.startswith(k.split(",")[0]):
                    task_result = v
                    break
        if task_result is None:
            return None

        # Try primary metric keys (with and without ,none suffix)
        want = BENCH_PRIMARY_METRIC.get(bench, "accuracy")
        for suffix in ("", ",none", ",stderr"):
            key = want + suffix
            if key in task_result and isinstance(task_result[key], (int, float)):
                if "stderr" not in key:
                    return float(task_result[key])

        # Fallback: first float that isn't stderr
        for k, v in task_result.items():
            if isinstance(v, float) and "stderr" not in k and not k.startswith("_"):
                return v
    except Exception:
        pass
    return None


def extract_sub_metrics(result_json: Path, bench: str) -> dict:
    """Return all numeric sub-metrics from a result (excluding stderr)."""
    out = {}
    try:
        data = json.loads(result_json.read_text())
        results = data.get("results", {})
        task_result = None
        if bench in results:
            task_result = results[bench]
        else:
            for k, v in results.items():
                if k.startswith(bench) or bench.startswith(k.split(",")[0]):
                    task_result = v
                    break
        if task_result is None:
            return out
        for k, v in task_result.items():
            if isinstance(v, float) and "stderr" not in k and not k.startswith("_"):
                # strip aggregation suffix
                bare = k.split(",")[0]
                out[bare] = v
    except Exception:
        pass
    return out


def render_table(
    registry: dict,
    benchmarks: list[str],
    output_dir: Path,
    show_sub: bool = True,
) -> str:
    ckpt_reg = registry["checkpoints"]
    # Ordered list: first base, then by training_id_m4 desc, then by name
    ckpt_names = []
    for name in ["base"] + [n for n in ckpt_reg if n != "base"]:
        path = ckpt_reg[name]["path"]
        # Check if HF dir exists (skip unavailable)
        if name == "base" or path.startswith("Qwen/"):
            ckpt_names.append(name)
        elif Path(ckpt_reg["base_ckpt_dir"] if "base_ckpt_dir" in registry else "")  \
                .joinpath(path).exists():
            ckpt_names.append(name)
        else:
            base_dir = registry.get("base_ckpt_dir", "")
            if Path(base_dir).joinpath(path).exists():
                ckpt_names.append(name)

    # Collect all results
    results = {}
    for name in ckpt_names:
        results[name] = {}
        for bench in benchmarks:
            rj = find_result_json(output_dir, name, bench)
            if rj:
                results[name][bench] = {
                    "primary": extract_primary(rj, bench),
                    "sub": extract_sub_metrics(rj, bench),
                }

    base_primary = {b: (results.get("base", {}).get(b) or {}).get("primary")
                    for b in benchmarks}

    # ── Markdown table ─────────────────────────────────────────────────────
    now = time.strftime("%Y-%m-%d %H:%M")
    lines = []
    lines.append(f"# EASI Spatial Probe Results")
    lines.append(f"")
    lines.append(f"> Auto-generated by `scripts/probe_table.py` · Last updated: {now}")
    lines.append(f"> Base model: **Qwen/Qwen2.5-VL-3B-Instruct**  ")
    lines.append(f"> Backend: lmms-eval-revsi · `—` = not yet evaluated")
    lines.append(f"")

    # Header row
    header_bench = [BENCH_SHORT.get(b, b) for b in benchmarks]
    header = "| Checkpoint | Train ID_m4 | " + " | ".join(header_bench) + " |"
    sep = "| --- | ---: | " + " | ".join(["---:"] * len(benchmarks)) + " |"
    lines.append(header)
    lines.append(sep)

    for name in ckpt_names:
        info = ckpt_reg[name]
        id_m4 = info.get("training_id_m4")
        id_m4_str = f"{id_m4:.3f}" if id_m4 else "—"
        cells = [f"**{name}**" if name == "base" else name, id_m4_str]

        for bench in benchmarks:
            r = results[name].get(bench)
            if r is None or r["primary"] is None:
                # check if in-progress
                log_f = output_dir / name / bench / "lmms_eval.log"
                if log_f.exists() and log_f.stat().st_size > 0:
                    cells.append("⏳")
                else:
                    cells.append("—")
                continue
            pct = r["primary"] * 100
            cell = f"{pct:.1f}%"
            # delta vs base
            bp = base_primary.get(bench)
            if bp is not None and name != "base":
                delta = (r["primary"] - bp) * 100
                sign = "+" if delta >= 0 else ""
                cell += f" ({sign}{delta:.1f})"
            cells.append(cell)

        lines.append("| " + " | ".join(cells) + " |")

    lines.append("")

    # ── Sub-metric breakdown tables ────────────────────────────────────────
    if show_sub:
        for bench in benchmarks:
            # Collect all sub-metric keys across all checkpoints
            all_keys: set[str] = set()
            for name in ckpt_names:
                r = results[name].get(bench)
                if r:
                    all_keys.update(r["sub"].keys())
            if not all_keys:
                continue
            # only show if there's more than 1 metric
            if len(all_keys) <= 1:
                continue

            sub_keys = sorted(all_keys)
            lines.append(f"### {BENCH_SHORT.get(bench, bench)} — Sub-metrics")
            lines.append("")
            sh = "| Checkpoint | " + " | ".join(sub_keys) + " |"
            ss = "| --- | " + " | ".join(["---:"] * len(sub_keys)) + " |"
            lines.append(sh)
            lines.append(ss)
            for name in ckpt_names:
                r = results[name].get(bench)
                row_cells = [name]
                for k in sub_keys:
                    if r and k in r["sub"]:
                        row_cells.append(f"{r['sub'][k]*100:.1f}%")
                    else:
                        row_cells.append("—")
                lines.append("| " + " | ".join(row_cells) + " |")
            lines.append("")

    # ── Progress summary ───────────────────────────────────────────────────
    total = len(ckpt_names) * len(benchmarks)
    done = sum(
        1 for n in ckpt_names for b in benchmarks
        if results[n].get(b) and results[n][b]["primary"] is not None
    )
    lines.append(f"---")
    lines.append(f"**Progress: {done}/{total} evaluations completed**")
    lines.append("")

    # Active evaluations
    active = []
    for log_f in sorted(output_dir.glob("*/*/lmms_eval.log")):
        if not log_f.exists() or log_f.stat().st_size == 0:
            continue
        parts = log_f.parts
        ckpt_n = parts[-3]
        bench_n = parts[-2]
        rj = find_result_json(output_dir, ckpt_n, bench_n)
        if rj is None:  # still in progress
            last = log_f.read_text().strip().split("\n")[-1]
            active.append(f"  - ⏳ `{ckpt_n}/{bench_n}` — {last[:80]}")
    if active:
        lines.append("**In progress:**")
        lines.extend(active)
        lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="EASI probe results table")
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR))
    parser.add_argument("--registry", default=str(REGISTRY_PATH))
    parser.add_argument("--benchmarks", default="core5",
                        help="Comma-separated bench names or a group from registry")
    parser.add_argument("--out", default=None,
                        help="Write Markdown table to this file path")
    parser.add_argument("--watch", action="store_true",
                        help="Refresh every 60s until all done")
    parser.add_argument("--no-sub", action="store_true",
                        help="Skip sub-metric breakdown tables")
    args = parser.parse_args()

    import yaml
    registry = yaml.safe_load(open(args.registry))
    bench_groups = registry.get("benchmark_groups", {})
    if args.benchmarks in bench_groups:
        benchmarks = bench_groups[args.benchmarks]
    else:
        benchmarks = [b.strip() for b in args.benchmarks.split(",") if b.strip()]

    output_dir = Path(args.output_dir)
    out_path = Path(args.out) if args.out else None

    def run_once():
        table = render_table(registry, benchmarks, output_dir, show_sub=not args.no_sub)
        print(table)
        if out_path:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(table)
            print(f"\n[Saved to {out_path}]")

    if args.watch:
        while True:
            os.system("clear")
            run_once()
            # count done
            done = sum(1 for p in output_dir.glob("*/*/**/*_results.json"))
            print(f"\nRefreshing in 60s... (Ctrl+C to stop)")
            try:
                time.sleep(60)
            except KeyboardInterrupt:
                break
    else:
        run_once()


if __name__ == "__main__":
    main()
