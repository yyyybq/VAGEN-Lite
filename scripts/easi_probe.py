#!/usr/bin/env python3
"""
easi_probe.py — EASI-based spatial QA probe for RL training checkpoints

Wraps lmms-eval-revsi to provide one-click evaluation of multiple RL checkpoints
across multiple spatial benchmarks. Results are cached on disk and aggregated
into a comparison table.

Usage:
    # Evaluate all 'key' checkpoints on 'default' (mindcube_tiny)
    python scripts/easi_probe.py

    # Evaluate specific checkpoints by name
    python scripts/easi_probe.py --ckpts base,v33_step150,v35_step200

    # Evaluate a named group
    python scripts/easi_probe.py --ckpts recent --benchmarks core

    # Add an on-the-fly checkpoint (no registry entry needed)
    python scripts/easi_probe.py --add "v37_step100:path/to/actor/huggingface" --benchmarks default

    # Force re-evaluation (ignore cached results)
    python scripts/easi_probe.py --ckpts v35_step200 --rerun

    # Multi-GPU evaluation (faster but needs accelerate)
    python scripts/easi_probe.py --ckpts key --benchmarks core --nproc 4

    # List all registered checkpoints and benchmarks
    python scripts/easi_probe.py --list

    # Show results only (no re-evaluation, just parse cached results)
    python scripts/easi_probe.py --results-only
"""

import argparse
import glob
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

# ─────────────────────── defaults ────────────────────────────────────────────

SCRIPT_DIR = Path(__file__).parent
REGISTRY_PATH = SCRIPT_DIR / "probe_registry.yaml"
OUTPUT_DIR = SCRIPT_DIR.parent / "easi_probe_results"
PYTHON = sys.executable  # use same python as this script

# Ensure stdout is unbuffered when running under nohup / redirected pipes
if not sys.stdout.line_buffering:
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, line_buffering=True)


# ─────────────────────── helpers ─────────────────────────────────────────────

def load_registry(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_ckpt_path(path: str, base_dir: str) -> str:
    """
    Return an absolute path.
    - HuggingFace Hub IDs (e.g., 'Qwen/...') are returned as-is.
    - Absolute paths are returned as-is.
    - Relative paths are resolved against base_dir.
    """
    if path.startswith("/") or path.startswith("~"):
        return str(Path(path).expanduser())
    if re.match(r"^[A-Za-z0-9_\-]+/[A-Za-z0-9_\-.]+$", path):
        # Looks like an HF Hub ID (e.g., "Qwen/Qwen2.5-VL-3B-Instruct")
        return path
    return str(Path(base_dir) / path)


def sanitize_model_name(model_name: str) -> str:
    """Mirrors lmms-eval's sanitize_model_name (last-two-parts variant)."""
    parts = model_name.split("/")
    last_two = "/".join(parts[-2:]) if len(parts) > 1 else parts[-1]
    return re.sub(r'["<>:/|\\?*\[\]]+', "__", last_two)


def find_result_json(output_path: Path, bench: str) -> Optional[Path]:
    """
    Given the per-checkpoint/benchmark output_path, find the latest results JSON.
    lmms-eval writes: {output_path}/{model_sanitized}/{datetime}_results.json
    """
    pattern = str(output_path / "**" / "*_results.json")
    candidates = glob.glob(pattern, recursive=True)
    if not candidates:
        return None
    # Pick the most recently modified one
    return Path(max(candidates, key=os.path.getmtime))


def parse_result(result_json: Path, bench: str) -> Optional[Dict]:
    """Parse a lmms-eval results.json and extract the benchmark metrics."""
    try:
        data = json.loads(result_json.read_text())
        results = data.get("results", {})
        # The task name in results may differ slightly from bench name;
        # find the closest matching key.
        if bench in results:
            return results[bench]
        # fallback: look for any key that starts with bench
        for k, v in results.items():
            if k.startswith(bench) or bench.startswith(k):
                return v
        return None
    except Exception as e:
        print(f"  [warn] Failed to parse {result_json}: {e}")
        return None


def run_lmms_eval(
    ckpt_name: str,
    ckpt_path: str,
    bench: str,
    output_path: Path,
    lmms_eval_dir: str,
    gpu: str,
    nproc: int,
    batch_size: int,
    model_type: str = "qwen2_5_vl",
) -> bool:
    """
    Run lmms-eval for a single checkpoint × benchmark.
    Returns True on success.
    """
    output_path.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env["PYTHONPATH"] = lmms_eval_dir
    env.setdefault("HF_HOME", "/scratch/by2593/.huggingface")
    env["TOKENIZERS_PARALLELISM"] = "false"

    model_args = f"pretrained={ckpt_path}"

    if nproc > 1:
        # accelerate launch for multi-GPU
        cmd = [
            PYTHON, "-m", "accelerate.commands.launch",
            "--num_processes", str(nproc),
            "-m", "lmms_eval",
        ]
    else:
        cmd = [PYTHON, "-m", "lmms_eval"]

    cmd += [
        "--model", model_type,
        "--model_args", model_args,
        "--tasks", bench,
        "--batch_size", str(batch_size),
        "--output_path", str(output_path),
    ]

    log_file = output_path / "lmms_eval.log"
    print(f"\n  CMD: {' '.join(cmd)}")
    print(f"  LOG: {log_file}")

    with open(log_file, "w") as log_f:
        proc = subprocess.Popen(
            cmd,
            cwd=lmms_eval_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        output_lines = []
        for line in proc.stdout:
            log_f.write(line)
            log_f.flush()
            output_lines.append(line.rstrip())
        proc.wait()

    # Print last 20 lines for quick status
    for line in output_lines[-20:]:
        print(f"    {line}")

    if proc.returncode != 0:
        print(f"  [error] lmms_eval exited with code {proc.returncode}")
        return False
    return True


def format_metric(metrics: dict) -> str:
    """Format the primary metric from a benchmark result dict.

    lmms-eval can suffix metric names with ',none', ',stderr', etc.
    We strip those suffixes when matching priority names.
    """
    # Strip aggregation suffix for comparison (e.g. "overall_accuracy,none" → "overall_accuracy")
    def bare(k: str) -> str:
        return k.split(",")[0]

    priority = ["overall_accuracy", "acc_norm", "acc", "accuracy", "exact_match"]
    bare_to_key = {bare(k): k for k, v in metrics.items() if isinstance(v, float)}

    for p in priority:
        if p in bare_to_key:
            val = metrics[bare_to_key[p]]
            return f"{val*100:.1f}%"
    # Fallback: first float value that isn't stderr
    for k, v in metrics.items():
        if isinstance(v, float) and not k.startswith("_") and "stderr" not in k:
            return f"{v*100:.1f}%"
    return "?"


def print_comparison_table(
    results: Dict[str, Dict[str, Optional[dict]]],
    benchmarks: List[str],
    registry: dict,
):
    """Print a pretty comparison table."""
    ckpt_names = list(results.keys())
    col_w = max(14, max(len(b) for b in benchmarks) + 2)
    name_w = max(20, max(len(n) for n in ckpt_names) + 2)
    desc_w = 50

    # Header
    header = f"{'Checkpoint':<{name_w}}" + "".join(f"{b:^{col_w}}" for b in benchmarks)
    if any(registry["checkpoints"].get(n, {}).get("training_id_m4") for n in ckpt_names):
        header += f"{'ID_m4(train)':>14}"
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    # Base result for delta computation
    base_metrics = results.get("base", {})

    for name in ckpt_names:
        row = f"{name:<{name_w}}"
        for bench in benchmarks:
            m = results[name].get(bench)
            if m is None:
                row += f"{'—':^{col_w}}"
            else:
                primary = format_metric(m)
                # compute delta vs base
                base_m = base_metrics.get(bench)
                if base_m and name != "base":
                    try:
                        def _primary_float(d):
                            priority = ["overall_accuracy", "acc_norm", "acc",
                                        "accuracy", "exact_match"]
                            bare_map = {k.split(",")[0]: v for k, v in d.items()
                                        if isinstance(v, float)}
                            for p in priority:
                                if p in bare_map:
                                    return bare_map[p]
                            return next(v for k, v in d.items()
                                        if isinstance(v, float) and "stderr" not in k)
                        v = _primary_float(m)
                        bv = _primary_float(base_m)
                        delta = (v - bv) * 100
                        sign = "+" if delta >= 0 else ""
                        primary += f"({sign}{delta:.1f})"
                    except StopIteration:
                        pass
                row += f"{primary:^{col_w}}"

        ckpt_info = registry["checkpoints"].get(name, {})
        if ckpt_info.get("training_id_m4"):
            row += f"{ckpt_info['training_id_m4']:>14.3f}"
        print(row)

    print("=" * len(header))

    # Per-benchmark subtable with sub-metrics
    for bench in benchmarks:
        has_sub = False
        for name in ckpt_names:
            m = results[name].get(bench) or {}
            if len(m) > 1:
                has_sub = True
                break
        if not has_sub:
            continue

        print(f"\n  [{bench}] Sub-metrics:")
        all_keys = set()
        for name in ckpt_names:
            all_keys.update(k for k in (results[name].get(bench) or {}).keys()
                            if not k.startswith("_") and isinstance(
                                (results[name][bench] or {}).get(k), float))
        for key in sorted(all_keys):
            sub_row = f"    {key:<40}"
            for name in ckpt_names:
                m = results[name].get(bench) or {}
                val = m.get(key)
                if isinstance(val, float):
                    sub_row += f"{val*100:>8.1f}%"
                else:
                    sub_row += f"{'—':>9}"
            print(sub_row)


def main():
    parser = argparse.ArgumentParser(
        description="EASI-based spatial QA probe for RL training checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:")[1] if "Usage:" in __doc__ else "",
    )
    parser.add_argument(
        "--registry", default=str(REGISTRY_PATH),
        help="Path to probe_registry.yaml (default: scripts/probe_registry.yaml)",
    )
    parser.add_argument(
        "--ckpts", default="key",
        help="Comma-separated checkpoint names, a named group from registry, or 'all'. "
             "Default: 'key'",
    )
    parser.add_argument(
        "--benchmarks", default="default",
        help="Comma-separated lmms-eval task names, or a group name from registry. "
             "Default: 'default' (mindcube_tiny)",
    )
    parser.add_argument(
        "--add", action="append", default=[],
        metavar="NAME:PATH",
        help="Add an ad-hoc checkpoint not in registry. Can be repeated. "
             "Format: 'name:path/to/actor/huggingface'",
    )
    parser.add_argument(
        "--output_dir", default=str(OUTPUT_DIR),
        help="Directory to save lmms-eval outputs (default: easi_probe_results/)",
    )
    parser.add_argument(
        "--gpu", default="4",
        help="CUDA_VISIBLE_DEVICES value, e.g. '4' or '4,5,6,7'. Default: '4'",
    )
    parser.add_argument(
        "--nproc", type=int, default=1,
        help="Number of GPUs for accelerate launch. Default: 1",
    )
    parser.add_argument(
        "--batch_size", type=int, default=1,
        help="Batch size per GPU. Default: 1",
    )
    parser.add_argument(
        "--rerun", action="store_true",
        help="Force re-evaluation even if cached results exist",
    )
    parser.add_argument(
        "--results-only", action="store_true",
        help="Only parse and display existing results, do not run any evaluation",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all registered checkpoints and benchmark groups and exit",
    )
    args = parser.parse_args()

    # ── Load registry ──────────────────────────────────────────────────────
    registry = load_registry(Path(args.registry))
    base_ckpt_dir = registry.get("base_ckpt_dir", ".")
    lmms_eval_dir = registry.get("lmms_eval_dir",
        "/scratch/by2593/project/Active_Spatial/lmms-eval-revsi")

    # ── --list ─────────────────────────────────────────────────────────────
    if args.list:
        print("\n=== Registered Checkpoints ===")
        for name, info in registry["checkpoints"].items():
            id_m4 = f"  [ID_m4={info['training_id_m4']:.3f}]" if info.get("training_id_m4") else ""
            print(f"  {name:<25} {info.get('desc','')}{id_m4}")
        print("\n=== Checkpoint Groups ===")
        for gname, members in registry.get("checkpoint_groups", {}).items():
            print(f"  {gname:<15} {', '.join(members)}")
        print("\n=== Benchmark Groups ===")
        for gname, tasks in registry.get("benchmark_groups", {}).items():
            print(f"  {gname:<15} {', '.join(tasks)}")
        return

    # ── Resolve checkpoints ────────────────────────────────────────────────
    ckpt_groups = registry.get("checkpoint_groups", {})
    ckpt_registry = registry["checkpoints"]

    if args.ckpts == "all":
        ckpt_names = list(ckpt_registry.keys())
    elif args.ckpts in ckpt_groups:
        ckpt_names = ckpt_groups[args.ckpts]
    else:
        ckpt_names = [s.strip() for s in args.ckpts.split(",") if s.strip()]

    # Build (name, path, desc) list
    ckpts: List[Tuple[str, str, str]] = []
    for name in ckpt_names:
        if name not in ckpt_registry:
            print(f"[warn] Checkpoint '{name}' not in registry — skipping")
            continue
        info = ckpt_registry[name]
        path = resolve_ckpt_path(info["path"], base_ckpt_dir)
        ckpts.append((name, path, info.get("desc", "")))

    # Add on-the-fly checkpoints from --add
    for entry in args.add:
        if ":" not in entry:
            print(f"[error] --add format must be 'name:path', got: {entry}")
            sys.exit(1)
        name, path = entry.split(":", 1)
        path = resolve_ckpt_path(path, base_ckpt_dir)
        ckpts.append((name.strip(), path.strip(), "ad-hoc"))
        print(f"  [add] {name} → {path}")

    if not ckpts:
        print("[error] No valid checkpoints found. Use --list to see available checkpoints.")
        sys.exit(1)

    # ── Resolve benchmarks ────────────────────────────────────────────────
    bench_groups = registry.get("benchmark_groups", {})

    if args.benchmarks in bench_groups:
        benchmarks = bench_groups[args.benchmarks]
    else:
        benchmarks = [s.strip() for s in args.benchmarks.split(",") if s.strip()]

    if not benchmarks:
        print("[error] No benchmarks specified.")
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Print plan ────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  EASI Probe — {len(ckpts)} checkpoints × {len(benchmarks)} benchmarks")
    print(f"  GPU: {args.gpu}  nproc: {args.nproc}  batch_size: {args.batch_size}")
    print(f"  Output: {output_dir}")
    print(f"{'='*70}")
    for name, path, desc in ckpts:
        print(f"  • {name:<25} {desc}")
    print(f"\n  Benchmarks: {', '.join(benchmarks)}")
    print(f"{'='*70}\n")

    # ── Evaluate ─────────────────────────────────────────────────────────
    # results[ckpt_name][bench] = metric_dict or None
    all_results: Dict[str, Dict[str, Optional[dict]]] = {}

    for ckpt_name, ckpt_path, ckpt_desc in ckpts:
        all_results[ckpt_name] = {}
        for bench in benchmarks:
            out_path = output_dir / ckpt_name / bench
            existing = find_result_json(out_path, bench)

            if existing and not args.rerun and not args.results_only:
                print(f"  [cache] {ckpt_name}/{bench} → {existing.name}")
                metrics = parse_result(existing, bench)
                all_results[ckpt_name][bench] = metrics
                continue

            if args.results_only:
                if existing:
                    metrics = parse_result(existing, bench)
                    all_results[ckpt_name][bench] = metrics
                else:
                    print(f"  [miss]  {ckpt_name}/{bench} (no cached result)")
                    all_results[ckpt_name][bench] = None
                continue

            # Run evaluation
            print(f"\n{'─'*70}")
            print(f"  Evaluating: {ckpt_name}  bench={bench}")
            print(f"  Path: {ckpt_path}")
            print(f"{'─'*70}")

            if not Path(ckpt_path).exists() and "/" in ckpt_path and not ckpt_path.startswith("Qwen/"):
                print(f"  [error] Path does not exist: {ckpt_path}")
                all_results[ckpt_name][bench] = None
                continue

            t0 = time.time()
            ckpt_info = ckpt_registry.get(ckpt_name, {})
            model_type = ckpt_info.get("model_type", "qwen2_5_vl")
            success = run_lmms_eval(
                ckpt_name=ckpt_name,
                ckpt_path=ckpt_path,
                bench=bench,
                output_path=out_path,
                lmms_eval_dir=lmms_eval_dir,
                gpu=args.gpu,
                nproc=args.nproc,
                batch_size=args.batch_size,
                model_type=model_type,
            )
            elapsed = time.time() - t0
            print(f"  Elapsed: {elapsed/60:.1f} min")

            if success:
                result_json = find_result_json(out_path, bench)
                if result_json:
                    metrics = parse_result(result_json, bench)
                    all_results[ckpt_name][bench] = metrics
                    print(f"  Result: {format_metric(metrics) if metrics else '?'}")
                else:
                    print(f"  [warn] Evaluation succeeded but no result JSON found.")
                    all_results[ckpt_name][bench] = None
            else:
                all_results[ckpt_name][bench] = None

    # ── Summary table ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("  COMPARISON TABLE")
    print(f"{'='*70}")
    print_comparison_table(all_results, benchmarks, registry)

    # ── Save summary JSON ─────────────────────────────────────────────────
    summary_path = output_dir / "probe_summary.json"
    summary = {
        "checkpoints": [
            {"name": n, "path": p, "desc": d} for n, p, d in ckpts
        ],
        "benchmarks": benchmarks,
        "results": {
            name: {
                bench: metrics
                for bench, metrics in bench_dict.items()
            }
            for name, bench_dict in all_results.items()
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"\n  Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
