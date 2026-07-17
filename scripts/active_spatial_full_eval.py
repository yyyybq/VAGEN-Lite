#!/usr/bin/env python3
"""
One-command Active Spatial evaluation.

Runs three layers:
1. embodied navigation test sweep;
2. validation/test alignment through the navigation summary;
3. EASI static spatial-QA capability probe;
4. consolidated Markdown analysis report.

The command is restart-safe: existing navigation result JSON files and EASI
result JSON files are skipped unless --rerun is passed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from active_spatial_eval_sweep import (  # noqa: E402
    DEFAULT_EXP_ROOT,
    DEFAULT_OUT_ROOT,
    discover_experiments,
    safe_name,
    select_checkpoints,
    select_experiments,
)


def run_cmd(cmd: List[str], dry_run: bool = False) -> int:
    print()
    print("$ " + " ".join(cmd), flush=True)
    if dry_run:
        return 0
    return subprocess.run(cmd, cwd=str(PROJECT_ROOT)).returncode


def infer_model_type(exp_name: str, ckpt_path: Path) -> str:
    text = f"{exp_name} {ckpt_path}".lower()
    if "cambrian" in text or exp_name.startswith("c"):
        return "cambrians"
    return "qwen2_5_vl"


def resolve_eval_checkpoints(exp_root: Path, exps: str, steps: str, val_n: int) -> List[Tuple[str, Path, str]]:
    experiments = select_experiments(discover_experiments(exp_root), exps)
    out: List[Tuple[str, Path, str]] = []
    for exp in experiments:
        for ckpt in select_checkpoints(exp, steps, val_n):
            name = safe_name(f"{exp.name}_step{ckpt.step}")
            model_type = infer_model_type(exp.name, ckpt.model_dir)
            out.append((name, ckpt.model_dir, model_type))
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run full Active Spatial evaluation stack.")
    parser.add_argument("--exp-root", default=str(DEFAULT_EXP_ROOT), help="Experiment root.")
    parser.add_argument("--exps", required=True, help="'all', comma list, or re:<regex>.")
    parser.add_argument("--steps", default="best-val,latest", help="Checkpoint selector passed to sweep/probe.")
    parser.add_argument("--suite-config", default="examples/evaluate/active_spatial/test_suites.yaml")
    parser.add_argument("--sweep-name", default=None)
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--nav-suites", default="all")
    parser.add_argument("--nav-agents", default="model")
    parser.add_argument("--val-n", type=int, default=4)
    parser.add_argument("--include-nav", action="store_true", default=True)
    parser.add_argument("--no-nav", dest="include_nav", action="store_false")
    parser.add_argument("--include-easi", action="store_true", default=True)
    parser.add_argument("--no-easi", dest="include_easi", action="store_false")
    parser.add_argument("--include-analysis", action="store_true", default=True)
    parser.add_argument("--no-analysis", dest="include_analysis", action="store_false")
    parser.add_argument("--easi-registry", default="scripts/probe_registry.yaml")
    parser.add_argument("--easi-benchmarks", default="core5")
    parser.add_argument("--easi-gpu", default="4")
    parser.add_argument("--easi-nproc", type=int, default=1)
    parser.add_argument("--easi-batch-size", type=int, default=1)
    parser.add_argument("--easi-include-base", action="store_true", default=True)
    parser.add_argument("--no-easi-base", dest="easi_include_base", action="store_false")
    parser.add_argument("--run", action="store_true", help="Actually run evaluations.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands and write plan only.")
    parser.add_argument("--rerun", action="store_true", help="Force rerun existing results.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    dry_run = args.dry_run or not args.run
    exp_root = Path(args.exp_root).resolve()
    suite_config = Path(args.suite_config)
    suite_name = args.sweep_name or safe_name(Path(args.suite_config).stem)
    # If suite YAML has a name field, active_spatial_eval_sweep will use it.
    try:
        import yaml

        with suite_config.open("r") as f:
            suite_yaml = yaml.safe_load(f) or {}
        suite_name = args.sweep_name or safe_name(str(suite_yaml.get("name") or suite_config.stem))
    except Exception:
        pass

    full_out_root = Path(args.out_root).resolve() / suite_name
    full_out_root.mkdir(parents=True, exist_ok=True)
    plan_path = full_out_root / "full_eval_plan.json"

    ckpts = resolve_eval_checkpoints(exp_root, args.exps, args.steps, args.val_n)
    plan = {
        "exp_root": str(exp_root),
        "exps": args.exps,
        "steps": args.steps,
        "suite_config": str(suite_config),
        "nav_suites": args.nav_suites,
        "nav_agents": args.nav_agents,
        "easi_benchmarks": args.easi_benchmarks,
        "checkpoints": [
            {"name": name, "path": str(path), "model_type": model_type}
            for name, path, model_type in ckpts
        ],
    }
    plan_path.write_text(json.dumps(plan, indent=2))
    print(f"Plan: {plan_path}", flush=True)
    if not ckpts:
        print(f"[warn] No checkpoints resolved from {exp_root} for exps={args.exps} steps={args.steps}")

    failed = 0

    if args.include_nav:
        nav_cmd = [
            sys.executable,
            str(SCRIPTS_DIR / "active_spatial_eval_sweep.py"),
            "--suite-config",
            str(suite_config),
            "--exp-root",
            str(exp_root),
            "--out-root",
            args.out_root,
            "--sweep-name",
            suite_name,
            "--exps",
            args.exps,
            "--steps",
            args.steps,
            "--suites",
            args.nav_suites,
            "--agents",
            args.nav_agents,
            "--val-n",
            str(args.val_n),
        ]
        if args.rerun:
            nav_cmd.append("--rerun")
        if args.run and not args.dry_run:
            nav_cmd.append("--run")
        else:
            nav_cmd.append("--dry-run")
        failed += int(run_cmd(nav_cmd, dry_run=False) != 0)

    probe_summary = full_out_root / "easi_probe_results" / "probe_summary.json"
    if args.include_easi:
        easi_cmd = [
            sys.executable,
            str(SCRIPTS_DIR / "easi_probe.py"),
            "--registry",
            args.easi_registry,
            "--ckpts",
            "base" if args.easi_include_base else "",
            "--benchmarks",
            args.easi_benchmarks,
            "--output_dir",
            str(full_out_root / "easi_probe_results"),
            "--gpu",
            args.easi_gpu,
            "--nproc",
            str(args.easi_nproc),
            "--batch_size",
            str(args.easi_batch_size),
        ]
        for name, path, model_type in ckpts:
            easi_cmd.extend(["--add", f"{name}:{path}:{model_type}"])
        if args.rerun:
            easi_cmd.append("--rerun")
        if not args.run or args.dry_run:
            easi_cmd.append("--results-only")
        failed += int(run_cmd(easi_cmd, dry_run=False) != 0)

    if args.include_analysis:
        report_cmd = [
            sys.executable,
            str(SCRIPTS_DIR / "active_spatial_eval_report.py"),
            "--nav-summary",
            str(full_out_root / "summary.csv"),
            "--probe-summary",
            str(probe_summary),
            "--out",
            str(full_out_root / "analysis_report.md"),
        ]
        failed += int(run_cmd(report_cmd, dry_run=False) != 0)

    if failed:
        print(f"[error] {failed} stage(s) failed")
        return 1
    print()
    print(f"Full evaluation output: {full_out_root}")
    print(f"Analysis report: {full_out_root / 'analysis_report.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
