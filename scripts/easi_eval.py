#!/usr/bin/env python3
"""Run the official EASI suite for one or more registered checkpoints.

The default is the complete EASI-8 suite. MindCube is evaluated only as the
``mindcube_tiny`` member of EASI-8; there is no project-specific MindCube path.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
REGISTRY_PATH = SCRIPT_DIR / "easi_registry.yaml"
OUTPUT_DIR = PROJECT_ROOT / "easi_results"
EASI_8 = [
    "vsi_bench",
    "mmsi_bench",
    "mindcube_tiny",
    "viewspatial",
    "site",
    "blink",
    "3dsrbench",
    "embspatial",
]


def load_registry(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text()) or {}


def resolve_checkpoint(path: str, base_dir: str) -> str:
    expanded = str(Path(path).expanduser())
    if expanded.startswith("/"):
        return expanded
    if re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", path):
        return path
    return str((Path(base_dir) / path).resolve())


def resolve_names(value: str, groups: dict[str, list[str]], available: dict[str, Any]) -> list[str]:
    if value == "all":
        return list(available)
    if value in groups:
        return list(groups[value])
    return [part.strip() for part in value.split(",") if part.strip()]


def resolve_benchmarks(value: str, groups: dict[str, list[str]]) -> list[str]:
    if value in groups:
        return list(groups[value])
    return [part.strip() for part in value.split(",") if part.strip()]


def normalize_score(value: Any) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    score = float(value)
    return score / 100.0 if abs(score) > 1.0 else score


def load_scores(path: Path, benchmarks: list[str]) -> dict[str, float | None] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text())
    raw = payload.get("scores", {})
    return {bench: normalize_score(raw.get(bench)) for bench in benchmarks}


def is_complete(scores: dict[str, float | None] | None, benchmarks: list[str]) -> bool:
    return bool(scores) and all(scores.get(bench) is not None for bench in benchmarks)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate checkpoints with the official EASI suite (EASI-8 by default)."
    )
    parser.add_argument("--registry", default=str(REGISTRY_PATH))
    parser.add_argument("--ckpts", default="current_main", help="Checkpoint names/group, or 'all'. Default: current_main.")
    parser.add_argument(
        "--benchmarks",
        default="easi_8",
        help="Official EASI benchmark keys/group. Default: easi_8.",
    )
    parser.add_argument(
        "--add",
        action="append",
        default=[],
        metavar="NAME:PATH[:MODEL_TYPE]",
        help="Add an ad-hoc checkpoint.",
    )
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR))
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--nproc", type=int, default=1)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--results-only", action="store_true")
    parser.add_argument(
        "--no-summary",
        action="store_true",
        help="Do not write the shared easi_summary.json (used by parallel checkpoint workers).",
    )
    parser.add_argument("--no-accelerate", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Smoke-test sample limit passed to EASI/lmms-eval.")
    parser.add_argument("--list", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    registry_path = Path(args.registry).resolve()
    registry = load_registry(registry_path)
    checkpoints = dict(registry.get("checkpoints", {}))
    checkpoint_groups = dict(registry.get("checkpoint_groups", {}))
    benchmark_groups = dict(registry.get("benchmark_groups", {}))

    if args.list:
        print("Checkpoints:")
        for name, info in checkpoints.items():
            print(f"  {name:28s} {info.get('desc', '')}")
        print("Benchmark groups:")
        for name, members in benchmark_groups.items():
            print(f"  {name:28s} {', '.join(members)}")
        return 0

    names = resolve_names(args.ckpts, checkpoint_groups, checkpoints)
    selected: list[tuple[str, str, str, str]] = []
    base_dir = registry.get("base_ckpt_dir", str(PROJECT_ROOT))
    for name in names:
        info = checkpoints.get(name)
        if not info:
            print(f"[error] Unknown checkpoint: {name}", file=sys.stderr)
            return 2
        selected.append((
            name,
            resolve_checkpoint(str(info["path"]), base_dir),
            str(info.get("model_type", "qwen2_5_vl")),
            str(info.get("desc", "")),
        ))

    for item in args.add:
        parts = item.split(":")
        if len(parts) < 2:
            print(f"[error] Invalid --add value: {item}", file=sys.stderr)
            return 2
        if len(parts) >= 3 and parts[-1] in {"qwen2_5_vl", "cambrians"}:
            name, model_type = parts[0], parts[-1]
            raw_path = ":".join(parts[1:-1])
        else:
            name, raw_path, model_type = parts[0], ":".join(parts[1:]), "qwen2_5_vl"
        selected.append((name, resolve_checkpoint(raw_path, base_dir), model_type, "ad-hoc"))

    if not selected:
        print("[error] No checkpoints selected.", file=sys.stderr)
        return 2

    benchmarks = resolve_benchmarks(args.benchmarks, benchmark_groups)
    if not benchmarks:
        print("[error] No benchmarks selected.", file=sys.stderr)
        return 2

    easi_dir = Path(registry.get("easi_dir", PROJECT_ROOT / "third_party/EASI")).resolve()
    runner = easi_dir / "scripts/submissions/run_easi_eval.py"
    lmms_eval_dir = Path(registry.get("lmms_eval_dir", PROJECT_ROOT / "third_party/lmms-eval")).resolve()
    if not runner.exists():
        print(f"[error] Official EASI runner not found: {runner}", file=sys.stderr)
        return 2

    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    all_results: dict[str, dict[str, float | None] | None] = {}
    failures = 0

    print(f"EASI: {len(selected)} checkpoint(s) x {len(benchmarks)} benchmark(s)")
    print(f"Benchmarks: {', '.join(benchmarks)}")
    for name, checkpoint, model_type, desc in selected:
        checkpoint_out = output_root / name
        result_path = checkpoint_out / "easi_results.json"
        cached = load_scores(result_path, benchmarks)
        if is_complete(cached, benchmarks) and not args.rerun:
            print(f"[cache] {name}: complete EASI result")
            all_results[name] = cached
            continue
        if args.results_only:
            print(f"[{'partial' if cached else 'miss'}] {name}")
            all_results[name] = cached
            continue

        if not Path(checkpoint).exists() and not re.fullmatch(
            r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", checkpoint
        ):
            print(f"[error] Checkpoint does not exist: {checkpoint}", file=sys.stderr)
            all_results[name] = cached
            failures += 1
            continue

        cmd = [
            sys.executable,
            str(runner),
            "--backend", "lmms-eval",
            "--model", model_type,
            "--model-args", f"pretrained={checkpoint}",
            "--output-dir", str(checkpoint_out),
            "--nproc", str(args.nproc),
            "--no-rich",
        ]
        if benchmarks != EASI_8:
            cmd.extend(["--benchmarks", ",".join(benchmarks)])
        if args.rerun:
            cmd.append("--rerun")
        if args.no_accelerate:
            cmd.append("--no-accelerate")
        if args.limit is not None:
            cmd.extend(["--limit", str(args.limit)])

        env = os.environ.copy()
        env.setdefault("CUDA_VISIBLE_DEVICES", args.gpu)
        old_pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = str(lmms_eval_dir) + (os.pathsep + old_pythonpath if old_pythonpath else "")
        env.setdefault("TOKENIZERS_PARALLELISM", "false")
        print(f"\n[{name}] {desc}\n$ {' '.join(cmd)}", flush=True)
        completed = subprocess.run(cmd, cwd=easi_dir, env=env)
        scores = load_scores(result_path, benchmarks)
        all_results[name] = scores
        if completed.returncode or not is_complete(scores, benchmarks):
            failures += 1
            print(f"[error] {name}: incomplete EASI result", file=sys.stderr)

    summary = {
        "suite": "EASI-8" if benchmarks == EASI_8 else "EASI custom",
        "benchmarks": benchmarks,
        "checkpoints": [
            {"name": name, "path": path, "model_type": model_type, "desc": desc}
            for name, path, model_type, desc in selected
        ],
        "results": all_results,
    }
    if not args.no_summary:
        (output_root / "easi_summary.json").write_text(json.dumps(summary, indent=2))
        print(f"\nSummary: {output_root / 'easi_summary.json'}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
