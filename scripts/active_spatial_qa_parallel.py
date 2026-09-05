#!/usr/bin/env python3
"""Run static spatial QA for Active Spatial checkpoints across local GPUs."""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from active_spatial_eval_sweep import (  # noqa: E402
    discover_experiments,
    safe_name,
    select_checkpoints,
    select_experiments,
)
from active_spatial_full_eval import infer_model_type  # noqa: E402
from easi_eval import EASI_8, is_complete, load_scores, resolve_benchmarks, load_registry  # noqa: E402


@dataclasses.dataclass
class QATask:
    experiment: str
    step: int
    name: str
    checkpoint: Path
    model_type: str
    result_path: Path

    @property
    def label(self) -> str:
        return f"{self.experiment}/step{self.step}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exp-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--sweep-name", required=True)
    parser.add_argument("--exps", required=True)
    parser.add_argument("--steps", default="all")
    parser.add_argument("--benchmarks", default="easi_8")
    parser.add_argument("--registry", default="scripts/easi_registry.yaml")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--nproc", type=int, default=1)
    parser.add_argument("--val-n", type=int, default=4)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    return parser


def append_progress(path: Path, lock: threading.Lock, record: dict[str, Any]) -> None:
    payload = dict(record)
    payload["timestamp_utc"] = datetime.now(timezone.utc).isoformat()
    with lock:
        with path.open("a") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def run_shard(
    gpu: str,
    tasks: list[QATask],
    benchmarks_arg: str,
    registry: Path,
    output_root: Path,
    nproc: int,
    max_attempts: int,
    rerun: bool,
    progress_path: Path,
    progress_lock: threading.Lock,
) -> list[str]:
    failures: list[str] = []
    env = os.environ.copy()
    # The child sees one physical GPU remapped to cuda:0.
    env["CUDA_VISIBLE_DEVICES"] = gpu
    for shard_index, task in enumerate(tasks, start=1):
        existing_scores = load_scores(
            task.result_path,
            resolve_benchmarks(
                benchmarks_arg, load_registry(registry).get("benchmark_groups", {})
            ),
        )
        if (
            task.result_path.exists()
            and not rerun
            and is_complete(
                existing_scores,
                resolve_benchmarks(
                    benchmarks_arg, load_registry(registry).get("benchmark_groups", {})
                ),
            )
        ):
            append_progress(
                progress_path,
                progress_lock,
                {"event": "skip_existing", "gpu": gpu, "task": task.label},
            )
            continue

        task.result_path.parent.mkdir(parents=True, exist_ok=True)
        log_path = task.result_path.parent / "parallel_qa.log"
        cmd = [
            sys.executable,
            str(SCRIPTS_DIR / "easi_eval.py"),
            "--registry",
            str(registry),
            "--ckpts",
            "",
            "--benchmarks",
            benchmarks_arg,
            "--output_dir",
            str(output_root),
            "--gpu",
            "0",
            "--nproc",
            str(nproc),
            "--no-summary",
            "--add",
            f"{task.name}:{task.checkpoint}:{task.model_type}",
        ]
        if rerun:
            cmd.append("--rerun")

        ok = False
        returncode = None
        for attempt in range(1, max_attempts + 1):
            append_progress(
                progress_path,
                progress_lock,
                {
                    "event": "start",
                    "gpu": gpu,
                    "task": task.label,
                    "attempt": attempt,
                    "shard_index": shard_index,
                    "shard_size": len(tasks),
                },
            )
            with log_path.open("a") as log_handle:
                log_handle.write(
                    f"\n[parallel-qa] gpu={gpu} task={task.label} "
                    f"attempt={attempt}/{max_attempts}\n"
                )
                log_handle.flush()
                proc = subprocess.run(
                    cmd,
                    cwd=str(PROJECT_ROOT),
                    env=env,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                )
            returncode = proc.returncode
            scores = load_scores(task.result_path, resolve_benchmarks(
                benchmarks_arg, load_registry(registry).get("benchmark_groups", {})
            ))
            ok = returncode == 0 and is_complete(
                scores,
                resolve_benchmarks(
                    benchmarks_arg, load_registry(registry).get("benchmark_groups", {})
                ),
            )
            append_progress(
                progress_path,
                progress_lock,
                {
                    "event": "complete" if ok else "attempt_failed",
                    "gpu": gpu,
                    "task": task.label,
                    "attempt": attempt,
                    "returncode": returncode,
                    "result_exists": task.result_path.exists(),
                    "log": str(log_path),
                },
            )
            if ok:
                break
            if attempt < max_attempts:
                time.sleep(5)
        if not ok:
            failures.append(task.label)
    return failures


def main() -> int:
    args = build_parser().parse_args()
    exp_root = Path(args.exp_root).resolve()
    sweep_root = Path(args.out_root).resolve() / args.sweep_name
    output_root = sweep_root / "easi_results"
    registry = Path(args.registry).resolve()
    registry_payload = load_registry(registry)
    benchmarks = resolve_benchmarks(
        args.benchmarks, registry_payload.get("benchmark_groups", {})
    )
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    if not gpus:
        raise ValueError("--gpus must contain at least one device")

    experiments = select_experiments(discover_experiments(exp_root), args.exps)
    output_root.mkdir(parents=True, exist_ok=True)
    tasks: list[QATask] = []
    for experiment in experiments:
        for checkpoint in select_checkpoints(experiment, args.steps, args.val_n):
            name = safe_name(f"{experiment.name}_step{checkpoint.step}")
            tasks.append(
                QATask(
                    experiment=experiment.name,
                    step=checkpoint.step,
                    name=name,
                    checkpoint=checkpoint.model_dir,
                    model_type=infer_model_type(experiment.name, checkpoint.model_dir),
                    result_path=output_root / name / "easi_results.json",
                )
            )
    if not tasks:
        raise ValueError("No checkpoints matched the requested experiments and steps")

    shards: dict[str, list[QATask]] = {gpu: [] for gpu in gpus}
    for index, task in enumerate(tasks):
        shards[gpus[index % len(gpus)]].append(task)
    plan = {
        "sweep_name": args.sweep_name,
        "benchmarks": benchmarks,
        "task_count": len(tasks),
        "gpus": gpus,
        "shards": {gpu: [task.label for task in shard] for gpu, shard in shards.items()},
        "checkpoints": [
            {
                "experiment": task.experiment,
                "step": task.step,
                "name": task.name,
                "path": str(task.checkpoint),
                "model_type": task.model_type,
            }
            for task in tasks
        ],
    }
    (sweep_root / "qa_parallel_plan.json").write_text(json.dumps(plan, indent=2))
    print(json.dumps({"task_count": len(tasks), "benchmarks": benchmarks}, indent=2))
    if args.plan_only:
        return 0

    progress_path = sweep_root / "qa_parallel_progress.jsonl"
    progress_path.unlink(missing_ok=True)
    progress_lock = threading.Lock()
    failures: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = {
            executor.submit(
                run_shard,
                gpu,
                shard,
                args.benchmarks,
                registry,
                output_root,
                args.nproc,
                args.max_attempts,
                args.rerun,
                progress_path,
                progress_lock,
            ): gpu
            for gpu, shard in shards.items()
            if shard
        }
        for future in concurrent.futures.as_completed(futures):
            gpu = futures[future]
            try:
                failures.extend(future.result())
            except Exception as exc:
                failures.append(f"gpu={gpu}: {exc}")
                append_progress(
                    progress_path,
                    progress_lock,
                    {"event": "shard_exception", "gpu": gpu, "error": repr(exc)},
                )

    results = {
        task.name: load_scores(task.result_path, benchmarks)
        for task in tasks
    }
    summary = {
        "suite": "EASI-8" if benchmarks == EASI_8 else "EASI custom",
        "benchmarks": benchmarks,
        "checkpoints": plan["checkpoints"],
        "results": results,
    }
    (output_root / "easi_summary.json").write_text(json.dumps(summary, indent=2))
    complete = sum(is_complete(results[task.name], benchmarks) for task in tasks)
    completion = {
        "planned": len(tasks),
        "complete": complete,
        "failed": failures,
    }
    (sweep_root / "qa_parallel_completion.json").write_text(
        json.dumps(completion, indent=2)
    )
    print(json.dumps(completion, indent=2))
    return 1 if failures or complete != len(tasks) else 0


if __name__ == "__main__":
    raise SystemExit(main())
