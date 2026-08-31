#!/usr/bin/env python3
"""Run an Active Spatial checkpoint/suite matrix across fixed local GPUs."""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from active_spatial_eval_sweep import (  # noqa: E402
    append_manifest,
    discover_experiments,
    extract_result_row,
    filter_suites,
    load_suites,
    make_eval_config,
    result_json_path,
    safe_name,
    select_checkpoints,
    select_experiments,
    validation_success_by_step,
    write_csv,
    write_markdown,
    write_yaml,
)


@dataclasses.dataclass
class EvalTask:
    experiment: Any
    checkpoint: Any
    suite: Dict[str, Any]
    agent: str
    config_path: Path
    result_path: Path
    weight: int
    reused: bool = False

    @property
    def label(self) -> str:
        return f"{self.experiment.name}/step{self.checkpoint.step}/{self.suite['name']}/{self.agent}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite-config", required=True)
    parser.add_argument("--exp-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--sweep-name", required=True)
    parser.add_argument("--exps", required=True)
    parser.add_argument("--steps", default="all")
    parser.add_argument("--suites", default="all")
    parser.add_argument("--agents", default="model")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--val-n", type=int, default=4)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=None,
        help="Optional per-worker vLLM GPU memory utilization override.",
    )
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--plan-only", action="store_true")
    return parser


def count_episodes(config: Dict[str, Any]) -> int:
    jsonl_path = Path(config["env"]["jsonl_path"])
    if not jsonl_path.is_absolute():
        jsonl_path = PROJECT_ROOT / jsonl_path
    include = set(config.get("task_types") or config["env"].get("include_task_types") or [])
    exclude = set(config["env"].get("exclude_task_types") or [])
    limit = config.get("num_eval_episodes")
    count = 0
    with jsonl_path.open() as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            task_type = record.get("task_type")
            if include and task_type not in include:
                continue
            if task_type in exclude:
                continue
            count += 1
            if limit is not None and count >= int(limit):
                break
    return max(count, 1)


def append_progress(path: Path, lock: threading.Lock, record: Dict[str, Any]) -> None:
    record = dict(record)
    record["timestamp_utc"] = __import__("datetime").datetime.now(
        __import__("datetime").timezone.utc
    ).isoformat()
    with lock:
        with path.open("a") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def run_shard(
    gpu: str,
    tasks: List[EvalTask],
    progress_path: Path,
    progress_lock: threading.Lock,
    max_attempts: int,
) -> List[str]:
    failures: List[str] = []
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    for index, task in enumerate(tasks, start=1):
        if task.result_path.exists():
            append_progress(
                progress_path,
                progress_lock,
                {"event": "skip_existing", "gpu": gpu, "task": task.label},
            )
            continue

        task.result_path.parent.mkdir(parents=True, exist_ok=True)
        log_path = task.result_path.parent / "parallel_eval.log"
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "evaluation" / "run_eval.py"),
            "--config",
            str(task.config_path),
        ]
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
                    "shard_index": index,
                    "shard_size": len(tasks),
                    "estimated_episodes": task.weight,
                },
            )
            with log_path.open("a") as log_handle:
                log_handle.write(
                    f"\n[parallel] gpu={gpu} task={task.label} "
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
            ok = returncode == 0 and task.result_path.exists()
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
            append_progress(
                progress_path,
                progress_lock,
                {
                    "event": "failed",
                    "gpu": gpu,
                    "task": task.label,
                    "attempts": max_attempts,
                    "returncode": returncode,
                    "result_exists": task.result_path.exists(),
                    "log": str(log_path),
                },
            )
            failures.append(task.label)
    return failures


def main() -> int:
    args = build_parser().parse_args()
    suite_config_path = Path(args.suite_config).resolve()
    exp_root = Path(args.exp_root).resolve()
    out_root = Path(args.out_root).resolve() / args.sweep_name
    resume_root = Path(args.resume_from).resolve() if args.resume_from else None
    gpus = [item.strip() for item in args.gpus.split(",") if item.strip()]
    agents = [item.strip() for item in args.agents.split(",") if item.strip()]
    if not gpus:
        raise ValueError("--gpus must contain at least one device")

    suite_cfg = load_suites(suite_config_path)
    suites = filter_suites(suite_cfg["suites"], args.suites)
    experiments = select_experiments(discover_experiments(exp_root), args.exps)
    if not suites or not experiments:
        raise ValueError("No suites or experiments matched")

    out_root.mkdir(parents=True, exist_ok=True)
    manifest_path = out_root / "manifest.jsonl"
    progress_path = out_root / "parallel_progress.jsonl"
    manifest_path.unlink(missing_ok=True)
    progress_path.unlink(missing_ok=True)

    tasks: List[EvalTask] = []
    episode_count_cache: Dict[str, int] = {}
    for experiment in experiments:
        checkpoints = select_checkpoints(experiment, args.steps, args.val_n)
        for checkpoint in checkpoints:
            for suite in suites:
                for agent in agents:
                    suite_name = safe_name(str(suite["name"]))
                    output_dir = (
                        out_root
                        / experiment.name
                        / f"global_step_{checkpoint.step}"
                        / suite_name
                        / agent
                    )
                    config_path = output_dir / "eval_config.yaml"
                    result_path = result_json_path(output_dir, agent)
                    config = make_eval_config(
                        experiment, checkpoint, suite, suite_cfg, output_dir, agent
                    )
                    if args.gpu_memory_utilization is not None:
                        config.setdefault("model", {})["gpu_memory_utilization"] = (
                            args.gpu_memory_utilization
                        )
                    # Each subprocess sees exactly one physical GPU, mapped to cuda:0.
                    config["env"]["gpu_device"] = 0
                    write_yaml(config_path, config)
                    cache_key = json.dumps(
                        {
                            "jsonl": config["env"]["jsonl_path"],
                            "include": config.get("task_types"),
                            "exclude": config["env"].get("exclude_task_types"),
                            "limit": config.get("num_eval_episodes"),
                        },
                        sort_keys=True,
                    )
                    if cache_key not in episode_count_cache:
                        episode_count_cache[cache_key] = count_episodes(config)
                    task = EvalTask(
                        experiment=experiment,
                        checkpoint=checkpoint,
                        suite=suite,
                        agent=agent,
                        config_path=config_path,
                        result_path=result_path,
                        weight=episode_count_cache[cache_key],
                    )
                    tasks.append(task)
                    append_manifest(
                        manifest_path,
                        {
                            "experiment": experiment.name,
                            "step": checkpoint.step,
                            "suite": suite["name"],
                            "agent": agent,
                            "checkpoint": str(checkpoint.model_dir),
                            "config": str(config_path),
                            "output_dir": str(output_dir),
                            "result": str(result_path),
                            "estimated_episodes": task.weight,
                        },
                    )

    expected_tasks = sum(
        len(select_checkpoints(experiment, args.steps, args.val_n))
        for experiment in experiments
    ) * len(suites) * len(agents)
    if len(tasks) != expected_tasks:
        print(
            f"[warn] expected {expected_tasks} tasks from the selected matrix, "
            f"planned {len(tasks)}",
            flush=True,
        )

    reused = 0
    if resume_root and not args.rerun:
        for task in tasks:
            relative = task.result_path.relative_to(out_root)
            source = resume_root / relative
            if source.exists() and not task.result_path.exists():
                task.result_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, task.result_path)
                task.reused = True
                reused += 1

    pending = [task for task in tasks if args.rerun or not task.result_path.exists()]
    shards: Dict[str, List[EvalTask]] = {gpu: [] for gpu in gpus}
    shard_loads: Dict[str, int] = {gpu: 0 for gpu in gpus}
    for task in sorted(pending, key=lambda item: (-item.weight, item.label)):
        gpu = min(gpus, key=lambda item: (shard_loads[item], item))
        shards[gpu].append(task)
        shard_loads[gpu] += task.weight

    plan = {
        "sweep_name": args.sweep_name,
        "task_count": len(tasks),
        "pending_count": len(pending),
        "reused_count": reused,
        "gpus": gpus,
        "shards": {
            gpu: {
                "estimated_episodes": shard_loads[gpu],
                "task_count": len(shards[gpu]),
                "tasks": [task.label for task in shards[gpu]],
            }
            for gpu in gpus
        },
    }
    (out_root / "parallel_plan.json").write_text(json.dumps(plan, indent=2))
    print(json.dumps({k: plan[k] for k in ("task_count", "pending_count", "reused_count")}), flush=True)
    print(f"shard_loads={shard_loads}", flush=True)
    if args.plan_only:
        return 0

    progress_lock = threading.Lock()
    failures: List[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = {
            executor.submit(
                run_shard,
                gpu,
                shards[gpu],
                progress_path,
                progress_lock,
                args.max_attempts,
            ): gpu
            for gpu in gpus
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

    rows: List[Dict[str, Any]] = []
    val_metrics_cache: Dict[str, Dict[int, Dict[str, float]]] = {}
    for task in tasks:
        exp_name = task.experiment.name
        if exp_name not in val_metrics_cache:
            val_metrics_cache[exp_name] = validation_success_by_step(
                task.experiment, args.val_n
            )
        rows.append(
            extract_result_row(
                task.result_path,
                task.experiment,
                task.checkpoint,
                task.suite,
                task.agent,
                val_metrics_cache[exp_name],
            )
        )
    write_csv(out_root / "summary.csv", rows)
    write_markdown(out_root / "summary.md", rows)
    completion = {
        "planned": len(tasks),
        "complete": sum(task.result_path.exists() for task in tasks),
        "failed": failures,
    }
    (out_root / "parallel_completion.json").write_text(json.dumps(completion, indent=2))
    print(json.dumps(completion, indent=2), flush=True)
    return 1 if failures or completion["complete"] != len(tasks) else 0


if __name__ == "__main__":
    raise SystemExit(main())
