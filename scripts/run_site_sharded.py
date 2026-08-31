#!/usr/bin/env python3
"""Shard SITE-Bench (image+video) inference across many single-GPU processes."""

from __future__ import annotations

import argparse
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "exps/unified_eval_runs/easi_results"
N_IMAGE = 4449
N_VIDEO = 3619
TASKS = {
    "site_bench_image": N_IMAGE,
    "site_bench_video_multiimage": N_VIDEO,
}
CKPT_ORDER = [
    "qwen_pretrain_baseline",
    "v46_step200",
    "v50_step150",
    "cambrian_pretrain_baseline",
    "c8_step300",
]


def load_registry():
    reg = yaml.safe_load((ROOT / "scripts/easi_registry.yaml").read_text())
    base = Path(reg.get("base_ckpt_dir", str(ROOT)))
    ckpts = {}
    for name in CKPT_ORDER:
        info = reg["checkpoints"][name]
        path = Path(info["path"])
        if not path.is_absolute():
            path = (base / path).resolve()
        ckpts[name] = {
            "path": str(path),
            "model_type": info.get("model_type", "qwen2_5_vl"),
        }
    return ckpts


def shard_bounds(shard: int, num_shards: int, n: int) -> tuple[int, int]:
    chunk = math.ceil(n / num_shards)
    offset = shard * chunk
    limit = max(0, min(chunk, n - offset))
    return offset, limit


def build_jobs(num_shards: int, tasks: list[str]) -> list[dict]:
    jobs = []
    for ckpt in CKPT_ORDER:
        for task in tasks:
            n = TASKS[task]
            for shard in range(num_shards):
                offset, limit = shard_bounds(shard, num_shards, n)
                if limit <= 0:
                    continue
                jobs.append(
                    {
                        "ckpt": ckpt,
                        "task": task,
                        "shard": shard,
                        "num_shards": num_shards,
                        "offset": offset,
                        "limit": limit,
                        "n": n,
                    }
                )
    return jobs


def assign_slots(jobs: list[dict], gpus: list[int], procs_per_gpu: int) -> list[dict]:
    slots = [(g, s) for g in gpus for s in range(procs_per_gpu)]
    assigned = []
    for i, job in enumerate(jobs):
        if i >= len(slots):
            break
        gpu, slot = slots[i]
        assigned.append({**job, "gpu": gpu, "slot": slot})
    return assigned


def launch_one(job: dict, ckpts: dict, output_root: Path, env_base: dict) -> subprocess.Popen:
    info = ckpts[job["ckpt"]]
    shard_out = (
        output_root
        / job["ckpt"]
        / "shards"
        / f"{job['task']}_s{job['shard']:02d}_of_{job['num_shards']:02d}"
    )
    shard_out.mkdir(parents=True, exist_ok=True)
    log = (
        output_root
        / "shard_logs"
        / f"site_{job['ckpt']}_{job['task']}_s{job['shard']:02d}_gpu{job['gpu']}p{job['slot']}.log"
    )
    log.parent.mkdir(parents=True, exist_ok=True)
    py = env_base["PYTHON"]
    lmms = str(ROOT / "third_party/lmms-eval")
    cmd = [
        py, "-m", "lmms_eval",
        "--model", info["model_type"],
        "--model_args", f"pretrained={info['path']}",
        "--tasks", job["task"],
        "--batch_size", "1",
        "--output_path", str(shard_out),
        "--log_samples",
        "--offset", str(job["offset"]),
        "--limit", str(job["limit"]),
    ]
    env = os.environ.copy()
    env.update({k: v for k, v in env_base.items() if k != "PYTHON"})
    env["CUDA_VISIBLE_DEVICES"] = str(job["gpu"])
    env["HF_DATASETS_CACHE"] = env_base.get(
        "HF_DATASETS_CACHE_ROOT",
        "/mnt/umm/users/yinbaiqiao/.cache/huggingface/datasets_site_shared",
    )
    old_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = lmms + (os.pathsep + old_pp if old_pp else "")
    with log.open("w") as fh:
        fh.write(f"$ CUDA_VISIBLE_DEVICES={job['gpu']} {' '.join(cmd)}\n")
        fh.flush()
        proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env, stdout=fh, stderr=subprocess.STDOUT)
    print(
        f"[launch] {job['ckpt']} {job['task']} shard {job['shard']}/{job['num_shards']} "
        f"offset={job['offset']} limit={job['limit']} gpu={job['gpu']}p{job['slot']} pid={proc.pid}",
        flush=True,
    )
    return proc


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--procs-per-gpu", type=int, default=1)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--tasks", default="site_bench_image,site_bench_video_multiimage")
    parser.add_argument("--job-start", type=int, default=0)
    parser.add_argument("--job-end", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--datasets-cache-root",
        default="/mnt/umm/users/yinbaiqiao/.cache/huggingface/datasets_site_shared",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--wait-extract", action="store_true")
    args = parser.parse_args()

    extract_marker = Path(
        "/mnt/umm/users/yinbaiqiao/.cache/huggingface/sitebench/.extract_complete"
    )
    if args.wait_extract:
        print("[WAIT] sitebench extract...", flush=True)
        for _ in range(720):
            if extract_marker.exists():
                break
            time.sleep(30)
        if not extract_marker.exists():
            print("[FAIL] extract timeout", file=sys.stderr)
            return 2

    gpus = [int(x) for x in args.gpus.split(",") if x.strip() != ""]
    tasks = [t for t in args.tasks.split(",") if t.strip()]
    ckpts = load_registry()
    jobs = build_jobs(args.num_shards, tasks)
    end = len(jobs) if args.job_end is None else args.job_end
    slice_jobs = jobs[args.job_start:end]

    env_base = {
        "PYTHON": os.environ.get(
            "PYTHON", "/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite/bin/python"
        ),
        "HF_HOME": "/mnt/umm/users/yinbaiqiao/.cache/huggingface",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "CAMBRIAN_SRC": "/mnt/umm/users/yinbaiqiao/cambrian-s",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "TORCH_EXTENSIONS_DIR": "/mnt/umm/users/yinbaiqiao/.cache/torch_extensions_yinbaiqiao",
        "TORCH_CUDA_ARCH_LIST": "9.0",
        "HF_DATASETS_CACHE_ROOT": args.datasets_cache_root,
    }
    conda = Path("/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite")
    if (conda / "bin").is_dir():
        env_base["PATH"] = f"{conda}/bin:" + os.environ.get("PATH", "")
        env_base["CUDA_HOME"] = str(conda)
        cc = conda / "bin/x86_64-conda-linux-gnu-gcc"
        cxx = conda / "bin/x86_64-conda-linux-gnu-g++"
        if cc.exists():
            env_base["CC"] = str(cc)
        if cxx.exists():
            env_base["CXX"] = str(cxx)

    capacity = len(gpus) * args.procs_per_gpu
    print(
        f"SITE plan: {len(slice_jobs)} jobs, capacity/wave={capacity}, "
        f"gpus={gpus}, procs_per_gpu={args.procs_per_gpu}, "
        f"shards={args.num_shards}, tasks={tasks}",
        flush=True,
    )
    if args.dry_run:
        for j in slice_jobs[: min(20, len(slice_jobs))]:
            print(j)
        print(f"... total {len(slice_jobs)}")
        return 0

    failed = 0
    for wave_start in range(0, len(slice_jobs), capacity):
        wave = slice_jobs[wave_start : wave_start + capacity]
        assigned = assign_slots(wave, gpus, args.procs_per_gpu)
        print(f"[wave] {wave_start}-{wave_start + len(assigned)}", flush=True)
        procs = []
        for job in assigned:
            procs.append((job, launch_one(job, ckpts, args.output_dir.resolve(), env_base)))
            time.sleep(1.0)
        for job, proc in procs:
            rc = proc.wait()
            status = "OK" if rc == 0 else f"FAIL rc={rc}"
            print(
                f"[done] {job['ckpt']} {job['task']} shard {job['shard']} "
                f"gpu={job['gpu']}p{job['slot']} {status}",
                flush=True,
            )
            if rc != 0:
                failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
