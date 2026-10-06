#!/usr/bin/env python3
"""Fail-closed vLLM load and sampled multimodal inference preflight."""
import argparse
import gc
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from r1_run_canonical_dev_eval32 import VllmPolicy, verify_model_sha256s


def find_curand(cuda_home: Path):
    configured_include = Path(os.environ.get("CURAND_INCLUDE_DIR", ""))
    configured_library = Path(os.environ.get("CURAND_LIBRARY_DIR", ""))
    headers = [configured_include / "curand.h", cuda_home / "include/curand.h",
               cuda_home / "targets/x86_64-linux/include/curand.h"]
    libraries = list(configured_library.glob("libcurand.so*")) + list(cuda_home.glob("lib64/libcurand.so*")) + list(cuda_home.glob("targets/x86_64-linux/lib/libcurand.so*"))
    return next((p for p in headers if p.is_file()), None), sorted(str(p) for p in libraries if p.is_file())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--model-sha256s", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-model-len", type=int, default=4480)
    args = parser.parse_args()
    started = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cuda_home = Path(os.environ.get("CUDA_HOME", ""))
    header, libraries = find_curand(cuda_home)
    nvcc = shutil.which("nvcc")
    report = {
        "status": "RUNNING", "model_key": args.model_key,
        "cuda_home": str(cuda_home), "nvcc": nvcc,
        "curand_header": str(header) if header else None, "curand_libraries": libraries,
        "curand_include_dir": os.environ.get("CURAND_INCLUDE_DIR"),
        "curand_library_dir": os.environ.get("CURAND_LIBRARY_DIR"),
        "flashinfer_sampler": os.environ.get("VLLM_USE_FLASHINFER_SAMPLER"),
    }
    if not nvcc or not header or not libraries:
        raise RuntimeError(f"incomplete CUDA/curand development toolkit: {report}")
    report["nvcc_version"] = subprocess.run(
        [nvcc, "--version"], check=True, text=True, capture_output=True).stdout.strip()
    report["model_hash_verification"] = verify_model_sha256s(args.model_path, args.model_sha256s)
    policy_args = SimpleNamespace(
        model_key=args.model_key, model_path=args.model_path,
        tensor_parallel_size=1, gpu_memory_utilization=.72,
        max_model_len=args.max_model_len,
    )
    policy = VllmPolicy(policy_args, args.output_dir)
    task = "Position where chair appears to the left of table"
    policy.public_task = task
    text, inference_seconds = policy.generate(
        "You are a spatial navigation agent.",
        f"[Observation]:\n<image>\nTask: {task}\nReturn one valid action.",
        Image.new("RGB", (256, 256), (64, 96, 128)),
        20261003,
    )
    if not getattr(policy, "last_task_context_check", {}).get("task_present"):
        raise RuntimeError("sampled multimodal inference did not retain task context")
    report.update({
        "status": "PASS", "sampled_multimodal_inference": True,
        "task_context_check": policy.last_task_context_check,
        "completion_prefix": text[:200], "inference_seconds": inference_seconds,
        "elapsed_seconds": time.time() - started,
    })
    temporary = args.output_dir / "preflight.tmp"
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(args.output_dir / "preflight.json")
    policy.close()
    del policy
    gc.collect()
    print(json.dumps({"vllm_preflight": "PASS", "model_key": args.model_key}), flush=True)
    # vLLM/torch may leave non-daemon distributed threads after engine shutdown.
    # All evidence is atomically durable above; do not let interpreter teardown
    # block the shell from preflighting the second model.
    os._exit(0)


if __name__ == "__main__":
    main()
