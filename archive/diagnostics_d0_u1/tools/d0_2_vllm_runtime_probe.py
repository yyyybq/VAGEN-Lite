#!/usr/bin/env python3
"""Minimal vLLM runtime config probe for D0.2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/mnt/umm/users/yinbaiqiao/hf_cache/Cambrian-S-7B-LFP")
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.45)
    parser.add_argument("--limit-images", type=int, default=1)
    args = parser.parse_args()

    from vagen.models.cambrian_plugin import register

    register()
    import vagen.models.cambrian_vllm  # noqa: F401
    from vllm import LLM

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        enforce_eager=True,
        limit_mm_per_prompt={"image": args.limit_images},
    )
    mc = llm.llm_engine.model_config
    pc = getattr(llm.llm_engine, "parallel_config", None)
    if pc is None and hasattr(llm.llm_engine, "vllm_config"):
        pc = getattr(llm.llm_engine.vllm_config, "parallel_config", None)
    result = {
        "model": args.model,
        "requested_max_model_len": args.max_model_len,
        "effective_max_model_len": getattr(mc, "max_model_len", None),
        "dtype": str(getattr(mc, "dtype", None)),
        "tensor_parallel_size": getattr(pc, "tensor_parallel_size", None) if pc is not None else None,
        "limit_mm_per_prompt": getattr(mc, "limit_mm_per_prompt", None),
        "enforce_eager": getattr(mc, "enforce_eager", None),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
