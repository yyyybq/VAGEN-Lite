#!/usr/bin/env python3
"""Fail-fast compatibility check for the Qwen3-VL Active Spatial stack.

This intentionally does not download weights.  Run it on a training node after
installing the pinned environment and before allocating GPUs.
"""
import importlib
import sys

def version(name):
    try:
        mod = importlib.import_module(name)
        return getattr(mod, "__version__", "unknown")
    except Exception as exc:
        return f"MISSING ({exc})"

print("transformers:", version("transformers"))
print("vllm:", version("vllm"))
print("sglang:", version("sglang"))

try:
    import transformers
    version_parts = tuple(map(int, transformers.__version__.split(".")[:2]))
    required = ("AutoProcessor", "AutoModelForVision2Seq")
    missing = [name for name in required if not hasattr(transformers, name)]
    if missing:
        raise RuntimeError(
            "Transformers is too old for Qwen3-VL; missing " + ", ".join(missing)
        )
    if version_parts < (4, 57) or version_parts >= (5, 0):
        raise RuntimeError(
            f"Transformers {transformers.__version__} is outside the supported "
            "Qwen3-VL/VAGEN range; install transformers==4.57.6"
        )
    from transformers.models.qwen3_vl import Qwen3VLForConditionalGeneration, Qwen3VLProcessor
except Exception as exc:
    print(f"ERROR: {exc}", file=sys.stderr)
    raise SystemExit(2)

print("Qwen3-VL environment: OK")
