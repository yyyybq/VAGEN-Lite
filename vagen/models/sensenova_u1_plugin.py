"""
vLLM general plugin bootstrap for SenseNova-U1.
"""

from __future__ import annotations

import os
import sys


def register() -> None:
    from vllm import ModelRegistry

    target = "vagen.models.sensenova_u1_vllm:SenseNovaU1VLLMForCausalLM"
    for arch in ("NEOChatModel", "SenseNovaU1ForCausalLMAdapter"):
        ModelRegistry.register_model(arch, target)
        print(f"[sensenova_u1_plugin] Registered {arch} -> {target}", flush=True)

    src = os.environ.get("SENSENOVA_U1_SRC", "/mnt/umm/users/yinbaiqiao/SenseNova-U1/src")
    if src not in sys.path:
        sys.path.insert(0, src)
    try:
        import sensenova_u1  # noqa: F401
    except Exception as e:
        print(f"[sensenova_u1_plugin] sensenova_u1 import deferred/failed: {e}", flush=True)
