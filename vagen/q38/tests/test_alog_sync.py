"""A_log reaches vLLM in fp32 in the colocated runs (STACK_SUPPORT 2(c) and 3.3; DESIGN_ASYNC 5.4 Z1).

vLLM keeps every GDN layer's A_log in fp32 (vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:450-457,
loaded by sharded_weight_loader -> default_weight_loader, a plain copy_).  verl's checkpoint engines for separate
rollout cards cast every tensor to rollout_dtype bf16 (nccl_checkpoint_engine.py:141-147, kimi 42-50), which pear
worked around with its own engine.  The colocated runs (sync, colocate_async) do not use them: V1 forces the backend
"naive" (trainer_base.py:406-407) and the actor worker then sends ``module.state_dict()`` of the FSDP engine as it is
(engine_workers.py:765-813, transformer_impl.py:1058-1115) through BucketedWeightSender, whose cast is commented out
(bucketed_weight_transfer.py:118-125: "we should not force cast weight here because some parameters ... have to keep
fp32 precision").  So A_log arrives in fp32 exactly when the actor's master weights are fp32 (fsdp_config.model_dtype
fp32, verl's default and the q38 configs).  Three checks:

1. the sender and receiver, run on the CPU over shared memory, hand an fp32 A_log over bit for bit (no bf16
   rounding) and keep each tensor's dtype;
2. the installed vLLM still declares A_log as an fp32 parameter loaded by sharded_weight_loader (source tripwire);
3. the contract refuses a colocated run whose actor master weights are not fp32 or whose backend is not naive.

Run (login node, 1 thread):
    OMP_NUM_THREADS=1 ... nice -n 19 python -B -m pytest vagen/q38/tests/test_alog_sync.py -q -x -p no:cacheprovider
"""

from __future__ import annotations

import asyncio
import importlib.util
import re
import threading
import types
import uuid
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf, open_dict

from vagen.q38 import contract as C

REPO = Path(__file__).resolve().parents[3]
SMOKE = REPO / "vagen" / "q38" / "configs" / "smoke_4b_example.yaml"


def _roundtrip(weights, monkeypatch):
    from verl.workers.rollout.vllm_rollout import bucketed_weight_transfer as bwt

    device = types.SimpleNamespace(synchronize=lambda: None, empty_cache=lambda: None, ipc_collect=lambda: None)
    monkeypatch.setattr(bwt, "get_torch_device", lambda: device)  # the CPU has no empty_cache / ipc_collect
    monkeypatch.setattr(bwt, "is_support_ipc", lambda: False)
    handle = f"ipc://@q38-alog-{uuid.uuid4().hex}"  # abstract socket: nothing on disk
    received, errors = [], []

    def on_bucket(bucket, is_last):
        received.extend((name, t.clone()) for name, t in bucket)

    receiver = bwt.BucketedWeightReceiver(handle, torch.device("cpu"), use_shm=True)

    def run_receiver():
        try:
            receiver.receive_weights(on_bucket)
        except Exception as e:  # pragma: no cover - reported below
            errors.append(e)

    thread = threading.Thread(target=run_receiver, daemon=True)
    thread.start()
    sender = bwt.BucketedWeightSender(handle, bucket_size_mb=1, use_shm=True)
    asyncio.run(asyncio.wait_for(sender.async_send_weights(iter(weights)), timeout=60))
    thread.join(timeout=60)
    assert not thread.is_alive() and not errors, errors
    return dict(received)


def test_naive_sync_keeps_fp32_a_log(monkeypatch):
    g = torch.Generator().manual_seed(0)
    # values with low mantissa bits set: bf16 keeps 8 bits of mantissa, so a cast would change them
    a_log = (torch.rand(48, generator=g, dtype=torch.float64) * 3 - 1).to(torch.float32)
    a_log[0] = 1.0 + 2.0 ** -20
    assert not torch.equal(a_log.to(torch.bfloat16).to(torch.float32), a_log)
    weights = [
        ("model.language_model.layers.0.linear_attn.A_log", a_log),
        ("model.language_model.layers.0.linear_attn.dt_bias", torch.randn(48, generator=g)),
        ("model.language_model.layers.0.mlp.down_proj.weight", torch.randn(600, 400, generator=g)),  # 960 kB
        ("model.language_model.layers.0.mlp.up_proj.weight", torch.randn(600, 400, generator=g)),  # next bucket
        ("model.language_model.layers.1.linear_attn.A_log", a_log.flip(0).contiguous()),
        ("model.visual.blocks.0.norm1.weight", torch.randn(1152, generator=g).to(torch.bfloat16)),
    ]
    got = _roundtrip(weights, monkeypatch)
    assert list(got) == [n for n, _ in weights]
    for name, sent in weights:
        assert got[name].dtype == sent.dtype, name  # no cast on the way, whatever the dtype
        assert torch.equal(got[name].view(torch.uint8), sent.view(torch.uint8)), name  # bit for bit
    assert got["model.language_model.layers.0.linear_attn.A_log"].dtype == torch.float32


def _vllm_gdn_source() -> str:
    spec = importlib.util.find_spec("vllm")
    assert spec is not None and spec.submodule_search_locations, "vLLM is not installed in this environment"
    path = Path(list(spec.submodule_search_locations)[0]) / "model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py"
    return path.read_text()


def test_vllm_keeps_a_log_fp32():
    src = _vllm_gdn_source()
    decl = re.search(r"self\.A_log\s*=\s*nn\.Parameter\((.*?)\n\s*\)\n", src, re.S)
    assert decl and "dtype=torch.float32" in decl.group(1), "vLLM's A_log is no longer declared fp32"
    assert re.search(r'set_weight_attrs\(self\.A_log,\s*\{"weight_loader":\s*sharded_weight_loader\(0\)\}\)', src)


def _cfg():
    cfg = OmegaConf.load(SMOKE)
    return cfg


def _violations(cfg):
    report = C.evaluate(cfg, endpoint=None, check_paths=False, read_files=False)
    return set(report["violations"])


def test_contract_master_weights_fp32():
    key = "actor_rollout_ref.actor.fsdp_config.model_dtype"
    assert key not in _violations(_cfg())
    for value in ("bf16", "bfloat16", "fp16"):
        cfg = _cfg()
        with open_dict(cfg):
            OmegaConf.update(cfg, key, value)
        assert key in _violations(cfg), value
    cfg = _cfg()
    with open_dict(cfg):
        OmegaConf.update(cfg, key, "float32")
    assert key not in _violations(cfg)


def test_contract_naive_backend_when_colocated():
    key = "actor_rollout_ref.rollout.checkpoint_engine.backend"
    assert key not in _violations(_cfg())
    for mode in ("colocate_async", "sync"):
        cfg = _cfg()
        with open_dict(cfg):
            OmegaConf.update(cfg, "trainer.v1.trainer_mode", mode)
            OmegaConf.update(cfg, key, "nccl")
        assert key in _violations(cfg), mode
    cfg = _cfg()
    with open_dict(cfg):
        OmegaConf.update(cfg, "trainer.v1.trainer_mode", "separate_async")
        OmegaConf.update(cfg, key, "nccl")
    assert key not in _violations(cfg)  # phase 4: its own engine (bf16 cast, STACK_SUPPORT 3.3) is a separate question
