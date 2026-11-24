"""PORT_P23 item 9 (2026-11-24): the critic's ranks under init_empty_weights build the value model from its config.

verl's FSDP engine builds the critic (model_type value_model) with load_valuehead_model, i.e.
AutoModelForTokenClassification.from_pretrained, on every rank (verl/verl/workers/engine/fsdp/transformer_impl.py
_build_module, verl/verl/utils/model.py load_valuehead_model).  transformers 5 reads and converts the whole checkpoint
there even under init_empty_weights: for the 27B critic in fp32 that is up to 8 x ~108 GB of host memory at once.  The
language model branch already builds such ranks with from_config (verl 0a7be8b).  The change (PATCH below) makes the
value model branch do the same: AutoModelForTokenClassification.from_config on a copy of the config, with the source
rank's dtype and attention implementation (flash_attention_2), left in eval mode as from_pretrained leaves the source
rank; a failure there raises with the rank instead of falling back to TRL's value-head model.  The weights of those
ranks come from the source rank's broadcast after sharding (FSDP2: fsdp2_load_full_state_dict; FSDP1:
sync_module_states), as before.

PATCH: placement (a) is a verl change (load_valuehead_model(..., from_config=...), passed by _build_module); placement
(b) is vagen/q38/critic_init.py, a monkeypatch installed in the critic's worker.  This file is the same for both but
for ``patched()`` / ``install_in_this_process()``, which install the monkeypatch under (b) and do nothing under (a).

On a tiny random qwen3_5 config (test_critic_model_cpu.tiny_config) and a bf16 Qwen3_5ForConditionalGeneration
checkpoint (the 27B C0 is bf16 on disk; the critic is built in fp32):

- one process, the source rank / another rank by torch.distributed.get_rank (no mesh): the other rank calls neither
  from_pretrained nor safe_open and holds meta parameters only (fp32, the source rank's class, the same attention
  implementation asked for: flash_attention_2, dtype by ``dtype=``), in eval mode as the source rank, on a copy of the
  engine's config (which is left as the source rank's build leaves it); the source rank reads the checkpoint, its
  weights equal it;
- one process, rank 1 with an injected from_config failure: a RuntimeError naming the rank and from_config, logged
  with the exception; no step of the TRL fallback (is_trl_available, the causal LM / vision2seq from_pretrained, TRL's
  value-head from_pretrained) and no read of the checkpoint;
- two processes (gloo over loopback TCP, 1 thread each; a 1-D CPU mesh as create_device_mesh builds it for
  fsdp_size -1; the engine's own _build_module and _build_fsdp_module with strategy fsdp2, i.e. apply_fsdp2 +
  fsdp2_load_full_state_dict), each rank seeded differently (seed + rank): rank 1 reads nothing; after the sync both
  ranks' full state dicts and buffers equal a plain from_pretrained built with rank 0's seed, bitwise (the backbone is
  the checkpoint upcast to fp32; the 1-unit score head is the source rank's random init, broadcast to every rank, as
  before; a control shows rank 1's seed gives another head);
- two processes, rank 1 with an injected from_config failure: rank 1 raises the RuntimeError in _build_module, before
  any collective and without the TRL fallback; rank 0 then fails in the sync (the gloo peer is gone; under NCCL it
  would wait for the collective's timeout unless the job is stopped, which Ray does when a worker's init fails);
- unchanged: the language model branch (rank 1 from_config, rank 0 from_pretrained) and load_valuehead_model's
  default call (from_pretrained with the same arguments as before).

CPU has no flash-attention kernels: the from_pretrained / from_config calls are wrapped to record the attention
implementation asked for and to build "eager" instead (the weights do not depend on it); so are the TRL fallback's
from_pretrained calls, so that a fallback (the draft's negative control) runs as it would on GPU.  verl's qwen3_5
monkey patch is not applied (it patches classes process-wide and does not touch the weights).

Run (login node, 1 thread a process; the two-process tests start 2 processes):
    OMP_NUM_THREADS=1 ... nice -n 19 python -B -m pytest vagen/q38/tests/test_critic_meta_init_cpu.py -q -s -p no:cacheprovider
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import time
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vagen.q38.tests.test_critic_model_cpu import tiny_config as _tiny_config

torch.set_num_threads(1)

SEED = 1234
INJECTED = "injected from_config failure (test)"
_MISSING = object()


# ---- the placement's hook: (a) verl change, nothing to install ----

def patched():
    """The context the tests run in: nothing under placement (a) (the change is in verl)."""
    return contextlib.nullcontext()


def install_in_this_process():
    """A spawned rank's set-up: nothing under placement (a)."""


@pytest.fixture(autouse=True)
def _placement():
    with patched():
        yield


# ---- helpers ----

def tiny_config():
    """test_critic_model_cpu's tiny qwen3_5 config, untied as the 27B C0 (tie_word_embeddings false at both levels):
    get_init_weight_context_manager uses the meta device only for an untied model."""
    cfg = _tiny_config()
    cfg.tie_word_embeddings = False
    cfg.text_config.tie_word_embeddings = False
    return cfg


class _Cfg(SimpleNamespace):
    """The engine config's attribute and dict access (FSDPEngineConfig is read both ways)."""

    def get(self, key, default=None):
        return getattr(self, key, default)


def _patch_attr(patches, obj, name, value):
    patches.append((obj, name, obj.__dict__.get(name, _MISSING)))
    setattr(obj, name, value)


def _restore(patches):
    for obj, name, saved in reversed(patches):
        if saved is _MISSING:
            delattr(obj, name)
        else:
            setattr(obj, name, saved)


@contextlib.contextmanager
def recording(fail_from_config=False):
    """Record AutoModelForTokenClassification.from_pretrained / from_config calls and transformers' safe_open calls;
    "flash_attention_2" is built as "eager" (no kernels on CPU).  ``fail_from_config``: from_config raises after the
    record.  Restored afterwards."""
    import transformers.modeling_utils as MU
    from transformers import AutoModelForTokenClassification as A

    rec = {"from_pretrained": [], "from_config": [], "safe_open": []}
    orig = {name: getattr(A, name) for name in ("from_pretrained", "from_config")}
    orig_safe_open = MU.safe_open
    patches = []

    def wrap(name):
        def call(*args, **kwargs):
            config = kwargs.get("config", args[0] if args else None)
            dtype_kwarg = [k for k in ("dtype", "torch_dtype") if k in kwargs]
            rec[name].append({"attn_implementation": kwargs.get("attn_implementation"),
                              "dtype": str(kwargs.get("dtype", kwargs.get("torch_dtype"))),
                              "dtype_kwarg": dtype_kwarg,
                              "path": str(kwargs.get("pretrained_model_name_or_path", "")),
                              "config_id": id(config)})
            if name == "from_config" and fail_from_config:
                raise ValueError(INJECTED)
            if kwargs.get("attn_implementation") == "flash_attention_2":
                kwargs["attn_implementation"] = "eager"
            return orig[name](*args, **kwargs)

        return call

    def safe_open(*args, **kwargs):
        rec["safe_open"].append(str(args[0]) if args else str(kwargs.get("filename")))
        return orig_safe_open(*args, **kwargs)

    _patch_attr(patches, A, "from_pretrained", wrap("from_pretrained"))
    _patch_attr(patches, A, "from_config", wrap("from_config"))
    _patch_attr(patches, MU, "safe_open", safe_open)
    try:
        yield rec
    finally:
        _restore(patches)


@contextlib.contextmanager
def fallback_recording():
    """Record the steps of load_valuehead_model's TRL fallback: is_trl_available, the causal LM / vision2seq
    from_pretrained, TRL's value-head from_pretrained; "flash_attention_2" is built as "eager", so that a fallback runs
    as it would on GPU.  Restored afterwards."""
    import verl.utils.model as VM
    from transformers import AutoModelForCausalLM

    rec, patches = [], []

    def recorder(label, original):
        def call(*args, **kwargs):
            rec.append(label)
            if kwargs.get("attn_implementation") == "flash_attention_2":
                kwargs["attn_implementation"] = "eager"
            return original(*args, **kwargs)

        return call

    _patch_attr(patches, VM, "is_trl_available", recorder("is_trl_available", VM.is_trl_available))
    targets = [AutoModelForCausalLM]
    if VM.AutoModelForVision2Seq is not None and VM.AutoModelForVision2Seq is not AutoModelForCausalLM:
        targets.append(VM.AutoModelForVision2Seq)
    try:
        from trl.experimental.ppo import AutoModelForCausalLMWithValueHead as value_head  # type: ignore
    except ImportError:
        try:
            from trl import AutoModelForCausalLMWithValueHead as value_head  # type: ignore
        except ImportError:
            value_head = None
    if value_head is not None:
        targets.append(value_head)
    for cls in targets:
        _patch_attr(patches, cls, "from_pretrained", recorder(f"{cls.__name__}.from_pretrained", cls.from_pretrained))
    try:
        yield rec
    finally:
        _restore(patches)


def make_checkpoint(path, seed=1):
    """A tiny Qwen3_5ForConditionalGeneration saved in bf16 (the C0 checkpoint's dtype on disk)."""
    from transformers import Qwen3_5ForConditionalGeneration

    torch.manual_seed(seed)
    lm = Qwen3_5ForConditionalGeneration._from_config(tiny_config(), attn_implementation="eager",
                                                      torch_dtype=torch.float32)
    lm.to(torch.bfloat16).save_pretrained(path)
    return str(path)


def engine_for(path, model_type="value_model", mesh=None):
    """FSDPEngine without __init__ (as verl's test_fsdp_build_module_meta_ranks_on_cpu.py), the critic's settings:
    fp32 build (model_dtype null, training), FSDP2, no mixed-precision section (param bf16 / reduce fp32)."""
    from verl.workers.engine.fsdp.transformer_impl import FSDPEngine

    engine = object.__new__(FSDPEngine)
    engine.engine_config = _Cfg(model_dtype=None, forward_only=False, mixed_precision=None, strategy="fsdp2",
                                wrap_policy={}, reshard_after_forward=True, offload_policy=False,
                                forward_prefetch=False, use_orig_params=False)
    engine.model_config = SimpleNamespace(hf_config=tiny_config(), model_type=model_type, local_path=path,
                                          trust_remote_code=False, use_liger=False, fused_kernel_options=None,
                                          use_fused_kernels=False, enable_gradient_checkpointing=False,
                                          lora_rank=0, enable_activation_offload=False)
    engine.device_mesh = mesh
    engine.use_remove_padding = True
    engine.ulysses_sequence_parallel_size = 1
    return engine


def reference_value_model(path, seed=SEED):
    """A plain from_pretrained of the value model (fp32), seeded as the source rank's build."""
    from transformers import AutoModelForTokenClassification

    torch.manual_seed(seed)
    return AutoModelForTokenClassification.from_pretrained(path, config=tiny_config(), torch_dtype=torch.float32,
                                                           attn_implementation="eager")


def param_summary(module):
    params = list(module.named_parameters())
    return {"class": type(module).__name__, "params": len(params),
            "meta_params": sum(p.is_meta for _, p in params),
            "nonmeta_param_bytes": sum(p.numel() * p.element_size() for _, p in params if not p.is_meta),
            "dtypes": sorted({str(p.dtype) for _, p in params}), "training": module.training}


def config_state(cfg):
    """A config's serialised form and the fields from_config / from_pretrained set: attention implementation and
    dtype at the root and the text / vision sub-configs."""
    return {"json": cfg.to_json_string(),
            **{f"{part}_{field}": str(getattr(c, field))
               for part, c in (("root", cfg), ("text", cfg.text_config), ("vision", cfg.vision_config))
               for field in ("_attn_implementation", "dtype")}}


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    return make_checkpoint(tmp_path_factory.mktemp("tiny_qwen35_bf16"))


@pytest.fixture
def no_class_patch(monkeypatch):
    """verl's qwen3_5 monkey patch off (process-wide class patches; not part of what is tested)."""
    import verl.workers.engine.fsdp.transformer_impl as TI

    calls = []
    monkeypatch.setattr(TI, "apply_monkey_patch", lambda **kw: calls.append(kw))
    return calls


def build_on_rank(monkeypatch, path, rank, model_type="value_model"):
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: rank)
    engine = engine_for(path, model_type=model_type)
    with recording() as rec:
        torch.manual_seed(SEED)
        module = engine._build_module()
    return module, rec, engine.model_config.hf_config


# ---- one process ----

def test_only_the_source_rank_reads_the_checkpoint(checkpoint, monkeypatch, no_class_patch):
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForTokenClassification

    other, rec1, cfg1 = build_on_rank(monkeypatch, checkpoint, rank=1)
    source, rec0, cfg0 = build_on_rank(monkeypatch, checkpoint, rank=0)
    s1, s0 = param_summary(other), param_summary(source)
    print(f"[critic meta] rank 1: {s1}, calls {len(rec1['from_pretrained'])} from_pretrained / "
          f"{len(rec1['from_config'])} from_config / {len(rec1['safe_open'])} safe_open")
    print(f"[critic meta] rank 0: {s0}, calls {len(rec0['from_pretrained'])} from_pretrained / "
          f"{len(rec0['from_config'])} from_config / {len(rec0['safe_open'])} safe_open")
    m1, m0 = config_state(other.config), config_state(source.config)
    print("[critic meta] module config (rank 1 / rank 0):", {k: (m1[k], m0[k]) for k in m1 if k != "json"})
    # rank 1: no read, meta parameters only, the source rank's class / dtype / attention implementation asked for
    assert rec1["from_pretrained"] == [] and rec1["safe_open"] == [], rec1
    assert len(rec1["from_config"]) == 1
    assert type(other) is Qwen3_5ForTokenClassification and type(source) is Qwen3_5ForTokenClassification
    assert s1["meta_params"] == s1["params"] == s0["params"] and s1["nonmeta_param_bytes"] == 0
    assert s1["dtypes"] == s0["dtypes"] == ["torch.float32"]
    assert other.score.out_features == source.score.out_features == 1
    assert [n for n, _ in other.named_parameters()] == [n for n, _ in source.named_parameters()]
    assert rec1["from_config"][0]["attn_implementation"] == rec0["from_pretrained"][0]["attn_implementation"] \
        == "flash_attention_2"
    assert rec1["from_config"][0]["dtype"] == rec0["from_pretrained"][0]["dtype"] == "torch.float32"
    assert rec1["from_config"][0]["dtype_kwarg"] == ["dtype"]  # not the deprecated torch_dtype
    assert rec0["from_pretrained"][0]["dtype_kwarg"] == ["torch_dtype"]  # the source rank's call as before
    # both ranks leave the same mode and the engine's config alike; rank 1 built on a copy of it
    assert s1["training"] is False and s0["training"] is False
    assert rec1["from_config"][0]["config_id"] != id(cfg1) and other.config is not cfg1
    assert config_state(cfg1) == config_state(cfg0)
    assert m1["root__attn_implementation"] == m0["root__attn_implementation"]
    assert m1["text__attn_implementation"] == m0["text__attn_implementation"]
    assert m1["vision__attn_implementation"] == m0["vision__attn_implementation"]
    # rank 0: reads the checkpoint (as before), its backbone equals it
    assert len(rec0["from_pretrained"]) == 1 and rec0["from_config"] == [] and len(rec0["safe_open"]) >= 1
    assert s0["meta_params"] == 0
    from transformers import Qwen3_5ForConditionalGeneration

    lm = Qwen3_5ForConditionalGeneration.from_pretrained(checkpoint, config=tiny_config(), torch_dtype=torch.float32)
    lm_sd, sd = lm.state_dict(), source.state_dict()
    shared = [k for k in sd if not k.startswith("score.")]
    assert shared and all(k in lm_sd for k in shared), sorted(set(shared) - set(lm_sd))[:5]
    assert all(torch.equal(sd[k], lm_sd[k]) for k in shared)  # the bf16 checkpoint upcast, as before the patch
    assert no_class_patch  # the engine applied its (stubbed) monkey patch on both ranks, as before


def test_a_from_config_failure_raises_on_that_rank_without_the_trl_fallback(checkpoint, monkeypatch, no_class_patch,
                                                                              caplog):
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)
    caplog.set_level(logging.ERROR)
    outcome = {"raised": None}
    with recording(fail_from_config=True) as rec, fallback_recording() as fallback:
        torch.manual_seed(SEED)
        try:
            module = engine_for(checkpoint)._build_module()
            outcome["built"] = type(module).__name__
        except Exception as e:  # the outcome is printed first (negative controls), then checked
            outcome["raised"] = e
    err = outcome["raised"]
    logged = [r for r in caplog.records if r.levelno >= logging.ERROR and "rank 1" in r.getMessage()
              and "from_config failed" in r.getMessage()]
    print(f"[critic meta] rank 1 from_config failure: outcome {type(err).__name__ if err else outcome}: {err}; "
          f"fallback steps {fallback}; calls {[(k, len(v)) for k, v in rec.items()]}; error lines logged {len(logged)}")
    assert len(rec["from_config"]) == 1, rec
    assert isinstance(err, RuntimeError), outcome
    msg = str(err)
    assert "rank 1" in msg and "from_config" in msg and INJECTED in msg, msg
    assert "AutoModelForCausalLMWithValueHead" not in msg  # the Z4 pass line greps the logs for that class name
    assert isinstance(err.__cause__, ValueError) and str(err.__cause__) == INJECTED
    assert fallback == [] and rec["from_pretrained"] == [] and rec["safe_open"] == []
    # the error line carries the injected failure's traceback (logger.error(..., exc_info=True)): exc_info=False would
    # leave record.exc_info False, not None, so the check is on the exception itself
    assert len(logged) == 1, [r.getMessage() for r in caplog.records]
    assert logged[0].exc_info and logged[0].exc_info[1] is err.__cause__, logged[0].exc_info


def test_load_valuehead_model_default_call_unchanged(checkpoint):
    """Without from_config (every caller but the engine's other ranks): from_pretrained with the same arguments."""
    from verl.utils.model import load_valuehead_model

    cfg = tiny_config()
    with recording() as rec:
        torch.manual_seed(SEED)
        model = load_valuehead_model(checkpoint, torch.float32, cfg, False)
    assert len(rec["from_pretrained"]) == 1 and rec["from_config"] == []
    call = rec["from_pretrained"][0]
    assert call["attn_implementation"] == "flash_attention_2" and call["dtype"] == "torch.float32"
    assert call["dtype_kwarg"] == ["torch_dtype"]
    assert call["path"] == checkpoint and call["config_id"] == id(cfg)
    ref = reference_value_model(checkpoint)
    a, b = model.state_dict(), ref.state_dict()
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


def test_the_language_model_branch_is_unchanged(checkpoint, monkeypatch, no_class_patch):
    """The actor / reference branch (model_type language_model, verl 0a7be8b): rank 1 from_config on meta, rank 0
    from_pretrained equal to the checkpoint; the value-model loader is not called."""
    import verl.utils.model as VM
    from transformers import Qwen3_5ForConditionalGeneration

    called = []
    monkeypatch.setattr(VM, "load_valuehead_model", lambda *a, **kw: called.append(kw))
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 1)
    other = engine_for(checkpoint, model_type="language_model")._build_module()
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: 0)
    source = engine_for(checkpoint, model_type="language_model")._build_module()
    assert called == []
    assert type(other) is type(source) is Qwen3_5ForConditionalGeneration
    assert all(p.is_meta for p in other.parameters()) and not any(p.is_meta for p in source.parameters())
    ref = Qwen3_5ForConditionalGeneration.from_pretrained(checkpoint, config=tiny_config(), torch_dtype=torch.float32)
    a, b = source.state_dict(), ref.state_dict()
    assert a.keys() == b.keys() and all(torch.equal(a[k], b[k]) for k in a)


# ---- two processes: the engine's build and FSDP2 sync on a CPU mesh (gloo) ----

def _rank_main(rank, world, path, store, out_dir, seed, fail_rank, timeout_s):
    """One rank: _build_module, then _build_fsdp_module (FSDP2: apply_fsdp2 + fsdp2_load_full_state_dict), then the
    full state dict and the buffers, saved to out_dir/rank<r>.pt; the record (or the exception, which is not raised
    again) to out_dir/rank<r>.json.  ``fail_rank``: that rank's from_config raises."""
    torch.set_num_threads(1)
    import torch.distributed as dist

    install_in_this_process()
    # loopback TCP store (a file:// store needs flock, which the shared file system refuses with EAGAIN)
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    dist.init_process_group("gloo", init_method=store, rank=rank, world_size=world,
                            timeout=timedelta(seconds=timeout_s))
    record = {"rank": rank, "stage": "build", "error": None}
    start = time.monotonic()
    try:
        from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor import DTensor

        import verl.utils.fsdp_utils as FU
        import verl.workers.engine.fsdp.transformer_impl as TI

        FU.get_device_id = lambda: "cpu"  # no accelerator: fsdp2_load_full_state_dict moves the module here
        TI.apply_monkey_patch = lambda **kw: None
        mesh = init_device_mesh("cpu", (world,), mesh_dim_names=("fsdp",))  # create_device_mesh, fsdp_size -1
        engine = engine_for(path, mesh=mesh)
        with recording(fail_from_config=rank == fail_rank) as rec, fallback_recording() as fallback:
            record["calls"], record["fallback"] = rec, fallback
            torch.manual_seed(seed + rank)
            module = engine._build_module()
        record["before_sync"] = param_summary(module)
        record["module_config"] = {k: v for k, v in config_state(module.config).items() if k != "json"}
        record["stage"] = "sync"
        module = engine._build_fsdp_module(module)
        record["sharded_params"] = sum(isinstance(p, DTensor) for p in module.parameters())
        full = get_model_state_dict(module, options=StateDictOptions(full_state_dict=True))
        full = {k: (v.full_tensor() if isinstance(v, DTensor) else v).detach().clone() for k, v in full.items()}
        buffers = {n: b.detach().clone() for n, b in module.named_buffers()}
        torch.save({"state": full, "buffers": buffers}, Path(out_dir) / f"rank{rank}.pt")
        record["stage"] = "done"
    except Exception as e:
        record["error"] = {"type": type(e).__name__, "message": str(e)[:600]}
    record["seconds"] = round(time.monotonic() - start, 2)
    calls = record.pop("calls", {})
    record["calls"] = {k: len(v) for k, v in calls.items()}
    record["attn_requested"] = [c["attn_implementation"] for c in calls.get("from_pretrained", []) +
                                calls.get("from_config", [])]
    record["fallback"] = list(record.get("fallback", []))
    Path(out_dir, f"rank{rank}.json").write_text(json.dumps(record, indent=1))
    try:
        dist.destroy_process_group()
    except Exception:
        pass


def run_two_ranks(path, work_dir, seed=SEED, fail_rank=None, timeout_s=60):
    """Both ranks (spawned, 1 thread each) and their records.  Unless a rank failed, each rank's full state is also
    compared with a plain from_pretrained of rank 0's seed."""
    import socket

    import torch.multiprocessing as tmp

    work = Path(work_dir)
    work.mkdir(parents=True, exist_ok=True)
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        store = f"tcp://127.0.0.1:{s.getsockname()[1]}"
    start = time.monotonic()
    tmp.start_processes(_rank_main, args=(2, path, store, str(work), seed, fail_rank, timeout_s), nprocs=2,
                        join=True, start_method="spawn")
    out = {"seconds": round(time.monotonic() - start, 2), "ranks": [], "compare": []}
    out["ranks"] = [json.loads((work / f"rank{r}.json").read_text()) for r in range(2)]
    if any(r["stage"] != "done" for r in out["ranks"]):
        return out
    ref = reference_value_model(path, seed)
    ref_state = {k: v.detach() for k, v in ref.state_dict().items()}
    ref_buffers = {n: b.detach() for n, b in ref.named_buffers()}
    other_seed = reference_value_model(path, seed + 1).state_dict()["score.weight"]
    out["control_other_seed_head_differs"] = not torch.equal(other_seed, ref_state["score.weight"])
    for r in range(2):
        got = torch.load(work / f"rank{r}.pt")
        state, bufs = got["state"], got["buffers"]
        diff_keys = sorted(set(state) ^ set(ref_state))
        unequal = [k for k in ref_state if k in state and not torch.equal(state[k], ref_state[k])]
        dtype_bad = [k for k in state if state[k].dtype != torch.float32]
        buf_unequal = [n for n in ref_buffers if n not in bufs or not torch.equal(bufs[n], ref_buffers[n])]
        max_abs = max((float((state[k] - ref_state[k]).abs().max()) for k in ref_state if k in state), default=0.0)
        out["compare"].append({"rank": r, "tensors": len(state), "key_mismatch": diff_keys, "unequal": unequal[:10],
                               "n_unequal": len(unequal), "non_fp32": dtype_bad[:5], "buffers": len(bufs),
                               "buffers_unequal": buf_unequal, "max_abs_diff": max_abs,
                               "score_weight_equal": torch.equal(state["score.weight"], ref_state["score.weight"])})
    return out


def test_two_ranks_after_the_fsdp2_sync_equal_a_plain_from_pretrained(checkpoint, tmp_path):
    result = run_two_ranks(checkpoint, tmp_path / "two_ranks")
    print("[critic meta] two ranks:", json.dumps(result))
    r0, r1 = result["ranks"]
    assert r0["stage"] == r1["stage"] == "done", result
    assert r0["calls"]["from_pretrained"] == 1 and r0["calls"]["safe_open"] >= 1 and r0["calls"]["from_config"] == 0
    assert r1["calls"] == {"from_pretrained": 0, "from_config": 1, "safe_open": 0}
    assert r0["fallback"] == r1["fallback"] == []
    assert r1["before_sync"]["meta_params"] == r1["before_sync"]["params"] and r0["before_sync"]["meta_params"] == 0
    assert r0["before_sync"]["training"] is False and r1["before_sync"]["training"] is False
    assert r0["attn_requested"] == r1["attn_requested"] == ["flash_attention_2"]
    assert r0["sharded_params"] == r1["sharded_params"] == r0["before_sync"]["params"]
    assert result["control_other_seed_head_differs"]  # rank 1's own seed would give another score head
    for c in result["compare"]:
        assert c["key_mismatch"] == [] and c["n_unequal"] == 0 and c["non_fp32"] == [], c
        assert c["buffers_unequal"] == [] and c["score_weight_equal"] and c["max_abs_diff"] == 0.0, c


def test_two_ranks_a_from_config_failure_raises_at_its_rank(checkpoint, tmp_path):
    result = run_two_ranks(checkpoint, tmp_path / "two_ranks_fail", fail_rank=1, timeout_s=60)
    print("[critic meta] two ranks, rank 1 from_config fails:", json.dumps(result))
    r0, r1 = result["ranks"]
    # rank 1: the RuntimeError in _build_module (before any collective), no TRL fallback, nothing read
    assert r1["stage"] == "build" and r1["error"] is not None and r1["error"]["type"] == "RuntimeError", r1
    msg = r1["error"]["message"]
    assert "rank 1" in msg and "from_config" in msg and INJECTED in msg, msg
    assert r1["fallback"] == [] and r1["calls"] == {"from_pretrained": 0, "from_config": 1, "safe_open": 0}, r1
    # rank 0 built as before and then fails in the sync, whose peer is gone (gloo); it does not finish
    assert r0["calls"]["from_pretrained"] == 1 and r0["fallback"] == [], r0
    assert r0["stage"] == "sync" and r0["error"] is not None, r0
    assert result["seconds"] < 300, result
