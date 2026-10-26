"""Runtime compat for SenseNova-U1 under VERL (local /tmp; no NFS writes)."""
from __future__ import annotations

from pathlib import Path

import importlib.util
import sys
import types

# Prefer package import; keep file-load fallback for /tmp external_lib.
_HERE = Path(__file__).resolve().parent if "__file__" in globals() else None
REGISTER_PATH = str((_HERE / "sensenova_u1_register.py") if _HERE else "/mnt/umm/users/yinbaiqiao/VAGEN-Lite/vagen/models/sensenova_u1_register.py")
if "/tmp" not in sys.path:
    sys.path.insert(0, "/tmp")


def _patch_neo_chat_config_compat(config_cls) -> None:
    if not isinstance(config_cls, type):
        return
    if not any(b.__name__ == "PretrainedConfig" for b in getattr(config_cls, "__mro__", ())):
        return
    if "_vagen_text_config_compat" in getattr(config_cls, "__dict__", {}):
        return
    if "text_config" not in config_cls.__dict__:
        config_cls.text_config = property(lambda self: self.llm_config)
    for attr in (
        "num_attention_heads",
        "num_key_value_heads",
        "hidden_size",
        "num_hidden_layers",
        "vocab_size",
    ):
        if attr not in config_cls.__dict__:
            setattr(
                config_cls,
                attr,
                property(lambda self, a=attr: getattr(self.llm_config, a)),
            )
    config_cls._vagen_text_config_compat = True


def _ensure_config_compat(cfg) -> None:
    if cfg is None or getattr(cfg, "model_type", None) != "neo_chat":
        return
    _patch_neo_chat_config_compat(type(cfg))
    # Enable vLLM V1 3D M-RoPE position buffers for MoT-und indexes.
    # Actual RoPE math is U1MotRotaryEmbedding (dual-base); this only flips uses_mrope.
    if getattr(cfg, "rope_scaling", None) is None:
        llm = getattr(cfg, "llm_config", cfg)
        head_dim = int(getattr(llm, "head_dim", 128))
        cfg.rope_scaling = {
            "rope_type": "default",
            "mrope_section": [head_dim // 4, head_dim // 8, head_dim // 8],
        }
        print(
            f"[u1_neo_compat] Injected rope_scaling mrope_section={cfg.rope_scaling['mrope_section']}",
            flush=True,
        )


def _is_wrapped(fn) -> bool:
    return isinstance(fn, types.FunctionType) and fn.__dict__.get("_u1_compat", False)


def _patch_monkey_patch() -> None:
    from verl.models.transformers import monkey_patch as mp

    orig = mp.apply_monkey_patch
    if _is_wrapped(orig):
        wrapped = orig
    else:
        def wrapped(model, *args, **kwargs):
            _ensure_config_compat(getattr(model, "config", None))
            return orig(model, *args, **kwargs)

        wrapped._u1_compat = True
        mp.apply_monkey_patch = wrapped
        print("[u1_neo_compat] Patched monkey_patch.apply_monkey_patch")

    # Direct rewrite of known call sites
    for modname in (
        "verl.workers.fsdp_workers",
        "verl.models.transformers.monkey_patch",
    ):
        try:
            mod = importlib.import_module(modname)
            setattr(mod, "apply_monkey_patch", wrapped)
            print(f"[u1_neo_compat] Set {modname}.apply_monkey_patch")
        except Exception as e:
            print(f"[u1_neo_compat] skip {modname}: {e}")


import importlib

try:
    import vagen.models.sensenova_u1_register as _reg_mod
    sys.modules["sensenova_u1_register_real"] = _reg_mod
    mod = _reg_mod
except Exception:
    spec = importlib.util.spec_from_file_location("sensenova_u1_register_real", REGISTER_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sensenova_u1_register_real"] = mod
    spec.loader.exec_module(mod)

from sensenova_u1.models.neo_unify import NEOChatConfig  # noqa: E402

_patch_neo_chat_config_compat(NEOChatConfig)

# Dense SenseNova-U1-8B has Qwen3DecoderLayer, not MoE layers. VERL FSDP wrap
# fails if any name in _no_split_modules is missing from the module tree.
try:
    from sensenova_u1.models.neo_unify.modeling_neo_chat import NEOChatModel

    NEOChatModel._no_split_modules = ["Qwen3DecoderLayer", "NEOVisionModel"]
except Exception as e:
    print(f"[u1_neo_compat] NEOChatModel._no_split_modules skip: {e}")
for cls_name in ("SenseNovaU1ForCausalLMAdapter", "SenseNovaU1ForTokenClassification"):
    cls = getattr(mod, cls_name, None)
    if cls is not None:
        cls._no_split_modules = ["Qwen3DecoderLayer", "NEOVisionModel"]
        print(f"[u1_neo_compat] {cls_name}._no_split_modules -> {cls._no_split_modules}")

_patch_monkey_patch()

try:
    from transformers import AutoConfig

    _orig_ac = AutoConfig.from_pretrained.__func__

    def _patched_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        cfg = _orig_ac(cls, pretrained_model_name_or_path, *args, **kwargs)
        _ensure_config_compat(cfg)
        return cfg

    AutoConfig.from_pretrained = classmethod(_patched_from_pretrained)
    print("[u1_neo_compat] Patched AutoConfig.from_pretrained")
except Exception as e:
    print(f"[u1_neo_compat] AutoConfig patch skipped: {e}")


# Ensure SenseNova-U1 vLLM architecture is registered in this process.
try:
    import vagen.models.sensenova_u1_vllm  # noqa: F401
    print("[u1_neo_compat] Imported sensenova_u1_vllm")
except Exception as e:
    print(f"[u1_neo_compat] sensenova_u1_vllm import: {e}")


def _filter_u1_rollout_weight_keys(params):
    """Drop MoT gen / FM modules before FSDP full_tensor() gather to GPU."""
    if not isinstance(params, dict):
        return params
    kept = {
        k: v
        for k, v in params.items()
        if ("mot_gen" not in k) and (not str(k).startswith("fm_modules")) and (".fm_modules." not in str(k))
    }
    if len(kept) != len(params):
        print(
            f"[u1_neo_compat] Filtered rollout weights {len(params)} -> {len(kept)} "
            f"(dropped mot_gen/fm_modules)",
            flush=True,
        )
    return kept


def _patch_convert_weight_keys() -> None:
    try:
        import verl.utils.model as model_utils
    except Exception as e:
        print(f"[u1_neo_compat] convert_weight_keys import failed: {e}", flush=True)
        return
    orig = model_utils.convert_weight_keys
    if getattr(orig, "_u1_compat", False):
        return

    def wrapped(params, *args, **kwargs):
        out = orig(params, *args, **kwargs)
        return _filter_u1_rollout_weight_keys(out)

    wrapped._u1_compat = True
    model_utils.convert_weight_keys = wrapped
    try:
        import verl.workers.fsdp_workers as fw
        if hasattr(fw, "convert_weight_keys"):
            fw.convert_weight_keys = wrapped
            print("[u1_neo_compat] Patched fsdp_workers.convert_weight_keys", flush=True)
    except Exception as e:
        print(f"[u1_neo_compat] fsdp_workers patch skip: {e}", flush=True)
    print("[u1_neo_compat] Patched convert_weight_keys filter", flush=True)


_patch_convert_weight_keys()


_ADAPTER_METHODS = (
    "forward",
    "_build_indexes",
    "_build_block_attention_mask",
    "_prepare_single_gen_target",
    "_predict_x_from_hidden",
    "_detach_past_key_values",
    "_compute_fm_aux_split",
    "compute_pending_fm_aux_loss",
)


def _promote_adapter_onto(cls, label: str) -> None:
    try:
        from sensenova_u1_register_real import SenseNovaU1ForCausalLMAdapter
    except Exception as e:
        print(f"[u1_neo_compat] promote import skip: {e}", flush=True)
        return
    if not isinstance(cls, type):
        return
    for name in _ADAPTER_METHODS:
        if hasattr(SenseNovaU1ForCausalLMAdapter, name):
            setattr(cls, name, getattr(SenseNovaU1ForCausalLMAdapter, name))
    cls._no_split_modules = ["Qwen3DecoderLayer", "NEOVisionModel"]
    print(f"[u1_neo_compat] Promoted adapter methods onto {label} ({cls})", flush=True)


def _promote_adapter_methods_onto_neochat() -> None:
    """HF trust_remote_code loads NEOChatModel via auto_map; promote onto all copies."""
    import os

    try:
        from sensenova_u1.models.neo_unify.modeling_neo_chat import NEOChatModel

        _promote_adapter_onto(NEOChatModel, "sensenova_u1 NEOChatModel")
    except Exception as e:
        print(f"[u1_neo_compat] package promote skip: {e}", flush=True)

    for modname, mod in list(sys.modules.items()):
        if mod is None or "neo_chat" not in modname.lower():
            continue
        cls = getattr(mod, "NEOChatModel", None)
        if cls is not None and getattr(cls, "__name__", "") == "NEOChatModel":
            _promote_adapter_onto(cls, modname)

    try:
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        model_dir = os.environ.get(
            "SENSENOVA_U1_MODEL_PATH",
            "/mnt/umm/users/yinbaiqiao/hf_cache/SenseNova-U1-8B-MoT-SFT",
        )
        dyn = get_class_from_dynamic_module("modeling_neo_chat.NEOChatModel", model_dir)
        _promote_adapter_onto(dyn, "dynamic_module NEOChatModel")
    except Exception as e:
        print(f"[u1_neo_compat] dynamic_module promote skip: {e}", flush=True)


def _patch_automodel_from_pretrained_promote() -> None:
    try:
        from transformers import AutoModelForCausalLM
    except Exception as e:
        print(f"[u1_neo_compat] AutoModel patch skip: {e}", flush=True)
        return
    orig = AutoModelForCausalLM.from_pretrained
    if getattr(orig, "_u1_compat", False):
        return

    def wrapped(*args, **kwargs):
        model = orig(*args, **kwargs)
        try:
            _promote_adapter_methods_onto_neochat()
        except Exception as e:
            print(f"[u1_neo_compat] post-load promote skip: {e}", flush=True)
        return model

    wrapped._u1_compat = True
    AutoModelForCausalLM.from_pretrained = wrapped
    print("[u1_neo_compat] Patched AutoModelForCausalLM.from_pretrained for promote", flush=True)


_promote_adapter_methods_onto_neochat()
_patch_automodel_from_pretrained_promote()


def _register_automodels_for_all_neochat_configs() -> None:
    """HF trust_remote_code uses transformers_modules NEOChatConfig (different id)."""
    try:
        from transformers import AutoModelForCausalLM, AutoModelForTokenClassification
        from sensenova_u1_register_real import (
            SenseNovaU1ForCausalLMAdapter,
            SenseNovaU1ForTokenClassification,
        )
    except Exception as e:
        print(f"[u1_neo_compat] AutoModel re-register import skip: {e}", flush=True)
        return

    cfgs = []
    try:
        from sensenova_u1.models.neo_unify import NEOChatConfig as PkgCfg
        cfgs.append(("package", PkgCfg))
    except Exception:
        pass
    for modname, mod in list(sys.modules.items()):
        if mod is None or "configuration_neo_chat" not in modname.lower():
            continue
        cls = getattr(mod, "NEOChatConfig", None)
        if cls is not None and getattr(cls, "__name__", "") == "NEOChatConfig":
            cfgs.append((modname, cls))
    try:
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        import os

        model_dir = os.environ.get(
            "SENSENOVA_U1_MODEL_PATH",
            "/mnt/umm/users/yinbaiqiao/hf_cache/SenseNova-U1-8B-MoT-SFT",
        )
        dyn = get_class_from_dynamic_module("configuration_neo_chat.NEOChatConfig", model_dir)
        cfgs.append(("dynamic_module", dyn))
    except Exception as e:
        print(f"[u1_neo_compat] dynamic config fetch skip: {e}", flush=True)

    seen = set()
    for label, cfg in cfgs:
        if cfg is None or id(cfg) in seen:
            continue
        seen.add(id(cfg))
        try:
            AutoModelForCausalLM.register(cfg, SenseNovaU1ForCausalLMAdapter, exist_ok=True)
            AutoModelForTokenClassification.register(
                cfg, SenseNovaU1ForTokenClassification, exist_ok=True
            )
            print(f"[u1_neo_compat] Registered AutoModels for {label}", flush=True)
        except Exception as e:
            print(f"[u1_neo_compat] register {label} skip: {e}", flush=True)


_register_automodels_for_all_neochat_configs()

print("[u1_neo_compat] Ready")
