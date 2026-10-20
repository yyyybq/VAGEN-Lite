#!/usr/bin/env python3
"""Load one start weight as the vision-language backbone of StarVLA's QwenOFT model and run a fake batch.

Run in the starVLA Python environment with third_party/starVLA on PYTHONPATH, one weight per process (so peak
memory is per weight):

    python downstream/load_test/load_in_starvla.py --name RS --path /hf/dir --config template.yaml --out RS.json

Reads the weight only by path (StarVLA's own ``from_pretrained``); writes nothing next to it.  Records:
  * the loading info transformers reports (missing / unexpected / mismatched keys) for StarVLA's own call;
  * a full comparison of every tensor in the safetensors files with the loaded model (after the key renaming
    transformers applies), so "loaded" means "equal to the file", not just "no error";
  * dtypes and parameter counts of the backbone and the new action head, and a hash of the action head's
    initial weights and of the torch RNG state when it is built (same seed -> same head for every start?);
  * a forward pass (training path, L1 action loss) and ``predict_action`` (inference path) on a fake batch of
    three 224x224 camera views, with peak GPU memory after loading and after each pass;
  * as extra information (not part of pass / fail), whether one backward pass fits on one card.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np


def sha(t) -> str:
    import torch

    data = t.detach().float().cpu().contiguous().numpy().tobytes() if isinstance(t, torch.Tensor) else bytes(t)
    return hashlib.sha256(data).hexdigest()[:16]


def norm_key(key: str) -> str:
    """One name for both Qwen2.5-VL key layouts (old checkpoints: model.layers / visual; new: model.language_model /
    model.visual), with StarVLA's wrapper prefix removed."""
    for prefix in ("qwen_vl_interface.model.",):
        if key.startswith(prefix):
            key = key[len(prefix):]
    if key.startswith("model.language_model."):
        return "LM." + key[len("model.language_model."):]
    if key.startswith("model.visual."):
        return "VIS." + key[len("model.visual."):]
    if key.startswith("visual."):
        return "VIS." + key[len("visual."):]
    if key.startswith("model."):
        return "LM." + key[len("model."):]
    return key


def compare_with_files(model, root: Path) -> dict:
    """Every tensor of the export against the loaded backbone (cast to the loaded dtype)."""
    import torch
    from safetensors import safe_open

    params = {norm_key(k): v for k, v in model.state_dict().items()}
    index = root / "model.safetensors.index.json"
    shards = sorted(set(json.loads(index.read_text())["weight_map"].values())) if index.is_file() else ["model.safetensors"]
    equal, differ, absent, max_abs = 0, [], [], 0.0
    file_keys = set()
    for name in shards:
        with safe_open(str(root / name), framework="pt", device="cpu") as handle:
            for key in handle.keys():
                nk = norm_key(key)
                file_keys.add(nk)
                if nk not in params:
                    absent.append(key)
                    continue
                target = params[nk]
                source = handle.get_tensor(key).to(dtype=target.dtype)
                if source.shape != target.shape:
                    differ.append(f"{key}: shape {tuple(source.shape)} vs {tuple(target.shape)}")
                    continue
                if torch.equal(source, target.detach().cpu()):
                    equal += 1
                else:
                    diff = (source.float() - target.detach().cpu().float()).abs().max().item()
                    max_abs = max(max_abs, diff)
                    differ.append(f"{key}: max abs diff {diff:.3g}")
    not_in_file = sorted(k for k in params if k not in file_keys)
    return {"file_tensors": len(file_keys), "equal": equal, "differ": differ[:20], "differ_count": len(differ),
            "max_abs_diff": max_abs, "file_keys_absent_in_model": absent[:20], "absent_count": len(absent),
            "model_keys_not_in_file": not_in_file[:20], "not_in_file_count": len(not_in_file)}


def dtype_counts(module) -> dict:
    out = {}
    for p in module.parameters():
        out[str(p.dtype)] = out.get(str(p.dtype), 0) + p.numel()
    return out


def fake_examples(batch: int, horizon: int, action_dim: int, seed: int):
    from PIL import Image

    rng = np.random.default_rng(seed)
    examples = []
    for i in range(batch):
        views = [Image.fromarray(rng.integers(0, 256, (224, 224, 3), dtype=np.uint8)) for _ in range(3)]
        examples.append({"image": views, "lang": f"open the drawer next to the sink ({i})",
                         "action": rng.uniform(-1, 1, (horizon, action_dim)).astype(np.float32)})
    return examples


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--name", required=True)
    ap.add_argument("--path", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True, help="StarVLA QwenOFT yaml; base_vlm is replaced")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--seed", type=int, default=1234, help="torch seed set right before the model is built")
    ap.add_argument("--no-compare", action="store_true", help="skip the tensor-by-tensor file comparison")
    args = ap.parse_args()

    import torch
    from omegaconf import OmegaConf

    report = {"name": args.name, "path": str(args.path), "status": "running", "seed": args.seed,
              "host": os.uname().nodename, "torch": torch.__version__,
              "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.out.write_text(json.dumps(report, indent=1, default=str) + "\n")

    save()
    try:
        import transformers
        from transformers import Qwen2_5_VLForConditionalGeneration

        report["transformers"] = transformers.__version__
        from starVLA.model.framework.share_tools import apply_config_compat
        from starVLA.training.trainer_utils.config_tracker import wrap_config

        cfg = OmegaConf.load(args.config)
        cfg.framework.qwenvl.base_vlm = str(args.path)
        cfg.framework.qwenvl.attn_implementation = "sdpa"
        cfg = wrap_config(apply_config_compat(cfg))  # as train_starvla.py does before build_framework

        loading = {}
        original = Qwen2_5_VLForConditionalGeneration.from_pretrained.__func__

        def from_pretrained(cls, *a, **kw):  # StarVLA's own call, plus the loading report
            kw["output_loading_info"] = True
            model, info = original(cls, *a, **kw)
            loading.update({k: sorted(str(x) for x in v) for k, v in info.items()})
            loading["kwargs"] = {k: str(v) for k, v in kw.items()}
            return model

        Qwen2_5_VLForConditionalGeneration.from_pretrained = classmethod(from_pretrained)

        import starVLA.model.framework.VLM4A.QwenOFT as qwen_oft

        head_rng = {}
        original_head = qwen_oft.get_action_model

        def get_action_model(config=None):  # the RNG state the action head is initialised from
            head_rng["cpu_rng_sha"] = hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()[:16]
            return original_head(config=config)

        qwen_oft.get_action_model = get_action_model
        from starVLA.model.framework.base_framework import build_framework

        torch.manual_seed(args.seed)
        t0 = time.time()
        model = build_framework(cfg)
        report["build_seconds"] = round(time.time() - t0, 1)
        report["loading_info"] = loading
        report["action_head_rng"] = head_rng
        vlm = model.qwen_vl_interface.model
        head = model.action_model
        report["params"] = {
            "total": sum(p.numel() for p in model.parameters()),
            "backbone": sum(p.numel() for p in vlm.parameters()),
            "action_head": sum(p.numel() for p in head.parameters()),
            "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
        }
        report["dtypes"] = {"backbone": dtype_counts(vlm), "action_head": dtype_counts(head)}
        action_cfg = model.config.framework.action_model  # QwenOFT's merged config (sets action_hidden_dim)
        report["hidden_size"] = int(vlm.config.hidden_size)
        report["action_hidden_dim"] = int(action_cfg.action_hidden_dim)
        report["action_token_id"] = int(model.action_token_id)
        report["action_head_init_sha"] = sha(torch.cat([p.detach().flatten().float() for p in head.parameters()]))
        save()
        if not args.no_compare:
            t0 = time.time()
            report["file_comparison"] = compare_with_files(vlm, args.path)
            report["file_comparison"]["seconds"] = round(time.time() - t0, 1)
            save()

        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        model = model.to("cuda")
        torch.cuda.synchronize()
        report["to_cuda_seconds"] = round(time.time() - t0, 1)
        report["gpu_mem_gb"] = {"after_load": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
        horizon = int(action_cfg.action_horizon)
        action_dim = int(action_cfg.action_dim)
        examples = fake_examples(args.batch, horizon, action_dim, args.seed)

        model.train()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        out = model.forward(examples)  # the training path: L1 loss of the action head on the fake actions
        loss = out["action_loss"]
        torch.cuda.synchronize()
        report["train_forward"] = {
            "batch": args.batch, "loss": float(loss.item()), "loss_finite": bool(torch.isfinite(loss).item()),
            "seconds": round(time.time() - t0, 2),
            "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        }
        save()
        # Extra information, not part of pass/fail: does one backward pass fit on one card in the loaded dtype?
        try:
            t0 = time.time()
            loss.backward()
            torch.cuda.synchronize()
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            report["backward_info"] = {
                "fits_one_card": True, "params_with_grad": sum(g.numel() for g in grads),
                "grads_finite": bool(all(torch.isfinite(g).all().item() for g in grads)),
                "seconds": round(time.time() - t0, 2),
                "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            }
            del grads
        except torch.cuda.OutOfMemoryError as exc:
            report["backward_info"] = {"fits_one_card": False, "error": str(exc).splitlines()[0][:300]}
        model.zero_grad(set_to_none=True)
        del out, loss
        torch.cuda.empty_cache()
        save()

        model.eval()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        pred = model.predict_action(examples=[{k: v for k, v in examples[0].items() if k != "action"}])
        actions = pred["normalized_actions"]
        report["predict_action"] = {
            "shape": list(actions.shape), "finite": bool(np.isfinite(actions).all()),
            "seconds": round(time.time() - t0, 2),
            "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        }
        ok = (not loading.get("missing_keys") and not loading.get("unexpected_keys")
              and not loading.get("mismatched_keys")
              and report["train_forward"]["loss_finite"] and report["predict_action"]["finite"]
              and list(actions.shape) == [1, horizon, action_dim])
        if "file_comparison" in report:
            fc = report["file_comparison"]
            ok = ok and fc["differ_count"] == 0 and fc["absent_count"] == 0 and fc["equal"] == fc["file_tensors"]
        report["status"] = "ok" if ok else "loaded_with_problems"
    except Exception as exc:
        report["status"] = "error"
        report["error"] = f"{type(exc).__name__}: {exc}"
        report["traceback"] = traceback.format_exc()
    save()
    print(json.dumps({k: report.get(k) for k in ("name", "status", "params", "dtypes", "gpu_mem_gb", "error")},
                     default=str))
    return 0 if report["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
