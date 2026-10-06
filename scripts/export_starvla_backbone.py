#!/usr/bin/env python3
"""Export the VLM inside a trained QwenOFT policy for Active Spatial retention eval.

CPU operation. The 12-D action head is intentionally excluded; retain the full
StarVLA checkpoint separately for robot control. Does not merge architectures.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vagen.envs.robocasa.starvla_contract import checkpoint_config


def extract_backbone(state):
    prefix = "qwen_vl_interface.model."
    result = {key[len(prefix):]: tensor for key, tensor in state.items() if key.startswith(prefix)}
    if not result or not any(key.endswith("embed_tokens.weight") for key in result):
        raise ValueError("Checkpoint does not contain a QwenOFT VLM backbone")
    return result


def main():
    import torch
    from huggingface_hub import save_torch_state_dict

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    cfg, _ = checkpoint_config(args.ckpt)
    base = Path(cfg["framework"]["qwenvl"]["base_vlm"]).resolve(strict=True)
    model_cfg = json.loads((base / "config.json").read_text())
    if model_cfg.get("model_type") not in {"qwen2_5_vl", "qwen3_vl"}:
        raise ValueError("Only Qwen2.5-VL / Qwen3-VL exports are supported")
    if args.out.exists():
        raise FileExistsError(args.out)
    state = torch.load(args.ckpt, map_location="cpu", weights_only=True, mmap=True)
    backbone = extract_backbone(state)
    text_cfg = model_cfg.get("text_config", model_cfg)
    expected = (text_cfg["vocab_size"], text_cfg["hidden_size"])
    for name, value in backbone.items():
        if name.endswith("embed_tokens.weight") and tuple(value.shape) != expected:
            raise ValueError("Vocabulary changed during action training; source processor cannot be reused")
    args.out.mkdir(parents=True)
    save_torch_state_dict(backbone, args.out, max_shard_size="5GB", safe_serialization=True,
                          shared_tensors_to_discard=["lm_head.weight"] if model_cfg.get("tie_word_embeddings") else None)
    for path in base.iterdir():
        if path.is_file() and (path.suffix in {".json", ".txt", ".jinja", ".model"}) and not path.name.endswith(".index.json"):
            shutil.copy2(path, args.out / path.name)
    (args.out / "starvla_export.json").write_text(json.dumps({
        "source_policy": str(args.ckpt.resolve()), "source_vlm": str(base),
        "purpose": "Active Spatial retention evaluation after action fine-tuning",
        "backbone_tensors": len(backbone), "action_head_exported": False,
    }, indent=2) + "\n")
    print(args.out.resolve())


if __name__ == "__main__":
    main()
