#!/usr/bin/env python3
"""Audit Phase4B checkpoint completeness and prove FM parameters did not move."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "exps/vagen_active_spatial/protocol_only_prepost"
DEFAULT_BASE = Path("/mnt/umm/users/yinbaiqiao/hf_cache/SenseNova-U1-8B-MoT-SFT")


def load_weight_map(model_dir: Path) -> dict[str, str]:
    index = model_dir / "model.safetensors.index.json"
    if index.is_file():
        return json.loads(index.read_text(encoding="utf-8"))["weight_map"]
    single = model_dir / "model.safetensors"
    if single.is_file():
        with safe_open(single, framework="pt", device="cpu") as handle:
            return {key: single.name for key in handle.keys()}
    raise FileNotFoundError(f"no safetensors model/index under {model_dir}")


def is_fm_parameter(name: str) -> bool:
    return name.startswith("fm_modules.") or "_mot_gen" in name


def tensor_digest(model_dir: Path, selected_keys: list[str]) -> dict[str, Any]:
    weight_map = load_weight_map(model_dir)
    missing = sorted(set(selected_keys) - set(weight_map))
    if missing:
        raise RuntimeError(f"{model_dir}: missing {len(missing)} FM keys, first={missing[:5]}")
    by_file: dict[str, list[str]] = {}
    for key in selected_keys:
        by_file.setdefault(weight_map[key], []).append(key)
    aggregate = hashlib.sha256()
    per_module: dict[str, hashlib._Hash] = {}
    tensor_count = 0
    element_count = 0
    l2_sq = 0.0
    for filename in sorted(by_file):
        with safe_open(model_dir / filename, framework="pt", device="cpu") as handle:
            for key in sorted(by_file[filename]):
                tensor = handle.get_tensor(key).detach().contiguous()
                raw = tensor.view(torch.uint8).numpy().tobytes()
                header = f"{key}\0{tensor.dtype}\0{tuple(tensor.shape)}\0".encode()
                aggregate.update(header)
                aggregate.update(raw)
                module = "fm_modules" if key.startswith("fm_modules.") else "language_model_mot_gen"
                module_hash = per_module.setdefault(module, hashlib.sha256())
                module_hash.update(header)
                module_hash.update(raw)
                tensor_count += 1
                element_count += tensor.numel()
                l2_sq += float(torch.sum(tensor.float().square()).item())
    return {
        "sha256": aggregate.hexdigest(),
        "per_module_sha256": {name: digest.hexdigest() for name, digest in sorted(per_module.items())},
        "tensor_count": tensor_count,
        "element_count": element_count,
        "l2_norm": l2_sq**0.5,
    }


def checkpoint_hf_path(root: Path, step: int) -> Path:
    return root / "checkpoints" / f"global_step_{step}" / "actor" / "huggingface"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    ap.add_argument("--base", type=Path, default=DEFAULT_BASE)
    ap.add_argument("--steps", type=int, nargs="+", default=[8, 16, 24, 32])
    args = ap.parse_args()

    base_map = load_weight_map(args.base)
    fm_keys = sorted(key for key in base_map if is_fm_parameter(key))
    if not fm_keys:
        raise RuntimeError("no FM parameters selected")
    base_digest = tensor_digest(args.base, fm_keys)
    checkpoints = []
    for step in args.steps:
        folder = args.root / "checkpoints" / f"global_step_{step}"
        hf_path = checkpoint_hf_path(args.root, step)
        required = {
            "complete": folder / "COMPLETE",
            "actor": folder / "actor",
            "critic": folder / "critic",
            "resolved_config": folder / "resolved_config.yaml",
            "metadata": folder / "checkpoint_metadata.json",
            "dataloader": folder / "data.pt",
            "hf_config": hf_path / "config.json",
        }
        presence = {name: path.exists() for name, path in required.items()}
        if not all(presence.values()):
            raise RuntimeError(f"step {step} incomplete: {presence}")
        metadata = json.loads(required["metadata"].read_text(encoding="utf-8"))
        if int(metadata["global_step"]) != step or str(metadata.get("u1_fm_backprop")) != "0":
            raise RuntimeError(f"step {step} metadata mismatch: {metadata}")
        digest = tensor_digest(hf_path, fm_keys)
        checkpoints.append(
            {
                "step": step,
                "folder": str(folder),
                "hf_model": str(hf_path),
                "presence": presence,
                "metadata": metadata,
                "fm_digest": digest,
                "fm_parameter_delta_exact_zero": digest["sha256"] == base_digest["sha256"],
                "fm_module_delta_exact_zero": {
                    name: value == base_digest["per_module_sha256"].get(name)
                    for name, value in digest["per_module_sha256"].items()
                },
            }
        )

    all_zero = all(entry["fm_parameter_delta_exact_zero"] for entry in checkpoints)
    output = {
        "status": "PASS" if all_zero else "FAIL",
        "checkpoint_completeness": "PASS",
        "base_model": str(args.base),
        "fm_selection": {
            "rule": "name starts with fm_modules. OR contains _mot_gen",
            "parameter_tensor_count": len(fm_keys),
        },
        "base_fm_digest": base_digest,
        "checkpoints": checkpoints,
        "fm_backward_expected": 0,
        "fm_optimizer_update_observed": not all_zero,
        "fm_parameter_delta_exact_zero_all_milestones": all_zero,
    }
    out = args.root / "checkpoint_fm_audit.json"
    out.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    return 0 if output["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
