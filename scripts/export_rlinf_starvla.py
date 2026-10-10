#!/usr/bin/env python3
"""Export a dense RLinf StarVLA actor to the native deterministic policy format.

Requires actor/model_state_dict/full_weights.pt, not DCP/local shards. The
original StarVLA run provides the architecture, processor path and action stats.
Keep the RLinf checkpoint separately to resume optimizers and Gaussian policy.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from prepare_rlinf_robocasa import file_identity, inspect_policy


def extract_policy(state, reference):
    prefix = "starvla_model."
    policy = {
        name[len(prefix) :]: value
        for name, value in state.items()
        if name.startswith(prefix)
    }
    extras = [
        name
        for name in state
        if not name.startswith(prefix)
        and name != "actor_logstd"
        and not name.startswith("value_head.")
    ]
    if extras or not policy or set(policy) != set(reference):
        raise ValueError(
            "RLinf actor does not match the source StarVLA state dict; adapters/shards need explicit conversion"
        )
    for name, value in policy.items():
        if tuple(value.shape) != tuple(reference[name].shape):
            raise ValueError(f"Parameter shape changed: {name}")
    return policy


def main():
    import torch

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument(
        "--source-ckpt",
        type=Path,
        required=True,
        help="The StarVLA .pt used to initialize this RLinf run",
    )
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    _, _, source = inspect_policy(args.source_ckpt)
    state = torch.load(args.weights, map_location="cpu", weights_only=True, mmap=True)
    reference = torch.load(
        args.source_ckpt, map_location="cpu", weights_only=True, mmap=True
    )
    policy = extract_policy(state, reference)
    target = args.out / "checkpoints/policy.pt"
    target.parent.mkdir(parents=True)
    torch.save(policy, target)
    run = args.source_ckpt.resolve().parent.parent
    for name in ("config.yaml", "config.full.yaml", "dataset_statistics.json"):
        path = run / name
        if path.is_file():
            shutil.copy2(path, args.out / name)
    (args.out / "rlinf_export.json").write_text(
        json.dumps(
            {
                "source_starvla": source,
                "rlinf_weights": file_identity(args.weights),
                "exported_parameters": len(policy),
                "purpose": "Deterministic action-mean evaluation and subsequent HF backbone export",
                "excluded": ["actor_logstd", "value_head", "optimizer", "scheduler"],
            },
            indent=2,
        )
        + "\n"
    )
    print(target.resolve())


if __name__ == "__main__":
    main()
