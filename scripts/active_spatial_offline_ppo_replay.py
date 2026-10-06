#!/usr/bin/env python3
"""Replay an immutable Active Spatial PPO snapshot without an optimizer step.

The optional adapter is intentionally explicit.  It must load the checkpoint
bound in manifest.json and return ``actor``, ``critic``, ``actor_logprob`` and
``critic_value``.  The script never creates an optimizer or calls ``step``.
"""

from __future__ import annotations

import argparse
import importlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from vagen.utils.active_spatial_ppo_replay import gradient_diagnosis, recompute_branches
from vagen.utils.active_spatial_ppo_snapshot import SnapshotValidationError, load_snapshot


def _tensor_summary(value: torch.Tensor, mask: torch.Tensor) -> dict[str, float | int]:
    selected = value[mask.bool()].detach().float().cpu()
    return {
        "count": int(selected.numel()),
        "mean": float(selected.mean()) if selected.numel() else 0.0,
        "std": float(selected.std(unbiased=False)) if selected.numel() else 0.0,
        "min": float(selected.min()) if selected.numel() else 0.0,
        "max": float(selected.max()) if selected.numel() else 0.0,
    }


def _family_stats(advantages: torch.Tensor, mask: torch.Tensor, families: list[Any]) -> dict[str, dict[str, float | int]]:
    rows: dict[str, list[torch.Tensor]] = defaultdict(list)
    for row, family in enumerate(families):
        rows[str(family)].append(advantages[row][mask[row].bool()].detach().float().cpu())
    return {family: _tensor_summary(torch.cat(values), torch.ones_like(torch.cat(values), dtype=torch.bool)) for family, values in rows.items()}


def _adapter(spec: str):
    module_name, separator, attribute = spec.partition(":")
    if not separator:
        raise SnapshotValidationError("--adapter must be MODULE:FACTORY")
    factory = getattr(importlib.import_module(module_name), attribute)
    adapter = factory()
    required = ("actor", "critic", "actor_logprob", "critic_value")
    missing = [key for key in required if key not in adapter]
    if missing:
        raise SnapshotValidationError(f"adapter missing: {', '.join(missing)}")
    return adapter


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--adapter", help="MODULE:FACTORY; required only for --backward")
    parser.add_argument("--backward", action="store_true")
    args = parser.parse_args()
    payload, manifest, config = load_snapshot(args.snapshot)
    replay = recompute_branches(payload, config)
    tensors, metadata = payload["tensor_batch"], payload["metadata"]
    historical_error = (replay["historical"]["advantages"] - tensors["advantages"]).abs().max().item()
    report: dict[str, Any] = {
        "status": "PASS" if historical_error <= 1e-6 else "BLOCKED",
        "scope": "FIXTURE_ONLY unless snapshot was captured from a data-gate PASS real batch",
        "snapshot_schema_version": manifest["schema_version"],
        "snapshot_runtime_identity": manifest["runtime_identity"],
        "historical_no_concat_gae_max_abs_error": historical_error,
        "optimizer_step_called": False,
        "branches": {},
    }
    for name, branch in replay["branches"].items():
        report["branches"][name] = {
            "turn_total_reward": branch["turn_totals"],
            "raw_advantage": _tensor_summary(branch["raw_advantages"], tensors["response_mask"]),
            "independent_whitened_advantage": _tensor_summary(branch["independent_advantages"], tensors["response_mask"]),
            "fixed_reference_whitened_advantage": _tensor_summary(branch["fixed_reference_advantages"], tensors["response_mask"]),
            "by_task_family": _family_stats(branch["independent_advantages"], tensors["response_mask"], metadata["task_type"]),
        }
    if args.backward:
        if not args.adapter:
            raise SnapshotValidationError("--backward requires an immutable-checkpoint adapter")
        adapter = _adapter(args.adapter)
        report["gradient_diagnosis"] = gradient_diagnosis(payload, config, **adapter)
    Path(args.output).write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SnapshotValidationError as exc:
        print(f"BLOCKED: {exc}")
        raise SystemExit(2)
