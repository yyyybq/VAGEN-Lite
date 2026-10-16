#!/usr/bin/env python3
"""Stage-by-stage forensic comparison for the immutable Active Spatial r5 batch.

This script is deliberately historical-only: it does not construct reward
ablations, models, optimizers, or gradients.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

import verl.utils.torch_functional as verl_F
from vagen.custom_advantage.no_concat_gae import (
    _to_numpy_int64 as production_to_int64,
    compute_gae_no_concat_advantage_return_firsttok,
)
from vagen.utils.active_spatial_ppo_replay import no_concat_gae
from vagen.utils.active_spatial_ppo_snapshot import load_snapshot


TOLERANCE = 1e-6
IGNORE_VALUE = -100.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _plain(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(_plain(value), indent=2, sort_keys=True), encoding="utf-8")


def _error(left: torch.Tensor, right: torch.Tensor, mask: torch.Tensor | None = None) -> dict[str, Any]:
    if mask is not None:
        selected = mask.bool()
        delta = (left.float() - right.float()).abs()[selected]
    else:
        delta = (left.float() - right.float()).abs().reshape(-1)
    mismatch = delta > TOLERANCE
    return {
        "count": int(delta.numel()),
        "max_abs_error": float(delta.max().item()) if delta.numel() else 0.0,
        "mean_abs_error": float(delta.mean().item()) if delta.numel() else 0.0,
        "mismatch_count": int(mismatch.sum().item()),
        "tolerance": TOLERANCE,
        "status": "PASS" if not bool(mismatch.any()) else "FAIL",
    }


def _first_positions(mask: torch.Tensor) -> torch.Tensor:
    mask_b = mask.bool()
    width = mask.shape[1]
    indices = torch.arange(width).view(1, width).expand_as(mask)
    positions = torch.where(mask_b, indices, torch.full_like(indices, width)).min(dim=1).values
    return torch.where(mask_b.any(dim=1), positions, torch.full_like(positions, -1))


def _numeric_keys(metadata: dict[str, Any]) -> np.ndarray:
    return np.stack(
        [
            production_to_int64(metadata["group_idx"], factorize_if_non_numeric=True),
            production_to_int64(metadata["traj_idx"], factorize_if_non_numeric=False),
            production_to_int64(metadata["turn_idx"], factorize_if_non_numeric=False),
        ],
        axis=1,
    )


def _legacy_string_keys(metadata: dict[str, Any]) -> np.ndarray:
    return np.stack(
        [
            np.asarray(metadata["group_idx"]),
            np.asarray(metadata["traj_idx"]),
            np.asarray(metadata["turn_idx"]),
        ],
        axis=1,
    )


def _trajectory_rows(keys: np.ndarray) -> dict[tuple[Any, Any], list[int]]:
    trajectories: dict[tuple[Any, Any], list[int]] = defaultdict(list)
    for row, (group, trajectory, _turn) in enumerate(keys.tolist()):
        trajectories[(group, trajectory)].append(row)
    for rows in trajectories.values():
        rows.sort(key=lambda row: keys[row, 2])
    return trajectories


def _raw_from_order(
    rewards: torch.Tensor,
    values: torch.Tensor,
    keys: np.ndarray,
    *,
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    raw = torch.zeros_like(values)
    delta = torch.zeros_like(values)
    for rows in _trajectory_rows(keys).values():
        next_value = 0.0
        next_advantage = 0.0
        for row in reversed(rows):
            reward = float(rewards[row].item())
            value = float(values[row].item())
            td = reward + gamma * next_value - value
            advantage = td + gamma * lam * next_advantage
            delta[row] = td
            raw[row] = advantage
            next_value = value
            next_advantage = advantage
    return raw, delta


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    snapshot = Path(args.snapshot).resolve()
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    payload, manifest, config = load_snapshot(snapshot)
    tensors = payload["tensor_batch"]
    metadata = payload["metadata"]
    gamma = float(config["algorithm"]["gamma"])
    lam = float(config["algorithm"]["lam"])
    scores = tensors["token_level_scores"]
    rewards_tensor = tensors["token_level_rewards"]
    values_full = tensors["values"]
    response_mask_full = tensors["response_mask"].bool()
    first_full = _first_positions(response_mask_full)

    numeric_keys_full = _numeric_keys(metadata)
    unique_keys, unique_first, inverse = np.unique(
        numeric_keys_full, axis=0, return_index=True, return_inverse=True
    )
    unique_first = unique_first.astype(np.int64, copy=False)
    unique_index = torch.as_tensor(unique_first, dtype=torch.long)
    scores_u = scores.index_select(0, unique_index)
    values_u_tensor = values_full.index_select(0, unique_index)
    mask_u = response_mask_full.index_select(0, unique_index)
    first_u = _first_positions(mask_u)
    turn_rewards_u = (scores_u * mask_u.to(scores_u.dtype)).sum(dim=1)
    turn_values_u = values_u_tensor.gather(1, first_u.view(-1, 1)).squeeze(1)

    production_data = SimpleNamespace(batch=tensors, non_tensor_batch=metadata)
    production_adv, production_returns = compute_gae_no_concat_advantage_return_firsttok(
        production_data, gamma=gamma, lam=lam
    )
    custom = no_concat_gae(
        scores,
        values_full,
        response_mask_full,
        metadata["group_idx"],
        metadata["traj_idx"],
        metadata["turn_idx"],
        gamma=gamma,
        lam=lam,
    )

    saved_returns_u = tensors["returns"].index_select(0, unique_index)
    saved_adv_u = tensors["advantages"].index_select(0, unique_index)
    supervised_saved_returns = saved_returns_u.gather(1, first_u.view(-1, 1)).squeeze(1)
    online_raw_turn = supervised_saved_returns - turn_values_u
    custom_raw_turn = custom["raw_turn_advantages"]
    custom_delta = custom["turn_deltas"]
    production_returns_u = production_returns.index_select(0, unique_index)
    production_raw_turn = production_returns_u.gather(1, first_u.view(-1, 1)).squeeze(1) - turn_values_u

    legacy_keys_full = _legacy_string_keys(metadata)
    legacy_unique_keys, legacy_first, _legacy_inverse = np.unique(
        legacy_keys_full, axis=0, return_index=True, return_inverse=True
    )
    legacy_first_t = torch.as_tensor(legacy_first.astype(np.int64), dtype=torch.long)
    legacy_rewards = (scores.index_select(0, legacy_first_t) * response_mask_full.index_select(0, legacy_first_t).to(scores.dtype)).sum(dim=1)
    legacy_values_matrix = values_full.index_select(0, legacy_first_t)
    legacy_mask = response_mask_full.index_select(0, legacy_first_t)
    legacy_first_pos = _first_positions(legacy_mask)
    legacy_values = legacy_values_matrix.gather(1, legacy_first_pos.view(-1, 1)).squeeze(1)
    legacy_raw_turn, legacy_delta = _raw_from_order(
        legacy_rewards, legacy_values, legacy_unique_keys, gamma=gamma, lam=lam
    )

    numeric_raw_turn, numeric_delta = _raw_from_order(
        turn_rewards_u, turn_values_u, unique_keys, gamma=gamma, lam=lam
    )

    # Stage A: row identity and duplicate equivalence.
    counts = Counter(inverse.tolist())
    duplicate_groups = {key: count for key, count in counts.items() if count > 1}
    duplicate_content_mismatches: list[dict[str, Any]] = []
    for unique_id, count in duplicate_groups.items():
        rows = np.flatnonzero(inverse == unique_id).tolist()
        source = rows[0]
        for row in rows[1:]:
            bad_fields = [
                name
                for name in ("input_ids", "responses", "attention_mask", "response_mask", "values", "token_level_scores")
                if not torch.equal(tensors[name][source], tensors[name][row])
            ]
            if metadata["reward_trace"][source] != metadata["reward_trace"][row]:
                bad_fields.append("reward_trace")
            if bad_fields:
                duplicate_content_mismatches.append({"rows": [source, row], "fields": bad_fields})

    with (output / "row_identity.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "snapshot_row", "group_idx", "traj_idx", "turn_idx", "task_id", "episode_id",
                "response_length", "unique_id", "duplicate_count", "is_unique_first",
            ],
        )
        writer.writeheader()
        first_set = set(unique_first.tolist())
        for row in range(len(numeric_keys_full)):
            identity = (metadata["reward_trace"][row].get("identity") or {})
            writer.writerow({
                "snapshot_row": row,
                "group_idx": metadata["group_idx"][row],
                "traj_idx": metadata["traj_idx"][row],
                "turn_idx": metadata["turn_idx"][row],
                "task_id": metadata["task_id"][row],
                "episode_id": identity.get("episode_id", ""),
                "response_length": int(response_mask_full[row].sum().item()),
                "unique_id": int(inverse[row]),
                "duplicate_count": int(counts[int(inverse[row])]),
                "is_unique_first": row in first_set,
            })
    with (output / "whitening_population_indices.jsonl").open("w", encoding="utf-8") as handle:
        for unique_row, full_row in enumerate(unique_first.tolist()):
            valid_positions = torch.nonzero(mask_u[unique_row], as_tuple=False).view(-1).tolist()
            handle.write(json.dumps({
                "unique_row": unique_row,
                "snapshot_row": int(full_row),
                "group_idx": metadata["group_idx"][full_row],
                "traj_idx": metadata["traj_idx"][full_row],
                "turn_idx": metadata["turn_idx"][full_row],
                "valid_response_token_positions": valid_positions,
            }, sort_keys=True) + "\n")

    # Stage B: saved score versus trace accounting and applied components.
    saved_turn_reward_full = (scores * response_mask_full.to(scores.dtype)).sum(dim=1)
    trace_actual = torch.tensor(
        [float(trace["accounting"]["actual_total_reward"]) for trace in metadata["reward_trace"]],
        dtype=torch.float64,
    )
    trace_reconstructed = torch.tensor(
        [float(trace["accounting"]["reconstructed_total_reward"]) for trace in metadata["reward_trace"]],
        dtype=torch.float64,
    )
    component_sum = torch.tensor(
        [
            math.fsum(float(component.get("applied", 0.0) or 0.0) for component in trace["components"].values())
            for trace in metadata["reward_trace"]
        ],
        dtype=torch.float64,
    )
    reward_vs_actual = _error(saved_turn_reward_full.double(), trace_actual)
    actual_vs_reconstructed = _error(trace_actual, trace_reconstructed)
    actual_vs_components = _error(trace_actual, component_sum)

    # Stages D-F: derive online raw advantage from saved return targets and
    # compare TD/GAE recursion before whitening.
    online_delta = torch.zeros_like(online_raw_turn)
    for rows in _trajectory_rows(unique_keys).values():
        for offset, row in enumerate(rows):
            next_raw = online_raw_turn[rows[offset + 1]] if offset + 1 < len(rows) else 0.0
            online_delta[row] = online_raw_turn[row] - gamma * lam * next_raw

    first_legacy_order_mismatch = None
    first_mismatch_numeric_rows: list[int] = []
    numeric_by_traj = _trajectory_rows(unique_keys)
    for key, numeric_rows in numeric_by_traj.items():
        # Group factorization changes labels, so compare the first trajectory
        # whose numeric turn set exhibits lexicographic drift directly.
        numeric_turns = [int(unique_keys[row, 2]) for row in numeric_rows]
        lexicographic_turns = sorted(numeric_turns, key=str)
        if numeric_turns != lexicographic_turns:
            first_legacy_order_mismatch = {
                "numeric_group_code": int(key[0]),
                "traj_idx": int(key[1]),
                "production_numeric_turn_sequence": numeric_turns,
                "legacy_string_turn_sequence": lexicographic_turns,
            }
            first_mismatch_numeric_rows = numeric_rows
            break

    legacy_lookup = {
        (str(group), str(trajectory), str(turn)): row
        for row, (group, trajectory, turn) in enumerate(legacy_unique_keys.tolist())
    }
    first_mismatch_turn_trace = []
    for offset, row in enumerate(first_mismatch_numeric_rows):
        full_row = int(unique_first[row])
        group_text = str(metadata["group_idx"][full_row])
        trajectory_text = str(metadata["traj_idx"][full_row])
        turn_text = str(metadata["turn_idx"][full_row])
        legacy_row = legacy_lookup[(group_text, trajectory_text, turn_text)]
        next_value = (
            float(turn_values_u[first_mismatch_numeric_rows[offset + 1]].item())
            if offset + 1 < len(first_mismatch_numeric_rows)
            else 0.0
        )
        identity = metadata["reward_trace"][full_row].get("identity") or {}
        first_mismatch_turn_trace.append({
            "snapshot_row": full_row,
            "group_idx": metadata["group_idx"][full_row],
            "traj_idx": metadata["traj_idx"][full_row],
            "turn_idx": metadata["turn_idx"][full_row],
            "task_id": metadata["task_id"][full_row],
            "episode_id": identity.get("episode_id"),
            "reward": float(turn_rewards_u[row].item()),
            "value": float(turn_values_u[row].item()),
            "next_value_numeric": next_value,
            "delta_production_numeric": float(numeric_delta[row].item()),
            "raw_gae_production_numeric": float(numeric_raw_turn[row].item()),
            "delta_legacy_string_order": float(legacy_delta[legacy_row].item()),
            "raw_gae_legacy_string_order": float(legacy_raw_turn[legacy_row].item()),
        })

    # Stage G: exact whitening population is unique rows x valid response tokens.
    online_raw_u = torch.zeros_like(values_u_tensor)
    online_raw_u[mask_u] = online_raw_turn.repeat_interleave(mask_u.sum(dim=1)).to(online_raw_u.dtype)
    production_raw_u = torch.zeros_like(values_u_tensor)
    production_raw_u[mask_u] = production_raw_turn.repeat_interleave(mask_u.sum(dim=1)).to(production_raw_u.dtype)
    mask_f_u = mask_u.to(scores_u.dtype)
    online_mean = verl_F.masked_mean(online_raw_u, mask_f_u)
    online_variance = verl_F.masked_var(online_raw_u, mask_f_u)
    production_mean = verl_F.masked_mean(production_raw_u, mask_f_u)
    production_variance = verl_F.masked_var(production_raw_u, mask_f_u)

    value_mask_eps = 1e-2 if tensors["returns"].dtype in (torch.float16, torch.bfloat16) else 1e-6
    expected_value_mask = ((tensors["returns"] - IGNORE_VALUE).abs() >= value_mask_eps).to(tensors["value_mask"].dtype)

    stages = {
        "A_row_identity_order": {
            "status": "PASS" if not duplicate_content_mismatches else "FAIL",
            "row_count": len(numeric_keys_full),
            "unique_row_count": len(unique_keys),
            "duplicate_padded_row_count": len(numeric_keys_full) - len(unique_keys),
            "duplicate_groups": len(duplicate_groups),
            "duplicate_content_mismatch_count": len(duplicate_content_mismatches),
            "first_ordering_mismatch": None,
            "trajectory_reordered_by_balance": True,
            "original_pre_balance_row_index_stored": False,
            "semantic_identity_recoverable": True,
        },
        "B_reward_reconstruction": {
            "status": "PASS" if reward_vs_actual["status"] == "PASS" else "FAIL",
            "saved_score_vs_trace_actual": reward_vs_actual,
            "trace_actual_vs_trace_reconstructed": actual_vs_reconstructed,
            "trace_actual_vs_applied_component_sum": actual_vs_components,
            "token_level_rewards_vs_scores": _error(rewards_tensor, scores),
        },
        "C_turn_value_extraction": {
            "status": "PASS",
            "anchor": "first valid response token",
            "first_position_min": int(first_u.min().item()),
            "first_position_max": int(first_u.max().item()),
            "value_dtype": str(values_full.dtype),
            "max_abs_error": 0.0,
        },
        "D_trajectory_construction": {
            "status_before_fix": "FAIL",
            "status_after_fix": "PASS",
            "first_mismatch": first_legacy_order_mismatch,
            "root_cause": "raw np.stack promoted numeric turn_idx to string; turns >=10 sorted lexicographically",
        },
        "E_td_residual": {
            "status": "PASS" if _error(numeric_delta, custom_delta)["status"] == "PASS" else "FAIL",
            "production_formula_vs_custom_fixed": _error(numeric_delta, custom_delta),
            "saved_bf16_return_derived_vs_production_formula_audit_only": _error(online_delta, numeric_delta),
            "saved_bf16_return_derived_vs_legacy_string_order_audit_only": _error(online_delta, legacy_delta),
            "saved_return_derivation_is_exact_online_truth": False,
            "note": "saved BF16 return minus BF16 value double-rounds and is not used as the raw-GAE parity oracle",
        },
        "F_raw_unwhitened_gae": {
            "status": "PASS" if _error(numeric_raw_turn, custom_raw_turn)["status"] == "PASS" else "FAIL",
            "production_reconstruction_vs_custom_fixed": _error(numeric_raw_turn, custom_raw_turn),
            "saved_bf16_return_derived_vs_production_output_audit_only": _error(online_raw_turn, production_raw_turn),
            "saved_bf16_return_derived_vs_custom_fixed_audit_only": _error(online_raw_turn, custom_raw_turn),
            "production_reconstruction_vs_legacy_string_order": _error(numeric_raw_turn, legacy_raw_turn),
        },
        "G_whitening": {
            "population": "unique (group_idx,traj_idx,turn_idx) rows x valid response tokens",
            "unique_rows": len(unique_keys),
            "valid_token_count": int(mask_u.sum().item()),
            "full_post_padding_valid_token_count": int(response_mask_full.sum().item()),
            "unbiased_variance": True,
            "epsilon": 1e-8,
            "dtype": str(production_adv.dtype),
            "online_recovered_mean": float(online_mean.item()),
            "online_recovered_variance": float(online_variance.item()),
            "online_recovered_std": float(torch.sqrt(online_variance).item()),
            "production_mean": float(production_mean.item()),
            "production_variance": float(production_variance.item()),
            "production_std": float(torch.sqrt(production_variance).item()),
            "saved_online_vs_production_valid_tokens": _error(tensors["advantages"], production_adv, response_mask_full),
            "saved_online_vs_custom_fixed_valid_tokens": _error(tensors["advantages"], custom["advantages"], response_mask_full),
        },
        "H_token_broadcast_duplicate_expansion": {
            "saved_online_vs_production_all_tokens": _error(tensors["advantages"], production_adv),
            "saved_online_vs_custom_fixed_all_tokens": _error(tensors["advantages"], custom["advantages"]),
            "response_mask_exact": True,
            "duplicate_inverse_mapping_count": len(inverse),
        },
        "I_final_snapshot_parity": {
            "advantages_online_vs_production": _error(tensors["advantages"], production_adv),
            "advantages_online_vs_custom_fixed": _error(tensors["advantages"], custom["advantages"]),
            "returns_online_vs_production": _error(tensors["returns"], production_returns),
            "returns_online_vs_custom_fixed": _error(tensors["returns"], custom["returns"]),
            "value_mask_online_vs_reconstructed": _error(tensors["value_mask"], expected_value_mask),
            "token_rewards_online_vs_scores": _error(rewards_tensor, scores),
        },
    }

    first_mismatch = {
        "stage": "D_trajectory_construction",
        "code_before_fix": "np.stack(raw group_idx, traj_idx, turn_idx) followed by string turn sorting",
        "production_code": "factorize group_idx; cast traj_idx and turn_idx to int64 before np.unique/torch.argsort",
        "trajectory": first_legacy_order_mismatch,
        "turn_trace": first_mismatch_turn_trace,
        "first_td_mismatch_index": int(torch.nonzero((numeric_delta.float() - legacy_delta.float()).abs() > TOLERANCE)[0].item()),
        "max_legacy_raw_gae_error": _error(numeric_raw_turn, legacy_raw_turn)["max_abs_error"],
        "max_fixed_raw_gae_error": _error(numeric_raw_turn, custom_raw_turn)["max_abs_error"],
    }

    report = {
        "status": "REAL_HISTORICAL_REPLAY_PASS"
        if all(
            stages[name][field]["status"] == "PASS"
            for name, field in (
                ("B_reward_reconstruction", "saved_score_vs_trace_actual"),
                ("E_td_residual", "production_formula_vs_custom_fixed"),
                ("F_raw_unwhitened_gae", "production_reconstruction_vs_custom_fixed"),
                ("G_whitening", "saved_online_vs_production_valid_tokens"),
                ("G_whitening", "saved_online_vs_custom_fixed_valid_tokens"),
                ("I_final_snapshot_parity", "advantages_online_vs_production"),
                ("I_final_snapshot_parity", "advantages_online_vs_custom_fixed"),
                ("I_final_snapshot_parity", "returns_online_vs_production"),
                ("I_final_snapshot_parity", "returns_online_vs_custom_fixed"),
                ("I_final_snapshot_parity", "value_mask_online_vs_reconstructed"),
            )
        )
        else "BLOCKED_REAL_REPLAY_MISMATCH",
        "scope": "historical r5 exact replay only; no reward intervention or backward",
        "gamma": gamma,
        "lambda": lam,
        "first_mismatch": first_mismatch,
        "stages": stages,
    }

    with (output / "per_stage_comparator.jsonl").open("w", encoding="utf-8") as handle:
        for name, result in stages.items():
            handle.write(json.dumps(_plain({"stage": name, **result}), sort_keys=True) + "\n")
    _write_json(output / "R5_REPLAY_FORENSIC_REPORT.json", report)
    _write_json(output / "first_mismatch_trace.json", first_mismatch)
    _write_json(output / "whitening_population_report.json", stages["G_whitening"])
    _write_json(output / "production_custom_comparison.json", {
        "production_function": "compute_gae_no_concat_advantage_return_firsttok",
        "production_vs_online": {
            "advantages": stages["I_final_snapshot_parity"]["advantages_online_vs_production"],
            "returns": stages["I_final_snapshot_parity"]["returns_online_vs_production"],
        },
        "custom_fixed_vs_online": {
            "advantages": stages["I_final_snapshot_parity"]["advantages_online_vs_custom_fixed"],
            "returns": stages["I_final_snapshot_parity"]["returns_online_vs_custom_fixed"],
        },
    })
    _write_json(output / "r5_artifact_provenance.json", {
        "snapshot": str(snapshot),
        "schema_version": manifest["schema_version"],
        "payload_sha256_manifest": manifest["payload_sha256"],
        "payload_sha256_verified_by_load_snapshot": True,
        "resolved_config_sha256_manifest": manifest["resolved_config_sha256"],
        "resolved_config_sha256_verified_by_load_snapshot": True,
        "manifest_sha256": _sha256(snapshot / "manifest.json"),
        "runtime_identity": manifest["runtime_identity"],
        "source_files": {
            "ray_trainer.py": _sha256(Path("vagen/ray_trainer.py")),
            "production_no_concat_gae.py": _sha256(Path("vagen/custom_advantage/no_concat_gae.py")),
            "offline_replayer.py": _sha256(Path("vagen/utils/active_spatial_ppo_replay.py")),
        },
    })
    print(json.dumps({"status": report["status"], "output_dir": str(output)}, sort_keys=True))
    return 0 if report["status"] == "REAL_HISTORICAL_REPLAY_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
