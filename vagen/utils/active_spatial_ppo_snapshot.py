"""Exact, opt-in snapshots for Active Spatial no-concat PPO batches.

This is deliberately a diagnostic boundary, not a training feature.  Nothing is
written unless ``VAGEN_ACTIVE_SPATIAL_PPO_SNAPSHOT_DIR`` is set.  The writer is
fail-closed: a partial text/score log is never presented as a replayable PPO
batch.
"""

from __future__ import annotations

import hashlib
import json
import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch


SCHEMA_VERSION = "active_spatial_ppo_batch_snapshot_v1"
SNAPSHOT_REQUIRED_TENSORS = (
    "input_ids", "responses", "attention_mask", "response_mask", "position_ids",
    "old_log_probs", "values", "token_level_scores", "token_level_rewards",
    "advantages", "returns", "value_mask",
)
SNAPSHOT_REQUIRED_METADATA = (
    "group_idx", "traj_idx", "turn_idx", "__last_turn__", "reward_trace",
    "task_id", "task_type", "terminated", "truncated", "termination_reason",
    "action_text", "parsed_primitive_actions", "bootstrap_context",
)


class SnapshotValidationError(RuntimeError):
    """Raised when a batch cannot be replayed exactly from a snapshot."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def directory_manifest_sha256(root: str | Path) -> str:
    """Content identity for a newly exported immutable model directory.

    A filename/path is not an identity.  The digest commits to every regular
    file's relative path and bytes, without requiring a particular checkpoint
    serialization layout.
    """
    path = Path(root)
    if not path.is_dir():
        raise SnapshotValidationError(f"checkpoint export directory is missing: {path}")
    entries = []
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        entries.append({"path": str(item.relative_to(path)), "sha256": _sha256_bytes(item.read_bytes())})
    if not entries:
        raise SnapshotValidationError(f"checkpoint export directory is empty: {path}")
    return _sha256_bytes(_canonical_json(entries))


def _canonical_json(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")


def _json_scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return value


def _to_cpu_tree(value: Any) -> Any:
    """Convert tensor/numpy containers into a deterministic torch-save payload."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return [_to_cpu_tree(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _to_cpu_tree(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_cpu_tree(item) for item in value]
    if isinstance(value, bytes):
        return {"__snapshot_bytes_hex__": value.hex()}
    # Ray/DataProto metadata commonly carries numpy scalar indices (for
    # example ``np.int64`` group/trajectory ids).  They have an exact Python
    # scalar representation, unlike arbitrary extension objects, so accepting
    # them preserves the fail-closed boundary while allowing real batches.
    if value is None or isinstance(value, (str, int, float, bool, np.generic)):
        return _json_scalar(value)
    raise SnapshotValidationError(f"unsupported snapshot payload type: {type(value).__name__}")


def _config_to_plain(config: Any) -> Any:
    try:
        from omegaconf import OmegaConf

        # DictConfig implements Mapping.  Check OmegaConf first so a resolved
        # training config never retains nested DictConfig/ListConfig values
        # that would make the immutable JSON sidecar non-serializable.
        if OmegaConf.is_config(config):
            return OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    except ImportError:
        pass
    if isinstance(config, Mapping):
        return deepcopy(config)
    raise SnapshotValidationError(
        f"resolved config must be a Mapping or OmegaConf config, got {type(config).__name__}"
    )


def _get_path(mapping: Mapping[str, Any], dotted: str, default: Any = None) -> Any:
    cur: Any = mapping
    for part in dotted.split("."):
        if not isinstance(cur, Mapping) or part not in cur:
            return default
        cur = cur[part]
    return cur


def _sequence(value: Any, name: str, batch_size: int) -> list[Any]:
    if isinstance(value, np.ndarray):
        value = value.tolist()
    elif isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    elif not isinstance(value, (list, tuple)):
        raise SnapshotValidationError(f"{name} is not batch-indexable")
    if len(value) != batch_size:
        raise SnapshotValidationError(f"{name} length {len(value)} != batch size {batch_size}")
    return [_to_cpu_tree(item) for item in value]


def snapshot_contract() -> dict[str, list[str]]:
    """Field classification exposed in both schema docs and manifests."""
    return {
        "required_for_training_replay": list(SNAPSHOT_REQUIRED_TENSORS) + [
            "multi_modal_inputs", "group_idx", "traj_idx", "turn_idx", "__last_turn__",
            "reward_trace", "terminated", "truncated", "termination_reason",
            "checkpoint_identity", "resolved_config",
        ],
        "deterministically_reconstructable": [
            "optimization_mask_from_response_mask", "turn_token_map_from_response_mask_and_indices",
            "historical_total_reward_from_reward_trace", "historical_raw_advantage_from_scores_values_indices",
        ],
        "audit_metadata": [
            "task_id", "task_type", "episode_id", "action_text", "parsed_primitive_actions",
            "rollout_temperature", "logprob_temperature", "rng_identity", "sampler_identity",
            "bootstrap_context", "snapshot_hook",
        ],
    }


def _checkpoint_identity_from_env(role: str) -> dict[str, str]:
    upper = role.upper()
    prefix = f"VAGEN_ACTIVE_SPATIAL_PPO_SNAPSHOT_{upper}_CHECKPOINT_"
    path = os.environ.get(prefix + "PATH")
    digest = os.environ.get(prefix + "SHA256")
    if not path or not digest:
        raise SnapshotValidationError(
            f"missing immutable {role} checkpoint identity; set {prefix}PATH and {prefix}SHA256"
        )
    if len(digest) != 64 or any(char not in "0123456789abcdefABCDEF" for char in digest):
        raise SnapshotValidationError(f"{prefix}SHA256 is not a SHA256 hex digest")
    return {"role": role, "path": path, "content_sha256": digest.lower()}


def runtime_identity_from_env(config: Any, global_step: int) -> dict[str, Any]:
    """Construct strict runtime binding without hashing multi-GB checkpoints in the trainer."""
    plain = _config_to_plain(config)
    return {
        "global_step": int(global_step),
        "actor": _checkpoint_identity_from_env("actor"),
        "critic": _checkpoint_identity_from_env("critic"),
        "reference": _checkpoint_identity_from_env("reference"),
        "rollout_temperature": _get_path(plain, "actor_rollout_ref.rollout.temperature"),
        "logprob_temperature": _get_path(plain, "actor_rollout_ref.rollout.logprob_temperature", 1.0),
        "rng_identity": {
            "torch_initial_seed": int(torch.initial_seed()),
            "numpy_state_sha256": _sha256_bytes(repr(np.random.get_state()).encode("utf-8")),
        },
        "sampler_identity": {
            "rollout_name": _get_path(plain, "actor_rollout_ref.rollout.name"),
            "rollout_n": _get_path(plain, "actor_rollout_ref.rollout.n"),
        },
    }


def _make_turn_token_map(response_mask: torch.Tensor, metadata: Mapping[str, list[Any]]) -> list[dict[str, Any]]:
    out = []
    for row in range(response_mask.shape[0]):
        out.append({
            "batch_row": row,
            "group_idx": metadata["group_idx"][row],
            "traj_idx": metadata["traj_idx"][row],
            "turn_idx": metadata["turn_idx"][row],
            "valid_response_token_positions": torch.nonzero(response_mask[row].bool(), as_tuple=False).view(-1).tolist(),
        })
    return out


def build_snapshot_payload(data: Any, config: Any, runtime_identity: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Extract the exact post-GAE, pre-update DataProto boundary into plain data."""
    tensor_batch = data.batch
    non_tensor_batch = data.non_tensor_batch
    missing = [key for key in SNAPSHOT_REQUIRED_TENSORS if key not in tensor_batch]
    if missing:
        raise SnapshotValidationError(f"missing tensor fields: {', '.join(missing)}")
    batch_size = int(tensor_batch["input_ids"].shape[0])
    metadata: dict[str, list[Any]] = {}
    for key in SNAPSHOT_REQUIRED_METADATA:
        if key not in non_tensor_batch:
            raise SnapshotValidationError(f"missing metadata field: {key}")
        metadata[key] = _sequence(non_tensor_batch[key], key, batch_size)
    tensors = {key: _to_cpu_tree(tensor_batch[key]) for key in SNAPSHOT_REQUIRED_TENSORS}
    if "ref_log_prob" in tensor_batch:
        tensors["ref_log_prob"] = _to_cpu_tree(tensor_batch["ref_log_prob"])
    if "rollout_log_probs" in tensor_batch:
        tensors["rollout_log_probs"] = _to_cpu_tree(tensor_batch["rollout_log_probs"])
    if "multi_modal_inputs" in non_tensor_batch:
        multimodal = _sequence(non_tensor_batch["multi_modal_inputs"], "multi_modal_inputs", batch_size)
    else:
        multimodal = [None] * batch_size
    plain_config = _config_to_plain(config)
    config_bytes = _canonical_json(plain_config)
    payload = {
        "tensor_batch": tensors,
        "metadata": metadata,
        "multi_modal_inputs": multimodal,
        "turn_token_map": _make_turn_token_map(tensors["response_mask"], metadata),
        "optimization_mask": tensors["response_mask"].clone(),
        "runtime_identity": _to_cpu_tree(runtime_identity),
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "resolved_config_sha256": _sha256_bytes(config_bytes),
        "runtime_identity": _to_cpu_tree(runtime_identity),
        "field_contract": snapshot_contract(),
        "batch_size": batch_size,
        "required_tensor_fields": list(tensors),
        "required_metadata_fields": list(metadata),
    }
    return payload, {"manifest": manifest, "resolved_config": plain_config}


def write_snapshot(
    output_dir: str | Path, data: Any, config: Any, runtime_identity: Mapping[str, Any], *, snapshot_name: str | None = None
) -> Path:
    """Atomically write a validated snapshot directory. Existing targets are refused."""
    payload, sidecar = build_snapshot_payload(data, config, runtime_identity)
    root = Path(output_dir)
    name = snapshot_name or f"global_step_{int(runtime_identity['global_step']):06d}_pre_update"
    target = root / name
    if target.exists():
        raise SnapshotValidationError(f"snapshot target already exists: {target}")
    target.mkdir(parents=True, exist_ok=False)
    payload_path = target / "payload.pt"
    config_path = target / "resolved_config.json"
    torch.save(payload, payload_path)
    config_path.write_bytes(_canonical_json(sidecar["resolved_config"]))
    manifest = dict(sidecar["manifest"])
    manifest["payload_sha256"] = _sha256_bytes(payload_path.read_bytes())
    manifest["resolved_config_sha256"] = _sha256_bytes(config_path.read_bytes())
    (target / "manifest.json").write_bytes(_canonical_json(manifest))
    return target


def load_snapshot(path: str | Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    root = Path(path)
    manifest_path, payload_path, config_path = root / "manifest.json", root / "payload.pt", root / "resolved_config.json"
    if not all(item.is_file() for item in (manifest_path, payload_path, config_path)):
        raise SnapshotValidationError(f"incomplete snapshot directory: {root}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise SnapshotValidationError(f"unsupported snapshot schema: {manifest.get('schema_version')!r}")
    if manifest.get("payload_sha256") != _sha256_bytes(payload_path.read_bytes()):
        raise SnapshotValidationError("payload SHA256 mismatch")
    config_bytes = config_path.read_bytes()
    if manifest.get("resolved_config_sha256") != _sha256_bytes(config_bytes):
        raise SnapshotValidationError("resolved config SHA256 mismatch")
    try:
        # Real VLM snapshots can contain a large immutable multimodal tensor
        # that numeric reward/GAE replay does not touch.  Memory mapping keeps
        # loading hash-validated payloads fast and read-only; older Torch
        # versions retain the eager-load fallback below.
        payload = torch.load(payload_path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:  # torch versions without weights_only and/or mmap
        payload = torch.load(payload_path, map_location="cpu")
    for key in ("tensor_batch", "metadata", "multi_modal_inputs", "turn_token_map", "optimization_mask", "runtime_identity"):
        if key not in payload:
            raise SnapshotValidationError(f"payload missing {key}")
    return payload, manifest, json.loads(config_bytes)


def maybe_write_pre_update_snapshot(data: Any, config: Any, global_step: int) -> Path | None:
    """Hook entrypoint. Empty env var means no validation, I/O, or semantic effect."""
    destination = os.environ.get("VAGEN_ACTIVE_SPATIAL_PPO_SNAPSHOT_DIR")
    if not destination:
        return None
    runtime_identity = runtime_identity_from_env(config, global_step)
    return write_snapshot(destination, data, config, runtime_identity)
