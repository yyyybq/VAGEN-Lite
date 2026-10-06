# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import hashlib
import os
import re
import time
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pprint import pprint
from typing import Optional

import numpy as np
import ray
import torch
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.metric import reduce_metrics
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from vagen.utils.image_dump_actor import ImageDumpActor
from vagen.utils.upload_hugging_face import HFUploadManager
from vagen.utils.image_validation_logger import ValidationGenerationsLogger
from vagen.utils.concat_val_multi_turn import concat_val_multi_turn
from vagen.utils.image_token_utils import replace_image_tokens_for_logging
from vagen.utils.active_spatial_ppo_snapshot import (
    directory_manifest_sha256,
    maybe_write_pre_update_snapshot,
)
import vagen.custom_advantage
from vagen.custom_metric.metric import METRIC_REGISTRY
from vagen.custom_filter.filter import FILTER_REGISTRY
@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create Ray resource pools for distributed training.

        Initializes resource pools based on the resource pool specification,
        with each pool managing GPU resources across multiple nodes.
        For FSDP backend, uses max_colocate_count=1 to merge WorkerGroups.
        For Megatron backend, uses max_colocate_count>1 for different models.
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray._private.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld
    token_level_kl_penalty = -beta * kld
    data.batch["token_level_kl_penalty"] = token_level_kl_penalty

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def _d0_13_trainer_marker(stage: str, **extra) -> None:
    out_dir = os.environ.get("VAGEN_D0_13_TIMELINE_DIR")
    if not out_dir:
        return
    rec = {
        "stage": stage,
        "pid": int(os.getpid()),
        "time": time.time(),
        **extra,
    }
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "trainer.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")


def _d0_3_quantile(values: torch.Tensor, q: float) -> float:
    if values.numel() == 0:
        return float("nan")
    return float(torch.quantile(values.float(), q).detach().cpu().item())


def _d0_3_response_mask_parity(data: DataProto, path: str, tag: str, global_step: int) -> dict:
    """Write D0.3 rollout-vs-FSDP token logprob parity for the real PPO batch."""
    if "rollout_log_probs" not in data.batch or "old_log_probs" not in data.batch:
        return {}
    response_mask = data.batch.get("response_mask")
    if response_mask is None:
        return {}
    rollout = data.batch["rollout_log_probs"].detach().float().cpu()
    actor = data.batch["old_log_probs"].detach().float().cpu()
    mask = response_mask.detach().bool().cpu()
    if rollout.shape != actor.shape:
        return {f"d0_3/{tag}/shape_mismatch": 1.0}
    delta = (actor - rollout)[mask]
    if delta.numel() == 0:
        return {f"d0_3/{tag}/token_count": 0}
    ratio = torch.exp(delta.clamp(min=-80.0, max=80.0))
    result = {
        "tag": tag,
        "global_step": int(global_step),
        "token_count": int(delta.numel()),
        "delta_logp_mean": float(delta.mean().item()),
        "delta_logp_median": float(delta.median().item()),
        "delta_logp_std": float(delta.std(unbiased=False).item()),
        "delta_logp_max_abs": float(delta.abs().max().item()),
        "ratio_mean": float(ratio.mean().item()),
        "ratio_median": float(ratio.median().item()),
        "ratio_std": float(ratio.std(unbiased=False).item()),
        "ratio_p1": _d0_3_quantile(ratio, 0.01),
        "ratio_p5": _d0_3_quantile(ratio, 0.05),
        "ratio_p95": _d0_3_quantile(ratio, 0.95),
        "ratio_p99": _d0_3_quantile(ratio, 0.99),
        "frac_abs_ratio_minus_1_gt_0p01": float(((ratio - 1.0).abs() > 0.01).float().mean().item()),
        "frac_abs_ratio_minus_1_gt_0p05": float(((ratio - 1.0).abs() > 0.05).float().mean().item()),
    }
    os.makedirs(path, exist_ok=True)
    out = os.path.join(path, f"{tag}_response_mask_parity_step{global_step}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return {f"d0_3/{tag}/{k}": v for k, v in result.items() if isinstance(v, (int, float))}


def _d0_4_pearson(x: torch.Tensor, y: torch.Tensor) -> float:
    if x.numel() < 2:
        return float("nan")
    x = x.float()
    y = y.float()
    vx = x - x.mean()
    vy = y - y.mean()
    denom = torch.sqrt((vx * vx).sum() * (vy * vy).sum())
    if denom.item() == 0:
        return float("nan")
    return float(((vx * vy).sum() / denom).detach().cpu().item())


def _d0_4_pair_stats(lhs: torch.Tensor, rhs: torch.Tensor, mask: torch.Tensor) -> dict:
    valid = mask.bool()
    if lhs.shape != rhs.shape:
        return {"shape_mismatch": 1, "lhs_shape": list(lhs.shape), "rhs_shape": list(rhs.shape)}
    if valid.sum().item() == 0:
        return {"token_count": 0}
    delta = (rhs.float() - lhs.float())[valid]
    ratio = torch.exp(delta.clamp(min=-80.0, max=80.0))
    abs_delta = delta.abs()
    return {
        "token_count": int(delta.numel()),
        "delta_logp_mean": float(delta.mean().item()),
        "delta_logp_median": float(delta.median().item()),
        "delta_logp_std": float(delta.std(unbiased=False).item()),
        "delta_logp_max_abs": float(abs_delta.max().item()),
        "mean_abs_delta_logp": float(abs_delta.mean().item()),
        "median_abs_delta_logp": float(abs_delta.median().item()),
        "p95_abs_delta_logp": _d0_3_quantile(abs_delta, 0.95),
        "ratio_mean": float(ratio.mean().item()),
        "ratio_median": float(ratio.median().item()),
        "ratio_std": float(ratio.std(unbiased=False).item()),
        "ratio_p1": _d0_3_quantile(ratio, 0.01),
        "ratio_p5": _d0_3_quantile(ratio, 0.05),
        "ratio_p95": _d0_3_quantile(ratio, 0.95),
        "ratio_p99": _d0_3_quantile(ratio, 0.99),
        "frac_abs_ratio_minus_1_gt_0p01": float(((ratio - 1.0).abs() > 0.01).float().mean().item()),
        "frac_abs_ratio_minus_1_gt_0p05": float(((ratio - 1.0).abs() > 0.05).float().mean().item()),
        "pearson": _d0_4_pearson(lhs.float()[valid], rhs.float()[valid]),
    }


def _d0_4_shift_stats(rollout: torch.Tensor, fsdp: torch.Tensor, mask: torch.Tensor) -> dict:
    out = {"same_index": _d0_4_pair_stats(rollout, fsdp, mask)}
    if rollout.size(1) > 1:
        out["fsdp_k_minus_1"] = _d0_4_pair_stats(rollout[:, 1:], fsdp[:, :-1], mask[:, 1:] & mask[:, :-1])
        out["fsdp_k_plus_1"] = _d0_4_pair_stats(rollout[:, :-1], fsdp[:, 1:], mask[:, :-1] & mask[:, 1:])
    return out


def _d0_4_tensor_digest(tensor: torch.Tensor) -> dict:
    flat = tensor.detach().float().cpu().reshape(-1)
    sample = flat[: min(16, flat.numel())]
    return {
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
        "mean": float(flat.mean().item()) if flat.numel() else 0.0,
        "std": float(flat.std(unbiased=False).item()) if flat.numel() else 0.0,
        "l2": float(torch.linalg.vector_norm(flat).item()) if flat.numel() else 0.0,
        "first_values": [float(x) for x in sample.tolist()],
        "sha1_first_4096": hashlib.sha1(flat[: min(4096, flat.numel())].numpy().tobytes()).hexdigest(),
    }


def _d0_4_token_type(token_id: int, start: int, end: int, text_l: str, eos_id: int | None, pad_id: int | None) -> str:
    if pad_id is not None and token_id == pad_id:
        return "pad"
    if eos_id is not None and token_id == eos_id:
        return "eos"

    def intersects(a: int, b: int, c: int, d: int) -> bool:
        return max(a, c) < min(b, d)

    for tag in ("<action>", "</action>", "<think>", "</think>"):
        pos = text_l.find(tag)
        while pos >= 0:
            if intersects(start, end, pos, pos + len(tag)):
                return "action_tag" if "action" in tag else "think_tag"
            pos = text_l.find(tag, pos + 1)

    action_start = text_l.find("<action>")
    action_end = text_l.find("</action>", action_start + 1) if action_start >= 0 else -1
    if action_start >= 0 and action_end >= 0 and start >= action_start + len("<action>") and end <= action_end:
        return "action_name"
    think_start = text_l.find("<think>")
    think_end = text_l.find("</think>", think_start + 1) if think_start >= 0 else -1
    if think_start >= 0 and (think_end < 0 or (start >= think_start + len("<think>") and end <= think_end)):
        return "think_text"
    return "other"


def _d0_6_decode_with_image_blocks(
    tokenizer,
    token_ids: list[int],
    *,
    attention_mask: list[int] | None = None,
    pad_id: int | None = None,
    image_token_id: int = -200,
) -> tuple[str, list[dict]]:
    """Decode text while compacting expanded Cambrian image token runs."""
    pieces: list[str] = []
    text_buf: list[int] = []
    image_blocks: list[dict] = []
    image_start: int | None = None
    image_len = 0

    def flush_text() -> None:
        nonlocal text_buf
        if text_buf:
            pieces.append(tokenizer.decode(text_buf, skip_special_tokens=False))
            text_buf = []

    def flush_image() -> None:
        nonlocal image_start, image_len
        if image_start is not None:
            pieces.append(f"<image_block:{image_len}>")
            image_blocks.append({"start": int(image_start), "length": int(image_len)})
            image_start = None
            image_len = 0

    for idx, tid in enumerate(token_ids):
        if attention_mask is not None and idx < len(attention_mask) and int(attention_mask[idx]) == 0:
            continue
        tid = int(tid)
        if tid == image_token_id:
            flush_text()
            if image_start is None:
                image_start = idx
                image_len = 0
            image_len += 1
            continue
        flush_image()
        if pad_id is not None and tid == pad_id:
            continue
        if tid >= 0:
            text_buf.append(tid)
        else:
            pieces.append(f"<unknown_negative_token:{tid}>")
    flush_text()
    flush_image()
    return "".join(pieces), image_blocks


def _d0_4_write_localization_artifacts(data: DataProto, path: str, tokenizer, global_step: int) -> dict:
    if "rollout_log_probs" not in data.batch or "old_log_probs" not in data.batch:
        return {}
    import csv

    os.makedirs(path, exist_ok=True)
    rollout = data.batch["rollout_log_probs"].detach().float().cpu()
    fsdp_raw = data.batch["old_log_probs"].detach().float().cpu()
    fsdp_sampling = data.batch.get("old_log_probs_sampling_temperature")
    if fsdp_sampling is not None:
        fsdp_sampling = fsdp_sampling.detach().float().cpu()
    responses = data.batch["responses"].detach().cpu()
    response_mask = data.batch["response_mask"].detach().bool().cpu()
    attention_mask = data.batch["attention_mask"].detach().cpu()
    input_ids = data.batch["input_ids"].detach().cpu()
    position_ids = data.batch["position_ids"].detach().cpu()
    prompts = data.batch.get("prompts")
    if prompts is not None:
        prompts = prompts.detach().cpu()
    pad_id = getattr(tokenizer, "pad_token_id", None)
    eos_id = getattr(tokenizer, "eos_token_id", None)

    post_fix = {
        "tag": "d0_4_post_fix_raw_logprob",
        "global_step": int(global_step),
        "semantics": "rollout raw vLLM sampled-token logprob vs FSDP log_softmax(logits / logprob_temperature)",
        "logprob_temperature": data.meta_info.get("temperature"),
        "sampling_temperature": data.meta_info.get("sampling_temperature"),
        "overall": _d0_4_pair_stats(rollout, fsdp_raw, response_mask),
        "shift_ablation": _d0_4_shift_stats(rollout, fsdp_raw, response_mask),
    }
    pre_fix = None
    if fsdp_sampling is not None:
        pre_fix = {
            "tag": "d0_4_pre_fix_sampling_temperature_logprob",
            "global_step": int(global_step),
            "semantics": "rollout raw vLLM sampled-token logprob vs legacy FSDP log_softmax(logits / sampling_temperature)",
            "logprob_temperature": data.meta_info.get("sampling_temperature"),
            "sampling_temperature": data.meta_info.get("sampling_temperature"),
            "overall": _d0_4_pair_stats(rollout, fsdp_sampling, response_mask),
            "shift_ablation": _d0_4_shift_stats(rollout, fsdp_sampling, response_mask),
        }

    request_ids = data.non_tensor_batch.get("d0_4_request_id")
    manifest = {
        "global_step": int(global_step),
        "batch_size": int(responses.shape[0]),
        "response_shape": list(responses.shape),
        "input_shape": list(input_ids.shape),
        "attention_mask_shape": list(attention_mask.shape),
        "position_ids_shape": list(position_ids.shape),
        "rollout_temperature": data.meta_info.get("sampling_temperature"),
        "fsdp_logprob_temperature": data.meta_info.get("temperature"),
        "top_p": None,
        "top_k": None,
        "pad_token_id": pad_id,
        "eos_token_id": eos_id,
        "samples": [],
    }
    if "multi_modal_inputs" in data.non_tensor_batch:
        manifest["multi_modal_inputs"] = []
        for i, mmi in enumerate(data.non_tensor_batch["multi_modal_inputs"]):
            rec = {"sample": int(i), "keys": sorted(list(mmi.keys())) if isinstance(mmi, dict) else None}
            if isinstance(mmi, dict):
                for key in ("pixel_values", "nfp_pixel_values", "nfp_loss_mask"):
                    if key in mmi and torch.is_tensor(mmi[key]):
                        rec[key] = _d0_4_tensor_digest(mmi[key])
            manifest["multi_modal_inputs"].append(rec)

    rows = []
    top_candidates = []
    token_type_masks: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for b in range(responses.shape[0]):
        active_len = int(response_mask[b].sum().item())
        resp_ids = [int(x) for x in responses[b].tolist()]
        compact_resp_ids = [tid for tid, m in zip(resp_ids, response_mask[b].tolist()) if m]
        full_text = tokenizer.decode(compact_resp_ids, skip_special_tokens=False) if compact_resp_ids else ""
        text_l = full_text.lower()
        spans = []
        cursor = 0
        for j in range(len(resp_ids)):
            prefix = tokenizer.decode([tid for tid, m in zip(resp_ids[:j], response_mask[b, :j].tolist()) if m], skip_special_tokens=False)
            piece = tokenizer.decode([resp_ids[j]], skip_special_tokens=False) if response_mask[b, j] else ""
            spans.append((len(prefix), len(prefix) + len(piece), piece))
        prompt_len = int(input_ids.shape[1] - responses.shape[1])
        prompt_hash = None
        if prompts is not None:
            prompt_hash = hashlib.sha1(np.asarray(prompts[b].tolist(), dtype=np.int64).tobytes()).hexdigest()
        manifest["samples"].append(
            {
                "sample": int(b),
                "request_id": None if request_ids is None else str(request_ids[b]),
                "response_active_tokens": active_len,
                "prompt_token_count": None if prompts is None else int((prompts[b] != (pad_id or 0)).sum().item()),
                "prompt_hash": prompt_hash,
                "response_hash": hashlib.sha1(np.asarray(resp_ids, dtype=np.int64).tobytes()).hexdigest(),
                "image_token_positions": [int(i) for i, tid in enumerate(input_ids[b].tolist()) if int(tid) == -200],
                "position_id_first_response": int(position_ids[b, prompt_len].item()) if position_ids.dim() == 2 and prompt_len < position_ids.shape[1] else None,
            }
        )
        for j in range(responses.shape[1]):
            tid = resp_ids[j]
            start, end, piece = spans[j]
            token_type = _d0_4_token_type(tid, start, end, text_l, eos_id, pad_id)
            token_type_masks[token_type].append((b, j))
            raw_delta = float(fsdp_raw[b, j].item() - rollout[b, j].item())
            sampling_logp = None if fsdp_sampling is None else float(fsdp_sampling[b, j].item())
            sampling_delta = None if fsdp_sampling is None else float(fsdp_sampling[b, j].item() - rollout[b, j].item())
            abs_pos = prompt_len + j
            pos_id = None
            if position_ids.dim() == 2 and abs_pos < position_ids.shape[1]:
                pos_id = int(position_ids[b, abs_pos].item())
            row = {
                "sample": b,
                "request_id": None if request_ids is None else str(request_ids[b]),
                "idx": j,
                "token_id": tid,
                "decoded_token": piece,
                "response_mask": int(response_mask[b, j].item()),
                "absolute_sequence_position": abs_pos,
                "position_id": pos_id,
                "token_type": token_type,
                "vllm_logp": float(rollout[b, j].item()),
                "fsdp_logp_raw_T1": float(fsdp_raw[b, j].item()),
                "delta_raw_T1": raw_delta,
                "ratio_raw_T1": float(np.exp(np.clip(raw_delta, -80.0, 80.0))),
                "fsdp_logp_sampling_T": sampling_logp,
                "delta_sampling_T": sampling_delta,
                "ratio_sampling_T": None if sampling_delta is None else float(np.exp(np.clip(sampling_delta, -80.0, 80.0))),
            }
            rows.append(row)
            if response_mask[b, j]:
                top_candidates.append((abs(raw_delta), "raw_T1", b, j, row))
                if sampling_delta is not None:
                    top_candidates.append((abs(sampling_delta), "sampling_T", b, j, row))

    type_stats = {}
    for token_type, pairs in token_type_masks.items():
        if not pairs:
            continue
        mask = torch.zeros_like(response_mask)
        for b, j in pairs:
            mask[b, j] = response_mask[b, j]
        type_stats[token_type] = _d0_4_pair_stats(rollout, fsdp_raw, mask)
    post_fix["token_type_stats"] = type_stats
    if pre_fix is not None and fsdp_sampling is not None:
        pre_fix["token_type_stats"] = {}
        for token_type, pairs in token_type_masks.items():
            mask = torch.zeros_like(response_mask)
            for b, j in pairs:
                mask[b, j] = response_mask[b, j]
            pre_fix["token_type_stats"][token_type] = _d0_4_pair_stats(rollout, fsdp_sampling, mask)

    d0_6_dir = os.environ.get("VAGEN_D0_6_DIR")
    if d0_6_dir:
        per_sample = []
        for b in range(responses.shape[0]):
            valid = response_mask[b]
            if not bool(valid.any()):
                continue
            delta = (fsdp_raw[b] - rollout[b]).abs()
            per_sample.append(
                {
                    "sample": int(b),
                    "mean_abs_delta": float(delta[valid].mean().item()),
                    "max_abs_delta": float(delta[valid].max().item()),
                }
            )
        selected = []
        if per_sample:
            selected.append(min(per_sample, key=lambda x: x["mean_abs_delta"]))
            selected.append(max(per_sample, key=lambda x: x["max_abs_delta"]))
        selected_samples = []
        for item in selected:
            b = item["sample"]
            if b in {s["sample"] for s in selected_samples}:
                continue
            prompt_len = int(input_ids.shape[1] - responses.shape[1])
            resp_ids = [int(x) for x in responses[b].tolist()]
            active_resp_ids = [tid for tid, m in zip(resp_ids, response_mask[b].tolist()) if m]
            prompt_ids = [int(x) for x in input_ids[b, :prompt_len].tolist()]
            full_input_ids = [int(x) for x in input_ids[b].tolist()]
            prompt_attention = [int(x) for x in attention_mask[b, :prompt_len].tolist()]
            full_attention = [int(x) for x in attention_mask[b].tolist()]
            decoded_prompt, prompt_image_blocks = _d0_6_decode_with_image_blocks(
                tokenizer,
                prompt_ids,
                attention_mask=prompt_attention,
                pad_id=pad_id,
            )
            decoded_full, full_image_blocks = _d0_6_decode_with_image_blocks(
                tokenizer,
                full_input_ids,
                attention_mask=full_attention,
                pad_id=pad_id,
            )
            mmi_rec = None
            if "multi_modal_inputs" in data.non_tensor_batch:
                raw_mmi = data.non_tensor_batch["multi_modal_inputs"][b]
                if isinstance(raw_mmi, dict):
                    mmi_rec = {
                        key: _d0_4_tensor_digest(raw_mmi[key])
                        for key in ("pixel_values", "nfp_pixel_values", "nfp_loss_mask")
                        if key in raw_mmi and torch.is_tensor(raw_mmi[key])
                    }
            selected_samples.append(
                {
                    "selection": "normal_sample" if item["mean_abs_delta"] == min(x["mean_abs_delta"] for x in per_sample) else "top_mismatch_sample",
                    "sample": int(b),
                    "request_id": None if request_ids is None else str(request_ids[b]),
                    "mean_abs_delta": item["mean_abs_delta"],
                    "max_abs_delta": item["max_abs_delta"],
                    "input_ids": full_input_ids,
                    "prompt_token_ids": prompt_ids,
                    "response_token_ids": resp_ids,
                    "active_response_token_ids": active_resp_ids,
                    "response_mask": [int(x) for x in response_mask[b].tolist()],
                    "attention_mask": full_attention,
                    "position_ids": [int(x) for x in position_ids[b].tolist()],
                    "compact_image_token_positions": [int(i) for i, tid in enumerate(prompt_ids) if int(tid) == -200],
                    "expanded_image_token_positions": [int(i) for i, tid in enumerate(full_input_ids) if int(tid) == -200],
                    "prompt_image_blocks": prompt_image_blocks,
                    "full_image_blocks": full_image_blocks,
                    "image_order": [0],
                    "image_artifact": mmi_rec,
                    "exact_decoded_prompt_image_blocks": decoded_prompt,
                    "exact_decoded_response": tokenizer.decode(active_resp_ids, skip_special_tokens=False),
                    "exact_decoded_full_image_blocks": decoded_full,
                    "vllm_rollout_log_probs": [float(x) for x in rollout[b].tolist()],
                    "fsdp_log_probs_T1": [float(x) for x in fsdp_raw[b].tolist()],
                    "delta_log_probs_T1": [float(x) for x in (fsdp_raw[b] - rollout[b]).tolist()],
                    "top_residual_tokens": [
                        dict(row)
                        for _abs_delta, variant, tb, _tj, row in sorted(top_candidates, key=lambda x: x[0], reverse=True)
                        if tb == b and variant == "raw_T1"
                    ][:10],
                }
            )
        d0_6_manifest = {
            "tag": "d0_6_fixed_sequence_manifest",
            "source": "production DataProto captured after rollout and before PPO update",
            "model_checkpoint": os.environ.get("VAGEN_D0_6_MODEL_CHECKPOINT"),
            "global_step": int(global_step),
            "sampling_temperature": data.meta_info.get("sampling_temperature"),
            "logprob_temperature": data.meta_info.get("temperature"),
            "random_seed": int(os.environ.get("PYTHONHASHSEED", "0")),
            "input_shape": list(input_ids.shape),
            "attention_mask_shape": list(attention_mask.shape),
            "position_ids_shape": list(position_ids.shape),
            "response_shape": list(responses.shape),
            "samples": selected_samples,
        }
        with open(os.path.join(d0_6_dir, "d0_6_fixed_sequence_manifest.json"), "w", encoding="utf-8") as f:
            json.dump(d0_6_manifest, f, indent=2, ensure_ascii=False, sort_keys=True)
        d0_6_docs_dir = os.environ.get("VAGEN_D0_6_DOCS_DIR")
        if d0_6_docs_dir:
            os.makedirs(d0_6_docs_dir, exist_ok=True)
            with open(os.path.join(d0_6_docs_dir, "d0_6_fixed_sequence_manifest.json"), "w", encoding="utf-8") as f:
                json.dump(d0_6_manifest, f, indent=2, ensure_ascii=False, sort_keys=True)

    with open(os.path.join(path, "d0_4_fixed_batch_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False, sort_keys=True)
    with open(os.path.join(path, "d0_4_post_fix_parity.json"), "w", encoding="utf-8") as f:
        json.dump(post_fix, f, indent=2, ensure_ascii=False, sort_keys=True)
    if pre_fix is not None:
        with open(os.path.join(path, "d0_4_pre_fix_parity.json"), "w", encoding="utf-8") as f:
            json.dump(pre_fix, f, indent=2, ensure_ascii=False, sort_keys=True)

    csv_path = os.path.join(path, "d0_4_token_alignment.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)

    top = []
    for _abs_delta, variant, b, j, row in sorted(top_candidates, key=lambda x: x[0], reverse=True)[:20]:
        prev_ids = [int(x) for x in responses[b, max(0, j - 5):j].tolist()]
        next_ids = [int(x) for x in responses[b, j + 1:min(responses.shape[1], j + 6)].tolist()]
        rec = dict(row)
        rec["variant"] = variant
        rec["previous_5_tokens"] = tokenizer.decode(prev_ids, skip_special_tokens=False)
        rec["next_5_tokens"] = tokenizer.decode(next_ids, skip_special_tokens=False)
        top.append(rec)
    with open(os.path.join(path, "d0_4_top_mismatch_tokens.json"), "w", encoding="utf-8") as f:
        json.dump(top, f, indent=2, ensure_ascii=False, sort_keys=True)

    metrics = {}
    for prefix, payload in (("d0_4/post_fix", post_fix), ("d0_4/pre_fix", pre_fix)):
        if payload is None:
            continue
        for key, val in payload["overall"].items():
            if isinstance(val, (int, float)):
                metrics[f"{prefix}/{key}"] = val
    return metrics


def _d0_15_write_pre_update_ratio_audit(
    batch: DataProto,
    current_log_prob: DataProto,
    path: str,
    tokenizer,
    global_step: int,
    *,
    actor_config=None,
    rollout_corr_config=None,
) -> dict:
    if "rollout_log_probs" not in batch.batch or "old_log_probs" not in batch.batch:
        return {}
    if "old_log_probs" not in current_log_prob.batch:
        return {}

    os.makedirs(path, exist_ok=True)
    rollout = batch.batch["rollout_log_probs"].detach().float().cpu()
    old = batch.batch["old_log_probs"].detach().float().cpu()
    current = current_log_prob.batch["old_log_probs"].detach().float().cpu()
    response_mask = batch.batch["response_mask"].detach().bool().cpu()
    responses = batch.batch["responses"].detach().cpu()
    advantages = batch.batch.get("advantages")
    if advantages is not None:
        advantages = advantages.detach().float().cpu()

    pad_id = getattr(tokenizer, "pad_token_id", None)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    token_type_masks: dict[str, torch.Tensor] = defaultdict(lambda: torch.zeros_like(response_mask))
    token_rows = []
    for b in range(responses.shape[0]):
        resp_ids = [int(x) for x in responses[b].tolist()]
        compact_resp_ids = [tid for tid, m in zip(resp_ids, response_mask[b].tolist()) if m]
        full_text = tokenizer.decode(compact_resp_ids, skip_special_tokens=False) if compact_resp_ids else ""
        text_l = full_text.lower()
        spans = []
        for j in range(len(resp_ids)):
            prefix = tokenizer.decode(
                [tid for tid, m in zip(resp_ids[:j], response_mask[b, :j].tolist()) if m],
                skip_special_tokens=False,
            )
            piece = tokenizer.decode([resp_ids[j]], skip_special_tokens=False) if response_mask[b, j] else ""
            spans.append((len(prefix), len(prefix) + len(piece), piece))
        for j, tid in enumerate(resp_ids):
            start, end, piece = spans[j]
            token_type = _d0_4_token_type(tid, start, end, text_l, eos_id, pad_id)
            token_type_masks[token_type][b, j] = response_mask[b, j]
            if response_mask[b, j]:
                delta = float(current[b, j].item() - rollout[b, j].item())
                token_rows.append(
                    {
                        "sample": int(b),
                        "idx": int(j),
                        "token_id": int(tid),
                        "decoded_token": piece,
                        "token_type": token_type,
                        "rollout_logp": float(rollout[b, j].item()),
                        "current_hf_logp": float(current[b, j].item()),
                        "old_logp_consumed": float(old[b, j].item()),
                        "delta_current_minus_rollout": delta,
                        "ratio_current_over_rollout": float(np.exp(np.clip(delta, -80.0, 80.0))),
                        "advantage": None if advantages is None else float(advantages[b, j].item()),
                    }
                )

    def _union_mask(*names: str) -> torch.Tensor:
        mask = torch.zeros_like(response_mask)
        for name in names:
            if name in token_type_masks:
                mask |= token_type_masks[name].bool()
        return mask & response_mask

    split_masks = {
        "all_response": response_mask,
        "think_text": _union_mask("think_text"),
        "action_tag+name": _union_mask("action_tag", "action_name"),
        "action_name": _union_mask("action_name"),
    }
    split_stats = {name: _d0_4_pair_stats(rollout, current, mask) for name, mask in split_masks.items()}
    old_vs_rollout = _d0_4_pair_stats(rollout, old, response_mask)

    ratio = torch.exp((current - rollout).clamp(min=-80.0, max=80.0))
    clip_ratio = 0.2
    clip_ratio_low = clip_ratio
    clip_ratio_high = clip_ratio
    if actor_config is not None:
        clip_ratio = float(actor_config.get("clip_ratio", clip_ratio))
        clip_ratio_low = float(actor_config.get("clip_ratio_low", clip_ratio) or clip_ratio)
        clip_ratio_high = float(actor_config.get("clip_ratio_high", clip_ratio) or clip_ratio)
    valid_ratio = ratio[response_mask]
    bypass_ratio_outside_clip = float(
        ((valid_ratio < (1.0 - clip_ratio_low)) | (valid_ratio > (1.0 + clip_ratio_high))).float().mean().item()
    ) if valid_ratio.numel() else 0.0

    is_threshold = 2.0
    rollout_is_mode = None
    rollout_is_batch_normalize = False
    if rollout_corr_config is not None:
        rollout_is_mode = rollout_corr_config.get("rollout_is", None)
        is_threshold = float(rollout_corr_config.get("rollout_is_threshold", is_threshold))
        rollout_is_batch_normalize = bool(rollout_corr_config.get("rollout_is_batch_normalize", False))
    token_is_clipped = ratio.clamp(max=is_threshold)
    if rollout_is_batch_normalize and bool(response_mask.any()):
        mean_weight = token_is_clipped[response_mask].mean()
        if mean_weight.item() > 1e-8:
            token_is_clipped = token_is_clipped / mean_weight

    correction_comparison = {
        "A_bypass_mode_true": {
            "old_log_prob_source": "rollout_log_probs",
            "ppo_ratio": split_stats,
            "ratio_outside_ppo_clip_range_all_response": bypass_ratio_outside_clip,
            "changes_ppo_clip_anchor": True,
        },
        "B_decoupled_rollout_correction_monitor_only": {
            "old_log_prob_source": "HF/FSDP recompute",
            "ppo_ratio_before_update": "exactly 1.0 by construction (current_log_prob == recomputed old_log_prob before optimizer.step)",
            "rollout_is_weights_applied": False,
            "note": "This is what bypass_mode=False with rollout_is=null does: it monitors rollout mismatch but does not importance-weight the loss.",
        },
        "B_decoupled_token_IS_if_enabled": {
            "old_log_prob_source": "HF/FSDP recompute",
            "ppo_ratio_before_update": "exactly 1.0 by construction",
            "rollout_is_weight_source": "exp(HF_current_log_prob - vLLM_rollout_log_prob)",
            "configured_rollout_is": rollout_is_mode,
            "configured_rollout_is_threshold": is_threshold,
            "configured_batch_normalize": rollout_is_batch_normalize,
            "unclipped_is_weight_stats": split_stats,
            "clipped_token_is_weight_all_response": _d0_4_pair_stats(torch.zeros_like(token_is_clipped), torch.log(token_is_clipped.clamp(min=1e-12)), response_mask),
        },
    }

    top = sorted(token_rows, key=lambda row: abs(row["delta_current_minus_rollout"]), reverse=True)[:25]
    payload = {
        "tag": "d0_15_pre_update_ppo_ratio_audit",
        "global_step": int(global_step),
        "semantics": "current_log_prob=HF/FSDP pre-update differentiable-policy path; old_log_prob=vLLM rollout behavior logprob",
        "batch_shape": {
            "responses": list(responses.shape),
            "response_mask": list(response_mask.shape),
        },
        "old_log_probs_equals_rollout_log_probs": old_vs_rollout,
        "current_vs_rollout_split_stats": split_stats,
        "minimal_comparison": correction_comparison,
        "top_abs_delta_tokens": top,
    }
    out_file = os.path.join(path, f"d0_15_pre_update_ratio_audit_step{global_step}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=True)

    metric_out = {}
    for split, stats in split_stats.items():
        prefix = f"d0_15/{split}"
        for key in ("mean_abs_delta_logp", "median_abs_delta_logp", "p95_abs_delta_logp", "ratio_median", "ratio_p5", "ratio_p95", "frac_abs_ratio_minus_1_gt_0p01", "frac_abs_ratio_minus_1_gt_0p05"):
            if key in stats and isinstance(stats[key], (int, float)):
                metric_out[f"{prefix}/{key}"] = stats[key]
    metric_out["d0_15/old_equals_rollout_max_abs_delta_logp"] = old_vs_rollout.get("delta_logp_max_abs", 0.0)
    metric_out["d0_15/bypass_ratio_outside_ppo_clip_range"] = bypass_ratio_outside_clip
    return metric_out


def _d0_16_weight_stats(weights: torch.Tensor, mask: torch.Tensor) -> dict:
    valid = mask.bool()
    if weights.shape != mask.shape:
        return {"shape_mismatch": 1, "weights_shape": list(weights.shape), "mask_shape": list(mask.shape)}
    if valid.sum().item() == 0:
        return {"token_count": 0}
    vals = weights.detach().float()[valid]
    ess = (vals.sum() * vals.sum()) / (vals.square().sum() + 1e-12)
    return {
        "token_count": int(vals.numel()),
        "mean": float(vals.mean().item()),
        "median": float(vals.median().item()),
        "p5": _d0_3_quantile(vals, 0.05),
        "p95": _d0_3_quantile(vals, 0.95),
        "max": float(vals.max().item()),
        "min": float(vals.min().item()),
        "ess": float(ess.item()),
        "ess_over_n": float((ess / max(vals.numel(), 1)).item()),
        "frac_abs_weight_minus_1_gt_0p01": float(((vals - 1.0).abs() > 0.01).float().mean().item()),
        "frac_abs_weight_minus_1_gt_0p05": float(((vals - 1.0).abs() > 0.05).float().mean().item()),
    }


def _d0_16_write_decoupled_correction_audit(
    batch: DataProto,
    current_log_prob: DataProto,
    path: str,
    tokenizer,
    global_step: int,
    *,
    response_mask_before_correction: torch.Tensor | None = None,
    rollout_corr_config=None,
) -> dict:
    required = ("rollout_log_probs", "old_log_probs", "response_mask", "responses")
    if any(key not in batch.batch for key in required):
        return {}
    if "old_log_probs" not in current_log_prob.batch:
        return {}

    os.makedirs(path, exist_ok=True)
    rollout = batch.batch["rollout_log_probs"].detach().float().cpu()
    old = batch.batch["old_log_probs"].detach().float().cpu()
    current = current_log_prob.batch["old_log_probs"].detach().float().cpu()
    response_mask = batch.batch["response_mask"].detach().bool().cpu()
    original_response_mask = (
        response_mask_before_correction.detach().bool().cpu()
        if response_mask_before_correction is not None
        else response_mask
    )
    responses = batch.batch["responses"].detach().cpu()
    rollout_is_weights = batch.batch.get("rollout_is_weights")
    if rollout_is_weights is not None:
        rollout_is_weights = rollout_is_weights.detach().float().cpu()

    pad_id = getattr(tokenizer, "pad_token_id", None)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    token_type_masks: dict[str, torch.Tensor] = defaultdict(lambda: torch.zeros_like(response_mask))
    token_rows = []
    for b in range(responses.shape[0]):
        resp_ids = [int(x) for x in responses[b].tolist()]
        compact_resp_ids = [tid for tid, m in zip(resp_ids, response_mask[b].tolist()) if m]
        full_text = tokenizer.decode(compact_resp_ids, skip_special_tokens=False) if compact_resp_ids else ""
        text_l = full_text.lower()
        spans = []
        for j in range(len(resp_ids)):
            prefix = tokenizer.decode(
                [tid for tid, m in zip(resp_ids[:j], response_mask[b, :j].tolist()) if m],
                skip_special_tokens=False,
            )
            piece = tokenizer.decode([resp_ids[j]], skip_special_tokens=False) if response_mask[b, j] else ""
            spans.append((len(prefix), len(prefix) + len(piece), piece))
        for j, tid in enumerate(resp_ids):
            start, end, piece = spans[j]
            token_type = _d0_4_token_type(tid, start, end, text_l, eos_id, pad_id)
            token_type_masks[token_type][b, j] = response_mask[b, j]
            if response_mask[b, j]:
                a_log = float(current[b, j].item() - rollout[b, j].item())
                b_log = float(current[b, j].item() - old[b, j].item())
                c_log = float(old[b, j].item() - rollout[b, j].item())
                token_rows.append(
                    {
                        "sample": int(b),
                        "idx": int(j),
                        "token_id": int(tid),
                        "decoded_token": piece,
                        "token_type": token_type,
                        "rollout_vllm_logp": float(rollout[b, j].item()),
                        "old_hf_logp": float(old[b, j].item()),
                        "current_hf_logp": float(current[b, j].item()),
                        "A_log_current_minus_rollout": a_log,
                        "B_log_current_minus_old": b_log,
                        "C_log_old_minus_rollout": c_log,
                        "A_ratio": float(np.exp(np.clip(a_log, -80.0, 80.0))),
                        "B_ratio": float(np.exp(np.clip(b_log, -80.0, 80.0))),
                        "C_is_unclipped": float(np.exp(np.clip(c_log, -80.0, 80.0))),
                        "rollout_is_weight_tensor": None
                        if rollout_is_weights is None
                        else float(rollout_is_weights[b, j].item()),
                    }
                )

    def _union_mask(*names: str) -> torch.Tensor:
        mask = torch.zeros_like(response_mask)
        for name in names:
            if name in token_type_masks:
                mask |= token_type_masks[name].bool()
        return mask & response_mask

    split_masks = {
        "all_response": response_mask,
        "think_text": _union_mask("think_text"),
        "action_tag+name": _union_mask("action_tag", "action_name"),
        "action_name": _union_mask("action_name"),
    }

    a_stats = {name: _d0_4_pair_stats(rollout, current, mask) for name, mask in split_masks.items()}
    b_stats = {name: _d0_4_pair_stats(old, current, mask) for name, mask in split_masks.items()}
    c_stats = {name: _d0_4_pair_stats(rollout, old, mask) for name, mask in split_masks.items()}

    identity_log_err = (current - rollout) - ((current - old) + (old - rollout))
    identity_ratio_err = torch.exp((current - rollout).clamp(min=-80.0, max=80.0)) - (
        torch.exp((current - old).clamp(min=-80.0, max=80.0))
        * torch.exp((old - rollout).clamp(min=-80.0, max=80.0))
    )
    identity = {}
    for name, mask in split_masks.items():
        valid = mask.bool()
        if valid.sum().item() == 0:
            identity[name] = {"token_count": 0}
            continue
        log_abs = identity_log_err.float()[valid].abs()
        ratio_abs = identity_ratio_err.float()[valid].abs()
        identity[name] = {
            "token_count": int(log_abs.numel()),
            "max_abs_log_error": float(log_abs.max().item()),
            "mean_abs_log_error": float(log_abs.mean().item()),
            "max_abs_ratio_error": float(ratio_abs.max().item()),
            "mean_abs_ratio_error": float(ratio_abs.mean().item()),
        }

    rollout_is_threshold = 2.0
    rollout_is_mode = None
    rollout_rs_mode = None
    if rollout_corr_config is not None:
        rollout_is_threshold = float(rollout_corr_config.get("rollout_is_threshold", rollout_is_threshold))
        rollout_is_mode = rollout_corr_config.get("rollout_is", None)
        rollout_rs_mode = rollout_corr_config.get("rollout_rs", None)

    c_unclipped = torch.exp((old - rollout).clamp(min=-80.0, max=80.0))
    is_weight_stats = {
        name: _d0_16_weight_stats(
            rollout_is_weights if rollout_is_weights is not None else c_unclipped.clamp(max=rollout_is_threshold),
            mask,
        )
        for name, mask in split_masks.items()
    }
    unclipped_is_stats = {name: _d0_16_weight_stats(c_unclipped, mask) for name, mask in split_masks.items()}
    clip_fraction = {}
    for name, mask in split_masks.items():
        valid = mask.bool()
        clip_fraction[name] = (
            0.0
            if valid.sum().item() == 0
            else float((c_unclipped[valid] > rollout_is_threshold).float().mean().item())
        )

    accepted = response_mask.float().sum().item()
    original = original_response_mask.float().sum().item()
    rs_acceptance_fraction = 1.0 if original <= 0 else float(accepted / original)

    payload = {
        "tag": "d0_16_decoupled_rollout_correction_audit",
        "global_step": int(global_step),
        "semantics": {
            "A_bypass_ratio": "exp(current_HF - rollout_vLLM)",
            "B_ppo_proximal_ratio": "exp(current_HF - old_HF)",
            "C_rollout_is_weight": "exp(old_HF - rollout_vLLM)",
            "identity": "A == B * C in log space by tensor construction",
        },
        "config": {
            "bypass_mode": None if rollout_corr_config is None else bool(rollout_corr_config.get("bypass_mode", False)),
            "rollout_is": rollout_is_mode,
            "rollout_is_threshold": rollout_is_threshold,
            "rollout_rs": rollout_rs_mode,
            "has_rollout_is_weights_tensor": rollout_is_weights is not None,
        },
        "A_bypass_current_HF_vs_rollout_vLLM": a_stats,
        "B_ppo_proximal_current_HF_vs_old_HF": b_stats,
        "C_old_HF_vs_rollout_vLLM": c_stats,
        "identity_A_equals_B_times_C": identity,
        "C_unclipped_is_weight_stats": unclipped_is_stats,
        "rollout_is_weight_tensor_stats": is_weight_stats,
        "rollout_is_clip_fraction": clip_fraction,
        "rs_acceptance_fraction": rs_acceptance_fraction,
        "top_abs_A_tokens": sorted(token_rows, key=lambda row: abs(row["A_log_current_minus_rollout"]), reverse=True)[:25],
    }
    out_file = os.path.join(path, f"d0_16_decoupled_correction_audit_step{global_step}.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, sort_keys=True)

    metric_out = {}
    for split, stats in a_stats.items():
        for key in ("mean_abs_delta_logp", "ratio_median", "frac_abs_ratio_minus_1_gt_0p05"):
            if key in stats:
                metric_out[f"d0_16/A/{split}/{key}"] = stats[key]
    for split, stats in b_stats.items():
        for key in ("mean_abs_delta_logp", "ratio_median", "frac_abs_ratio_minus_1_gt_0p05"):
            if key in stats:
                metric_out[f"d0_16/B/{split}/{key}"] = stats[key]
    metric_out["d0_16/rs_acceptance_fraction"] = rs_acceptance_fraction
    metric_out["d0_16/is_ess_over_n_all_response"] = is_weight_stats.get("all_response", {}).get("ess_over_n", 0.0)
    metric_out["d0_16/identity_max_abs_log_error_all_response"] = identity.get("all_response", {}).get(
        "max_abs_log_error", 0.0
    )
    return metric_out


def compute_custom_metrics(data: DataProto, prefix: str = "custom_metrics") -> dict:
    """Compute all custom metrics registered in METRIC_REGISTRY.

    Args:
        data (DataProto): The data containing batch information.
        prefix (str): Prefix for metric names in the returned dictionary.

    Returns:
        dict: A dictionary containing all computed custom metrics with appropriate prefixes.
    """
    custom_metrics = {}

    for metric_name, metric_fn in METRIC_REGISTRY.items():
        try:
            metric_value = metric_fn(data)
            custom_metrics[f"{prefix}/{metric_name}"] = metric_value
        except Exception as e:
            print(f"Warning: Failed to compute custom metric '{metric_name}': {e}")

    return custom_metrics

def _default_eps(
    x: torch.Tensor,
    small_eps: float = 1e-2,
    large_eps: float = 1e-6,
) -> float:
    """
    Choose a comparison tolerance (eps) based on tensor dtype.
    """
    if x.dtype in (torch.float16, torch.bfloat16):
        return small_eps
    return large_eps


def compute_value_mask(
    data: DataProto,
    ignore_value: float = -100.0,
    eps: float | None = None,
) -> torch.Tensor:
    """Compute mask for token positions whose return is real, not sentinel."""
    returns = data.batch["returns"]

    if eps is None:
        eps = _default_eps(returns)

    is_ignored = (returns - ignore_value).abs() < eps
    return (~is_ignored).to(dtype=data.batch["response_mask"].dtype)


def compute_unique_sample_mask(data: DataProto) -> np.ndarray:
    """Mask first occurrence of each no-concat sample, excluding pad duplicates."""
    keys = ("group_idx", "traj_idx", "turn_idx")
    if not all(k in data.non_tensor_batch for k in keys):
        bsz = len(next(iter(data.non_tensor_batch.values()))) if data.non_tensor_batch else len(data.batch["responses"])
        return np.ones(int(bsz), dtype=bool)

    cols = [np.asarray(data.non_tensor_batch[k]).astype(str) for k in keys]
    triples = np.stack(cols, axis=1)
    _, first_idx = np.unique(triples, axis=0, return_index=True)
    mask = np.zeros(triples.shape[0], dtype=bool)
    mask[first_idx] = True
    return mask


def compute_valid_return_metrics(data: DataProto, ignore_value: float = -100.0) -> dict[str, float]:
    """Log return stats on real value-supervision positions only."""
    if "returns" not in data.batch or "response_mask" not in data.batch:
        return {}
    value_mask = compute_value_mask(data, ignore_value=ignore_value).bool()
    response_mask = data.batch["response_mask"].bool()
    valid_mask = value_mask & response_mask
    response_count = int(response_mask.sum().detach().item())
    valid_count = int(valid_mask.sum().detach().item())
    sentinel_count = max(response_count - valid_count, 0)
    metrics = {
        "critic/returns/valid_count": valid_count,
        "critic/returns/sentinel_ratio": float(sentinel_count / max(response_count, 1)),
    }
    if "attention_mask" in data.batch:
        response_width = int(response_mask.shape[-1])
        attention_response_count = int(
            data.batch["attention_mask"][:, -response_width:].bool().sum().detach().item()
        )
        metrics["actor/valid_response_token_ratio"] = float(
            response_count / max(attention_response_count, 1)
        )
    if "advantages" in data.batch and response_count > 0:
        valid_advantages = torch.masked_select(data.batch["advantages"], response_mask)
        metrics.update({
            "critic/advantages/valid_mean": torch.mean(valid_advantages.float()).detach().item(),
            "critic/advantages/valid_std": torch.std(valid_advantages.float(), unbiased=False).detach().item(),
        })
    if valid_count > 0:
        valid_returns = torch.masked_select(data.batch["returns"], valid_mask)
        metrics.update({
            "critic/returns/valid_mean": torch.mean(valid_returns).detach().item(),
            "critic/returns/valid_std": torch.std(valid_returns.float(), unbiased=False).detach().item(),
            "critic/returns/valid_min": torch.min(valid_returns).detach().item(),
            "critic/returns/valid_max": torch.max(valid_returns).detach().item(),
        })
        if "values" in data.batch:
            valid_values = torch.masked_select(data.batch["values"], valid_mask)
            value_error = valid_values.float() - valid_returns.float()
            return_var = torch.var(valid_returns.float())
            metrics.update({
                "critic/values/valid_mean": torch.mean(valid_values).detach().item(),
                "critic/values/valid_min": torch.min(valid_values).detach().item(),
                "critic/values/valid_max": torch.max(valid_values).detach().item(),
                "critic/values/mae_valid": torch.mean(value_error.abs()).detach().item(),
                "critic/values/rmse_valid": torch.sqrt(torch.mean(value_error.square())).detach().item(),
                "critic/values/explained_var_valid": (
                    1.0 - torch.var(value_error) / (return_var + 1e-5)
                ).detach().item(),
            })
    return metrics


_ACTION_TAG_RE = re.compile(r"<action>(.*?)</action>", re.IGNORECASE | re.DOTALL)


def _extract_action_sequence(text: str) -> tuple[str, ...]:
    """Normalize all action tags in a generated trajectory."""
    actions: list[str] = []
    for match in _ACTION_TAG_RE.finditer(text or ""):
        raw = match.group(1)
        actions.extend(a.strip().lower() for a in raw.split("|") if a.strip())
    return tuple(actions)


def _stable_text_hash(text: str) -> str:
    normalized = " ".join((text or "").split())
    return hashlib.sha1(normalized.encode("utf-8")).hexdigest()


def compute_validation_diversity_metrics(
    data_sources: np.ndarray,
    sample_uids: list,
    sample_outputs: list[str],
) -> dict[str, float]:
    """Measure whether repeated validation rollouts are actually diverse."""
    source_uid_to_outputs: dict[tuple[str, str], list[str]] = defaultdict(list)
    for data_source, uid, output in zip(data_sources, sample_uids, sample_outputs, strict=False):
        source_uid_to_outputs[(str(data_source), str(uid))].append(output)

    source_to_stats: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for (data_source, _uid), outputs in source_uid_to_outputs.items():
        if not outputs:
            continue
        action_sequences = [_extract_action_sequence(output) for output in outputs]
        trajectory_hashes = [_stable_text_hash(output) for output in outputs]
        first_actions = [seq[0] if seq else "<empty>" for seq in action_sequences]
        first_action_counts = defaultdict(int)
        for action in first_actions:
            first_action_counts[action] += 1

        n = len(outputs)
        source_to_stats[data_source]["unique_action_sequence_count@4"].append(float(len(set(action_sequences))))
        source_to_stats[data_source]["trajectory_hash_unique@4"].append(float(len(set(trajectory_hashes))))
        source_to_stats[data_source]["all_four_identical_ratio"].append(
            float(n >= 4 and len(set(action_sequences[:4])) == 1)
        )
        source_to_stats[data_source]["first_action_agreement@4"].append(
            float(max(first_action_counts.values()) / max(n, 1))
        )

    metrics: dict[str, float] = {}
    for data_source, stat_name_to_values in source_to_stats.items():
        for stat_name, values in stat_name_to_values.items():
            if values:
                metrics[f"val-aux/{data_source}/diversity/{stat_name}"] = float(np.mean(values))
    return metrics


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]

        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "data": data,
            "config": config,
            "gamma": gamma,
            "lam": lam,
            "num_repeat": num_repeat,
            "norm_adv_by_std_in_grpo": norm_adv_by_std_in_grpo,
        }
        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.role_worker_mapping)
        self.use_rm = need_reward_model(self.role_worker_mapping)
        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )
        self._image_dump_actors = {}
        self._pending_dump_futures = []
        self._log_image_cfg = self.config.trainer.get("log_image", {})
        self._log_image_enable = self._log_image_cfg.get("enable", False)
        self._max_pending_dumps = self._log_image_cfg.get("max_pending", 2)

        # HuggingFace Hub upload
        self._hf_upload_manager = HFUploadManager(config)

        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = (
            config.actor_rollout_ref.model.get("lora_rank", 0) > 0
            or config.actor_rollout_ref.model.get("lora_adapter_path") is not None
        )

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("train_max_samples", -1),
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("val_max_samples", -1),
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, images, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(
                json.dumps(
                    entry,
                    ensure_ascii=False,
                    default=lambda value: value.item() if isinstance(value, np.generic) else str(value),
                )
            )

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

        # Save images to subfolders
        if images and self._log_image_enable:
            actor = self._image_dump_actors.get(dump_path)
            if actor is None:
                actor = ImageDumpActor.remote(base_dir=dump_path)
                self._image_dump_actors[dump_path] = actor

            compress_level = self._log_image_cfg.get("png_compress_level", 0)
            fut = actor.dump_images.remote(
                step=self.global_steps,
                images=images,
                compress_level=compress_level,
            )
            self._pending_dump_futures.append(fut)

            if self._max_pending_dumps > 0 and len(self._pending_dump_futures) > self._max_pending_dumps:
                done, rest = ray.wait(self._pending_dump_futures, num_returns=1)
                ray.get(done)
                self._pending_dump_futures = rest

    def _flush_image_dumps(self):
        if not self._pending_dump_futures:
            return
        ray.get(self._pending_dump_futures)
        self._pending_dump_futures = []

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            
            inputs = batch.batch["prompts"]
            outputs = batch.batch["responses"]
            
            # remove pad tokens for logging (keeps other special tokens like <|endoftext|>
            # visible so we can spot degenerate model outputs)
            pad_token_id = self.tokenizer.pad_token_id
            skip_pad_tokens = self.config.trainer.get("skip_pad_tokens", True)
            if skip_pad_tokens:
                inputs = self.tokenizer.batch_decode(
                    [s[-l:] if l else [] for s, l in zip(inputs.tolist(),  (inputs  != pad_token_id).sum(1).tolist())],
                    skip_special_tokens=False)
                outputs = self.tokenizer.batch_decode(
                    [s[:l]  if l else [] for s, l in zip(outputs.tolist(), (outputs != pad_token_id).sum(1).tolist())],
                    skip_special_tokens=False)
            else:
                inputs = self.tokenizer.batch_decode(inputs.tolist(), skip_special_tokens=False)
                outputs = self.tokenizer.batch_decode(outputs.tolist(), skip_special_tokens=False)

            if self.config.trainer.get("replace_image_tokens_for_logging", False):
                inputs = replace_image_tokens_for_logging(inputs, processor=self.processor, tokenizer=self.tokenizer)
                outputs = replace_image_tokens_for_logging(outputs, processor=self.processor, tokenizer=self.tokenizer)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]
            # Extract images from non_tensor_batch (extra_fields are stored there)
            sample_images=[]
            if "image_data" in batch.non_tensor_batch:
                batch_images = batch.non_tensor_batch["image_data"]
                sample_images.extend(batch_images.tolist() if hasattr(batch_images, 'tolist') else batch_images)
            else:
                sample_images.extend([None] * len(outputs))
            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_dict.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                images=sample_images,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
            )

    def _maybe_log_val_generations(self, inputs, outputs, scores, images=None):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score, image) and sort by input text
        if images is None or len(images) == 0:
            images = [None] * len(inputs)
        else:
            non_none_count = sum(1 for img in images if img is not None)
            print(f"Logging {non_none_count}/{len(images)} validation samples with images to wandb")

        samples = list(zip(inputs, outputs, scores, images, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def _assign_group_and_traj_idx(self, gen_batch: DataProto, num_traj_per_sample: int) -> None:
        """Assign group_idx and traj_idx for no-concat mode.

        Args:
            gen_batch: The generated batch after repeat operation
            num_traj_per_sample: Number of trajectories per sample (repeat_times)
        """
        # Assign group_idx from uid
        gen_batch.non_tensor_batch["group_idx"] = gen_batch.non_tensor_batch["uid"]

        # Assign traj_idx based on repeat pattern
        # Since repeat with interleave=True creates [A, A, A, B, B, B, C, C, C] pattern,
        # traj_idx should be [0, 1, 2, 0, 1, 2, 0, 1, 2]
        batch_size = len(gen_batch.non_tensor_batch["uid"])
        traj_idx = np.tile(np.arange(num_traj_per_sample), batch_size // num_traj_per_sample)
        gen_batch.non_tensor_batch["traj_idx"] = traj_idx


    def _post_process_no_concat_batch(self, batch: DataProto, gen_batch_output: DataProto) -> DataProto:
        """Re-align and union batch with gen_batch_output in no-concat mode.

        In no-concat mode, each trajectory has multiple prompt-response pairs with varying lengths.
        Each original batch item may correspond to a different number of gen_batch_output items
        depending on how many turns were generated for that trajectory.

        The key insight: we build a selection index list that maps each gen_batch_output item
        to its corresponding original batch item, then use select_idxs to replicate and reorder
        the batch to match gen_batch_output's uid sequence.

        Args:
            batch: Original batch with reward model keys (uid, data_source, etc.)
            gen_batch_output: Generated output with sequences and uid (variable items per original uid)

        Returns:
            DataProto: Aligned and unified batch ready for downstream processing

        Example:
            Original batch: [item_0 (uid=C), item_1 (uid=B), item_2 (uid=A)]
            gen_batch_output uids: [A, A, A, B, B, C, C, C, C]
            -> selection_indices: [2, 2, 2, 1, 1, 0, 0, 0, 0]
            -> select_idxs([2,2,2,1,1,0,0,0,0]): [A, A, A, B, B, C, C, C, C]
            -> Perfectly aligned with gen_batch_output!
        """
        # Step 1: Verify uid exists in both batches
        assert "uid" in batch.non_tensor_batch, "batch must contain 'uid' in non_tensor_batch for alignment"
        gen_batch_output.non_tensor_batch["uid"]=gen_batch_output.non_tensor_batch["group_idx"]
        assert "uid" in gen_batch_output.non_tensor_batch, (
            "gen_batch_output must contain 'uid' in non_tensor_batch for alignment"
        )

        # Step 2: Build uid to index mapping for original batch
        batch_uid_to_idx = {str(uid): idx for idx, uid in enumerate(batch.non_tensor_batch["uid"])}

        # Step 3: Build selection indices by mapping each gen_batch_output uid to its batch index
        # This automatically handles:
        # - Variable repetition (each uid can appear different number of times)
        # - Arbitrary ordering (gen_batch_output uids can be in any order)
        selection_indices = []

        for gen_uid in gen_batch_output.non_tensor_batch["uid"]:
            gen_uid_str = str(gen_uid)
            if gen_uid_str not in batch_uid_to_idx:
                raise ValueError(
                    f"uid '{gen_uid_str}' from gen_batch_output not found in batch. "
                    f"Available uids: {list(batch_uid_to_idx.keys())[:5]}... "
                    f"This suggests a data alignment issue in agent loop."
                )
            batch_idx = batch_uid_to_idx[gen_uid_str]
            selection_indices.append(batch_idx)

        # Step 4: Use select_idxs to replicate and reorder batch to match gen_batch_output
        # This single operation handles both repetition and reordering
        batch = batch.select_idxs(selection_indices)

        # Step 5: Verify the size matches
        assert len(batch) == len(gen_batch_output), (
            f"After alignment, batch size ({len(batch)}) should match gen_batch_output size ({len(gen_batch_output)}). "
            f"selection_indices length: {len(selection_indices)}"
        )

        # Step 6: Union the aligned batches
        batch = batch.union(gen_batch_output)

        return batch

    def _validate(self):
        data_source_lst = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)
        custom_metrics_accumulator: dict[str, list] = defaultdict(list)

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_uids = []
        sample_images = []

        pad_token_id = self.tokenizer.pad_token_id
        skip_pad_tokens = self.config.trainer.get("skip_pad_tokens", True)

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch.batch))], dtype=object
                )

            # repeat test batch
            test_batch = test_batch.repeat(
                repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True
            )

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            
            sample_uids.extend(test_batch.non_tensor_batch["uid"])

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            test_gen_batch = self._get_gen_batch(test_batch)

            if not self.concat_multi_turn:
                # we need to create group_idx, traj_idx for each traj in no-concat mode
                num_traj_per_sample = self.config.actor_rollout_ref.rollout.val_kwargs.n
                self._assign_group_and_traj_idx(test_gen_batch, num_traj_per_sample)

            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate": True,
                "global_steps": self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )

            # In no-concat mode, save original uids before padding for filtering later
            if not self.concat_multi_turn:
                original_uids = set(test_gen_batch.non_tensor_batch["uid"])

            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            if self.concat_multi_turn:
                test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            else:
                # In no-concat mode, filter by uid since each input generates variable number of outputs
                # We need to keep only outputs whose uid is in the original (pre-padding) uid set
                valid_indices = [
                    i for i, uid in enumerate(test_output_gen_batch_padded.non_tensor_batch["group_idx"]) # uid in test_gen become group index in test_output_gen
                    if uid in original_uids
                ]
                test_output_gen_batch = test_output_gen_batch_padded.select_idxs(valid_indices)
                # Concatenate multi-turn trajectories into single entries
                test_output_gen_batch = concat_val_multi_turn(test_output_gen_batch, test_gen_batch,self.tokenizer)
                # after this, we can assume no-concat mode and concat_multi_turn can be handled equally


            print("validation generation end")
            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True
            # Store generated outputs
            
            inputs = test_batch.batch["prompts"]
            outputs = test_batch.batch["responses"]
            if skip_pad_tokens:
                inputs = self.tokenizer.batch_decode(
                    [s[-l:] if l else [] for s, l in zip(inputs.tolist(),  (inputs  != pad_token_id).sum(1).tolist())],
                    skip_special_tokens=False)
                outputs = self.tokenizer.batch_decode(
                    [s[:l]  if l else [] for s, l in zip(outputs.tolist(), (outputs != pad_token_id).sum(1).tolist())],
                    skip_special_tokens=False)
            else:
                inputs = self.tokenizer.batch_decode(inputs.tolist(), skip_special_tokens=False)
                outputs = self.tokenizer.batch_decode(outputs.tolist(), skip_special_tokens=False)
           
            sample_inputs.extend(inputs)
            sample_outputs.extend(outputs)

            # Extract images from non_tensor_batch (extra_fields are stored there)
            if "image_data" in test_batch.non_tensor_batch:
                batch_images = test_batch.non_tensor_batch["image_data"]
                sample_images.extend(batch_images.tolist() if hasattr(batch_images, 'tolist') else batch_images)
            else:
                sample_images.extend([None] * len(outputs))

            
            
            
            
            # evaluate using reward_function
            if self.val_reward_fn is None:
                raise ValueError("val_reward_fn must be provided for validation.")
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            if "reward_extra_info" in result:
                for key, lst in result["reward_extra_info"].items():
                    reward_extra_infos_dict[key].extend(lst)

            # Add token_level_scores to batch for custom metrics computation
            test_batch.batch["token_level_scores"] = reward_tensor

            # Compute custom metrics for validation
            custom_val_metrics = compute_custom_metrics(test_batch, prefix="custom_metrics")
            for metric_name, metric_value in custom_val_metrics.items():
                custom_metrics_accumulator[metric_name].append(metric_value)

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            data_source_lst.append(test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0]))
        
        if self.config.trainer.get("replace_image_tokens_for_logging", False):
            sample_inputs = replace_image_tokens_for_logging(sample_inputs, processor=self.processor, tokenizer=self.tokenizer)
            sample_outputs = replace_image_tokens_for_logging(sample_outputs, processor=self.processor, tokenizer=self.tokenizer)
            
        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores, images=sample_images)
        # dump generations
        reward_extra_infos_dict["sample_uid"] = [str(uid) for uid in sample_uids]
        reward_extra_infos_dict["action_sequence"] = [
            "|".join(_extract_action_sequence(output)) for output in sample_outputs
        ]
        reward_extra_infos_dict["trajectory_hash"] = [_stable_text_hash(output) for output in sample_outputs]

        # Write data_source into the JSONL dump so OOD splits can be
        # distinguished in downstream analysis (analyze_experiments.py).
        reward_extra_infos_dict["data_source"] = [
            src for batch_srcs in data_source_lst for src in batch_srcs
        ]

        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                images=sample_images,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        data_sources = np.concatenate(data_source_lst, axis=0)

        data_src2var2metric2val = process_validation_metrics(data_sources, sample_uids, reward_extra_infos_dict)
        metric_dict = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max([int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys()])
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else: 
                        metric_sec = "val-aux"
                    pfx = f"{metric_sec}/{data_source}/{var_name}/{metric_name}"
                    metric_dict[pfx] = metric_val

        metric_dict.update(compute_validation_diversity_metrics(data_sources, sample_uids, sample_outputs))

        if len(sample_turns) > 0:
            sample_turns = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = sample_turns.min()
            metric_dict["val-aux/num_turns/max"] = sample_turns.max()
            metric_dict["val-aux/num_turns/mean"] = sample_turns.mean()

        # Add aggregated custom metrics to metric_dict
        for metric_name, values in custom_metrics_accumulator.items():
            if len(values) > 0:
                # Use mean aggregation for custom metrics
                metric_dict[f"custom_metrics/val/{metric_name.split('/')[-1]}"] = np.mean(values)

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role=str(Role.ActorRollout),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.ActorRollout)] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cfg = omega_conf_to_dataclass(self.config.critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool][str(Role.RewardModel)] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
            self.ref_policy_wg.init_model()

        self.rm_wg = None
        # initalization of rm_wg will be deprecated in the future
        if self.use_rm:
            self.rm_wg = all_wg[str(Role.RewardModel)]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg[str(Role.ActorRollout)]
        self.actor_rollout_wg.init_model()

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        self.concat_multi_turn = True # whether to concat history in async rollout --> if not, one traj has multiple prompt-response pairs
        if self.config.actor_rollout_ref.rollout.mode == "async":
            self.async_rollout_mode = True
            if self.config.trainer.get("concat_multi_turn", True):
                from verl.experimental.agent_loop import AgentLoopManager
            else:
                from .agent_loop.agent_loop_no_concat import AgentLoopManager
                self.concat_multi_turn = False
            self.async_rollout_manager = AgentLoopManager(
                    config=self.config, worker_group=self.actor_rollout_wg, rm_wg=self.rm_wg
                )


    def _rotate_complete_checkpoints(self, current_folder: str):
        """Delete old COMPLETE checkpoint folders atomically; never leave hollow dirs.

        Keeps: latest, optional best, fixed milestones, and the newest N COMPLETE folders.
        Incomplete folders (no COMPLETE marker) are never treated as loadable and are removed
        only when they are clearly hollow (missing actor/ or critic/ shards).
        """
        import shutil

        ckpt_root = os.path.dirname(current_folder.rstrip("/"))
        keep_n = int(self.config.trainer.get("max_actor_ckpt_to_keep", 5) or 5)
        milestones = set(int(x) for x in self.config.trainer.get("ckpt_milestones", [10, 20, 30, 50, 100]) or [])
        protect_steps = set(milestones)
        protect_steps.add(int(self.global_steps))
        latest_path = os.path.join(ckpt_root, "latest_checkpointed_iteration.txt")
        if os.path.exists(latest_path):
            try:
                protect_steps.add(int(open(latest_path).read().strip()))
            except Exception:
                pass
        best_path = os.path.join(ckpt_root, "best_checkpointed_iteration.txt")
        if os.path.exists(best_path):
            try:
                protect_steps.add(int(open(best_path).read().strip()))
            except Exception:
                pass

        entries = []
        for name in os.listdir(ckpt_root):
            if not name.startswith("global_step_"):
                continue
            folder = os.path.join(ckpt_root, name)
            if not os.path.isdir(folder):
                continue
            try:
                step = int(name.split("global_step_")[-1])
            except Exception:
                continue
            complete = os.path.exists(os.path.join(folder, "COMPLETE"))
            has_actor = os.path.isdir(os.path.join(folder, "actor"))
            has_critic = os.path.isdir(os.path.join(folder, "critic")) or os.path.isdir(
                os.path.join(folder, str(Role.Critic))
            )
            # Clean clearly hollow incomplete folders (e.g. data.pt only).
            if (not complete) and (not has_actor) and (not has_critic) and step not in protect_steps:
                print(f"[ckpt-rotate] removing hollow incomplete folder: {folder}")
                shutil.rmtree(folder, ignore_errors=True)
                continue
            if complete:
                entries.append((step, folder))

        entries.sort(key=lambda x: x[0])
        # Keep newest keep_n complete folders plus protected steps.
        keep_steps = {s for s, _ in entries[-keep_n:]} | protect_steps
        for step, folder in entries:
            if step in keep_steps:
                continue
            # Never delete the folder currently being written.
            if os.path.abspath(folder) == os.path.abspath(current_folder):
                continue
            print(f"[ckpt-rotate] removing old COMPLETE checkpoint folder: {folder}")
            shutil.rmtree(folder, ignore_errors=True)

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        # Role-wise max_ckpt_to_keep deleted actor/critic shards independently and left
        # hollow global_step_* folders. Retention is handled by _rotate_complete_checkpoints.
        max_actor_ckpt_to_keep = None
        max_critic_ckpt_to_keep = None
        if remove_previous_ckpt_in_save:
            print(
                "Note: remove_previous_ckpt_in_save ignored for role managers; "
                "using trainer-level COMPLETE folder rotation instead"
            )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # Persist the exact resolved config and checkpoint metadata before the
        # COMPLETE marker. A folder cannot become loadable while any of its
        # actor/critic/config/metadata components are still being written.
        resolved_config_path = os.path.join(local_global_step_folder, "resolved_config.yaml")
        OmegaConf.save(config=self.config, f=resolved_config_path, resolve=True)
        checkpoint_metadata_path = os.path.join(local_global_step_folder, "checkpoint_metadata.json")
        checkpoint_metadata = {
            "global_step": int(self.global_steps),
            "experiment_name": str(self.config.trainer.experiment_name),
            "actor_path": actor_local_path,
            "critic_path": critic_local_path if self.use_critic else None,
            "has_actor": os.path.isdir(actor_local_path),
            "has_critic": (not self.use_critic) or os.path.isdir(critic_local_path),
            "has_dataloader_state": os.path.isfile(dataloader_local_path),
            "has_resolved_config": os.path.isfile(resolved_config_path),
            "u1_fm_backprop": os.environ.get("U1_FM_BACKPROP", "unset"),
            "created_unix_time": time.time(),
        }
        if not checkpoint_metadata["has_actor"] or not checkpoint_metadata["has_critic"]:
            raise RuntimeError(f"refusing COMPLETE marker for incomplete checkpoint: {checkpoint_metadata}")
        metadata_tmp = checkpoint_metadata_path + ".tmp"
        with open(metadata_tmp, "w", encoding="utf-8") as f:
            json.dump(checkpoint_metadata, f, indent=2, sort_keys=True)
            f.write("\n")
        os.replace(metadata_tmp, checkpoint_metadata_path)

        # Atomic-ish completeness marker for this global_step folder.
        complete_marker = os.path.join(local_global_step_folder, "COMPLETE")
        complete_tmp = complete_marker + ".tmp"
        with open(complete_tmp, "w", encoding="utf-8") as f:
            f.write(str(self.global_steps))
        os.replace(complete_tmp, complete_marker)

        # Only publish latest after COMPLETE exists, and replace atomically so a
        # crash cannot point resume/evaluation at a partially written folder.
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        latest_tmp = local_latest_checkpointed_iteration + ".tmp"
        with open(latest_tmp, "w", encoding="utf-8") as f:
            f.write(str(self.global_steps))
        os.replace(latest_tmp, local_latest_checkpointed_iteration)

        # Whole-folder rotation (atomic completeness). Role-wise shard deletion is disabled
        # below to avoid hollow global_step_* dirs that retain only data.pt.
        self._rotate_complete_checkpoints(local_global_step_folder)

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            # NOTE: while there is no checkpoint to load, we still need to offload the model and optimizer to CPU
            self.actor_rollout_wg.load_checkpoint(None)
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                self.actor_rollout_wg.load_checkpoint(None)
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)
            if self.use_rm:
                self.rm_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm:
                self.rm_wg.stop_profile()

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1)  # (train_batch_size,)
        global_seqlen_lst = calculate_workload(global_seqlen_lst)
        world_size = self.actor_rollout_wg.world_size
        if keep_minibatch:
            # Decouple the DP balancing and mini-batching.
            minibatch_size = self.config.actor_rollout_ref.actor.get("ppo_mini_batch_size")
            minibatch_num = len(global_seqlen_lst) // minibatch_size
            global_partition_lst = [[] for _ in range(world_size)]
            for i in range(minibatch_num):
                rearrange_minibatch_lst = get_seqlen_balanced_partitions(
                    global_seqlen_lst[i * minibatch_size : (i + 1) * minibatch_size],
                    k_partitions=world_size,
                    equal_size=True,
                )
                for j, part in enumerate(rearrange_minibatch_lst):
                    global_partition_lst[j].extend([x + minibatch_size * i for x in part])
        else:
            global_partition_lst = get_seqlen_balanced_partitions(
                global_seqlen_lst, k_partitions=world_size, equal_size=True
            )
        # Place smaller micro-batches at both ends to reduce the bubbles in pipeline parallel.
        for idx, partition in enumerate(global_partition_lst):
            partition.sort(key=lambda x: (global_seqlen_lst[x], x))
            ordered_partition = partition[::2] + partition[1::2][::-1]
            global_partition_lst[idx] = ordered_partition
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                self._flush_image_dumps()
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                with marked_timer("start_profile", timing_raw):
                    self._start_profiling(
                        not prev_step_profile and curr_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)

                # pass global_steps to trace
                gen_batch.meta_info["global_steps"] = self.global_steps
                gen_batch_output = gen_batch.repeat(
                    repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True
                )
                if not self.concat_multi_turn:
                    # we need to create group_idx, traj_idx for each traj in no-concat mode
                    num_traj_per_sample = self.config.actor_rollout_ref.rollout.n
                    self._assign_group_and_traj_idx(gen_batch_output, num_traj_per_sample)

                is_last_step = self.global_steps >= self.total_training_steps
                with marked_timer("step", timing_raw):
                    # generate a batch
                    with marked_timer("gen", timing_raw, color="red"):
                        if not self.async_rollout_mode:
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                        else:
                            gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)

                        timing_raw.update(gen_batch_output.meta_info["timing"])
                        gen_batch_output.meta_info.pop("timing", None)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        if not self.concat_multi_turn:
                            raise NotImplementedError("REMAX advantage estimation is not supported in no-concat mode yet.")
                        if self.reward_fn is None:
                            raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                        with marked_timer("gen_max", timing_raw, color="purple"):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            if not self.async_rollout_mode:
                                gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            else:
                                gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                            batch = batch.union(gen_baseline_output)
                            # compute reward model score on batch
                            rm_scores = None
                            if self.use_rm and "rm_scores" not in batch.batch.keys():
                                rm_scores = self.rm_wg.compute_rm_score(batch)
                                batch = batch.union(rm_scores)
                            reward_baseline_tensor, _ = compute_reward(batch, self.reward_fn)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            keys_to_pop = set(gen_baseline_output.batch.keys())
                            if rm_scores is not None:
                                keys_to_pop.update(rm_scores.batch.keys())
                            batch.pop(batch_keys=list(keys_to_pop))

                            batch.batch["reward_baselines"] = reward_baseline_tensor

                            del rm_scores, gen_baseline_batch, gen_baseline_output
                    # repeat to align with repeated responses in rollout
                    if self.concat_multi_turn:
                        batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                        batch = batch.union(gen_batch_output)
                    else:
                        # In no-concat mode, each trajectory has multiple prompt-response pairs.
                        # We need to re-generate batch to align with gen_batch_output.
                        batch = self._post_process_no_concat_batch(batch, gen_batch_output)

                    if "response_mask" not in batch.batch.keys():
                        batch.batch["response_mask"] = compute_response_mask(batch)
                    
                    # Balance the number of valid tokens across DP ranks.
                    # NOTE: This usually changes the order of data in the `batch`,
                    # which won't affect the advantage calculation (since it's based on uid),
                    # but might affect the loss calculation (due to the change of mini-batching).
                    if self.config.trainer.balance_batch:
                        if not self.concat_multi_turn: # pad to divisor of dp_size
                            divisor_size = self.actor_rollout_wg.world_size
                            batch_size = len(batch.batch["attention_mask"])
                            batch, pad_size = pad_dataproto_to_divisor(batch, divisor_size)
                            print(f"Pad {pad_size} samples to make batch size {batch_size} divisible by {divisor_size} dp_workers")
                        self._balance_batch(batch, metrics=metrics)

                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                    rollout_cfg = self.config.actor_rollout_ref.rollout
                    batch.meta_info["sampling_temperature"] = float(rollout_cfg.temperature)
                    batch.meta_info["temperature"] = float(rollout_cfg.get("logprob_temperature", 1.0))

                    with marked_timer("reward", timing_raw, color="yellow"):
                        # compute reward model score
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        if self.config.reward_model.launch_reward_fn_async:
                            future_reward = compute_reward_async.remote(
                                data=batch, config=self.config, tokenizer=self.tokenizer
                            )
                        else:
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)

                    # Operating Mode Selection:
                    # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
                    # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
                    #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
                    rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                    bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                    d0_16_response_mask_before_correction = None
                    if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                        from verl.trainer.ppo.rollout_corr_helper import apply_rollout_correction

                        apply_rollout_correction(
                            batch=batch,
                            rollout_corr_config=rollout_corr_config,
                            policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                        )
                    else:  # Recompute old_log_probs
                        with marked_timer("old_log_prob", timing_raw, color="blue"):
                            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                            entropys = old_log_prob.batch["entropys"]
                            response_masks = batch.batch["response_mask"]
                            loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                            entropy_agg = agg_loss(
                                loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode
                            )
                            old_log_prob_metrics = {"actor/entropy": entropy_agg.detach().item()}
                            metrics.update(old_log_prob_metrics)
                            old_log_prob.batch.pop("entropys")
                            batch = batch.union(old_log_prob)
                            if "rollout_log_probs" in batch.batch.keys():
                                # TODO: we may want to add diff of probs too.
                                from verl.utils.debug.metrics import calculate_debug_metrics

                                metrics.update(calculate_debug_metrics(batch))
                                d0_3_dir = os.environ.get("VAGEN_D0_3_SMOKE_DIR")
                                if d0_3_dir:
                                    metrics.update(
                                        _d0_3_response_mask_parity(
                                            batch,
                                            d0_3_dir,
                                            tag="pre_update",
                                            global_step=self.global_steps,
                                        )
                                    )
                                d0_4_dir = os.environ.get("VAGEN_D0_4_DIR")
                                if d0_4_dir:
                                    metrics.update(
                                        _d0_4_write_localization_artifacts(
                                            batch,
                                            d0_4_dir,
                                            tokenizer=self.tokenizer,
                                            global_step=self.global_steps,
                                        )
                                    )

                    assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'
                    if os.environ.get("VAGEN_D0_6_STOP_AFTER_LOGPROB") == "1":
                        print(
                            "[D0.6] Stopping after rollout + FSDP old_log_prob artifact capture; "
                            "no ref/critic/actor optimizer update will be run.",
                            flush=True,
                        )
                        self._flush_image_dumps()
                        return

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                            if not self.ref_in_actor:
                                ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            else:
                                ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with marked_timer("values", timing_raw, color="cyan"):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with marked_timer("adv", timing_raw, color="brown"):
                        # we combine with rule-based rm
                        reward_extra_infos_dict: dict[str, list]
                        if self.config.reward_model.launch_reward_fn_async:
                            reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                        batch.batch["token_level_scores"] = reward_tensor

                        if reward_extra_infos_dict:
                            batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                            # Train-time rollout metrics. In no-concat mode the batch is flattened
                            # to one row per environment turn, so episode-level metrics must be
                            # grouped before averaging. Padding duplicates are removed by
                            # compute_unique_sample_mask().
                            _unique_sample_mask = compute_unique_sample_mask(batch)

                            def _masked_reward_extra_array(name: str) -> np.ndarray | None:
                                values = reward_extra_infos_dict.get(name)
                                if values is None or len(values) == 0:
                                    return None
                                arr = np.asarray(values, dtype=np.float32)
                                mask = (
                                    _unique_sample_mask[: arr.shape[0]]
                                    if _unique_sample_mask.shape[0] >= arr.shape[0]
                                    else np.ones(arr.shape[0], dtype=bool)
                                )
                                arr = arr[mask]
                                return arr if arr.size > 0 else None

                            def _log_transition_metric(name: str, metric_prefix: str) -> None:
                                arr = _masked_reward_extra_array(name)
                                if arr is None:
                                    return
                                metrics[f"{metric_prefix}/mean"] = float(arr.mean())
                                metrics[f"{metric_prefix}/std"] = float(arr.std())
                                metrics[f"{metric_prefix}/sum"] = float(arr.sum())
                                metrics[f"{metric_prefix}/count"] = int(arr.size)

                            def _log_transition_reward() -> None:
                                if "token_level_scores" not in batch.batch:
                                    return
                                rewards = batch.batch["token_level_scores"].sum(-1).detach().float().cpu().numpy()
                                mask = (
                                    _unique_sample_mask[: rewards.shape[0]]
                                    if _unique_sample_mask.shape[0] >= rewards.shape[0]
                                    else np.ones(rewards.shape[0], dtype=bool)
                                )
                                rewards = rewards[mask]
                                if rewards.size == 0:
                                    return
                                metrics["transition/reward_mean"] = float(rewards.mean())
                                metrics["transition/reward_std"] = float(rewards.std())
                                metrics["transition/reward_min"] = float(rewards.min())
                                metrics["transition/reward_max"] = float(rewards.max())
                                metrics["transition/reward_count"] = int(rewards.size)

                            def _log_episode_success() -> None:
                                values = reward_extra_infos_dict.get("traj_success")
                                if values is None or len(values) == 0:
                                    return
                                required = ("group_idx", "traj_idx")
                                if not all(k in batch.non_tensor_batch for k in required):
                                    arr = _masked_reward_extra_array("traj_success")
                                    if arr is None:
                                        return
                                    success_count = float(arr.sum())
                                    num_episodes = int(arr.size)
                                else:
                                    arr = np.asarray(values, dtype=np.float32)
                                    n = arr.shape[0]
                                    mask = (
                                        _unique_sample_mask[:n]
                                        if _unique_sample_mask.shape[0] >= n
                                        else np.ones(n, dtype=bool)
                                    )
                                    group_idx = np.asarray(batch.non_tensor_batch["group_idx"][:n]).astype(str)
                                    traj_idx = np.asarray(batch.non_tensor_batch["traj_idx"][:n]).astype(str)
                                    episode_success: dict[tuple[str, str], float] = {}
                                    for gid, tid, success, keep in zip(group_idx, traj_idx, arr, mask):
                                        if not keep:
                                            continue
                                        key = (gid, tid)
                                        episode_success[key] = max(episode_success.get(key, 0.0), float(success))
                                    if not episode_success:
                                        return
                                    success_values = np.asarray(list(episode_success.values()), dtype=np.float32)
                                    success_count = float(success_values.sum())
                                    num_episodes = int(success_values.size)
                                success_rate = float(success_count / max(num_episodes, 1))
                                metrics["episode/success_rate"] = success_rate
                                metrics["episode/success_count"] = success_count
                                metrics["episode/num_episodes"] = num_episodes
                                # Backward-compatible alias, now episode-level rather than turn-level.
                                metrics["train/traj_success/mean"] = success_rate
                                metrics["train/traj_success/sum"] = success_count
                                metrics["train/traj_success/count"] = num_episodes

                            _log_episode_success()
                            _log_transition_metric("invalid_action", "transition/invalid_action_rate")
                            _log_transition_metric("empty_action", "transition/empty_action_rate")
                            _log_transition_metric("contradictory_action", "transition/contradictory_action_rate")
                            _log_transition_metric("missing_action_tag", "transition/missing_action_tag_rate")
                            _log_transition_metric("empty_action_body", "transition/empty_action_body_rate")
                            _log_transition_metric("unknown_action_name", "transition/unknown_action_name_rate")
                            _log_transition_metric("truncated_before_action", "transition/truncated_before_action_rate")
                            _log_transition_metric("multiple_action_tag", "transition/multiple_action_tag_rate")
                            _log_transition_metric("env_exception", "transition/env_exception_rate")
                            _log_transition_metric("strict_action_tag_rate", "transition/strict_action_tag_rate")
                            _log_transition_metric("tool_call_rate", "transition/tool_call_rate")
                            _log_transition_metric("fallback_parse_rate", "transition/fallback_parse_rate")
                            _log_transition_metric("strict_parse_success_rate", "transition/strict_parse_success_rate")
                            _log_transition_metric("format_penalty_rate", "transition/format_penalty_rate")
                            _log_transition_metric("collision_termination", "transition/collision_termination_rate")
                            _log_transition_metric("low_info_termination", "transition/low_info_termination_rate")
                            _log_transition_metric("renderer_failure", "transition/renderer_failure_rate")
                            _log_transition_metric("episode_length", "transition/episode_length")
                            _log_transition_metric("final_score", "transition/final_score")
                            _log_transition_metric("score_improvement", "transition/score_improvement")
                            for _action_name in (
                                "move_forward", "move_backward", "move_left", "move_right", "turn_left", "turn_right"
                            ):
                                _log_transition_metric(
                                    f"action_{_action_name}", f"transition/action_distribution/{_action_name}"
                                )
                            _log_transition_reward()

                        # compute rewards. apply_kl_penalty if available
                        if self.config.algorithm.use_kl_in_reward:
                            batch, kl_metrics = apply_kl_penalty(
                                batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                            )
                            metrics.update(kl_metrics)
                        else:
                            batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                        # Compute rollout correction: IS weights, rejection sampling, and metrics
                        # Only runs in decoupled mode (computes once per batch using stable π_old)
                        # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
                        if (
                            rollout_corr_config is not None
                            and "rollout_log_probs" in batch.batch
                            and not bypass_recomputing_logprobs  # Only in decoupled mode
                        ):
                            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                            # Compute IS weights, apply rejection sampling, compute metrics
                            if os.environ.get("VAGEN_D0_16_DIR"):
                                d0_16_response_mask_before_correction = batch.batch["response_mask"].detach().clone()
                            batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                            # IS and off-policy metrics already have rollout_corr/ prefix
                            metrics.update(is_metrics)

                        # compute advantages, executed on the driver process
                        norm_adv_by_std_in_grpo = self.config.algorithm.get(
                            "norm_adv_by_std_in_grpo", True
                        )  # GRPO adv normalization factor

                        batch = compute_advantage(
                            batch,
                            adv_estimator=self.config.algorithm.adv_estimator,
                            gamma=self.config.algorithm.gamma,
                            lam=self.config.algorithm.lam,
                            num_repeat=self.config.actor_rollout_ref.rollout.n,
                            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                            config=self.config.algorithm,
                        )

                    if self.config.algorithm.adv_estimator in ["no_concat_gae_last", "no_concat_gae_first", "no_concat_gae"]:
                        batch.batch["value_mask"] = compute_value_mask(batch)

                    # compute custom metrics
                    with marked_timer("custom_metrics", timing_raw, color="magenta"):
                        custom_train_metrics = compute_custom_metrics(batch, prefix="custom_metrics/train")
                        metrics.update(custom_train_metrics)

                    
                    
                    # filter the training batch for effective update (Refer to STARPO-S and DAPO)
                    if self.config.filter.get("enable", False):
                        batch,metrics = FILTER_REGISTRY.get(self.config.filter.name)(batch, metrics,**self.config.filter.filter_kwargs)
                        if self.config.trainer.balance_batch:
                            # re-balance after filtering
                            divisor_size = self.actor_rollout_wg.world_size
                            batch_size = len(batch.batch["attention_mask"])
                            batch, pad_size = pad_dataproto_to_divisor(batch, divisor_size)
                            print(f"After filtering: Pad {pad_size} samples to make batch size {batch_size} divisible by {divisor_size} dp_workers")
                            self._balance_batch(batch, metrics=metrics, logging_prefix="filtered_global_seqlen")

                    # Opt-in exact post-GAE/post-filter boundary for offline PPO
                    # diagnosis.  With the environment variable unset this does
                    # nothing at all; when armed it runs before either optimizer.
                    # A post-filter-only capture cannot faithfully replay GAE
                    # whitening if the filter removed trajectory rows.  The
                    # diagnostic contract therefore fails closed until a future
                    # hook persists the complete pre-filter batch plus index map.
                    if (
                        os.environ.get("VAGEN_ACTIVE_SPATIAL_PPO_SNAPSHOT_DIR")
                        and self.config.filter.get("enable", False)
                    ):
                        raise RuntimeError(
                            "exact PPO snapshot refuses filter.enable=true: "
                            "capture requires the complete pre-filter batch and mapping"
                        )
                    # A step-0 critic has a random value head.  When explicitly
                    # requested, export its *current, pre-update* distributed
                    # state and bind the snapshot to a content digest.  This is
                    # a new immutable diagnostic artifact, never an update to
                    # an existing checkpoint, and is intentionally unavailable
                    # unless the snapshot hook itself is armed.
                    critic_initial_export = os.environ.get(
                        "VAGEN_ACTIVE_SPATIAL_PPO_SNAPSHOT_CRITIC_INITIAL_EXPORT_DIR"
                    )
                    if critic_initial_export:
                        if os.path.exists(critic_initial_export):
                            raise RuntimeError(
                                "critic initial export target already exists; "
                                "refusing to overwrite a diagnostic identity"
                            )
                        if not self.use_critic:
                            raise RuntimeError("exact PPO snapshot requires an initialized critic")
                        self.critic_wg.save_checkpoint(
                            critic_initial_export, None, self.global_steps, max_ckpt_to_keep=None
                        )
                        os.environ["VAGEN_ACTIVE_SPATIAL_PPO_SNAPSHOT_CRITIC_CHECKPOINT_PATH"] = (
                            critic_initial_export
                        )
                        os.environ["VAGEN_ACTIVE_SPATIAL_PPO_SNAPSHOT_CRITIC_CHECKPOINT_SHA256"] = (
                            directory_manifest_sha256(critic_initial_export)
                        )
                    snapshot_path = maybe_write_pre_update_snapshot(
                        batch, self.config, global_step=int(self.global_steps)
                    )
                    if snapshot_path is not None:
                        print(f"[active_spatial_snapshot] wrote exact pre-update batch to {snapshot_path}")
                        if os.environ.get("VAGEN_ACTIVE_SPATIAL_PPO_SNAPSHOT_STOP_AFTER_WRITE") == "1":
                            print("[active_spatial_snapshot] stopping before critic/actor optimizer steps.")
                            self._flush_image_dumps()
                            return

                    d0_15_dir = os.environ.get("VAGEN_D0_15_DIR")
                    if d0_15_dir and "rollout_log_probs" in batch.batch:
                        with marked_timer("d0_15_pre_update_current_log_prob", timing_raw, color="blue"):
                            d0_15_current_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        d0_15_current_log_prob.batch.pop("entropys", None)
                        d0_15_metrics = _d0_15_write_pre_update_ratio_audit(
                            batch=batch,
                            current_log_prob=d0_15_current_log_prob,
                            path=d0_15_dir,
                            tokenizer=self.tokenizer,
                            global_step=int(self.global_steps),
                            actor_config=self.config.actor_rollout_ref.actor,
                            rollout_corr_config=self.config.algorithm.get("rollout_correction", None),
                        )
                        metrics.update(d0_15_metrics)
                        if os.environ.get("VAGEN_D0_15_STOP_AFTER_AUDIT") == "1":
                            print(
                                f"[D0.15] Wrote pre-update PPO ratio audit to {d0_15_dir}; "
                                "stopping before critic/actor optimizer steps."
                            )
                            self._flush_image_dumps()
                            return

                    d0_16_dir = os.environ.get("VAGEN_D0_16_DIR")
                    if d0_16_dir and "rollout_log_probs" in batch.batch and "old_log_probs" in batch.batch:
                        with marked_timer("d0_16_pre_update_current_log_prob", timing_raw, color="blue"):
                            d0_16_current_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        d0_16_current_log_prob.batch.pop("entropys", None)
                        d0_16_metrics = _d0_16_write_decoupled_correction_audit(
                            batch=batch,
                            current_log_prob=d0_16_current_log_prob,
                            path=d0_16_dir,
                            tokenizer=self.tokenizer,
                            global_step=int(self.global_steps),
                            response_mask_before_correction=d0_16_response_mask_before_correction,
                            rollout_corr_config=self.config.algorithm.get("rollout_correction", None),
                        )
                        metrics.update(d0_16_metrics)
                        if os.environ.get("VAGEN_D0_16_STOP_AFTER_AUDIT") == "1":
                            print(
                                f"[D0.16] Wrote decoupled rollout-correction audit to {d0_16_dir}; "
                                "stopping before critic/actor optimizer steps."
                            )
                            self._flush_image_dumps()
                            return
                    
                    # update critic
                    if self.use_critic:
                        with marked_timer("update_critic", timing_raw, color="pink"):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                        metrics.update(critic_output_metrics)
                        if "critic/vf_loss" in metrics:
                            metrics["critic/value_loss"] = metrics["critic/vf_loss"]
                        if "critic/vpred_mean" in metrics:
                            metrics["critic/pred/mean"] = metrics["critic/vpred_mean"]

                    # Critic warmup should only delay actor updates when a critic is actually used.
                    actor_warmup_steps = int(self.config.trainer.critic_warmup) if self.use_critic else 0
                    actor_update_ready = actor_warmup_steps <= self.global_steps
                    metrics["trainer/actor_update_performed"] = float(actor_update_ready)
                    metrics["trainer/actor_update_skipped_by_warmup"] = float(not actor_update_ready)
                    metrics["trainer/actor_warmup_steps_effective"] = actor_warmup_steps
                    if actor_update_ready:
                        # update actor
                        with marked_timer("update_actor", timing_raw, color="red"):
                            batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                            _d0_13_trainer_marker("before_update_actor_call", global_step=int(self.global_steps))
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                            _d0_13_trainer_marker("after_update_actor_call", global_step=int(self.global_steps))
                        _d0_13_trainer_marker("before_actor_metrics_reduce", global_step=int(self.global_steps))
                        actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                        _d0_13_trainer_marker(
                            "after_actor_metrics_reduce",
                            global_step=int(self.global_steps),
                            metric_keys=len(actor_output_metrics),
                        )
                        metrics.update(actor_output_metrics)
                        d0_3_dir = os.environ.get("VAGEN_D0_3_SMOKE_DIR")
                        if d0_3_dir:
                            with marked_timer("d0_3_post_update_resync", timing_raw, color="blue"):
                                _d0_13_trainer_marker("before_post_update_resync", global_step=int(self.global_steps))
                                self.actor_rollout_wg.d0_3_sync_actor_to_rollout()
                                _d0_13_trainer_marker("after_post_update_resync", global_step=int(self.global_steps))
                            os.makedirs(d0_3_dir, exist_ok=True)
                            path = os.path.join(d0_3_dir, f"post_update_resync_step{self.global_steps}.json")
                            payload = {
                                "tag": "post_update_resync",
                                "global_step": int(self.global_steps),
                                "used_production_rollout_mode": True,
                            }
                            with open(path, "w", encoding="utf-8") as f:
                                json.dump(payload, f, indent=2, sort_keys=True)
                            metrics["d0_3/post_update_resync_called"] = 1.0

                    # Log rollout generations if enabled
                    rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                    if rollout_data_dir:
                        self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

                # validate
                if (
                    self.val_reward_fn is not None
                    and self.config.trainer.test_freq > 0
                    and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
                ):
                    with marked_timer("testing", timing_raw, color="green"):
                        val_metrics: dict = self._validate()
                        if is_last_step:
                            last_val_metrics = val_metrics
                    metrics.update(val_metrics)

                # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
                esi_close_to_expiration = should_save_ckpt_esi(
                    max_steps_duration=self.max_steps_duration,
                    redundant_time=self.config.trainer.esi_redundant_time,
                )
                # Check if the conditions for saving a checkpoint are met.
                # The conditions include a mandatory condition (1) and
                # one of the following optional conditions (2/3/4):
                # 1. The save frequency is set to a positive value.
                # 2. It's the last training step.
                # 3. The current step number is a multiple of the save frequency.
                # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
                should_save_ckpt = self.config.trainer.save_freq > 0 and (
                    is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration
                )
                should_upload_hf = self._hf_upload_manager.should_upload(self.global_steps)

                if should_save_ckpt or should_upload_hf:
                    # Flush pending HF uploads before saving to avoid conflicts
                    # with checkpoint deletion (max_actor_ckpt_to_keep)
                    self._hf_upload_manager.flush()
                    if esi_close_to_expiration:
                        print("Force saving checkpoint: ESI instance expiration approaching.")
                    with marked_timer("save_checkpoint", timing_raw, color="green"):
                        self._save_checkpoint()

                if should_upload_hf:
                    self._hf_upload_manager.maybe_upload(self.global_steps)

                with marked_timer("stop_profile", timing_raw):
                    next_step_profile = (
                        self.global_steps + 1 in self.config.global_profiler.steps
                        if self.config.global_profiler.steps is not None
                        else False
                    )
                    self._stop_profiling(
                        curr_step_profile and not next_step_profile
                        if self.config.global_profiler.profile_continuous_steps
                        else curr_step_profile
                    )
                    prev_step_profile = curr_step_profile
                    curr_step_profile = next_step_profile

                steps_duration = timing_raw["step"]
                self.max_steps_duration = max(self.max_steps_duration, steps_duration)

                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch": epoch,
                    }
                )
                # collect metrics
                data_metrics = compute_data_metrics(batch=batch, use_critic=self.use_critic)
                if self.config.algorithm.adv_estimator in [
                    "no_concat_gae_last",
                    "no_concat_gae_first",
                    "no_concat_gae",
                    "masked_gae",
                ]:
                    # These estimators use -100 sentinel returns for positions without
                    # value supervision. Raw return stats are dominated by sentinels;
                    # compute_valid_return_metrics() logs the interpretable values.
                    for key in (
                        "critic/returns/mean",
                        "critic/returns/max",
                        "critic/returns/min",
                        "critic/values/mean",
                        "critic/values/max",
                        "critic/values/min",
                        "critic/vf_explained_var",
                    ):
                        data_metrics.pop(key, None)
                metrics.update(data_metrics)
                metrics.update(compute_valid_return_metrics(batch))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

                # this is experimental and may be changed/removed in the future in favor of a general-purpose one
                if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                    self.train_dataloader.sampler.update(batch=batch)

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                progress_bar.update(1)
                self.global_steps += 1

                if (
                    hasattr(self.config.actor_rollout_ref.actor, "profiler")
                    and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
                ):
                    self.actor_rollout_wg.dump_memory_snapshot(
                        tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                    )

                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    self._flush_image_dumps()
                    self._hf_upload_manager.flush()
                    return

                # this is experimental and may be changed/removed in the future
                # in favor of a general-purpose data buffer pool
                if hasattr(self.train_dataset, "on_batch_end"):
                    # The dataset may be changed after each training batch
                    self.train_dataset.on_batch_end(batch=batch)
