"""Per-token rollout-vs-trainer log-probabilities (the model line, 2026-10-26, the user's decision 1): where the
difference between vLLM's sampled-token log-probabilities and the trainer's recomputed ones sits.

With the audit on (rl-config --rollout-log-probs, VAGEN_D0_3_SMOKE_DIR) and PPO config trainer.logprob_tokens
(rl-config --logprob-tokens), the trainer writes, before each update, VAGEN_D0_3_SMOKE_DIR/tokens/step_NNNN.npz with
one entry per trained response token (response_mask 1):
    sample          the sequence's row in the batch
    token           the token id
    rollout         vLLM's log-probability of the token (rollout_log_probs)
    trainer         the trainer's (old_log_probs, before the update)
    turn            0 for the first answer of the episode, 1 for the second, ... (the runs of response_mask)
    pos_in_turn     the token's place in its answer (0 = the answer's first token)
    image_distance  sequence positions back to the nearest image token (<|image_pad|>, or a negative id), -1 if none
Nothing else changes; model_matrix/logprob/breakdown.py reads the files.
"""
from __future__ import annotations

import os
from typing import Dict, Iterable

import numpy as np
import torch

IMAGE_PAD = 151655  # <|image_pad|> of the Qwen2.5-VL / Qwen3-VL tokenizers (and the Cambrian trainer expansion)


def _run_starts(mask: torch.Tensor) -> torch.Tensor:
    prev = torch.cat([torch.zeros(1, dtype=torch.bool), mask[:-1]])
    return mask & ~prev


def token_records(data, image_ids: Iterable[int] = (IMAGE_PAD,)) -> Dict[str, np.ndarray]:
    batch = data.batch
    rollout = batch["rollout_log_probs"].detach().float().cpu()
    trainer = batch["old_log_probs"].detach().float().cpu()
    responses = batch["responses"].detach().cpu()
    mask = batch["response_mask"].detach().bool().cpu()
    input_ids = batch["input_ids"].detach().cpu()
    n_resp = responses.shape[1]
    prompt_len = input_ids.shape[1] - n_resp
    images = torch.tensor(sorted({int(i) for i in image_ids}), dtype=input_ids.dtype)
    cols = {k: [] for k in ("sample", "token", "rollout", "trainer", "turn", "pos_in_turn", "image_distance")}
    for b in range(responses.shape[0]):
        m = mask[b]
        if not bool(m.any()):
            continue
        ids = input_ids[b]
        pos = torch.arange(ids.shape[0])
        is_image = torch.isin(ids, images) | (ids < 0)
        last_image = torch.cummax(torch.where(is_image, pos, torch.full_like(pos, -1)), dim=0).values
        j = m.nonzero(as_tuple=True)[0]
        absolute = prompt_len + j
        last = last_image[absolute]
        starts = _run_starts(m)
        turn = torch.cumsum(starts.long(), dim=0) - 1
        run_start = torch.cummax(torch.where(starts, torch.arange(n_resp), torch.zeros(n_resp, dtype=torch.long)),
                                 dim=0).values
        cols["sample"].append(np.full(j.shape[0], b, dtype=np.int32))
        cols["token"].append(responses[b, j].numpy().astype(np.int64))
        cols["rollout"].append(rollout[b, j].numpy().astype(np.float32))
        cols["trainer"].append(trainer[b, j].numpy().astype(np.float32))
        cols["turn"].append(turn[j].numpy().astype(np.int32))
        cols["pos_in_turn"].append((j - run_start[j]).numpy().astype(np.int32))
        cols["image_distance"].append(torch.where(last >= 0, absolute - last, torch.full_like(last, -1)).numpy()
                                      .astype(np.int64))
    return {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in cols.items()}


def image_token_ids(tokenizer) -> set:
    ids = {IMAGE_PAD}
    try:
        tid = tokenizer.convert_tokens_to_ids("<|image_pad|>")
        if isinstance(tid, int) and tid >= 0:
            ids.add(tid)
    except Exception:
        pass
    return ids


def write_logprob_tokens(data, path: str, global_step: int, tokenizer=None) -> Dict[str, float]:
    if "rollout_log_probs" not in data.batch or "old_log_probs" not in data.batch:
        return {}
    os.makedirs(path, exist_ok=True)
    rec = token_records(data, image_token_ids(tokenizer) if tokenizer is not None else (IMAGE_PAD,))
    tmp = os.path.join(path, f"step_{int(global_step):04d}.tmp.npz")
    np.savez_compressed(tmp, **rec)
    os.replace(tmp, os.path.join(path, f"step_{int(global_step):04d}.npz"))
    return {"logprob_tokens/tokens": float(len(rec["token"]))}
