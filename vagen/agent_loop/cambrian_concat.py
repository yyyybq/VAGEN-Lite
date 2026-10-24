"""Cambrian-S on the concat path, one sequence per episode (the model line, step 2 of 2026-10-24).

The agent loop (vagen.agent_loop.gym_agent_loop.GymAgentLoop) and vLLM keep the compact form: one <image> token
(151665, added to the tokenizer by CambrianProcessorWrapper) per frame, which vLLM's Cambrian processor expands to
756 <|image_pad|> positions (vagen/models/cambrian_vllm.py).  The trainer needs the expanded form, so this worker
expands every sample after verl's padding (expand_sample):

  - each <image> becomes TOKENS_PER_IMAGE (756) <|image_pad|> tokens (151655) in prompts, responses and input_ids --
    the sequence vLLM sampled from; the adapter takes them as its image positions when its config has
    image_token_alias = 151655 (vagen/models/cambrian_register.py, set through actor_rollout_ref.model.override_config),
    so the trainer's labels and decoding only meet an ordinary special token;
  - attention mask 1, response mask 0 and rollout log-probability 0 at those positions;
  - the episode's frames through SigLIP2 (the processor's preprocess_images) as multi_modal_inputs.pixel_values;
  - 1-D positions over the expanded sequence;
  - fixed lengths, so every worker's batch concatenates: the prompt left-padded to rollout.prompt_length (the Cambrian
    RL config raises that budget by 756 tokens per frame, since vLLM's max_model_len is prompt_length +
    response_length and counts expanded tokens), the response right-padded to rollout.response_length + 755 x
    cambrian_concat.max_images.  A sample that does not fit is an error, never cut.
The invalid-turn spans (vagen/invalid_turns.py) move with their tokens.

Selected by trainer.agent_loop_manager = vagen.agent_loop.cambrian_concat.CambrianAgentLoopManager
(vagen/ray_trainer.py); training_prep rl-config --model-type cambrian_qwen writes it (vagen/models/model_types.py).
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import ray
import torch

from verl.experimental.agent_loop.agent_loop import AgentLoopManager, AgentLoopWorkerBase, _InternalAgentLoopOutput
from verl.utils.model import compute_position_id_with_mask

IMAGE_TOKEN_ID = 151665  # the compact <image> (CambrianProcessorWrapper)
IMAGE_PAD_ID = 151655  # <|image_pad|>: the expanded positions, as in vLLM (cambrian_vllm.IMAGE_PAD_TOKEN_ID)
TOKENS_PER_IMAGE = 756  # 27 x (27 + 1)


def _expand(ids: Sequence[int], image_id: int, alias: int, per_image: int):
    """(expanded ids, image flags, where): ``where[i]`` is the expanded index of compact token i, and
    ``where[len(ids)]`` the expanded length."""
    out: List[int] = []
    image: List[bool] = []
    where: List[int] = []
    for t in ids:
        where.append(len(out))
        if t == image_id:
            out.extend([alias] * per_image)
            image.extend([True] * per_image)
        else:
            out.append(int(t))
            image.append(False)
    where.append(len(out))
    return out, image, where


def expand_sample(prompt_ids: torch.Tensor, prompt_attention: torch.Tensor, response_ids: torch.Tensor,
                  response_attention: torch.Tensor, response_mask: torch.Tensor,
                  response_logprobs: Optional[torch.Tensor], *, prompt_length: int, response_length: int,
                  pad_id: int, image_id: int = IMAGE_TOKEN_ID, alias: int = IMAGE_PAD_ID,
                  per_image: int = TOKENS_PER_IMAGE) -> Dict[str, Any]:
    """One sample of verl's agent loop (tensors of shape (1, n): the prompt left-padded, the response right-padded)
    in the expanded form at the fixed lengths.  Returns the tensors (shape (1, n)), the frames counted in the prompt
    and the response, and ``where`` (compact response index -> expanded response index)."""
    p_valid = prompt_ids[0][prompt_attention[0].bool()].tolist()
    n_resp = int(response_attention[0].sum().item())
    if not bool(response_attention[0, :n_resp].all()):
        raise ValueError("the response is not right-padded")
    r_valid = response_ids[0, :n_resp].tolist()
    p_ids, _, _ = _expand(p_valid, image_id, alias, per_image)
    r_ids, r_image, where = _expand(r_valid, image_id, alias, per_image)
    if len(p_ids) > prompt_length:
        raise ValueError(f"expanded prompt of {len(p_ids)} tokens > prompt_length {prompt_length}")
    if len(r_ids) > response_length:
        raise ValueError(f"expanded response of {len(r_ids)} tokens > {response_length}: more frames than "
                         "cambrian_concat.max_images")
    mask_c = response_mask[0, :n_resp].tolist()
    if any(m and t == image_id for m, t in zip(mask_c, r_valid)):
        raise ValueError("an <image> token is marked as generated")
    r_mask: List[int] = []
    for i, m in enumerate(mask_c):
        r_mask.extend([0] * (where[i + 1] - where[i]) if r_valid[i] == image_id else [int(m)])
    lp: Optional[List[float]] = None
    if response_logprobs is not None:
        lp_c = response_logprobs[0, :n_resp].tolist()
        lp = []
        for i, v in enumerate(lp_c):
            lp.extend([0.0] * (where[i + 1] - where[i]) if r_valid[i] == image_id else [float(v)])
    p_pad, r_pad = prompt_length - len(p_ids), response_length - len(r_ids)
    prompt = torch.tensor([[pad_id] * p_pad + p_ids], dtype=prompt_ids.dtype)
    p_att = torch.tensor([[0] * p_pad + [1] * len(p_ids)], dtype=prompt_attention.dtype)
    response = torch.tensor([r_ids + [pad_id] * r_pad], dtype=response_ids.dtype)
    r_att = torch.tensor([[1] * len(r_ids) + [0] * r_pad], dtype=response_attention.dtype)
    attention_mask = torch.cat([p_att, r_att], dim=1)
    out = {"prompt_ids": prompt, "response_ids": response, "input_ids": torch.cat([prompt, response], dim=1),
           "attention_mask": attention_mask, "position_ids": compute_position_id_with_mask(attention_mask),
           "response_mask": torch.tensor([r_mask + [0] * r_pad], dtype=response_mask.dtype),
           "response_logprobs": None if lp is None else torch.tensor([lp + [0.0] * r_pad],
                                                                    dtype=response_logprobs.dtype),
           "frames_prompt": sum(1 for t in p_valid if t == image_id),
           "frames_response": sum(1 for t in r_valid if t == image_id), "where": where}
    if sum(r_image) != per_image * out["frames_response"]:
        raise AssertionError("expanded image positions do not match the frames")
    return out


def move_spans(spans: Any, where: Sequence[int]) -> Any:
    """The agent loop's invalid-turn spans ({"spans": [[start, end], ...]}, compact response indices) in expanded
    indices (spans past the response end are clipped to it, as vagen.invalid_turns.spans_mask does)."""
    if not isinstance(spans, dict) or not spans.get("spans"):
        return spans
    last = len(where) - 1
    return dict(spans, spans=[[where[min(int(s), last)], where[min(int(e), last)]] for s, e in spans["spans"]])


def expanded_lengths(config) -> Dict[str, int]:
    """The fixed lengths of the expanded samples, after checking the config holds what the expansion needs."""
    alias = (config.actor_rollout_ref.model.get("override_config") or {}).get("image_token_alias")
    if alias is None or int(alias) != IMAGE_PAD_ID:
        raise ValueError("Cambrian-S on the concat path needs actor_rollout_ref.model.override_config."
                         f"image_token_alias = {IMAGE_PAD_ID} (the adapter's image positions)")
    max_images = int((config.get("cambrian_concat") or {}).get("max_images") or 0)
    if max_images <= 0:
        raise ValueError("cambrian_concat.max_images (the frames an episode can have) is needed")
    rollout = config.actor_rollout_ref.rollout
    return {"max_images": max_images, "prompt": int(rollout.prompt_length),
            "response": int(rollout.response_length) + (TOKENS_PER_IMAGE - 1) * max_images}


class CambrianAgentLoopWorkerBase(AgentLoopWorkerBase):
    """verl's agent-loop worker with Cambrian-S's processor (the compact <image> token for the agent loop and vLLM)
    and the expansion for the trainer."""

    def __init__(self, config, server_handles, reward_router_address: str = None):
        lengths = expanded_lengths(config)  # before verl's set-up: a wrong config fails at once
        super().__init__(config, server_handles, reward_router_address)
        from vagen.models.cambrian_processor import CambrianProcessorWrapper

        self.processor = CambrianProcessorWrapper(self.tokenizer)  # adds <image> to this worker's tokenizer
        if self.processor.image_token_id != IMAGE_TOKEN_ID:
            raise ValueError(f"<image> is token {self.processor.image_token_id}, the expansion expects {IMAGE_TOKEN_ID}")
        self.expanded_prompt_length = lengths["prompt"]
        self.expanded_response_length = lengths["response"]

    def expand_output(self, out: _InternalAgentLoopOutput) -> _InternalAgentLoopOutput:
        pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
        e = expand_sample(out.prompt_ids, out.attention_mask[:, : out.prompt_ids.shape[1]], out.response_ids,
                          out.attention_mask[:, out.prompt_ids.shape[1]:], out.response_mask, out.response_logprobs,
                          prompt_length=self.expanded_prompt_length, response_length=self.expanded_response_length,
                          pad_id=pad_id)
        frames = e["frames_prompt"] + e["frames_response"]
        images = list((out.multi_modal_data or {}).get("image") or [])
        if len(images) < frames:
            raise ValueError(f"{frames} <image> tokens but {len(images)} frames")
        # frames are kept in order; a response cut at its length drops only trailing ones
        multi_modal_inputs = dict(self.processor.preprocess_images(images[:frames])) if frames else None
        extra = dict(out.extra_fields)
        if "invalid_turn_spans" in extra:
            extra["invalid_turn_spans"] = move_spans(extra["invalid_turn_spans"], e["where"])
        return out.model_copy(update={
            "prompt_ids": e["prompt_ids"], "response_ids": e["response_ids"], "input_ids": e["input_ids"],
            "position_ids": e["position_ids"], "response_mask": e["response_mask"],
            "attention_mask": e["attention_mask"], "response_logprobs": e["response_logprobs"],
            "multi_modal_inputs": multi_modal_inputs, "extra_fields": extra})

    async def _run_agent_loop(self, sampling_params: dict[str, Any], trajectory: dict[str, Any], *,
                              agent_name: str, **kwargs) -> _InternalAgentLoopOutput:
        out = await super()._run_agent_loop(sampling_params, trajectory, agent_name=agent_name, **kwargs)
        return self.expand_output(out)


CambrianAgentLoopWorker = ray.remote(CambrianAgentLoopWorkerBase)


class CambrianAgentLoopManager(AgentLoopManager):
    """verl's agent-loop manager with CambrianAgentLoopWorker."""

    def __init__(self, config, worker_group=None, rm_wg=None):
        self.agent_loop_workers_class = CambrianAgentLoopWorker
        super().__init__(config, worker_group=worker_group, rm_wg=rm_wg)
