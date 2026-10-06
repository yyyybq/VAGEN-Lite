# Copyright 2025 Bytedance Ltd.
# Licensed under the Apache License, Version 2.0

import asyncio
import copy
import logging
import os
import re
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from PIL import Image
from .agent_loop_no_concat import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from ..envs.gym_image_env import GymImageEnv
from omegaconf import OmegaConf
from vagen.utils.observation_history import ObservationHistory
import traceback
import importlib
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))
from .gym_agent_loop import convert_obs_to_content, extract_success, _flatten_text_only_content, _normalize_images


def _has_contradictory_actions(actions: Any) -> bool:
    names = {str(a).lower() for a in (actions or [])}
    contradictory_pairs = (
        ("turn_left", "turn_right"),
        ("look_up", "look_down"),
        ("move_forward", "move_backward"),
        ("move_left", "move_right"),
    )
    return any(a in names and b in names for a, b in contradictory_pairs)

class AgentState(Enum):
    PENDING = "pending"
    GENERATING = "generating"
    INTERACTING = "interacting"
    TERMINATED = "terminated"


class AgentData:
    """Container for all mutable trajectory state."""
    def __init__(
        self,
        metrics: Dict[str, Any],
        request_id: str,
        env: GymImageEnv,
        response_limit: int,
        env_name: str,
        sys_msg: Optional[Dict[str, Any]] = None,
        sys_images: Optional[List[Image.Image]] = None,
        cur_msg: Optional[Dict[str, Any]] = None,
        cur_images: Optional[List[Image.Image]] = None,
        group_idx: int = 0,
        traj_idx: int = 0,
    ):
        self.sys_msg: Optional[Dict[str, Any]] = sys_msg
        self.sys_images: Optional[List[Image.Image]] = sys_images
        
        self.cur_msg: Optional[Dict[str, Any]] = cur_msg
        self.cur_images: Optional[List[Image.Image]] = cur_images
        
        self.metrics = metrics
        self.request_id = request_id
        self.env = env
        self.response_limit = response_limit
        self.env_name = env_name
        self.group_idx = group_idx
        self.traj_idx = traj_idx
        # Token buffers
        self.turn_prompt_ids: Optional[List[int]] = None
        self.turn_response_ids: Optional[List[int]] = None
        self.turn_response_mask: Optional[List[int]] = None
        self.turn_response_logprobs: Optional[List[int]] = None

        # Env stats
        self.env_turns: int = 0

        # Episode metadata – populated from env info during the episode.
        # Written into every turn's reward_extra_info so concat_val_multi_turn
        # can propagate them (it takes reward_extra_info from the last turn).
        self.task_type: str = "unknown"
        self.scene_id: str = "unknown"
        self.object_label: str = "unknown"
        self.preset: str = "unknown"
        self.task_id: str = "unknown"        # f"{scene_id}/{object_label}/{preset}/{jsonl_idx}"
        self.initial_score: float = 0.0        # potential-field score at reset
        self.initial_distance: float = -1.0    # Euclidean dist to target at t=0
        self.final_score: float = 0.0          # potential-field score at terminal
        self.final_position_score: float = 0.0
        self.final_orientation_score: float = 0.0
        self.score_improvement: float = 0.0
        self.n_primitive_steps: int = 0        # total primitive actions executed
        self.env_exception_count: int = 0

        # Cached assistant text to step env
        self.last_assistant_text: Optional[str] = None
        self.outputs: List[AgentLoopOutput] = []
        self.history = ObservationHistory(1)

# -------------------- Gym Agent Loop --------------------

class GymAgentLoop(AgentLoopBase):
    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        print("Performing class-level GymAgentLoop initialization")

        cls.tokenizer = tokenizer
        cls.processor = processor
        cls.multi_turn_cfg = config.actor_rollout_ref.rollout.multi_turn
        
        # Store module paths for lazy loading; environments are imported on first use
        cls.env_registry_paths = dict(config.env_registry.items())
        cls.env_registry = {}
            
        cls.apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})
        cls.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        cls.response_length = config.actor_rollout_ref.rollout.response_length
        

    @rollout_trace_op
    async def run(self, sampling_params: Dict[str, Any], **kwargs) -> AgentLoopOutput:
        metrics: Dict[str, Any] = {}
        request_id = uuid4().hex

        # Build env (lazy import on first use)
        env_name = kwargs["env_name"]
        if env_name not in self.env_registry:
            if env_name not in self.env_registry_paths:
                raise KeyError(f"Unknown env: {env_name}. Available: {list(self.env_registry_paths.keys())}")
            module_path, class_name = self.env_registry_paths[env_name].rsplit(".", 1)
            module = importlib.import_module(module_path)
            self.env_registry[env_name] = getattr(module, class_name)
        env_cls = self.env_registry[env_name]
        env_config = kwargs["config"]
        seed = kwargs["seed"]
        self.env_max_turns = kwargs.get("max_turns", None)
        env: GymImageEnv = env_cls(env_config=env_config)

        # Bootstrap: reset -> system_prompt (message order: system, then initial user)
        init_obs, info = await env.reset(seed=seed)
        sys_obs = await env.system_prompt()

        # Capture episode metadata from reset info
        _scene_id     = info.get("scene_id",                "unknown") if info else "unknown"
        _task_type    = info.get("task_type",               "unknown") if info else "unknown"
        _object_label = info.get("object_label",            "unknown") if info else "unknown"
        _preset       = info.get("preset",                  "unknown") if info else "unknown"
        _jsonl_idx    = info.get("jsonl_idx",               -1)         if info else -1
        _init_score   = float(info.get("initial_potential_score", 0.0)) if info else 0.0
        # Canonical Active Spatial intentionally withholds target distance to
        # avoid leaking oracle information to the policy.  Keep the legacy
        # diagnostic field numeric without requiring the environment to expose
        # that value.
        _distance_value = info.get("distance") if info else None
        _init_dist = float(_distance_value) if _distance_value is not None else -1.0
        _task_id      = f"{_scene_id}/{_object_label}/{_preset}/{_jsonl_idx}"

        sys_msg={"role": "system", "content": convert_obs_to_content(sys_obs, **kwargs)}
        sys_images=_normalize_images(sys_obs.get("multi_modal_input", {}).get("<image>", []) or [])
        
        cur_msg={"role": "user", "content": convert_obs_to_content(init_obs, **kwargs)}
        cur_images=_normalize_images(init_obs.get("multi_modal_input", {}).get("<image>", []) or [])

        per_turn_response_limit = int(kwargs.get("response_length_per_turn") or self.response_length)
        per_turn_response_limit = min(per_turn_response_limit, self.response_length)
        if per_turn_response_limit <= 0:
            per_turn_response_limit = 1

        agent_data = AgentData(
            sys_msg=sys_msg,
            sys_images=sys_images,
            cur_msg=cur_msg,
            cur_images=cur_images,
            metrics=metrics,
            request_id=request_id,
            env=env,
            response_limit=per_turn_response_limit,
            env_name=kwargs["env_name"],
            group_idx=kwargs["group_idx"],
            traj_idx=kwargs["traj_idx"],
        )
        agent_data.history = ObservationHistory(int(env_config.get("history_window_size", 1)))
        agent_data.public_task = str((info or {}).get("task_prompt") or "")
        if agent_data.env_name == "ActiveSpatial" and not agent_data.public_task:
            raise ValueError("ActiveSpatial reset did not supply a public task")
        agent_data.history.append(cur_msg, cur_images)
        agent_data.task_type       = _task_type
        agent_data.scene_id        = _scene_id
        agent_data.object_label    = _object_label
        agent_data.preset          = _preset
        agent_data.task_id         = _task_id
        agent_data.initial_score   = _init_score
        agent_data.initial_distance = _init_dist

        # State machine: always GENERATE -> INTERACT, and decide termination inside INTERACT
        state = AgentState.PENDING
        while state != AgentState.TERMINATED:
            if state == AgentState.PENDING:
                state = await self._handle_pending_state(agent_data, sampling_params)
            elif state == AgentState.GENERATING:
                state = await self._handle_generating_state(agent_data, sampling_params)
            elif state == AgentState.INTERACTING:
                state = await self._handle_env_state(agent_data, **kwargs)
            else:
                logger.error(f"Invalid state: {state}")
                state = AgentState.TERMINATED

        # Close env after loop
        await env.close()
        return agent_data.outputs

    async def _handle_pending_state(self, agent_data: AgentData, sampling_params: Dict[str, Any]) -> AgentState:
        """Encode initial (system + first user) messages into prompt_ids."""
        image_data = agent_data.sys_images + agent_data.history.images
        if self.processor is not None:
            raw_prompt = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    [agent_data.sys_msg] + agent_data.history.messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            # SenseNova-U1: keep compact ``<image>`` for vLLM PromptReplacement.
            # Pre-expanding here removes the placeholder and makes vLLM assert
            # "Failed to apply prompt replacement". FSDP expand happens later in
            # agent_loop_no_concat postprocess.
            is_u1 = (
                hasattr(self.processor, "image_processor")
                and "SenseNovaU1ImageProcessor"
                in self.processor.image_processor.__class__.__name__
            )
            if is_u1:
                def _tokenize_u1_compact(prompt: str):
                    from vagen.models.sensenova_u1_processor import (
                        IMAGE_TOKEN,
                        locate_image_placeholder,
                    )

                    ids = self.tokenizer.encode(prompt, add_special_tokens=False)
                    bare = self.tokenizer.encode(IMAGE_TOKEN, add_special_tokens=False)
                    # Normalize BPE-fused ``<image>\n\n`` -> bare ``<image>`` + rest
                    # so vLLM string/token target ``<image>`` can match.
                    out: list[int] = []
                    rest = ids
                    while True:
                        idx, span, prefix, suffix = locate_image_placeholder(
                            rest, self.tokenizer
                        )
                        if idx < 0:
                            out.extend(rest)
                            break
                        out.extend(rest[:idx])
                        out.extend(prefix)
                        out.extend(bare)
                        out.extend(suffix)
                        rest = rest[idx + span :]
                    return out

                agent_data.turn_prompt_ids = await self.loop.run_in_executor(
                    None, lambda: _tokenize_u1_compact(raw_prompt)
                )
            else:
                model_inputs = self.processor(
                    text=[raw_prompt], images=image_data, return_tensors="pt"
                )
                agent_data.turn_prompt_ids = model_inputs.pop("input_ids").squeeze(0).tolist()
        else:
            if image_data:
                raise ValueError("Environment returned images but `processor` is None.")
            flat_messages = [_flatten_text_only_content(m) for m in [agent_data.sys_msg] + agent_data.history.messages]
            agent_data.turn_prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    flat_messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
        
        if len(agent_data.turn_prompt_ids) > self.prompt_length:
            if agent_data.history.drop_oldest_turn():
                return await self._handle_pending_state(agent_data, sampling_params)
            raise ValueError("current observation exceeds prompt_length; increase it instead of truncating image tokens")
        return AgentState.GENERATING

    
    async def _handle_generating_state(
        self, agent_data: AgentData, sampling_params: Dict[str, Any]
    ) -> AgentState:
        """Generate assistant output and mark generated tokens with mask=1."""
        sampling_params_for_turn = sampling_params.copy()
        max_new_tokens=sampling_params_for_turn.get("max_new_tokens", None) or agent_data.response_limit
        max_new_tokens = min(max_new_tokens, agent_data.response_limit)
        sampling_params_for_turn["max_new_tokens"] = max_new_tokens
        image_data = agent_data.sys_images + agent_data.history.images
        prompt_ids = agent_data.turn_prompt_ids
        if len(prompt_ids) > self.prompt_length:
            raise ValueError("policy prompt exceeds budget; refusing lossy truncation")
        if getattr(agent_data, "public_task", ""):
            from vagen.utils.task_context import check_policy_tokens
            agent_data.task_context_check = check_policy_tokens(
                self.tokenizer, prompt_ids, agent_data.public_task, self.prompt_length)

        with simple_timer("generate_sequences", agent_data.metrics):
            output = await self.server_manager.generate(
                request_id = agent_data.request_id,
                prompt_ids = prompt_ids,
                sampling_params = sampling_params_for_turn,
                image_data = image_data,
            )


        agent_data.turn_response_ids = output.token_ids
        agent_data.turn_response_mask = [1] * len(output.token_ids)
        agent_data.turn_prompt_ids += agent_data.turn_response_ids
        if output.log_probs:
            agent_data.turn_response_logprobs = output.log_probs

        # Cache assistant text and add assistant message (text-only)
        assistant_message = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(agent_data.turn_response_ids, skip_special_tokens=True)
        )
        agent_data.last_assistant_text = assistant_message
        return AgentState.INTERACTING

    async def _handle_env_state(self, agent_data: AgentData, **kwargs) -> AgentState:
        """
        Step the environment with last assistant action; always collect reward first.
        If terminal (done/success/turn-limit/token-limit), stop WITHOUT appending user suffix,
        so the episode ends on an assistant turn.
        """
        action_str = agent_data.last_assistant_text or ""
        try:
            obs, reward, done, info = await agent_data.env.step(action_str)
        except Exception as exc:
            logger.error(
                "Environment step failed in '%s' with action %r: %s",
                agent_data.env_name,
                action_str,
                exc,
            )
            logger.error("Environment traceback:\n%s", traceback.format_exc())
            if kwargs.get("env_exception_fail_fast", True):
                raise RuntimeError(
                    f"Environment step failed in {agent_data.env_name}: {type(exc).__name__}: {exc}"
                ) from exc
            agent_data.env_exception_count += 1
            obs, reward, done, info = {"obs_str": "Environment Error"}, 0.0, True, {
                "traj_success": False,
                "env_exception": True,
                "env_exception_type": type(exc).__name__,
            }

        traj_success = extract_success(info)
        info = info or {}
        env_metrics = info.get("metrics") or {}
        turn_metrics = env_metrics.get("turn_metrics") or {}
        traj_metrics = env_metrics.get("traj_metrics") or {}
        action_list = info.get("actions") or []
        action_valid = bool(turn_metrics.get("action_is_valid", bool(action_list)))
        env_exception = bool(info.get("env_exception", False))
        empty_action = len(action_list) == 0
        contradictory_action = _has_contradictory_actions(action_list)
        agent_data.env_turns += 1
        last_turn=False
        turn_limit_truncated = False

        # Keep an episode-level snapshot on every turn. Validation concatenation
        # takes reward_extra_info from the last emitted turn, which may be caused
        # by max-turn truncation rather than env done=True.
        agent_data.final_score = float(
            info.get("final_score", info.get("current_potential_score", agent_data.final_score))
        )
        _tm = (info.get("metrics") or {}).get("traj_metrics", {})
        agent_data.final_position_score = float(_tm.get("final_position_score", agent_data.final_position_score))
        agent_data.final_orientation_score = float(_tm.get("final_orientation_score", agent_data.final_orientation_score))
        agent_data.n_primitive_steps = int(info.get("env_step", agent_data.n_primitive_steps))
        agent_data.score_improvement = agent_data.final_score - agent_data.initial_score

        if done:
            last_turn = True

        if self.env_max_turns is not None and agent_data.env_turns >= int(self.env_max_turns):
            last_turn = True
            turn_limit_truncated = not bool(done)

        reward_trace = copy.deepcopy(info.get("reward_trace") or {})
        reward_trace_flat = dict(info.get("reward_trace_flat") or {})
        if turn_limit_truncated:
            reward_trace.setdefault("outcome", {})["terminated"] = False
            reward_trace["outcome"]["truncated"] = True
            reward_trace["outcome"]["termination_reason"] = "max_llm_turns"
            reward_trace_flat["reward_trace/terminated"] = 0.0
            reward_trace_flat["reward_trace/truncated"] = 1.0
            reward_trace_flat["reward_trace/termination_reason"] = "max_llm_turns"

        turn_images = agent_data.sys_images + agent_data.history.images

        # NFP: collect next-frame images (the observation rendered AFTER this action).
        # These serve as the prediction target for the Next Frame Prediction head.
        # For terminal turns (done=True), there is no genuine next frame; we use
        # the current frame as a dummy so all batch samples always carry this key.
        # The nfp_loss_mask will be all-zeros for terminal turns, so the dummy
        # image contributes zero loss.
        _raw_next_images = _normalize_images(obs.get("multi_modal_input", {}).get("<image>", []) or [])
        # Only when the snapshot hook is explicitly armed, retain an immutable
        # next-observation source for the terminal-aware GAE *candidate*.  It is
        # metadata, never used by rollout or PPO, and is deliberately absent in
        # ordinary training to avoid carrying renderer frames through the batch.
        snapshot_capture = bool(os.environ.get("VAGEN_ACTIVE_SPATIAL_PPO_SNAPSHOT_DIR"))
        bootstrap_context = {
            "capture_enabled": snapshot_capture,
            "next_state_available": bool(_raw_next_images),
            "next_observation_text": str(obs.get("obs_str", "")),
            "next_images": [],
        }
        if snapshot_capture:
            for image in _raw_next_images:
                if not isinstance(image, Image.Image):
                    continue
                bootstrap_context["next_images"].append({
                    "mode": image.mode,
                    "size": list(image.size),
                    "pixel_bytes": image.tobytes(),
                })
        snapshot_extra_fields = {}
        if snapshot_capture:
            snapshot_extra_fields = {
                "action_text": action_str,
                "parsed_primitive_actions": [str(action) for action in action_list],
                "bootstrap_context": bootstrap_context,
            }
        if last_turn or not _raw_next_images:
            # Terminal or empty observation: dummy = first current-frame image
            nfp_target_images = agent_data.cur_images[:1] if agent_data.cur_images else []
            nfp_valid = False
        else:
            nfp_target_images = _raw_next_images
            nfp_valid = True
        
        resp_len = len(agent_data.turn_response_mask)
        response_ids = agent_data.turn_prompt_ids[-resp_len:] if resp_len else []
        prompt_ids = agent_data.turn_prompt_ids[: len(agent_data.turn_prompt_ids) - resp_len]
        multi_modal_data = {"image": turn_images} if turn_images else {}
        output = AgentLoopOutput(
            prompt_ids=prompt_ids[-self.prompt_length:],
            response_ids=response_ids[: self.response_length],
            response_mask=agent_data.turn_response_mask[: self.response_length],
            multi_modal_data=multi_modal_data,
            response_logprobs=(
                agent_data.turn_response_logprobs[: self.response_length] if agent_data.turn_response_logprobs else None
            ),
            reward_score=float(reward),
            num_turns=1,
            metrics=agent_data.metrics,
            extra_fields={"reward_extra_info": {
                "task_context_present": float(bool(getattr(agent_data, "task_context_check", {}).get("task_present"))),
                "traj_success":             float(traj_success),
                "task_type":                agent_data.task_type,
                "scene_id":                 agent_data.scene_id,
                "object_label":             agent_data.object_label,
                "preset":                   agent_data.preset,
                "task_id":                  agent_data.task_id,
                "initial_score":            agent_data.initial_score,
                "initial_distance":         agent_data.initial_distance,
                "final_score":              agent_data.final_score,
                "final_position_score":     agent_data.final_position_score,
                "final_orientation_score":  agent_data.final_orientation_score,
                "score_improvement":        agent_data.score_improvement,
                "n_primitive_steps":        float(agent_data.n_primitive_steps),
                "env_exception":           float(env_exception),
                "env_exception_count":     float(agent_data.env_exception_count),
                "invalid_action":          float(not action_valid),
                "empty_action":            float(empty_action),
                "contradictory_action":    float(contradictory_action),
                "missing_action_tag":      float(turn_metrics.get("missing_action_tag", 0.0)),
                "empty_action_body":       float(turn_metrics.get("empty_action_body", 0.0)),
                "unknown_action_name":     float(turn_metrics.get("unknown_action_name", 0.0)),
                "truncated_before_action": float(turn_metrics.get("truncated_before_action", 0.0)),
                "multiple_action_tag":     float(turn_metrics.get("multiple_action_tag", 0.0)),
                "strict_action_tag_rate":  float(turn_metrics.get("strict_action_tag", 0.0)),
                "tool_call_rate":          float(turn_metrics.get("tool_call", 0.0)),
                "fallback_parse_rate":     float(turn_metrics.get("fallback_parse", 0.0)),
                "strict_parse_success_rate": float(turn_metrics.get("strict_parse_success", 0.0)),
                "strict_format_correct_rate": float(turn_metrics.get("strict_format_correct", 0.0)),
                "format_penalty_rate":      float(not bool(turn_metrics.get("strict_format_correct", 0.0))),
                "collision_termination":    float(bool(info.get("early_terminated_collision", False))),
                "low_info_termination":     float(bool(info.get("early_terminated_low_info", False))),
                "renderer_failure":         float(bool(info.get("renderer_failure", False))),
                "episode_length":           float(agent_data.env_turns),
                "action_move_forward":      float("move_forward" in action_list),
                "action_move_backward":     float("move_backward" in action_list),
                "action_move_left":         float("move_left" in action_list),
                "action_move_right":        float("move_right" in action_list),
                "action_turn_left":         float("turn_left" in action_list),
                "action_turn_right":        float("turn_right" in action_list),
                "invalid_action_count":    float(traj_metrics.get("invalid_action_count", 0.0)),
                "episode_invalid_action_count": float(traj_metrics.get("invalid_action_count", 0.0)),
                "episode_empty_action_count": float(traj_metrics.get("empty_action_count", 0.0)),
                "episode_contradictory_action_count": float(traj_metrics.get("contradictory_action_count", 0.0)),
                "episode_empty_action_rate": float(traj_metrics.get("empty_action_rate", 0.0)),
                "episode_contradictory_action_rate": float(traj_metrics.get("contradictory_action_rate", 0.0)),
                "terminated":               float(bool(info.get("terminated", done)) and not turn_limit_truncated),
                "truncated":                float(bool(info.get("truncated", False)) or turn_limit_truncated),
                "turn_limit_truncated":     float(turn_limit_truncated),
                "termination_reason":       (
                    "max_llm_turns" if turn_limit_truncated else str(info.get("termination_reason", "continuing"))
                ),
                **reward_trace_flat,
                },
                "reward_trace": reward_trace,
                **snapshot_extra_fields,
                "image_data": turn_images,
                "last_turn": last_turn,
                "group_idx": agent_data.group_idx,
                "traj_idx": agent_data.traj_idx,
                "turn_idx": agent_data.env_turns,
                "d0_4_request_id": agent_data.request_id,
                # NFP next-frame targets
                "nfp_target_images": nfp_target_images,
                "nfp_valid": nfp_valid,
            },
        )
        agent_data.outputs.append(output)
        
        # update cur msg and images
        cur_msg={"role": "user", "content": convert_obs_to_content(obs, **kwargs)}
        cur_images=_normalize_images(obs.get("multi_modal_input", {}).get("<image>", []) or [])
        agent_data.cur_msg = cur_msg
        agent_data.cur_images = cur_images
        agent_data.history.append({"role": "assistant", "content": agent_data.last_assistant_text or ""})
        agent_data.history.append(cur_msg, cur_images)
        if last_turn:
            return AgentState.TERMINATED

        return AgentState.PENDING
