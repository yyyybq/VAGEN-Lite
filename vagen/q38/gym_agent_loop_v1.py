# Copyright 2025 Bytedance Ltd.
# Licensed under the Apache License, Version 2.0
"""The q38 agent loop for the V1 trainer family (docs/q38/DESIGN_ASYNC.md section 2.5.3).

One episode is one sequence ("concat" mode, as the 8B GymAgentLoop, vagen/agent_loop/gym_agent_loop.py):

    initial prompt (system + first observation + generation prompt)      -> AgentLoopOutput.prompt_ids
    answer_1 | <\\n> user_2 + generation prompt | answer_2 | ...          -> AgentLoopOutput.response_ids

The tokens are built with verl's Continuous Token builder (AgentLoopBase.ct_build_initial_tokens /
ct_merge_assistant_token / ct_merge_context_msg): the model's answer tokens go into the stream as generated (mask 1,
their rollout logprobs), every observation is rendered by the model's own chat template through the processor (mask 0).
For the Qwen family the builder inserts the template's "\\n" after a generated <|im_end|> (mask 0), which the 8B loop
did not do (recipe change R9 (1), DESIGN_ASYNC section 3.3).

What is kept from the 8B loop (GL = vagen/agent_loop/gym_agent_loop.py at qwen38 551c241e), line for line where noted:
  - the env is built in-process from the row (env_name / config / seed, GL:240-251) and stepped with the answer decoded
    with skip_special_tokens=True (GL:434-436);
  - the termination rules, in GL's order (GL:480-490): #3 env done or success, #2 env_turns >= max_turns,
    #1 the response budget is used up;
  - an exception in env.step ends the episode as done with reward 0 (GL:448-459); here it is also counted
    (vagen_stats.step_exceptions, MF5);
  - reward_score = the sum of the env rewards; reward_extra_info has the 8B scalars (GL:339-352).

What is new:
  - each turn asks for max_tokens = min(response_length_per_turn, response budget left): an explicit limit is required,
    otherwise FullyAsyncLLMServerClient takes rollout.response_length (4096) as the cumulative budget of a resumed turn
    (verl/workers/rollout/llm_server.py:254-274); the last turn of a long episode is therefore cut at the budget, where
    the 8B loop generated a full turn and cut the sequence afterwards (R9 (2), MF11);
  - an observation that does not fit the budget is not appended: the episode ends on the assistant turn
    (as verl/experimental/agent_loop/tool_agent_loop.py:426-427), so image placeholders are never cut;
  - the observation's images are carried inside the user message's content blocks ({"type": "image", "image": PIL}),
    which the VL builder needs to render the observation (continuous_token.py:1153-1170, 1248-1269);
  - the weight versions of the episode (extra_fields min_global_steps / max_global_steps, the min and the max over its
    generate calls, as tool_agent_loop.py:283-292) for the V1 staleness tags (agent_loop_tq.py:229-233) and the
    sampler's version check (MF3);
  - extra_fields.vagen_stats: per-episode counters and timings for the sampler's vagen/env/* and vagen/gen/* metrics;
  - the render service address is the current process's R1_RENDER_URL, not the one the row carries
    (refresh_render_url): the env yaml has client_url: ${oc.env:R1_RENDER_URL}, which the dataset resolves when it
    loads, and a resume re-issues the in-flight rows from the TransferQueue snapshot with the address of the run that
    wrote the checkpoint, while the render service (its own sco job) has a new address after every restart.  An
    episode whose row carried another address counts vagen_stats.render_url_refreshed = 1
    (vagen/env/render_url_refreshed), and each process logs the first such replacement once (WARNING).

Every extra field is JSON-style (str keys, lists, Python scalars): a tuple comes back from TransferQueue as a list, and
every episode carries the same keys (TransferQueue's kv_batch_get refuses samples with different top-level fields).

The data row's max_turns and response_length_per_turn must equal vagen.q38.contract.MAX_TURNS / RESPONSE_LENGTH_PER_TURN
(12 / 384, MF8); the episode's images may not exceed rollout.limit_images (contract.LIMIT_IMAGES = 13 when unset).

Configuration read here (all optional):
  vagen.parity_mode                (false)
  vagen.parity_sampling_override   (null) a dict merged into every turn's sampling params (the 8B parity run samples at
                                          T 0.8 / top-p 0.92, DESIGN_ASYNC 5.3); refused without vagen.parity_mode
  env_registry                     name -> "module.Class" (GL:204); ActiveSpatial by default
"""

from __future__ import annotations

import importlib
import logging
import os
import time
import traceback
from collections.abc import Mapping
from typing import Any, Optional
from uuid import uuid4

from omegaconf import DictConfig, OmegaConf
from PIL import Image

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput
from verl.utils.rollout_trace import rollout_trace_op

from vagen.q38 import contract

# the 8B loop's helpers, imported unchanged (GL is read-only for this line)
from vagen.agent_loop.gym_agent_loop import (
    _normalize_images,
    convert_obs_to_content,
    extract_success,
    turn_invalid,
)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

DEFAULT_ENV_REGISTRY = {
    "ActiveSpatial": "vagen.envs.active_spatial.active_spatial_env.ActiveSpatialGymEnv",
}

# the keys of extra_fields.vagen_stats, in a fixed set (every episode carries all of them)
VAGEN_STATS_KEYS = (
    "step_exceptions",
    "render_s",
    "turn_gen_s",
    "turn_max_tokens",
    "turn_spans",
    "endoftext_stops",
    "length_stops",
    "turns_resumed",
    "num_preempted",
    "gen_turns",
    "env_turns",
    "termination",
    "render_url_refreshed",
)

# the render service address: the env yaml's client_url is ${oc.env:R1_RENDER_URL} (8B train_env.yaml), and every Ray
# process gets R1_RENDER_URL from the entry (vagen/q38/ppo_v1.py runtime_env_vars)
RENDER_URL_ENV = "R1_RENDER_URL"
RENDER_URL_KEY = "client_url"

# the keys of extra_fields.reward_extra_info: the 8B scalars (GL:339-352)
REWARD_EXTRA_INFO_KEYS = (
    "traj_success",
    "task_type",
    "scene_id",
    "task_id",
    "initial_score",
    "final_score",
    "score_improvement",
    "n_primitive_steps",
    "n_turns",
    "invalid_action",
)


def _select(config: Any, key: str, default: Any = None) -> Any:
    """OmegaConf.select on a DictConfig, a nested dict lookup otherwise."""
    if isinstance(config, DictConfig):
        return OmegaConf.select(config, key, default=default)
    node = config
    for part in key.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node


def _plain(value: Any) -> Any:
    """A DictConfig / ListConfig as plain Python containers, anything else unchanged."""
    if isinstance(value, (DictConfig,)) or OmegaConf.is_list(value):
        return OmegaConf.to_container(value, resolve=True)
    return value


def _opt_int(value: Any) -> Optional[int]:
    return None if value is None else int(value)


def refresh_render_url(env_config: Any, environ: Optional[Mapping[str, str]] = None) -> tuple[Any, Optional[str]]:
    """The row's env config with client_url = the current R1_RENDER_URL; (config, the row's old address or None).

    The dataset resolves ``client_url: ${oc.env:R1_RENDER_URL}`` when it loads the env yaml, so a row carries the
    address of the process that loaded it; after a resume the in-flight rows come back from the TransferQueue snapshot
    with the address of the run that wrote the checkpoint (S3 phase b, 2026-11-18).  The current process's variable
    wins.  Returned unchanged (the same object, None) when R1_RENDER_URL is unset or empty, when the config has no
    client_url key (another env), or when the address is already the current one; otherwise a shallow copy with the
    new address (the row's own dict is not edited) and the address the row carried."""
    environ = os.environ if environ is None else environ
    current = environ.get(RENDER_URL_ENV)
    if not current:
        return env_config, None
    if not isinstance(env_config, Mapping) or RENDER_URL_KEY not in env_config:
        return env_config, None
    old = env_config[RENDER_URL_KEY]
    if old == current:
        return env_config, None
    refreshed = dict(env_config)
    refreshed[RENDER_URL_KEY] = current
    return refreshed, str(old)


def attach_images(content: list[dict[str, Any]], images: list[Image.Image]) -> list[dict[str, Any]]:
    """Put the observation's images, in order, into its {"type": "image"} content blocks.

    convert_obs_to_content (GL:152-186) gives bare image blocks and has already checked that the obs text has one
    <image> per image; the VL Continuous Token builder finds the images of an appended message in its blocks.
    """
    out, it = [], iter(images)
    for block in content:
        if block.get("type") == "image":
            try:
                block = {**block, "image": next(it)}
            except StopIteration:
                raise ValueError("observation has more <image> placeholders than images") from None
            out.append(block)
        else:
            out.append(block)
    if next(it, None) is not None:
        raise ValueError("observation has more images than <image> placeholders")
    return out


class _Episode:
    """Mutable state of one episode."""

    def __init__(self, request_id: str, want_logprobs: bool):
        self.request_id = request_id
        self.messages: list[dict[str, Any]] = []
        self.images: list[Image.Image] = []
        self.prompt_ids: list[int] = []  # the runtime token stream: initial prompt + response so far
        self.initial_prompt_len = 0
        self.response_mask: list[int] = []
        self.response_logprobs: Optional[list[float]] = [] if want_logprobs else None
        self.last_answer_text = ""

        # env stats, as GL's AgentData (GL:93-116)
        self.env_rewards: list[float] = []
        self.traj_success = False
        self.env_turns = 0
        self.invalid_turns = 0
        self.task_type = "unknown"
        self.scene_id = "unknown"
        self.task_id = ""
        self.initial_score = 0.0
        self.final_score = 0.0
        self.n_primitive_steps = 0

        # weight versions over the episode's generate calls
        self.min_global_steps: Optional[int] = None
        self.max_global_steps: Optional[int] = None

        # vagen_stats
        self.step_exceptions = 0
        self.render_s: list[float] = []
        self.turn_gen_s: list[float] = []
        self.turn_max_tokens: list[int] = []
        self.turn_spans: list[list[int]] = []
        self.endoftext_stops = 0
        self.length_stops = 0
        self.turns_resumed = 0
        self.num_preempted: Optional[int] = None
        self.termination = ""
        self.render_url_refreshed = 0

    @property
    def want_logprobs(self) -> bool:
        """Whether the episode carries rollout logprobs (rollout.calculate_log_probs)."""
        return self.response_logprobs is not None

    def add_versions(self, extra: dict[str, Any]) -> None:
        """Fold one generate call's weight versions in (sync clients only give global_steps)."""
        global_steps = extra.get("global_steps")
        lo = extra.get("min_global_steps")
        hi = extra.get("max_global_steps")
        lo = _opt_int(global_steps if lo is None else lo)
        hi = _opt_int(global_steps if hi is None else hi)
        if lo is not None:
            self.min_global_steps = lo if self.min_global_steps is None else min(self.min_global_steps, lo)
        if hi is not None:
            self.max_global_steps = hi if self.max_global_steps is None else max(self.max_global_steps, hi)
        if lo is not None and hi is not None and lo != hi:
            self.turns_resumed += 1

    def vagen_stats(self) -> dict[str, Any]:
        stats = {
            "step_exceptions": int(self.step_exceptions),
            "render_s": [float(x) for x in self.render_s],
            "turn_gen_s": [float(x) for x in self.turn_gen_s],
            "turn_max_tokens": [int(x) for x in self.turn_max_tokens],
            "turn_spans": [[int(s), int(e)] for s, e in self.turn_spans],
            "endoftext_stops": int(self.endoftext_stops),
            "length_stops": int(self.length_stops),
            "turns_resumed": int(self.turns_resumed),
            "num_preempted": -1 if self.num_preempted is None else int(self.num_preempted),
            "gen_turns": len(self.turn_gen_s),
            "env_turns": int(self.env_turns),
            "termination": str(self.termination),
            "render_url_refreshed": int(self.render_url_refreshed),
        }
        assert tuple(stats) == VAGEN_STATS_KEYS
        return stats

    def reward_extra_info(self) -> dict[str, Any]:
        # GL:339-352, the same keys and values
        info = {
            "traj_success": float(self.traj_success),
            "task_type": str(self.task_type),
            "scene_id": str(self.scene_id),
            "task_id": str(self.task_id),
            "initial_score": float(self.initial_score),
            "final_score": float(self.final_score),
            "score_improvement": float(self.final_score - self.initial_score),
            "n_primitive_steps": int(self.n_primitive_steps),
            "n_turns": int(self.env_turns),
            "invalid_action": self.invalid_turns / max(1, self.env_turns),
        }
        assert tuple(info) == REWARD_EXTRA_INFO_KEYS
        return info


class GymAgentLoopV1(AgentLoopBase):
    """Concat-mode multi-turn agent loop over a GymImageEnv, on Continuous Token (DESIGN_ASYNC section 2.5.3).

    Registered by vagen/q38/configs/agent_loop_v1.yaml under the rows' agent_name "gym_agent"; AgentLoopWorker builds
    one instance per episode (verl/experimental/agent_loop/agent_loop.py:710-722).
    """

    # env classes imported so far, shared by the episodes of a worker process
    _env_classes: dict[str, type] = {}
    # (row address, current address) pairs already logged by this process (refresh_render_url)
    _render_url_logged: set[tuple[str, str]] = set()
    # what the data rows must carry (MF8); None skips the check
    expected_max_turns: Optional[int] = contract.MAX_TURNS
    expected_turn_limit: Optional[int] = contract.RESPONSE_LENGTH_PER_TURN

    def __init__(self, *args, **kwargs):
        kwargs.pop("tools", None)  # AgentLoopWorker passes its tool list; this loop has no tools
        super().__init__(*args, **kwargs)
        self.prompt_length = int(self.rollout_config.prompt_length)
        self.response_length = int(self.rollout_config.response_length)
        limit_images = _select(self.rollout_config, "limit_images", None)
        self.limit_images = int(contract.LIMIT_IMAGES if limit_images is None else limit_images)

        override = _plain(_select(self.config, "vagen.parity_sampling_override", None))
        parity_mode = bool(_select(self.config, "vagen.parity_mode", False))
        if override and not parity_mode:
            raise ValueError(
                f"vagen.parity_sampling_override is only allowed with vagen.parity_mode: true (got {override!r})"
            )
        self.sampling_override: dict[str, Any] = dict(override or {})

        registry = dict(DEFAULT_ENV_REGISTRY)
        registry.update({str(k): str(v) for k, v in (_plain(_select(self.config, "env_registry", None)) or {}).items()})
        self.env_registry_paths = registry

        tok = self.tokenizer
        self._im_end_id = self._token_id("<|im_end|>")
        self._endoftext_id = self._token_id("<|endoftext|>")
        eos = getattr(tok, "eos_token_id", None)
        eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
        self._eos_ids = {t for t in (self._im_end_id, self._endoftext_id, *eos_ids) if t is not None}

    def _token_id(self, token: str) -> Optional[int]:
        tid = self.tokenizer.convert_tokens_to_ids(token)
        if tid is None or (self.tokenizer.unk_token_id is not None and tid == self.tokenizer.unk_token_id):
            return None
        return int(tid)

    # ------------------------------------------------------------------ env

    def _make_env(self, env_name: str, env_config: Any):
        """Build the row's env (GL:240-251: lazy import of the registered class)."""
        cls = self._env_classes.get(env_name)
        if cls is None:
            if env_name not in self.env_registry_paths:
                raise KeyError(f"Unknown env: {env_name}. Available: {list(self.env_registry_paths)}")
            module_path, class_name = self.env_registry_paths[env_name].rsplit(".", 1)
            cls = getattr(importlib.import_module(module_path), class_name)
            self._env_classes[env_name] = cls
        return cls(env_config=env_config)

    def _env_config(self, ep: _Episode, env_config: Any) -> Any:
        """The row's env config with the current render service address (refresh_render_url), counted in ep."""
        env_config, old = refresh_render_url(env_config)
        if old is not None:
            ep.render_url_refreshed = 1
            current = str(env_config[RENDER_URL_KEY])
            if (old, current) not in self._render_url_logged:
                self._render_url_logged.add((old, current))
                logger.warning(
                    "render service address: the data row carries %s=%s, using the current %s=%s (logged once per "
                    "address pair; every such episode counts vagen/env/render_url_refreshed)",
                    RENDER_URL_KEY, old, RENDER_URL_ENV, current,
                )
        return env_config

    async def _close_env(self, env) -> None:
        try:
            await env.close()
        except Exception:  # a close error must not hide the episode's own exception
            logger.error("env.close failed:\n%s", traceback.format_exc())

    # ------------------------------------------------------------------ run

    def _row_limits(self, kwargs: dict[str, Any]) -> tuple[Optional[int], int]:
        """max_turns and the per-turn token limit from the data row (GL:250, GL:274-277), checked against the
        configured values (MF8)."""
        max_turns = kwargs.get("max_turns", None)
        max_turns = None if max_turns is None else int(max_turns)
        if self.expected_max_turns is not None and max_turns != int(self.expected_max_turns):
            raise ValueError(f"data row max_turns={max_turns}, expected {self.expected_max_turns} (MF8)")
        per_turn = int(kwargs.get("response_length_per_turn") or self.response_length)
        if self.expected_turn_limit is not None and per_turn != int(self.expected_turn_limit):
            raise ValueError(
                f"data row response_length_per_turn={per_turn}, expected {self.expected_turn_limit} (MF8)"
            )
        per_turn = max(1, min(per_turn, self.response_length))
        return max_turns, per_turn

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        max_turns, per_turn_limit = self._row_limits(kwargs)
        if sampling_params.get("topk_log_probs"):
            raise ValueError("GymAgentLoopV1 does not carry sampler top-k heads (rollout.topk_log_probs must be 0)")
        sampling_params = {**sampling_params, **self.sampling_override}
        want_logprobs = bool(sampling_params.get("logprobs"))

        ep = _Episode(request_id=uuid4().hex, want_logprobs=want_logprobs)  # one request id an episode (sticky server)
        self.episode = ep  # the instance lives for one episode; kept for inspection (tests)
        env = self._make_env(kwargs["env_name"], self._env_config(ep, kwargs["config"]))
        try:
            await self._reset(env, ep, kwargs)
            while True:
                if not await self._generate_turn(ep, sampling_params, per_turn_limit):
                    break
                if not await self._env_turn(env, ep, max_turns, kwargs):
                    break
        finally:
            await self._close_env(env)
        return self._output(ep)

    async def _reset(self, env, ep: _Episode, kwargs: dict[str, Any]) -> None:
        """reset -> system prompt -> initial tokens (GL:253-293, GL:372-385 with Continuous Token)."""
        t0 = time.perf_counter()
        init_obs, info = await env.reset(seed=kwargs["seed"])
        ep.render_s.append(time.perf_counter() - t0)
        sys_obs = await env.system_prompt()
        if info:
            ep.task_type = info.get("task_type", "unknown")
            ep.scene_id = info.get("scene_id", "unknown")
            ep.initial_score = float(info.get("initial_potential_score", 0.0))
            ep.task_id = str(info.get("task_id") or "")

        if sys_obs:
            sys_imgs = _normalize_images(sys_obs.get("multi_modal_input", {}).get("<image>", []) or [])
            content = attach_images(convert_obs_to_content(sys_obs, **kwargs), sys_imgs)
            ep.messages.append({"role": "system", "content": content})
            ep.images.extend(sys_imgs)
        if init_obs:
            init_imgs = _normalize_images(init_obs.get("multi_modal_input", {}).get("<image>", []) or [])
            content = attach_images(convert_obs_to_content(init_obs, **kwargs), init_imgs)
            ep.messages.append({"role": "user", "content": content})
            ep.images.extend(init_imgs)

        self._assert_mm_supported(bool(ep.images))
        ep.prompt_ids = await self.ct_build_initial_tokens(ep.messages, images=ep.images or None)
        ep.initial_prompt_len = len(ep.prompt_ids)

    async def _generate_turn(self, ep: _Episode, sampling_params: dict[str, Any], per_turn_limit: int) -> bool:
        """One assistant turn. Returns False when the response budget is already used up (rule #1)."""
        remaining = self.response_length - len(ep.response_mask)
        if remaining <= 0:
            ep.termination = "response_budget"
            return False
        max_tokens = min(per_turn_limit, remaining)
        turn_params = {k: v for k, v in sampling_params.items() if k != "max_new_tokens"}
        turn_params["max_tokens"] = max_tokens

        t0 = time.perf_counter()
        output = await self.server_manager.generate(
            request_id=ep.request_id,
            prompt_ids=ep.prompt_ids,
            sampling_params=turn_params,
            image_data=ep.images or None,
        )
        ep.turn_gen_s.append(time.perf_counter() - t0)
        ep.turn_max_tokens.append(max_tokens)
        if output.stop_reason in ("aborted", "abort"):
            # only possible with async_training.partial_rollout false, which the contract refuses: a cut answer
            # must not reach the env (DESIGN_ASYNC section 3.1)
            raise RuntimeError(f"generation aborted without resume (request {ep.request_id})")
        if output.num_preempted is not None:
            ep.num_preempted = (ep.num_preempted or 0) + int(output.num_preempted)
        ep.add_versions(output.extra_fields or {})

        token_ids = list(output.token_ids)
        log_probs = list(output.log_probs) if output.log_probs else []
        if len(token_ids) > max_tokens:
            logger.warning("server returned %d tokens for max_tokens=%d; cut", len(token_ids), max_tokens)
            token_ids, log_probs = token_ids[:max_tokens], log_probs[:max_tokens]
        if token_ids and token_ids[-1] == self._endoftext_id:
            ep.endoftext_stops += 1  # Continuous Token inserts no "\n" after it
        if not token_ids or token_ids[-1] not in self._eos_ids:
            ep.length_stops += 1

        merge_result, mask, logprobs = await self.ct_merge_assistant_token(
            ep.prompt_ids,
            token_ids,
            ep.response_mask,
            ep.response_logprobs,
            assistant_logprobs=log_probs if ep.want_logprobs else None,
        )
        ep.prompt_ids = merge_result.token_ids
        ep.response_mask = mask
        if ep.want_logprobs:
            ep.response_logprobs = logprobs
        end = len(ep.response_mask)
        ep.turn_spans.append([end - merge_result.appended_token_count, end])

        # the env reads the answer without special tokens, as the 8B loop (GL:434-438); the assistant message only
        # keeps the message list in step with the tokens (Continuous Token renders appended messages alone)
        ep.last_answer_text = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(token_ids, skip_special_tokens=True)
        )
        ep.messages.append({"role": "assistant", "content": ep.last_answer_text})
        return True

    async def _env_turn(self, env, ep: _Episode, max_turns: Optional[int], kwargs: dict[str, Any]) -> bool:
        """Step the env with the last answer; returns False when the episode ends (GL:441-534)."""
        action_str = ep.last_answer_text or ""
        error = False
        t0 = time.perf_counter()
        try:
            obs, reward, done, info = await env.step(action_str)
        except Exception as exc:  # GL:451-459: the episode ends as done with reward 0; counted here (MF5)
            logger.error("Environment step failed in '%s' with action %r: %s", kwargs.get("env_name"), action_str, exc)
            logger.error("Environment traceback:\n%s", traceback.format_exc())
            obs, reward, done, info = {"obs_str": "Environment Error"}, 0.0, True, {"traj_success": False}
            ep.step_exceptions += 1
            error = True
        ep.render_s.append(time.perf_counter() - t0)

        # GL:461-479
        ep.env_rewards.append(float(reward))
        ep.traj_success = extract_success(info)
        ep.env_turns += 1
        ep.invalid_turns += int(turn_invalid(info))
        if info:
            ep.n_primitive_steps = int(info.get("env_step", ep.env_turns))
            if "final_score" in info:
                ep.final_score = float(info["final_score"])
            elif "current_potential_score" in info:
                ep.final_score = float(info["current_potential_score"])

        # Termination rule #3: env done or success (GL:480-482)
        if done or ep.traj_success:
            ep.termination = "env_error" if error else ("success" if ep.traj_success else "env_done")
            return False
        # Termination rule #2: env turn-limit (GL:484-486)
        if max_turns is not None and ep.env_turns >= max_turns:
            ep.termination = "max_turns"
            return False
        # Termination rule #1: response token-limit (GL:488-490)
        if len(ep.response_mask) >= self.response_length:
            ep.termination = "response_budget"
            return False

        # not terminal: the next observation as a user message (GL:492-534 with Continuous Token)
        new_images = _normalize_images(obs.get("multi_modal_input", {}).get("<image>", []) or [])
        user_msg = {"role": "user", "content": attach_images(convert_obs_to_content(obs, **kwargs), new_images)}
        self._assert_mm_supported(bool(new_images))
        previous = list(ep.messages)
        updated = previous + [user_msg]
        merge_result, mask, logprobs = await self.ct_merge_context_msg(
            previous, updated, ep.prompt_ids, ep.response_mask, ep.response_logprobs
        )
        if len(mask) >= self.response_length:
            # the observation does not fit: end on the assistant turn (tool_agent_loop.py:426-427)
            ep.termination = "obs_overflow"
            return False
        ep.messages = updated
        ep.prompt_ids = merge_result.token_ids
        ep.response_mask = mask
        if ep.want_logprobs:
            ep.response_logprobs = logprobs
        ep.images.extend(new_images)
        return True

    # ------------------------------------------------------------------ output

    def _output(self, ep: _Episode) -> AgentLoopOutput:
        p0 = ep.initial_prompt_len
        response_ids = ep.prompt_ids[p0:]
        if len(response_ids) != len(ep.response_mask):
            raise RuntimeError(
                f"response tokens ({len(response_ids)}) and response mask ({len(ep.response_mask)}) differ"
            )
        if len(response_ids) > self.response_length:
            raise RuntimeError(f"response of {len(response_ids)} tokens over rollout.response_length")
        if len(ep.images) > self.limit_images:
            raise ValueError(f"episode has {len(ep.images)} images, over rollout.limit_images={self.limit_images}")
        if ep.response_logprobs is not None and len(ep.response_logprobs) != len(ep.response_mask):
            raise RuntimeError("response logprobs and response mask differ in length")

        return AgentLoopOutput(
            prompt_ids=ep.prompt_ids[:p0],
            response_ids=response_ids,
            response_mask=ep.response_mask,
            response_logprobs=ep.response_logprobs,
            multi_modal_data={"images": ep.images} if ep.images else {},
            reward_score=float(sum(ep.env_rewards)) if ep.env_rewards else 0.0,
            num_turns=ep.env_turns,  # as GL:335
            metrics={
                "generate_sequences": float(sum(ep.turn_gen_s)),
                "tool_calls": float(sum(ep.render_s)),
                "num_preempted": -1 if ep.num_preempted is None else int(ep.num_preempted),
            },
            extra_fields={
                "min_global_steps": ep.min_global_steps,
                "max_global_steps": ep.max_global_steps,
                "reward_extra_info": ep.reward_extra_info(),
                "vagen_stats": ep.vagen_stats(),
            },
        )
