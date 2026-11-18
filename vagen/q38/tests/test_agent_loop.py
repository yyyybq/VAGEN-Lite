"""CPU tests of vagen.q38.gym_agent_loop_v1.GymAgentLoopV1 (docs/q38/DESIGN_ASYNC.md section 5.1, T6 and T7).

No GPU, no model weights, no real render service:
  - a fake LLM server (FakeServer) answers each turn with scripted tokens and logprobs;
  - a fake env (FakeEnv) gives one 256x256 noise frame per observation;
  - the real ActiveSpatialGymEnv test talks to a local fake HTTP render service (FakeRenderService) and reads the 8B
    main line's task rows and rollout dumps read-only.
The three processors (Qwen3.8-27B, Qwen3.5-4B, the 8B C0 SFT checkpoint of Qwen3-VL-8B) are loaded without weights;
a model whose folder is not readable is skipped.  The TransferQueue test starts a local Ray with num_cpus=2.

Run (login node, DESIGN_ASYNC 5 rules):
  cd /mnt/umm/users/yinbaiqiao/VAGEN-Lite-q38 && OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false RAYON_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=$PWD nice -n 19 /mnt/umm/users/yinbaiqiao/q38/envs/verl/bin/python -B -m pytest \
    vagen/q38/tests/test_agent_loop.py -q -x -p no:cacheprovider
"""

from __future__ import annotations

import asyncio
import functools
import http.server
import json
import os
import re
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image

from vagen.q38 import contract
from vagen.q38.gym_agent_loop_v1 import REWARD_EXTRA_INFO_KEYS, VAGEN_STATS_KEYS, GymAgentLoopV1, _Episode
from verl.workers.rollout.replica import TokenOutput

REPO = Path(__file__).resolve().parents[3]
AGENT_LOOP_YAML = REPO / "vagen" / "q38" / "configs" / "agent_loop_v1.yaml"
# the 8B main line V4 run (DESIGN_ASYNC R8), read only
R8 = Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_train_q3s3_20261027")

MODEL_PATHS = {
    "qwen38_27b": os.environ.get("Q38_TEST_MODEL_27B", "/mnt/umm/users/yinbaiqiao/q38/models/Qwen3.8-27B"),
    "qwen35_4b": os.environ.get(
        "Q38_TEST_MODEL_4B",
        "/mnt/umm/shared_model/huggingface/hub/models--Qwen--Qwen3.5-4B/snapshots/"
        "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a",
    ),
    # the 8B C0 SFT checkpoint the 8B main line started from (Y8:258), its own chat template
    "qwen3vl_8b": os.environ.get(
        "Q38_TEST_MODEL_8B",
        "/mnt/umm/users/yinbaiqiao/models_runs/chains/m_qwen3vl8b_formal_sft_20261026/lf_q3formal/model/checkpoint-1232",
    ),
}
MODEL_NAMES = tuple(MODEL_PATHS)
# the empty thinking block of the 27B generation prompt (docs/q38/TEMPLATE_CHECK.md item 4)
EMPTY_THINK_27B = [248068, 271, 248069, 271]
IMAGE_TOKENS_256PX = 64  # one 256x256 frame (TEMPLATE_CHECK item 5)

SAMPLING = dict(temperature=0.8, top_p=1.0, top_k=-1, repetition_penalty=1.0, logprobs=True, topk_log_probs=0)
CLEAN_ANSWERS = [
    "<action>turn_left|move_forward|</action>",
    "<action>move_forward|move_forward|turn_right|</action>",
    "<action>move_left|</action>",
]


# --------------------------------------------------------------------------------------------------- fixtures


@functools.lru_cache(maxsize=None)
def _load_model(name: str) -> SimpleNamespace:
    from transformers import AutoConfig

    from verl.utils.tokenizer.tokenizer import hf_processor, hf_tokenizer

    path = MODEL_PATHS[name]
    tokenizer = hf_tokenizer(path)
    processor = hf_processor(path)
    assert processor is not None, f"no multimodal processor for {path}"
    model_type = AutoConfig.from_pretrained(path).model_type
    return SimpleNamespace(name=name, path=path, tokenizer=tokenizer, processor=processor, model_type=model_type)


def load_model(name: str) -> SimpleNamespace:
    if not os.path.isfile(os.path.join(MODEL_PATHS[name], "tokenizer_config.json")):
        pytest.skip(f"{name}: {MODEL_PATHS[name]} not readable")
    return _load_model(name)


def default_model() -> SimpleNamespace:
    for name in MODEL_NAMES:
        if os.path.isfile(os.path.join(MODEL_PATHS[name], "tokenizer_config.json")):
            return _load_model(name)
    pytest.skip("no model folder readable")


def noise_image(seed: int, size: int = 256) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (size, size, 3), dtype=np.uint8), "RGB")


class FakeEnv:
    """A GymImageEnv stand-in: one 256x256 noise frame an observation, behaviour from its config."""

    closed: list = []

    def __init__(self, env_config):
        cfg = dict(env_config)
        self.done_at = cfg.get("done_at")  # the step that ends the episode (None: never)
        self.success = bool(cfg.get("success", True))
        self.raise_at = cfg.get("raise_at")  # the step that raises
        self.obs_words = int(cfg.get("obs_words", 0))  # filler words in each step observation
        self.invalid = bool(cfg.get("invalid", False))  # every answer unreadable
        self.t = 0
        self.seed = 0
        self.actions: list[str] = []

    async def system_prompt(self):
        # trailing whitespace on purpose: the 27B / 4B templates trim it, the 8B one keeps it
        return {"obs_str": "You are a navigation agent.\nAnswer with <action>a|b|</action>.\n"}

    async def reset(self, seed):
        self.seed, self.t = int(seed), 0
        obs = {
            "obs_str": "[Initial Observation]:\n<image>\nTask: reach the chair.\n",
            "multi_modal_input": {"<image>": [noise_image(1000 + self.seed)]},
        }
        info = {"task_type": "fake_type", "scene_id": "fake_scene", "task_id": f"fake_{seed}",
                "initial_potential_score": 0.25}
        return obs, info

    async def step(self, action_str):
        self.t += 1
        self.actions.append(action_str)
        if self.raise_at is not None and self.t == int(self.raise_at):
            raise RuntimeError("fake env failure")
        done = self.done_at is not None and self.t >= int(self.done_at)
        success = done and self.success
        info = {
            "success": success,
            "env_step": 2 * self.t,
            "current_potential_score": 0.25 + 0.1 * self.t,
            "metrics": {"turn_metrics": {"action_is_valid": not self.invalid}},
        }
        obs = {
            "obs_str": f"[Observation {self.t}]:\n<image>\nFeedback: step {self.t} done.{' x' * self.obs_words}\n",
            "multi_modal_input": {"<image>": [noise_image(2000 + 100 * self.seed + self.t)]},
        }
        return obs, (5.0 if success else -0.1), done, info

    async def close(self):
        FakeEnv.closed.append(self)


class FakeServer:
    """LLMServerClient stand-in: scripted answers, distinct logprobs, weight versions per call."""

    def __init__(self, tokenizer, answers=None, ends=None, fill=False, raise_at=None, versions=None, extra=None):
        self.tokenizer = tokenizer
        self.answers = answers or CLEAN_ANSWERS
        self.ends = ends  # per turn: "im_end" (default), "endoftext" or None (no stop token)
        self.fill = fill  # answer exactly max_tokens filler tokens, no stop token
        self.raise_at = raise_at
        self.versions = versions  # per turn (min, max)
        self.extra = extra  # fixed extra_fields instead of versions (sync clients)
        self.calls: list[dict] = []
        self.generated: list[list[int]] = []
        self.logprobs: list[list[float]] = []
        self._end_ids = {
            "im_end": [tokenizer.convert_tokens_to_ids("<|im_end|>")],
            "endoftext": [tokenizer.convert_tokens_to_ids("<|endoftext|>")],
            None: [],
        }
        self._filler = tokenizer.encode("a", add_special_tokens=False)[0]

    async def generate(self, request_id, *, prompt_ids, sampling_params, image_data=None, **kwargs):
        i = len(self.calls)
        self.calls.append({"request_id": request_id, "prompt_ids": list(prompt_ids),
                           "sampling_params": dict(sampling_params), "n_images": len(image_data or [])})
        if self.raise_at is not None and i + 1 == self.raise_at:
            raise RuntimeError("fake generate failure")
        max_tokens = sampling_params["max_tokens"]
        if self.fill:
            ids = [self._filler] * max_tokens
        else:
            end = self.ends[i % len(self.ends)] if self.ends else "im_end"
            ids = self.tokenizer.encode(self.answers[i % len(self.answers)], add_special_tokens=False)
            ids = (ids + self._end_ids[end])[:max_tokens]
        lps = [-0.001 * (i + 1) - 1e-5 * k for k in range(len(ids))]
        self.generated.append(ids)
        self.logprobs.append(lps)
        if self.extra is not None:
            extra = dict(self.extra)
        else:
            lo, hi = self.versions[i % len(self.versions)] if self.versions else (7, 7)
            extra = {"global_steps": hi, "min_global_steps": lo, "max_global_steps": hi}
        return TokenOutput(token_ids=ids, log_probs=lps if sampling_params.get("logprobs") else None,
                           stop_reason="completed", num_preempted=0, extra_fields=extra)


@pytest.fixture(autouse=True)
def _fake_env_registered(monkeypatch):
    monkeypatch.setitem(GymAgentLoopV1._env_classes, "FakeEnv", FakeEnv)
    FakeEnv.closed = []
    yield


def make_config(rollout=None, vagen=None, data=None):
    cfg = {
        "actor_rollout_ref": {"rollout": {"prompt_length": 4096, "response_length": 4096, "limit_images": 13,
                                          "calculate_log_probs": True}},
        "data": {"apply_chat_template_kwargs": {"enable_thinking": False}},
        "vagen": {"parity_mode": False, "parity_sampling_override": None},
    }
    cfg["actor_rollout_ref"]["rollout"].update(rollout or {})
    cfg["vagen"].update(vagen or {})
    cfg["data"].update(data or {})
    return OmegaConf.create(cfg)


def make_loop(model, server, cfg=None):
    """Build the loop the way AgentLoopWorker does: from the registry yaml through hydra (AL:558-564, 710-721)."""
    import hydra

    from verl.experimental.agent_loop.agent_loop import DictConfigWrap, ToolListWrap

    cfg = cfg if cfg is not None else make_config()
    entry = next(e for e in OmegaConf.load(AGENT_LOOP_YAML) if e.name == "gym_agent")
    return hydra.utils.instantiate(
        config=entry,
        trainer_config=DictConfigWrap(config=cfg),
        server_manager=server,
        tokenizer=model.tokenizer,
        processor=model.processor,
        hf_model_type=model.model_type,
        dataset_cls=None,
        data_config=DictConfigWrap(cfg.data),
        tools=ToolListWrap([]),
    )


def row(seed=0, max_turns=12, response_length_per_turn=384, **env_cfg):
    """The kwargs AgentLoopWorkerTQ hands to run() for one session (a dataset row + uid/session/global_steps)."""
    return dict(env_name="FakeEnv", seed=seed, config=dict(env_cfg), max_turns=max_turns,
                response_length_per_turn=response_length_per_turn, data_source="fake", raw_prompt=[],
                index=seed, uid=f"uid{seed}", session_id=0, global_steps=7)


def run_episode(model, server, cfg=None, sampling=None, **row_kw):
    async def go():
        loop = make_loop(model, server, cfg)
        out = await loop.run(dict(sampling or SAMPLING), **row(**row_kw))
        return loop, out

    return asyncio.run(go())


THINK_TEXT = "<think>\n\n</think>\n\n"


def template_drops_history_think(model) -> bool:
    """Whether the chat template renders the assistant turns before the last user turn without their (empty) thinking
    block: Qwen3.5-4B's template has no preserve_thinking switch and does; Qwen3.8-27B's keeps them by default
    (TEMPLATE_CHECK item 1)."""
    return model.model_type == "qwen3_5" and "preserve_thinking" not in (model.processor.chat_template or "")


def keep_history_think(text: str) -> str:
    """The whole-conversation text with the empty thinking block put back into every assistant turn."""
    return re.sub(r"<\|im_start\|>assistant\n(?!<think>)", "<|im_start|>assistant\n" + THINK_TEXT, text)


def full_render(model, messages, patch=None):
    """Token ids of the whole conversation rendered at once by the model's chat template and processor (``patch``
    edits the rendered text before it is tokenized)."""
    from verl.utils.tokenizer.chat_template import apply_chat_template
    from verl.utils.tokenizer.tokenizer import build_multimodal_processor_inputs, normalize_token_ids

    text = apply_chat_template(model.processor, messages, tokenize=False, add_generation_prompt=False,
                               enable_thinking=False)
    if patch is not None:
        text = patch(text)
    images = [b["image"] for m in messages if isinstance(m["content"], list) for b in m["content"]
              if b.get("type") == "image"]
    out = build_multimodal_processor_inputs(model.processor, text=[text], images=images or None, videos=None,
                                            audio=None, mm_processor_kwargs=None)
    return normalize_token_ids(out["input_ids"])


def image_token_id(model) -> int:
    from verl.utils.tokenizer import get_processor_token_id

    return get_processor_token_id(model.processor, "image")


def think_block_ids(model) -> list[int]:
    tok = model.tokenizer
    nn = tok.encode("\n\n", add_special_tokens=False)
    return [tok.convert_tokens_to_ids("<think>"), *nn, tok.convert_tokens_to_ids("</think>"), *nn]


def assert_json_style(value, path="extra_fields"):
    """str keys, lists, Python scalars only (a tuple would come back from TransferQueue as a list)."""
    if isinstance(value, dict):
        for k, v in value.items():
            assert isinstance(k, str), f"{path}: key {k!r}"
            assert_json_style(v, f"{path}.{k}")
    elif isinstance(value, list):
        for i, v in enumerate(value):
            assert_json_style(v, f"{path}[{i}]")
    else:
        assert value is None or type(value) in (str, int, float, bool), f"{path}: {type(value).__name__}"


def tq_fields(model, cfg, output, kwargs):
    """What AgentLoopWorkerTQ._agent_loop_postprocess stores for one output: a copy of
    verl/verl/trainer/ppo/v1/agent_loop_tq.py:196-235 (verl tree 4bfa94d), with AgentLoopWorker's own multimodal and
    position-id helpers called on a stand-in worker."""
    from verl.experimental.agent_loop.agent_loop import AgentLoopWorker

    worker = SimpleNamespace(processor=model.processor, tokenizer=model.tokenizer, config=cfg, mm_processor_kwargs={})
    worker._get_mm_processor_kwargs = lambda audio_data=None: {}
    prompts = torch.tensor(output.prompt_ids, dtype=torch.int64)
    responses = torch.tensor(output.response_ids, dtype=torch.int64)
    input_ids = torch.cat([prompts, responses], dim=0)
    attention_mask = torch.ones_like(input_ids, dtype=torch.int64)
    multi_modal_inputs = AgentLoopWorker._compute_multi_modal_inputs(worker, output, input_ids)
    position_ids = AgentLoopWorker._compute_position_ids(
        worker, input_ids.unsqueeze(0), attention_mask.unsqueeze(0), multi_modal_inputs
    ).squeeze(0)
    field = output.as_dict()
    field.update(kwargs)
    field.pop("multi_modal_data", None)
    field["loss_mask"] = field["response_mask"]
    field["input_ids"] = input_ids
    field["position_ids"] = position_ids
    field["multi_modal_inputs"] = multi_modal_inputs
    prompt_len, response_len = field["prompts"].size(0), field["responses"].size(0)
    tag = {
        "status": "success",
        "prompt_len": prompt_len,
        "response_len": response_len,
        "seq_len": prompt_len + response_len,
        "global_steps": kwargs["global_steps"],
        "min_global_steps": field["extra_fields"].get("min_global_steps"),
        "max_global_steps": field["extra_fields"].get("max_global_steps"),
    }
    return field, tag


# --------------------------------------------------------------------------------------------------- T7


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_t7_three_turns_match_whole_conversation(name):
    """3 turns: the Continuous Token stream = the whole conversation rendered by the template (only the final "\\n"
    after the last <|im_end|> is missing), masks / logprobs / images / generation prompt / max_tokens / versions."""
    model = load_model(name)
    tok = model.tokenizer
    server = FakeServer(tok, versions=[(3, 3), (3, 4), (4, 4)])
    loop, out = run_episode(model, server, done_at=3)
    ep = loop.episode
    stream = out.prompt_ids + out.response_ids

    # the whole conversation, rendered at once
    full = full_render(model, ep.messages)
    newline = tok.encode("\n", add_special_tokens=False)
    if template_drops_history_think(model):
        # Qwen3.5-4B: every turn was generated after the generation prompt's empty thinking block, which the stream
        # keeps; the template drops it from the turns before the last user turn when it re-renders the history
        assert full != stream + newline
        full = full_render(model, ep.messages, patch=keep_history_think)
    assert full == stream + newline, f"{name}: CT stream differs from the whole-conversation render"
    # every turn's prompt plus its answer is a prefix of the whole conversation
    for call, gen in zip(server.calls, server.generated, strict=True):
        p = call["prompt_ids"]
        assert full[: len(p) + len(gen)] == p + gen

    # mask 1 exactly on the generated turns, which hold exactly the generated tokens
    stats = out.extra_fields["vagen_stats"]
    spans = stats["turn_spans"]
    assert len(spans) == 3 and spans[0][0] == 0
    ones = [i for i, m in enumerate(out.response_mask) if m == 1]
    assert ones == [i for s, e in spans for i in range(s, e)]
    for (s, e), gen in zip(spans, server.generated, strict=True):
        assert out.response_ids[s:e] == gen
    # logprobs: the server's on the generated tokens, 0 on the context tokens
    lps = out.response_logprobs
    assert len(lps) == len(out.response_mask)
    for (s, e), want in zip(spans, server.logprobs, strict=True):
        assert lps[s:e] == want
    assert all(lp == 0.0 for i, lp in enumerate(lps) if out.response_mask[i] == 0)
    # the Qwen boundary: a "\n" (mask 0) after each generated <|im_end|> that is followed by context
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    for s, e in spans[:-1]:
        assert out.response_ids[e - 1] == im_end and out.response_ids[e] == newline[0]
        assert out.response_mask[e] == 0
    assert out.response_ids[-1] == im_end  # the episode ends on the assistant turn

    # images: the first frame and one per non-final step; 64 placeholders each
    images = out.multi_modal_data["images"]
    assert len(images) == 3
    assert [c["n_images"] for c in server.calls] == [1, 2, 3]
    assert stream.count(image_token_id(model)) == IMAGE_TOKENS_256PX * len(images)

    # generation prompt: the 4-token empty thinking block for the Qwen3.5 family, none for Qwen3-VL
    if model.model_type == "qwen3_5":
        think = think_block_ids(model)
        if name == "qwen38_27b":
            assert think == EMPTY_THINK_27B
        for call in server.calls:
            assert call["prompt_ids"][-4:] == think
    else:
        assert tok.convert_tokens_to_ids("<think>") not in stream

    # per-turn limits, sampling params and one request id an episode
    p0 = len(out.prompt_ids)
    for call in server.calls:
        before = len(call["prompt_ids"]) - p0
        sp = call["sampling_params"]
        assert sp["max_tokens"] == min(384, 4096 - before)
        assert sp["temperature"] == 0.8 and sp["top_p"] == 1.0 and sp["logprobs"] is True
        assert "max_new_tokens" not in sp
    assert len({c["request_id"] for c in server.calls}) == 1

    # the env got the answers decoded without special tokens
    env = FakeEnv.closed[-1]
    assert env.actions == CLEAN_ANSWERS

    # versions, reward, 8B scalars, counters
    ef = out.extra_fields
    assert (ef["min_global_steps"], ef["max_global_steps"]) == (3, 4)
    assert out.reward_score == pytest.approx(-0.1 - 0.1 + 5.0)
    rei = ef["reward_extra_info"]
    assert tuple(rei) == REWARD_EXTRA_INFO_KEYS
    assert rei["traj_success"] == 1.0 and rei["n_turns"] == 3 and rei["invalid_action"] == 0.0
    assert rei["task_id"] == "fake_0" and rei["n_primitive_steps"] == 6
    assert rei["initial_score"] == 0.25 and rei["final_score"] == pytest.approx(0.55)
    assert rei["score_improvement"] == pytest.approx(0.30)
    assert tuple(stats) == VAGEN_STATS_KEYS
    assert stats["termination"] == "success" and stats["step_exceptions"] == 0
    assert stats["gen_turns"] == 3 and stats["env_turns"] == 3 and len(stats["render_s"]) == 4
    assert stats["turns_resumed"] == 1 and stats["endoftext_stops"] == 0 and stats["length_stops"] == 0
    assert stats["turn_max_tokens"] == [c["sampling_params"]["max_tokens"] for c in server.calls]
    assert set(ef) == {"min_global_steps", "max_global_steps", "reward_extra_info", "vagen_stats"}
    assert_json_style(ef)
    assert out.num_turns == 3

    # the trajectory as AgentLoopWorkerTQ stores it: position ids (4, L), one 16x16 grid per frame
    cfg = make_config()
    field, tag = tq_fields(model, cfg, out, row(done_at=3))
    assert tuple(field["position_ids"].shape) == (4, len(stream))
    grid = field["multi_modal_inputs"]["image_grid_thw"].tolist()
    assert grid == [[1, 16, 16]] * len(images)
    assert field["rm_scores"].sum().item() == pytest.approx(out.reward_score)
    assert field["rollout_log_probs"].shape == field["response_mask"].shape
    assert (tag["min_global_steps"], tag["max_global_steps"]) == (3, 4)


@pytest.mark.parametrize("name", MODEL_NAMES)
def test_t7_answer_whitespace_is_the_only_trim_difference(name):
    """An answer with leading / trailing whitespace: the template (27B, 4B) trims it in the history, the CT stream keeps
    the generated tokens; trimming the answers in the stream gives the whole-conversation render (8B: no trim)."""
    model = load_model(name)
    tok = model.tokenizer
    answers = [" <action>turn_left|</action>\n", "<action>move_forward|</action>  "]
    server = FakeServer(tok, answers=answers)
    loop, out = run_episode(model, server, done_at=2)
    stream = out.prompt_ids + out.response_ids
    patch = keep_history_think if template_drops_history_think(model) else None
    full_text = tok.decode(full_render(model, loop.episode.messages, patch=patch), skip_special_tokens=False)

    p0 = len(out.prompt_ids)
    pieces, cursor, im_end = [], 0, "<|im_end|>"
    for s, e in out.extra_fields["vagen_stats"]["turn_spans"]:
        s, e = s + p0, e + p0
        pieces.append(tok.decode(stream[cursor:s], skip_special_tokens=False))
        seg = tok.decode(stream[s:e], skip_special_tokens=False)
        assert seg.endswith(im_end)
        pieces.append(seg[: -len(im_end)].strip() + im_end)
        cursor = e
    pieces.append(tok.decode(stream[cursor:], skip_special_tokens=False))
    trimmed = "".join(pieces) + "\n"
    exact = tok.decode(stream, skip_special_tokens=False) + "\n"
    if model.model_type == "qwen3_5":
        assert trimmed == full_text and exact != full_text
    else:
        assert exact == full_text
    # the env still got the answers as generated (8B: GL:434-436)
    assert FakeEnv.closed[-1].actions == [tok.decode(g, skip_special_tokens=True) for g in server.generated]


def test_t7_twelve_turn_cap():
    """An env that never reads an answer: the loop stops at the 12th turn (GL:484-486), 12 images <= 13."""
    model = default_model()
    server = FakeServer(model.tokenizer)
    _, out = run_episode(model, server, invalid=True)
    stats, rei = out.extra_fields["vagen_stats"], out.extra_fields["reward_extra_info"]
    assert len(server.calls) == 12 and stats["env_turns"] == 12 and stats["termination"] == "max_turns"
    assert len(out.multi_modal_data["images"]) == 12 <= contract.LIMIT_IMAGES
    assert rei["n_turns"] == 12 and rei["invalid_action"] == 1.0 and rei["traj_success"] == 0.0
    assert out.reward_score == pytest.approx(-1.2)
    assert out.response_ids[-1] == model.tokenizer.convert_tokens_to_ids("<|im_end|>")


def test_t7_last_turn_gets_the_remaining_budget():
    """Answers that always run to max_tokens: every turn asks for 384 until less is left, the last one for exactly the
    remaining budget, and the response ends at 4096 tokens (R9 (2), MF11)."""
    model = default_model()
    server = FakeServer(model.tokenizer, fill=True)
    _, out = run_episode(model, server, invalid=True)
    stats = out.extra_fields["vagen_stats"]
    assert stats["termination"] == "response_budget"
    assert len(out.response_ids) == len(out.response_mask) == 4096
    p0 = len(out.prompt_ids)
    budgets = [4096 - (len(c["prompt_ids"]) - p0) for c in server.calls]
    asked = [c["sampling_params"]["max_tokens"] for c in server.calls]
    assert asked[:-1] == [384] * (len(asked) - 1)
    assert asked[-1] == budgets[-1] < 384
    assert stats["length_stops"] == stats["gen_turns"] == len(server.calls) < 12
    assert out.extra_fields["vagen_stats"]["turn_spans"][-1][1] == 4096


def test_t7_observation_that_does_not_fit_ends_on_the_assistant_turn():
    """A next observation larger than the budget left is not appended (no cut image placeholders)."""
    model = default_model()
    server = FakeServer(model.tokenizer)
    loop, out = run_episode(model, server, invalid=True, obs_words=2500)
    stats = out.extra_fields["vagen_stats"]
    assert stats["termination"] == "obs_overflow"
    assert len(out.response_ids) < 4096
    assert out.response_ids[-1] == model.tokenizer.convert_tokens_to_ids("<|im_end|>")
    n_images = len(out.multi_modal_data["images"])
    stream = out.prompt_ids + out.response_ids
    assert stream.count(image_token_id(model)) == IMAGE_TOKENS_256PX * n_images
    assert n_images == stats["gen_turns"]  # the first frame plus every appended observation
    assert loop.episode.messages[-1]["role"] == "assistant"


def test_t7_env_step_exception_ends_the_episode_as_done():
    """env.step raises: done, reward 0 for the turn, counted (GL:451-459, MF5); the env is closed."""
    model = default_model()
    server = FakeServer(model.tokenizer)
    _, out = run_episode(model, server, raise_at=2)
    stats, rei = out.extra_fields["vagen_stats"], out.extra_fields["reward_extra_info"]
    assert stats["step_exceptions"] == 1 and stats["termination"] == "env_error"
    assert stats["env_turns"] == 2 and len(server.calls) == 2
    assert out.reward_score == pytest.approx(-0.1 + 0.0)
    assert rei["traj_success"] == 0.0 and rei["n_turns"] == 2
    assert len(FakeEnv.closed) == 1


def test_t7_generate_exception_propagates_and_closes_the_env():
    """A generate error is not caught (the group becomes a failure for the sampler's failure_policy)."""
    model = default_model()
    server = FakeServer(model.tokenizer, raise_at=2)
    with pytest.raises(RuntimeError, match="fake generate failure"):
        run_episode(model, server, done_at=5)
    assert len(FakeEnv.closed) == 1 and FakeEnv.closed[0].t == 1


def test_t7_endoftext_stop_is_counted_and_gets_no_newline():
    model = default_model()
    tok = model.tokenizer
    server = FakeServer(tok, ends=["endoftext", "im_end"])
    _, out = run_episode(model, server, done_at=2)
    stats = out.extra_fields["vagen_stats"]
    assert stats["endoftext_stops"] == 1 and stats["length_stops"] == 0
    (s0, e0), _ = stats["turn_spans"]
    assert out.response_ids[e0 - 1] == tok.convert_tokens_to_ids("<|endoftext|>")
    assert out.response_ids[e0] == tok.convert_tokens_to_ids("<|im_start|>")  # no "\n" inserted


def test_t7_versions_from_sync_clients_and_missing_versions():
    model = default_model()
    _, out = run_episode(model, FakeServer(model.tokenizer, extra={"global_steps": 5}), done_at=2)
    ef = out.extra_fields
    assert (ef["min_global_steps"], ef["max_global_steps"]) == (5, 5)
    assert ef["vagen_stats"]["turns_resumed"] == 0
    _, out = run_episode(model, FakeServer(model.tokenizer, extra={}), done_at=2)
    assert (out.extra_fields["min_global_steps"], out.extra_fields["max_global_steps"]) == (None, None)


def test_t7_row_limits_are_checked():
    model = default_model()
    with pytest.raises(ValueError, match="max_turns=10"):
        run_episode(model, FakeServer(model.tokenizer), max_turns=10, done_at=1)
    with pytest.raises(ValueError, match="response_length_per_turn=256"):
        run_episode(model, FakeServer(model.tokenizer), response_length_per_turn=256, done_at=1)
    assert len(FakeEnv.closed) == 0  # refused before any env was built


def test_t7_parity_sampling_override():
    from hydra.errors import InstantiationException

    model = default_model()
    override = dict(contract.PARITY_SAMPLING)
    # hydra (as AgentLoopWorker calls it) wraps the constructor's ValueError
    with pytest.raises((ValueError, InstantiationException), match="parity_mode"):
        run_episode(model, FakeServer(model.tokenizer), cfg=make_config(vagen={"parity_sampling_override": override}),
                    done_at=1)
    server = FakeServer(model.tokenizer)
    cfg = make_config(vagen={"parity_mode": True, "parity_sampling_override": override})
    run_episode(model, server, cfg=cfg, done_at=2)
    for call in server.calls:
        assert call["sampling_params"]["temperature"] == 0.8 and call["sampling_params"]["top_p"] == 0.92


def test_t7_limit_images_and_logprobs_off():
    model = default_model()
    with pytest.raises(ValueError, match="limit_images=2"):
        run_episode(model, FakeServer(model.tokenizer), cfg=make_config(rollout={"limit_images": 2}), done_at=3)
    _, out = run_episode(model, FakeServer(model.tokenizer), sampling={**SAMPLING, "logprobs": False}, done_at=2)
    assert out.response_logprobs is None and "rollout_log_probs" not in out.as_dict()
    with pytest.raises(ValueError, match="topk"):
        run_episode(model, FakeServer(model.tokenizer), sampling={**SAMPLING, "topk_log_probs": 4}, done_at=1)


# ----------------------------------------------------------------- T7: the render service address after a resume

# S3 phase b (2026-11-18): the row re-issued from global_step_7's TransferQueue snapshot carried phase a's address
OLD_RENDER_URL = "http://10.119.28.126:8915/render"
NEW_RENDER_URL = "http://10.119.29.141:8915/render"


class RecordingEnv(FakeEnv):
    """FakeEnv that keeps the env config it was built with."""

    configs: list = []

    def __init__(self, env_config):
        RecordingEnv.configs.append(env_config)
        super().__init__(env_config)


def test_refresh_render_url():
    from vagen.q38.gym_agent_loop_v1 import refresh_render_url

    row_cfg = {"client_url": OLD_RENDER_URL, "render_backend": "http", "nested": {"a": 1}}
    new, old = refresh_render_url(row_cfg, {"R1_RENDER_URL": NEW_RENDER_URL})
    assert new == {**row_cfg, "client_url": NEW_RENDER_URL} and old == OLD_RENDER_URL
    assert row_cfg["client_url"] == OLD_RENDER_URL  # the row's own dict is not edited
    # unchanged (the same object): variable unset or empty, address already current, no client_url (another env)
    for environ in ({}, {"R1_RENDER_URL": ""}, {"R1_RENDER_URL": OLD_RENDER_URL}):
        same, old = refresh_render_url(row_cfg, environ)
        assert same is row_cfg and old is None
    other = {"done_at": 2}
    assert refresh_render_url(other, {"R1_RENDER_URL": NEW_RENDER_URL}) == (other, None)
    # an unresolved interpolation (a row that kept the yaml's text) gets the address as well
    raw = {"client_url": "${oc.env:R1_RENDER_URL}"}
    assert refresh_render_url(raw, {"R1_RENDER_URL": NEW_RENDER_URL})[0]["client_url"] == NEW_RENDER_URL


def _run_rows(model, rows):
    async def go():
        outs = []
        for kwargs in rows:
            loop = make_loop(model, FakeServer(model.tokenizer))
            outs.append(await loop.run(dict(SAMPLING), **kwargs))
        return outs

    return asyncio.run(go())


def test_t7_env_gets_the_current_render_url(monkeypatch, caplog):
    """A row carrying the old address (in-flight prompt restored after a resume): the env is built with the current
    process's R1_RENDER_URL, the episode counts render_url_refreshed, the row is not edited, one warning per pair."""
    import logging

    model = default_model()
    monkeypatch.setitem(GymAgentLoopV1._env_classes, "FakeEnv", RecordingEnv)
    monkeypatch.setattr(GymAgentLoopV1, "_render_url_logged", set())
    RecordingEnv.configs = []
    monkeypatch.setenv("R1_RENDER_URL", NEW_RENDER_URL)
    rows = [row(seed=s, done_at=2, client_url=OLD_RENDER_URL) for s in (0, 1)]
    rows.append(row(seed=2, done_at=2, client_url=NEW_RENDER_URL))
    with caplog.at_level(logging.WARNING):
        outs = _run_rows(model, rows)
    assert [c["client_url"] for c in RecordingEnv.configs] == [NEW_RENDER_URL] * 3
    assert [o.extra_fields["vagen_stats"]["render_url_refreshed"] for o in outs] == [1, 1, 0]
    assert [r["config"]["client_url"] for r in rows] == [OLD_RENDER_URL, OLD_RENDER_URL, NEW_RENDER_URL]
    warned = [r for r in caplog.records if "render service address" in r.getMessage()]
    assert len(warned) == 1 and OLD_RENDER_URL in warned[0].getMessage() and NEW_RENDER_URL in warned[0].getMessage()
    assert all(o.extra_fields["vagen_stats"]["termination"] == "success" for o in outs)


def test_t7_render_url_kept_without_the_variable(monkeypatch):
    """R1_RENDER_URL unset: the row's address is used as before, nothing counted."""
    model = default_model()
    monkeypatch.setitem(GymAgentLoopV1._env_classes, "FakeEnv", RecordingEnv)
    RecordingEnv.configs = []
    monkeypatch.delenv("R1_RENDER_URL", raising=False)
    kwargs = row(seed=0, done_at=2, client_url=OLD_RENDER_URL)
    (out,) = _run_rows(model, [kwargs])
    assert RecordingEnv.configs[0] is kwargs["config"] and RecordingEnv.configs[0]["client_url"] == OLD_RENDER_URL
    assert out.extra_fields["vagen_stats"]["render_url_refreshed"] == 0


# ----------------------------------------------------------------- T7: the real env against the 8B rollout dumps


class _RenderHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802 (http.server API)
        from vagen.envs.active_spatial.render.service_http.multipart import decode_multipart, encode_multipart

        body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        meta, _ = decode_multipart(self.headers.get("Content-Type", ""), body)
        self.server.requests.append(meta)
        images = [noise_image(7 * len(self.server.requests) + i) for i in range(len(meta.get("tasks") or []))]
        boundary, out = encode_multipart({"ok": True}, images)
        self.send_response(200)
        self.send_header("Content-Type", f'multipart/form-data; boundary="{boundary}"')
        self.send_header("Content-Length", str(len(out)))
        self.end_headers()
        self.wfile.write(out)

    def log_message(self, *args):
        pass


@pytest.fixture
def fake_render(monkeypatch):
    """A local stand-in of the InteriorGS HTTP render service (POST /render, multipart in and out)."""
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _RenderHandler)
    server.requests = []
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    for var in ("http_proxy", "HTTP_PROXY", "https_proxy", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    url = f"http://127.0.0.1:{server.server_address[1]}"
    monkeypatch.setenv("R1_RENDER_URL", url)
    yield SimpleNamespace(url=url, server=server)
    server.shutdown()
    server.server_close()


def _trim_messages(text: str) -> str:
    """Strip each ChatML message body (what the 27B template's |trim does)."""
    return re.sub(r"(<\|im_start\|>[a-z]+\n)(.*?)(<\|im_end\|>)",
                  lambda m: m.group(1) + m.group(2).strip() + m.group(3), text, flags=re.S)


def test_t7_first_prompts_match_the_8b_rollout_dump(fake_render):
    """20 tasks of the 8B main line (rl_run_qw3v4s3 steps 1-2): the real ActiveSpatialGymEnv (fake render service)
    through GymAgentLoopV1's reset gives, with the 8B C0 processor, exactly the dumped 'input' text (the first-turn
    prompt, image placeholders shown as <image>); the 27B processor gives the same text up to the per-message trim
    and the empty thinking block (TEMPLATE_CHECK items 2-4)."""
    run_dir, spec_dir = R8 / "rl_run_qw3v4s3", R8 / "rl_qw3v4s3"
    if not (run_dir / "rollout_data" / "1.jsonl").is_file():
        pytest.skip(f"{run_dir} not readable")
    m8 = load_model("qwen3vl_8b")
    from vagen.envs.active_spatial.active_spatial_env import ActiveSpatialGymEnv
    from vagen.utils.image_token_utils import replace_image_tokens_for_logging

    rows, seen = [], set()
    for step in (1, 2):
        with open(run_dir / "rollout_data" / f"{step}.jsonl") as f:
            for line in f:
                d = json.loads(line)
                if d["task_id"] not in seen:
                    seen.add(d["task_id"])
                    rows.append(d)
    rows = rows[:20]
    assert len(rows) == 20
    want = {r["task_id"] for r in rows}
    line_of = {}
    with open(spec_dir / "train.jsonl") as f:
        for i, line in enumerate(f):
            tid = json.loads(line).get("task_id")
            if tid in want:
                line_of[tid] = i
    assert set(line_of) == want

    spec = OmegaConf.load(spec_dir / "train_env.yaml").envs[0]
    env_cfg = OmegaConf.to_container(spec.config, resolve=True)
    assert env_cfg["client_url"] == fake_render.url
    kwargs = dict(env_name=spec.name, config=env_cfg, max_turns=int(spec.max_turns),
                  response_length_per_turn=int(spec.response_length_per_turn), data_source=spec.data_source)
    m27 = None
    if os.path.isfile(os.path.join(MODEL_PATHS["qwen38_27b"], "tokenizer_config.json")):
        m27 = _load_model("qwen38_27b")

    async def go():
        loop8 = make_loop(m8, server=None)
        loop27 = make_loop(m27, server=None) if m27 is not None else None
        env = ActiveSpatialGymEnv(env_config=env_cfg)
        results = []
        try:
            for r in rows:
                ep = _Episode("t7", True)
                await loop8._reset(env, ep, {**kwargs, "seed": line_of[r["task_id"]]})
                text8 = replace_image_tokens_for_logging(
                    m8.tokenizer.decode(ep.prompt_ids, skip_special_tokens=False), processor=m8.processor)
                text27 = None
                if loop27 is not None:
                    ids27 = await loop27.ct_build_initial_tokens(ep.messages, images=ep.images)
                    text27 = replace_image_tokens_for_logging(
                        m27.tokenizer.decode(ids27, skip_special_tokens=False), processor=m27.processor)
                results.append((r, ep, text8, text27))
        finally:
            await env.close()
        return results

    results = asyncio.run(go())
    assert len(fake_render.server.requests) == 20  # one frame a reset, through HTTP
    for r, ep, text8, text27 in results:
        assert ep.task_id == r["task_id"]
        assert len(ep.images) == 1
        assert text8 == r["input"], f"task {r['task_id']}: first prompt differs from the 8B dump"
        if text27 is not None:
            assert text27.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
            assert text27.replace("<think>\n\n</think>\n\n", "") == _trim_messages(r["input"])


# --------------------------------------------------------------------------------------------------- T6


# The real-TransferQueue tests (T6 round trip and tag merge) need a local Ray.  They are skipped unless
# Q38_TEST_REAL_TQ=1: on the fujian login node (2026-11-18) four local starts (num_cpus=2, no dashboard, 200 MB object
# store) all failed after 68-82 s with "The current node timed out during startup" ("GCS cannot find the node" /
# "RPC error: Deadline Exceeded"), and a failed start leaves raylet / GCS processes on the shared CPU quota.  They are
# run in the Qwen3.5-4B smoke instead (DESIGN_ASYNC 5.2 S0: "TQ fields complete"), or on a one-card job with
# Q38_TEST_REAL_TQ=1.  The codec-level test below (no Ray) runs everywhere.
REAL_TQ = os.environ.get("Q38_TEST_REAL_TQ", "") == "1"
REAL_TQ_SKIP = (
    "real TransferQueue needs a local Ray, which does not start on the fujian login node (raylet start-up timeout, "
    "2026-11-18); set Q38_TEST_REAL_TQ=1 on a node where it starts; covered by the 4B smoke S0"
)


@pytest.fixture(scope="module")
def tq_ray():
    """A local Ray of 2 CPUs: the TransferQueue controller and one storage unit (one CPU each).

    Skipped (REAL_TQ_SKIP) unless Q38_TEST_REAL_TQ=1, and when Ray still cannot start."""
    if not REAL_TQ:
        pytest.skip(REAL_TQ_SKIP)
    import ray
    import transfer_queue as tq

    try:
        ray.init(num_cpus=2, include_dashboard=False, log_to_driver=False, object_store_memory=200 * 1024**2)
    except Exception as exc:  # noqa: BLE001 (any start-up failure means: no cluster here)
        try:
            ray.shutdown()
        finally:
            pytest.skip(f"local Ray could not start on this node: {str(exc)[:160]}")
    tq.init(OmegaConf.create({"backend": {"SimpleStorage": {"num_data_storage_units": 1}}}))
    yield tq
    tq.close()
    ray.shutdown()


def test_t6_extra_fields_survive_the_transfer_queue_codec():
    """Codec level only (no Ray): TransferQueue's message codec (transfer_queue/utils/serial_utils.py encode / decode,
    msgpack with a pickle fallback, used for every ZMQ message, zmq_utils.py:225-252) gives the episode's extra_fields
    back equal; a tuple would come back as a list, which is why the loop writes lists only."""
    from transfer_queue.utils import serial_utils

    model = default_model()
    _, out = run_episode(model, FakeServer(model.tokenizer, versions=[(3, 3), (3, 4)]), done_at=2)
    ef = out.extra_fields
    assert_json_style(ef)
    assert serial_utils.decode(serial_utils.encode({"body": [ef, ef]})) == {"body": [ef, ef]}
    assert serial_utils.decode(serial_utils.encode({"span": (1, 2)})) == {"span": [1, 2]}


def _row_of(td, key, i):
    v = td[key]
    if isinstance(v, torch.Tensor):
        return v.unbind()[i] if v.is_nested else v[i]
    return getattr(list(v)[i], "data", list(v)[i])


def test_t6_extra_fields_round_trip_through_transfer_queue(tq_ray):
    """Two episodes stored as AgentLoopWorkerTQ stores them, read back as the sampler reads them (kv_batch_get, then
    list() and .data per item, interface note i2 5.1): extra_fields come back equal, tags carry the versions."""
    import uuid

    from verl.utils.tensordict_utils import list_of_dict_to_tensordict

    tq = tq_ray
    model = default_model()
    cfg = make_config()
    pid = f"t6-{uuid.uuid4().hex}"
    outs, keys = [], []
    for seed, versions in ((0, [(3, 3), (3, 4)]), (1, [(4, 4)])):
        server = FakeServer(model.tokenizer, versions=versions)
        kwargs = row(seed=seed, done_at=2)
        _, out = run_episode(model, server, seed=seed, done_at=2)
        field, tag = tq_fields(model, cfg, out, kwargs)
        key = f"{kwargs['uid']}_{kwargs['session_id']}_0"
        tq.kv_batch_put(keys=[key], partition_id=pid, fields=list_of_dict_to_tensordict([field]), tags=[tag])
        outs.append(out)
        keys.append(key)

    td = tq.kv_batch_get(keys=keys, partition_id=pid,
                         select_fields=["rm_scores", "extra_fields", "prompts", "responses", "response_mask"])
    for i, out in enumerate(outs):
        got = _row_of(td, "extra_fields", i)
        assert got == out.extra_fields
        assert_json_style(got)
        assert _row_of(td, "rm_scores", i).sum().item() == pytest.approx(out.reward_score)
        assert _row_of(td, "responses", i).tolist() == out.response_ids
        assert _row_of(td, "response_mask", i).tolist() == out.response_mask
    tags = tq.kv_list(partition_id=pid)[pid]
    assert (tags[keys[0]]["min_global_steps"], tags[keys[0]]["max_global_steps"]) == (3, 4)
    assert (tags[keys[1]]["min_global_steps"], tags[keys[1]]["max_global_steps"]) == (4, 4)
    tq.kv_clear(keys=keys, partition_id=pid)


def test_t6_missing_top_level_field_and_tag_merge(tq_ray):
    """Recorded behaviour: (a) reading a field one of the samples lacks raises ValueError (so every episode must carry
    the same top-level fields); (b) a later tag-only put merges into the key's tags (a draw_batch tag survives a status
    update, which the trainer's re-issue after a resume relies on, DESIGN_ASYNC 2.5.2)."""
    import uuid

    from verl.utils.tensordict_utils import list_of_dict_to_tensordict

    tq = tq_ray
    pid = f"t6m-{uuid.uuid4().hex}"
    full = {"rm_scores": torch.zeros(3), "extra_fields": {"vagen_stats": {"step_exceptions": 0}}}
    tq.kv_batch_put(keys=["a_0_0"], partition_id=pid, fields=list_of_dict_to_tensordict([full]),
                    tags=[{"status": "success"}])
    tq.kv_batch_put(keys=["b_0_0"], partition_id=pid, fields=list_of_dict_to_tensordict([{"rm_scores": torch.ones(2)}]),
                    tags=[{"status": "success"}])
    with pytest.raises(ValueError):
        tq.kv_batch_get(keys=["a_0_0", "b_0_0"], partition_id=pid, select_fields=["extra_fields"])

    tq.kv_put(key="g", partition_id=pid, tag={"is_prompt": True, "status": "pending", "global_steps": 3,
                                              "draw_batch": 3, "draw_index": 5})
    tq.kv_put(key="g", partition_id=pid, tag={"status": "running"})
    tag = tq.kv_list(partition_id=pid)[pid]["g"]
    assert tag["status"] == "running" and tag["draw_batch"] == 3 and tag["draw_index"] == 5
    tq.kv_clear(keys=["a_0_0", "b_0_0", "g"], partition_id=pid)
