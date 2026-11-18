"""q38: evaluation.py's model-type switch (TEMPLATE_CHECK 1, STACK_SUPPORT 2.5; DESIGN_ASYNC question 7).

A Qwen3.5 / Qwen3.8 model (config.json model_type "qwen3_5") gets enable_thinking False at every chat call and
generation_config "vllm" on its LLM; every other model type -- the 8B main line's qwen3_vl above all -- gets exactly the
calls and model.json it had before the switch.  The second is checked against the evaluation.py of d0c87a12 (the q38
branch before the switch), loaded from git: the same fake vLLM records every call of both versions.  The episodes
themselves need a GPU and the render service (the S3 smoke's 2-card evaluation).
"""

import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from data_gen.interiorgs_v10 import evaluation as E
from data_gen.interiorgs_v10 import grounding_probe

REPO = Path(__file__).resolve().parents[3]
BEFORE = "d0c87a12"  # the q38 branch's HEAD before the switch
EVAL_PY = "data_gen/interiorgs_v10/evaluation.py"
SMOKE_4B_HF = Path("/mnt/umm/users/yinbaiqiao/q38/runs/q38-smoke-4b-s3/checkpoints/global_step_8/actor/huggingface")
Q38_27B = Path("/mnt/umm/users/yinbaiqiao/q38/models/Qwen3.8-27B")
EMPTY_THINK = "<|im_start|>assistant\n<think>\n\n</think>\n\n"


def _image():
    from PIL import Image

    return Image.new("RGB", (4, 4), (10, 20, 30))


class _Params:  # vllm.SamplingParams stand-in: equal when given the same arguments
    def __init__(self, **kw):
        self.kw = kw

    def __eq__(self, other):
        return isinstance(other, _Params) and self.kw == other.kw

    def __repr__(self):
        return f"SamplingParams({self.kw})"


def _fake_vllm(log):
    out = SimpleNamespace(prompt_token_ids=[1, 2, 3],
                          outputs=[SimpleNamespace(text="<action>move_forward</action>", finish_reason="stop",
                                                   stop_reason="</action>", logprobs=None)])

    class LLM:
        def __init__(self, **kw):
            log.append(("init", kw))

        def chat(self, messages, **kw):
            log.append(("chat", messages, kw))
            return [out]

        def preprocess_chat(self, messages, **kw):
            log.append(("preprocess_chat", messages, kw))
            return ["PROMPT"]

        def generate(self, prompts, **kw):
            log.append(("generate", prompts, kw))
            return [out]

        def get_tokenizer(self):
            return None

    return types.SimpleNamespace(LLM=LLM, SamplingParams=_Params)


def _model_dir(root, model_type):
    d = Path(root) / (model_type or "no_config")
    d.mkdir()
    if model_type:
        (d / "config.json").write_text(json.dumps({"model_type": model_type}))
    return str(d)


def _drive(mod, model):
    """Every vLLM call of an action call with history and a grounding side call, as the fake LLM records them."""
    log = []
    image = _image()
    with mock.patch.dict(sys.modules, {"vllm": _fake_vllm(log)}), \
            mock.patch.object(grounding_probe, "prefixed_prompt", lambda prompt, tok, prefix: (prompt, [])):
        policy = mod.VllmPolicy(model, max_model_len=16384, max_images=12)
        text = policy.generate("SYS\n", "turn 2 <image>\n", image,
                               earlier=[("turn 1 <image>\n", image, "<action>turn_left</action>")])
        policy.last_prompt_ids = None  # no layout check (the fake tokenizer has no ids)
        side = policy.side_generate("SYS\n", "turn 2 <image>\n", image, prefix="<state>bearing")
    return log, text, side


def _args(model):
    return SimpleNamespace(name="q38_test", model=model, samples=1, temperature=0.0, top_p=1.0,
                           sample_seed=20261016, max_turns=12, history=True, sft_text=False, entropy=False)


def _record(mod, model, probe_key=None):
    return mod.model_record(_args(model), {"prompt_format": "no_think"}, {"prompt_format_example": "action_names"},
                            False, dict(probe_key or {}), 12.5)


def _before_module():
    """evaluation.py as of BEFORE, as a module (None when git or the commit is not there)."""
    try:
        src = subprocess.run(["git", "-C", str(REPO), "show", f"{BEFORE}:{EVAL_PY}"], capture_output=True,
                             text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None
    mod = types.ModuleType(f"evaluation_{BEFORE}")
    mod.__file__ = str(REPO / EVAL_PY)
    sys.modules[mod.__name__] = mod
    try:
        exec(compile(src, f"{BEFORE}:{EVAL_PY}", "exec"), mod.__dict__)
    finally:
        sys.modules.pop(mod.__name__, None)
    return mod


class ModelTypeSwitchTest(unittest.TestCase):
    def test_tables(self):
        self.assertEqual(E.model_type_kwargs("qwen3_5"), ({"enable_thinking": False}, {"generation_config": "vllm"}))
        for kind in ("qwen3_vl", "qwen2_5_vl", "cambrian_qwen", None, ""):
            self.assertEqual(E.model_type_kwargs(kind), ({}, {}))
        chat, llm = E.model_type_kwargs("qwen3_5")
        chat["enable_thinking"] = True
        llm.clear()
        self.assertEqual(E.model_type_kwargs("qwen3_5"), ({"enable_thinking": False}, {"generation_config": "vllm"}))

    def test_qwen3_5_gets_the_switch(self):
        with tempfile.TemporaryDirectory() as root:
            model = _model_dir(root, "qwen3_5")
            log, text, side = _drive(E, model)
            self.assertEqual(text, "<action>move_forward</action>")
            (init,), = [e[1:] for e in log if e[0] == "init"]
            self.assertEqual(init, {"model": model, "gpu_memory_utilization": 0.6, "max_model_len": 16384,
                                    "enforce_eager": True, "enable_prefix_caching": True,
                                    "limit_mm_per_prompt": {"image": 12}, "trust_remote_code": False,
                                    "generation_config": "vllm"})
            (_, chat_kw), = [e[1:] for e in log if e[0] == "chat"]
            self.assertEqual(chat_kw["chat_template_kwargs"], {"enable_thinking": False})
            self.assertEqual(set(chat_kw), {"sampling_params", "use_tqdm", "chat_template_kwargs"})
            (_, pre_kw), = [e[1:] for e in log if e[0] == "preprocess_chat"]
            self.assertEqual(pre_kw, {"chat_template_kwargs": {"enable_thinking": False}})
            with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1"}):
                rec = _record(E, model, {"grounding_probe": "v2"})
            keys = list(rec)
            self.assertEqual(keys[-5:], ["model_type", "chat_template_kwargs", "llm_kwargs", "cuda_visible_devices",
                                         "grounding_probe"])  # the switch's keys before the probes' keys
            self.assertEqual((rec["model_type"], rec["chat_template_kwargs"], rec["llm_kwargs"],
                              rec["cuda_visible_devices"]),
                             ("qwen3_5", {"enable_thinking": False}, {"generation_config": "vllm"}, "1"))

    def test_other_types_exact_calls(self):
        """Without the switch: the LLM's arguments and every call's arguments as they always were."""
        with tempfile.TemporaryDirectory() as root:
            for kind in ("qwen3_vl", "qwen2_5_vl", None):
                model = _model_dir(root, kind)
                log, _, _ = _drive(E, model)
                (init,), = [e[1:] for e in log if e[0] == "init"]
                self.assertEqual(init, {"model": model, "gpu_memory_utilization": 0.6, "max_model_len": 16384,
                                        "enforce_eager": True, "enable_prefix_caching": True,
                                        "limit_mm_per_prompt": {"image": 12}, "trust_remote_code": False}, kind)
                (_, chat_kw), = [e[1:] for e in log if e[0] == "chat"]
                self.assertEqual(set(chat_kw), {"sampling_params", "use_tqdm"}, kind)
                (_, pre_kw), = [e[1:] for e in log if e[0] == "preprocess_chat"]
                self.assertEqual(pre_kw, {}, kind)
                with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1"}):
                    rec = _record(E, model)
                self.assertEqual(list(rec), ["name", "model", "version", "prompt_format", "greedy", "samples",
                                             "temperature", "top_p", "sample_seed", "env_switches", "max_turns",
                                             "history", "sft_text", "entropy", "load_seconds"], kind)

    def test_other_types_same_as_before_the_switch(self):
        """The 8B path, word for word: every vLLM call (arguments, messages, sampling parameters) and model.json of
        qwen3_vl, qwen2_5_vl and a folder without config.json equal those of BEFORE's evaluation.py."""
        old = _before_module()
        if old is None:
            self.skipTest(f"git show {BEFORE}:{EVAL_PY} is not available")
        self.assertFalse(hasattr(old, "model_type_kwargs"))  # really the version before the switch
        with tempfile.TemporaryDirectory() as root:
            for kind in ("qwen3_vl", "qwen2_5_vl", None):
                model = _model_dir(root, kind)
                self.assertEqual(_drive(E, model), _drive(old, model), kind)
                for probe in ({}, {"grounding_probe": "v2"}, {"critic_probe": "critic_probe_v1",
                                                               "critic_probe_tables": 3}):
                    new_rec, old_rec = _record(E, model, probe), _record(old, model, probe)
                    self.assertEqual(json.dumps(new_rec), json.dumps(old_rec), (kind, probe))  # key order too
            # the chat messages and the default prompt layout are the same functions
            image = _image()
            args = ("SYS\n", [("turn 1 <image>\n", image, "a")], "turn 2 <image>\n", image, E._image_url)
            self.assertEqual(E.chat_messages(*args), old.chat_messages(*args[:4], old._image_url))

    def test_policy_built_without_init(self):
        """A policy built with __new__ (test_grounding_probe's SidePolicyTest) calls chat as before the switch."""
        log = []
        fake = _fake_vllm(log)
        policy = E.VllmPolicy.__new__(E.VllmPolicy)
        policy.llm, policy.cambrian, policy.temperature, policy.top_p, policy.entropy = fake.LLM(), None, 0.0, 1.0, False
        with mock.patch.dict(sys.modules, {"vllm": fake}):
            policy.generate("SYS\n", "turn 1 <image>\n", _image())
        (_, chat_kw), = [e[1:] for e in log if e[0] == "chat"]
        self.assertEqual(set(chat_kw), {"sampling_params", "use_tqdm"})

    def test_resume_ignores_the_cards(self):
        with tempfile.TemporaryDirectory() as root:
            out = Path(root)
            rec = {"name": "a", "model": "m", "model_type": "qwen3_5", "cuda_visible_devices": "1"}
            (out / "model.json").write_text(json.dumps(rec))
            (out / "results.jsonl").write_text(json.dumps({"index": 0}) + "\n")
            E.check_resume_record(out, dict(rec, name="b", cuda_visible_devices="0"))
            with self.assertRaises(SystemExit):
                E.check_resume_record(out, dict(rec, model_type="qwen3_vl"))


class TemplateTest(unittest.TestCase):
    """The real chat templates (tokenizer files only, no weights): the switch's arguments give the empty think block
    as the generation prompt and no reasoning-effort note; without them the model would think."""

    def _render(self, path, **kw):
        if not (path / "tokenizer_config.json").is_file():
            self.skipTest(f"{path} is not here")
        try:
            from transformers import AutoTokenizer

            tok = AutoTokenizer.from_pretrained(str(path))
        except Exception as exc:  # e.g. transformers older than the template's needs
            self.skipTest(f"{path}: tokenizer not loadable here ({type(exc).__name__})")
        msgs = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "U1"},
                {"role": "assistant", "content": "A1"}, {"role": "user", "content": "U2"}]
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)

    def _check(self, path):
        from vagen.models.model_types import of_checkpoint

        if of_checkpoint(str(path)) is not None:
            self.assertEqual(of_checkpoint(str(path)), "qwen3_5")
        chat, _ = E.model_type_kwargs("qwen3_5")
        on = self._render(path, **chat)
        self.assertTrue(on.endswith(EMPTY_THINK), on[-60:])
        self.assertNotIn("Reasoning effort", on)
        off = self._render(path)
        self.assertTrue(off.endswith("<|im_start|>assistant\n<think>\n"), off[-60:])
        return on

    def test_smoke_4b_checkpoint(self):
        on = self._check(SMOKE_4B_HF)
        # the 4B template drops the think block of earlier turns (RL's token history keeps it; noted in the S3 report)
        self.assertIn("<|im_start|>assistant\nA1<|im_end|>", on)

    def test_27b(self):
        on = self._check(Q38_27B)
        self.assertIn("<|im_start|>assistant\n<think>\n\n</think>\n\nA1<|im_end|>", on)  # TEMPLATE_CHECK 1


if __name__ == "__main__":
    unittest.main()
