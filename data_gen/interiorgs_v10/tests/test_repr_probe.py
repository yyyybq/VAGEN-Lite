"""Gate 0 of docs/PREREG_PPO2.md (the representation probe): evaluation run --critic-probe (the recording mode) and
repr_probe.py (extract's training-layout sequence and positions, the check against the run, the targets, the folds, the
probes, the baselines, the reading).  CPU only, no model: a Qwen-like stand-in processor; the agent loop's own handlers
(vagen/agent_loop/gym_agent_loop.py) with verl's infrastructure stubbed; the real processor's test runs only where its
runtime is."""

import argparse
import asyncio
import contextlib
import dataclasses
import hashlib
import importlib
import importlib.util
import json
import os
import random
import re
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest import mock

import numpy as np
import torch
from PIL import Image

from data_gen.interiorgs_v10 import evaluation as E
from data_gen.interiorgs_v10 import repr_probe as R

MODEL_KEYS = {"name", "model", "version", "prompt_format", "greedy", "samples", "temperature", "top_p", "sample_seed",
              "env_switches", "max_turns", "history", "sft_text", "entropy", "load_seconds"}  # model.json, plain run
PROBE_MODEL_KEYS = {"critic_probe", "critic_probe_tables"}
PROBE_TURN_KEYS = {"reward_exact", "critic_features", "critic_features_error", "prompt_ids_sha256", "finish_reason",
                   "stop_reason"}
PROBE_ROW_KEYS = {"system", "final_critic_features", "final_critic_features_error"}


def item(i=0, scene="s0", family="absolute_positioning", d_star=3):
    return {"task_id": f"t{i}", "scene_id": scene, "task_type": family, "difficulty": {"d_star": d_star},
            "task_description": f"task {i}"}


def frame(pose, index=0):
    """A distinct frame per pose (its size changes the stand-in processor's image tokens)."""
    return Image.new("RGB", (4 + pose % 3, 4), color=(10 * pose, 3 * index, 7))


def feats(pose, family="absolute_positioning"):
    return {"v": 1, "d": max(0, 3 - pose), "near": pose == 2, "table": True,
            "bearing": R.BEARINGS[pose % 4], "condition": R.CONDITIONS[pose % 3] if family == R.AP else None,
            "family": family, "phi": 0.1 * pose}


class ProbeEnv:
    """A scripted env: pose p's frame is frame(p) and its truth feats(p); reset is pose 0, each step one pose on;
    the env's critic_features (reset / step info) only when its config turns them on, as the real env."""

    def __init__(self, cfg=None, items=None, turns=3, features=None, fail_pose=None):
        self.cfg, self.config = cfg, cfg
        self.items, self.turns, self.fail_pose = items or [item()], turns, fail_pose
        self.features = bool(getattr(cfg, "critic_features", False)) if features is None else features
        self.read, self.current_item, self.pose, self._current_step = [], None, 0, 0

    def info(self):
        if not self.features:
            return {}
        if self.pose == self.fail_pose:
            return {"critic_features": None, "critic_features_error": "OSError: table unreadable"}
        return {"critic_features": feats(self.pose, self.current_item["task_type"])}

    def obs(self):
        return {"obs_str": f"<image>\nturn {self.pose}\n", "multi_modal_data": {"<image>": [frame(self.pose, self.k)]}}

    def reset(self, seed):
        self.k = seed
        self.current_item, self.pose, self._current_step = self.items[seed % len(self.items)], 0, 0
        return self.obs(), {"task_id": self.current_item["task_id"], **self.info()}

    def system_prompt(self):
        return "sys prompt\n"

    def step(self, text):
        self.read.append(text)
        self.pose += 1
        self._current_step += 1
        n = self.turns + self.k % 2  # some episodes one turn longer
        reward = -0.0123456789 * self.pose * (1 + self.k % 3) + (5.0 if self.pose == n and self.k % 2 == 0 else 0.0)
        info = {"actions": ["move_forward"], "metrics": {}, **self.info()}
        return self.obs(), reward, self.pose >= n, info

    def _calculate_canonical_metric(self):
        n = self.turns + self.k % 2
        return {"success": self.pose >= n and self.k % 2 == 0}


# ---------------------------------------------------------------------------
# a Qwen-like processor and tokenizer
# ---------------------------------------------------------------------------
SPECIAL = ("<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|image_pad|>", "<|vision_end|>")
VOCAB = {}  # id -> token, filled by tid


def tid(token):
    """A fixed id per token (the same in every instance)."""
    i = int(hashlib.sha256(token.encode()).hexdigest()[:8], 16) % 100000
    VOCAB[i] = token
    return i


def tokens(text):
    """Special tokens whole, a run of newlines one token (as Qwen's pre-tokeniser merges "\\n\\n"), words, single
    characters."""
    return re.findall("|".join(re.escape(s) for s in SPECIAL) + r"|\n+|\w+|[^\w\n]", text)


class FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [tid(t) for t in tokens(text)]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(VOCAB[int(i)] for i in ids if not (skip_special_tokens and VOCAB[int(i)] in SPECIAL))

    def convert_tokens_to_ids(self, token):
        return tid(token)


class FakeProcessor:
    """The Qwen chat template (string or list content, image parts as vision placeholders, other keyword arguments
    ignored) and processor: each image placeholder becomes (w * h) // 4 image tokens of its frame; tensors as the
    real one returns them (input_ids [1, L], the frames' pixel_values and image_grid_thw)."""

    def __init__(self):
        self.tokenizer = FakeTokenizer()

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
        assert not tokenize
        out = []
        for m in messages:
            c = m["content"]
            body = c if isinstance(c, str) else "".join(
                "<|vision_start|><|image_pad|><|vision_end|>" if p.get("type") == "image" else p["text"] for p in c)
            out.append(f"<|im_start|>{m['role']}\n{body}<|im_end|>\n")
        return "".join(out) + ("<|im_start|>assistant\n" if add_generation_prompt else "")

    def __call__(self, text, images=None, return_tensors=None):
        (t,) = text
        images, toks, k = list(images or []), [], 0
        for tok in tokens(t):
            if tok == "<|image_pad|>":
                toks += [tok] * ((images[k].size[0] * images[k].size[1]) // 4)
                k += 1
            else:
                toks.append(tok)
        assert k == len(images), (k, len(images))
        out = {"input_ids": torch.tensor([[tid(x) for x in toks]], dtype=torch.long),
               "attention_mask": torch.ones(1, len(toks), dtype=torch.long)}
        if images:
            out["pixel_values"] = torch.cat([torch.tensor(np.asarray(im), dtype=torch.float32).reshape(-1, 3)
                                             for im in images])
            out["image_grid_thw"] = torch.tensor([[1, im.size[1], im.size[0]] for im in images])
        return out


def ids_of(batch):
    return [int(i) for i in batch["input_ids"][0].tolist()]


class FakePolicy:
    """vLLM's chat in miniature: the chat of chat_messages through the processor, the prompt's ids kept; each answer
    ends as ``finishes`` says (vLLM's finish_reason)."""

    def __init__(self, *a, answers=None, finishes=None, **k):  # VllmPolicy(model, ...) in cmd_run
        self.answers = answers or ["<action>move_forward|</action>"]
        self.finishes = finishes or ["stop"]
        self.proc, self.calls = FakeProcessor(), []
        self.last_prompt_tokens = self.last_prompt_ids = self.last_finish = self.last_stop_reason = None

    def generate(self, system, user_text, image, earlier=(), as_sft=False, seed=None):
        messages = E.chat_messages(system, earlier, user_text, image, R.image_placeholder, as_sft)
        images = [im for _, im, _ in earlier] + [image]
        text = self.proc.apply_chat_template(messages, add_generation_prompt=True)
        self.last_prompt_ids = ids_of(self.proc(text=[text], images=images))
        self.last_prompt_tokens = len(self.last_prompt_ids)
        k = len(self.calls)
        self.last_finish = self.finishes[k % len(self.finishes)]
        self.last_stop_reason = "</action>" if self.last_finish == "stop" else None
        self.calls.append({"messages": messages, "images": [np.asarray(im).copy() for im in images], "seed": seed,
                           "ids": list(self.last_prompt_ids)})
        return self.answers[k % len(self.answers)]


def strip_probe(row):
    """A probe row less the probe's fields, its frames renamed back to .jpg."""
    out = {k: v for k, v in row.items() if k not in PROBE_ROW_KEYS}
    out["trajectory"] = [{k: (v.replace(".png", ".jpg") if k == "frame" else v) for k, v in t.items()
                          if k not in PROBE_TURN_KEYS} for t in row["trajectory"]]
    out["final_frame"] = row["final_frame"].replace(".png", ".jpg")
    return out


# ---------------------------------------------------------------------------
# A. evaluation run --critic-probe
# ---------------------------------------------------------------------------
class RunEpisodeProbeTest(unittest.TestCase):
    def test_off_is_unchanged(self):
        """Off: no new field, JPEG frames, even with an env that reports the truth; on, less its fields, the same."""
        with tempfile.TemporaryDirectory() as tmp:
            off = E.run_episode(ProbeEnv(features=True), FakePolicy(), 0, 12, frames=Path(tmp) / "a" / "frames",
                                history=True, sample=0, seed=5)
            on = E.run_episode(ProbeEnv(features=True), FakePolicy(), 0, 12, frames=Path(tmp) / "b" / "frames",
                               history=True, sample=0, seed=5, critic_probe=True)
            jpgs = sorted(p.name for p in (Path(tmp) / "a" / "frames").iterdir())
            pngs = sorted(p.name for p in (Path(tmp) / "b" / "frames").iterdir())
        self.assertFalse(PROBE_ROW_KEYS & set(off))
        self.assertFalse(any(PROBE_TURN_KEYS & set(t) for t in off["trajectory"]))
        self.assertEqual(jpgs, ["000_s0_00.jpg", "000_s0_01.jpg", "000_s0_02.jpg", "000_s0_03.jpg"])
        self.assertEqual(pngs, [n.replace(".jpg", ".png") for n in jpgs])
        self.assertEqual(strip_probe(on), off)

    def test_on_records_each_turn(self):
        env, policy = ProbeEnv(features=True, fail_pose=2), FakePolicy(finishes=["stop", "length", "stop"])
        with tempfile.TemporaryDirectory() as tmp:
            frames = Path(tmp) / "frames"
            row = E.run_episode(env, policy, 0, 12, frames=frames, history=True, critic_probe=True)
            loaded = [np.asarray(R.load_png(Path(tmp) / t["frame"])) for t in row["trajectory"]]
        traj = row["trajectory"]
        self.assertEqual(row["turns"], 3)
        self.assertEqual(row["system"], "sys prompt\n")
        # turn t answered the observation at pose t: reset's truth for turn 0, step t-1's after
        self.assertEqual([t["critic_features"] for t in traj], [feats(0), feats(1), None])
        self.assertEqual(traj[2]["critic_features_error"], "OSError: table unreadable")
        self.assertNotIn("critic_features_error", traj[1])
        self.assertEqual(row["final_critic_features"], feats(3))
        for t, pix in enumerate(loaded):  # the PNG is the frame the model got at that turn, pixel for pixel
            self.assertTrue(np.array_equal(pix, np.asarray(frame(t))), t)
            self.assertTrue(np.array_equal(pix, policy.calls[t]["images"][-1]), t)
        self.assertEqual([t["reward_exact"] for t in traj],
                         [-0.0123456789 * 1, -0.0123456789 * 2, -0.0123456789 * 3 + 5.0])  # the env's, unrounded
        self.assertEqual([t["reward"] for t in traj], [round(t["reward_exact"], 4) for t in traj])
        self.assertEqual([t["prompt_ids_sha256"] for t in traj], [E.ids_digest(c["ids"]) for c in policy.calls])
        self.assertEqual([t["prompt_tokens"] for t in traj], [len(c["ids"]) for c in policy.calls])
        self.assertEqual([(t["finish_reason"], t["stop_reason"]) for t in traj],
                         [("stop", "</action>"), ("length", None), ("stop", "</action>")])

    def test_final_truth_when_the_turn_limit_ends_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            row = E.run_episode(ProbeEnv(features=True, turns=5), FakePolicy(), 0, 2, frames=Path(tmp) / "frames",
                                history=True, critic_probe=True)
        self.assertEqual((row["turns"], row["end_reason"]), (2, "max_turns"))
        self.assertEqual([t["critic_features"] for t in row["trajectory"]], [feats(0), feats(1)])
        self.assertEqual(row["final_critic_features"], feats(2))  # the observation after the last answer

    def test_refusals(self):
        with tempfile.TemporaryDirectory() as tmp:
            frames = Path(tmp) / "frames"
            for kw in ({"grounding": True}, {"grounding_side": True}):
                with self.assertRaises(ValueError):
                    E.run_episode(ProbeEnv(features=True), FakePolicy(), 0, 12, frames=frames, critic_probe=True, **kw)
            with self.assertRaises(ValueError):  # the probe needs the trajectory
                E.run_episode(ProbeEnv(features=True), FakePolicy(), 0, 12, critic_probe=True)
            with self.assertRaises(RuntimeError):  # an env without the truth: loud
                E.run_episode(ProbeEnv(features=False), FakePolicy(), 0, 12, frames=frames, critic_probe=True)

    def test_overrides(self):
        o = E.critic_probe_overrides("/tables/b3000")
        from data_gen.interiorgs_v10.training_prep import reward_overrides

        self.assertEqual(o, {**reward_overrides("D_cond"), "potential_field_reward_scale": 3.3333333333333335,
                             "critic_features": True, "critic_feature_tables_dir": os.path.abspath("/tables/b3000")})
        self.assertEqual((o["progress_version"], o["enable_potential_shaping_reward"]), ("grouped_gated", True))
        self.assertEqual(E.switch_overrides(argparse.Namespace()), {})
        self.assertEqual(E.switch_overrides(argparse.Namespace(critic_probe=None, check_each_action=True)),
                         {"success_check_each_action": True})
        self.assertEqual(E.switch_overrides(argparse.Namespace(critic_probe="/tables/b3000")), o)
        for empty in ("", "  "):  # not a silent "off"
            with self.assertRaises(SystemExit):
                E.switch_overrides(argparse.Namespace(critic_probe=empty))


class RealEnvConfigTest(unittest.TestCase):
    """The probe's env keys against the real ActiveSpatialEnvConfig and the real system prompt (no renderer)."""

    def test_keys_and_system_prompt(self):
        from vagen.envs.active_spatial.env_config import ActiveSpatialEnvConfig

        from data_gen.interiorgs_v10.training_prep import env_system_prompt
        from vagen.feature_critic import FAMILIES

        fields = set(ActiveSpatialEnvConfig.__dataclass_fields__)
        self.assertLessEqual(set(E.critic_probe_overrides("/t")), fields)
        template = Path(E.__file__).with_name("configs") / "rl_env_v10_S0.yaml"
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"R1_RENDER_URL": "http://x.invalid"}):
            tasks = Path(tmp) / "tasks.jsonl"
            tasks.write_text(json.dumps(item()) + "\n")
            for extra in ({}, {"format_example": "action_names"}):  # the plan gives --format-example action_names
                plain = E.env_config(template, tasks, "/gs", None, E.switch_overrides(argparse.Namespace(**extra)))
                probe = E.env_config(template, tasks, "/gs", None,
                                     E.switch_overrides(argparse.Namespace(critic_probe="/t", **extra)))
                self.assertLessEqual(set(probe) - set(plain), fields)
                for family in FAMILIES:  # V4's reward keys do not change what the model is told
                    self.assertEqual(env_system_prompt(plain, family), env_system_prompt(probe, family), family)


def make_cfg(*missing):
    """An env-config stand-in with the fields the switches set (less ``missing``)."""
    names = ["jsonl_path", "prompt_format", "prompt_format_example"] + list(E.critic_probe_overrides("/t"))
    return dataclasses.make_dataclass("Cfg", [(n, Any, None) for n in names if n not in missing])


def make_tables(root, items):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    for it in items:
        (root / f"{it['task_id']}.npz").write_bytes(b"")
    return root


class CmdRunProbeTest(unittest.TestCase):
    def run_cmd(self, tmp, name, opts, items=None, cfg=None, env_cls=ProbeEnv):
        envs = []

        def build(c):
            envs.append(env_cls(c, items=items))
            return envs[-1]

        env_mod = types.ModuleType("vagen.envs.active_spatial.env")
        env_mod.ActiveSpatialEnv = build
        cfg_mod = types.ModuleType("vagen.envs.active_spatial.env_config")
        cfg_mod.ActiveSpatialEnvConfig = cfg or make_cfg()
        ev = Path(tmp)
        (ev / "tasks.jsonl").write_text("".join(json.dumps(it) + "\n" for it in (items or [item()])))
        with mock.patch.dict(sys.modules, {env_mod.__name__: env_mod, cfg_mod.__name__: cfg_mod}), \
                mock.patch.object(E, "VllmPolicy", FakePolicy), \
                mock.patch.dict(os.environ, {"R1_RENDER_URL": "http://scripted.invalid"}):
            E.main(["run", "--eval", str(ev), "--name", name, "--model", "/m", *opts])
        out = ev / name
        rows = [json.loads(l) for l in (out / "results.jsonl").read_text().splitlines()]
        return json.loads((out / "model.json").read_text()), rows, (envs[-1] if envs else None)

    def test_off_and_on(self):
        with tempfile.TemporaryDirectory() as tmp:
            tables = make_tables(Path(tmp) / "tables", [item()])
            m0, plain, env0 = self.run_cmd(tmp, "plain", ["--temperature", "0.8", "--top-p", "0.92"])
            self.assertEqual(set(m0), MODEL_KEYS)
            self.assertEqual(m0["env_switches"], {})
            self.assertFalse(env0.cfg.critic_features)
            self.assertFalse(PROBE_ROW_KEYS & set(plain[0]))
            m1, probe, env1 = self.run_cmd(tmp, "probe", ["--temperature", "0.8", "--top-p", "0.92",
                                                          "--critic-probe", str(tables)])
            self.assertEqual(set(m1), MODEL_KEYS | PROBE_MODEL_KEYS)
            self.assertEqual(m1["critic_probe"], E.CRITIC_PROBE_VERSION)
            self.assertEqual(m1["critic_probe_tables"], {"tasks": 1, "tables_found": 1})
            self.assertEqual(m1["env_switches"], E.critic_probe_overrides(str(tables)))
            # the env got the switches: the truth on, V4's progress scale and grouped progress
            self.assertEqual((env1.cfg.critic_features, env1.cfg.critic_feature_tables_dir,
                              env1.cfg.potential_field_reward_scale, env1.cfg.progress_version),
                             (True, os.path.abspath(str(tables)), 3.3333333333333335, "grouped_gated"))
            self.assertEqual([t["critic_features"] for t in probe[0]["trajectory"]], [feats(0), feats(1), feats(2)])
            self.assertTrue(probe[0]["trajectory"][0]["frame"].endswith("_s0_00.png"))
            self.assertEqual(strip_probe(probe[0]), plain[0])  # the stand-in env's reward does not depend on it
            for m in (m0, m1):
                m.pop("load_seconds"), m.pop("name"), m.pop("env_switches")
            m1.pop("critic_probe"), m1.pop("critic_probe_tables")
            self.assertEqual(m0, m1)  # model.json: only the probe's keys differ

    def test_refusals_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            tables = make_tables(Path(tmp) / "tables", [item()])
            other = make_tables(Path(tmp) / "tables2", [item()])
            empty = make_tables(Path(tmp) / "tables3", [])
            self.run_cmd(tmp, "plain", [])
            self.run_cmd(tmp, "probe", ["--critic-probe", str(tables)])
            for name, opts, why in (
                    ("plain", ["--critic-probe", str(tables)], "critic_probe None"),  # a plain folder stays plain
                    ("probe", [], "critic_probe 'critic_probe_v1'"),  # a probe folder stays a probe folder
                    ("probe", ["--critic-probe", str(other)], "critic_feature_tables_dir"),
                    ("probe", ["--critic-probe", str(tables), "--temperature", "0.8"], "differs from this run's"),
                    ("x", ["--critic-probe", str(tables), "--grounding-probe"], "not with --grounding"),
                    ("x", ["--critic-probe", str(tables), "--grounding-side"], "not with --grounding"),
                    ("x", ["--critic-probe", str(tables), "--no-history"], "needs --history"),
                    ("x", ["--critic-probe", str(tables), "--sft-text"], "not with --sft-text"),
                    ("x", ["--critic-probe", ""], "an empty value"),
                    ("x", ["--critic-probe", str(empty)], "1 of 1 tasks have no distance table"),
                    ("x", ["--critic-probe", str(Path(tmp) / "missing")], "not a directory")):
                with self.subTest(name=name, opts=opts), self.assertRaises(SystemExit) as ctx:
                    self.run_cmd(tmp, name, opts)
                self.assertIn(why, str(ctx.exception))
            self.assertFalse((Path(tmp) / "x" / "model.json").exists())  # refused before anything is written
            self.assertFalse((Path(tmp) / "x" / "results.jsonl").exists())
            m, rows, _ = self.run_cmd(tmp, "probe", ["--critic-probe", str(tables)])  # the same probe resumes
            self.assertEqual(len(rows), 1)
            plain, rows, _ = self.run_cmd(tmp, "plain", ["--temperature", "0.8"])  # plain folders: as before (the
            self.assertEqual(len(rows), 2)  # sampled episode is added, no refusal)

    def test_switches_reach_the_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            tables = make_tables(Path(tmp) / "tables", [item()])
            with self.assertRaises(SystemExit) as ctx:  # a switch the env config does not have: refused
                self.run_cmd(tmp, "a", ["--critic-probe", str(tables)], cfg=make_cfg("potential_field_gamma"))
            self.assertIn("potential_field_gamma", str(ctx.exception))

            class Changing(ProbeEnv):  # an env that changes a setting on the way
                def __init__(self, cfg=None, **kw):
                    super().__init__(cfg, **kw)
                    self.config = dataclasses.replace(cfg, potential_field_reward_scale=10.0)

            with self.assertRaises(SystemExit) as ctx:
                self.run_cmd(tmp, "b", ["--critic-probe", str(tables)], env_cls=Changing)
            self.assertIn("potential_field_reward_scale", str(ctx.exception))
            self.run_cmd(tmp, "c", ["--format-example", "action_names", "--critic-probe", str(tables)])

    def test_errors_in_a_row_stop_the_run(self):
        items = [item(i) for i in range(12)]

        class Broken(ProbeEnv):
            def reset(self, seed):
                raise RuntimeError("render service down")

        with tempfile.TemporaryDirectory() as tmp:
            tables = make_tables(Path(tmp) / "tables", items)
            with self.assertRaises(SystemExit) as ctx:
                self.run_cmd(tmp, "p", ["--critic-probe", str(tables)], items=items, env_cls=Broken)
            self.assertIn("10 episodes in a row", str(ctx.exception))
            out = Path(tmp) / "p"
            self.assertEqual(len((out / "results.jsonl").read_text().splitlines()), 10)
            self.assertFalse((out / "done.json").exists())
            _, rows, _ = self.run_cmd(tmp, "q", [], items=items, env_cls=Broken)  # without the switch: as before
            self.assertEqual(len(rows), 12)
            self.assertTrue((Path(tmp) / "q" / "done.json").exists())


# ---------------------------------------------------------------------------
# B. repr_probe: the training layout, the check against the run, the targets
# ---------------------------------------------------------------------------
def write_run(root, name, items, n_episodes, answers=None, model="/m", turns=3, finishes=None):
    """A finished critic-probe result folder from the stand-in env and policy (cmd_run's files)."""
    out = Path(root) / name
    env, policy = ProbeEnv(features=True, items=items, turns=turns), FakePolicy(answers=answers, finishes=finishes)
    rows = [E.run_episode(env, policy, i, 12, frames=out / "frames", history=True, sample=0, seed=7,
                          critic_probe=True) for i in range(n_episodes)]
    rows.append({"index": n_episodes, "sample": 0, "error": "RuntimeError: render"})  # an infrastructure error
    (out / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (out / "model.json").write_text(json.dumps({"name": name, "model": model, "history": True, "sft_text": False,
                                                "max_turns": 12, "critic_probe": E.CRITIC_PROBE_VERSION,
                                                "env_switches": E.critic_probe_overrides("/tables")}))
    (out / "done.json").write_text("{}")
    return out, policy


def stub_verl():
    """Stand-ins for the verl infrastructure gym_agent_loop.py imports (the handlers under test do not use it)."""
    stubs = {}

    def module(name, **attrs):
        m = types.ModuleType(name)
        m.__path__ = []
        m.__dict__.update(attrs)
        stubs[name] = m

    class AgentLoopBase:
        _class_initialized = False

    @contextlib.contextmanager
    def simple_timer(name, metrics):
        yield

    for name in ("verl", "verl.experimental", "verl.experimental.agent_loop", "verl.utils"):
        module(name)
    module("verl.experimental.agent_loop.agent_loop", AgentLoopBase=AgentLoopBase, AgentLoopOutput=dict,
           register=lambda *a, **k: (lambda c: c))
    module("verl.utils.profiler", simple_timer=simple_timer)
    module("verl.utils.rollout_trace", rollout_trace_op=lambda f: f)
    if importlib.util.find_spec("omegaconf") is None:
        module("omegaconf", OmegaConf=object)
    return stubs


class NS(dict):
    __getattr__ = dict.__getitem__


GAL_NAME = "vagen.agent_loop.gym_agent_loop"


@contextlib.contextmanager
def agent_loop_module():
    """gym_agent_loop.py imported fresh against the verl stand-ins; afterwards the stand-ins go and whatever was there
    before (the real verl, the real module) comes back; nothing else in sys.modules is touched."""
    import vagen.agent_loop as package
    import vagen.envs.gym_image_env  # noqa: F401 - its real imports, before the stand-ins
    import vagen.invalid_turns  # noqa: F401
    import vagen.split_reward_adv  # noqa: F401
    import vagen.turn_gae  # noqa: F401

    stubs = stub_verl()
    saved = {name: sys.modules.get(name) for name in list(stubs) + [GAL_NAME]}
    saved_attr = getattr(package, "gym_agent_loop", None)
    sys.modules.update(stubs)
    sys.modules.pop(GAL_NAME, None)
    try:
        yield importlib.import_module(GAL_NAME)
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
        if saved_attr is None:
            package.__dict__.pop("gym_agent_loop", None)
        else:
            package.gym_agent_loop = saved_attr


def agent_loop_ids(system, turns, outputs):
    """RL's sequence for one episode, from GymAgentLoop's own handlers (vagen/agent_loop/gym_agent_loop.py): init_class
    (the placeholder system's ids), _handle_pending_state, then _handle_generating_state (the generation server stands
    in: it returns ``outputs[t]`` as turn t's token ids) and _handle_env_state (an env that returns the next user turn)
    until the episode ends.  Returns (the ids, the first prompt's length, the turn spans, the frames in order, the
    prompts the server got)."""
    with agent_loop_module() as gal:
        loop_cls = type("Loop", (gal.GymAgentLoop,), {"_class_initialized": False})
        config = NS(actor_rollout_ref=NS(rollout=NS(multi_turn=NS(), prompt_length=10 ** 6, response_length=10 ** 6)),
                    env_registry=NS(), data=NS(apply_chat_template_kwargs=dict(R.TEMPLATE_KWARGS)), algorithm=None)
        proc = FakeProcessor()
        loop_cls.init_class(config, proc.tokenizer, proc)
        agent = loop_cls.__new__(loop_cls)
        agent.env_max_turns = None
        prompts = []

        class Server:
            async def generate(self, request_id, prompt_ids, sampling_params, image_data):
                prompts.append(list(prompt_ids))
                return SimpleNamespace(token_ids=list(outputs[len(prompts) - 1]), log_probs=None)

        class Env:  # the gym wrapper's observation: obs_str and multi_modal_input
            k = 0

            async def step(self, text):
                self.k += 1
                done = self.k >= len(turns)
                user, image, _ = turns[min(self.k, len(turns) - 1)]
                return {"obs_str": user, "multi_modal_input": {"<image>": [image]}}, 0.0, done, {"success": False}

        agent.server_manager = Server()

        async def drive():
            agent.loop = asyncio.get_running_loop()
            sys_obs = {"obs_str": system}  # GymAgentLoop.run: system first, then the first user turn
            init_obs = {"obs_str": turns[0][0], "multi_modal_input": {"<image>": [turns[0][1]]}}
            messages = [{"role": "system", "content": gal.convert_obs_to_content(sys_obs)},
                        {"role": "user", "content": gal.convert_obs_to_content(init_obs)}]
            data = gal.AgentData(messages=messages, image_data=gal._normalize_images([turns[0][1]]), metrics={},
                                 request_id="r", env=Env(), response_limit=10 ** 6, env_name="active_spatial")
            state = await agent._handle_pending_state(data, {})
            prompt0 = len(data.prompt_ids)
            while state != gal.AgentState.TERMINATED:
                if state == gal.AgentState.GENERATING:
                    state = await agent._handle_generating_state(data, {})
                else:
                    state = await agent._handle_env_state(data)
            return data, prompt0

        data, prompt0 = asyncio.run(drive())
    return data.prompt_ids, prompt0, data.turn_spans, data.image_data, prompts


class TrainingLayoutTest(unittest.TestCase):
    TURNS_TEXT = [("<image>\nturn 0\n", "<action>turn_left|</action>"),
                  ("<image>\nturn 1\n", "\n<action>move_forward|</action>"),  # a newline first: merges if re-rendered
                  ("Look: <image>\nturn 2\n", "<action>move_forward|move_forward|</action> and more"),
                  ("<image>\nturn 3\n", "<action>move_backward|</action>")]

    def turns(self):
        return [(u, frame(t), a) for t, (u, a) in enumerate(self.TURNS_TEXT)]

    def test_equals_the_agent_loop(self):
        """training_layout's ids, positions and frames are exactly what GymAgentLoop builds when its generations are
        the answers' tokens with the end-of-turn token (none for an answer cut by the limit)."""
        turns, tok = self.turns(), FakeTokenizer()
        finishes = ["stop", "stop", "length", None]
        eos = tok.convert_tokens_to_ids(R.END_OF_TURN)
        outputs = [tok.encode(a) + ([] if f == "length" else [eos]) for (_, _, a), f in zip(turns, finishes)]
        rl_ids, prompt0, spans, images, prompts = agent_loop_ids("sys prompt\n", turns, outputs)
        layout = R.training_layout(FakeProcessor(), "sys prompt\n", turns, finishes)
        self.assertEqual(layout["input_ids"], rl_ids)
        self.assertEqual(layout["prompt0_len"], prompt0)
        self.assertEqual(layout["positions"], [prompt0 + s - 1 for s, _ in spans])  # before each turn's first token
        self.assertEqual(prompts, [rl_ids[: p + 1] for p in layout["positions"]])  # what the model got each turn
        self.assertEqual((layout["eos_appended"], layout["length_cut"], layout["finish_unknown"]), (3, 1, 1))
        batch = R.model_batch(layout["input_ids"], layout["parts"])
        self.assertEqual(batch["input_ids"][0].tolist(), rl_ids)
        self.assertTrue(torch.equal(batch["pixel_values"], FakeProcessor()(text=["<|image_pad|>" * len(images)],
                                                                           images=images)["pixel_values"]))
        self.assertEqual(batch["image_grid_thw"].tolist(), [[1, im.size[1], im.size[0]] for im in images])
        for p in layout["positions"]:  # the generation prompt's last token
            self.assertEqual([VOCAB[i] for i in rl_ids[p - 2: p + 1]], ["<|im_start|>", "assistant", "\n"])

    def test_differs_from_the_eval_layout(self):
        """The first prompt is the same call in both layouts; after it each earlier turn ends <|im_end|> in RL's
        sequence and <|im_end|>\\n in the re-rendered chat, and a leading newline stays its own token."""
        turns, proc = self.turns(), FakeProcessor()
        layout = R.training_layout(proc, "sys prompt\n", turns)
        evals = [R.encode(proc, R.conversation("sys prompt\n", turns, t), [im for _, im, _ in turns[: t + 1]], True)[0]
                 for t in range(len(turns))]
        self.assertEqual(layout["input_ids"][: layout["prompt0_len"]], evals[0])
        # each earlier turn: one token more in the eval layout ("\n" after <|im_end|>), but turn 1's leading newline
        # merges there with the generation prompt's ("\n\n" one token), one fewer
        self.assertEqual([p + 1 for p in layout["positions"]], [len(e) - d for e, d in zip(evals, (0, 1, 1, 2))])
        p1 = layout["positions"][1] + 1
        self.assertNotEqual(layout["input_ids"][:p1], evals[1][:p1])

    def test_template_kwargs_must_not_change_the_first_prompt(self):
        class Thinking(FakeProcessor):  # a template that reads enable_thinking
            def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False, **kwargs):
                text = super().apply_chat_template(messages, tokenize, add_generation_prompt)
                return text + ("<think>\n\n</think>\n\n" if add_generation_prompt and "enable_thinking" in kwargs else "")

        with tempfile.TemporaryDirectory() as tmp:
            res, _ = write_run(tmp, "run", [item()], 1)
            ep = R.list_episodes([res])[0][0]
            with self.assertRaises(R.Mismatch) as ctx:
                R.episode_inputs(ep, Thinking())
        self.assertIn("first prompt differs", str(ctx.exception))


class ChatAndCheckTest(unittest.TestCase):
    def episode(self, tmp, answers=None, finishes=None):
        res, policy = write_run(tmp, "run", [item()], 1, answers, finishes=finishes)
        episodes, info = R.list_episodes([res])
        self.assertEqual((len(episodes), info["run_errors"]), (1, 1))
        return episodes[0], policy

    def test_rebuilt_chat_is_the_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            ep, policy = self.episode(tmp)
            traj = ep["row"]["trajectory"]
            turns = [(t["user"], R.load_png(Path(ep["result"]) / t["frame"]), t["answer"]) for t in traj]
            for t, call in enumerate(policy.calls):  # the messages and frames of each model call, rebuilt
                self.assertEqual(R.conversation(ep["row"]["system"], turns, t), call["messages"])
                self.assertEqual(len(call["images"]), t + 1)
                for a, (_, im, _) in zip(call["images"], turns[: t + 1]):
                    self.assertTrue(np.array_equal(a, np.asarray(im)))
            full = R.conversation(ep["row"]["system"], turns, len(turns))
            self.assertEqual(full[:-1], R.conversation(ep["row"]["system"], turns, len(turns) - 1))
            self.assertEqual(full[-1], {"role": "assistant", "content": traj[-1]["answer"]})
            with self.assertRaises(ValueError):
                R.conversation("s", turns, len(turns) + 1)

    def test_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            ep, policy = self.episode(tmp, answers=["<action>turn_left|</action>", "\n<action>move_forward|</action>"],
                                      finishes=["stop", "length"])
            inp = R.episode_inputs(ep, FakeProcessor())
            turns = [(t["user"], R.load_png(Path(ep["result"]) / t["frame"]), t["answer"])
                     for t in ep["row"]["trajectory"]]
        layout = R.training_layout(FakeProcessor(), "sys prompt\n", turns, ["stop", "length", "stop"])
        self.assertEqual(inp["batch"]["input_ids"][0].tolist(), layout["input_ids"])
        self.assertEqual(inp["positions"], layout["positions"])
        self.assertEqual(inp["prompt_lens"], [p + 1 for p in layout["positions"]])
        self.assertEqual(inp["eval_prompt_lens"], [len(c["ids"]) for c in policy.calls])  # what vLLM ran
        self.assertEqual((inp["prompt_checked"], inp["prompt_unchecked"]), (3, 0))  # a leading newline is fine
        self.assertEqual((inp["eos_appended"], inp["length_cut"], inp["finish_unknown"]), (2, 1, 0))
        self.assertEqual(inp["seq_len"], len(layout["input_ids"]))
        self.assertEqual(inp["batch"]["image_grid_thw"].shape[0], 3)

    def test_run_prompt_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            ep, _ = self.episode(tmp)
            traj = ep["row"]["trajectory"]
            traj[1]["prompt_ids_sha256"] = "0" * 64
            with self.assertRaises(R.Mismatch) as ctx:
                R.episode_inputs(ep, FakeProcessor())
            self.assertIn("token ids differ", str(ctx.exception))
            off = R.episode_inputs(ep, FakeProcessor(), run_check=False)  # extract --skip-run-check
            self.assertEqual((off["prompt_checked"], off["prompt_unchecked"]), (0, 3))
            traj[1].pop("prompt_ids_sha256")
            self.assertEqual(R.episode_inputs(ep, FakeProcessor())["prompt_unchecked"], 1)
            traj[1]["prompt_tokens"] += 1
            with self.assertRaises(R.Mismatch):
                R.episode_inputs(ep, FakeProcessor())

    def test_refused_folders(self):
        with tempfile.TemporaryDirectory() as tmp:
            res, _ = write_run(tmp, "run", [item()], 1)
            model = json.loads((res / "model.json").read_text())
            for change, why in ((dict(critic_probe=None), "critic_probe"), (dict(history=False), "history"),
                                (dict(sft_text=True), "--sft-text")):
                (res / "model.json").write_text(json.dumps({**model, **change}))
                with self.assertRaises(SystemExit) as ctx:
                    R.list_episodes([res])
                self.assertIn(why, str(ctx.exception))
            (res / "model.json").write_text(json.dumps(model))
            (res / "done.json").unlink()
            with self.assertRaises(SystemExit):
                R.list_episodes([res])


class TargetsTest(unittest.TestCase):
    def row(self, family, rewards, fs, success):
        return {"index": 0, "task_type": family, "success": success,
                "trajectory": [{"reward_exact": r, "reward": round(r, 4), "critic_features": f}
                               for r, f in zip(rewards, fs)]}

    def test_returns_and_turns_left(self):
        rows = R.episode_targets(self.row(R.AP, [-0.01, 0.25, 5.125], [feats(0), feats(1), None], True))
        self.assertEqual([r["G_t"] for r in rows], [5.125 + 0.25 - 0.01, 5.125 + 0.25, 5.125])
        self.assertEqual([r["turns_left"] for r in rows], [3, 2, 1])
        self.assertEqual([r["reward"] for r in rows], [-0.01, 0.25, 5.125])
        self.assertEqual([r["d"] for r in rows], [3, 2, None])
        self.assertEqual([r["condition"] for r in rows], ["too_close", "ok", None])
        self.assertEqual([r["phi"] for r in rows], [0.0, 0.1, None])
        self.assertTrue(all(r["success"] for r in rows))
        other = R.episode_targets(self.row("fov_inclusion", [0.0, 1.0], [dict(feats(0), condition="ok"), feats(1)],
                                           False))
        self.assertEqual([r["condition"] for r in other], [None, None])  # absolute_positioning only
        self.assertEqual([r["bearing"] for r in other], ["front", "right"])
        with self.assertRaises(ValueError):  # a plain run's row: refused
            R.episode_targets({"index": 0, "trajectory": [{"reward": 0.1}]})

    def test_rows_from_the_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            res, _ = write_run(tmp, "run", [item()], 1)
            row = R.list_episodes([res])[0][0]["row"]
        rows = R.episode_targets(row)
        np.testing.assert_allclose([r["G_t"] for r in rows],
                                   [sum(t["reward_exact"] for t in row["trajectory"][k:]) for k in range(3)], rtol=1e-12)
        self.assertGreater(rows[0]["G_t"], 4.9)  # the success bonus on the last turn
        self.assertEqual([r["critic_features"] for r in rows], [feats(0), feats(1), feats(2)])


# ---------------------------------------------------------------------------
# B. folds, probes, metrics, the reading
# ---------------------------------------------------------------------------
class FoldTest(unittest.TestCase):
    def test_grouped_deterministic_balanced(self):
        rng = random.Random(3)
        groups = [f"scene{rng.randrange(23)}" for _ in range(700)]
        f1, f2 = R.assign_folds(groups, 5), R.assign_folds(groups, 5)
        self.assertTrue(np.array_equal(f1, f2))
        by = {}
        for g, f in zip(groups, f1):
            by.setdefault(g, set()).add(int(f))
        self.assertTrue(all(len(v) == 1 for v in by.values()))  # a scene in one fold only
        order = list(range(len(groups)))
        rng.shuffle(order)  # the rows' order does not matter
        shuffled = R.assign_folds([groups[i] for i in order], 5)
        self.assertTrue(np.array_equal(shuffled, f1[order]))
        sizes = np.bincount(f1, minlength=5)
        self.assertLessEqual(sizes.max() - sizes.min(), max(groups.count(g) for g in set(groups)))
        self.assertEqual(set(f1.tolist()), set(range(5)))
        with self.assertRaises(ValueError):
            R.assign_folds(["a", "b", "c"], 5)


def planted(n=600, d=12, scenes=25, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, d)) * rng.uniform(0.5, 30.0, size=d)  # very different scales: the standardisation
    groups = np.array([f"s{i % scenes}" for i in range(n)])
    return x, groups, R.assign_folds(groups, 5), rng


class RidgeTest(unittest.TestCase):
    def test_path_matches_the_closed_form(self):
        rng = np.random.default_rng(1)
        for n, d in ((50, 8), (12, 30)):  # the primal and the dual way
            x, y, xte = rng.normal(size=(n, d)), rng.normal(size=(n, 2)), rng.normal(size=(7, d))
            mu, sd = x.mean(0), x.std(0)
            a, b = (x - mu) / sd, (xte - mu) / sd
            for k, alpha in enumerate((0.1, 10.0)):
                w = np.linalg.solve(a.T @ a + alpha * np.eye(d), a.T @ (y - y.mean(0)))
                got = R.ridge_path(x, y, xte, (0.1, 10.0))[k]
                np.testing.assert_allclose(got, b @ w + y.mean(0), rtol=1e-8, atol=1e-8)

    def test_grid(self):
        self.assertEqual(len(R.ALPHAS), 10)
        self.assertAlmostEqual(min(R.ALPHAS), 1e-2)
        self.assertAlmostEqual(max(R.ALPHAS), 1e7)
        np.testing.assert_allclose(np.array(R.CS) * np.array(R.ALPHAS), 1.0)

    def test_planted_signal_and_noise(self):
        x, groups, folds, rng = planted()
        w = rng.normal(size=x.shape[1]) / x.std(0)
        y = x @ w + 0.2 * rng.normal(size=len(x))
        noise = rng.normal(size=len(x))
        r = R.ridge_cv(x, np.stack([y, noise], 1), folds, groups)
        self.assertGreater(R.explained_variance(y, r["oof"][:, 0]), 0.9)
        self.assertTrue(-0.1 < R.explained_variance(noise, r["oof"][:, 1]) < 0.05)
        self.assertEqual(len(r["alphas"]), 5)
        self.assertTrue(all(a[1] >= a[0] for a in r["alphas"]))  # noise: at least as much penalty
        again = R.ridge_cv(x, np.stack([y, noise], 1), folds, groups)
        self.assertTrue(np.array_equal(again["oof"], r["oof"]))  # deterministic


class LogisticTest(unittest.TestCase):
    def test_binary(self):
        x, groups, folds, rng = planted()
        z = (x / x.std(0)) @ rng.normal(size=x.shape[1]) * 1.5
        y = (rng.uniform(size=len(x)) < 1 / (1 + np.exp(-z))).astype(int)
        r = R.logistic_cv(x, y, 2, folds, groups)
        self.assertGreater(R.auc(y, r["oof"][:, 1]), 0.9)
        self.assertGreater(np.mean(r["oof"].argmax(1) == y), np.mean(r["majority"] == y))
        noise = rng.integers(0, 2, size=len(x))
        rn = R.logistic_cv(x, noise, 2, folds, groups)
        self.assertLess(abs(R.auc(noise, rn["oof"][:, 1]) - 0.5), 0.1)
        np.testing.assert_allclose(r["oof"].sum(1), 1.0)

    def test_multinomial(self):
        x, groups, folds, rng = planted(n=700, d=10)
        y = ((x / x.std(0)) @ rng.normal(size=(x.shape[1], 4)) * 2.0).argmax(1)
        r = R.logistic_cv(x, y, 4, folds, groups)
        rep = R._class_report(y, r, R.BEARINGS)
        self.assertGreater(rep["accuracy"], 0.8)
        self.assertLess(rep["majority_accuracy"], 0.5)
        self.assertEqual(sum(rep["classes"].values()), len(y))

    def test_warm_started_path_is_the_fit(self):
        """The warm-started C path ends where a fit of the last C alone does (a convex problem)."""
        x, groups, folds, rng = planted(n=300, d=6)
        y = (x[:, 0] / x[:, 0].std() + rng.normal(size=len(x)) > 0).astype(int)
        path = R.logistic_path(x, y, 2, x[:20], [1e-3, 1e-2, 1.0])[-1]
        alone = R.logistic_path(x, y, 2, x[:20], [1.0])[0]
        np.testing.assert_allclose(path, alone, atol=1e-5)

    def test_auc(self):
        self.assertEqual(R.auc([0, 0, 1, 1], [0.1, 0.4, 0.35, 0.8]), 0.75)
        self.assertEqual(R.auc([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]), 0.5)  # ties
        self.assertIsNone(R.auc([1, 1], [0.1, 0.2]))
        self.assertEqual(R.explained_variance([1.0, 2.0, 3.0], [1.0, 2.0, 3.0]), 1.0)
        self.assertIsNone(R.explained_variance([2.0, 2.0], [1.0, 3.0]))


class ReadingTest(unittest.TestCase):
    def test_thresholds(self):
        self.assertEqual(R.reading_of(0.3), "linearly readable")
        self.assertEqual(R.reading_of(0.75), "linearly readable")
        self.assertEqual(R.reading_of(0.2999), "weak")
        self.assertEqual(R.reading_of(0.1), "weak")
        self.assertEqual(R.reading_of(0.0999), "not readable")
        self.assertEqual(R.reading_of(-0.4), "not readable")
        self.assertEqual(R.reading_of(None), "undetermined")
        self.assertEqual(R.reading_of(float("nan")), "undetermined")

    def test_grid_edges(self):
        res = {"readings": {"layer9": {"G_t": {"alpha_per_fold": [1e7, 1.0, 1e7, 1.0, 1.0]},
                                       "success": {"C_per_fold": [1e-7, 1.0, 1.0, 1.0, 1.0]},
                                       "seconds": 1.0}},
               "baselines": {"turn_family_ridge": {"alpha_per_fold": [1e-2, 1.0, 1.0, 1.0, 1.0]}}}
        edges = R.grid_edges(res)
        self.assertEqual(len(edges), 3)
        self.assertIn("layer9 G_t: alpha 1e+07 at the grid's edge in folds [0, 2]", edges)
        self.assertTrue(any(e.startswith("layer9 success: C 1e-07") for e in edges))
        self.assertTrue(any(e.startswith("baseline turn_family_ridge G_t: alpha 0.01") for e in edges))

    def test_diagnosis(self):
        good = {"hidden_states_entries": 37, "norm_of_layer36_ok": True,
                "hidden_states_agree": {"layer9": True, "layer18": True, "layer27": True, "final_norm": True}}
        R.check_diagnosis(good)
        for bad in (dict(good, hidden_states_entries=36), dict(good, norm_of_layer36_ok=False),
                    dict(good, hidden_states_agree=dict(good["hidden_states_agree"], layer18=False)),
                    dict(good, hidden_states_error="TypeError: x"), {}):
            with self.assertRaises(SystemExit):
                R.check_diagnosis(bad)

    def test_threads(self):
        with mock.patch.dict(os.environ, {"OMP_NUM_THREADS": "3"}):
            self.assertEqual((R.threads_of(None, 8), R.threads_of(2, 8)), (3, 2))
        with mock.patch.dict(os.environ, {"OMP_NUM_THREADS": "", "MKL_NUM_THREADS": ""}):
            self.assertEqual(R.threads_of(None, 8), 8)


def synthetic_metas(n_episodes=80, scenes=10, seed=0):
    """Turn rows as extract writes them: G_t = 5 * success - 3.33 * phi + small rewards; success from d."""
    rng = random.Random(seed)
    metas = []
    for e in range(n_episodes):
        fam = ("absolute_positioning", "fov_inclusion", "equidistance")[e % 3]
        n = 2 + e % 3
        d0 = rng.randrange(1, 8)
        success = d0 <= 4
        phis = [rng.uniform(0, 1) for _ in range(n)]
        for t in range(n):
            f = {"v": 1, "d": d0, "near": False, "table": True, "bearing": R.BEARINGS[(e + t) % 4],
                 "condition": R.CONDITIONS[(e + t) % 3] if fam == R.AP else None, "family": fam, "phi": phis[t]}
            if e == 3 and t == 1:
                f = None  # a turn without the truth
            g = 5.0 * success - 3.33 * phis[t] + 0.01 * rng.uniform(-1, 1)
            metas.append({"key": f"k{e:03d}", "scene_id": f"scene{e % scenes}", "task_type": fam, "turn": t,
                          "G_t": g, "success": success, "turns_left": n - t, "d": None if f is None else f["d"],
                          "bearing": None if f is None else f["bearing"],
                          "condition": None if f is None else f["condition"], "phi": None if f is None else f["phi"],
                          "critic_features": f})
    return metas


class BaselineTest(unittest.TestCase):
    def test_privileged_mlp_and_turn_family(self):
        metas = [m for m in synthetic_metas(120, 12) if m["critic_features"] is not None]
        groups = np.array([m["scene_id"] for m in metas])
        folds = R.assign_folds(groups, 5)
        b = R.mlp_baseline(metas, folds, 12, epochs=150)
        self.assertGreater(b["ev_G_t"], 0.8)  # success from d, phi in the read-out
        self.assertEqual(len(b["readout_a_b_c_e_per_fold"]), 5)
        self.assertLess(b["readout_a_b_c_e_per_fold"][0][3], -2.0)  # e: about -3.33
        x = R.turn_family_design(metas, 12)
        self.assertEqual(x.shape, (len(metas), 18))
        self.assertTrue(np.array_equal(x.sum(1), np.full(len(metas), 2.0)))
        tf = R.ridge_cv(x, np.array([m["G_t"] for m in metas]), folds, groups)
        self.assertLess(R.explained_variance([m["G_t"] for m in metas], tf["oof"][:, 0]), 0.3)


# ---------------------------------------------------------------------------
# B. extract (stand-in processor and forward) -> fit
# ---------------------------------------------------------------------------
def fake_read(batch, positions):
    """Readings that name their token position (to check the rows' alignment)."""
    p = np.asarray(positions, dtype=np.float64)[:, None]
    return {name: np.concatenate([p, np.sin(p * (k + 1) + np.arange(5)[None])], 1)
            for k, name in enumerate(R.READINGS)}


MANIFEST = {"version": R.VERSION, "model": "/m", "readings": list(R.READINGS), "layers": list(R.LAYERS),
            "max_turns": 12}


class ExtractFitTest(unittest.TestCase):
    def test_shards_resume_and_fit(self):
        items = [item(i, scene=f"scene{i % 7}", family=("absolute_positioning", "fov_inclusion")[i % 2])
                 for i in range(21)]
        with tempfile.TemporaryDirectory() as tmp:
            res, _ = write_run(tmp, "run", items, 21)
            episodes, info = R.list_episodes([res])
            self.assertEqual(len(episodes), 21)
            out = Path(tmp) / "feat"
            for i in range(2):
                shard = R.shard_of(episodes, i, 2)
                sdir = R.shard_dir(out, i, 2)
                first = R.extract_shard(shard, sdir, FakeProcessor(), fake_read, {**MANIFEST, "shard": [i, 2]},
                                        limit=3, chunk_episodes=2, log=lambda s: None)
                self.assertEqual((first["status"], first["counts"]["episodes_written"]), ("partial", 3))
                done = R.extract_shard(shard, sdir, FakeProcessor(), fake_read, {**MANIFEST, "shard": [i, 2]},
                                       chunk_episodes=2, log=lambda s: None)
                self.assertEqual((done["status"], done["counts"]["episodes_written"]), ("done", len(shard)))
                turns = done["counts"]["turns_written"]
                self.assertEqual((done["counts"]["prompt_checked"], done["counts"]["eos_appended"],
                                  done["counts"]["length_cut"], done["counts"]["finish_unknown"]), (turns, turns, 0, 0))
                again = R.extract_shard(shard, sdir, FakeProcessor(), fake_read, {**MANIFEST, "shard": [i, 2]},
                                        log=lambda s: None)  # nothing left to do
                self.assertEqual(again["counts"]["turns_written"], done["counts"]["turns_written"])
            metas, arrays, _ = R.load_features(out)
            self.assertEqual(len({m["key"] for m in metas}), 21)
            self.assertEqual(len(metas), sum(len(e["row"]["trajectory"]) for e in episodes))
            for name in R.READINGS:  # each row is its turn's reading
                self.assertEqual(arrays[name].dtype, np.float32)
                np.testing.assert_allclose(arrays[name][:, 0], [m["position"] for m in metas])
            m0, m1 = metas[0], metas[1]
            self.assertEqual((m0["turn"], m0["position"], m0["prompt_tokens"], m0["response_offset"]),
                             (0, m0["prompt_tokens"] - 1, m0["prompt_tokens"], 0))
            self.assertEqual(m1["eval_prompt_tokens"] - m1["prompt_tokens"], 1)  # the eval layout's "\n" after turn 0
            # another model's shard, or a mismatch: refused
            with self.assertRaises(SystemExit):
                R.extract_shard(R.shard_of(episodes, 0, 2), R.shard_dir(out, 0, 2), FakeProcessor(), fake_read,
                                {**MANIFEST, "model": "/other"})
            bad = dict(episodes[0], key="bad", row=dict(episodes[0]["row"], trajectory=[
                dict(t, prompt_ids_sha256="0" * 64) for t in episodes[0]["row"]["trajectory"]]))
            with self.assertRaises(SystemExit):
                R.extract_shard([bad], Path(tmp) / "x" / "shard_000_of_001", FakeProcessor(), fake_read, MANIFEST)
            skip = R.extract_shard([bad], Path(tmp) / "y" / "shard_000_of_001", FakeProcessor(), fake_read,
                                   {**MANIFEST, "shard": [0, 1]}, on_mismatch="skip")
            self.assertEqual((skip["status"], skip["counts"]["skipped"]), ("done", 1))

    def test_non_finite_values_are_refused(self):
        def nan_read(batch, positions):
            out = fake_read(batch, positions)
            out["layer27"][0, 2] = np.nan
            return out

        with tempfile.TemporaryDirectory() as tmp:
            res, _ = write_run(tmp, "run", [item()], 2)
            episodes, _ = R.list_episodes([res])
            sdir = Path(tmp) / "f" / "shard_000_of_001"
            with self.assertRaises(SystemExit) as ctx:
                R.extract_shard(episodes, sdir, FakeProcessor(), nan_read, {**MANIFEST, "shard": [0, 1]})
            self.assertIn("non-finite", str(ctx.exception))
            self.assertEqual(list(sdir.glob("chunk_*")), [])

    def test_fit_reading(self):
        """fit_all on planted readings: layer36 carries G_t, the others noise; the reading follows the best one."""
        metas = synthetic_metas(100, 10)
        rng = np.random.default_rng(0)
        n = len(metas)
        arrays = {name: rng.normal(size=(n, 6)).astype(np.float32) for name in R.READINGS}
        arrays["layer36"][:, 0] = [m["G_t"] + 0.05 * rng.normal() for m in metas]
        with mock.patch.object(R, "ALPHAS", (0.1, 100.0)), mock.patch.object(R, "CS", (1.0, 0.01)):
            res = R.fit_all(metas, arrays, 12, max_iter=60, mlp_epochs=8)
            again = R.fit_all(metas, arrays, 12, max_iter=60, mlp_epochs=8)
            md = R.results_md(res)

        def timeless(r):
            return json.dumps({**r, "readings": {k: {**v, "seconds": None} for k, v in r["readings"].items()}},
                              sort_keys=True)

        self.assertEqual(timeless(res), timeless(again))  # deterministic
        self.assertEqual((res["best_reading"], res["verdict"]), ("layer36", "linearly readable"))
        self.assertGreater(res["best_ev_G_t"], 0.9)
        self.assertLess(res["readings"]["layer9"]["G_t"]["ev"], 0.1)
        self.assertEqual(res["turns_without_features"], 1)
        self.assertEqual(res["turns_used"], n - 1)
        self.assertEqual(res["grid"], {"ridge_alpha": [0.1, 100.0], "logistic_C": [1.0, 0.01]})
        r36 = res["readings"]["layer36"]
        self.assertEqual(set(r36), {"G_t", "turns_left", "d", "success", "bearing", "condition", "seconds"})
        self.assertEqual(r36["condition"]["n"], sum(1 for m in metas if m["task_type"] == R.AP and m["condition"]))
        self.assertEqual(r36["d"]["n"], n - 1)
        self.assertIn("auc", r36["success"])
        self.assertIn("majority_accuracy", r36["bearing"])
        self.assertIn("privileged_mlp", res["baselines"])
        self.assertIsInstance(res["grid_edges"], list)
        self.assertIn("Reading: linearly readable", md)
        self.assertEqual(md.count("| layer"), 4)
        self.assertIn("grid's edge", md)
        self.assertIn("grouped_gated", md)
        self.assertIn("training concat layout", md)


TOKENIZER = "/mnt/umm/users/yinbaiqiao/models_runs/chains/m_qwen3vl8b_formal_sft_20261026/lf_q3formal/model/checkpoint-1232"


@unittest.skipUnless(Path(TOKENIZER, "tokenizer.json").is_file() and importlib.util.find_spec("transformers")
                     and importlib.util.find_spec("transformers.models.qwen3_vl"),
                     "the C0 SFT checkpoint's processor and transformers >= 4.57 (the Qwen3-VL runtime)")
class RealProcessorTest(unittest.TestCase):
    """The model's own processor (no weights): the training layout with it, its positions at the generation prompt's
    last token, the first prompt equal to the eval layout's, a leading newline kept apart, the end of turn known to
    the tokenizer and the generation config."""

    def test_training_layout_with_the_qwen3vl_processor(self):
        from transformers import AutoProcessor

        proc = AutoProcessor.from_pretrained(TOKENIZER)
        tok = proc.tokenizer
        turns = [(f"<image>\nCurrent camera pose: turn {t}\n", Image.new("RGB", (64, 48), (40 * t, 9, 9)),
                  ("\n" if t == 1 else "") + "<action>move_forward|turn_left|</action>") for t in range(3)]
        layout = R.training_layout(proc, "sys\n", turns)
        ids = layout["input_ids"]
        for p in layout["positions"]:
            self.assertEqual(tok.decode(ids[p - 2: p + 1]), "<|im_start|>assistant\n")
        eval0 = R.encode(proc, R.conversation("sys\n", turns, 0), [turns[0][1]], True)[0]
        self.assertEqual(ids[: layout["prompt0_len"]], eval0)
        p1 = layout["positions"][1]
        self.assertEqual(tok.decode(ids[p1 + 1: p1 + 2]), "\n")  # the leading newline, its own token
        eos = tok.convert_tokens_to_ids(R.END_OF_TURN)
        self.assertEqual(ids[-1], eos)
        gen = json.loads(Path(TOKENIZER, "generation_config.json").read_text())["eos_token_id"]
        self.assertIn(eos, gen if isinstance(gen, list) else [gen])
        batch = R.model_batch(ids, layout["parts"])
        self.assertEqual(int(batch["image_grid_thw"].shape[0]), 3)
        image_pad = tok.convert_tokens_to_ids("<|image_pad|>")
        merge = proc.image_processor.merge_size
        self.assertEqual(ids.count(image_pad), int(batch["image_grid_thw"].prod(1).sum()) // merge ** 2)


if __name__ == "__main__":
    unittest.main()
