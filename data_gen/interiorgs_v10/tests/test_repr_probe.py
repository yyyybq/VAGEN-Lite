"""Gate 0 of docs/PREREG_PPO2.md (the representation probe): evaluation run --critic-probe (the recording mode) and
repr_probe.py (extract's chat rebuild and positions, the targets, the folds, the probes, the baselines, the reading).
CPU only, no model: a Qwen-like stand-in processor; the real processor's test runs only where its runtime is."""

import argparse
import dataclasses
import hashlib
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
from unittest import mock

import numpy as np
from PIL import Image

from data_gen.interiorgs_v10 import evaluation as E
from data_gen.interiorgs_v10 import repr_probe as R

MODEL_KEYS = {"name", "model", "version", "prompt_format", "greedy", "samples", "temperature", "top_p", "sample_seed",
              "env_switches", "max_turns", "history", "sft_text", "entropy", "load_seconds"}  # model.json, plain run
PROBE_TURN_KEYS = {"reward_exact", "critic_features", "critic_features_error", "prompt_ids_sha256"}
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
        self.cfg, self.items, self.turns, self.fail_pose = cfg, items or [item()], turns, fail_pose
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


class FakeProcessor:
    """A Qwen-like chat template and processor: special tokens whole, a run of newlines one token (as Qwen's
    pre-tokeniser merges "\\n\\n"), words and single characters; each image placeholder becomes (w * h) // 4 image
    tokens of its frame; token ids a fixed hash of the token (the same in every instance)."""
    SPECIAL = ("<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|image_pad|>", "<|vision_end|>")

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert not tokenize
        out = []
        for m in messages:
            c = m["content"]
            body = c if isinstance(c, str) else "".join(
                "<|vision_start|><|image_pad|><|vision_end|>" if p.get("type") == "image" else p["text"] for p in c)
            out.append(f"<|im_start|>{m['role']}\n{body}<|im_end|>\n")
        return "".join(out) + ("<|im_start|>assistant\n" if add_generation_prompt else "")

    def tokens(self, text):
        return re.findall("|".join(re.escape(s) for s in self.SPECIAL) + r"|\n+|\w+|[^\w\n]", text)

    @staticmethod
    def tid(token):
        return int(hashlib.sha256(token.encode()).hexdigest()[:8], 16) % 100000

    def __call__(self, text, images=None, return_tensors=None):
        (t,) = text
        images, toks, k = list(images or []), [], 0
        for tok in self.tokens(t):
            if tok == "<|image_pad|>":
                toks += [tok] * ((images[k].size[0] * images[k].size[1]) // 4)
                k += 1
            else:
                toks.append(tok)
        assert k == len(images), (k, len(images))
        return {"input_ids": [[self.tid(x) for x in toks]], "images": images}


class FakePolicy:
    """vLLM's chat in miniature: the chat of chat_messages through the processor, the prompt's ids kept."""

    def __init__(self, *a, answers=None, **k):  # VllmPolicy(model, ...) in cmd_run
        self.answers = answers or ["<action>move_forward|</action>"]
        self.proc, self.calls = FakeProcessor(), []
        self.last_prompt_tokens = self.last_prompt_ids = None

    def generate(self, system, user_text, image, earlier=(), as_sft=False, seed=None):
        messages = E.chat_messages(system, earlier, user_text, image, R.image_placeholder, as_sft)
        images = [im for _, im, _ in earlier] + [image]
        text = self.proc.apply_chat_template(messages, add_generation_prompt=True)
        self.last_prompt_ids = self.proc(text=[text], images=images)["input_ids"][0]
        self.last_prompt_tokens = len(self.last_prompt_ids)
        self.calls.append({"messages": messages, "images": [np.asarray(im).copy() for im in images], "seed": seed,
                           "ids": list(self.last_prompt_ids)})
        return self.answers[(len(self.calls) - 1) % len(self.answers)]


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
        env, policy = ProbeEnv(features=True, fail_pose=2), FakePolicy()
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


@dataclasses.dataclass
class Cfg:
    jsonl_path: str = ""
    prompt_format: str = ""
    critic_features: bool = False
    critic_feature_tables_dir: str = None
    potential_field_reward_scale: float = 1.0
    progress_version: str = "product"


class CmdRunProbeTest(unittest.TestCase):
    def run_cmd(self, tmp, name, opts, items=None):
        envs = []

        def build(cfg):
            envs.append(ProbeEnv(cfg, items=items))
            return envs[-1]

        env_mod = types.ModuleType("vagen.envs.active_spatial.env")
        env_mod.ActiveSpatialEnv = build
        cfg_mod = types.ModuleType("vagen.envs.active_spatial.env_config")
        cfg_mod.ActiveSpatialEnvConfig = Cfg
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
            tables = Path(tmp) / "tables"
            tables.mkdir()
            m0, plain, env0 = self.run_cmd(tmp, "plain", ["--temperature", "0.8", "--top-p", "0.92"])
            self.assertEqual(set(m0), MODEL_KEYS)
            self.assertEqual(m0["env_switches"], {})
            self.assertFalse(env0.cfg.critic_features)
            self.assertFalse(PROBE_ROW_KEYS & set(plain[0]))
            m1, probe, env1 = self.run_cmd(tmp, "probe", ["--temperature", "0.8", "--top-p", "0.92",
                                                          "--critic-probe", str(tables)])
            self.assertEqual(set(m1), MODEL_KEYS | {"critic_probe"})
            self.assertEqual(m1["critic_probe"], E.CRITIC_PROBE_VERSION)
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
            m1.pop("critic_probe")
            self.assertEqual(m0, m1)  # model.json: only the probe's keys differ

    def test_refusals_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            tables, other = Path(tmp) / "tables", Path(tmp) / "tables2"
            tables.mkdir(), other.mkdir()
            self.run_cmd(tmp, "plain", [])
            self.run_cmd(tmp, "probe", ["--critic-probe", str(tables)])
            for name, opts, why in (
                    ("plain", ["--critic-probe", str(tables)], "critic_probe None"),  # a plain folder stays plain
                    ("probe", [], "critic_probe 'critic_probe_v1'"),  # a probe folder stays a probe folder
                    ("probe", ["--critic-probe", str(other)], "critic_feature_tables_dir"),
                    ("x", ["--critic-probe", str(tables), "--grounding-probe"], "not with --grounding"),
                    ("x", ["--critic-probe", str(tables), "--grounding-side"], "not with --grounding"),
                    ("x", ["--critic-probe", str(tables), "--no-history"], "needs --history"),
                    ("x", ["--critic-probe", str(Path(tmp) / "missing")], "not a directory")):
                with self.subTest(name=name, opts=opts), self.assertRaises(SystemExit) as ctx:
                    self.run_cmd(tmp, name, opts)
                self.assertIn(why, str(ctx.exception))
            self.assertFalse((Path(tmp) / "x" / "model.json").exists())  # refused before anything is written
            self.assertFalse((Path(tmp) / "x" / "results.jsonl").exists())
            m, rows, _ = self.run_cmd(tmp, "probe", ["--critic-probe", str(tables)])  # the same probe resumes
            self.assertEqual(len(rows), 1)


# ---------------------------------------------------------------------------
# B. repr_probe: the chat, the positions, the targets
# ---------------------------------------------------------------------------
def write_run(root, name, items, n_episodes, answers=None, model="/m", turns=3):
    """A finished critic-probe result folder from the stand-in env and policy (cmd_run's files)."""
    out = Path(root) / name
    env, policy = ProbeEnv(features=True, items=items, turns=turns), FakePolicy(answers=answers)
    rows = [E.run_episode(env, policy, i, 12, frames=out / "frames", history=True, sample=0, seed=7,
                          critic_probe=True) for i in range(n_episodes)]
    rows.append({"index": n_episodes, "sample": 0, "error": "RuntimeError: render"})  # an infrastructure error
    (out / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (out / "model.json").write_text(json.dumps({"name": name, "model": model, "history": True, "sft_text": False,
                                                "max_turns": 12, "critic_probe": E.CRITIC_PROBE_VERSION,
                                                "env_switches": E.critic_probe_overrides("/tables")}))
    (out / "done.json").write_text("{}")
    return out, policy


class ChatAndPositionTest(unittest.TestCase):
    def episode(self, tmp, answers=None):
        res, policy = write_run(tmp, "run", [item()], 1, answers)
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
            sft = R.conversation(" sys\n\n\n", turns, 1, as_sft=True)
            self.assertEqual(sft[0]["content"], "sys")  # sft_text as the run's chat_messages applies it
            with self.assertRaises(ValueError):
                R.conversation("s", turns, len(turns) + 1)

    def test_positions(self):
        with tempfile.TemporaryDirectory() as tmp:
            ep, policy = self.episode(tmp)
            inp = R.episode_inputs(ep, FakeProcessor())
        full = inp["batch"]["input_ids"][0]
        self.assertEqual(inp["positions"], [n - 1 for n in inp["prompt_lens"]])
        self.assertEqual(inp["prompt_lens"], [len(call["ids"]) for call in policy.calls])  # the policy's prompts
        for call, p in zip(policy.calls, inp["positions"]):
            self.assertEqual(full[: p + 1], call["ids"])
        tid = FakeProcessor.tid
        for p in inp["positions"]:  # the last token of the generation prompt; the answer starts right after
            self.assertEqual(full[p - 2: p + 2], [tid("<|im_start|>"), tid("assistant"), tid("\n"), tid("<")])
        self.assertEqual((inp["prompt_checked"], inp["prompt_unchecked"]), (3, 0))
        self.assertEqual(len(inp["batch"]["images"]), 3)
        self.assertEqual(inp["seq_len"], len(full))

    def test_prefix_mismatch_is_refused(self):
        """An answer that starts with a newline: in the episode's sequence it merges with the generation prompt's
        newline, so the turn's prompt is not the first tokens of the sequence."""
        with tempfile.TemporaryDirectory() as tmp:
            ep, _ = self.episode(tmp, answers=["<action>turn_left|</action>", "\n<action>move_forward|</action>"])
            with self.assertRaises(R.Mismatch) as ctx:
                R.episode_inputs(ep, FakeProcessor())
        self.assertIn("turn 1", str(ctx.exception))
        with self.assertRaises(R.Mismatch):
            R.turn_positions([1, 2, 3], [[1, 2, 3]])  # no answer token after the prompt
        with self.assertRaises(R.Mismatch):
            R.turn_positions([1, 2, 3, 4, 5], [[1, 2], [1]])  # not growing
        self.assertEqual(R.turn_positions([1, 2, 3, 4, 5], [[1, 2], [1, 2, 3, 4]]), [1, 3])

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
            for change, why in ((dict(critic_probe=None), "critic_probe"), (dict(history=False), "history")):
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


class ExtractFitTest(unittest.TestCase):
    def test_shards_resume_and_fit(self):
        items = [item(i, scene=f"scene{i % 7}", family=("absolute_positioning", "fov_inclusion")[i % 2])
                 for i in range(21)]
        with tempfile.TemporaryDirectory() as tmp:
            res, _ = write_run(tmp, "run", items, 21)
            episodes, info = R.list_episodes([res])
            self.assertEqual(len(episodes), 21)
            out = Path(tmp) / "feat"
            man = {"version": R.VERSION, "model": "/m", "readings": list(R.READINGS), "layers": list(R.LAYERS),
                   "max_turns": 12}
            for i in range(2):
                shard = R.shard_of(episodes, i, 2)
                sdir = R.shard_dir(out, i, 2)
                first = R.extract_shard(shard, sdir, FakeProcessor(), fake_read, {**man, "shard": [i, 2]}, limit=3,
                                        chunk_episodes=2, log=lambda s: None)
                self.assertEqual((first["status"], first["counts"]["episodes_written"]), ("partial", 3))
                done = R.extract_shard(shard, sdir, FakeProcessor(), fake_read, {**man, "shard": [i, 2]},
                                       chunk_episodes=2, log=lambda s: None)
                self.assertEqual((done["status"], done["counts"]["episodes_written"]), ("done", len(shard)))
                again = R.extract_shard(shard, sdir, FakeProcessor(), fake_read, {**man, "shard": [i, 2]},
                                        log=lambda s: None)  # nothing left to do
                self.assertEqual(again["counts"]["turns_written"], done["counts"]["turns_written"])
            metas, arrays, _ = R.load_features(out)
            self.assertEqual(len({m["key"] for m in metas}), 21)
            self.assertEqual(len(metas), sum(len(e["row"]["trajectory"]) for e in episodes))
            for name in R.READINGS:  # each row is its turn's reading
                self.assertEqual(arrays[name].dtype, np.float32)
                np.testing.assert_allclose(arrays[name][:, 0], [m["position"] for m in metas])
            m0 = metas[0]
            self.assertEqual((m0["turn"], m0["position"], m0["prompt_tokens"]), (0, m0["prompt_tokens"] - 1,
                                                                                    m0["prompt_tokens"]))
            # a shard missing, or a mismatch: refused
            with self.assertRaises(SystemExit):
                R.extract_shard(R.shard_of(episodes, 0, 2), R.shard_dir(out, 0, 2), FakeProcessor(), fake_read,
                                {**man, "model": "/other"})
            bad = dict(episodes[0], key="bad", row=dict(episodes[0]["row"], trajectory=[
                dict(t, prompt_ids_sha256="0" * 64) for t in episodes[0]["row"]["trajectory"]]))
            with self.assertRaises(SystemExit):
                R.extract_shard([bad], Path(tmp) / "x" / "shard_000_of_001", FakeProcessor(), fake_read, man)
            skip = R.extract_shard([bad], Path(tmp) / "y" / "shard_000_of_001", FakeProcessor(), fake_read,
                                   {**man, "shard": [0, 1]}, on_mismatch="skip")
            self.assertEqual((skip["status"], skip["counts"]["skipped"]), ("done", 1))

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
        md = R.results_md(res)
        self.assertIn("Reading: linearly readable", md)
        self.assertEqual(md.count("| layer"), 4)


TOKENIZER = "/mnt/umm/users/yinbaiqiao/models_runs/chains/m_qwen3vl8b_formal_sft_20261026/lf_q3formal/model/checkpoint-1232"


@unittest.skipUnless(Path(TOKENIZER, "tokenizer.json").is_file() and importlib.util.find_spec("transformers")
                     and importlib.util.find_spec("transformers.models.qwen3_vl"),
                     "the C0 SFT checkpoint's processor and transformers >= 4.57 (the Qwen3-VL runtime)")
class RealProcessorTest(unittest.TestCase):
    """The model's own processor (no weights): each turn's prompt is the first tokens of the episode's sequence, the
    position is the generation prompt's last token, and an answer starting with a newline is refused."""

    def test_positions_with_the_qwen3vl_processor(self):
        from transformers import AutoProcessor

        proc = AutoProcessor.from_pretrained(TOKENIZER)
        turns = [(f"<image>\nCurrent camera pose: turn {t}\n", Image.new("RGB", (64, 48), (40 * t, 9, 9)),
                  "<action>move_forward|turn_left|</action>") for t in range(3)]
        full, batch = R.encode(proc, R.conversation("sys\n", turns, 3), [im for _, im, _ in turns], False)
        prompts = [R.encode(proc, R.conversation("sys\n", turns, t), [im for _, im, _ in turns[: t + 1]], True)[0]
                   for t in range(3)]
        pos = R.turn_positions(full, prompts)
        tok = proc.tokenizer
        for p in pos:
            self.assertEqual(tok.decode(full[p - 2: p + 1]), "<|im_start|>assistant\n")
        self.assertEqual(int(batch["image_grid_thw"].shape[0]), 3)
        bad = [(u, im, "\n" + a if t == 0 else a) for t, (u, im, a) in enumerate(turns)]
        full2, _ = R.encode(proc, R.conversation("sys\n", bad, 3), [im for _, im, _ in bad], False)
        p0 = R.encode(proc, R.conversation("sys\n", bad, 0), [bad[0][1]], True)[0]
        with self.assertRaises(R.Mismatch):
            R.turn_positions(full2, [p0])


if __name__ == "__main__":
    unittest.main()
