"""q38 base zero-shot (2026-11-20): the QA evaluations of the 27B model (qwen3_5) through the training chain's QA jobs.

Changed for the q38 type only -- every other model type must run exactly as before, so each check below compares the
new code with the q38 branch's a264ae46 (loaded from git), with vLLM, the processor and the subprocesses replaced by
fakes:

- data_gen/interiorgs_qa/evaluate.py (P0 in-domain QA): the LLM(...) arguments, the chat-template call, the video
  input (with metadata) and the sampling parameters; for qwen3_5 enable_thinking False, generation_config "vllm",
  the video metadata and --gpu-memory-utilization; enforce_eager (2026-11-20: vLLM 0.28's CUDA-graph check refused
  the 27B hybrid model, max_num_seqs 1024 > 307 Mamba cache blocks), and no max_num_seqs;
- data_gen/interiorgs_v10/qa_eval.py: the easi and p0 commands and environments; the new probe command;
- scripts/easi_eval.py: the EASI runner's command (qwen3_5: lmms-eval's qwen3_vl class with enable_thinking=False);
- vagen/q38/qa_probe.py (the q38 copy of eval/diagnostics/qa_probe.py): the run path's vLLM calls against the
  original (VAGEN-Lite-data, when it is on this machine with the pinned sha256), and the scoring against the 8B base
  model's saved probe answers (they must be read and scored exactly as stored).

    PYTHONPATH=. python -B -m unittest vagen.q38.tests.test_qa_q38
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[3]
BEFORE = "a264ae46"  # the q38 branch's HEAD before these changes
ORIGINAL_PROBE = Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite-data/eval/diagnostics/qa_probe.py")
ORIGINAL_PROBE_SHA = "d70bbb61b29de8271e1ae715fa7b2ce55f2e7423d2d62857c61bc39cbfb3c680"
BASE_PROBE_8B = Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite-data/exps/diagnostics_20261101/probe_v3g_q3")
P1_PROBE = Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite-data/exps/transition_qa_20261019b/p1_probe.jsonl")


def _git_source(path: str):
    try:
        return subprocess.run(["git", "-C", str(REPO), "show", f"{BEFORE}:{path}"], capture_output=True, text=True,
                              check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None


BEFORE_MAX_TOKENS = "17c6894a"  # the q38 branch's HEAD before probe --max-tokens (the exploratory probe variant)


def _before_max_tokens(path: str, package: str = ""):
    """``path`` at BEFORE_MAX_TOKENS as a module, from this tree's git or Q38_BEFORE_GIT (another checkout of the
    branch); None when neither has the commit."""
    for repo in (REPO, os.environ.get("Q38_BEFORE_GIT")):
        if not repo:
            continue
        try:
            src = subprocess.run(["git", "-C", str(repo), "show", f"{BEFORE_MAX_TOKENS}:{path}"], capture_output=True,
                                 text=True, check=True).stdout
        except (OSError, subprocess.CalledProcessError):
            continue
        return _module_from_source(src, f"_before_mt_{path.replace('/', '_')[:-3]}", REPO / path, package)
    return None


def _module_from_source(src: str, name: str, file: Path, package: str = ""):
    mod = types.ModuleType(name)
    mod.__file__ = str(file)
    mod.__package__ = package
    sys.modules[name] = mod
    try:
        exec(compile(src, f"{name}", "exec"), mod.__dict__)
    finally:
        sys.modules.pop(name, None)
    return mod


def _before(path: str, package: str = ""):
    src = _git_source(path)
    if src is None:
        return None
    return _module_from_source(src, f"_before_{path.replace('/', '_')[:-3]}", REPO / path, package)


def _model_dir(root, model_type):
    d = Path(root) / (model_type or "no_type")
    d.mkdir(exist_ok=True)
    (d / "config.json").write_text(json.dumps({"model_type": model_type} if model_type else {}))
    return str(d)


class _Rec:
    """A fake vllm + transformers.AutoProcessor that records every call."""

    def __init__(self):
        self.calls = []

    def modules(self):
        rec = self

        class LLM:
            def __init__(self, **kw):
                rec.calls.append(("LLM", kw))

            def generate(self, batch, params):
                rec.calls.append(("generate", [self._mm(b) for b in batch], params))
                return [types.SimpleNamespace(outputs=[types.SimpleNamespace(text="A")], prompt_token_ids=[1, 2])
                        for _ in batch]

            def chat(self, messages, sampling_params=None, use_tqdm=True, **kw):
                rec.calls.append(("chat", len(messages), sampling_params, kw))
                return [types.SimpleNamespace(outputs=[types.SimpleNamespace(text="A")]) for _ in messages]

            @staticmethod
            def _mm(b):
                out = {"prompt": b["prompt"]}
                for k, v in b["multi_modal_data"].items():
                    if k == "video":
                        out["video"] = ("with_metadata", v[1]) if isinstance(v, tuple) else ("frames", len(v))
                    else:
                        out[k] = len(v)
                return out

        def SamplingParams(**kw):
            rec.calls.append(("SamplingParams", kw))
            return ("params", tuple(sorted(kw.items())))

        class AutoProcessor:
            @staticmethod
            def from_pretrained(model):
                return AutoProcessor()

            def apply_chat_template(self, conv, **kw):
                rec.calls.append(("apply_chat_template", kw))
                return "PROMPT"

        vllm = types.ModuleType("vllm")
        vllm.LLM, vllm.SamplingParams = LLM, SamplingParams
        tr = types.ModuleType("transformers")
        tr.AutoProcessor = AutoProcessor
        return {"vllm": vllm, "transformers": tr}


def _frames(root, n=2):
    from PIL import Image

    files = []
    for i in range(n):
        f = Path(root) / f"f{i}.png"
        Image.new("RGB", (8, 8), (i, 0, 0)).save(f)
        files.append(str(f))
    return files


class EvaluateP0(unittest.TestCase):
    """data_gen/interiorgs_qa/evaluate.py run_vllm."""

    @classmethod
    def setUpClass(cls):
        from data_gen.interiorgs_qa import evaluate as new

        cls.new = new
        cls.old = _before("data_gen/interiorgs_qa/evaluate.py", "data_gen.interiorgs_qa")

    def _calls(self, mod, model, gpu=None, kw=True):
        rec = _Rec()
        with tempfile.TemporaryDirectory() as d:
            files = _frames(d)
            reqs = [{"media": "video", "files": files, "text": "q1"}, {"media": "image", "files": files[:1], "text": "q2"}]
            with mock.patch.dict(sys.modules, rec.modules()):
                if kw:
                    mod.run_vllm(model, reqs, 12288, 512, 2.0, gpu)
                else:
                    mod.run_vllm(model, reqs, 12288, 512, 2.0)
        return rec.calls

    def test_other_types_unchanged(self):
        if self.old is None:
            self.skipTest(f"git or {BEFORE} not available")
        with tempfile.TemporaryDirectory() as root:
            for t in ("qwen3_vl", "qwen2_5_vl", None):
                m = _model_dir(root, t)
                self.assertEqual(self._calls(self.new, m), self._calls(self.old, m, kw=False), t)

    def test_other_types_no_eager(self):
        """The eager switch is the q38 type's only (the 8B P0 ran with CUDA graphs, vLLM 0.11, 2026-11-03)."""
        with tempfile.TemporaryDirectory() as root:
            for t in ("qwen3_vl", "qwen2_5_vl", None):
                llm = [c[1] for c in self._calls(self.new, _model_dir(root, t)) if c[0] == "LLM"][0]
                self.assertEqual(sorted(llm), ["gpu_memory_utilization", "limit_mm_per_prompt", "max_model_len",
                                               "model", "seed"], t)
                self.assertEqual(self.new.q38_record(t or "", None), {}, t)

    def test_qwen3_5(self):
        with tempfile.TemporaryDirectory() as root:
            m = _model_dir(root, "qwen3_5")
            calls = self._calls(self.new, m, 0.9)
            llm = [c[1] for c in calls if c[0] == "LLM"][0]
            self.assertEqual(llm["gpu_memory_utilization"], 0.9)
            self.assertEqual(llm["generation_config"], "vllm")
            self.assertIs(llm["enforce_eager"], True)  # 2026-11-20: CUDA graphs refused (Mamba cache blocks)
            self.assertNotIn("max_num_seqs", llm)  # vLLM's default, as every other evaluator of the line
            chats = [c[1] for c in calls if c[0] == "apply_chat_template"]
            self.assertEqual(len(chats), 2)
            self.assertTrue(all(c["enable_thinking"] is False for c in chats))
            batch = [c[1] for c in calls if c[0] == "generate"][0]
            self.assertEqual(batch[0]["video"][0], "with_metadata")
            self.assertEqual(batch[0]["video"][1]["frames_indices"], [0, 1])
            default = [c[1] for c in self._calls(self.new, m) if c[0] == "LLM"][0]
            self.assertEqual(default["gpu_memory_utilization"], 0.85)
            self.assertEqual([c[1] for c in calls if c[0] == "SamplingParams"], [{"temperature": 0.0, "max_tokens": 512}])

    def test_record(self):
        self.assertEqual(self.new.q38_record("qwen3_vl", None), {})
        self.assertEqual(self.new.q38_record("", None), {})
        r = self.new.q38_record("qwen3_5", 0.9)
        self.assertEqual(r["chat_template_kwargs"], {"enable_thinking": False})
        self.assertEqual(r["llm_kwargs"], {"generation_config": "vllm", "enforce_eager": True})
        self.assertEqual(r["gpu_memory_utilization"], 0.9)


class QaEval(unittest.TestCase):
    """data_gen/interiorgs_v10/qa_eval.py: the commands it runs."""

    @classmethod
    def setUpClass(cls):
        from data_gen.interiorgs_v10 import qa_eval as new

        cls.new = new
        cls.old = _before("data_gen/interiorgs_v10/qa_eval.py")

    def _run(self, mod, argv, write=None):
        seen = []

        def fake_run(cmd, cwd=None, env=None):
            seen.append((list(cmd), cwd, {k: v for k, v in (env or {}).items() if k in (
                "PYTHONPATH", "HF_HUB_OFFLINE", "HF_HOME", "HF_DATASETS_CACHE", "TOKENIZERS_PARALLELISM", "NO_PROXY")}))
            if write:
                write(cmd)
            return types.SimpleNamespace(returncode=0)

        with mock.patch.object(mod.subprocess, "run", side_effect=fake_run), mock.patch.object(sys, "argv", argv), \
                mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}):  # qa_eval easi reads gpus[0]
            rc = mod.main()
        return rc, seen

    def _norm(self, seen, out):
        # the accelerate wrapper folder carries the process id
        return [([c for c in cmd], cwd, env) for cmd, cwd, env in seen]

    def test_easi_and_p0_unchanged(self):
        if self.old is None:
            self.skipTest(f"git or {BEFORE} not available")
        with tempfile.TemporaryDirectory() as d:
            for argv in (["qa_eval", "easi", "--name", "m", "--model", d, "--out", f"{d}/o1", "--benchmark", "blink",
                          "--model-type", "qwen3_vl"],
                         ["qa_eval", "easi", "--name", "m", "--model", d, "--out", f"{d}/o2", "--benchmark",
                          "site_video", "--model-type", "qwen2_5_vl", "--limit", "4"],
                         ["qa_eval", "p0", "--name", "m", "--model", d, "--out", f"{d}/o3", "--selection", "S",
                          "--model-type", "qwen3_vl"],
                         ["qa_eval", "p0", "--name", "m", "--model", d, "--out", f"{d}/o4", "--selection", "S",
                          "--res", "512", "--limit", "8"]):
                _, a = self._run(self.new, argv)
                _, b = self._run(self.old, argv)
                self.assertEqual(a, b, argv)

    def test_qwen3_5_commands(self):
        with tempfile.TemporaryDirectory() as d:
            _, seen = self._run(self.new, ["qa_eval", "p0", "--name", "m", "--model", d, "--out", f"{d}/p0",
                                           "--selection", "S", "--model-type", "qwen3_5"])
            self.assertEqual(seen[0][0][-2:], ["--gpu-memory-utilization", "0.9"])
            _, seen = self._run(self.new, ["qa_eval", "easi", "--name", "m", "--model", d, "--out", f"{d}/e",
                                           "--benchmark", "blink", "--model-type", "qwen3_5"])
            self.assertIn(f"m:{d}:qwen3_5", seen[0][0])
            expect_py = self.new.Q38_EASI_PYTHON if Path(self.new.Q38_EASI_PYTHON).is_file() else sys.executable
            self.assertEqual(seen[0][0][0], expect_py)

    def test_probe(self):
        with tempfile.TemporaryDirectory() as d:
            probe = Path(d) / "probe.jsonl"
            probe.write_text("".join(json.dumps({"sample_id": f"s{i}"}) + "\n" for i in range(5)))
            out = Path(d) / "probe_out"

            def write(cmd):  # qa_probe writes OUT/NAME/NAME.json
                (out / "m" / "m.json").write_text(json.dumps({"all": {"accuracy": 0.5}}))

            rc, seen = self._run(self.new, ["qa_eval", "probe", "--name", "m", "--model", d, "--out", str(out),
                                            "--probe", str(probe), "--model-type", "qwen3_5", "--limit", "3"], write)
            self.assertEqual(rc, 0)
            cmd = seen[0][0]
            self.assertEqual(cmd[1:4], ["-m", "vagen.q38.qa_probe", "run"])
            self.assertEqual(cmd[cmd.index("--model") + 1], f"m={d}")
            self.assertEqual(cmd[-2:], ["--gpu-memory-utilization", "0.9"])
            small = Path(cmd[cmd.index("--probe") + 1])
            self.assertEqual(len(small.read_text().splitlines()), 3)
            done = json.loads((out / "m" / "done.json").read_text())
            self.assertEqual((done["benchmark"], done["limit"], done["accuracy"]), ("probe", 3, 0.5))
            # without --limit the probe itself; another type gets no memory option
            rc, seen = self._run(self.new, ["qa_eval", "probe", "--name", "n", "--model", d, "--out", str(out),
                                            "--probe", str(probe), "--model-type", "qwen3_vl"])
            cmd = seen[0][0]
            self.assertEqual(cmd[cmd.index("--probe") + 1], str(probe))
            self.assertNotIn("--gpu-memory-utilization", cmd)
            self.assertEqual(rc, 1)  # no n.json written: not done
            self.assertFalse((out / "n" / "done.json").exists())

    def _probe_full(self, mod, argv, out: Path):
        """qa_eval probe with qa_probe faked: (rc, [(command, folder, full environment)], done.json text)."""
        seen = []

        def fake_run(cmd, cwd=None, env=None):
            seen.append((list(cmd), cwd, dict(env or {})))
            (out / "m" / "m.json").write_text(json.dumps({"all": {"accuracy": 0.5}}))
            return types.SimpleNamespace(returncode=0)

        with mock.patch.object(mod.subprocess, "run", side_effect=fake_run), mock.patch.object(sys, "argv", argv), \
                mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}):
            rc = mod.main()
        done = out / "m" / "done.json"
        return rc, seen, done.read_text() if done.is_file() else None

    def test_probe_without_max_tokens_unchanged(self):
        """q38 (2026-11-20): probe without --max-tokens -- the same command, folder, environment and done.json bytes as
        17c6894a's qa_eval."""
        import shutil

        old = _before_max_tokens("data_gen/interiorgs_v10/qa_eval.py")
        if old is None:
            self.skipTest(f"git or {BEFORE_MAX_TOKENS} not available (set Q38_BEFORE_GIT)")
        with tempfile.TemporaryDirectory() as d:
            probe = Path(d) / "probe.jsonl"
            probe.write_text("".join(json.dumps({"sample_id": f"s{i}"}) + "\n" for i in range(5)))
            out = Path(d) / "probe_out"
            for kind in ("qwen3_5", "qwen3_vl"):
                for extra in ([], ["--limit", "3"]):
                    argv = ["qa_eval", "probe", "--name", "m", "--model", d, "--out", str(out), "--probe", str(probe),
                            "--model-type", kind, *extra]
                    got = []
                    for mod in (self.new, old):
                        shutil.rmtree(out, ignore_errors=True)
                        got.append(self._probe_full(mod, argv, out))
                    self.assertEqual(got[0], got[1], argv)
                    self.assertNotIn("--max-tokens", got[0][1][0][0])
                    self.assertNotIn("max_tokens", json.loads(got[0][2]))

    def test_probe_max_tokens(self):
        """probe --max-tokens 1024: the same command and --max-tokens 1024; done.json with "max_tokens": 1024."""
        with tempfile.TemporaryDirectory() as d:
            probe = Path(d) / "probe.jsonl"
            probe.write_text(json.dumps({"sample_id": "s0"}) + "\n")
            out = Path(d) / "probe_out"
            argv = ["qa_eval", "probe", "--name", "m", "--model", d, "--out", str(out), "--probe", str(probe),
                    "--model-type", "qwen3_5"]
            rc0, plain, done0 = self._probe_full(self.new, argv, out)
            rc1, more, done1 = self._probe_full(self.new, argv + ["--max-tokens", "1024"], out)
            self.assertEqual((rc0, rc1), (0, 0))
            self.assertEqual(more[0][0], plain[0][0] + ["--max-tokens", "1024"])
            self.assertEqual(more[0][1:], plain[0][1:])
            self.assertEqual(json.loads(done1), {**json.loads(done0), "max_tokens": 1024})


class EasiEval(unittest.TestCase):
    """scripts/easi_eval.py: the EASI runner's command."""

    @classmethod
    def setUpClass(cls):
        path = REPO / "scripts/easi_eval.py"
        cls.new = _module_from_source(path.read_text(), "_easi_eval_new", path)
        cls.old = _before("scripts/easi_eval.py")

    def _cmd(self, mod, kind, root):
        easi = Path(root) / "easi"
        (easi / "scripts/submissions").mkdir(parents=True, exist_ok=True)
        (easi / "scripts/submissions/run_easi_eval.py").write_text("")
        reg = Path(root) / "reg.yaml"
        reg.write_text(f"easi_dir: {easi}\nlmms_eval_dir: {root}/lmms\ncheckpoints: {{}}\n")
        ckpt = Path(root) / "ckpt"
        ckpt.mkdir(exist_ok=True)
        seen = []

        def fake_run(cmd, cwd=None, env=None):
            seen.append(list(cmd))
            return types.SimpleNamespace(returncode=0)

        argv = ["easi_eval.py", "--registry", str(reg), "--ckpts", "", "--add", f"m:{ckpt}:{kind}", "--benchmarks",
                "blink", "--output_dir", f"{root}/out", "--gpu", "0", "--nproc", "1", "--no-summary"]
        with mock.patch.object(mod.subprocess, "run", side_effect=fake_run), mock.patch.object(sys, "argv", argv):
            mod.main()
        return seen[0]

    def test_other_types_unchanged(self):
        if self.old is None:
            self.skipTest(f"git or {BEFORE} not available")
        with tempfile.TemporaryDirectory() as d:
            for kind in ("qwen3_vl", "qwen2_5_vl", "cambrians"):
                self.assertEqual(self._cmd(self.new, kind, d), self._cmd(self.old, kind, d), kind)

    def test_qwen3_5(self):
        with tempfile.TemporaryDirectory() as d:
            cmd = self._cmd(self.new, "qwen3_5", d)
            self.assertEqual(cmd[cmd.index("--model") + 1], "qwen3_vl")
            self.assertEqual(cmd[cmd.index("--model-args") + 1], f"pretrained={d}/ckpt,enable_thinking=False")


class QaProbe(unittest.TestCase):
    """vagen/q38/qa_probe.py against the original and the 8B base model's saved answers."""

    @classmethod
    def setUpClass(cls):
        from vagen.q38 import qa_probe as new

        cls.new = new
        cls.old = None
        if ORIGINAL_PROBE.is_file() and hashlib.sha256(ORIGINAL_PROBE.read_bytes()).hexdigest() == ORIGINAL_PROBE_SHA:
            cls.old = _module_from_source(ORIGINAL_PROBE.read_text(), "_qa_probe_original", ORIGINAL_PROBE)

    def test_same_functions_as_original(self):
        """Everything but the run path is the original's text."""
        if not ORIGINAL_PROBE.is_file():
            self.skipTest("the original is not on this machine")
        import ast

        def defs(src):
            tree = ast.parse(src)
            return {n.name: ast.dump(n) for n in tree.body if isinstance(n, (ast.FunctionDef, ast.ClassDef))}

        a, b = defs((REPO / "vagen/q38/qa_probe.py").read_text()), defs(ORIGINAL_PROBE.read_text())
        changed = {n for n in b if a.get(n) != b[n]}
        self.assertEqual(changed, {"run_one", "cmd_run", "main"})
        self.assertEqual(set(a) - set(b), {"model_type_of", "q38_record"})

    def _run_one(self, mod, model, gpu=None, new=True, **kw):
        rec = _Rec()
        with tempfile.TemporaryDirectory() as d:
            probe = Path(d) / "p.jsonl"
            img = _frames(d, 1)[0]
            row = {"sample_id": "s1", "qtype": "goal_now", "qclass": "state", "family": "x", "images": [img],
                   "messages": [{"role": "user", "content": "<image>\nIs it? Answer Yes or No."},
                                {"role": "assistant", "content": "Yes"}]}
            probe.write_text(json.dumps(row) + "\n")
            with mock.patch.dict(sys.modules, rec.modules()):
                if new:
                    mod.run_one("m", model, probe, Path(d), False, gpu, **kw)
                else:
                    mod.run_one("m", model, probe, Path(d), False)
            meta = json.loads((Path(d) / "m.json").read_text())
        assert meta["probe"] == str(probe)
        return rec.calls, {k: v for k, v in meta.items() if k not in ("load_seconds", "answer_seconds", "probe")}

    def test_run_other_types_unchanged(self):
        if self.old is None:
            self.skipTest("the original (pinned sha256) is not on this machine")
        with tempfile.TemporaryDirectory() as root:
            for t in ("qwen3_vl", "qwen2_5_vl", None):
                m = _model_dir(root, t)
                self.assertEqual(self._run_one(self.new, m), self._run_one(self.old, m, new=False), t)

    def test_run_qwen3_5(self):
        with tempfile.TemporaryDirectory() as root:
            m = _model_dir(root, "qwen3_5")
            calls, meta = self._run_one(self.new, m, 0.9)
            llm = [c[1] for c in calls if c[0] == "LLM"][0]
            self.assertEqual((llm["gpu_memory_utilization"], llm["generation_config"], llm["max_model_len"]),
                             (0.9, "vllm", 4096))
            chat = [c for c in calls if c[0] == "chat"][0]
            self.assertEqual(chat[3], {"chat_template_kwargs": {"enable_thinking": False}})
            self.assertEqual(meta["model_type"], "qwen3_5")
            self.assertEqual(meta["gpu_memory_utilization"], 0.9)

    def test_cmd_run_subprocess(self):
        seen = []
        args = argparse.Namespace(out=tempfile.mkdtemp(), model=["m=/x"], from_eval="", names="", probe="p.jsonl",
                                  no_images=False, in_process=False, gpu_memory_utilization=0.9)
        with mock.patch.object(self.new.subprocess, "run",
                               side_effect=lambda cmd, env=None: seen.append(cmd) or types.SimpleNamespace(returncode=0)):
            self.new.cmd_run(args)
        self.assertEqual(seen[0][1:3], ["-m", "vagen.q38.qa_probe"])
        self.assertEqual(seen[0][-2:], ["--gpu-memory-utilization", "0.9"])

    def test_run_max_tokens(self):
        """q38 (2026-11-20): run_one without max_tokens samples at most 256 new tokens and writes OUT/<name>.json
        exactly as 17c6894a's (the same calls, keys and key order); max_tokens 1024 samples 1,024 and records
        "max_tokens"; 256 given explicitly is the default."""
        old = _before_max_tokens("vagen/q38/qa_probe.py")
        with tempfile.TemporaryDirectory() as root:
            for t, gpu in (("qwen3_5", 0.9), ("qwen3_5", None), ("qwen3_vl", None), (None, None)):
                m = _model_dir(root, t)
                calls, meta = self._run_one(self.new, m, gpu)
                self.assertEqual([c[1] for c in calls if c[0] == "SamplingParams"],
                                 [{"temperature": 0.0, "max_tokens": 256}], t)
                self.assertNotIn("max_tokens", meta)
                self.assertEqual(self.new.MAX_TOKENS, 256)
                if old is not None:
                    calls_old, meta_old = self._run_one(old, m, gpu)
                    self.assertEqual(calls, calls_old, t)
                    self.assertEqual(list(meta.items()), list(meta_old.items()), t)
                self.assertEqual(self._run_one(self.new, m, gpu, max_tokens=256), (calls, meta), t)
                calls2, meta2 = self._run_one(self.new, m, gpu, max_tokens=1024)
                self.assertEqual([c[1] for c in calls2 if c[0] == "SamplingParams"],
                                 [{"temperature": 0.0, "max_tokens": 1024}], t)
                self.assertEqual([c for c in calls2 if c[0] != "SamplingParams"],
                                 [c if c[0] != "chat" else (c[0], c[1], ("params", (("max_tokens", 1024),
                                                                                    ("temperature", 0.0))), c[3])
                                  for c in calls if c[0] != "SamplingParams"], t)
                self.assertEqual(meta2, {**meta, "max_tokens": 1024}, t)
            if old is None:
                self.skipTest(f"git or {BEFORE_MAX_TOKENS} not available (set Q38_BEFORE_GIT): compared without it")

    def test_cmd_run_max_tokens(self):
        """cmd_run passes --max-tokens to the per-model process only when it is not 256, and to run_one in process."""
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)

        def sub_cmd(**kw):
            seen = []
            args = argparse.Namespace(out=f"{tmp.name}/sub", model=["m=/x"], from_eval="", names="", probe="p.jsonl",
                                      no_images=False, in_process=False, gpu_memory_utilization=0.9, **kw)
            with mock.patch.object(self.new.subprocess, "run", side_effect=lambda cmd, env=None: seen.append(cmd)
                                   or types.SimpleNamespace(returncode=0)):
                self.new.cmd_run(args)
            return seen[0]

        plain = sub_cmd()
        self.assertEqual(sub_cmd(max_tokens=256), plain)
        self.assertEqual(sub_cmd(max_tokens=1024), plain + ["--max-tokens", "1024"])
        with self.assertRaises(SystemExit):
            sub_cmd(max_tokens=0)
        for i, (kw, want) in enumerate((({}, 256), ({"max_tokens": 1024}, 1024))):
            ran = []
            args = argparse.Namespace(out=f"{tmp.name}/in{i}", model=["m=/x"], from_eval="", names="", probe="p.jsonl",
                                      no_images=False, in_process=True, gpu_memory_utilization=None, **kw)

            def fake_run_one(name, model, probe, out_dir, blind, gpu, max_tokens=None):
                ran.append(max_tokens)
                (out_dir / f"{name}.json").write_text(json.dumps({"all": {}}))

            with mock.patch.object(self.new, "run_one", side_effect=fake_run_one):
                self.new.cmd_run(args)
            self.assertEqual(ran, [want])
        parsed = []
        with mock.patch.object(self.new, "cmd_run", side_effect=parsed.append):
            self.new.main(["run", "--probe", "p", "--out", "o"])
            self.new.main(["run", "--probe", "p", "--out", "o", "--max-tokens", "1024"])
        self.assertEqual([a.max_tokens for a in parsed], [256, 1024])

    def test_scores_8b_base_answers_as_stored(self):
        """The 8B base model's probe answers (2026-11-01) are read and scored exactly as stored."""
        jl, js = BASE_PROBE_8B / "base_qwen3.jsonl", BASE_PROBE_8B / "base_qwen3.json"
        if not (jl.is_file() and js.is_file() and P1_PROBE.is_file()):
            self.skipTest("the 8B base probe results are not on this machine")
        rows = {r["sample_id"]: r for r in map(json.loads, P1_PROBE.read_text().splitlines()) if r}
        old = [json.loads(l) for l in jl.read_text().splitlines() if l.strip()]
        self.assertEqual(len(old), 838)
        answers = self.new.score([rows[a["sample_id"]] for a in old], [a["output"] for a in old])
        # the stored fields (the file predates the later additions for pointing answers, which add fields only)
        self.assertEqual([{k: a.get(k) for k in o} for a, o in zip(answers, old)], old)
        stored = json.loads(js.read_text())["all"]
        now = self.new.summary(answers)["all"]
        self.assertEqual({k: now.get(k) for k in stored}, stored)


if __name__ == "__main__":
    unittest.main()
