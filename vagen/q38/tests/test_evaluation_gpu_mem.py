"""q38 smoke Z3: evaluation.py's --gpu-memory-utilization (DESIGN_ASYNC 5.4).

The 27B model's bf16 weights (55.6 GB) do not fit in VllmPolicy's default 0.6 of an 80 GB card, so its evaluation gives
vLLM a larger share.  Without the option -- every run before it, the 8B main line's above all -- the run must be exactly
as before: run's call of VllmPolicy and model.json are checked against the evaluation.py of 19b939bd (the q38 branch
before the option), loaded from git, with the env, the render service and vLLM replaced by fakes.
"""

import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from data_gen.interiorgs_v10 import evaluation as E

REPO = Path(__file__).resolve().parents[3]
BEFORE = "19b939bd"  # the q38 branch's HEAD before --gpu-memory-utilization
EVAL_PY = "data_gen/interiorgs_v10/evaluation.py"


class _Stop(Exception):
    pass


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


def _model_dir(root, model_type):
    d = Path(root) / (model_type or "no_config")
    d.mkdir(exist_ok=True)
    if model_type:
        (d / "config.json").write_text(json.dumps({"model_type": model_type}))
    return str(d)


def _parse(mod, argv):
    """The namespace mod.main gives cmd_run for argv (cmd_run replaced, nothing runs)."""
    seen = []
    with mock.patch.object(mod, "cmd_run", side_effect=seen.append):
        mod.main(argv)
    (args,) = seen
    return args


def _env_modules():
    env = types.ModuleType("vagen.envs.active_spatial.env")
    env.ActiveSpatialEnv = lambda cfg: ("ENV", cfg)
    cfg = types.ModuleType("vagen.envs.active_spatial.env_config")

    class ActiveSpatialEnvConfig:
        __dataclass_fields__ = {}

        def __init__(self, **kw):
            self.kw = kw

    cfg.ActiveSpatialEnvConfig = ActiveSpatialEnvConfig
    return {env.__name__: env, cfg.__name__: cfg}


def _policy_call(mod, argv):
    """(args, kwargs) of cmd_run's VllmPolicy call for argv: the env and vLLM are fakes, the run stops right there."""
    calls = []

    def fake_policy(*a, **kw):
        calls.append((a, kw))
        raise _Stop

    args = _parse(mod, argv)
    with mock.patch.dict(sys.modules, _env_modules()), \
            mock.patch.object(mod, "env_config", lambda *a, **kw: {"prompt_format": "no_think"}), \
            mock.patch.object(mod, "VllmPolicy", fake_policy):
        try:
            mod.cmd_run(args)
        except _Stop:
            pass
    (call,) = calls
    return call


def _argv(ev, model, *extra):
    return ["run", "--eval", str(ev), "--name", "q38_gpu_mem_test", "--model", model,
            "--format-example", "action_names", *extra]


class GpuMemoryOptionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.ev = self.root / "eval"
        self.ev.mkdir()
        (self.ev / "tasks.jsonl").write_text(json.dumps({"task_id": "t0"}) + "\n")

    def tearDown(self):
        self.tmp.cleanup()

    def test_default_is_none(self):
        args = _parse(E, _argv(self.ev, _model_dir(self.root, "qwen3_vl")))
        self.assertIsNone(args.gpu_memory_utilization)
        self.assertEqual(E.vllm_memory_kwargs(args), {})
        self.assertEqual(E.vllm_memory_kwargs(types.SimpleNamespace()), {})  # a namespace built without the option

    def test_bounds(self):
        for bad in (0.0, -0.1, 0.96, 1.0):
            with self.assertRaises(SystemExit, msg=bad):
                E.vllm_memory_kwargs(types.SimpleNamespace(gpu_memory_utilization=bad))
        for good in (0.05, 0.6, 0.9, 0.95):
            self.assertEqual(E.vllm_memory_kwargs(types.SimpleNamespace(gpu_memory_utilization=good)),
                             {"gpu_memory_utilization": good})

    def test_bad_value_stops_before_anything_is_written(self):
        args = _parse(E, _argv(self.ev, _model_dir(self.root, "qwen3_5"), "--gpu-memory-utilization", "1.2"))
        with mock.patch.dict(sys.modules, _env_modules()), self.assertRaises(SystemExit):
            E.cmd_run(args)
        self.assertFalse((self.ev / "q38_gpu_mem_test").exists())

    def test_option_reaches_the_policy_and_the_llm(self):
        model = _model_dir(self.root, "qwen3_5")
        (pos, kw) = _policy_call(E, _argv(self.ev, model, "--gpu-memory-utilization", "0.9"))
        self.assertEqual(pos, (model,))
        self.assertEqual(kw, {"max_model_len": 16384, "max_images": 12, "temperature": 0.0, "top_p": 1.0,
                              "entropy": False, "gpu_memory_utilization": 0.9})
        log = []

        class LLM:
            def __init__(self, **k):
                log.append(k)

        with mock.patch.dict(sys.modules, {"vllm": types.SimpleNamespace(LLM=LLM)}):
            E.VllmPolicy(model, **kw)
        (init,) = log
        self.assertEqual(init["gpu_memory_utilization"], 0.9)
        self.assertEqual(init["max_model_len"], 16384)
        self.assertEqual(init["generation_config"], "vllm")

    def test_model_json_records_it_before_the_probe_keys(self):
        model = _model_dir(self.root, "qwen3_5")
        args = _parse(E, _argv(self.ev, model, "--gpu-memory-utilization", "0.9"))
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1"}):
            rec = E.model_record(args, {"prompt_format": "no_think"}, {"prompt_format_example": "action_names"},
                                 False, {"grounding_probe": "v2"}, 12.5)
        self.assertEqual(list(rec)[-7:], ["load_seconds", "model_type", "chat_template_kwargs", "llm_kwargs",
                                          "cuda_visible_devices", "gpu_memory_utilization", "grounding_probe"])
        self.assertEqual(rec["gpu_memory_utilization"], 0.9)

    def test_without_the_option_same_as_before(self):
        """The 8B path word for word: run's VllmPolicy call and model.json (keys in order) equal those of BEFORE's
        evaluation.py, for qwen3_vl, qwen2_5_vl, a folder without config.json and qwen3_5."""
        old = _before_module()
        if old is None:
            self.skipTest(f"git show {BEFORE}:{EVAL_PY} is not available")
        self.assertFalse(hasattr(old, "vllm_memory_kwargs"))  # really the version before the option
        for kind in ("qwen3_vl", "qwen2_5_vl", None, "qwen3_5"):
            model = _model_dir(self.root, kind)
            for extra in ((), ("--no-history",), ("--temperature", "0.8", "--samples", "2")):
                argv = _argv(self.ev, model, *extra)
                new_call, old_call = _policy_call(E, argv), _policy_call(old, argv)
                self.assertEqual(new_call, old_call, (kind, extra))
                self.assertNotIn("gpu_memory_utilization", new_call[1])
                new_args, old_args = _parse(E, argv), _parse(old, argv)
                for probe in ({}, {"grounding_probe": "v2"}, {"critic_probe": "critic_probe_v1",
                                                               "critic_probe_tables": 3}):
                    with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1"}):
                        new_rec = E.model_record(new_args, {"prompt_format": "no_think"}, {}, False, probe, 1.0)
                        old_rec = old.model_record(old_args, {"prompt_format": "no_think"}, {}, False, probe, 1.0)
                    self.assertEqual(json.dumps(new_rec), json.dumps(old_rec), (kind, extra, probe))
        # the one new parser option is the only difference of the run parser
        new_args = vars(_parse(E, _argv(self.ev, _model_dir(self.root, "qwen3_vl"))))
        old_args = vars(_parse(old, _argv(self.ev, _model_dir(self.root, "qwen3_vl"))))
        self.assertEqual(set(new_args) - set(old_args), {"gpu_memory_utilization"})
        self.assertEqual({k: v for k, v in new_args.items() if k != "gpu_memory_utilization"}, old_args)


if __name__ == "__main__":
    unittest.main()
