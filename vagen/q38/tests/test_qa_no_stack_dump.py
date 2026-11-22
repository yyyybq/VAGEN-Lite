"""q38 (2026-11-22, user OK): plan "qa_eval" "no_stack_dump" and qa_eval easi --no-stack-dump.

C0 e5's one-card mmsi_bench and 3dsrbench children died of SIGSEGV during the 15-min faulthandler dump (4 of 4 runs,
the dump cut off mid-frame); the base model's mmsi_bench ran without the thread and finished.  The switch turns the
thread off for the benchmarks a plan names.  Off unless given, so every plan without the key and every qa_eval run
without the flag must run exactly as at c2c6ed47 (git show; Q38_BEFORE_GIT names another checkout of the branch when
this tree has no .git), with sco and the subprocesses replaced by fakes:

- train_chain.Chain.qa_jobs and eval_jobs: the same jobs, cards, pools and job script arguments for plans without
  "no_stack_dump" (or with an empty list); with it, only the named benchmarks' jobs change, by a trailing
  --no-stack-dump (a job with more cards keeps them); bad values stop the chain (ChainError);
- qa_eval easi without --no-stack-dump: the same command, working folder, full environment and done.json as before;
  with it: no _stack_dump_ folder, PYTHONPATH as the job's own, everything else as before, done.json with
  "stack_dump": false.

    PYTHONPATH=.:verl python -B -m unittest vagen.q38.tests.test_qa_no_stack_dump
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

REPO = Path(__file__).resolve().parents[3]
BEFORE = "c2c6ed47"  # the qwen38 branch's HEAD before this change (qwen38-mx starts there)


def _git_source(path: str):
    for repo in (REPO, os.environ.get("Q38_BEFORE_GIT")):
        if not repo:
            continue
        try:
            return subprocess.run(["git", "-C", str(repo), "show", f"{BEFORE}:{path}"], capture_output=True,
                                  text=True, check=True).stdout
        except (OSError, subprocess.CalledProcessError):
            continue
    return None


def _before(path: str, name: str, package: str = ""):
    src = _git_source(path)
    if src is None:
        return None
    mod = types.ModuleType(name)
    mod.__file__ = str(REPO / path)
    mod.__package__ = package
    sys.modules[name] = mod
    try:
        exec(compile(src, name, "exec"), mod.__dict__)
    finally:
        sys.modules.pop(name, None)
    return mod


def _models(root: Path):
    for m in ("a", "b"):
        (root / m).mkdir(exist_ok=True)
        (root / m / "config.json").write_text("{}")
    return {"a": (str(root / "a"), None, []), "b": (str(root / "b"), None, ["--max-turns", "9"])}


class ChainJobs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from data_gen.interiorgs_v10 import train_chain as new

        cls.new = new
        cls.old = _before("data_gen/interiorgs_v10/train_chain.py", "data_gen.interiorgs_v10._train_chain_before_nsd",
                          "data_gen.interiorgs_v10")

    @staticmethod
    def _base(root: Path):
        return {"tr": str(root / "tr"), "sft_dir": str(root), "eval_only": True,
                "eval": {"models": {"a": {"model": str(root / "a")}, "b": {"model": str(root / "b")}}}}

    def _plans(self, root: Path):
        base, qa = self._base(root), {"out": str(root / "qa")}
        yield {**base, "qa_eval": {**qa, "easi": ["vsi_bench", "site"]}}
        yield {**base, "model_type": "qwen3_vl", "qa_eval": {**qa, "easi": "easi_8", "limit": 5}}
        yield {**base, "model_type": "qwen3_5", "eval_cards": 2,
               "qa_eval": {**qa, "easi": "easi_8", "p0": "S", "p0_res": 512, "probe": "P", "probe_models": ["b"],
                           "cards": {"site_video": 2}, "probe_variants": {"probe1024": {"max_tokens": 1024}}}}
        yield {**base, "model_type": "qwen3_5", "qa_eval": {**qa, "easi": ["mmsi_bench", "3dsrbench"],
                                                            "easi_models": ["a"]}}
        yield {**base, "model_type": "qwen3_5", "qa_eval": None}

    def _run_eval_jobs(self, mod, plan, jobs, root):
        from data_gen.interiorgs_v10 import job_sco

        chain = mod.Chain(plan, sleep=lambda s: None)
        seen, pools = {}, []
        chain.job_pool = lambda key=None, cards=8: pools.append(cards)

        def prepare(run, gpus, args, job_dir="job", pool=None):
            seen[run.name] = (gpus, list(args))
            (run / job_dir).mkdir(parents=True, exist_ok=True)

        def submit(run, job_dir="job"):
            (run / job_dir / "submission.json").write_text(json.dumps({"job_id": "pt-" + run.name}))

        def status(run, job_dir="job"):
            j = jobs[run.name]
            (j[0] / j[1]).mkdir(parents=True, exist_ok=True)
            (j[0] / j[1] / "done.json").write_text("{}")
            return {"state": "SUCCEEDED", "finished": True}

        with mock.patch.multiple(job_sco, prepare=prepare, submit=submit, status=status, delete=lambda *a, **k: None):
            chain.eval_jobs("evaluate", jobs)
        return seen, sorted(pools)

    def _clear(self, root: Path):
        for d in ("tr", "qa", "ev"):
            shutil.rmtree(root / d, ignore_errors=True)

    def test_plans_without_the_key_unchanged(self):
        if self.old is None:
            self.skipTest(f"git or {BEFORE} not available")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models = _models(root)
            for plan in self._plans(root):
                for p in (plan, *([{**plan, "qa_eval": {**plan["qa_eval"], "no_stack_dump": []}}]
                                  if plan.get("qa_eval") else [])):
                    new_jobs = self.new.Chain(p, sleep=lambda s: None).qa_jobs(models)
                    old_jobs = self.old.Chain(plan, sleep=lambda s: None).qa_jobs(models)
                    self.assertEqual(new_jobs, old_jobs, p)
                    self.assertFalse(any("--no-stack-dump" in j[4] for j in new_jobs.values()))
                    jobs = {**new_jobs, "val__a": (root / "ev", "a", str(root / "a"), None, [])}
                    got = []
                    for mod, pl in ((self.new, p), (self.old, plan)):
                        self._clear(root)
                        got.append(self._run_eval_jobs(mod, pl, jobs, root))
                    self.assertEqual(got[0], got[1], p)
                    self.assertEqual(len(got[0][0]), len(jobs))

    def test_only_the_named_benchmarks_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models = _models(root)
            plan = {**self._base(root), "model_type": "qwen3_5", "eval_cards": 2,
                    "qa_eval": {"out": str(root / "qa"), "easi": "easi_8", "p0": "S", "probe": "P",
                                "cards": {"site_video": 2}}}
            plain = self.new.Chain(plan, sleep=lambda s: None).qa_jobs(models)
            plan["qa_eval"]["no_stack_dump"] = ["mmsi_bench", "3dsrbench", "site_video", "mmsi_bench"]
            got = self.new.Chain(plan, sleep=lambda s: None).qa_jobs(models)
            self.assertEqual(set(got), set(plain))
            changed = {k for k in got if got[k] != plain[k]}
            self.assertEqual(changed, {f"qa_{b}__{n}" for b in ("mmsi_bench", "3dsrbench", "site_video")
                                       for n in ("a", "b")})
            for k in changed:
                self.assertEqual(got[k], (*plain[k][:4], [*plain[k][4], "--no-stack-dump"], *plain[k][5:]), k)
            self.assertEqual(got["qa_site_video__a"][4][-3:], ["--cards", "2", "--no-stack-dump"])
            self.assertEqual(got["qa_site_video__a"][5:], ("qa", 2))  # the job keeps its two cards
            self.assertEqual(got["qa_mmsi_bench__a"][5:], ("qa",))
            seen, pools = self._run_eval_jobs(self.new, plan, got, root)
            self.assertEqual({n for n, (_, args) in seen.items() if "--no-stack-dump" in args}, changed)
            self.assertEqual(seen["qa_mmsi_bench__a"][1].count("--no-stack-dump"), 1)
            self.assertEqual({n for n, (g, _) in seen.items() if g == 2}, {"qa_site_video__a", "qa_site_video__b"})
            self._clear(root)
            seen_plain, pools_plain = self._run_eval_jobs(self.new, {**plan, "qa_eval": {
                k: v for k, v in plan["qa_eval"].items() if k != "no_stack_dump"}}, plain, root)
            self.assertEqual(pools, pools_plain)
            for n in seen:  # the arguments differ only by the flag
                self.assertEqual([x for x in seen[n][1] if x != "--no-stack-dump"], seen_plain[n][1], n)

    def test_bad_values_stop_the_chain(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models = _models(root)
            for easi, given in ((["mmsi_bench"], "mmsi_bench"), (["mmsi_bench"], ["3dsrbench"]),
                                ("easi_8", ["site"]), (["mmsi_bench"], [1]), (["mmsi_bench"], {"mmsi_bench": True})):
                plan = {**self._base(root), "model_type": "qwen3_5",
                        "qa_eval": {"out": str(root / "qa"), "easi": easi, "no_stack_dump": given}}
                with self.assertRaises(self.new.ChainError, msg=repr(given)):
                    self.new.Chain(plan, sleep=lambda s: None).qa_jobs(models)


class QaEvalNoStackDump(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from data_gen.interiorgs_v10 import qa_eval as new

        cls.new = new
        cls.old = _before("data_gen/interiorgs_v10/qa_eval.py", "_qa_eval_before_nsd")

    def _run(self, mod, benchmark, model_type, root: Path, *extra):
        out = root / "o"
        shutil.rmtree(out, ignore_errors=True)
        seen = []
        key = "site" if benchmark.startswith("site") else benchmark

        def fake_run(cmd, cwd=None, env=None):
            seen.append((list(cmd), cwd, dict(env or {})))
            res = out / "m" / "easi_results.json"
            res.parent.mkdir(parents=True, exist_ok=True)
            res.write_text(json.dumps({"scores": {key: 0.5}, "subScores": {key: {"x": 1}}}))
            return types.SimpleNamespace(returncode=1 if key != benchmark else 0)

        argv = ["qa_eval", "easi", "--name", "m", "--model", str(root), "--out", str(out), "--benchmark", benchmark,
                "--model-type", model_type, *extra]
        with mock.patch.object(mod.subprocess, "run", side_effect=fake_run), mock.patch.object(sys, "argv", argv), \
                mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0", "PYTHONPATH": "/job/own/path"}):
            os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
            rc = mod.main()
        done = out / "m" / "done.json"
        dumps = sorted(p.name for p in (out / "m").glob("_stack_dump_*"))
        return rc, seen, done.read_text() if done.is_file() else None, dumps

    CASES = (("mmsi_bench", "qwen3_5"), ("3dsrbench", "qwen3_5"), ("blink", "qwen3_5"), ("site_video", "qwen3_5"),
             ("site_image", "qwen3_vl"), ("mmsi_bench", "qwen3_vl"), ("vsi_bench", "qwen2_5_vl"))

    def test_without_the_flag_unchanged(self):
        if self.old is None:
            self.skipTest(f"git or {BEFORE} not available")
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for b, kind in self.CASES:
                for extra in ([], ["--cards", "2"], ["--limit", "3"]):
                    self.assertEqual(self._run(self.new, b, kind, root, *extra),
                                     self._run(self.old, b, kind, root, *extra), (b, kind, extra))

    def test_the_flag_turns_the_dump_off(self):
        if self.old is None:
            self.skipTest(f"git or {BEFORE} not available")
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for b, kind in self.CASES:
                for extra in ([], ["--cards", "2"]):
                    rc_n, seen_n, done_n, dumps_n = self._run(self.new, b, kind, root, *extra, "--no-stack-dump")
                    rc_o, seen_o, done_o, dumps_o = self._run(self.old, b, kind, root, *extra)
                    (cmd_n, cwd_n, env_n), (cmd_o, cwd_o, env_o) = seen_n[0], seen_o[0]
                    self.assertEqual((rc_n, cmd_n, cwd_n), (rc_o, cmd_o, cwd_o), (b, kind, extra))
                    self.assertEqual(dumps_n, [], (b, kind))
                    self.assertEqual(env_n["PYTHONPATH"], "/job/own/path", (b, kind))
                    dumped = b.startswith("site") or kind == "qwen3_5"
                    self.assertEqual(len(dumps_o), 1 if dumped else 0, (b, kind))
                    if dumped:
                        self.assertTrue(env_o["PYTHONPATH"].endswith(os.pathsep + "/job/own/path"))
                    self.assertEqual({k: v for k, v in env_n.items() if k != "PYTHONPATH"},
                                     {k: v for k, v in env_o.items() if k != "PYTHONPATH"}, (b, kind, extra))
                    self.assertEqual(json.loads(done_n), {**json.loads(done_o), "stack_dump": False}, (b, kind))


if __name__ == "__main__":
    unittest.main()
