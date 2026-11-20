"""q38 (2026-11-20): QA jobs with more than one card (plan "qa_eval" "cards", the 27B model's SITE video OOMed on one).

Off unless given, so every plan without the key must run exactly as before: each check compares the new code with the
q38 branch's 6a163462 (git show; Q38_BEFORE_GIT names another checkout of the branch when this tree has no .git), with
sco and the subprocesses replaced by fakes:

- train_chain.Chain.qa_jobs and eval_jobs: the same jobs, cards, pools and job script arguments for plans without
  "cards" (easi lists, easi_8, p0, probe, limit; model types qwen2_5_vl, qwen3_vl, qwen3_5; eval_cards 1 and 2);
- qa_eval easi without --cards: the same command, working folder, full environment and done.json as before; with
  --cards 2: the same command, CUDA_VISIBLE_DEVICES 0,1, PYTORCH_CUDA_ALLOC_CONF expandable_segments:True (unless
  set), done.json with "cards": 2;
- scripts/easi_eval.py (unchanged) hands the EASI runner the CUDA_VISIBLE_DEVICES it was given.

    PYTHONPATH=.:verl python -B -m unittest vagen.q38.tests.test_qa_cards
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

REPO = Path(__file__).resolve().parents[3]
BEFORE = "6a163462"  # the q38 branch's HEAD before this change


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


def _module_from_source(src: str, name: str, file: Path, package: str = ""):
    mod = types.ModuleType(name)
    mod.__file__ = str(file)
    mod.__package__ = package
    sys.modules[name] = mod
    try:
        exec(compile(src, name, "exec"), mod.__dict__)
    finally:
        sys.modules.pop(name, None)
    return mod


def _before(path: str, name: str, package: str = ""):
    src = _git_source(path)
    return None if src is None else _module_from_source(src, name, REPO / path, package)


class ChainJobs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from data_gen.interiorgs_v10 import train_chain as new

        cls.new = new
        cls.old = _before("data_gen/interiorgs_v10/train_chain.py", "data_gen.interiorgs_v10._train_chain_before",
                          "data_gen.interiorgs_v10")

    def _plans(self, root: Path):
        base = {"tr": str(root / "tr"), "sft_dir": str(root), "eval_only": True,
                "eval": {"models": {"a": {"model": str(root / "a")}, "b": {"model": str(root / "b")}}}}
        qa = {"out": str(root / "qa")}
        yield {**base, "qa_eval": {**qa, "easi": ["vsi_bench", "site"]}}
        yield {**base, "model_type": "qwen3_vl", "qa_eval": {**qa, "easi": "easi_8", "limit": 5}}
        yield {**base, "model_type": "qwen3_5", "eval_cards": 2,
               "qa_eval": {**qa, "easi": "easi_8", "p0": "S", "p0_res": 512, "probe": "P", "probe_models": ["b"]}}
        yield {**base, "model_type": "qwen3_5", "qa_eval": {**qa, "easi": ["site_video"], "easi_models": ["a"]}}
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

    def test_plans_without_cards_unchanged(self):
        if self.old is None:
            self.skipTest(f"git or {BEFORE} not available")
        import shutil

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for m in ("a", "b"):
                (root / m).mkdir()
                (root / m / "config.json").write_text("{}")
            models = {"a": (str(root / "a"), None, []), "b": (str(root / "b"), None, ["--max-turns", "9"])}
            for plan in self._plans(root):
                new_jobs = self.new.Chain(plan, sleep=lambda s: None).qa_jobs(models)
                old_jobs = self.old.Chain(plan, sleep=lambda s: None).qa_jobs(models)
                self.assertEqual(new_jobs, old_jobs, plan)
                self.assertTrue(all(len(j) == 6 for j in new_jobs.values()))
                jobs = {**new_jobs, "val__a": (root / "ev", "a", str(root / "a"), None, [])}
                got = []
                for mod in (self.new, self.old):
                    shutil.rmtree(root / "tr", ignore_errors=True)
                    shutil.rmtree(root / "qa", ignore_errors=True)
                    shutil.rmtree(root / "ev", ignore_errors=True)
                    got.append(self._run_eval_jobs(mod, plan, jobs, root))
                self.assertEqual(got[0], got[1], plan)
                self.assertEqual(len(got[0][0]), len(jobs))
                self.assertTrue(all(g == 1 for n, (g, _) in got[0][0].items() if n != "val__a"))

    def test_cards_only_where_given(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for m in ("a", "b"):
                (root / m).mkdir()
                (root / m / "config.json").write_text("{}")
            models = {"a": (str(root / "a"), None, []), "b": (str(root / "b"), None, [])}
            plan = {"tr": str(root / "tr"), "sft_dir": str(root), "eval_only": True, "model_type": "qwen3_5",
                    "eval_cards": 2, "eval": {"models": {"a": {"model": str(root / "a")},
                                                         "b": {"model": str(root / "b")}}},
                    "qa_eval": {"out": str(root / "qa"), "easi": "easi_8", "p0": "S"}}
            plain = self.new.Chain(plan, sleep=lambda s: None).qa_jobs(models)
            plan["qa_eval"]["cards"] = {"site_video": 2}
            got = self.new.Chain(plan, sleep=lambda s: None).qa_jobs(models)
            changed = {k for k in got if got[k] != plain[k]}
            self.assertEqual(changed, {"qa_site_video__a", "qa_site_video__b"})
            for k in changed:
                self.assertEqual(got[k], (*plain[k][:4], [*plain[k][4], "--cards", "2"], "qa", 2))
            seen, pools = self._run_eval_jobs(self.new, plan, got, root)
            self.assertEqual({n for n, (g, _) in seen.items() if g == 2}, changed)
            self.assertEqual(seen["qa_site_video__a"][1][-2:], ["--cards", "2"])
            self.assertEqual(pools.count(2), 2)


class QaEvalCards(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from data_gen.interiorgs_v10 import qa_eval as new

        cls.new = new
        cls.old = _before("data_gen/interiorgs_v10/qa_eval.py", "_qa_eval_before")

    def _run(self, mod, argv, out: Path, environ=None):
        seen = []

        def fake_run(cmd, cwd=None, env=None):
            seen.append((list(cmd), cwd, dict(env or {})))
            res = out / "m" / "easi_results.json"
            res.parent.mkdir(parents=True, exist_ok=True)
            res.write_text(json.dumps({"scores": {"site": 0.41}, "subScores": {"site": {"x": 1}}}))
            return types.SimpleNamespace(returncode=1)  # a SITE part: easi_eval calls it incomplete, as on the cluster

        env = {"CUDA_VISIBLE_DEVICES": "0", **(environ or {})}
        with mock.patch.object(mod.subprocess, "run", side_effect=fake_run), mock.patch.object(sys, "argv", argv), \
                mock.patch.dict(os.environ, env):
            if "PYTORCH_CUDA_ALLOC_CONF" not in env:
                os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
            rc = mod.main()
        done = out / "m" / "done.json"
        return rc, seen, done.read_text() if done.is_file() else None

    def _argv(self, d, out, *extra):
        return ["qa_eval", "easi", "--name", "m", "--model", d, "--out", str(out), "--benchmark", "site_video",
                "--model-type", "qwen3_5", *extra]

    def test_without_cards_unchanged(self):
        if self.old is None:
            self.skipTest(f"git or {BEFORE} not available")
        import shutil

        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "o"
            for extra in ([], ["--limit", "4"], ["--cards", "1"]):
                got = []
                for mod in (self.new, self.old):
                    shutil.rmtree(out, ignore_errors=True)
                    argv = self._argv(d, out, *[x for x in extra if mod is self.new or extra[:1] != ["--cards"]])
                    got.append(self._run(mod, argv, out))
                self.assertEqual(got[0], got[1], extra)  # command, folder, full environment, done.json bytes
                self.assertNotIn("PYTORCH_CUDA_ALLOC_CONF", got[0][1][0][2])
                self.assertEqual(got[0][1][0][2]["CUDA_VISIBLE_DEVICES"], "0")

    def test_two_cards(self):
        with tempfile.TemporaryDirectory() as d:
            out = Path(d) / "o"
            rc1, one, done1 = self._run(self.new, self._argv(d, out), out)
            rc2, two, done2 = self._run(self.new, self._argv(d, out, "--cards", "2"), out)
            self.assertEqual((rc1, rc2), (0, 0))
            self.assertEqual(two[0][0], one[0][0])  # the same EASI command (--gpu 0 --nproc 1)
            self.assertEqual(two[0][2]["CUDA_VISIBLE_DEVICES"], "0,1")
            self.assertEqual(two[0][2]["PYTORCH_CUDA_ALLOC_CONF"], "expandable_segments:True")
            self.assertEqual({k: v for k, v in two[0][2].items() if k not in ("CUDA_VISIBLE_DEVICES",
                                                                              "PYTORCH_CUDA_ALLOC_CONF")},
                             {k: v for k, v in one[0][2].items() if k != "CUDA_VISIBLE_DEVICES"})
            self.assertEqual({**json.loads(done1), "cards": 2}, json.loads(done2))
            # a PYTORCH_CUDA_ALLOC_CONF the job already has is kept
            _, kept, _ = self._run(self.new, self._argv(d, out, "--cards", "2"), out,
                                   {"PYTORCH_CUDA_ALLOC_CONF": "max_split_size_mb:512"})
            self.assertEqual(kept[0][2]["PYTORCH_CUDA_ALLOC_CONF"], "max_split_size_mb:512")


class EasiEvalPassesCards(unittest.TestCase):
    """scripts/easi_eval.py (unchanged) only sets CUDA_VISIBLE_DEVICES when it is unset: the EASI runner (and so
    accelerate launch and lmms-eval, which do not set it either) sees the cards qa_eval gave."""

    def test_env(self):
        path = REPO / "scripts/easi_eval.py"
        mod = _module_from_source(path.read_text(), "_easi_eval_cards", path)
        with tempfile.TemporaryDirectory() as root:
            easi = Path(root) / "easi"
            (easi / "scripts/submissions").mkdir(parents=True)
            (easi / "scripts/submissions/run_easi_eval.py").write_text("")
            reg = Path(root) / "reg.yaml"
            reg.write_text(f"easi_dir: {easi}\nlmms_eval_dir: {root}/lmms\ncheckpoints: {{}}\n")
            (Path(root) / "ckpt").mkdir()
            for visible, want in (("0,1", "0,1"), (None, "0")):
                seen = []

                def fake_run(cmd, cwd=None, env=None):
                    seen.append(env["CUDA_VISIBLE_DEVICES"])
                    return types.SimpleNamespace(returncode=0)

                argv = ["easi_eval.py", "--registry", str(reg), "--ckpts", "", "--add", f"m:{root}/ckpt:qwen3_5",
                        "--benchmarks", "site_video", "--output_dir", f"{root}/out", "--gpu", "0", "--nproc", "1",
                        "--no-summary"]
                with mock.patch.object(mod.subprocess, "run", side_effect=fake_run), \
                        mock.patch.object(sys, "argv", argv), mock.patch.dict(os.environ, {}):
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                    if visible:
                        os.environ["CUDA_VISIBLE_DEVICES"] = visible
                    mod.main()
                self.assertEqual(seen, [want])


if __name__ == "__main__":
    unittest.main()
