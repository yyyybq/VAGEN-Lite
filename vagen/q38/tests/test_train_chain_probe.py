"""q38 base zero-shot (2026-11-20; part of the gated train_chain diff): plan qa_eval "probe" adds one QA job a model,
qa_eval probe in the plan's model type, results <out>/probe/<name>; without it the QA jobs are exactly as before.

q38 (2026-11-20, EXPLORATORY): plan qa_eval "probe_variants" {VARIANT: {"max_tokens": N}} adds one job a probe model,
"qa_<VARIANT>__<name>", the main probe's arguments and --max-tokens N, results <out>/<VARIANT>/<name>.  Plans without
the key must give the same jobs (and, through eval_jobs, the same job script arguments, cards and pools) as the q38
branch's 17c6894a (git show; Q38_BEFORE_GIT names another checkout of the branch when this tree lacks the commit)."""

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

from data_gen.interiorgs_v10 import train_chain as T

REPO = Path(__file__).resolve().parents[3]
BEFORE = "17c6894a"  # the q38 branch's HEAD before plan qa_eval "probe_variants"


def _before_chain():
    """train_chain.py at BEFORE as a module (None without git or the commit)."""
    path = "data_gen/interiorgs_v10/train_chain.py"
    for repo in (REPO, os.environ.get("Q38_BEFORE_GIT")):
        if not repo:
            continue
        try:
            src = subprocess.run(["git", "-C", str(repo), "show", f"{BEFORE}:{path}"], capture_output=True, text=True,
                                 check=True).stdout
        except (OSError, subprocess.CalledProcessError):
            continue
        name = "data_gen.interiorgs_v10._train_chain_before_probe_variants"
        mod = types.ModuleType(name)
        mod.__file__, mod.__package__ = str(REPO / path), "data_gen.interiorgs_v10"
        sys.modules[name] = mod
        try:
            exec(compile(src, name, "exec"), mod.__dict__)
        finally:
            sys.modules.pop(name, None)
        return mod
    return None


def _models(root: Path):
    for m in ("q38base", "other"):
        (root / m).mkdir(exist_ok=True)
        (root / m / "config.json").write_text("{}")
    return {"q38base": (str(root / "q38base"), None, []), "other": (str(root / "other"), None, ["--max-turns", "9"])}


def _plan(root: Path, **qa):
    return {"tr": str(root / "tr"), "sft_dir": str(root / "sft"), "eval_only": True, "model_type": "qwen3_5",
            "eval_sets": {}, "eval": {"models": {"q38base": {"model": str(root / "q38base")},
                                                 "other": {"model": str(root / "other")}}},
            "qa_eval": {"out": str(root / "qa"), **qa}}


def _run_eval_jobs(mod, plan, jobs):
    """eval_jobs with sco replaced by fakes: {run name: (cards, job script arguments)}, the pools' cards."""
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


class QaProbeJobs(unittest.TestCase):
    def test_probe_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for m in ("q38base", "other"):
                (root / m).mkdir()
                (root / m / "config.json").write_text("{}")
            models = {"q38base": (str(root / "q38base"), None, []), "other": (str(root / "other"), None, [])}
            plan = {"tr": tmp, "sft_dir": str(root / "sft"), "eval_only": True, "model_type": "qwen3_5",
                    "eval_sets": {}, "eval": {"models": {"q38base": {"model": str(root / "q38base")},
                                                         "other": {"model": str(root / "other")}}}}
            probe = str(root / "p1_probe.jsonl")
            qa = {"out": str(root / "qa"), "easi": [], "p0": str(root / "sel.jsonl")}
            before = T.Chain({**plan, "qa_eval": qa}, sleep=lambda s: None).qa_jobs(models)
            self.assertEqual(sorted(before), ["qa_p0__other", "qa_p0__q38base"])
            jobs = T.Chain({**plan, "qa_eval": {**qa, "probe": probe, "probe_models": ["q38base"]}},
                           sleep=lambda s: None).qa_jobs(models)
            self.assertEqual(sorted(jobs), ["qa_p0__other", "qa_p0__q38base", "qa_probe__q38base"])
            self.assertEqual({k: v for k, v in jobs.items() if not k.startswith("qa_probe")}, before)
            self.assertEqual(jobs["qa_probe__q38base"],
                             (root / "qa" / "probe", "q38base", str(root / "q38base"), None,
                              ["probe", "--probe", probe, "--model-type", "qwen3_5"], "qa"))
            limited = T.Chain({**plan, "qa_eval": {"out": str(root / "qa"), "easi": [], "probe": probe, "limit": 5}},
                              sleep=lambda s: None).qa_jobs(models)
            self.assertEqual(sorted(limited), ["qa_probe__other", "qa_probe__q38base"])
            self.assertEqual(limited["qa_probe__other"][4][-2:], ["--limit", "5"])
            with self.assertRaises(T.ChainError):  # a probe model the evaluate step lacks
                T.Chain({**plan, "qa_eval": {**qa, "probe": probe, "probe_models": ["nope"]}},
                        sleep=lambda s: None).qa_jobs(models)


class ProbeVariants(unittest.TestCase):
    """Plan qa_eval "probe_variants" (2026-11-20, exploratory: the 27B base model with up to 1,024 new tokens)."""

    @classmethod
    def setUpClass(cls):
        cls.old = _before_chain()

    def _without_variants(self, root: Path):
        p = str(root / "p1_probe.jsonl")
        yield _plan(root, easi=[], probe=p)
        yield _plan(root, easi=[], probe=p, probe_models=["q38base"], p0="S")
        yield _plan(root, easi="easi_8", probe=p, limit=5)
        yield _plan(root, easi=["vsi_bench", "site"], p0="S", p0_res=512)
        yield {**_plan(root, easi=[], probe=p), "model_type": "qwen3_vl", "eval_cards": 2}
        yield {**_plan(root), "qa_eval": None}

    def test_plans_without_variants_unchanged(self):
        """The same jobs, job script arguments, cards and pools as 17c6894a; "probe_variants" {} or null as absent."""
        if self.old is None:
            self.skipTest(f"git or {BEFORE} not available (set Q38_BEFORE_GIT)")
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models = _models(root)
            n = 0
            for plan in self._without_variants(root):
                old_jobs = self.old.Chain(plan, sleep=lambda s: None).qa_jobs(models)
                plans = [plan]
                if plan.get("qa_eval"):
                    plans += [{**plan, "qa_eval": {**plan["qa_eval"], "probe_variants": v}} for v in ({}, None)]
                for p in plans:
                    new_jobs = T.Chain(p, sleep=lambda s: None).qa_jobs(models)
                    self.assertEqual(new_jobs, old_jobs, p)
                    self.assertEqual(list(new_jobs), list(old_jobs), p)  # the same order (eval_jobs' queue)
                    jobs = {**new_jobs, "val__q38base": (root / "ev", "q38base", str(root / "q38base"), None, [])}
                    got = []
                    for mod in (T, self.old):
                        for d in ("tr", "qa", "ev"):
                            shutil.rmtree(root / d, ignore_errors=True)
                        got.append(_run_eval_jobs(mod, p, jobs))
                    self.assertEqual(got[0], got[1], p)
                    self.assertFalse(any("--max-tokens" in a for _, a in got[0][0].values()), p)
                    n += 1
            self.assertEqual(n, 16)

    def test_variant_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models = _models(root)
            p = str(root / "p1_probe.jsonl")
            for extra, limit in (({}, []), ({"limit": 5}, ["--limit", "5"])):
                plain = _plan(root, easi=[], p0="S", probe=p, probe_models=["q38base"], **extra)
                base = T.Chain(plain, sleep=lambda s: None).qa_jobs(models)
                plan = {**plain, "qa_eval": {**plain["qa_eval"], "probe_variants": {"probe1024": {"max_tokens": 1024}}}}
                got = T.Chain(plan, sleep=lambda s: None).qa_jobs(models)
                self.assertEqual({k: v for k, v in got.items() if k != "qa_probe1024__q38base"}, base)
                self.assertEqual(list(got)[:len(base)], list(base))  # the variant after every main job
                want = (root / "qa" / "probe1024", "q38base", str(root / "q38base"), None,
                        ["probe", "--probe", p, "--model-type", "qwen3_5", *limit, "--max-tokens", "1024"], "qa")
                self.assertEqual(got["qa_probe1024__q38base"], want)
                self.assertEqual(got["qa_probe1024__q38base"][4][:-2], base["qa_probe__q38base"][4])
                seen, pools = _run_eval_jobs(T, plan, got)
                self.assertEqual(seen["qa_probe1024__q38base"],
                                 (1, ["qa", str(root / "qa" / "probe1024"), "q38base", str(root / "q38base"),
                                      "probe", "--probe", p, "--model-type", "qwen3_5", *limit, "--max-tokens", "1024"]))
                self.assertEqual(pools, [1] * len(got))
                self.assertTrue((root / "qa" / "probe1024" / "q38base" / "done.json").is_file())
                self.assertFalse((root / "qa" / "probe" / "probe1024").exists())
                shutil.rmtree(root / "qa", ignore_errors=True)
                shutil.rmtree(root / "tr", ignore_errors=True)
            # two variants, every probe model (default "models"); the bounds 1 and 3072
            plan = _plan(root, easi=[], probe=p, probe_variants={"probe1024": {"max_tokens": 1024},
                                                                 "probe_one": {"max_tokens": 1},
                                                                 "probe3072": {"max_tokens": 3072}})
            got = T.Chain(plan, sleep=lambda s: None).qa_jobs(models)
            self.assertEqual(sorted(got), sorted(f"qa_{v}__{n}" for v in ("probe", "probe1024", "probe_one", "probe3072")
                                                 for n in ("q38base", "other")))
            self.assertEqual({k: v[0].name for k, v in got.items()}, {k: k[3:].split("__")[0] for k in got})
            self.assertEqual(got["qa_probe_one__other"][4][-2:], ["--max-tokens", "1"])
            self.assertEqual(got["qa_probe3072__other"][4][-2:], ["--max-tokens", "3072"])

    def test_variant_errors(self):
        """A bad variant stops qa_jobs (the evaluate step) before any job: ChainError."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            models = _models(root)
            p = str(root / "p1_probe.jsonl")
            ok = {"max_tokens": 1024}
            bad = [{"probe": ok}, {"p1024": ok}, {"probe-1024": ok}, {"probe 1": ok}, {"": ok}, {5: ok},
                   {"probe1024": {"max_tokens": 0}}, {"probe1024": {"max_tokens": 3073}},
                   {"probe1024": {"max_tokens": -1}}, {"probe1024": {"max_tokens": True}},
                   {"probe1024": {"max_tokens": "1024"}}, {"probe1024": {"max_tokens": 1024.0}},
                   {"probe1024": {"max_tokens": None}}, {"probe1024": {"max_tokens": 1024, "temperature": 0.8}},
                   {"probe1024": {}}, {"probe1024": 1024}, {"probe1024": None}, ["probe1024"], "probe1024", 1024,
                   {"probe1024": ok, "probe": ok}]
            for b in bad:
                with self.assertRaises(T.ChainError, msg=repr(b)):
                    T.Chain(_plan(root, easi=[], probe=p, probe_variants=b), sleep=lambda s: None).qa_jobs(models)
            for no_probe in (_plan(root, easi=[], probe_variants={"probe1024": ok}),
                             _plan(root, easi=[], probe="", p0="S", probe_variants={"probe1024": ok})):
                with self.assertRaises(T.ChainError):  # a variant needs the main probe's file
                    T.Chain(no_probe, sleep=lambda s: None).qa_jobs(models)


if __name__ == "__main__":
    unittest.main()
