"""q38 base zero-shot (2026-11-20; part of the gated train_chain diff): plan qa_eval "probe" adds one QA job a model,
qa_eval probe in the plan's model type, results <out>/probe/<name>; without it the QA jobs are exactly as before."""

import tempfile
import unittest
from pathlib import Path

from data_gen.interiorgs_v10 import train_chain as T


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


if __name__ == "__main__":
    unittest.main()
