"""train_chain.py: the unattended v10 training chain (progress readers, stop on failure, resume)."""

import json
import tempfile
import unittest
from pathlib import Path

from data_gen.interiorgs_v10 import train_chain as T

RL_LOG = """(TaskRunner pid=1) step:0 - val-aux/active_spatial/traj_success/mean@4:np.float64(0.03125) - x:1
(TaskRunner pid=1) step:1 - episode/success_rate:0.125 - timing_s/step:153.2
(TaskRunner pid=1) step:2 - episode/success_rate:0.25 - timing_s/step:89.0 - val-aux/active_spatial/traj_success/mean@4:np.float64(0.0625)
"""


class ProgressTest(unittest.TestCase):
    def test_readers(self):
        rl = T.rl_progress(RL_LOG)
        self.assertEqual(rl["last_update"], 2)
        self.assertEqual(rl["train_success_last"], 0.25)
        self.assertAlmostEqual(rl["train_success_mean_last10"], 0.1875)
        self.assertEqual(rl["val_success"], [(0, 0.03125), (2, 0.0625)])
        greedy = T.rl_progress("step:0 - val-aux/active_spatial/traj_success/mean@1:np.float64(0.25) - x:1\n")
        self.assertEqual(greedy["val_success"], [(0, 0.25)])  # round 10: one greedy trajectory per val task
        self.assertEqual(T.pick_best({"a_e1": 0.3, "b_e1": 0.3, "a_e2": 0.2, "a_e3": None},
                                     ["a_e1", "b_e1", "a_e2", "a_e3"]), "a_e1")
        self.assertEqual(T.pick_best({"a_e1": 0.3, "b_e1": 0.4}, ["a_e1", "b_e1"]), "b_e1")
        self.assertIsNone(T.pick_best({"a": None}, ["a"]))
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "trainer_log.jsonl").write_text(
                json.dumps({"current_steps": 5, "total_steps": 90, "loss": 0.8, "epoch": 0.2}) + "\n"
                + json.dumps({"current_steps": 10, "total_steps": 90, "eval_loss": 0.5}) + "\n")
            self.assertEqual(T.sft_progress(d), {"current_steps": 10, "total_steps": 90, "loss": 0.8, "epoch": 0.2,
                                                 "eval_loss": 0.5, "eval_at_step": 10})
        self.assertIsNone(T.keep_training([(0, 0.1)]))
        self.assertTrue(T.keep_training([(0, 0.35), (25, 0.40), (150, 0.31)]))   # 9 points below the best: noise
        self.assertFalse(T.keep_training([(0, 0.35), (25, 0.40), (150, 0.29)]))  # more than 10 below: stop
        self.assertTrue(T.keep_training([(0, 0.1), (25, 0.2), (50, 0.3)]))


class ChainTest(unittest.TestCase):
    def test_stops_on_failure_and_resumes(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = {"tr": tmp, "sft_dir": tmp, "commit": "x", "endpoint": 150}
            calls = []

            def make(fail_at=None):
                chain = T.Chain(plan, sleep=lambda s: None)
                for step in T.STEPS:
                    def fn(step=step):
                        calls.append(step)
                        if step == fail_at:
                            raise T.ChainError("job sft ended FAILED")
                    setattr(chain, step, fn)
                return chain

            self.assertEqual(make(fail_at="sft").run(), 1)
            self.assertEqual(calls, ["check_code", "render", "sft_data", "sft"])  # nothing after the failure
            state = json.loads((Path(tmp) / "progress.json").read_text())
            self.assertEqual(state["state"], "failed")
            self.assertEqual(state["steps"]["sft"]["state"], "failed")
            self.assertIn("ended FAILED", state["steps"]["sft"]["error"])
            self.assertEqual(state["steps"]["rl"]["state"], "pending")
            self.assertIn("失败", (Path(tmp) / "PROGRESS.md").read_text(encoding="utf-8"))
            calls.clear()
            self.assertEqual(make().run(), 0)  # a new run continues at the failed step
            self.assertEqual(calls, ["sft", "sft_eval", "pick_sft", "rl_config", "rl", "rl_extend", "evaluate"])
            self.assertEqual(json.loads((Path(tmp) / "progress.json").read_text())["state"], "done")

    def test_progress_without_the_new_steps(self):
        with tempfile.TemporaryDirectory() as tmp:  # a progress.json written by the six-step chain
            steps = {s: {"state": "done"} for s in ("check_code", "render", "sft_data", "sft", "rl_config", "rl")}
            (Path(tmp) / "progress.json").write_text(json.dumps({"plan": {}, "steps": steps}))
            chain = T.Chain({"tr": tmp, "sft_dir": tmp, "endpoint": 150}, sleep=lambda s: None)
            self.assertEqual(chain.run(), 0)  # neither extension nor evaluation planned
            state = json.loads((Path(tmp) / "progress.json").read_text())
            self.assertEqual(state["steps"]["evaluate"], dict(state["steps"]["evaluate"], state="done",
                                                              note="not planned"))
            for s in ("sft_eval", "pick_sft"):  # the round-10 steps are not planned either
                self.assertEqual(state["steps"][s]["note"], "not planned")

    def _chain(self, tmp, **plan):
        chain = T.Chain(dict({"tr": tmp, "sft_dir": tmp, "endpoint": 150}, **plan), sleep=lambda s: None)
        logs = Path(tmp) / "rl_run" / "logs"
        logs.mkdir(parents=True, exist_ok=True)
        return chain, logs

    def test_rl_extend(self):
        steady = "".join(f"step:{s} - val-aux/active_spatial/traj_success/mean@4:{v}\n"
                         for s, v in ((0, 0.30), (100, 0.34), (125, 0.38), (150, 0.42)))
        dropped = steady.replace("0.42", "0.27")  # 11 points below the best earlier value (0.38)
        for log, want in ((steady, 250), (dropped, None)):
            with tempfile.TemporaryDirectory() as tmp:
                chain, logs = self._chain(tmp, extend_to=250)
                (logs / "rl-1.log").write_text(log)
                ppo = []
                chain._ppo = lambda *a, **k: ppo.append((a, k))
                chain.rl_extend()
                self.assertEqual(chain.state["steps"]["rl_extend"]["decision"]["extend_to"], want)
                self.assertEqual(ppo, [(("rl_extend", 250), {"suffix": "_ext"})] if want else [])
                if want:  # resumed after the extension started: the decision stands though validation moved on
                    (logs / "rl-2.log").write_text("step:175 - val-aux/active_spatial/traj_success/mean@4:0.1\n")
                    resumed = T.Chain(chain.plan, sleep=lambda s: None)
                    resumed._ppo = lambda *a, **k: ppo.append((a, k))
                    resumed.rl_extend()
                    self.assertEqual(len(ppo), 2)
                    self.assertEqual(resumed.final_update(), 250)
                    self.assertIn("续训到 250 次", (Path(tmp) / "PROGRESS.md").read_text(encoding="utf-8"))
                else:
                    self.assertEqual(chain.final_update(), 150)
        with tempfile.TemporaryDirectory() as tmp:
            chain, _ = self._chain(tmp)
            chain.rl_extend()
            self.assertEqual(chain.state["steps"]["rl_extend"]["note"], "not planned")

    def test_evaluate(self):
        from unittest import mock

        from data_gen.interiorgs_v10 import job_sco

        with tempfile.TemporaryDirectory() as tmp:
            ev = Path(tmp) / "eval"
            extra = Path(tmp) / "base"
            chain, _ = self._chain(tmp, eval={"dir": str(ev), "every": 50,
                                              "models": {"base": {"model": str(extra)},
                                                         "r1": {"model": str(extra), "prompt_format": "free_think"}}})
            for m in [extra] + [Path(tmp) / "rl_run" / "checkpoints" / f"global_step_{s}" / "actor" / "huggingface"
                                for s in (50, 100, 150)]:
                m.mkdir(parents=True)
                (m / "config.json").write_text("{}")
            (ev / "v10_rl_050").mkdir(parents=True)
            (ev / "v10_rl_050" / "done.json").write_text("{}")  # evaluated before: no new job
            prepared, deleted, fail = {}, [], set()

            def prepare(run, gpus, args, job_dir="job", pool=None):
                prepared[run.name] = (gpus, args)
                (run / job_dir).mkdir(parents=True, exist_ok=True)

            def submit(run, job_dir="job"):
                (run / job_dir / "submission.json").write_text(json.dumps({"job_id": "pt-" + run.name}))

            def status(run, job_dir="job"):
                if run.name not in fail:
                    (ev / run.name).mkdir(exist_ok=True)
                    (ev / run.name / "done.json").write_text("{}")
                return {"state": "FAILED" if run.name in fail else "SUCCEEDED", "finished": True}

            def delete(run, job_dir="job"):
                deleted.append(run.name)
                receipt = run / job_dir / "submission.json"
                receipt.replace(receipt.with_name(f"submission_{len(deleted)}_deleted.json"))

            tables = []
            chain.sh = lambda args, **k: tables.append(args)
            with mock.patch.multiple(job_sco, prepare=prepare, submit=submit, status=status, delete=delete):
                chain.evaluate()
                self.assertEqual(sorted(prepared), ["base", "r1", "v10_rl_100", "v10_rl_150"])
                self.assertEqual(prepared["r1"], (1, ["eval", str(ev), "r1", str(extra), "free_think"]))
                self.assertEqual(prepared["v10_rl_150"][1][3],
                                 str(Path(tmp) / "rl_run" / "checkpoints" / "global_step_150" / "actor" / "huggingface"))
                self.assertEqual(sorted(deleted), sorted(prepared))  # every job deleted after it ended
                self.assertEqual(tables[0][-2:], ["--eval", str(ev)])

                (ev / "base" / "done.json").unlink()  # a failed job: the step fails, its job is still deleted
                fail.add("base")
                prepared.clear()
                deleted.clear()
                with self.assertRaises(T.ChainError):
                    chain.evaluate()
                self.assertEqual((list(prepared), deleted), (["base"], ["base"]))

                chain.plan["eval"]["models"]["missing"] = {"model": str(Path(tmp) / "nowhere")}
                with self.assertRaises(T.ChainError):
                    chain.evaluate()

    def test_sft_groups_epoch_evaluation_and_pick(self):
        """Round 10: two SFT groups, every epoch checkpoint evaluated on two sets, the RL start picked by val."""
        from unittest import mock

        from data_gen.interiorgs_v10 import job_sco

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sft_dir = root / "sft"
            sft_dir.mkdir()
            for f in ("sft_trajectories.jsonl", "sft_trajectories_formal.jsonl", "split_manifest.json",
                      "task_registry.jsonl", "rows.jsonl"):
                (sft_dir / f).write_text("")
            val, test = root / "val200", root / "test"
            chain = T.Chain({"tr": tmp, "sft_dir": str(sft_dir), "commit": "x", "endpoint": 150, "model": "/base",
                             "ppo_template": "/S0.yaml", "gs_root": "/ply", "scenes_root": "/meta", "keep_text": True,
                             "sft_epochs": 5, "sft_groups": {"formal": "sft_trajectories_formal.jsonl",
                                                             "formal_reserve": "sft_trajectories.jsonl"},
                             "eval_sets": {"val200": str(val), "test": str(test)}, "select_by": "val200",
                             "rl": {"history_mode": "concat", "critic_warmup": 10, "test_freq": 25,
                                    "val_tasks": str(val / "tasks.jsonl"), "greedy_val": True},
                             "eval": {"every": 50, "options": ["--x"],
                                      "models": {"r8_sft_e3": {"model": "/r8", "options": ["--sft-text"]}}},
                             "max_eval_jobs": 3}, sleep=lambda s: None)
            calls = []
            chain.sh = lambda args, **k: calls.append(args)
            # sft_data: the conversations with scenes, then one recipe per group with 5 epochs and verbatim text
            for g in ("lf_formal", "lf_formal_reserve"):
                (root / g).mkdir()
                (root / g / "train.yaml").write_text("")
            chain.sft_data()
            self.assertEqual(calls[0][2:6], ["data_gen.interiorgs_v10.training_prep", "sft-records", "--dir", str(sft_dir)])
            self.assertEqual(calls[0][-2:], ["--scenes-root", "/meta"])
            self.assertEqual(len(calls), 3)
            formal = calls[1]
            self.assertEqual(formal[formal.index("--trajectory") + 1], str(sft_dir / "sft_trajectories_formal.jsonl"))
            self.assertEqual(formal[formal.index("--epochs") + 1], "5")
            self.assertIn("--keep-text", formal)
            self.assertEqual(formal[formal.index("--out") + 1], str(root / "lf_formal"))
            self.assertEqual(calls[2][calls[2].index("--out") + 1], str(root / "lf_formal_reserve"))
            # sft: one 8-card job per group, both submitted before waiting
            prepared, deleted = {}, []

            def prepare(run, gpus, args, job_dir="job", pool=None):
                prepared[run.name] = (gpus, args)
                (run / job_dir).mkdir(parents=True, exist_ok=True)

            def submit(run, job_dir="job"):
                (run / job_dir / "submission.json").write_text(json.dumps({"job_id": "pt-" + run.name}))

            def delete(run, job_dir="job"):
                deleted.append(run.name)
                receipt = run / job_dir / "submission.json"
                receipt.replace(receipt.with_name(f"submission_{len(deleted)}_deleted.json"))

            results = {"val200": {"sft_formal_e1": 0.2, "sft_formal_e2": 0.4, "sft_formal_reserve_e1": 0.4,
                                  "sft_formal_reserve_e2": 0.3},
                       "test": {"sft_formal_e1": 0.1, "sft_formal_e2": 0.2, "sft_formal_reserve_e1": 0.5,
                                "sft_formal_reserve_e2": 0.2}}

            def status(run, job_dir="job"):
                if run.name.startswith("sft_"):
                    return {"state": "SUCCEEDED", "finished": True}
                set_name, result = run.name.split("__", 1)
                d = (val if set_name == "val200" else test) / result
                d.mkdir(parents=True, exist_ok=True)
                rate = results.get(set_name, {}).get(result, 0.0)
                rows = [{"task_type": "fov_inclusion", "d_star": 2, "success": i < int(rate * 10), "turns": 1,
                         "actions": 1, "collisions": 0, "invalid_turns": 0, "actions_to_success": 1} for i in range(10)]
                (d / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
                (d / "done.json").write_text("{}")
                return {"state": "SUCCEEDED", "finished": True}

            with mock.patch.multiple(job_sco, prepare=prepare, submit=submit, status=status, delete=delete):
                for g in ("formal", "formal_reserve"):
                    model = root / f"lf_{g}" / "model"
                    for step in (163, 326):  # two epochs saved
                        (model / f"checkpoint-{step}").mkdir(parents=True)
                        (model / f"checkpoint-{step}" / "config.json").write_text("{}")
                    (model / "config.json").write_text("{}")
                    (model / "m.safetensors").write_text("")
                chain.sft()
                self.assertEqual(sorted(prepared), ["sft_formal", "sft_formal_reserve"])
                self.assertEqual(prepared["sft_formal"], (8, ["sft", str(root / "lf_formal")]))
                self.assertEqual(sorted(deleted), ["sft_formal", "sft_formal_reserve"])
                # sft_eval: 2 groups x 2 epochs x 2 sets = 8 single-card jobs, the plan options on each
                prepared.clear()
                deleted.clear()
                calls.clear()
                chain.sft_eval()
                self.assertEqual(len(prepared), 8)
                self.assertEqual(prepared["val200__sft_formal_e2"],
                                 (1, ["eval", str(val), "sft_formal_e2", str(root / "lf_formal" / "model" / "checkpoint-326"),
                                      "-", "--x"]))
                self.assertEqual(sorted(deleted), sorted(prepared))
                self.assertEqual([c[-1] for c in calls], [str(val), str(test)])  # a table per set
                # pick: the best val success; ties go to the earlier epoch (formal_reserve e1 over formal e2)
                chain.pick_sft()
                d = chain.state["steps"]["pick_sft"]["decision"]
                best_model = root / "lf_formal_reserve" / "model" / "checkpoint-163"
                self.assertEqual((d["best"], d["model"]), ("sft_formal_reserve_e1", str(best_model)))
                self.assertEqual(d["scores"]["sft_formal_e2"], 0.4)  # the tie, lost on the epoch
                self.assertEqual(d["other_sets"]["test"]["sft_formal_reserve_e1"], 0.5)  # recorded, not used
                self.assertEqual(chain.sft_model(), best_model)
                # rl_config: the picked model and the RL options of the plan
                calls.clear()
                chain.rl_config()
                rc = calls[0]
                self.assertEqual(rc[rc.index("--model") + 1], str(best_model))
                for flag, value in (("--history-mode", "concat"), ("--critic-warmup", "10"), ("--test-freq", "25"),
                                    ("--val-tasks", str(val / "tasks.jsonl"))):
                    self.assertEqual(rc[rc.index(flag) + 1], value)
                self.assertIn("--greedy-val", rc)
                # evaluate: RL checkpoints every 50 and the extra model on both sets, per-model options
                for s in (50, 100, 150):
                    m = root / "rl_run" / "checkpoints" / f"global_step_{s}" / "actor" / "huggingface"
                    m.mkdir(parents=True)
                    (m / "config.json").write_text("{}")
                (root / "r8").mkdir()
                (root / "r8" / "config.json").write_text("{}")
                chain.plan["eval"]["models"]["r8_sft_e3"]["model"] = str(root / "r8")
                prepared.clear()
                calls.clear()
                chain.evaluate()
                self.assertEqual(len(prepared), 8)
                self.assertEqual(prepared["test__r8_sft_e3"],
                                 (1, ["eval", str(test), "r8_sft_e3", str(root / "r8"), "-", "--sft-text"]))
                self.assertEqual(prepared["val200__v10_rl_100"][1][-1], "--x")
                self.assertIn("挑出 sft_formal_reserve_e1", (root / "PROGRESS.md").read_text(encoding="utf-8"))

    def test_family_success_from_rollout_dumps(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "rollout_data"
            d.mkdir()
            for step, rows in ((1, [("fov_inclusion", 1.0), ("fov_inclusion", 0.0), ("equidistance", 0.0)]),
                               (25, [("fov_inclusion", 1.0)]), (26, [("equidistance", 1.0)])):
                (d / f"{step}.jsonl").write_text("".join(json.dumps({"task_type": f, "traj_success": v}) + "\n"
                                                         for f, v in rows) + "not json\n")
            got = T.family_success(d)
            self.assertEqual(list(got), ["1-25", "26-50"])
            self.assertEqual(got["1-25"]["fov_inclusion"], {"success": round(2 / 3, 4), "episodes": 3})
            self.assertEqual(got["1-25"]["equidistance"], {"success": 0.0, "episodes": 1})
            self.assertEqual(got["26-50"], {"equidistance": {"success": 1.0, "episodes": 1}})
            self.assertEqual(T.family_success(Path(tmp) / "missing"), {})

    def test_two_rl_runs_and_variants(self):
        """Round 11: R-S0 and R-D from the picked SFT checkpoint at the same time; a failed run does not stop the
        other; every run's checkpoints and the sampled variants are evaluated."""
        from unittest import mock

        from data_gen.interiorgs_v10 import job_sco

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            val, test = root / "val400", root / "test"
            rl = {"history_mode": "concat", "critic_warmup": 10, "train_batch_size": 24, "ppo_mini_batch_size": 16,
                  "save_freq": 50, "no_internal_val": True}
            chain = T.Chain({"tr": tmp, "sft_dir": tmp, "endpoint": 100, "ppo_template": "/S0.yaml", "gs_root": "/ply",
                             "pool": "h800", "eval_sets": {"val400": str(val), "test": str(test)},
                             "rl_runs": {"rs0": {"reward_preset": "S0", "rl": rl}, "rd": {"reward_preset": "dense", "rl": rl}},
                             "eval": {"every": 50, "variants": [{"suffix": "_s4", "sets": ["val400"],
                                                                 "options": ["--samples", "4", "--temperature", "0.8"],
                                                                 "models": ["sft_pick", "rl", "r10_sft_e5"]}],
                                      "models": {"r10_sft_e5": {"model": "/r10"}}}}, sleep=lambda s: None)
            chain.state["steps"]["pick_sft"]["decision"] = {"best": "sft_e7", "model": "/lf/model/checkpoint-7",
                                                            "select_by": "val400", "scores": {"sft_e7": 0.5}}
            calls = []
            chain.sh = lambda args, **k: calls.append(args)
            chain.rl_config()
            self.assertEqual(len(calls), 2)
            by_out = {c[c.index("--out") + 1]: c for c in calls}
            rd = by_out[str(root / "rl_rd")]
            self.assertEqual(rd[rd.index("--reward-preset") + 1], "dense")
            self.assertEqual(rd[rd.index("--run-root") + 1], str(root / "rl_run_rd"))
            self.assertEqual(rd[rd.index("--model") + 1], str(Path("/lf/model/checkpoint-7")))
            for flag, value in (("--train-batch-size", "24"), ("--ppo-mini-batch-size", "16"), ("--save-freq", "50"),
                                ("--critic-warmup", "10"), ("--history-mode", "concat")):
                self.assertEqual(rd[rd.index(flag) + 1], value)
            self.assertIn("--no-internal-val", rd)
            self.assertEqual(by_out[str(root / "rl_rs0")][by_out[str(root / "rl_rs0")].index("--reward-preset") + 1], "S0")

            # PPO: rs0 finishes, rd's render service dies; rs0 is not stopped, the step fails naming rd
            prepared, deleted, pools = {}, [], set()
            polls = {"n": 0}

            def prepare(run, gpus, args, job_dir="job", pool=None):
                prepared[(run.name, job_dir)] = (gpus, args)
                pools.add(pool)
                (run / job_dir).mkdir(parents=True, exist_ok=True)

            def submit(run, job_dir="job"):
                (run / job_dir / "submission.json").write_text(json.dumps({"job_id": f"pt-{run.name}-{job_dir}"}))

            def status(run, job_dir="job"):
                polls["n"] += 1
                if run.name == "rl_run_rd" and job_dir == "job_renderer":
                    return {"state": "FAILED", "finished": True}
                if run.name == "rl_run_rs0" and job_dir == "job_rl" and polls["n"] > 4:
                    (run / "checkpoints" / "global_step_100").mkdir(parents=True, exist_ok=True)
                    return {"state": "SUCCEEDED", "finished": True}
                return {"state": "RUNNING", "finished": False}

            def delete(run, job_dir="job"):
                deleted.append((run.name, job_dir))
                receipt = run / job_dir / "submission.json"
                if receipt.exists():
                    receipt.replace(receipt.with_name(f"submission_{len(deleted)}_deleted.json"))

            with mock.patch.multiple(job_sco, prepare=prepare, submit=submit, status=status, delete=delete):
                with self.assertRaises(T.ChainError) as err:
                    chain.rl()
                self.assertIn("rd (renderer FAILED)", str(err.exception))
                self.assertNotIn("rs0", str(err.exception))
                self.assertEqual(prepared[("rl_run_rs0", "job_rl")], (8, ["rl", str(root / "rl_rs0"), "100"]))
                self.assertEqual(prepared[("rl_run_rd", "job_renderer")], (1, ["renderer"]))
                self.assertEqual(pools, {"h800"})  # the plan's pool reaches every job (round 11: zoetrope was full)
                self.assertEqual(sorted(set(deleted)), [("rl_run_rd", "job_renderer"), ("rl_run_rd", "job_rl"),
                                                        ("rl_run_rs0", "job_renderer"), ("rl_run_rs0", "job_rl")])
                self.assertTrue((root / "rl_run_rs0" / "STOP_RENDERER").exists())
                # a resumed chain trains only the run without its endpoint checkpoint
                prepared.clear()
                (root / "rl_run_rd" / "checkpoints" / "global_step_100").mkdir(parents=True)
                chain.rl()
                self.assertEqual(prepared, {})

            # evaluation: every run's checkpoints on both sets, the sampled variant on val400 only
            for run in ("rs0", "rd"):
                for st in (50, 100):
                    m = root / f"rl_run_{run}" / "checkpoints" / f"global_step_{st}" / "actor" / "huggingface"
                    m.mkdir(parents=True, exist_ok=True)
                    (m / "config.json").write_text("{}")
            for m in (Path("/r10"), Path("/lf/model/checkpoint-7")):
                pass
            jobs = {}
            chain.eval_jobs = lambda step, j: jobs.update(j)
            chain.tables = lambda: None
            chain.evaluate()
            names = sorted(jobs)
            self.assertIn("val400__rs0_050", names)
            self.assertIn("test__rd_100", names)
            self.assertIn("val400__rd_100_s4", names)
            self.assertIn("val400__sft_e7_s4", names)
            self.assertIn("val400__r10_sft_e5_s4", names)
            self.assertNotIn("test__rd_100_s4", names)
            self.assertEqual(jobs["val400__sft_e7_s4"][2], "/lf/model/checkpoint-7")
            self.assertEqual(jobs["val400__rd_050_s4"][4], ["--samples", "4", "--temperature", "0.8"])
            self.assertEqual(len([n for n in names if n.endswith("_s4")]), 6)  # 4 RL checkpoints, the pick, r10
            chain.state["rl_runs"] = {"rd": {"last_update": 25, "family_success": {"1-25": {"fov_inclusion": {
                "success": 0.5, "episodes": 8}}}}}
            md = T.render_markdown(chain.state)
            self.assertIn("**RL rd：**", md)
            self.assertIn("| 1-25 | 0.50（8） |", md)

    def test_block_stats(self):
        """Round 11: per 25 updates, training success, the share of tasks with both outcomes, reward, entropy, KL."""
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp) / "rollout_data"
            d.mkdir()
            ep = lambda fam, task, ok, score: json.dumps({"task_type": fam, "scene_id": "s", "gts": {"t": task},  # noqa: E731
                                                          "traj_success": ok, "score": score}) + "\n"
            (d / "1.jsonl").write_text(ep("fov_inclusion", 1, True, 5.0) + ep("fov_inclusion", 1, False, -1.0)
                                       + ep("equidistance", 2, True, 5.0) + ep("equidistance", 2, True, 4.0))
            (d / "2.jsonl").write_text(ep("fov_inclusion", 1, False, -1.0) + ep("fov_inclusion", 1, False, -1.0))
            (d / "26.jsonl").write_text(ep("fov_inclusion", 1, True, 5.0) + ep("fov_inclusion", 1, False, 0.0))
            log = ("(Runner) step:1 - actor/entropy:np.float64(0.5) - actor/kl_loss:np.float64(0.001) - actor/ppo_kl:0.0\n"
                   "step:2 - actor/entropy:0.3 - actor/kl_loss:0.003 - actor/ppo_kl:0.002\nstep:26 - actor/entropy:0.2\n")
            got = T.block_stats(d, log)
            self.assertEqual(list(got), ["1-25", "26-50"])
            b = got["1-25"]
            self.assertEqual((b["updates"], b["episodes"], b["tasks"]), (2, 6, 3))  # task 1 in update 1, 2 in 1, 1 in 2
            self.assertEqual((b["success"], b["mixed_share"], b["reward_mean"]), (0.5, round(1 / 3, 4), round(11 / 6, 4)))
            self.assertEqual((b["entropy"], b["kl"], b["ppo_kl"]), (0.4, 0.002, 0.001))
            self.assertEqual(b["families"]["fov_inclusion"], {"success": 0.25, "episodes": 4})
            self.assertEqual((got["26-50"]["mixed_share"], got["26-50"]["kl"]), (1.0, None))
            self.assertEqual(T.block_stats(Path(tmp) / "missing"), {})

    def test_reserve_tasks_signal_check_and_fallback(self):
        """Round 11, the user's decisions: RL on the reserve tasks (checked against the SFT conversations and the depth
        flags); at update 25 each run needs 40-60% training success and >= 30% mixed tasks, else the runs stop, move
        to rl_attempts/ and RL restarts from the fallback SFT checkpoint; a second miss stops the chain."""
        from unittest import mock

        from data_gen.interiorgs_v10 import job_sco

        def chain_for(root):
            rl = {"history_mode": "concat", "critic_warmup": 10, "train_batch_size": 24, "save_freq": 50,
                  "no_internal_val": True, "task_pool": "reserve", "rows": "/rl_rows/rows.jsonl", "exclude": "/flags.json",
                  "signal_check": {"at": 25, "success_min": 0.4, "success_max": 0.6, "mixed_min": 0.3}}
            chain = T.Chain({"tr": str(root), "sft_dir": str(root / "sft"), "endpoint": 50, "ppo_template": "/S0.yaml",
                             "gs_root": "/ply", "sft_groups": {"formal": "sft_trajectories.jsonl"}, "rl": rl,
                             "rl_runs": {"rs0": {"reward_preset": "S0"}, "rd": {"reward_preset": "dense"}},
                             "rl_fallback_model": "/lf/model/checkpoint-924"}, sleep=lambda s: None)
            chain.state["steps"]["pick_sft"]["decision"] = {"best": "sft_formal_e9", "model": "/lf/model/checkpoint-1386",
                                                            "select_by": "val400", "scores": {"sft_formal_e9": 0.42}}
            return chain

        def jobs(chain, outcomes):
            """job_sco stand-ins: the trainer of attempt k writes 25 updates of rollouts with outcome pattern
            outcomes[k-1] (per task of 4 episodes), then the endpoint checkpoint."""
            def prepare(run, gpus, args, job_dir="job", pool=None):
                (run / job_dir).mkdir(parents=True, exist_ok=True)

            def submit(run, job_dir="job"):
                (run / job_dir / "submission.json").write_text(json.dumps({"job_id": f"pt-{run.name}-{job_dir}"}))

            def status(run, job_dir="job"):
                if job_dir == "job_renderer":
                    return {"state": "RUNNING", "finished": False}
                d = run / "rollout_data"
                if not d.is_dir():
                    d.mkdir(parents=True)
                    pattern = outcomes[int(chain.state.get("rl_attempt", 1)) - 1]
                    for step in range(1, 26):
                        rows = [{"task_type": "fov_inclusion", "scene_id": "s", "gts": {"t": t}, "traj_success": ok,
                                 "score": 5.0 if ok else -0.5} for t, oks in enumerate(pattern) for ok in oks]
                        (d / f"{step}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
                    return {"state": "RUNNING", "finished": False}
                (run / "checkpoints" / "global_step_50").mkdir(parents=True, exist_ok=True)
                return {"state": "SUCCEEDED", "finished": True}

            def delete(run, job_dir="job"):
                receipt = run / job_dir / "submission.json"
                if receipt.exists():
                    receipt.replace(receipt.with_name("submission_deleted.json"))

            return mock.patch.multiple(job_sco, prepare=prepare, submit=submit, status=status, delete=delete)

        memorised = [[True] * 4] * 4                                    # 100% success, no task with both outcomes
        learnable = [[True, False, True, False], [True, True, False, False], [False] * 4, [True] * 4]  # 50%, 50% mixed
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chain = chain_for(root)
            calls = []
            chain.sh = lambda args, **k: calls.append(args)
            chain.rl_config()
            rd = next(c for c in calls if c[c.index("--out") + 1] == str(root / "rl_rd"))
            self.assertEqual(rd[rd.index("--rows") + 1], "/rl_rows/rows.jsonl")
            self.assertEqual(rd[rd.index("--task-pool") + 1], "reserve")
            self.assertEqual(rd[rd.index("--sft-trajectories") + 1], str(root / "sft" / "sft_trajectories.jsonl"))
            self.assertEqual(rd[rd.index("--exclude") + 1], "/flags.json")
            self.assertEqual(rd[rd.index("--model") + 1], str(Path("/lf/model/checkpoint-1386")))
            calls.clear()
            with jobs(chain, [memorised, learnable]):
                chain.rl()
            checks = chain.state["signal_checks"]
            self.assertEqual([c["ok"] for c in checks], [False, True])
            self.assertEqual(checks[0]["runs"]["rs0"]["success"], 1.0)
            self.assertEqual(checks[0]["runs"]["rs0"]["mixed_share"], 0.0)
            self.assertEqual((checks[1]["runs"]["rd"]["success"], checks[1]["runs"]["rd"]["mixed_share"]), (0.5, 0.5))
            self.assertEqual(checks[1]["model"], str(Path("/lf/model/checkpoint-924")))
            # the missed attempt is kept aside; the restart was configured from the fallback checkpoint
            self.assertTrue((root / "rl_attempts" / "attempt1_rs0_run" / "rollout_data" / "25.jsonl").is_file())
            self.assertEqual({c[c.index("--model") + 1] for c in calls}, {str(Path("/lf/model/checkpoint-924"))})
            self.assertTrue((root / "rl_run_rd" / "checkpoints" / "global_step_50").is_dir())
            md = T.render_markdown(chain.state)
            self.assertIn("没通过", md)
            self.assertIn("rs0 训练成功率 1.000、有成有败的题 0.000（不满足）", md)
        with tempfile.TemporaryDirectory() as tmp:  # a second miss stops the chain
            chain = chain_for(Path(tmp))
            chain.sh = lambda args, **k: None
            with jobs(chain, [memorised, memorised]):
                with self.assertRaises(T.ChainError) as err:
                    chain.rl()
            self.assertIn("missed again", str(err.exception))
            self.assertEqual([c["ok"] for c in chain.state["signal_checks"]], [False, False])

    def test_signal_check_on_mixed_share_only(self):
        """Round 12 batch (RL from the original model): a signal check with only mixed_min ignores the training
        success; without a fallback start model a miss stops RL with a clear reason."""
        from unittest import mock

        from data_gen.interiorgs_v10 import job_sco

        def run(pattern):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                chain = T.Chain({"tr": tmp, "sft_dir": tmp, "endpoint": 50, "gs_root": "/ply",
                                 "rl": {"signal_check": {"at": 25, "mixed_min": 0.10}},
                                 "rl_runs": {"as0": {}, "ad": {}}}, sleep=lambda s: None)

                def prepare(run_dir, gpus, args, job_dir="job", pool=None):
                    (run_dir / job_dir).mkdir(parents=True, exist_ok=True)

                def submit(run_dir, job_dir="job"):
                    (run_dir / job_dir / "submission.json").write_text(json.dumps({"job_id": f"pt-{run_dir.name}-{job_dir}"}))

                def status(run_dir, job_dir="job"):
                    if job_dir == "job_renderer":
                        return {"state": "RUNNING", "finished": False}
                    d = run_dir / "rollout_data"
                    if not d.is_dir():
                        d.mkdir(parents=True)
                        for step in range(1, 26):
                            rows = [{"task_type": "fov_inclusion", "scene_id": f"s{t}", "gts": None, "traj_success": ok}
                                    for t, oks in enumerate(pattern) for ok in oks]
                            (d / f"{step}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
                        return {"state": "RUNNING", "finished": False}
                    (run_dir / "checkpoints" / "global_step_50").mkdir(parents=True, exist_ok=True)
                    return {"state": "SUCCEEDED", "finished": True}

                def delete(run_dir, job_dir="job"):
                    receipt = run_dir / job_dir / "submission.json"
                    if receipt.exists():
                        receipt.replace(receipt.with_name("submission_deleted.json"))

                with mock.patch.multiple(job_sco, prepare=prepare, submit=submit, status=status, delete=delete):
                    try:
                        chain.rl()
                        return chain.state["signal_checks"], None
                    except T.ChainError as exc:
                        return chain.state["signal_checks"], str(exc)

        # 1 success in 80 episodes (1%), but 1 of 10 tasks mixed (10%): passes
        rare = [[True] + [False] * 7] + [[False] * 8] * 9
        checks, err = run(rare)
        self.assertIsNone(err)
        self.assertTrue(checks[0]["ok"])
        self.assertAlmostEqual(checks[0]["runs"]["as0"]["success"], 1 / 80, places=4)
        self.assertEqual(checks[0]["runs"]["as0"]["mixed_share"], 0.1)
        checks, err = run([[False] * 8] * 10)  # nothing ever succeeds: no task is mixed
        self.assertFalse(checks[0]["ok"])
        self.assertIn("signal check missed, RL stopped (no fallback start model)", err)

    def test_variants_on_extra_sets_with_sft_epochs(self):
        """Round 11: SFT epoch checkpoints evaluated on a set of training tasks only (plan eval extra_sets)."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for step in (154, 308, 1386):
                (root / "lf_formal" / "model" / f"checkpoint-{step}").mkdir(parents=True)
            chain = T.Chain({"tr": tmp, "sft_dir": tmp, "endpoint": 50, "sft_groups": {"formal": "x.jsonl"},
                             "eval_sets": {"val400": str(root / "val400")},
                             "eval": {"every": 50, "extra_sets": {"train200": str(root / "train200")},
                                      "variants": [{"suffix": "", "sets": ["train200"],
                                                    "models": ["sft_formal_e1", "sft_formal_e3"]}]}},
                            sleep=lambda s: None)
            got = {}
            chain.eval_jobs = lambda step, j: got.update(j)
            chain.tables = lambda: None
            chain.evaluate()
            self.assertEqual(got["train200__sft_formal_e1"][:3], (root / "train200", "sft_formal_e1",
                                                                  str(root / "lf_formal" / "model" / "checkpoint-154")))
            self.assertEqual(got["train200__sft_formal_e3"][2], str(root / "lf_formal" / "model" / "checkpoint-1386"))
            self.assertFalse(any(k.startswith("train200__") and "rl" in k for k in got))  # only the listed models
            self.assertFalse(any(k.startswith("val400__sft_formal") for k in got))
            self.assertIsNone(chain.sft_epoch_model("sft_formal_e9"))

    def test_guard_reason(self):
        """Round 11: the user's stop conditions for the optimisation-package runs."""
        guards = {"kl_max": 0.2, "invalid_max": 0.05, "invalid_window": 3, "success_drop": 0.10, "drop_window": 4,
                  "start_updates": 2}
        line = lambda s, succ=0.5, kl=0.01, inv=0.0, extra="": (  # noqa: E731
            f"(Runner) step:{s} - episode/success_rate:{succ} - actor/kl_loss:np.float64({kl}) - "
            f"transition/invalid_action_rate/mean:{inv}{extra}\n")
        ok = "".join(line(s) for s in range(1, 8))
        self.assertIsNone(T.guard_reason(ok, guards))
        self.assertIsNone(T.guard_reason("", guards))
        self.assertIn("NaN in actor/pg_loss at update 3",
                      T.guard_reason(ok + line(3, extra=" - actor/pg_loss:np.float64(nan)"), guards))
        self.assertIsNone(T.guard_reason(ok + line(8, extra=" - val-core/x:nan"), guards))  # other metrics: not a stop
        self.assertIn("KL to the start model 0.250", T.guard_reason(ok + line(8, kl=0.25), guards))
        self.assertIsNone(T.guard_reason(ok + line(8, kl=0.15), guards))
        # unparseable answers: the mean of the last 3 updates
        self.assertIsNone(T.guard_reason(ok + line(8, inv=0.12), guards))  # (0 + 0 + 0.12) / 3 = 0.04
        self.assertIn("unparseable answers 0.070", T.guard_reason(ok + line(8, inv=0.1) + line(9, inv=0.11), guards))
        # training success 0.5 at the start; 4 consecutive updates below 0.4 stop the run, 3 do not
        low = "".join(line(s, succ=0.35) for s in range(8, 12))
        self.assertIn("training success below the start (0.500", T.guard_reason(ok + low, guards))
        self.assertIsNone(T.guard_reason(ok + low + line(12, succ=0.45), guards))

    def test_package_runs_with_guards(self):
        """Round 11: the package options reach rl-config; a run stopped by a guard does not fail the step and only
        its existing checkpoints are evaluated; the other run goes on to the endpoint."""
        from unittest import mock

        from data_gen.interiorgs_v10 import job_sco

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rl = {"history_mode": "concat", "save_freq": 50, "no_internal_val": True, "task_pool": "reserve",
                  "rows": "/rl_rows.jsonl", "opt_package": "p1", "candidates": 24, "target_groups": 12,
                  "guards": {"kl_max": 0.2}}
            chain = T.Chain({"tr": tmp, "sft_dir": str(root / "sft"), "endpoint": 100, "ppo_template": "/S0.yaml",
                             "gs_root": "/ply", "sft_groups": {"formal": "sft_trajectories.jsonl"}, "rl": rl,
                             "rl_runs": {"ps0": {"reward_preset": "S0"}, "pd": {"reward_preset": "dense"}},
                             "eval_sets": {"val400": str(root / "val400")}, "eval": {"every": 50}}, sleep=lambda s: None)
            chain.state["steps"]["pick_sft"]["decision"] = {"best": "sft_formal_e9", "model": "/lf/e9",
                                                            "select_by": "val400", "scores": {"sft_formal_e9": 0.42}}
            calls = []
            chain.sh = lambda args, **k: calls.append(args)
            chain.rl_config()
            ps0 = next(c for c in calls if c[c.index("--out") + 1] == str(root / "rl_ps0"))
            for flag, value in (("--opt-package", "p1"), ("--candidates", "24"), ("--target-groups", "12"),
                                ("--task-pool", "reserve")):
                self.assertEqual(ps0[ps0.index(flag) + 1], value)
            self.assertNotIn("--rollouts", ps0)

            def prepare(run, gpus, args, job_dir="job", pool=None):
                (run / job_dir).mkdir(parents=True, exist_ok=True)

            def submit(run, job_dir="job"):
                (run / job_dir / "submission.json").write_text(json.dumps({"job_id": f"pt-{run.name}-{job_dir}"}))

            polls = {"n": 0}

            def status(run, job_dir="job"):
                if job_dir == "job_renderer":
                    return {"state": "RUNNING", "finished": False}
                polls["n"] += 1
                logs = run / "logs"
                logs.mkdir(exist_ok=True)
                if run.name == "rl_run_pd":  # KL runs away after its first checkpoint
                    (run / "checkpoints" / "global_step_50" / "actor" / "huggingface").mkdir(parents=True, exist_ok=True)
                    (logs / "rl-1.log").write_text("step:60 - actor/kl_loss:np.float64(0.31)\n")
                    return {"state": "RUNNING", "finished": False}
                (logs / "rl-1.log").write_text("step:1 - actor/kl_loss:np.float64(0.01)\n")
                if polls["n"] > 3:
                    for st in (50, 100):
                        (run / "checkpoints" / f"global_step_{st}" / "actor" / "huggingface").mkdir(parents=True, exist_ok=True)
                    return {"state": "SUCCEEDED", "finished": True}
                return {"state": "RUNNING", "finished": False}

            deleted = []

            def delete(run, job_dir="job"):
                deleted.append((run.name, job_dir))
                receipt = run / job_dir / "submission.json"
                if receipt.exists():
                    receipt.replace(receipt.with_name("submission_deleted.json"))

            with mock.patch.multiple(job_sco, prepare=prepare, submit=submit, status=status, delete=delete):
                chain.rl()  # does not raise: pd was stopped by a guard, ps0 reached the endpoint
            self.assertIn("KL to the start model 0.310", chain.state["guard_stops"]["pd"]["reason"])
            self.assertIn(("rl_run_pd", "job_rl"), deleted)
            self.assertTrue((root / "rl_run_pd" / "STOP_RENDERER").exists())
            self.assertIn("**RL pd 被自动停止**", T.render_markdown(chain.state))
            jobs = {}
            chain.eval_jobs = lambda step, j: jobs.update(j)
            chain.tables = lambda: None
            chain.evaluate()
            self.assertEqual(sorted(jobs), ["val400__pd_050", "val400__ps0_050", "val400__ps0_100"])

    def test_startup_problem_and_job_logs(self):
        """Round 12 (the user's decision 5): which trainer jobs count as failed or stuck at Ray start-up."""
        failed = "ray init kwargs: {}\nException: The current node timed out during startup. This could happen ...\n"
        self.assertIn("ended FAILED at Ray start-up (timed out during startup)",
                      T.startup_problem(failed, "FAILED", True, None, 2400))
        self.assertIsNone(T.startup_problem("Traceback ...\nKeyError: 'x'\n", "FAILED", True, None, 2400))  # other errors
        self.assertIsNone(T.startup_problem(failed + "step:1 - x:1\n", "FAILED", True, None, 2400))  # past the start-up
        self.assertIsNone(T.startup_problem("", "SUCCEEDED", True, None, 2400))
        hung = "INFO worker.py:2024 -- Started a local Ray instance.\n"
        self.assertIn("no output for 41 minutes", T.startup_problem(hung, "RUNNING", False, 41 * 60, 2400))
        self.assertIsNone(T.startup_problem(hung, "RUNNING", False, 39 * 60, 2400))
        self.assertIsNone(T.startup_problem(hung + "(TaskRunner) step:3 - a:1\n", "RUNNING", False, 3600, 2400))
        self.assertIsNone(T.startup_problem("", "PENDING", False, None, 2400))  # queued: not counted
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "logs").mkdir()
            for name in ("rl-20261017T115911Z-pt-1-worker-0.log", "rl-20261017T121226Z-pt-2-worker-0.log",
                         "renderer-20261017T121226Z-pt-3-worker-0.log"):
                (d / "logs" / name).write_text("x")
            got = [p.name for p in T.job_logs(d, "rl", "2026-10-17T12:12:00+00:00")]
            self.assertEqual(got, ["rl-20261017T121226Z-pt-2-worker-0.log"])  # the earlier attempt's log is not this job's
            self.assertEqual(len(T.job_logs(d, "rl", None)), 2)
            self.assertEqual(T.job_logs(d / "none", "rl", None), [])

    def test_startup_resubmission(self):
        """Round 12: a trainer that fails at Ray start-up, then one silent for 40 minutes, are deleted and submitted
        again (the run then reaches its endpoint); a run silent a third time after 2 resubmissions is given up and
        the step fails; the render services are not touched; PROGRESS.md lists the resubmissions."""
        import datetime
        import time
        from unittest import mock

        from data_gen.interiorgs_v10 import job_sco

        def run_chain(root, scripts):
            """scripts: {run name: [behaviour of its 1st, 2nd, ... trainer job]}, behaviours "ray_fail", "hang", "ok"."""
            clock = {"t": time.time()}
            chain = T.Chain({"tr": str(root), "sft_dir": str(root / "sft"), "endpoint": 50, "gs_root": "/ply",
                             "rl": {"standard": "v10_p1_S0"}, "rl_runs": {n: {} for n in scripts}},
                            sleep=lambda s: clock.__setitem__("t", clock["t"] + 20 * 60), clock=lambda: clock["t"])
            submitted = {n: 0 for n in scripts}
            deleted = []

            def prepare(run, gpus, args, job_dir="job", pool=None):
                (run / job_dir).mkdir(parents=True, exist_ok=True)

            def submit(run, job_dir="job"):
                name = run.name[len("rl_run_"):]
                if job_dir == "job_rl":
                    submitted[name] += 1
                job_id = f"pt-{name}-{job_dir}-{submitted[name]}"
                (run / job_dir / "submission.json").write_text(json.dumps(
                    {"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "job_id": job_id}))

            def status(run, job_dir="job"):
                job_id = json.loads((run / job_dir / "submission.json").read_text())["job_id"]
                if job_dir == "job_renderer":
                    done = (run / "STOP_RENDERER").exists()
                    return {"job_id": job_id, "state": "SUCCEEDED" if done else "RUNNING", "finished": done}
                name = run.name[len("rl_run_"):]
                kind = scripts[name][submitted[name] - 1]
                log = run / "logs" / f"rl-{datetime.datetime.now(datetime.timezone.utc):%Y%m%dT%H%M%SZ}-{job_id}-worker-0.log"
                log.parent.mkdir(exist_ok=True)
                if not any(p.name.endswith(f"{job_id}-worker-0.log") for p in log.parent.glob("rl-*.log")):
                    log.write_text({"ray_fail": "Exception: The current node timed out during startup.\n",
                                    "hang": "INFO worker.py:2024 -- Started a local Ray instance.\n",
                                    "ok": "(TaskRunner) step:1 - episode/success_rate:0.4\n"}[kind])
                if kind == "ray_fail":
                    return {"job_id": job_id, "state": "FAILED", "finished": True}
                if kind == "ok":
                    (run / "checkpoints" / "global_step_50").mkdir(parents=True, exist_ok=True)
                    return {"job_id": job_id, "state": "SUCCEEDED", "finished": True}
                return {"job_id": job_id, "state": "RUNNING", "finished": False}

            def delete(run, job_dir="job"):
                receipt = run / job_dir / "submission.json"
                if receipt.exists():
                    deleted.append(json.loads(receipt.read_text())["job_id"])
                    receipt.replace(receipt.with_name(f"submission_{deleted[-1]}_deleted.json"))

            with mock.patch.multiple(job_sco, prepare=prepare, submit=submit, status=status, delete=delete):
                try:
                    chain.rl()
                    err = None
                except T.ChainError as exc:
                    err = exc
            return chain, submitted, deleted, err

        with tempfile.TemporaryDirectory() as tmp:
            chain, submitted, deleted, err = run_chain(Path(tmp), {"a": ["ray_fail", "hang", "ok"], "b": ["ok"]})
            self.assertIsNone(err)
            self.assertEqual(submitted, {"a": 3, "b": 1})
            retries = chain.state["startup_retries"]["a"]
            self.assertEqual([r["job_id"] for r in retries], ["pt-a-job_rl-1", "pt-a-job_rl-2"])
            self.assertEqual([r["new_job_id"] for r in retries], ["pt-a-job_rl-2", "pt-a-job_rl-3"])
            self.assertIn("timed out during startup", retries[0]["reason"])
            self.assertIn("no output for", retries[1]["reason"])
            self.assertNotIn("b", chain.state["startup_retries"])
            self.assertFalse(any("job_renderer" in j for j in deleted[:2]))  # the render service kept running
            self.assertIn("**RL a 启动失败，第 2 次自动重新提交**", T.render_markdown(chain.state))
        with tempfile.TemporaryDirectory() as tmp:
            chain, submitted, _, err = run_chain(Path(tmp), {"c": ["hang", "hang", "hang"], "b": ["ok"]})
            self.assertEqual(submitted, {"c": 3, "b": 1})  # 2 resubmissions, then given up
            self.assertIn("start-up: no output for", str(err))
            self.assertTrue((Path(tmp) / "rl_run_b" / "checkpoints" / "global_step_50").is_dir())

    def test_standard_and_seed_reach_rl_config(self):
        """Round 12: plan rl "standard" and "seed" reach rl-config (no --ppo-template then, and --reward-preset only
        when a run names one)."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chain = T.Chain({"tr": tmp, "sft_dir": str(root / "sft"), "endpoint": 300, "gs_root": "/ply",
                             "rl": {"standard": "v10_p1_S0", "seed": 20261019, "rows": "/rows.jsonl", "task_pool": "reserve"},
                             "rl_runs": {"ps0": {}, "pd": {"rl": {"standard": "v10_p1_D"}}}}, sleep=lambda s: None)
            chain.state["steps"]["pick_sft"]["decision"] = {"best": "sft_formal_e9", "model": "/lf/e9"}
            calls = {}
            chain.sh = lambda args, **k: calls.__setitem__(args[args.index("--out") + 1], args)
            chain.rl_config()
            ps0, pd = calls[str(root / "rl_ps0")], calls[str(root / "rl_pd")]
            self.assertEqual((ps0[ps0.index("--standard") + 1], pd[pd.index("--standard") + 1]), ("v10_p1_S0", "v10_p1_D"))
            self.assertEqual((ps0[ps0.index("--seed") + 1], pd[pd.index("--seed") + 1]), ("20261019", "20261019"))
            for args in (ps0, pd):
                self.assertNotIn("--ppo-template", args)
                self.assertNotIn("--reward-preset", args)
                self.assertNotIn("--curriculum-easy-updates", args)
            # round 12 batch: the easy-first options reach rl-config
            chain.plan["rl"].update(curriculum_easy_updates=50, curriculum_max_dstar=3)
            chain.rl_config()
            ps0 = calls[str(root / "rl_ps0")]
            self.assertEqual((ps0[ps0.index("--curriculum-easy-updates") + 1], ps0[ps0.index("--curriculum-max-dstar") + 1]),
                             ("50", "3"))

    def test_eval_only_plan(self):
        """Round 12: an eval_only plan runs check_code and evaluate only, on the listed models and variants."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            plan = {"tr": tmp, "sft_dir": tmp, "eval_only": True,
                    "eval_sets": {"val_oa": str(root / "val_oa"), "test_oa": str(root / "test_oa")},
                    "eval": {"models": {"sft_formal_e9": {"model": "/m/e9"}, "r1_final": {"model": "/m/r1", "prompt_format": "r1"}},
                             "extra_sets": {"val400": str(root / "val400")},
                             "variants": [{"suffix": "_s4", "sets": ["val400"], "models": ["sft_formal_e9"],
                                           "options": ["--samples", "4"]}]}}
            chain = T.Chain(plan, sleep=lambda s: None)
            chain.check_code = lambda: None
            got = {}
            chain.eval_jobs = lambda step, j: got.update(j)
            chain.tables = lambda: None
            self.assertEqual(chain.run(), 0)
            self.assertEqual(sorted(got), ["test_oa__r1_final", "test_oa__sft_formal_e9", "val400__sft_formal_e9_s4",
                                           "val_oa__r1_final", "val_oa__sft_formal_e9"])
            self.assertEqual(got["val_oa__r1_final"][3], "r1")
            self.assertEqual(got["val400__sft_formal_e9_s4"][4], ["--samples", "4"])
            steps = chain.state["steps"]
            self.assertEqual(steps["rl"]["note"], "not planned (eval_only)")
            self.assertTrue(all(steps[s]["state"] == "done" for s in T.STEPS))

    def test_code_changes(self):
        """Round 12: uncommitted changes under docs/ (another session's documents) do not stop the chain; any other
        tracked change does; the verl submodule pointer is ignored as before."""
        changed, ignored = T.code_changes([" M verl", " M docs/interiorgs_qa/audit/site_20261014.sh"])
        self.assertEqual((changed, ignored), ([], [" M docs/interiorgs_qa/audit/site_20261014.sh"]))
        changed, _ = T.code_changes([" M data_gen/interiorgs_v10/train_chain.py", "M  docs/x.md",
                                     "R  docs/a.md -> data_gen/a.py", " D vagen/env.py"])
        self.assertEqual(changed, [" M data_gen/interiorgs_v10/train_chain.py", "R  docs/a.md -> data_gen/a.py",
                                   " D vagen/env.py"])
        self.assertEqual(T.code_changes([]), ([], []))

    def test_sft_only_chain_with_chosen_epochs(self):
        """Round 12 batch (the data-scale curve): no sft-records call, only epochs 2/4/.. evaluated, only on the
        validation set, the pick among them, then the pick alone on the test set; no RL step, no RL checkpoint."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for e in range(1, 5):
                (root / "lf_s100" / "model" / f"checkpoint-{e * 10}").mkdir(parents=True)
            (root / "sft").mkdir()
            (root / "sft" / "traj_s100.jsonl").write_text("{}\n")
            plan = {"tr": tmp, "sft_dir": str(root / "sft"), "commit": "x", "model": "/base", "sft_epochs": 4,
                    "sft_groups": {"s100": "traj_s100.jsonl"}, "sft_records": False, "sft_only": True,
                    "sft_eval_epochs": [2, 4], "sft_eval_sets": ["val"], "select_by": "val",
                    "eval_sets": {"val": str(root / "val"), "test": str(root / "test")},
                    "eval": {"variants": [{"suffix": "", "sets": ["test"], "models": ["sft_pick"]}]}}
            chain = T.Chain(plan, sleep=lambda s: None)
            calls = []

            def sh(args, **k):
                calls.append(args)
                if "prepare_active_spatial_sft.py" in args:
                    (root / "lf_s100" / "train.yaml").write_text("x")

            chain.sh = sh
            chain.sft_data()
            self.assertFalse(any("sft-records" in a for a in calls))  # the conversation files exist already
            self.assertTrue(any(str(root / "sft" / "traj_s100.jsonl") in a for a in calls))
            jobs = {}
            chain.eval_jobs = lambda step, j: jobs.update(j)
            chain.tables = lambda: None
            chain.sft_eval()
            self.assertEqual(sorted(jobs), ["val__sft_s100_e2", "val__sft_s100_e4"])
            for name, rate in (("sft_s100_e2", 0.40), ("sft_s100_e4", 0.38)):
                d = root / "val" / name
                d.mkdir(parents=True)
                (d / "done.json").write_text("{}")
                rows = [{"success": i < rate * 100, "task_type": "fov_inclusion", "d_star": 3, "turns": 1, "actions": 1,
                         "collisions": 0, "invalid_turns": 0, "actions_to_success": 1} for i in range(100)]
                (d / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            chain.pick_sft()
            decision = chain.state["steps"]["pick_sft"]["decision"]
            self.assertEqual((decision["best"], decision["model"]), ("sft_s100_e2", str(root / "lf_s100" / "model" / "checkpoint-20")))
            self.assertEqual(decision["other_sets"], {})  # no test numbers for the unpicked epochs
            jobs.clear()
            chain.evaluate()
            self.assertEqual(sorted(jobs), ["test__sft_s100_e2"])  # only the pick on the test set, no RL checkpoint
            for step in T.STEPS:  # the RL steps are skipped by run()
                setattr(chain, step, lambda: None)
            chain.state["steps"] = {s: {"state": "pending"} for s in T.STEPS}
            self.assertEqual(chain.run(), 0)
            self.assertEqual(chain.state["steps"]["rl"]["note"], "not planned (sft_only)")

    def test_auto_pool_and_queue_times(self):
        """Round 12 batch: plan "pool": "auto" asks job_sco.pick_pool for every new job, an RL run's render service
        and trainer share the first choice; the time a job waited for cards is kept and shown."""
        from unittest import mock

        from data_gen.interiorgs_v10 import job_sco

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chain = T.Chain({"tr": tmp, "sft_dir": tmp, "pool": "auto"}, sleep=lambda s: None)
            picks = iter(["zoetrope", "h800", "zoetrope"])
            with mock.patch.object(job_sco, "pick_pool", side_effect=lambda: next(picks)):
                self.assertEqual(chain.job_pool("p1s0"), "zoetrope")
                self.assertEqual(chain.job_pool("p1s0"), "zoetrope")  # the run keeps its pool
                self.assertEqual(chain.job_pool("p1d"), "h800")
                self.assertEqual(chain.job_pool(), "zoetrope")  # an evaluation job: picked anew
            self.assertEqual(T.Chain({"tr": tmp, "sft_dir": tmp, "pool": "h800"}).job_pool("x"), "h800")
            run = root / "rl_run_p1s0"
            (run / "job_rl").mkdir(parents=True)
            (run / "job_rl" / "prepared.json").write_text(json.dumps({"pool": "zoetrope"}))
            chain.note_queue("rl/p1s0/rl", run, "job_rl", {"job_id": "pt-1", "create_time": "2026-10-20T01:00:00Z",
                                                          "start_time": "2026-10-20T01:42:00Z"})
            chain.note_queue("rl/p1s0/rl", run, "job_rl", {"job_id": "pt-2", "create_time": None})  # unknown: skipped
            q = chain.state["queue"]
            self.assertEqual(q["pt-1"], {"what": "rl/p1s0/rl", "pool": "zoetrope", "created": "2026-10-20T01:00:00Z",
                                         "started": "2026-10-20T01:42:00Z", "queued_s": 2520})
            self.assertNotIn("pt-2", q)
            self.assertIn("| rl/p1s0/rl | pt-1 | zoetrope | 42 | 已开始 |", T.render_markdown(chain.state))

    def test_retries_of_failed_jobs(self):
        """Round 12 batch (the user's rule): plan "retries": 2 resubmits a failed RL trainer (it resumes), a dead
        render service (with its trainer), an SFT job (model folder moved aside) and an evaluation job (going on from
        its results); a third failure fails the step; without the option nothing is resubmitted."""
        from unittest import mock

        from data_gen.interiorgs_v10 import job_sco

        def rl_chain(root, script, retries=2):
            """script: per run, the outcomes of its successive trainer / renderer jobs."""
            chain = T.Chain({"tr": str(root), "sft_dir": str(root), "endpoint": 50, "gs_root": "/ply", "retries": retries,
                             "rl_runs": {n: {} for n in script}}, sleep=lambda s: None)
            count = {}

            def prepare(run, gpus, args, job_dir="job", pool=None):
                (run / job_dir).mkdir(parents=True, exist_ok=True)

            def submit(run, job_dir="job"):
                key = (run.name, job_dir)
                count[key] = count.get(key, 0) + 1
                (run / job_dir / "submission.json").write_text(json.dumps({"job_id": f"pt-{run.name}-{job_dir}-{count[key]}"}))

            def status(run, job_dir="job"):
                name = run.name[len("rl_run_"):]
                n = count[(run.name, job_dir)]
                kind = script[name][job_dir][min(n, len(script[name][job_dir])) - 1]
                if kind == "ok" and job_dir == "job_rl":
                    (run / "checkpoints" / "global_step_50").mkdir(parents=True, exist_ok=True)
                    return {"job_id": f"x{n}", "state": "SUCCEEDED", "finished": True}
                if kind == "run":
                    if job_dir == "job_renderer" and (run / "STOP_RENDERER").exists():
                        return {"job_id": f"x{n}", "state": "SUCCEEDED", "finished": True}
                    return {"job_id": f"x{n}", "state": "RUNNING", "finished": False}
                return {"job_id": f"x{n}", "state": "FAILED", "finished": True}

            def delete(run, job_dir="job"):
                receipt = run / job_dir / "submission.json"
                if receipt.exists():
                    receipt.replace(receipt.with_name("submission_deleted.json"))

            return chain, count, mock.patch.multiple(job_sco, prepare=prepare, submit=submit, status=status, delete=delete)

        with tempfile.TemporaryDirectory() as tmp:
            chain, count, patch = rl_chain(Path(tmp), {
                "a": {"job_rl": ["fail", "fail", "ok"], "job_renderer": ["run"]},          # the trainer fails twice
                "b": {"job_rl": ["run", "ok"], "job_renderer": ["fail", "run"]}})          # the render service dies
            with patch:
                chain.rl()
            self.assertEqual(count[("rl_run_a", "job_rl")], 3)
            self.assertEqual((count[("rl_run_b", "job_renderer")], count[("rl_run_b", "job_rl")]), (2, 2))
            self.assertEqual([r["what"] for r in chain.state["job_retries"]["rl/a"]], ["trainer", "trainer"])
            self.assertEqual([r["what"] for r in chain.state["job_retries"]["rl/b"]], ["renderer"])
            self.assertIn("**rl/a 失败，第 2 次自动重新提交**", T.render_markdown(chain.state))
        with tempfile.TemporaryDirectory() as tmp:  # a third failure: given up, the step fails
            chain, count, patch = rl_chain(Path(tmp), {"a": {"job_rl": ["fail"] * 3, "job_renderer": ["run"]}})
            with patch, self.assertRaises(T.ChainError):
                chain.rl()
            self.assertEqual(count[("rl_run_a", "job_rl")], 3)
        with tempfile.TemporaryDirectory() as tmp:  # no "retries": as before
            chain, count, patch = rl_chain(Path(tmp), {"a": {"job_rl": ["fail", "ok"], "job_renderer": ["run"]}}, retries=0)
            with patch, self.assertRaises(T.ChainError):
                chain.rl()
            self.assertEqual(count[("rl_run_a", "job_rl")], 1)

        # SFT: the failed run's model folder is kept aside and the job resubmitted
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chain = T.Chain({"tr": tmp, "sft_dir": tmp, "retries": 2, "sft_groups": {"s050": "x.jsonl"}}, sleep=lambda s: None)
            (root / "lf_s050" / "model" / "checkpoint-5").mkdir(parents=True)
            states = iter(["FAILED", "SUCCEEDED"])

            def status(run, job_dir="job"):
                s = next(states)
                if s == "SUCCEEDED":
                    model = root / "lf_s050" / "model"
                    model.mkdir(parents=True, exist_ok=True)
                    (model / "config.json").write_text("{}")
                    (model / "model.safetensors").write_text("")
                return {"job_id": "pt-s", "state": s, "finished": True}

            submitted = []
            with mock.patch.multiple(job_sco, prepare=lambda *a, **k: None, status=status,
                                     submit=lambda run, job_dir="job": (submitted.append(run.name), (run / "job").mkdir(parents=True, exist_ok=True),
                                                                        (run / "job" / "submission.json").write_text('{"job_id": "pt-s"}')),
                                     delete=lambda run, job_dir="job": (run / "job" / "submission.json").unlink(missing_ok=True)):
                chain.sft()
            self.assertEqual(submitted, ["sft_s050", "sft_s050"])
            self.assertTrue((root / "lf_s050" / "model_failed_1" / "checkpoint-5").is_dir())

        # evaluation: the rerun goes on from the results written (a cut-off last line dropped)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            chain = T.Chain({"tr": tmp, "sft_dir": tmp, "retries": 2}, sleep=lambda s: None)
            (root / "m").mkdir()
            (root / "m" / "config.json").write_text("{}")
            res = root / "val" / "r1"
            res.mkdir(parents=True)
            (res / "results.jsonl").write_text('{"index": 0}\n{"index": 1, "succ')
            states = iter(["FAILED", "SUCCEEDED"])
            submitted = []

            def status(run, job_dir="job"):
                s = next(states)
                if s == "SUCCEEDED":
                    self.assertEqual((res / "results.jsonl").read_text(), '{"index": 0}\n')  # the partial line went
                    (res / "done.json").write_text("{}")
                return {"job_id": "pt-e", "state": s, "finished": True}

            def submit(run, job_dir="job"):
                submitted.append(run.name)
                (run / "job" / "submission.json").write_text("{}")

            with mock.patch.multiple(job_sco, status=status, submit=submit,
                                     prepare=lambda run, *a, **k: (run / "job").mkdir(parents=True, exist_ok=True),
                                     delete=lambda run, job_dir="job": (run / "job" / "submission.json").unlink(missing_ok=True)):
                chain.eval_jobs("evaluate", {"val__r1": (root / "val", "r1", str(root / "m"), None, [])})
            self.assertEqual(submitted, ["val__r1", "val__r1"])
            self.assertEqual(len(chain.state["job_retries"]["evaluate/val__r1"]), 1)

    def test_wait_jobs(self):
        with tempfile.TemporaryDirectory() as tmp:
            chain = T.Chain({"tr": tmp, "sft_dir": tmp}, sleep=lambda s: None)
            seq = iter([{"state": "RUNNING", "finished": False}, {"state": "SUCCEEDED", "finished": True}])
            done = []
            chain.wait_jobs("sft", {"sft": lambda: next(seq)}, "sft", on_finish=lambda: done.append(1))
            self.assertEqual(done, [1])
            with self.assertRaises(T.ChainError):  # the render service dies while PPO runs
                chain.wait_jobs("rl", {"rl": lambda: {"state": "RUNNING", "finished": False},
                                       "renderer": lambda: {"state": "FAILED", "finished": True}}, "rl")


if __name__ == "__main__":
    unittest.main()
