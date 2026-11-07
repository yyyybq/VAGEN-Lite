"""The table-label switch (the reward line, phase 3, experiment B) in rl-config and the chain: off unless asked for,
and then only actor_rollout_ref.actor.dagger (and the report's dagger) differ."""

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from data_gen.interiorgs_v10 import train_chain as T
from data_gen.interiorgs_v10 import training_prep as TP
from data_gen.interiorgs_v10.run_scene import RunConfig, _jsonable, generate_scene
from data_gen.interiorgs_v10.testing.synthetic_scene import write_synthetic_scenes


class RlConfigDaggerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls._tmp.name)
        paths = write_synthetic_scenes(cls.root / "scenes")
        rows = _jsonable(generate_scene(paths["two_rooms"], "two_rooms", RunConfig(target_tasks_per_scene=4)))["rows"]
        val = [dict(r, scene_id="two_rooms_val", task_id=r["task_id"] + "_val") for r in rows[:2]]
        run = cls.root / "run"
        run.mkdir()
        (run / "tasks.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows + val))
        manifest = cls.root / "split.json"
        manifest.write_text(json.dumps({"version": 2, "train_scenes": ["two_rooms"], "val_scenes": ["two_rooms_val"],
                                        "test_scenes": []}))
        sft = cls.root / "sft"
        TP.cmd_sft_frames(Namespace(run=str(run), split_manifest=str(manifest), splits="train,val", limit=0, out=str(sft)))
        rows = [json.loads(l) for l in (sft / "rows.jsonl").read_text().splitlines() if l.strip()]
        for r in rows:
            r["reserve"] = r["split"] == "train"
        cls.rows = sft / "rows_reserve.jsonl"
        cls.rows.write_text("".join(json.dumps(r) + "\n" for r in rows))
        cls.tables = cls.root / "tables"  # rl-config only checks that the folder holds tables
        cls.tables.mkdir()
        (cls.tables / "t1.npz").write_bytes(b"x")
        import hashlib

        cls.pool = cls.root / "pool"  # a valid understanding pool, to show the two switches refuse each other
        cls.pool.mkdir()
        (cls.pool / "pool.jsonl").write_text("".join(json.dumps({"und_index": i}) + "\n" for i in range(200)))
        sha = hashlib.sha256((cls.pool / "pool.jsonl").read_bytes()).hexdigest()
        (cls.pool / "manifest.json").write_text(json.dumps({"rows": 200, "pool_sha256": sha}))

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def config(self, name, **kw):
        import yaml

        args = dict(rows=str(self.rows), gs_root="/ply", model="/m/start", name="t", standard="v10_p1_Dcond",
                    seed=20261019, run_root=str(self.root / "runs" / "same"), val_limit=1, trial=False,
                    out=str(self.root / name))
        args.update(kw)
        TP.cmd_rl_config(Namespace(**args))
        out = self.root / name
        return (yaml.safe_load((out / "ppo.yaml").read_text()), json.loads((out / "rl_config_report.json").read_text()),
                (out / "ppo.yaml").read_text())

    def test_off_by_default_and_only_the_block_differs(self):
        ppo0, rep0, text0 = self.config("plain")
        _, rep_none, text_none = self.config("plain", dagger=None)  # the same folder: the same file, byte for byte
        self.assertEqual((text_none, rep_none), (text0, rep0))
        self.assertNotIn("dagger", ppo0["actor_rollout_ref"]["actor"])
        self.assertNotIn("dagger", rep0)
        ppo1, rep1, _ = self.config("labels", dagger=json.dumps({"tables": str(self.tables), "audit_only": True}))
        block = ppo1["actor_rollout_ref"]["actor"].pop("dagger")
        self.assertEqual(block, {"enable": True, "tables": str(self.tables.resolve()), "lam": 0.5, "cap": 96,
                                 "turns": "all", "seed": 20261019, "max_per_turn": 5, "audit_only": True,
                                 "table_cache": 8, "tokenizer": "/m/start",
                                 "log_dir": str((self.root / "runs" / "same").resolve() / "dagger")})
        self.assertEqual(rep1.pop("dagger"), block)
        dump = lambda d: json.dumps(d, sort_keys=True).replace("/labels/", "/X/").replace("/plain/", "/X/")  # noqa: E731
        self.assertEqual(dump(ppo1), dump(ppo0))
        self.assertEqual(dump(rep1), dump(rep0))

    def test_bad_requests_stop_rl_config(self):
        t = str(self.tables)
        for bad in ({"tables": t, "lr": 1}, {}, {"tables": str(self.root / "run")}, {"tables": t, "cap": 4},
                    {"tables": t, "turns": "wrong"}, {"tables": t, "lam": -1}):
            with self.assertRaises(SystemExit, msg=bad):
                self.config("bad", dagger=json.dumps(bad))
        self.config("mix_ok", und_mix=json.dumps({"pool": str(self.pool)}))  # the pool alone is fine
        with self.assertRaises(SystemExit) as cm:  # not together with the understanding batches
            self.config("bad", dagger=json.dumps({"tables": t}), und_mix=json.dumps({"pool": str(self.pool)}))
        self.assertIn("together", str(cm.exception))
        req = self.root / "dagger.json"
        req.write_text(json.dumps({"tables": t, "cap": 48}))
        ppo, _, _ = self.config("from_file", dagger=str(req))
        self.assertEqual(ppo["actor_rollout_ref"]["actor"]["dagger"]["cap"], 48)


class ChainDaggerTest(unittest.TestCase):
    def test_plan_rl_dagger_reaches_rl_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            req = {"tables": "/tables/b3000", "lam": 0.5}
            chain = T.Chain({"tr": tmp, "sft_dir": str(root / "sft"), "endpoint": 300, "gs_root": "/ply",
                             "rl": {"standard": "v10_p1_S0", "seed": 20261019, "rows": "/rows.jsonl"},
                             "rl_runs": {"plain": {}, "labels": {"rl": {"dagger": req}}}},
                            sleep=lambda s: None)
            chain.state["steps"]["pick_sft"]["decision"] = {"best": "sft_e9", "model": "/lf/e9"}
            calls = {}
            chain.sh = lambda args, **k: calls.__setitem__(args[args.index("--out") + 1], args)
            chain.rl_config()
            plain, labels = calls[str(root / "rl_plain")], calls[str(root / "rl_labels")]
            self.assertNotIn("--dagger", plain)
            self.assertEqual(json.loads(labels[labels.index("--dagger") + 1]), req)
            strip = list(labels)
            i = strip.index("--dagger")
            del strip[i:i + 2]
            self.assertEqual([a.replace("labels", "plain") for a in strip], plain)


if __name__ == "__main__":
    unittest.main()
