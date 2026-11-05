"""The understanding-batch switch (the reward line, 2026-11-05) in rl-config and the chain: off unless asked for,
and then only actor_rollout_ref.actor.und_mix (and the report's und_mix) differ."""

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from data_gen.interiorgs_v10 import train_chain as T
from data_gen.interiorgs_v10 import training_prep as TP
from data_gen.interiorgs_v10.run_scene import RunConfig, _jsonable, generate_scene
from data_gen.interiorgs_v10.testing.synthetic_scene import write_synthetic_scenes


class RlConfigUndMixTest(unittest.TestCase):
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
        cls.pool = cls.root / "pool"  # its manifest and the file it hashes are what rl-config reads
        cls.pool.mkdir()
        (cls.pool / "pool.jsonl").write_text("".join(json.dumps({"und_index": i}) + "\n" for i in range(40)))
        import hashlib

        cls.sha = hashlib.sha256((cls.pool / "pool.jsonl").read_bytes()).hexdigest()
        (cls.pool / "manifest.json").write_text(json.dumps({"rows": 40, "pool_sha256": cls.sha}))

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
        _, rep_none, text_none = self.config("plain", und_mix=None)  # the same folder: the same file, byte for byte
        self.assertEqual((text_none, rep_none), (text0, rep0))
        self.assertNotIn("und_mix", ppo0["actor_rollout_ref"]["actor"])
        self.assertNotIn("und_mix", rep0)
        ppo1, rep1, _ = self.config("mixed", und_mix=json.dumps({"pool": str(self.pool), "batch_size": 32}))
        block = ppo1["actor_rollout_ref"]["actor"].pop("und_mix")
        self.assertEqual(block, {"enable": True, "pool": str(self.pool.resolve()), "interval": 2, "weight": 0.1,
                                 "batch_size": 32, "micro_batch_size": 1, "seed": 20261105, "tokenizer": "/m/start",
                                 "log_dir": str((self.root / "runs" / "same").resolve() / "und_mix"), "pool_rows": 40,
                                 "pool_sha256": self.sha})
        self.assertEqual(rep1.pop("und_mix"), block)
        # file paths aside (the out folders differ), the two configs are the same
        dump = lambda d: json.dumps(d, sort_keys=True).replace("/mixed/", "/X/").replace("/plain/", "/X/")  # noqa: E731
        self.assertEqual(dump(ppo1), dump(ppo0))
        self.assertEqual(dump(rep1), dump(rep0))

    def test_bad_requests_stop_rl_config(self):
        # unknown key, no pool, more samples than rows (also the default 128 here), and 12 samples on 8 GPUs
        for bad in ({"pool": str(self.pool), "lr": 1}, {"batch_size": 32}, {"pool": str(self.pool), "batch_size": 64},
                    {"pool": str(self.pool)}, {"pool": str(self.pool), "batch_size": 12}):
            with self.assertRaises(SystemExit, msg=bad):
                self.config("bad", und_mix=json.dumps(bad))
        # a .json file holding the request works like the request itself
        req = self.root / "req.json"
        req.write_text(json.dumps({"pool": str(self.pool), "batch_size": 32}))
        ppo, _, _ = self.config("from_file", und_mix=str(req))
        self.assertEqual(ppo["actor_rollout_ref"]["actor"]["und_mix"]["batch_size"], 32)


class ChainUndMixTest(unittest.TestCase):
    def test_plan_rl_und_mix_reaches_rl_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            req = {"pool": "/pools/d", "batch_size": 32}
            chain = T.Chain({"tr": tmp, "sft_dir": str(root / "sft"), "endpoint": 300, "gs_root": "/ply",
                             "rl": {"standard": "v10_p1_S0", "seed": 20261019, "rows": "/rows.jsonl"},
                             "rl_runs": {"plain": {}, "mixed": {"rl": {"standard": "v10_p1_Dcond", "und_mix": req}}}},
                            sleep=lambda s: None)
            chain.state["steps"]["pick_sft"]["decision"] = {"best": "sft_e9", "model": "/lf/e9"}
            calls = {}
            chain.sh = lambda args, **k: calls.__setitem__(args[args.index("--out") + 1], args)
            chain.rl_config()
            plain, mixed = calls[str(root / "rl_plain")], calls[str(root / "rl_mixed")]
            self.assertNotIn("--und-mix", plain)
            self.assertEqual(json.loads(mixed[mixed.index("--und-mix") + 1]), req)
            # without the flag the two runs' arguments differ only by name, folders and standard
            strip = [a for a in mixed]
            i = strip.index("--und-mix")
            del strip[i:i + 2]
            self.assertEqual(len(strip), len(plain))
            # a request given as a file's path reaches rl-config as that path
            chain.plan["rl_runs"]["mixed"]["rl"]["und_mix"] = "/plans/und_mix.json"
            chain.rl_config()
            mixed = calls[str(root / "rl_mixed")]
            self.assertEqual(mixed[mixed.index("--und-mix") + 1], "/plans/und_mix.json")


if __name__ == "__main__":
    unittest.main()
