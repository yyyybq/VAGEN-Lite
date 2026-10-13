"""Work-unit budgets (reproducible) and the run_all config files."""

import json
import tempfile
import unittest
from pathlib import Path

from data_gen.interiorgs_v10 import contract as C
from data_gen.interiorgs_v10 import initial_states as I
from data_gen.interiorgs_v10.contract import stable_hash
from data_gen.interiorgs_v10.run_all import _CLI_FIELDS, explicit_dests, load_config, merge_config
from data_gen.interiorgs_v10.run_scene import DEFAULT_WORK_WEIGHTS, RunConfig, _jsonable, generate_scene
from data_gen.interiorgs_v10.tests.test_allocation import _sig
from data_gen.interiorgs_v10.tests.test_difficulty import InitialStatesTest
from data_gen.interiorgs_v10.testing.synthetic_scene import write_synthetic_scenes


class WorkMeterTest(unittest.TestCase):
    def test_counts_and_units(self):
        m = C.WorkMeter()
        m.add("a", 3)
        m.add("b")
        m.add("a")
        self.assertEqual(m.snapshot(), {"a": 4, "b": 1})
        self.assertAlmostEqual(m.units({"a": 0.5, "b": 2.0, "c": 9.0}), 4.0)
        m.reset()
        self.assertEqual(m.units({"a": 1.0}), 0.0)


class EpisodeBudgetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        InitialStatesTest.setUpClass.__func__(cls)  # scene + three Projective entries with endpoints

    def test_work_budget_is_deterministic_and_bounds_the_search(self):
        def run(budget):
            C.WORK.reset()
            return I.episodes_for_scene(self.scene, self.entries, target_per_scene=12, seed=7,
                                        work_budget=budget, work_weights=DEFAULT_WORK_WEIGHTS)

        free, _, rep_free = run(None)
        big, _, rep_big = run(1e9)
        self.assertEqual(_sig(free), _sig(big))
        self.assertFalse(rep_big["timed_out"])
        a, fa, rep_a = run(0.5)
        b, fb, rep_b = run(0.5)
        self.assertEqual(_sig(a), _sig(b))       # same input, same output, whatever the machine load
        self.assertEqual(fa.to_dict(), fb.to_dict())
        self.assertTrue(rep_a["timed_out"])
        self.assertLess(len(a), len(free))


class SceneBudgetTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.paths = write_synthetic_scenes(Path(cls._tmp.name))

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_generate_scene_work_mode_repeats_exactly(self):
        cfg = RunConfig(target_tasks_per_scene=4, budget_mode="work", budget_work_units=3.0)
        r1 = _jsonable(generate_scene(self.paths["two_rooms"], "two_rooms", cfg))
        r2 = _jsonable(generate_scene(self.paths["two_rooms"], "two_rooms", cfg))
        self.assertEqual(stable_hash(r1["rows"]), stable_hash(r2["rows"]))
        self.assertEqual(r1["funnel"], r2["funnel"])
        self.assertEqual({k: v for k, v in r1["summary"].items() if k != "seconds"},
                         {k: v for k, v in r2["summary"].items() if k != "seconds"})
        w = r1["summary"]["work"]
        self.assertEqual(w["budget_units"], 3.0)
        self.assertGreater(w["units_total"], 0.0)
        self.assertTrue(w["endpoint_stage_hit"] or w["initial_state_stage_hit"])  # 3 units is a tiny budget
        self.assertNotIn("work", _jsonable(generate_scene(self.paths["two_rooms"], "two_rooms",
                                                          RunConfig(target_tasks_per_scene=2)))["summary"])
        with self.assertRaises(ValueError):
            generate_scene(self.paths["two_rooms"], "two_rooms", RunConfig(budget_mode="cpu"))

    def test_defaults_keep_the_wall_clock_budget(self):
        cfg = RunConfig()
        self.assertEqual((cfg.budget_mode, cfg.scene_time_budget_s), ("wall_clock", 600.0))


class ConfigFileTest(unittest.TestCase):
    def test_load_and_merge(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cfg.json"
            path.write_text(json.dumps({"run_config": {"reserve_k": 16, "families": ["projective_relations"],
                                                       "budget_mode": "work"}}))
            fields = load_config(path)
            self.assertEqual(fields["families"], ("projective_relations",))
            cli = {field: f"cli_{field}" for field in _CLI_FIELDS.values()}
            merged = merge_config(fields, cli, {"reserve_k"})
            self.assertEqual(merged["reserve_k"], "cli_reserve_k")      # explicit flag wins
            self.assertEqual(merged["budget_mode"], "work")             # file value kept
            self.assertNotIn("diagnostics", merged)                     # neither: RunConfig default
            path.write_text(json.dumps({"run_config": {"no_such_field": 1}}))
            with self.assertRaises(ValueError):
                load_config(path)

    def test_config_files(self):
        import dataclasses

        root = Path(__file__).resolve().parents[1] / "configs"
        old = RunConfig(**load_config(root / "generation_20261011c.json"))
        # the 20261011c file spells out the defaults, so it reproduces that run
        self.assertEqual(dataclasses.asdict(old), dataclasses.asdict(RunConfig()))
        final_path = root / "generation_final_20261012.json"
        if final_path.exists():
            final = RunConfig(**load_config(final_path))
            self.assertEqual(final.budget_mode, "work")
            self.assertEqual(final.terminal_identifiable, "family")
            self.assertTrue(final.family_quota_soft_max)
            self.assertEqual(final.reserve_k, 16)
            self.assertEqual(final.motion_model, "r1")
            self.assertFalse(final.include_fov)
            self.assertEqual(final.work_weights, DEFAULT_WORK_WEIGHTS)  # the defaults are the fitted weights
            from data_gen.interiorgs_v10.run_scene import ignored_object_ids

            self.assertIn("19", ignored_object_ids(final.ignored_objects, "0068_839928"))
        r4_path = root / "generation_r4_20261013.json"
        if r4_path.exists():
            r4 = RunConfig(**load_config(r4_path))
            self.assertEqual(r4.visibility_rule, "face_v1")
            self.assertIsNone(r4.ignored_objects)
            self.assertIn("fov_inclusion", r4.families)
            self.assertIn("fov_inclusion:1:3", r4.family_quotas)
            self.assertTrue(r4.runtime_success_check)
            from data_gen.interiorgs_v10.run_scene import ignored_object_ids

            self.assertIn("19", ignored_object_ids(r4.no_task_objects, "0068_839928"))  # merged wardrobe
            self.assertIn("29", ignored_object_ids(r4.no_task_objects, "0003_839989"))  # an uncertain box
        r5_path = root / "generation_r5_20261014.json"
        if r5_path.exists():
            r5 = RunConfig(**load_config(r5_path))
            self.assertEqual(r5.visibility_rule, "env")
            self.assertEqual(r5.object_preset, "v10_r5")
            from data_gen.interiorgs_v10.objects import get_preset

            self.assertEqual(get_preset(r5.object_preset).referring.rules, "v2")
            # everything else as on 2026-10-13
            r4_fields = dataclasses.asdict(RunConfig(**load_config(r4_path)))
            diff = {k for k, v in dataclasses.asdict(r5).items() if r4_fields[k] != v}
            self.assertEqual(diff, {"visibility_rule", "object_preset"})

    def test_explicit_dests(self):
        import argparse

        ap = argparse.ArgumentParser()
        ap.add_argument("--reserve-k", type=int, default=0)
        ap.add_argument("--budget-mode", default="wall_clock")
        self.assertEqual(explicit_dests(ap, ["--reserve-k", "3"]), {"reserve_k"})
        self.assertEqual(explicit_dests(ap, ["--budget-mode=work"]), {"budget_mode"})


if __name__ == "__main__":
    unittest.main()
