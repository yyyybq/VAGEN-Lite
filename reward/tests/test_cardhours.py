"""reward/cardhours.py on a fake runs root in a temporary folder (never the real reward_runs)."""

import datetime as dt
import json
import os
import tempfile
import unittest

from reward import cardhours

UTC = dt.timezone.utc


def _touch(path, text="", when=None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
    if when is not None:
        ts = when.timestamp()
        os.utime(path, (ts, ts))


class CardHoursTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = self.tmp.name
        t0 = dt.datetime(2026, 11, 1, 10, 0, 0, tzinfo=UTC)
        logs = os.path.join(self.root, "runs", "chain_a", "rl_run_x", "logs")
        # a trainer: 2 hours on 8 cards
        _touch(os.path.join(logs, "rl-20261101T100000Z-pt-aaa-worker-0.log"), "x", t0 + dt.timedelta(hours=2))
        # a render service: its log is written once at start, it ends 3 hours later (renderer_exit.json)
        _touch(os.path.join(logs, "renderer-20261101T100000Z-pt-bbb-worker-0.log"), "x", t0)
        _touch(os.path.join(logs, "renderer-20261101T103000Z-pt-bbb-worker-0.log"), "x", t0)  # same job restarted
        _touch(os.path.join(logs, "renderer_exit.json"),
               json.dumps({"mode": "renderer", "ended_utc": "2026-11-01T13:00:00Z"}))
        # an evaluation: 30 minutes on 1 card
        ev = os.path.join(self.root, "runs", "chain_a", "eval_runs", "val__x_050", "logs")
        _touch(os.path.join(ev, "eval-20261101T120000Z-pt-ccc-worker-0.log"), "x", t0 + dt.timedelta(hours=2.5))
        # a one-card job elsewhere, and files that are not job logs
        jb = os.path.join(self.root, "jobs", "replay", "logs")
        _touch(os.path.join(jb, "run-20261101T100000Z-pt-ddd-worker-0.log"), "x", t0 + dt.timedelta(minutes=6))
        _touch(os.path.join(jb, "notes.log"), "x", t0 + dt.timedelta(hours=9))
        _touch(os.path.join(self.root, "runs", "chain_a", "rl_run_x", "checkpoints", "global_step_1",
                            "rl-20261101T100000Z-pt-eee-worker-0.log"), "x", t0 + dt.timedelta(hours=50))

    def tearDown(self):
        self.tmp.cleanup()

    def test_spans_and_cards(self):
        got = {(os.path.basename(r[0]).split("-")[0], r[4]): (r[3] - r[2]).total_seconds() / 3600.0
               for r in cardhours.spans(self.root)}
        self.assertAlmostEqual(got[("rl", 8)], 2.0)
        self.assertAlmostEqual(got[("renderer", 1)], 3.0)  # first log of the job to renderer_exit.json
        self.assertAlmostEqual(got[("eval", 1)], 0.5)
        self.assertAlmostEqual(got[("run", 1)], 0.1)
        self.assertEqual(len(got), 4)  # checkpoints/ skipped, notes.log not a job log, one span per renderer job

    def test_groups(self):
        stages = {"A": ["runs/chain_a/"], "J": ["jobs/"]}
        total = {}
        for rel, mode, t0, t1, cards in cardhours.spans(self.root):
            g = cardhours.group_of(rel, stages)
            total[g] = total.get(g, 0.0) + (t1 - t0).total_seconds() / 3600.0 * cards
        self.assertAlmostEqual(total["A"], 2.0 * 8 + 3.0 + 0.5)
        self.assertAlmostEqual(total["J"], 0.1)
        self.assertEqual(cardhours.group_of("tables/jobs/x/run-1.log", None), "tables/jobs")
        self.assertEqual(cardhours.group_of("tables/jobs/x/run-1.log", stages), "other")


if __name__ == "__main__":
    unittest.main()
