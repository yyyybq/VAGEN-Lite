#!/usr/bin/env python3
"""Small deterministic regressions for independent local-action accounting."""
from __future__ import annotations

import unittest

from r1_summarize_independent_local_action_eval import ACTIONS, baseline, projection_change


class IndependentLocalActionAccountingTests(unittest.TestCase):
    def test_uniform_and_fixed_action_baselines(self) -> None:
        rows = [
            {"exact_distance": 1, "optimal_first_actions_audit_only": ["move_forward"]},
            {"exact_distance": 1, "optimal_first_actions_audit_only": ["move_forward", "turn_left"]},
            {"exact_distance": 2, "optimal_first_actions_audit_only": ["move_right"]},
        ]
        result = baseline(rows)
        self.assertEqual(result["overall"]["states"], 3)
        self.assertAlmostEqual(result["overall"]["uniform_six_action_expected_hits"], 4 / 6)
        self.assertEqual(result["overall"]["fixed_action_hits"]["move_forward"], 2)
        self.assertEqual(result["overall"]["fixed_action_hits"]["turn_left"], 1)
        self.assertEqual(result["d2"]["fixed_action_hits"]["move_right"], 1)
        self.assertEqual(set(result["overall"]["fixed_action_hits"]), set(ACTIONS))

    def test_numeric_and_gate_projection_change(self) -> None:
        row = {
            "projection_before": {"relation_margin_px": 2.0, "inside_frame_fraction_min": 0.8},
            "projection_after": {"relation_margin_px": 5.5, "inside_frame_fraction_min": 0.6},
            "canonical_before": {"gates": {"relation": False, "inside_frame": True}},
            "canonical_after": {"gates": {"relation": True, "inside_frame": False}},
            "collision_attempts": 1,
        }
        change = projection_change(row)
        self.assertAlmostEqual(change["relation_margin_delta"], 3.5)
        self.assertAlmostEqual(change["inside_fraction_min_delta"], -0.2)
        self.assertTrue(change["collision"])
        self.assertTrue(change["inside_gate_before"])
        self.assertFalse(change["inside_gate_after"])


if __name__ == "__main__":
    unittest.main()
