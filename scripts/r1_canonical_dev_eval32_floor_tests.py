#!/usr/bin/env python3
"""Small CPU regressions for the floor-diagnostic response transformation."""
from __future__ import annotations

import unittest

from r1_run_canonical_dev_eval32_floor_diagnostics import first_action_only_response
from vagen.envs.active_spatial.env import ActiveSpatialEnv


class ParserHarness:
    _allowed_actions = {
        "move_forward", "move_backward", "move_left", "move_right", "turn_left", "turn_right",
    }

    _default_parse_func = ActiveSpatialEnv._default_parse_func


class FirstActionOnlyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.parser = ParserHarness()

    def parse(self, value: str):
        return self.parser._default_parse_func(value, action_sep="|", max_actions=5)

    def test_preserves_first_and_discards_remainder(self) -> None:
        raw = "<think>turn then move</think><action>turn_left|move_forward|move_left|</action>"
        original = self.parse(raw)
        self.assertEqual(original["actions"], ["turn_left", "move_forward", "move_left"])
        transformed = first_action_only_response(raw, original["actions"][0])
        self.assertEqual(self.parse(transformed)["actions"], ["turn_left"])
        self.assertIn("<think>turn then move</think>", transformed)

    def test_case_insensitive_tag(self) -> None:
        raw = "<think>x</think><ACTION>move_right|turn_right|</ACTION>"
        original = self.parse(raw)
        self.assertEqual(original["actions"], ["move_right", "turn_right"])
        self.assertEqual(self.parse(first_action_only_response(raw, "move_right"))["actions"], ["move_right"])

    def test_invalid_output_is_not_transformable(self) -> None:
        raw = "<think>no action follows</think>"
        self.assertEqual(self.parse(raw)["actions"], [])
        with self.assertRaises(ValueError):
            first_action_only_response(raw, "move_forward")

    def test_unknown_action_remains_invalid(self) -> None:
        raw = "<think>x</think><action>move_forward|teleport|</action>"
        self.assertEqual(self.parse(raw)["actions"], [])


if __name__ == "__main__":
    unittest.main()
