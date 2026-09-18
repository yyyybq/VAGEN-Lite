#!/usr/bin/env python3
from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from r1_fov_observability import evaluate_fov_path


def frame(*, dual: bool = True) -> dict:
    return {
        "canonical_metric": {"success": False},
        "dual_target_observation": dual,
        "wall_like": False,
        "wall_clear": True,
    }


class FovObservabilityTests(unittest.TestCase):
    def test_initial_failure_terminal_success_with_discernible_targets_passes(self) -> None:
        path = [{"c2w": np.eye(4).tolist(), "action": None},
                {"c2w": np.eye(4).tolist(), "action": "move_backward"}]
        images = [Image.new("RGB", (8, 8)), Image.new("RGB", (8, 8))]
        with patch("r1_fov_observability.frame_observability", side_effect=[frame(), frame()]), \
             patch("r1_fov_observability.score_observation", side_effect=[{}, {}]), \
             patch("r1_fov_observability.canonical_fov",
                   side_effect=[{"success": False}, {"success": True}]):
            result = evaluate_fov_path({}, path, images, None)
        self.assertTrue(result["passed"])
        self.assertEqual(result["reasons"], [])

    def test_terminal_rgb_failure_remains_a_quality_rejection(self) -> None:
        path = [{"c2w": np.eye(4).tolist(), "action": None},
                {"c2w": np.eye(4).tolist(), "action": "move_backward"}]
        images = [Image.new("RGB", (8, 8)), Image.new("RGB", (8, 8))]
        with patch("r1_fov_observability.frame_observability",
                   side_effect=[frame(), frame(dual=False)]), \
             patch("r1_fov_observability.score_observation", side_effect=[{}, {}]), \
             patch("r1_fov_observability.canonical_fov",
                   side_effect=[{"success": False}, {"success": True}]):
            result = evaluate_fov_path({}, path, images, None)
        self.assertFalse(result["passed"])
        self.assertIn("terminal_dual_target_not_discernible", result["reasons"])


if __name__ == "__main__":
    unittest.main()
