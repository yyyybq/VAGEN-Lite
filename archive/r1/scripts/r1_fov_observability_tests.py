#!/usr/bin/env python3
from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from r1_fov_observability import evaluate_fov_path, evaluate_fov_path_v2


def frame(*, dual: bool = True) -> dict:
    return {
        "canonical_metric": {"success": False},
        "dual_target_observation": dual,
        "wall_like": False,
        "wall_clear": True,
    }


def v2_frame(*, inside: float = 0.0002) -> dict:
    """A deliberately clipped but visually supported FOV frame."""
    def obj() -> dict:
        return {
            "bbox_area_px": 7000.0,
            "inside_frame_fraction": inside,
            "gates": {"projectable": True},
            "rgb_patch": {
                "entropy_bits_32bin": 3.8,
                "edge_fraction_gt8": 0.08,
                "clipped_luminance_fraction": 0.02,
            },
        }
    return {
        "canonical_metric": {"success": False},
        "frame_rgb": {"pixel_count": 64, "mean_luminance": 100.0, "luminance_std": 20.0},
        "objects": [obj(), obj()],
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

    def test_v2_accepts_intentionally_clipped_but_textured_fov_keyframes(self) -> None:
        path = [
            {"c2w": np.eye(4).tolist(), "action": None},
            {"c2w": (np.eye(4) + np.diag([0.0, 0.0, 0.0, 0.0])).tolist(), "action": "move_backward"},
        ]
        # Give the second pose a real translation so the pose-integrity check
        # cannot mistake the fixture for an action that was not executed.
        path[1]["c2w"][0][3] = 0.3
        images = [Image.new("RGB", (8, 8), color=(100, 90, 80)), Image.new("RGB", (8, 8), color=(80, 90, 100))]
        with patch("r1_fov_observability.frame_observability", side_effect=[v2_frame(), v2_frame()]), \
             patch("r1_fov_observability.score_observation", side_effect=[{}, {}]), \
             patch("r1_fov_observability.canonical_fov", side_effect=[{"success": False}, {"success": True}]):
            result = evaluate_fov_path_v2({}, path, images, None)
        self.assertTrue(result["passed"])
        self.assertEqual(result["keyframe_dual_target_steps"], [0, 1])

    def test_v2_rejects_reused_rgb_for_distinct_pose(self) -> None:
        path = [{"c2w": np.eye(4).tolist(), "action": None},
                {"c2w": np.eye(4).tolist(), "action": "move_backward"}]
        path[1]["c2w"][0][3] = 0.3
        images = [Image.new("RGB", (8, 8), color=(100, 90, 80)), Image.new("RGB", (8, 8), color=(100, 90, 80))]
        with patch("r1_fov_observability.frame_observability", side_effect=[v2_frame(), v2_frame()]), \
             patch("r1_fov_observability.score_observation", side_effect=[{}, {}]), \
             patch("r1_fov_observability.canonical_fov", side_effect=[{"success": False}, {"success": True}]):
            result = evaluate_fov_path_v2({}, path, images, None)
        self.assertFalse(result["passed"])
        self.assertIn("step_1_renderer_image_reused_for_distinct_pose", result["reasons"])


if __name__ == "__main__":
    unittest.main()
