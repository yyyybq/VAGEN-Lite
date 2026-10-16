from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from r1_projective_medium_selector_v4_projection_frontier import frontier_priority, projection_quality


def _pose(x: float = 0.0) -> np.ndarray:
    pose = np.eye(4)
    pose[:3, 2] = [1.0, 0.0, 0.0]
    pose[0, 3] = x
    return pose


def _item() -> dict:
    return {"target_region": {"params": {"object_a_center": [2.0, -0.5, 1.0], "object_b_center": [2.0, 0.5, 1.0]}}}


def _observation(inside: float, area: float, *, visible: bool = True) -> dict:
    raw_width = 100.0
    clipped_width = raw_width * inside
    obj = {"bbox_raw": [0.0, 0.0, raw_width, area / raw_width],
           "bbox": [0.0, 0.0, clipped_width, area / raw_width],
           "center_in_front": True, "visible": visible}
    return {"visual_metrics": {"objects": [dict(obj), dict(obj)]}}


def test_projection_quality_prefers_better_inside_fraction_within_depth() -> None:
    pose = _pose()
    low = projection_quality(_item(), pose, _observation(0.25, 1000.0))
    high = projection_quality(_item(), pose, _observation(0.75, 1000.0))
    assert frontier_priority(4, high, pose) < frontier_priority(4, low, pose)


def test_depth_remains_primary_over_projection_quality() -> None:
    pose = _pose()
    low = projection_quality(_item(), pose, _observation(0.10, 100.0, visible=False))
    high = projection_quality(_item(), pose, _observation(1.0, 5000.0))
    assert frontier_priority(3, low, pose) < frontier_priority(4, high, pose)


def test_orientation_alignment_breaks_equal_projection_tie() -> None:
    aligned_pose = _pose()
    away_pose = np.array(aligned_pose, copy=True)
    away_pose[:3, 2] = [-1.0, 0.0, 0.0]
    observation = _observation(0.75, 1000.0)
    aligned = projection_quality(_item(), aligned_pose, observation)
    away = projection_quality(_item(), away_pose, observation)
    assert aligned["pair_alignment_cosine"] > away["pair_alignment_cosine"]
    assert frontier_priority(4, aligned, aligned_pose) < frontier_priority(4, away, away_pose)
