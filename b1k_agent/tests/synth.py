"""SYNTHETIC observation fixtures with the official evaluator's keys, shapes and dtypes
(OmniGibson/omnigibson/eval/evaluator.py _preprocess_obs/_batch_obs at v3.9.3-post2).

These are NOT recorded simulator observations. They exercise wiring only; a real recorded episode
(scripts/record_obs_fixture.py on server 105) replaces them for the red-line tests once available.
"""

from __future__ import annotations

import numpy as np

ROBOT = "robot_r1"
CAMS = {"head": "zed_link", "left_wrist": "left_realsense_link", "right_wrist": "right_realsense_link"}


def synth_obs(step: int, num_envs: int = 1, res=(224, 224), depth: bool = False, task_id: int = 0, seed: int = 0,
              extra: dict | None = None) -> dict:
    rng = np.random.default_rng(seed * 100003 + step)
    obs = {}
    for role, link in CAMS.items():
        h, w = res
        obs[f"{ROBOT}::{ROBOT}:{link}:Camera:0::rgb"] = rng.integers(0, 255, size=(num_envs, h, w, 4), dtype=np.uint8)
        if depth:
            obs[f"{ROBOT}::{ROBOT}:{link}:Camera:0::depth_linear"] = rng.uniform(0.2, 5.0, size=(num_envs, h, w)).astype(np.float32)
    proprio = np.zeros((num_envs, 61), np.float32)
    proprio[:, 3:10] = 0.1 * np.sin(step / 30.0)  # left arm moves slowly
    proprio[:, 24:26] = 0.05
    proprio[:, 49:51] = 0.05
    proprio[:, 53:57] = np.array([1.025, -1.45, -0.47, 0.0], np.float32)
    obs[f"{ROBOT}::proprio"] = proprio
    obs[f"{ROBOT}::cam_rel_poses"] = np.zeros((num_envs, 21), np.float32)
    obs["task_id"] = np.full((num_envs, 1), task_id, np.int64)
    if extra:
        obs.update(extra)
    return obs
