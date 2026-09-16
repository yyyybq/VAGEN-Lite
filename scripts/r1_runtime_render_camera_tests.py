#!/usr/bin/env python3
"""Regression checks for the formal R1 runtime render-camera contract."""

from __future__ import annotations

import numpy as np

from vagen.envs.active_spatial.canonical_camera import CANONICAL_CAMERA_H1_RESIZE_V1
from vagen.envs.active_spatial.canonical_task_metrics import CANONICAL_TASK_METRIC_VERSION
from vagen.envs.active_spatial.env import runtime_render_camera_parameters


def main() -> None:
    native_K = np.array(
        [[320.0, 0.0, 320.0], [0.0, 320.0, 240.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    native_before = native_K.copy()
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, 3] = [1.25, -2.5, 1.5]
    canonical = {
        "task_type": "projective_relations",
        "canonical_task_metric_version": CANONICAL_TASK_METRIC_VERSION,
        "camera_model_version": "canonical_h1_from_frozen_candidate_intrinsics_and_pose",
        "init_camera": {"intrinsics": native_K.tolist()},
    }
    expected = np.array(
        [[128.0, 0.0, 128.0], [0.0, 512.0 / 3.0, 128.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    first_K, first_w2c = runtime_render_camera_parameters(
        canonical, c2w, native_K, (256, 256)
    )
    second_K, second_w2c = runtime_render_camera_parameters(
        canonical, c2w, first_K, (256, 256)
    )
    assert np.allclose(first_K, expected)
    assert np.allclose(second_K, expected), "H1 K must not be resized a second time"
    assert np.array_equal(native_K, native_before), "native K must remain immutable"
    assert np.allclose(first_w2c, np.linalg.inv(c2w))
    assert np.allclose(second_w2c, first_w2c)

    legacy_K, legacy_w2c = runtime_render_camera_parameters(
        {"task_type": "projective_relations"}, c2w, native_K, (256, 256)
    )
    assert np.array_equal(legacy_K, native_K), "historical rows must retain legacy K"
    assert np.allclose(legacy_w2c, np.linalg.inv(c2w))

    wrong_version = dict(canonical, camera_model_version="historical_v46_unscaled_env256")
    try:
        runtime_render_camera_parameters(wrong_version, c2w, native_K, (256, 256))
    except ValueError as error:
        assert "camera_model_version" in str(error)
    else:
        raise AssertionError("canonical rows with a non-H1 camera must fail closed")

    print("PASS: canonical H1 runtime render K, no double scaling, legacy compatibility")


if __name__ == "__main__":
    main()
