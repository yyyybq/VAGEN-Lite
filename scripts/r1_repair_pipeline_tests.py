#!/usr/bin/env python3
"""Deterministic tests for R1 v4 layout and version gates."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from r1_repair_pipeline import SceneConstraints
from vagen.envs.active_spatial.canonical_camera import build_canonical_camera


def toy_item() -> dict:
    return {
        "scene_id": "toy_scene",
        "target_region": {
            "params": {
                "object_a_center": [1.5, 1.5, 1.0],
                "object_b_center": [2.0, 1.5, 1.0],
                "min_distance": 0.75,
            }
        },
    }


def main() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        scene = root / "toy_scene"
        scene.mkdir()
        (scene / "structure.json").write_text(
            json.dumps(
                {
                    "rooms": [
                        {"profile": [[0.0, 0.0], [4.0, 0.0], [4.0, 4.0], [0.0, 4.0]]}
                    ],
                    "walls": [],
                }
            )
        )
        (scene / "labels.json").write_text(
            json.dumps(
                [
                    {
                        "ins_id": "object_1",
                        "label": "cabinet",
                        "bounding_box": [
                            {"x": x, "y": y, "z": z}
                            for x in (1.0, 2.0)
                            for y in (1.0, 2.0)
                            for z in (0.0, 2.0)
                        ],
                    }
                ]
            )
        )

        constraints = SceneConstraints(root, min_wall_clearance=0.5, object_margin=0.2)
        item = toy_item()
        safe = constraints.validate(item, np.array([3.0, 2.0, 1.5]), check_pair_distance=True)
        assert safe["success"], safe
        outside = constraints.validate(item, np.array([5.0, 2.0, 1.5]), check_pair_distance=True)
        assert not outside["gates"]["inside_room"], outside
        near_wall = constraints.validate(item, np.array([0.1, 3.0, 1.5]), check_pair_distance=True)
        assert not near_wall["gates"]["wall_clearance"], near_wall
        collision = constraints.validate(item, np.array([1.5, 1.5, 1.5]), check_pair_distance=False)
        assert not collision["gates"]["object_collision_free"], collision
        too_close = constraints.validate(item, np.array([2.25, 1.5, 2.5]), check_pair_distance=True)
        assert not too_close["gates"]["pair_min_distance"], too_close
        item["target_region"]["params"]["min_distance"] = 2.0
        initial_point = np.array([2.5, 2.5, 2.5])
        target_check = constraints.validate(item, initial_point, check_pair_distance=True)
        initial_check = constraints.validate(item, initial_point, check_pair_distance=False)
        assert not target_check["success"] and initial_check["success"], (target_check, initial_check)

    try:
        build_canonical_camera(
            K_native=np.eye(3),
            native_size=(256, 256),
            render_size=(256, 256),
            transform="resize",
            camera_position=[0.0, 0.0, 0.0],
            camera_forward=[1.0, 0.0, 0.0],
        )
    except TypeError:
        pass
    else:
        raise AssertionError("camera_model_version must be a required explicit argument")

    print(json.dumps({"passed": True, "tests": 7, "generator_version": "projective_canonical_h1_v4_layout_gated"}, indent=2))


if __name__ == "__main__":
    main()
