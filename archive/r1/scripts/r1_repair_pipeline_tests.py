#!/usr/bin/env python3
"""Deterministic tests for R1 v4 layout and version gates."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from r1_repair_pipeline import (
    PROJECTIVE_GENERATOR_VERSION,
    SceneConstraints,
    absolute_pitch_degrees,
    horizontal_forward,
    reverse_turn_initial,
)
from vagen.envs.active_spatial.canonical_camera import build_canonical_camera
from vagen.envs.active_spatial.collision_detector import CollisionDetector
from vagen.envs.active_spatial.canonical_task_metrics import (
    CANONICAL_TASK_METRIC_VERSION,
    uses_canonical_backend,
)


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
        toy_labels = [
            {
                "ins_id": f"object_{index}",
                "label": "cabinet",
                "bounding_box": [
                    {"x": x, "y": y, "z": z}
                    for x in (1.0, 2.0)
                    for y in (1.0, 2.0)
                    for z in (0.0, 2.0)
                ],
            }
            for index in range(20)
        ]
        (scene / "labels.json").write_text(json.dumps(toy_labels))

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

    horizontal = horizontal_forward(np.array([1.0, 2.0, 9.0]))
    assert np.isclose(horizontal[2], 0.0) and np.isclose(np.linalg.norm(horizontal), 1.0)
    assert np.isclose(absolute_pitch_degrees(np.array([1.0, 0.0, 1.0])), 45.0)

    # InteriorGS structure files exist in both raw-Y and historical mirrored-Y
    # conventions.  The runtime collision gate must follow label alignment.
    detector = CollisionDetector()
    detector._label_xy_centers = [np.array([1.0, 2.0]), np.array([2.0, 2.0])]
    convention_scene = {
        "rooms": [{"profile": [[0.0, 1.0], [3.0, 1.0], [3.0, 3.0], [0.0, 3.0]]}]
    }
    assert detector._select_structure_y_sign(convention_scene) == 1.0
    assert detector.structure_convention_status == "ambiguous"
    detector._label_xy_centers = [np.array([1.0, -2.0]), np.array([2.0, -2.0])]
    assert detector._select_structure_y_sign(convention_scene) == -1.0
    detector._label_xy_centers = [np.array([1.5, 2.0])] * 20
    assert detector._select_structure_y_sign(convention_scene) == 1.0
    assert detector.structure_convention_status == "frozen"
    overridden = CollisionDetector(structure_y_sign_overrides={"reviewed": 1.0})
    overridden._label_xy_centers = [np.array([1.5, -2.0])] * 20
    assert overridden._select_structure_y_sign(convention_scene, "reviewed") == 1.0
    assert overridden.structure_convention_status == "frozen"
    assert overridden.convention_record()["override_source"] == "explicit_versioned_override"

    # Only explicitly versioned R1 rows select the canonical backend. Legacy
    # rows and unknown future versions must retain historical scoring.
    assert not uses_canonical_backend({"task_type": "fov_inclusion"})
    assert uses_canonical_backend(
        {
            "task_type": "fov_inclusion",
            "canonical_task_metric_version": CANONICAL_TASK_METRIC_VERSION,
        }
    )
    assert not uses_canonical_backend(
        {"task_type": "fov_inclusion", "canonical_task_metric_version": "unknown"}
    )

    success_pose = np.eye(4)
    initial_pose, certificate = reverse_turn_initial(success_pose)
    engine = __import__(
        "vagen.envs.active_spatial.utils", fromlist=["ViewManipulator"]
    ).ViewManipulator(step_translation=0.3, step_rotation_deg=20.0, world_up_axis="Z")
    engine.reset(initial_pose)
    for action in certificate:
        engine.step(action)
    assert len(certificate) == 9 and np.allclose(engine.get_pose(), success_pose, atol=1e-7)

    # The tiered planner reuses one manipulator but resets it before every
    # action.  This must be bit-equivalent to constructing a fresh instance.
    reused = type(engine)(step_translation=0.3, step_rotation_deg=20.0, world_up_axis="Z")
    for action in ("move_forward", "move_backward", "move_left", "move_right", "turn_left", "turn_right"):
        fresh = type(engine)(step_translation=0.3, step_rotation_deg=20.0, world_up_axis="Z")
        fresh.reset(initial_pose)
        reused.reset(initial_pose)
        assert np.array_equal(fresh.step(action), reused.step(action))

    print(json.dumps({"passed": True, "tests": 18, "generator_version": PROJECTIVE_GENERATOR_VERSION}, indent=2))


if __name__ == "__main__":
    main()
