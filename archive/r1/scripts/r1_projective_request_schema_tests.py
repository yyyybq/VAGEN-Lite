#!/usr/bin/env python3

import math

from r1_prepare_fullscale_canonical_expansion import template as phase_a_template
from r1_prepare_phase_b_from_pair_universe import template as phase_b_template
from r1_projective_request_schema import half_plane_geometry, validate_projective_params


def objects() -> list[dict]:
    return [
        {"id": "a", "label": "chair", "center": [0.0, 0.0, 0.5]},
        {"id": "b", "label": "table", "center": [0.0, 2.0, 0.7]},
    ]


def camera() -> dict:
    return {"intrinsics": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            "extrinsics": [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0],
                           [0.0, 0.0, 1.0, 1.5], [0.0, 0.0, 0.0, 1.0]]}


def assert_geometry(params: dict, relation: str) -> None:
    validate_projective_params(params)
    assert params["boundary_point"] == [0.0, 1.0]
    assert params["boundary_direction"] == [0.0, 1.0]
    assert params["normal"] == ([1.0, -0.0] if relation == "left" else [-1.0, 0.0])
    direction = params["boundary_direction"]
    normal = params["normal"]
    assert math.isclose(sum(x * x for x in direction), 1.0)
    assert math.isclose(sum(x * y for x, y in zip(direction, normal)), 0.0)


def test_frozen_half_plane_geometry_matches_task_generator_convention() -> None:
    assert half_plane_geometry([0, 0, 0], [0, 2, 1], "left")["normal"] == [1.0, -0.0]
    assert half_plane_geometry([0, 0, 0], [0, 2, 1], "right")["normal"] == [-1.0, 0.0]


def test_phase_a_and_phase_b_requests_include_frozen_projective_schema() -> None:
    parent = {"_index": 7, "scene_id": "scene", "task_type": "equidistance",
              "task_id": "parent", "init_camera": camera(),
              "target_object": {"objects": objects()}}
    phase_a = phase_a_template(parent, 0, "projective_relations", "left")
    assert_geometry(phase_a["target_region"]["params"], "left")

    pair = {"pair_id": "scene:a--b", "scene_id": "scene", "object_ids": ["a", "b"],
            "categories": ["chair", "table"], "category_pair": "chair--table",
            "objects": objects(), "source_construction": "test"}
    phase_b = phase_b_template(pair, camera(), "projective_relations", 0, "right")
    assert_geometry(phase_b["target_region"]["params"], "right")
