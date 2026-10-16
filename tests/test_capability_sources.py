"""Source provenance and reachability tests; no renderer or model is used."""
import copy
import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from data_gen.interiorgs_qa.capability import sources as S
from data_gen.interiorgs_v10.contract import pose_from_xy_yaw


class FakeScene:
    min_wall_distance = 0.5

    def __init__(self, wall_distance=1.0, barrier=False):
        self.clearance = wall_distance
        self.barrier = barrier
        self.detector = SimpleNamespace(check_collision=lambda p: SimpleNamespace(
            has_collision=False, collision_type="none"))

    def room_of(self, x, y):
        return 0 if 0 <= x < 4 and -2 < y < 2 else None

    def wall_distance(self, x, y):
        return self.clearance

    def transition_legal(self, a, b):
        if self.barrier and a[0, 3] < 1.2 < b[0, 3]:
            return False, "collision_wall"
        return True, "ok"


def fixture_source(tmp_path):
    p = pose_from_xy_yaw(1.0, 0.0, 0.0)
    task = {"task_id": "t", "scene_id": "scene", "split": "train", "reserve": False,
            "init_camera": {"intrinsics": [[320, 0, 320], [0, 320, 240], [0, 0, 1]],
                            "extrinsics": p.tolist()},
            "difficulty": {"certificate": ["move_forward", "turn_left"]}}
    poses = S.certificate_poses(task)
    paths = [str(tmp_path / f"t__s{k}.png") for k in range(3)]
    frames = {}
    for k, path in enumerate(paths):
        Image.new("RGB", (256, 256), (k * 30, 10, 20)).save(path)
        frames[f"t__s{k}"] = {"frame_id": f"t__s{k}", "task_id": "t", "scene_id": "scene",
                              "state_id": f"s{k}", "c2w": poses[k].tolist(),
                              "camera_item": {"init_camera": copy.deepcopy(task["init_camera"])}}
    traj = {"id": "sft_t", "source_task_id": "t", "parent_task_id": "t", "scene_id": "scene",
            "split": "train", "reserve": False, "success": True,
            "source_task_sha256": S.row_digest(task), "chunking": "action_runs", "blocked_insert": None,
            "primitive_trajectory_trace": [{"step": k, "action": None if k == 0 else task["difficulty"]["certificate"][k - 1],
                                            "pose_c2w": p.tolist()} for k, p in enumerate(poses)],
            "conversations": [{"role": "system", "content": "system"},
                              {"role": "user", "content": "input", "image_path": paths[0]},
                              {"role": "assistant", "content": "<action>move_forward|</action>"},
                              {"role": "user", "content": "input", "image_path": paths[1]},
                              {"role": "assistant", "content": "<action>turn_left|</action>"}],
            "conversation_image_paths": paths[:2], "image_paths": paths}
    return task, traj, frames


def test_formal_train_and_depth_filter_cannot_admit_reserve():
    formal = [{"task_id": k, "scene_id": "s"} for k in "abc"]
    published = [dict(r, split="train", reserve=r["task_id"] == "b") for r in formal]
    rows, reasons = S.select_tasks(formal, published, ["s"], {"c"}, scenes=["s"])
    assert [r["task_id"] for r in rows] == ["a"]
    assert reasons == {"not_formal_train": 1, "depth_excluded": 1}
    with pytest.raises(ValueError, match="frozen train"):
        S.select_tasks(formal, published, [], set(), scenes=["s"])


def test_changed_published_task_is_rejected():
    with pytest.raises(ValueError, match="differs"):
        S.select_tasks([{"task_id": "a", "scene_id": "s", "q": "original"}],
                       [{"task_id": "a", "scene_id": "s", "q": "changed", "split": "train", "reserve": False}],
                       ["s"], set(), scenes=["s"])


def test_existing_frames_replay_and_rgb_bindings(tmp_path):
    task, traj, frames = fixture_source(tmp_path)
    poses, records, expected = S.validate_trajectory(task, traj, frames, tmp_path, 5)
    assert len(poses) == len(records) == 3
    assert [r[-1] for r in records] == [1, 1, 0]
    assert len(expected) == 2
    frames["t__s1"]["c2w"][0][3] += 0.02
    with pytest.raises(ValueError, match="camera/task"):
        S.validate_trajectory(task, traj, frames, tmp_path, 5)


def test_grouped_actions_must_reproduce_real_certificate(tmp_path):
    task, traj, frames = fixture_source(tmp_path)
    traj["conversations"][2]["content"] = "<action>move_backward|</action>"
    with pytest.raises(ValueError, match="Grouped actions"):
        S.validate_trajectory(task, traj, frames, tmp_path, 5)


def test_source_hash_and_primitive_trace_are_checked(tmp_path):
    task, traj, frames = fixture_source(tmp_path)
    traj["source_task_sha256"] = "wrong"
    with pytest.raises(ValueError, match="hash"):
        S.validate_trajectory(task, traj, frames, tmp_path, 5)
    traj["source_task_sha256"] = S.row_digest(task)
    traj["primitive_trajectory_trace"][1]["pose_c2w"][0][3] += 0.01
    with pytest.raises(ValueError, match="Primitive trace"):
        S.validate_trajectory(task, traj, frames, tmp_path, 5)


def test_repeated_blocked_turn_does_not_create_a_new_state(tmp_path):
    task, traj, frames = fixture_source(tmp_path)
    traj["blocked_insert"] = {"turn": 0, "move": "move_backward"}
    first = traj["conversations"][1]
    traj["conversations"][1:1] = [copy.deepcopy(first), {"role": "assistant", "content": "<action>move_backward|</action>"}]
    traj["conversation_image_paths"].insert(0, traj["conversation_image_paths"][0])
    traj["image_paths"].insert(0, traj["image_paths"][0])
    _, records, _ = S.validate_trajectory(task, traj, frames, tmp_path, 5)
    assert len(records) == 3
    assert records[0][-1] == 2


def test_outside_room_is_rejected_without_a_collision():
    scene = FakeScene()
    start = pose_from_xy_yaw(0.1, 0, 0)
    outside = S.apply_action(start, "move_backward")
    result = S.check_path(scene, [start, outside])[-1]
    assert not result["runtime_legal"]
    assert result["runtime_rejections"][-1]["reason"] == "outside_rooms"


def test_crossed_wall_rejects_entire_prefix_even_if_endpoints_are_legal():
    start = pose_from_xy_yaw(1.0, 0, 0)
    end = S.apply_action(start, "move_forward")
    result = S.check_path(FakeScene(barrier=True), [start, end, S.apply_action(end, "turn_left")])[-1]
    assert not result["runtime_legal"]
    assert result["runtime_rejections"][0]["reason"] == "transition_collision_wall"


def test_extra_clearance_does_not_mislabel_runtime_path_illegal():
    result = S.check_path(FakeScene(wall_distance=0.4), [pose_from_xy_yaw(1, 0, 0)])[0]
    assert result["runtime_legal"]
    assert not result["sample_eligible"]
    assert result["sample_rejections"][0]["required_m"] == 0.5


def test_neighbors_are_exactly_one_reachable_layer_and_keep_parent(tmp_path):
    task, traj, frames = fixture_source(tmp_path)
    states, audit = S.build_task_states(task, traj, frames, tmp_path, FakeScene(), 5)
    existing = [s for s in states if s["existing_rgb"]]
    assert [s["source_kind"] for s in existing] == ["act_sft_input", "act_sft_input", "act_sft_terminal"]
    assert audit["extension_attempts"] == len(existing) * 6
    assert len({s["pose_key"] for s in states}) == len(states)
    for state in states:
        replay = np.array(task["init_camera"]["extrinsics"])
        for action in state["actions_from_initial"]:
            replay = S.apply_action(replay, action)
        assert np.allclose(replay, state["c2w"], atol=1e-9, rtol=0)
        assert state["parent_task_id"] == "t"
        if state["source_kind"] == "legal_extension":
            assert state["extension_from"] in {s["state_id"] for s in existing}
            assert state["existing_rgb"] is None and state["existing_rgb_sha256"] is None


def test_pose_key_can_reuse_rendering_across_parents_but_not_scenes():
    pose = pose_from_xy_yaw(1, 0, 0)
    K = np.eye(3)
    key = S.pose_key("s", pose, K)
    assert key == S.pose_key("s", pose.copy(), K)
    assert key != S.pose_key("other_scene", pose, K)


def test_existing_outputs_are_never_overwritten(tmp_path):
    (tmp_path / "states.jsonl").write_text("old\n")
    with pytest.raises(FileExistsError):
        S.build(tmp_path / "source", tmp_path)
    assert (tmp_path / "states.jsonl").read_text() == "old\n"


def test_epoch_budget_counts_reused_inputs_separately(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text(json.dumps({"images": ["a", "a", "b"]}) + "\n" + json.dumps({"images": ["b"]}) + "\n")
    result = S.epoch_budget(path)
    assert (result["trajectory_records"], result["image_input_occurrences"], result["unique_image_paths"]) == (2, 4, 2)
