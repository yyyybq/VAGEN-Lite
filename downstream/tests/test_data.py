"""CPU tests of downstream/demos (episode manifests, official subsets, training-directory preparation)."""
import json
import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "demos"))
import prepare_data  # noqa: E402
import subsets  # noqa: E402

LEROBOT_SLICES = {"base_motion": (0, 4), "control_mode": (4, 5), "end_effector_position": (5, 8),
                  "end_effector_rotation": (8, 11), "gripper_close": (11, 12)}
CAMS = ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand")


def robocasa_get_subset(ids, num_demos, seed=0):
    """RoboCasa 1.0.1 groot_dataset.get_subset_demos_filter_key, verbatim apart from the file reading."""
    all_demo_ids = list(ids)
    state = random.getstate()
    random.seed(seed)
    random.shuffle(all_demo_ids)
    random.setstate(state)
    return None if num_demos >= len(all_demo_ids) else all_demo_ids[:num_demos]


def test_official_subset_matches_robocasa_and_nests():
    ids = list(range(500, 0, -1)) + [700, 3]
    assert subsets.official_subset(ids, 50) == robocasa_get_subset(ids, 50)
    assert subsets.official_subset(ids, 150) == robocasa_get_subset(ids, 150)
    small, big = subsets.official_subset(ids, 50), subsets.official_subset(ids, 150)
    assert big[:50] == small and set(big) <= set(ids)
    assert subsets.official_subset(ids, 10_000) is None


def fake_lerobot(root: Path, task: str, n: int):
    """A tiny LeRobot v2.1 directory that passes scripts/check_starvla_navigatekitchen.check_lerobot."""
    src = root / "target/atomic" / task / "20250101/lerobot"
    (src / "meta").mkdir(parents=True)
    feats = {"action": {"shape": [12]}, "observation.state": {"shape": [16]}}
    feats.update({f"observation.images.{c}": {"shape": [256, 256, 3]} for c in CAMS})
    info = {"codebase_version": "v2.1", "fps": 20, "total_episodes": n, "chunks_size": 1000, "features": feats,
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4"}
    (src / "meta/info.json").write_text(json.dumps(info))
    modality = {"action": {k: {"start": a, "end": b} for k, (a, b) in LEROBOT_SLICES.items()},
                "video": {c: {"original_key": f"observation.images.{c}"} for c in CAMS},
                "annotation": {"human.task_description": {"original_key": "annotation.human.task_description"}}}
    (src / "meta/modality.json").write_text(json.dumps(modality))
    (src / "meta/tasks.jsonl").write_text('{"task_index": 0, "task": "do it"}\n')
    order = list(range(n))[::-1]  # file order differs from index order
    (src / "meta/episodes.jsonl").write_text("".join(json.dumps({"episode_index": i, "length": 10 + i}) + "\n"
                                                     for i in order))
    for i in range(n):
        (src / f"data/chunk-000/episode_{i:06d}.parquet").parent.mkdir(parents=True, exist_ok=True)
        (src / f"data/chunk-000/episode_{i:06d}.parquet").write_bytes(b"x")
        for c in CAMS:
            v = src / f"videos/chunk-000/observation.images.{c}/episode_{i:06d}.mp4"
            v.parent.mkdir(parents=True, exist_ok=True)
            v.write_bytes(b"x")
    return src


@pytest.fixture
def fake_target(tmp_path, monkeypatch):
    data_root = tmp_path / "v1.0"
    for task in subsets.STEP2_TASKS:
        fake_lerobot(data_root, task, 160)
    tasks = {"tasks": [{"group": "atomic_seen", "task": t, "horizon": 450} for t in subsets.STEP2_TASKS]}
    (tmp_path / "tasks.json").write_text(json.dumps(tasks))
    monkeypatch.setattr(subsets, "TASKS", tmp_path / "tasks.json")
    monkeypatch.setattr(subsets, "DATA_ROOT", data_root)
    return data_root


def test_target50_manifests_nest(fake_target, tmp_path):
    summary = subsets.target50(tmp_path / "m", data_root=fake_target)
    assert summary["nesting"] == {"010_in_030": True, "030_in_100": True}
    m10 = json.loads((tmp_path / "m/target50_010.json").read_text())
    m100 = json.loads((tmp_path / "m/target50_100.json").read_text())
    assert all(len(t["train_ids"]) == 50 for t in m10["tasks"])
    assert all(len(t["train_ids"]) == 160 for t in m100["tasks"])
    src = Path(m10["tasks"][0]["source"])
    file_order = [json.loads(x)["episode_index"] for x in (src / "meta/episodes.jsonl").read_text().splitlines()]
    assert m10["tasks"][0]["train_ids"] == sorted(robocasa_get_subset(file_order, 50))


def test_step2_and_prepare_and_check(fake_target, tmp_path):
    info = subsets.step2(tmp_path / "step2.json", n_train=100, n_val=20, data_root=fake_target)
    assert info["tasks"] == 3 and info["episodes"] == 300
    m = json.loads((tmp_path / "step2.json").read_text())
    assert m["tasks"][0]["train_ids"] == list(range(100)) and m["tasks"][0]["val_ids"] == list(range(140, 160))
    out = tmp_path / "data"
    result = prepare_data.prepare(tmp_path / "step2.json", out)
    assert result["train_episodes"] == 300 and result["val_episodes"] == 60
    assert prepare_data.check(out)["ok"]
    spec = json.loads((out / "mixture_spec.json").read_text())
    assert spec[0] == ["NavigateKitchen/lerobot", 1.0, "panda_omron_robocasa365"]
    extra = out / "train/NavigateKitchen/lerobot/data/chunk-000/episode_000150.parquet"
    extra.symlink_to(Path(m["tasks"][0]["source"]) / "data/chunk-000/episode_000150.parquet")
    assert not prepare_data.check(out)["ok"]  # a file outside the manifest is caught
    with pytest.raises(FileExistsError):
        prepare_data.prepare(tmp_path / "step2.json", out)
