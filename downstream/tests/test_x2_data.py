"""X2 data (the user's decision of 2026-10-26): 1% / 3% manifests nest in the 10% subset; 2 epochs from the dataset."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "demos"))
sys.path.insert(0, str(ROOT / "train"))
import epoch_steps  # noqa: E402
import subsets  # noqa: E402


def fake_task(root: Path, name: str, n: int):
    d = root / "target/atomic" / name / "2026-01-01/lerobot/meta"
    d.mkdir(parents=True)
    (d / "info.json").write_text("{}")
    (d / "episodes.jsonl").write_text("".join(json.dumps({"episode_index": i, "length": 100 + i}) + "\n"
                                              for i in range(n)))
    return {"task": name, "group": "atomic_seen", "horizon": 500}


def test_small_manifests_nest_and_leave_the_others_alone(tmp_path, monkeypatch):
    root = tmp_path / "data"
    tasks = [fake_task(root, "OpenDrawer", 60), fake_task(root, "CloseFridge", 55)]
    monkeypatch.setattr(subsets, "task_table", lambda path=None: tasks)
    out = tmp_path / "manifests"
    out.mkdir()
    (out / "target50_010.json").write_text("untouched")
    res = subsets.small(out, data_root=root)
    assert res["nesting"] == {"001_in_003": True, "003_in_010": True}
    m1 = json.loads((out / "target50_001.json").read_text())
    assert [len(r["train_ids"]) for r in m1["tasks"]] == [5, 5] and res["001"]["episodes"] == 10
    order = list(range(60))
    assert m1["tasks"][0]["train_ids"] == sorted(subsets.official_subset(order, 5))  # RoboCasa's permutation
    assert (out / "target50_010.json").read_text() == "untouched"


def test_batches_per_epoch_matches_the_trainer():
    # the 10% data: 3,547,850 samples, 8 per device on 8 cards -> 55,436 (the trainer's batches_per_epoch)
    assert epoch_steps.batches_per_epoch(3_547_850, 8, 8) == 55_436
    assert epoch_steps.batches_per_epoch(331_750, 8, 8) == 5_184  # 1%: 2 epochs = 10,368 steps
