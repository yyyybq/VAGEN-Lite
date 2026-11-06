"""n_stale.py (bad frames the frame scan misses; the user's decision 5 of 2026-11-06): a tile far from what the other
episodes of the same instance and camera setting rendered at the same pose is bad, including the start frame of an
episode whose moves were all blocked; the robot-camera tier is compared only with itself; a pose without enough
other tiles has no reference."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "n"))
import n_stale as S  # noqa: E402

PIL = pytest.importorskip("PIL")


def view(pose, cam="a"):
    """A deterministic textured 128 x 128 grey image per pose (and camera)."""
    rng = np.random.default_rng(abs(hash((round(pose[0], 3), round(pose[1], 3), round(pose[2], 3), cam))) % 2 ** 32)
    return rng.integers(0, 255, (128, 128)).astype(np.uint8)


def write_episode(folder: Path, iid: str, poses, tiles, success=False):
    from PIL import Image

    (folder / "frames").mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.concatenate(tiles, axis=1)).save(folder / "frames" / f"{iid}.png")
    row = {"instance_id": iid, "success": success, "turns": len(poses),
           "turn_log": [{"turn": k + 1, "pose_before": list(p)} for k, p in enumerate(poses)]}
    with open(folder / "episodes.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")


def test_a_frame_off_the_consensus_at_its_pose_is_flagged(tmp_path):
    start, step = (0.0, 0.0, 0.0), (0.3, 0.0, 0.0)
    for m in ("A", "B", "C"):  # three healthy runs: start, then one step
        write_episode(tmp_path / "n1" / m, "I1", [start, step], [view(start), view(step)])
    # D: blocked at the start every turn (pose never changes), its start frame is noise -> flagged at turn 1 and 2
    noise = np.full((128, 128), 7, np.uint8)
    write_episode(tmp_path / "n1" / "D", "I1", [start, start], [noise, noise])
    # E: healthy start, then the frame stops updating after a move -> flagged at turn 2 only
    write_episode(tmp_path / "gd" / "E", "I1", [start, step], [view(start), view(start)])
    # robot camera tier: its own views, compared only among themselves (two runs agree -> no flag)
    for m in ("A", "B", "C"):
        write_episode(tmp_path / "n3_20261103/e" / m, "I1", [start, step], [view(start, "r"), view(step, "r")])
    res = S.scan(tmp_path, tmp_path / "out", ["n3_20261103/e/"])
    rows = {r["folder"]: r for r in map(json.loads, (tmp_path / "out/stale.jsonl").read_text().splitlines())}
    assert set(rows) == {"n1/D", "gd/E"}
    assert rows["n1/D"]["bad_turns"] == [1, 2] and rows["n1/D"]["start_bad"]
    assert rows["gd/E"]["bad_turns"] == [2] and not rows["gd/E"]["start_bad"]
    assert res["flagged"] == 2 and res["instances"] == 1
    assert (tmp_path / "out/instances.txt").read_text() == "I1\n"
    summ = json.loads((tmp_path / "out/summary.json").read_text())
    assert summ["n3_20261103/e/A"]["flagged"] == 0 and summ["n1/A"]["checked_tiles"] == 2


def test_without_two_other_tiles_at_a_pose_there_is_no_reference(tmp_path):
    start = (1.0, 2.0, 0.5)
    write_episode(tmp_path / "x" / "A", "I2", [start], [view(start)])
    write_episode(tmp_path / "x" / "B", "I2", [start], [np.zeros((128, 128), np.uint8)])
    S.scan(tmp_path, tmp_path / "out", [])
    assert (tmp_path / "out/stale.jsonl").read_text() == ""
    summ = json.loads((tmp_path / "out/summary.json").read_text())
    assert summ["x/A"]["no_ref_tiles"] == 1 and summ["x/B"]["no_ref_tiles"] == 1


def test_error_rows_and_folders_outside_the_globs_are_skipped(tmp_path):
    start = (0.0, 0.0, 0.0)
    for m in ("A", "B"):
        write_episode(tmp_path / "keep" / m, "I3", [start], [view(start)])
    (tmp_path / "keep/A/episodes.jsonl").write_text(json.dumps({"instance_id": "I3", "error": "x"}) + "\n")
    write_episode(tmp_path / "other" / "C", "I3", [start], [view(start)])
    res = S.scan(tmp_path, tmp_path / "out", [], folders=["keep/*"])
    assert res["episodes"] == 1
