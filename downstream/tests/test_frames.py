"""CPU tests of the per-episode example strip of downstream/eval/closed_loop.py."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
import closed_loop  # noqa: E402

np = pytest.importorskip("numpy")
Image = pytest.importorskip("PIL.Image")


def test_pick_indices_spreads_and_keeps_both_ends():
    assert closed_loop.pick_indices(3, 8) == [0, 1, 2]
    assert closed_loop.pick_indices(29, 8) == [0, 4, 8, 12, 16, 20, 24, 28]
    picked = closed_loop.pick_indices(100, 8)
    assert len(picked) == 8 and picked[0] == 0 and picked[-1] == 99


def test_recorder_writes_one_strip_with_steps_and_outcome_bar(tmp_path):
    rec = closed_loop.FrameRecorder(tmp_path / "frames/OpenDrawer_s1000.jpg", size=32, keep=4)
    for step in range(0, 160, 16):
        rec.add(step, {closed_loop.FRAME_CAMERA: np.full((256, 256, 3), step, dtype=np.uint8), "other": 1})
    rec.add(999, {"no camera": 0})  # ignored
    out = rec.save(success=True)
    assert out == {"frames": "frames/OpenDrawer_s1000.jpg", "frame_steps": [0, 48, 96, 144]}
    sheet = Image.open(tmp_path / out["frames"])
    assert sheet.size == (4 * 32, 32 + 14 + 4)
    r, g, b = sheet.convert("RGB").getpixel((2, sheet.height - 2))
    assert g > 100 and r < 100  # green bar: success


def test_recorder_takes_the_last_image_of_a_stack_and_reports_missing_camera(tmp_path):
    rec = closed_loop.FrameRecorder(tmp_path / "f/x.jpg", size=16, keep=2)
    rec.add(0, {closed_loop.FRAME_CAMERA: np.zeros((2, 64, 64, 3), dtype=np.uint8)})
    assert rec.frames[0][1].shape == (16, 16, 3)
    assert "frames_error" in closed_loop.FrameRecorder(tmp_path / "f/y.jpg").save(False)


def test_safe_switches_off_after_a_failure():
    calls = []

    def flaky(x):
        calls.append(x)
        raise RuntimeError("boom")

    f = closed_loop._safe(flaky)
    f(1)
    f(2)
    assert calls == [1]
