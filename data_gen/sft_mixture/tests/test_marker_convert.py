"""data_gen/sft_mixture/marker_convert.py (the user's item 6c of 2026-11-07)."""
from unittest import mock

from PIL import Image

from data_gen.sft_mixture import marker_convert as MC


def test_convert_row_marks_each_frame_and_keeps_the_text(tmp_path):
    frames = []
    for k in range(2):
        p = tmp_path / f"v10_t__s{k}.png"
        Image.new("RGB", (256, 256), (128, 128, 128)).save(p)
        frames.append(str(p))
    row = {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "u0"},
                        {"role": "assistant", "content": "<action>move_forward|</action>"},
                        {"role": "user", "content": "u1"}, {"role": "assistant", "content": "<action>turn_left|</action>"}],
           "images": frames, "source_id": "v10_t"}
    MC._INPUTS["poses"] = {"v10_t__s0": "pose0", "v10_t__s1": "pose1"}
    out = tmp_path / "out"
    out.mkdir()
    marks = {"pose0": [("A", [10, 10, 100, 100]), ("B", [120, 20, 200, 90])], "pose1": []}
    with mock.patch("vagen.envs.active_spatial.oracle_marker.boxes", side_effect=lambda item, c2w, occ: marks[c2w]):
        new, st = MC.convert_row(row, {}, None, out)
    assert new["messages"] == row["messages"] and row["images"] == frames
    assert new["images"] == [str(out / "v10_t__s0.png"), str(out / "v10_t__s1.png")]
    assert st == {"frames": 2, "boxes": 2, "frames_no_box": 1, "boxes_A": 1, "boxes_B": 1}
    assert Image.open(new["images"][0]).getpixel((10, 50)) == (230, 30, 40)  # A's left edge, red
    assert Image.open(new["images"][1]).getpixel((10, 50)) == (128, 128, 128)  # no box on the second frame
