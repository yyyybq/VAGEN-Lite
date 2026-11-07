"""vagen/envs/active_spatial/oracle_marker.py (the user's item 6 of 2026-11-07)."""
import argparse
from unittest import mock

from PIL import Image

from vagen.envs.active_spatial import oracle_marker as OM


def test_draw_keeps_the_original_and_uses_the_fixed_colours():
    img = Image.new("RGB", (256, 256), (128, 128, 128))
    out = OM.draw(img, [("A", [40, 60, 120, 200]), ("B", [150, 30, 240, 100])])
    assert img.getpixel((40, 130)) == (128, 128, 128)  # the input is not drawn on
    assert out.getpixel((40, 130)) == tuple(OM.STYLE["colors"]["A"])  # A's left edge
    assert out.getpixel((239, 70)) == tuple(OM.STYLE["colors"]["B"])  # B's right edge
    assert out.getpixel((80, 130)) == (128, 128, 128)  # inside A: only the outline
    assert OM.draw(img, []).tobytes() == img.tobytes()


def test_boxes_scale_to_the_image():
    img = Image.new("RGB", (512, 512), (0, 0, 0))
    out = OM.draw(img, [("A", [40, 60, 120, 200])])
    assert out.getpixel((80, 260)) == tuple(OM.STYLE["colors"]["A"])  # x 40 -> 80 on a 512 image


def test_boxes_only_clearly_visible_targets():
    rows = [{"letter": "A", "in_view": True, "box": [1, 2, 3, 4]}, {"letter": "B", "in_view": False, "box": None}]
    with mock.patch("vagen.envs.active_spatial.pointing.truth", return_value=rows):
        assert OM.boxes({}, None, None) == [("A", [1, 2, 3, 4])]


def test_evaluation_switch():
    from data_gen.interiorgs_v10 import evaluation as EV

    assert EV.switch_overrides(argparse.Namespace(oracle_marker=True))["oracle_marker"] is True
    assert "oracle_marker" not in EV.switch_overrides(argparse.Namespace())
