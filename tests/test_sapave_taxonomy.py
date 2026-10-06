
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data_gen.sapave.taxonomy import (
    has_explicit_spatial_language,
    sapave_manip_family,
    sapave_modality,
    yaw_c2w,
)


def test_spatial_directive_from_projective():
    item = {
        "task_type": "projective_relations",
        "task_description": "Position where wardrobe appears to the left of coffee_maker",
        "preset": "left_of",
        "object_label": "wardrobe+coffee_maker",
    }
    assert sapave_modality(item) == "spatial_directive"
    assert has_explicit_spatial_language(item)
    assert sapave_manip_family(item) == "articulated"


def test_commonsense_without_direction():
    item = {
        "task_type": "absolute_positioning",
        "task_description": "Move to any position 2.9m from sofa",
        "preset": "closer",
        "object_label": "sofa",
    }
    assert sapave_modality(item) == "commonsense"
    assert not has_explicit_spatial_language(item)


def test_visual_centering():
    item = {"task_type": "centering", "task_description": "center the sofa", "object_label": "sofa"}
    assert sapave_modality(item) == "visual_centering"


def test_yaw_keeps_translation():
    c2w = [[1, 0, 0, 1.5], [0, 1, 0, 2.5], [0, 0, 1, 1.6], [0, 0, 0, 1]]
    yawed = yaw_c2w(c2w, 90.0)
    assert yawed[0][3] == 1.5
    assert yawed[1][3] == 2.5
    assert yawed[2][3] == 1.6
