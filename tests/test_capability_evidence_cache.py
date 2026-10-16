"""CPU checks for cached evidence and incomplete cropping comparisons."""
import json

import numpy as np
from PIL import Image
import pytest

from data_gen.interiorgs_qa.capability.evidence import fov_depth_agrees
from data_gen.interiorgs_qa.capability.render import cached_ok, sha


def make_cache(tmp_path, *, alpha=None, wide_alpha=None, wide_depth=None, include_alpha=True):
    rgb = tmp_path / 'rgb.png'
    Image.fromarray(np.zeros((256, 256, 3), np.uint8)).save(rgb)
    depth = tmp_path / 'depth.npz'
    arrays = {'depth': np.ones((256, 256), np.float32)}
    if include_alpha:
        arrays['alpha'] = np.ones((256, 256), np.float32) if alpha is None else alpha
    np.savez_compressed(depth, **arrays)
    doc = {'ok': True, 'view_key': 'camera', 'rgb': str(rgb), 'rgb_sha256': sha(rgb),
           'depth': str(depth), 'depth_sha256': sha(depth)}
    if wide_alpha is not None or wide_depth is not None:
        wide = tmp_path / 'wide.npz'
        np.savez_compressed(wide,
            depth=np.ones((384, 384), np.float32) if wide_depth is None else wide_depth,
            alpha=np.ones((384, 384), np.float32) if wide_alpha is None else wide_alpha)
        doc.update(wide_depth=str(wide), wide_depth_sha256=sha(wide))
    status = tmp_path / 'status.json'
    status.write_text(json.dumps(doc))
    return status


def test_valid_rgb_depth_and_wide_cache(tmp_path):
    status = make_cache(tmp_path, wide_alpha=np.ones((384, 384), np.float32))
    assert cached_ok(status, 'camera', True)
    assert cached_ok(status, 'camera', False)
    assert not cached_ok(status, 'other-camera', False)


@pytest.mark.parametrize('alpha', [np.ones((1, 256)), np.full((256, 256), np.nan),
                                  np.full((256, 256), -0.01), np.full((256, 256), 255)])
def test_wrong_alpha_cannot_be_reused_even_when_hash_matches(tmp_path, alpha):
    assert not cached_ok(make_cache(tmp_path, alpha=alpha), 'camera', False)


def test_missing_alpha_cannot_be_reused(tmp_path):
    assert not cached_ok(make_cache(tmp_path, include_alpha=False), 'camera', False)


@pytest.mark.parametrize('arrays', [
    {'wide_depth': np.ones((256, 256))},
    {'wide_depth': np.full((384, 384), np.inf)},
    {'wide_alpha': np.ones((1, 384))},
    {'wide_alpha': np.full((384, 384), np.nan)},
    {'wide_alpha': np.full((384, 384), 255)},
    {'wide_alpha': np.full((384, 384), -0.01)},
])
def test_invalid_wide_evidence_rejected_if_stored(tmp_path, arrays):
    status = make_cache(tmp_path, **arrays)
    assert not cached_ok(status, 'camera', True)
    # Finalization reads present wide arrays even for a non-FOV task.
    assert not cached_ok(status, 'camera', False)


def test_required_wide_missing_and_modified_payload_rejected(tmp_path):
    status = make_cache(tmp_path)
    assert not cached_ok(status, 'camera', True)
    (tmp_path / 'depth.npz').write_bytes(b'changed after rendering')
    assert not cached_ok(status, 'camera', False)


def fov_fixture():
    return [{'id': 'complete'}, {'id': 'cropped'}], {
        'complete': {'visible': True, 'edge_pixels': 0, 'wide_depth_evidence': {
            'depth_supported_pixels': 1000, 'outside_original_frame_fraction': 0,
            'wide_canvas_edge_pixels': 0}},
        'cropped': {'visible': True, 'edge_pixels': 20, 'wide_depth_evidence': {
            'depth_supported_pixels': 1000, 'outside_original_frame_fraction': .3,
            'wide_canvas_edge_pixels': 0}}}


def test_each_target_must_have_one_boolean_expected_flag():
    objects, stats = fov_fixture()
    assert fov_depth_agrees(objects, stats, [True, False]) == (True, None)
    for flags in ([], [True], [True, False, True]):
        assert fov_depth_agrees(objects, stats, flags) == (False, 'fov_target_flag_count_mismatch')
    assert fov_depth_agrees([], {}, []) == (False, 'fov_target_flag_count_mismatch')
    assert fov_depth_agrees(objects, stats, ['true', False]) == (False, 'fov_expected_flag_not_boolean')


def test_missing_second_target_evidence_cannot_pass():
    objects, stats = fov_fixture()
    del stats['cropped']
    assert fov_depth_agrees(objects, stats, [True, False])[0] is False
