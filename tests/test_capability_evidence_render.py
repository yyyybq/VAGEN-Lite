import json
from pathlib import Path

import numpy as np
from PIL import Image

from data_gen.interiorgs_qa.capability.evidence import fov_depth_agrees
from data_gen.interiorgs_qa.capability.render import cached_ok, sha, view_key


def test_fov_rejects_empty_box_corner_crop():
    objs = [{'id': 'a'}, {'id': 'b'}]
    full = {'visible': True, 'edge_pixels': 0, 'wide_depth_evidence': {
        'depth_supported_pixels': 300, 'outside_original_frame_fraction': 0.0,
        'wide_canvas_edge_pixels': 0}}
    stats = {'a': full, 'b': full}
    assert fov_depth_agrees(objs, stats, [True, True]) == (True, None)
    assert fov_depth_agrees(objs, stats, [True, False])[0] is False


def test_fov_requires_real_surface_at_image_edge():
    stat = {'visible': True, 'edge_pixels': 20, 'wide_depth_evidence': {
        'depth_supported_pixels': 500, 'outside_original_frame_fraction': 0.3,
        'wide_canvas_edge_pixels': 0}}
    assert fov_depth_agrees([{'id': 'a'}], {'a': stat}, [False])[0]
    stat['wide_depth_evidence']['wide_canvas_edge_pixels'] = 1
    assert not fov_depth_agrees([{'id': 'a'}], {'a': stat}, [False])[0]


def test_failed_or_changed_cache_cannot_be_reused(tmp_path):
    rgb, depth, status = (tmp_path / p for p in ['rgb.png', 'depth.npz', 'status.json'])
    Image.fromarray(np.zeros((256, 256, 3), np.uint8)).save(rgb)
    np.savez_compressed(depth, depth=np.ones((256, 256)), alpha=np.ones((256, 256)))
    doc = {'ok': True, 'view_key': 'key', 'rgb': str(rgb), 'rgb_sha256': sha(rgb),
           'depth': str(depth), 'depth_sha256': sha(depth)}
    status.write_text(json.dumps(doc))
    assert cached_ok(status, 'key', False)
    assert not cached_ok(status, 'key', True)
    doc['ok'] = False
    status.write_text(json.dumps(doc))
    assert not cached_ok(status, 'key', False)
    doc['ok'] = True
    status.write_text(json.dumps(doc))
    depth.write_bytes(b'changed')
    assert not cached_ok(status, 'key', False)


def test_camera_identity_includes_intrinsics_and_scene():
    pose, K = np.eye(4), np.eye(3)
    key = view_key('scene1', pose, K)
    assert key != view_key('scene2', pose, K)
    other = K.copy()
    other[0, 0] *= 2
    assert key != view_key('scene1', pose, other)
