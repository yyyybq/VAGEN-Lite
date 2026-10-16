"""Size descriptions need their real comparison object in the relevant image."""
import copy
import json
from types import SimpleNamespace

import numpy as np

from data_gen.interiorgs_qa.capability import evidence, pipeline
from data_gen.interiorgs_qa.geom import OBB
from data_gen.interiorgs_qa.visible import silhouette
from data_gen.interiorgs_v10.contract import NATIVE_K, Obj, pose_from_xy_yaw
from vagen.envs.active_spatial.target_visibility import OccluderSet, item_camera


def fixture():
    small = Obj('small', 'armchair', np.array([3., -1., 0.]), np.array([3.5, -.5, 1.]))
    large = Obj('large', 'armchair', np.array([3., 1., 0.]), np.array([3.8, 2., 1.5]))
    target = small.target_object_entry()
    target.update(ref='the smaller armchair', ref_method='size', ref_anchor=None)
    task = {'task_id': 'size-task', 'scene_id': 'actual-scene', 'task_type': 'absolute_positioning',
            'task_description': 'Keep the smaller armchair in view.',
            'camera_model_version': 'canonical_camera_h1_resize_v1',
            'canonical_task_metric_version': 'canonical_spatial_task_h1_v2',
            'init_camera': {'intrinsics': NATIVE_K}, 'target_object': {'objects': [target]}}
    scene = SimpleNamespace(scene_id='actual-scene', objects=[small, large])
    boxes = {o.id: OBB.axis_aligned(o.aabb_min, o.aabb_max) for o in scene.objects}
    pose = pose_from_xy_yaw(0, 0, 0)
    K, _, _ = item_camera(task)
    depth = np.full((256, 256), 100., np.float32)
    alpha = np.ones_like(depth)
    for box in boxes.values():
        mask, front, _ = silhouette(box, pose, K, 256)
        depth[mask] = np.minimum(depth[mask], front[mask] + .01)
    return task, scene, boxes, pose, depth, alpha


def test_binding_uses_scene_instances_and_never_edits_formal_target():
    task, scene, *_ = fixture(); before = copy.deepcopy(task)
    refs = evidence.bind_size_references(task, scene)
    assert refs[0]['id'] == 'large'
    assert refs[0]['comparison_for_target_id'] == 'small'
    assert refs[0]['source_scene_id'] == 'actual-scene'
    assert task == before
    scene.objects.append(Obj('third', 'armchair', np.array([4., 1., 0.]), np.array([5., 2., 2.])))
    assert evidence.bind_size_references(task, scene) == {0: None}
    assert not evidence.visible_references(task, {'large': {'visible': True}}, size_references={0: None})


def test_wrong_scene_missing_or_changed_target_cannot_bind():
    task, scene, *_ = fixture()
    scene.scene_id = 'other-scene'
    assert evidence.bind_size_references(task, scene) == {0: None}
    scene.scene_id = 'actual-scene'
    scene.objects = scene.objects[1:]
    assert evidence.bind_size_references(task, scene) == {0: None}
    task, scene, *_ = fixture()
    task['target_object']['objects'][0]['bbox_min'][0] += .1
    assert evidence.bind_size_references(task, scene) == {0: None}


def test_comparison_requires_both_geometry_and_depth_visibility():
    task, scene, boxes, pose, depth, alpha = fixture()
    refs = evidence.bind_size_references(task, scene)
    empty = OccluderSet.from_boxes([])
    stats, masks = evidence.analyse(task, pose, depth, alpha, empty, boxes, size_references=refs)
    assert stats['small']['visible'] and stats['large']['visible']
    assert evidence.visible_references(task, stats, size_references=refs)
    hidden_depth = depth.copy()
    hidden_depth[masks['large']['silhouette']] = .5
    blocked, _ = evidence.analyse(task, pose, hidden_depth, alpha, empty, boxes, size_references=refs)
    assert blocked['large']['face']['rule']['visible']
    assert not blocked['large']['face']['with_depth']['visible']
    assert blocked['small']['visible']
    assert not evidence.visible_references(task, blocked, size_references=refs)
    assert evidence.size_reference_failure(task, blocked, refs) == 'size_comparison_not_identifiable_with_real_depth'
    # Geometry alone can also disqualify the comparator despite supportive depth.
    wall = OccluderSet.from_boxes([('wall-object', 'cabinet', [1., .1, 0.], [1.5, 1.5, 2.])])
    blocked, _ = evidence.analyse(task, pose, depth, alpha, wall, boxes, size_references=refs)
    assert not blocked['large']['face']['rule']['visible']
    assert not evidence.visible_references(task, blocked, size_references=refs)


def test_slot_checks_follow_identifying_and_current_view_roles():
    task, scene, *_ = fixture()
    refs = evidence.bind_size_references(task, scene)
    task['target_object']['objects'].append({'id': 'b', 'ref_method': 'unique_label', 'ref_anchor': None})
    stats = {'large': {'visible': False}}
    assert not evidence.visible_references(task, stats, slots={0}, size_references=refs)
    assert evidence.visible_references(task, stats, slots={1}, size_references=refs)
    # A caller that forgot to bind the size comparator also rejects, not passes.
    assert not evidence.visible_references(task, {'large': {'visible': True}}, slots={0})


def test_finalize_records_rejection_instead_of_making_an_answer(tmp_path, monkeypatch):
    task, scene, boxes, pose, depth, alpha = fixture()
    K, _, _ = item_camera(task)
    comparison_mask, _, _ = silhouette(boxes['large'], pose, K, 256)
    depth[comparison_mask] = .5  # actual comparator is present in metadata, hidden in this test depth.
    root = tmp_path / 'source'; root.mkdir()
    np.savez_compressed(root / 'depth.npz', depth=depth, alpha=alpha)
    state = {'state_id': 's', 'parent_task_id': task['task_id'], 'scene_id': task['scene_id'],
             'c2w': pose.tolist(), 'actions_from_initial': []}
    rec = {'state_id': 's', 'parent_task_id': task['task_id'], 'scene_id': task['scene_id'],
           'view_key': 'v', 'drafts': [{'kind': 'absolute_distance'}], 'reasons': [],
           'measurement': {'objects': [{'id': 'small'}], 'metric': {'success': False}}}
    for name, rows in [('tasks.jsonl', [task]), ('states.jsonl', [state]), ('geometry.jsonl', [rec]),
                       ('render_manifest.jsonl', [{'view_key': 'v', 'rgb': str(root/'attempt'/'rgb.png'),
                                                  'depth': str(root/'depth.npz')}])]:
        pipeline.write_rows(root/name, rows)
    original = (root/'tasks.jsonl').read_bytes()
    monkeypatch.setattr(pipeline, 'cached_ok', lambda *args: True)
    monkeypatch.setattr(pipeline, 'load_task_scene', lambda *args: scene)
    monkeypatch.setattr(pipeline, 'face_occluders_for', lambda *args: OccluderSet.from_boxes([]))
    monkeypatch.setattr(evidence, 'load_boxes', lambda *args: boxes)
    dest = tmp_path/'result'
    pipeline.finalize(root, root, dest)
    ledger = json.loads((dest/'state_filter_ledger.jsonl').read_text())
    assert ledger['first_exclusion'] == 'size_comparison_not_identifiable_with_real_depth'
    assert '比较对象' in ledger['first_exclusion_zh']
    assert ledger['size_reference_exclusions'][0]['target_slot'] == 0
    assert (dest/'candidate_bank.jsonl').read_text() == ''
    assert (dest/'approved_training.jsonl').read_text() == ''
    assert (root/'tasks.jsonl').read_bytes() == original
