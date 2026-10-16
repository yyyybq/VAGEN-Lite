"""Rendered evidence for ability questions, separate from the environment labels.

Depth is real renderer output. Depth-supported pixels inside a labelled oriented
box are an attribution approximation, never an instance segmentation ground truth.
The default environment face-visibility function remains an additional requirement.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from data_gen.interiorgs_qa.geom import OBB
from data_gen.interiorgs_qa.visible import object_view, visible_mask
from data_gen.interiorgs_v10.depth_check import depth_hidden
from vagen.envs.active_spatial import target_visibility as TV


def load_boxes(scene_dir):
    result = {}
    for item in json.loads((Path(scene_dir) / 'labels.json').read_text()):
        corners = item.get('bounding_box', [])
        if len(corners) != 8:
            continue
        oid = str(item.get('ins_id') or item.get('id') or '')
        box = OBB.from_label_corners([[p['x'], p['y'], p['z']] for p in corners])
        if box.from_corners:
            result[oid] = box
    return result


def face_with_depth(obj, pose, K, depth, alpha, occ):
    geom = TV.FaceGeometry(obj['bbox_min'], obj['bbox_max'])
    blocked = TV.blocked_mask(occ, pose[None, :3, 3], geom.points, geom.lo, geom.hi,
                              exclude_ids=(str(obj.get('id', '')),))
    extra = depth_hidden(geom.points, pose, K, depth, alpha)[None, :]
    rule = TV.best_face(TV.face_visibility(pose[None, :3, :3], pose[None, :3, 3],
                                         geom, blocked, K, 256, 256))
    checked = TV.best_face(TV.face_visibility(pose[None, :3, :3], pose[None, :3, 3],
                                            geom, blocked | extra, K, 256, 256))
    return {'rule': rule, 'with_depth': checked, 'hidden_samples': int(extra.sum())}


def analyse(task, pose, depth, alpha, occ, boxes, wide=None):
    pose = np.asarray(pose, float)
    K, width, height = TV.item_camera(task)
    if (width, height) != (256, 256) or depth.shape != (256, 256):
        raise ValueError('camera/depth shape mismatch')
    if not np.isfinite(depth).all() or not np.isfinite(alpha).all():
        raise ValueError('invalid rendered depth/alpha')
    targets = task['target_object']['objects']
    refs = [r for _, r in all_target_references(task)]
    out, masks = {}, {}
    for obj in targets + refs:
        oid = str(obj['id'])
        if oid in out:
            continue
        if oid not in boxes:
            out[oid] = {'visible': False, 'reason': 'original_oriented_box_unavailable'}
            continue
        box = boxes[oid]
        ov, _ = object_view(oid, box, pose, depth, alpha, K)
        mask, sil, hidden = visible_mask(box, pose, depth, alpha, K)
        face = face_with_depth(obj, pose, K, depth, alpha, occ)
        stat = {'visible': bool(face['rule']['visible'] and face['with_depth']['visible']),
                'face': face, 'depth_supported_box_pixels': ov.to_json(),
                'edge_pixels': int(mask[:2, :].sum() + mask[-2:, :].sum() +
                                   mask[:, :2].sum() + mask[:, -2:].sum()),
                'attribution_is_instance_mask': False}
        if wide is not None:
            wd, wa = wide
            padding = (wd.shape[0] - 256) // 2
            wk = np.array(K, copy=True)
            wk[0, 2] += padding
            wk[1, 2] += padding
            wm, _, _ = visible_mask(box, pose, wd, wa, wk)
            crop = wm[padding:padding+256, padding:padding+256]
            stat['wide_depth_evidence'] = {
                'depth_supported_pixels': int(wm.sum()),
                'outside_original_frame_pixels': int(wm.sum() - crop.sum()),
                'outside_original_frame_fraction': float((wm.sum() - crop.sum()) / max(int(wm.sum()), 1)),
                'wide_canvas_edge_pixels': int(wm[0, :].sum()+wm[-1, :].sum()+wm[:, 0].sum()+wm[:, -1].sum()),
                'diagnostic_only': True,
            }
        out[oid] = stat
        masks[oid] = {'visible': mask, 'silhouette': sil, 'hidden': hidden}
    return out, masks


def all_target_references(task):
    """Include the hidden target's identifying anchor, not only success-required B."""
    return [(slot, obj["ref_anchor"]) for slot, obj in enumerate(task["target_object"]["objects"])
            if isinstance(obj.get("ref_anchor"), dict) and "bbox_min" in obj["ref_anchor"]
            and "bbox_max" in obj["ref_anchor"]]


def visible_references(task, stats, slots=None):
    for slot, obj in all_target_references(task):
        if slots is None or slot in slots:
            if not stats.get(str(obj['id']), {}).get('visible', False):
                return False
    return True


def fov_depth_agrees(objects, stats, expected_flags):
    """Require real surfaces outside the model image to confirm obvious cropping."""
    if not objects or len(objects) != len(expected_flags):
        return False, 'fov_target_flag_count_mismatch'
    if any(not isinstance(flag, (bool, np.bool_)) for flag in expected_flags):
        return False, 'fov_expected_flag_not_boolean'
    for obj, full in zip(objects, expected_flags):
        s = stats.get(str(obj['id']), {})
        w = s.get('wide_depth_evidence', {})
        if not s.get('visible') or w.get('depth_supported_pixels', 0) < 196:
            return False, 'fov_target_or_depth_insufficient'
        if w.get('wide_canvas_edge_pixels', 1) > 0:
            return False, 'fov_wider_inspection_view_still_cropped'
        fraction = w['outside_original_frame_fraction']
        if full:
            if fraction > 0.005 or s['edge_pixels'] > 0:
                return False, 'fov_box_and_rendered_surface_disagree_full'
        elif fraction < 0.20 or s['edge_pixels'] < 8:
            return False, 'fov_no_clear_rendered_surface_crop'
    return True, None


def occlusion_class(task, measured, stats, masks):
    """Conservative categories requiring both environment geometry and real depth.

    Ratios below are over the labelled target's projected box footprint. They
    are disclosed as proxies, not claimed to be percentages of its true contour.
    """
    a, b = measured['objects'][:2]
    aid, bid = str(a['id']), str(b['id'])
    if aid not in masks or bid not in masks or not stats[bid]['visible']:
        return None, {}, 'occluder_not_identifiable'
    target = masks[aid]
    count = max(int(target['silhouette'].sum()), 1)
    by_b = target['hidden'] & masks[bid]['visible']
    hidden = int(by_b.sum()) / count
    vis = int(target['visible'].sum()) / count
    in_frame = float(a['projection'].get('inside_fraction', 0))
    all_front = bool(a['projection'].get('all_corners_in_front', False))
    evidence = {'target_box_in_frame_fraction': in_frame, 'target_box_pixels': count,
                'depth_supported_blocker_coverage': hidden,
                'depth_supported_target_visible_fraction': vis,
                'definition_limit': 'Depth-supported box attribution; no native object mask.'}
    if in_frame <= 0.005 and int(target['visible'].sum()) == 0:
        return 'outside', evidence, None
    if not all_front or in_frame < 0.95:
        return None, evidence, 'occlusion_target_cropped_or_camera_plane'
    metric = measured['metric']
    success = bool(metric.get('success'))
    if hidden >= 0.95 and vis <= 0.01 and not stats[aid]['visible'] and success:
        return 'fully_hidden', evidence, None
    if 0.20 <= hidden <= 0.65 and vis >= 0.25 and stats[aid]['visible'] and not success:
        return 'partly_hidden', evidence, None
    if hidden <= 0.03 and vis >= 0.35 and stats[aid]['visible'] and not success:
        return 'not_hidden', evidence, None
    return None, evidence, 'occlusion_ambiguous_or_environment_disagreement'


def wide_center_agrees(depth, alpha, wide_depth, wide_alpha):
    """Check actual render consistency at the unchanged central pixel rays."""
    if depth.shape != (256, 256) or alpha.shape != depth.shape or wide_depth.shape != (384, 384) or wide_alpha.shape != wide_depth.shape:
        return False, {'reason':'wide_center_shape_mismatch'}
    center=wide_depth[64:320,64:320]; opacity=wide_alpha[64:320,64:320]
    finite=all(np.isfinite(x).all() for x in (depth,alpha,center,opacity))
    valid=finite and np.allclose(depth,center,atol=1e-4,rtol=1e-3) and np.allclose(alpha,opacity,atol=1e-4,rtol=1e-3)
    return bool(valid), {'depth_max_abs_m':float(np.max(np.abs(depth-center))),
                         'alpha_max_abs':float(np.max(np.abs(alpha-opacity))),
                         'absolute_tolerance':1e-4,'relative_tolerance':1e-3}
