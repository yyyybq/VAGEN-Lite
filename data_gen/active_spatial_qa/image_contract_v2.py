"""Versioned camera/image checks; never infer target visibility from variance."""
import numpy as np

VERSION = 'active_spatial_qa_image_contract_v3_scene_geometry'


def pose_legality(position, forward, *, min_height=0.05, max_height=8.0):
    """Validate camera geometry before scoring any state.

    This is deliberately a geometry gate, not a task-success gate: a valid
    camera may still be a legitimate negative example.
    """
    p = np.asarray(position, dtype=float)
    f = np.asarray(forward, dtype=float)
    reasons = []
    if p.shape != (3,) or not np.isfinite(p).all(): reasons.append('nonfinite_position')
    if f.shape != (3,) or not np.isfinite(f).all() or np.linalg.norm(f) < 1e-8: reasons.append('invalid_forward')
    if not reasons and not (float(min_height) <= p[2] <= float(max_height)): reasons.append('height_out_of_range')
    return {'valid': not reasons, 'reasons': reasons, 'position': p.tolist(), 'forward': f.tolist()}


def classify_scene_pose(position, collision_detector, layout=None, *, min_height=0.3, max_height=2.5):
    """Classify against the real room/object geometry, not scene AABB."""
    p = np.asarray(position, dtype=float)
    reasons = []
    if p.shape != (3,) or not np.isfinite(p).all():
        return {'status': 'coordinate_unconfirmed', 'reasons': ['nonfinite_position']}
    if p[2] < min_height or p[2] > max_height:
        reasons.append('height_illegal')
    if not getattr(collision_detector, 'scene_loaded', False):
        return {'status': 'coordinate_unconfirmed', 'reasons': reasons + ['scene_geometry_unloaded']}
    if getattr(collision_detector, 'structure_convention_status', 'unverified') not in ('frozen',):
        return {'status': 'coordinate_unconfirmed', 'reasons': reasons + ['structure_convention_unverified']}
    if layout is None or not getattr(layout, 'room_polys', None):
        return {'status': 'coordinate_unconfirmed', 'reasons': reasons + ['room_polygon_unavailable']}
    if not layout.contains_xy(p[:2]):
        reasons.append('outside_room_polygon')
    wall_distance = float(layout.wall_distance(p[:2]))
    required_clearance = collision_detector.camera_radius + collision_detector.safety_margin
    if wall_distance < required_clearance:
        reasons.append('room_wall_clearance')
    result = collision_detector.check_collision(p)
    if result.has_collision:
        reasons.append(f'collision_{result.collision_type}')
    object_hits = [box.ins_id for box in collision_detector.object_boxes if box.contains(p)]
    # A loaded detector with room wall segments is the strongest available
    # evidence; collision-free alone does not claim RGB/reconstruction cover.
    return {'status': 'legal_indoor' if not reasons else ('wall_or_obstacle' if any(x in reasons for x in ('collision_wall','collision_object','room_wall_clearance')) else 'illegal'),
            'reasons': reasons, 'collision': {'type': result.collision_type, 'object': result.collision_object,
                                               'distance': float(result.distance_to_collision)},
            'height': {'value': float(p[2]), 'min': min_height, 'max': max_height, 'valid': 'height_illegal' not in reasons},
            'object_collision': {'hits': object_hits, 'valid': not object_hits},
            'coordinate_convention': collision_detector.structure_convention_status,
            'room_polygon': {'contains': 'outside_room_polygon' not in reasons, 'wall_distance': wall_distance, 'required_clearance': required_clearance}}


def pose_from_forward(position, forward):
    p = np.asarray(position, dtype=float)
    z = np.asarray(forward, dtype=float)
    if p.shape != (3,) or z.shape != (3,) or not np.isfinite(p).all() or not np.isfinite(z).all():
        raise ValueError('position and forward must be finite length-three vectors')
    length = np.linalg.norm(z)
    if length < 1e-8:
        raise ValueError('zero camera forward')
    z = z / length
    up = np.array([0.0, 0.0, 1.0])
    right = np.cross(z, up)
    if np.linalg.norm(right) < 1e-8:
        right = np.cross(z, np.array([0.0, 1.0, 0.0]))
    right /= np.linalg.norm(right)
    down = np.cross(z, right)
    pose = np.eye(4)
    pose[:3, :3] = np.column_stack([right, down, z])
    pose[:3, 3] = p
    return pose


def image_stats(image):
    a = np.asarray(image.convert('RGB'), dtype=np.float32)
    spatial_std = float(a.std(axis=(0, 1), dtype=np.float64).max())
    return {'mean': float(a.mean()), 'std': float(a.std()), 'min': float(a.min()), 'max': float(a.max()),
            'spatial_channel_std_max': spatial_std, 'nonblank': spatial_std >= 5.0, 'quality_check_version': VERSION,
            'target_visibility_verified': False}
