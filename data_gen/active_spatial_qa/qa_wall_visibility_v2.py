"""QA-only, fail-closed structural sight-line prefilter (no renderer imports).

Use StructuralVisibility(scene_path, y_sign=goal['collision_convention']
['structure_y_sign']).check(camera_xyz, goal['target_object']['objects']).
Never infer or flip the coordinate convention. Walls are finite, thick prisms;
DOOR/WINDOW profiles define potential through-wall apertures. An open profile
does not establish that its door leaf/glass/furniture is transparent in the RGB.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

VERSION = 'qa_structural_visibility_v2'
_EPS = 1e-9


def _vec(value, length, name):
    result = np.asarray(value, dtype=float)
    if result.shape != (length,) or not np.isfinite(result).all():
        raise ValueError(name + ' must be a finite vector of length ' + str(length))
    return result


def _cross(a, b):
    return float(a[0] * b[1] - a[1] * b[0])


def _point_segment_distance(p, a, b):
    d = b - a
    norm2 = float(d @ d)
    if norm2 <= _EPS * _EPS:
        return float(np.linalg.norm(p-a))
    t = min(1., max(0., float((p-a) @ d) / norm2))
    return float(np.linalg.norm(p-a-t*d))


def _segment_distance(a, b, c, d):
    """Exact 2D intersection test, otherwise nearest endpoint-to-segment."""
    ab, cd = b-a, d-c
    den = _cross(ab, cd)
    if abs(den) > _EPS:
        t, u = _cross(c-a, cd)/den, _cross(c-a, ab)/den
        if -_EPS <= t <= 1+_EPS and -_EPS <= u <= 1+_EPS:
            return 0.
    return min(_point_segment_distance(a, c, d), _point_segment_distance(b, c, d),
               _point_segment_distance(c, a, b), _point_segment_distance(d, a, b))


def _inside_polygon(point, polygon):
    # Boundary is handled conservatively by the distance-to-edge check below.
    x, z = point
    inside = False
    for a, b in zip(polygon, np.roll(polygon, -1, axis=0)):
        if (a[1] > z) != (b[1] > z):
            at_x = a[0] + (z-a[1]) * (b[0]-a[0]) / (b[1]-a[1])
            if x < at_x:
                inside = not inside
    return inside


def _opening_contains_segment(start, end, polygon, edge_margin):
    """The entire path through wall thickness must fit the actual profile.

    Endpoints alone would miss a concave/arched profile crossing. Testing every
    polygon edge also rejects grazing a jamb, sill or lintel within the margin.
    """
    distance = min(_segment_distance(start, end, a, b)
                   for a, b in zip(polygon, np.roll(polygon, -1, axis=0)))
    ok = bool(_inside_polygon(start, polygon) and _inside_polygon(end, polygon)
              and distance > edge_margin + _EPS)
    return ok, distance


def _prism_interval(start, end, low, high):
    delta = end-start
    enter, leave = 0., 1.
    for axis in range(3):
        if abs(delta[axis]) < _EPS:
            if start[axis] < low[axis]-_EPS or start[axis] > high[axis]+_EPS:
                return None
            continue
        t0 = (low[axis]-start[axis])/delta[axis]
        t1 = (high[axis]-start[axis])/delta[axis]
        enter, leave = max(enter, min(t0,t1)), min(leave, max(t0,t1))
        if leave < enter-_EPS:
            return None
    # A point touch only at the target or camera endpoint is not an intervening wall.
    enter, leave = max(enter, 1e-7), min(leave, 1-1e-7)
    if leave-enter <= _EPS:
        return None
    return float(enter), float(leave)


class StructuralVisibility:
    def __init__(self, scene_path, y_sign, opening_edge_margin_m=0.002,
                 association_tolerance_m=0.005):
        self.path = Path(scene_path)
        if self.path.suffix != '.json':
            self.path = self.path/'structure.json'
        self.ready = False
        self.load_error = None
        self.structure_sha256 = None
        self.y_sign = None
        self.walls = []
        self.hole_associations = []
        self.opening_edge_margin_m = float(opening_edge_margin_m)
        try:
            if isinstance(y_sign, bool) or y_sign is None or float(y_sign) not in (-1., 1.):
                raise ValueError('explicit frozen structure_y_sign must be +1 or -1')
            self.y_sign = float(y_sign)
            if not np.isfinite(self.opening_edge_margin_m) or self.opening_edge_margin_m < 0:
                raise ValueError('invalid opening edge margin')
            tolerance = float(association_tolerance_m)
            if not np.isfinite(tolerance) or tolerance < 0 or tolerance > .02:
                raise ValueError('invalid opening association tolerance')
            raw = self.path.read_bytes()
            self.structure_sha256 = hashlib.sha256(raw).hexdigest()
            structure = json.loads(raw)
            if not isinstance(structure.get('walls'), list) or not structure['walls']:
                raise ValueError('no complete structural walls loaded')
            for wi, item in enumerate(structure['walls']):
                points = np.asarray(item['location'], dtype=float)
                if points.shape != (2,2) or not np.isfinite(points).all():
                    raise ValueError('invalid wall location at index '+str(wi))
                a, b = points * [1., self.y_sign]
                length = float(np.linalg.norm(b-a))
                height, thickness = float(item['height']), float(item['thickness'])
                # This project's 2D-location wall schema has a floor at z=0.
                # Explicit base_z is supported without silently discarding it.
                base_z = float(item.get('base_z', 0.))
                if not np.isfinite([length,height,thickness,base_z]).all() or min(length,height,thickness) <= 0:
                    raise ValueError('invalid wall extent at index '+str(wi))
                direction = (b-a)/length
                wall = {'index':wi, 'a':a, 'b':b, 'direction':direction,
                        'normal':np.array([-direction[1],direction[0]]), 'length':length,
                        'height':height, 'base_z':base_z, 'thickness':thickness, 'holes':[]}
                self.walls.append(wall)
            holes = structure.get('holes', [])
            if not isinstance(holes, list):
                raise ValueError('holes must be a list')
            for hi, item in enumerate(holes):
                kind = str(item['type']).upper()
                points = np.asarray(item['profile'], dtype=float)
                thickness = float(item['thickness'])
                if points.ndim != 2 or points.shape[1] != 3 or len(points)<3 or not np.isfinite(points).all() or not np.isfinite(thickness) or thickness <= 0:
                    raise ValueError('invalid 3D opening profile at index '+str(hi))
                points = points * [1., self.y_sign, 1.]
                associated = []
                for wall in self.walls:
                    normal = (points[:,:2]-wall['a']) @ wall['normal']
                    # A profile must be planar in the same wall frame, and on
                    # that wall's middle/face plane; no enclosing-rectangle cut.
                    if float(np.ptp(normal)) > tolerance or float(np.max(np.abs(normal))) > max(wall['thickness'],thickness)/2+tolerance:
                        continue
                    us = (points[:,:2]-wall['a']) @ wall['direction']
                    if float(us.max()) <= 0 or float(us.min()) >= wall['length']:
                        continue
                    if float(points[:,2].max()) <= wall['base_z'] or float(points[:,2].min()) >= wall['base_z']+wall['height']:
                        continue
                    polygon = np.column_stack([us,points[:,2]])
                    # Drop a repeated closing vertex, which is not an edge.
                    if np.linalg.norm(polygon[0]-polygon[-1]) <= _EPS:
                        polygon = polygon[:-1]
                    area = abs(sum(_cross(a,b) for a,b in zip(polygon,np.roll(polygon,-1,axis=0))))/2
                    if len(polygon)<3 or area <= _EPS:
                        raise ValueError('degenerate opening profile at index '+str(hi))
                    associated.append(wall['index'])
                    if kind in {'DOOR','WINDOW'}:
                        wall['holes'].append({'index':hi,'type':kind,'polygon_uz':polygon,
                                              'profile_xyz':points,'thickness':thickness})
                # An unassociated opening never creates transparency. It is
                # recorded, while every existing wall remains fully opaque.
                self.hole_associations.append({'hole_index':hi,'type':kind,
                    'wall_indices':associated,'supported_potential_aperture':kind in {'DOOR','WINDOW'},
                    'unassociated_treated_as_opaque':not bool(associated)})
            self.ready = True
        except (OSError, ValueError, KeyError, TypeError) as error:
            self.load_error = type(error).__name__+': '+str(error)

    def _metadata(self):
        return {'version':VERSION,'structure_path':str(self.path),
                'structure_sha256':self.structure_sha256,'structure_y_sign':self.y_sign,
                'structure_loaded':self.ready,'load_error':self.load_error,
                'opening_edge_margin_m':self.opening_edge_margin_m,
                'scope':'Structural geometry prefilter only; not an RGB visibility certificate.',
                'rgb_visibility_verified':False,'furniture_occlusion_verified':False,
                'opening_state':'DOOR/WINDOW profiles are potentially open; leaf/glass appearance unverified.'}

    def trace(self, camera_xyz, end_xyz):
        result = {'clear':False,'end_xyz':None,'blockers':[],'wall_crossings':[]}
        if not self.ready:
            return dict(result,reason='structure_unavailable_or_coordinate_unknown',load_error=self.load_error)
        try:
            start, end = _vec(camera_xyz,3,'camera'), _vec(end_xyz,3,'ray endpoint')
            result['end_xyz'] = end.tolist()
            if np.linalg.norm(end-start) <= _EPS:
                return dict(result,reason='degenerate_ray')
            for wall in self.walls:
                def local(p):
                    xy = p[:2]-wall['a']
                    return np.array([xy @ wall['direction'],xy @ wall['normal'],p[2]])
                p0, p1 = local(start), local(end)
                interval = _prism_interval(p0,p1,np.array([0.,-wall['thickness']/2,wall['base_z']]),
                    np.array([wall['length'],wall['thickness']/2,wall['base_z']+wall['height']]))
                if interval is None:
                    continue
                t0,t1 = interval
                enter,leave = p0+t0*(p1-p0),p0+t1*(p1-p0)
                crossing = {'wall_index':wall['index'],'ray_fraction_enter':t0,'ray_fraction_exit':t1,
                    'entry_xyz':(start+t0*(end-start)).tolist(),'exit_xyz':(start+t1*(end-start)).tolist(),
                    'entry_uz':enter[[0,2]].tolist(),'exit_uz':leave[[0,2]].tolist(),
                    'wall_height_m':wall['height'],'wall_base_z_m':wall['base_z'],
                    'wall_thickness_m':wall['thickness'],'through_opening':None,'opening_checks':[]}
                for hole in wall['holes']:
                    fits, clearance = _opening_contains_segment(enter[[0,2]],leave[[0,2]],
                                                hole['polygon_uz'],self.opening_edge_margin_m)
                    check = {'hole_index':hole['index'],'type':hole['type'],'fits_entire_wall_thickness':fits,
                             'profile_edge_clearance_m':clearance,'profile_uz':hole['polygon_uz'].tolist(),
                             'hole_thickness_m':hole['thickness']}
                    crossing['opening_checks'].append(check)
                    if fits:
                        crossing['through_opening'] = {'hole_index':hole['index'],'type':hole['type'],
                                                      'profile_edge_clearance_m':clearance}
                        break
                result['wall_crossings'].append(crossing)
                if crossing['through_opening'] is None:
                    result['blockers'].append(dict(crossing,reason='solid_wall_or_opening_edge'))
            result['blockers'].sort(key=lambda b:b['ray_fraction_enter'])
            result['wall_crossings'].sort(key=lambda b:b['ray_fraction_enter'])
            result['clear'] = not result['blockers']
            result['reason'] = 'structural_ray_clear' if result['clear'] else 'structural_wall_occlusion'
            return result
        except (ValueError,TypeError) as error:
            return dict(result,reason='invalid_ray_geometry',error=str(error))

    def check(self, camera_xyz, objects, min_clear_fraction=1.0):
        """Center + nine inset points per camera-facing AABB surface.

        All probes are required by default. A target's center must always be
        clear even if an explicit caller requests a lower surface fraction.
        Returned probes are actual checked endpoints for a separate furniture
        test, which must exclude only the target itself, not the other target.
        """
        result = dict(self._metadata(),ok=False,reason=None,targets=[],
                      hole_associations=self.hole_associations)
        if not self.ready:
            return dict(result,reason='structure_unavailable_or_coordinate_unknown')
        try:
            camera = _vec(camera_xyz,3,'camera')
            fraction = float(min_clear_fraction)
            if not np.isfinite(fraction) or not 0 < fraction <= 1:
                raise ValueError('min_clear_fraction must be in (0,1]')
            if not isinstance(objects,(list,tuple)) or not objects:
                raise ValueError('target objects must be a nonempty list')
            for oi, obj in enumerate(objects):
                low, high = _vec(obj['bbox_min'],3,'bbox_min'),_vec(obj['bbox_max'],3,'bbox_max')
                if not np.all(high-low > _EPS):
                    raise ValueError('target bbox must have positive extent')
                center = _vec(obj.get('center',(low+high)/2),3,'target center')
                if np.any(center < low-1e-6) or np.any(center > high+1e-6):
                    raise ValueError('target center outside bbox')
                target = {'object_id':str(obj.get('id',obj.get('ins_id',oi))), 'label':obj.get('label'),
                          'ok':False,'reason':None,'probes':[]}
                result['targets'].append(target)
                if np.all(camera >= low-_EPS) and np.all(camera <= high+_EPS):
                    target['reason'] = 'camera_inside_target_bbox'
                    continue
                samples = [('center',center,None)]
                for axis in range(3):
                    face = low[axis] if camera[axis] < low[axis] else high[axis] if camera[axis] > high[axis] else None
                    if face is None:
                        continue
                    others = [i for i in range(3) if i != axis]
                    for uv in itertools.product((.2,.5,.8),repeat=2):
                        point = center.copy()
                        point[axis] = face
                        for other, f in zip(others,uv):
                            point[other] = low[other]+f*(high[other]-low[other])
                        samples.append(('front_surface',point,{'axis':'xyz'[axis],
                            'side':'min' if face==low[axis] else 'max','inset_fractions':list(uv)}))
                for kind, point, face in samples:
                    probe = self.trace(camera,point)
                    probe.update(kind=kind,face=face)
                    target['probes'].append(probe)
                clear = sum(p['clear'] for p in target['probes'])
                target['center_clear'] = bool(target['probes'][0]['clear'])
                target['clear_fraction'] = clear/len(target['probes'])
                target['probe_count'] = len(target['probes'])
                target['ok'] = bool(target['center_clear'] and target['clear_fraction'] >= fraction)
                target['reason'] = 'sampled_structural_rays_clear' if target['ok'] else 'structural_wall_occlusion'
            result['min_clear_fraction'] = fraction
            result['camera_xyz'] = camera.tolist()
            result['ok'] = all(t['ok'] for t in result['targets'])
            result['reason'] = 'sampled_structural_rays_clear' if result['ok'] else 'target_structural_visibility_rejected'
            return result
        except (ValueError,KeyError,TypeError) as error:
            return dict(result,ok=False,reason='invalid_target_or_camera_geometry',error=str(error))
