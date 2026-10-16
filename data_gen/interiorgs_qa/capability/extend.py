"""Bounded CPU-only reachable-state extension of the frozen capability source.

The deterministic beam enumerates real environment actions. Answer deficits
select already measured states; they never change actions, tasks or question
rules. All geometry, especially two-image occlusion, awaits actual depth/RGB.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from data_gen.interiorgs_v10.contract import ACTIONS, apply_action
from data_gen.interiorgs_v10.occlusion import face_occluders_for, face_identifiable
from data_gen.interiorgs_v10.start_context import start_refs
from . import rules
from .sources import SCENES, check_path, jsonl, load_task_scene, pose_key, sha256

VERSION = "capability_reachable_extension_v2"
META = Path('/mnt/umm/users/yinbaiqiao/InteriorGS_meta')
SEMANTIC_KEYS = {
    'absolute_distance': ('distance_0', 'distance_1', 'distance_2', 'distance_3'),
    'equidistance_closer': ('a_closer', 'b_closer', 'same'),
    'projective_direction': ('left', 'right', 'front', 'behind'),
    'occlusion_closer': ('a_closer', 'b_closer'),
    'occlusion_state': ('fully_hidden', 'partly_hidden', 'not_hidden', 'outside'),
    'fov_complete': ('both', 'a_only', 'b_only', 'neither'),
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def write_rows(path, rows):
    with Path(path).open('x') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')


def refs_unchanged(scene, task, pose):
    current = start_refs(scene, task, pose, obs_fn=face_identifiable)
    if current is None:
        return False
    original = {str(o['id']): o for o in task['target_object']['objects']}
    return all(oid in original and original[oid]['ref'] == new['ref'] for oid, new in current.items())


def replay_seed(task, state, scene):
    """Recheck the entire original prefix, not just the extension endpoint."""
    actions = list(state['actions_from_initial'])
    if any(a not in ACTIONS for a in actions):
        raise ValueError('unknown source action')
    poses = [np.asarray(task['init_camera']['extrinsics'], float)]
    for action in actions:
        poses.append(apply_action(poses[-1], action))
    if not np.allclose(poses[-1], state['c2w'], atol=1e-8, rtol=0):
        raise ValueError('source action prefix does not reproduce its saved pose')
    result = check_path(scene, poses)[-1]
    return poses[-1], result


def check_step(scene, previous, candidate):
    # The prefix was already checked; reuse exactly the same endpoint,
    # transition and additional wall-clearance checks for the new edge.
    return check_path(scene, [previous, candidate])[-1]


def occlusion_geometry_candidate(measured):
    """Only a rendering priority, never a depth-verified occlusion answer."""
    if measured['task_type'] != 'occlusion_alignment':
        return None
    a, b = measured['objects']
    if not b['visibility'].get('visible') or any(not r['visibility'].get('visible') for r in b['references']):
        return None
    bp, ap = b['projection'], a['projection']
    if not bp['all_corners_in_front'] or bp['inside_fraction'] < .5:
        return None
    fraction = float(ap.get('inside_fraction', 0))
    hidden = float(measured['metric'].get('target_hidden_fraction', 0))
    visible = bool(a['visibility'].get('visible'))
    success = bool(measured['metric'].get('success'))
    if fraction <= .005 and not visible:
        key = 'outside'
    elif not ap['all_corners_in_front'] or fraction < .95:
        return None
    elif hidden >= .95 and not visible and success:
        key = 'fully_hidden'
    elif .20 <= hidden <= .65 and visible and not success:
        key = 'partly_hidden'
    elif hidden <= .03 and visible and not success:
        key = 'not_hidden'
    else:
        return None
    return {'kind': 'occlusion_state', 'answer_key': key, 'geometry_only': True,
            'final_eligible': False, 'needs_rendered_depth': True,
            'target_box_in_frame_fraction': fraction, 'environment_box_hidden_fraction': hidden,
            'definition_limit': 'Geometry sampling only; actual depth attribution and human review pending.'}


def identifying_prefix(task_id, actions, source_states, source_geometry):
    possible = []
    for s in source_states:
        prefix = s['actions_from_initial']
        if (s['parent_task_id'] != task_id or not s.get('sample_eligible')
                or not s.get('runtime_legal') or len(prefix) >= len(actions)
                or actions[:len(prefix)] != prefix):
            continue
        m = source_geometry.get(s['state_id'], {}).get('measurement')
        if not m:
            continue
        a = m['objects'][0]
        if a['visibility'].get('visible') and all(r['visibility'].get('visible') for r in a['references']):
            possible.append((len(prefix), s['state_id']))
    return min(possible)[1] if possible else None


def desired_counts(total):
    # Five source families share the target; the two occlusion question kinds
    # split that family's allocation. Never change a question to fill a slot.
    per_family = math.ceil(total / 5)
    out = Counter()
    for kind, keys in SEMANTIC_KEYS.items():
        budget = math.ceil(per_family / 2) if kind.startswith('occlusion_') else per_family
        for i, key in enumerate(keys):
            out[f'{kind}:{key}'] = budget // len(keys) + (i < budget % len(keys))
    return out


def rec_categories(record):
    out = {f"{d['kind']}:{d['answer_key']}" for d in record['drafts']}
    extra = record.get('occlusion_geometry_candidate')
    if extra:
        out.add(f"occlusion_state:{extra['answer_key']}")
    return out


def view_key(scene, pose, K):
    # Same identity as capability.render, without importing the GPU renderer.
    return digest({'scene': scene, 'pose': np.asarray(pose).round(12).tolist(),
                   'K': np.asarray(K).round(12).tolist()})[:24]


def build(root, out, existing_geometry, *, meta_root=META, max_checks=20000,
          per_parent=400, depth=5, beam_width=12, target_total=500):
    if not (2 <= depth <= 5 and 0 < max_checks <= 20000 and 0 < per_parent <= 400
            and 0 < beam_width <= 32 and target_total > 0):
        raise ValueError('extension limits exceed the authorised bounds')
    root, out, gp = Path(root).resolve(), Path(out).resolve(), Path(existing_geometry).resolve()
    if out.exists():
        raise FileExistsError('use a fresh extension version; previous output is immutable')
    sources = {n: root / n for n in ('tasks.jsonl', 'states.jsonl', 'source_inventory.json')}
    source_hashes = {n: sha256(p) for n, p in sources.items()}
    geometry_hash = sha256(gp)
    tasks, old_states = list(jsonl(sources['tasks.jsonl'])), list(jsonl(sources['states.jsonl']))
    by_task = {t['task_id']: t for t in tasks}
    if (len(tasks) != 56 or len(by_task) != 56 or set(t['scene_id'] for t in tasks) != set(SCENES)
            or any(t.get('split') != 'train' or t.get('reserve') is not False for t in tasks)):
        raise ValueError('extension requires exactly the same four-scene 56 formal train tasks')
    inventory = json.loads(sources['source_inventory.json'].read_text())
    for n in ('tasks.jsonl', 'states.jsonl'):
        if source_hashes[n] != inventory['output_sha256'][n]:
            raise ValueError('frozen input changed: ' + n)
    for files in inventory['assets'].values():
        for item in files.values():
            if sha256(item['path']) != item['sha256']:
                raise ValueError('scene metadata changed since source preparation')
    old_geometry = {r['state_id']: r for r in jsonl(gp)}
    if set(old_geometry) != {s['state_id'] for s in old_states}:
        raise ValueError('existing geometry must cover exactly the source states')
    existing_poses = {pose_key(s['scene_id'], s['c2w'], s['native_intrinsics']) for s in old_states}
    baseline, baseline_rows, baseline_poses = Counter(), [], set()
    for s in old_states:
        t, old = by_task[s['parent_task_id']], old_geometry[s['state_id']]
        rec = {k: v for k, v in old.items() if k not in ('drafts', 'reasons')}
        rec.update(drafts=[], reasons=['no_eligible_previous_measurement'])
        if s.get('runtime_legal') and s.get('sample_eligible') and old.get('measurement'):
            rec['drafts'], rec['reasons'] = rules.propose(t, old['measurement'])
            if identifying_prefix(t['task_id'], s['actions_from_initial'], old_states, old_geometry):
                candidate = occlusion_geometry_candidate(old['measurement'])
                if candidate:
                    rec['occlusion_geometry_candidate'] = candidate
            categories = rec_categories(rec)
            if categories and s['pose_key'] not in baseline_poses:
                baseline.update(categories)
                baseline_poses.add(s['pose_key'])
        baseline_rows.append(rec)
    targets, counts = desired_counts(target_total), baseline.copy()
    grouped = defaultdict(list)
    for s in old_states:
        if s.get('runtime_legal') and s.get('sample_eligible'):
            grouped[s['parent_task_id']].append(s)
    output, geometry, attempts, parent_ledgers = [], [], [], []
    accepted_poses, scenes = set(), {}
    checked, started = 0, time.monotonic()
    # Scarce families first; frontier/action order itself is answer-independent.
    def priority(t):
        kind = rules.TASK_KINDS.get(t['task_type'])
        return (sum(baseline[k] for k in targets if k.startswith(str(kind)+':')), t['task_id'])
    for task in sorted(tasks, key=priority):
        tid, sid = task['task_id'], task['scene_id']
        if task['task_type'] not in rules.TASK_KINDS:
            parent_ledgers.append({'parent_task_id': tid, 'reason': 'family_paused', 'checked': 0})
            continue
        if checked >= max_checks or len(baseline_poses) + len(output) >= target_total:
            break
        if sid not in scenes:
            scenes[sid] = load_task_scene(task, meta_root)
        scene, occ = scenes[sid], face_occluders_for(scenes[sid])
        if not math.isclose(float(scene.min_wall_distance), .5, abs_tol=1e-9):
            raise ValueError('additional sample wall clearance must remain 0.5 metres')
        seeds = sorted(grouped[tid], key=lambda s: (s.get('existing_rgb') is None, digest(s['state_id'])))[:8]
        frontier, visited = [], set()
        parent_start, parent_selected, rejects = checked, len(output), Counter()
        for s in seeds:
            pose, check = replay_seed(task, s, scene)
            if not check['sample_eligible']:
                rejects['source_prefix_not_eligible'] += 1
                continue
            key = pose_key(sid, pose, task['init_camera']['intrinsics'])
            if key not in visited:
                visited.add(key)
                frontier.append((pose, list(s['actions_from_initial']), s))
        for layer in range(1, depth + 1):
            next_layer = []
            for previous, actions, seed in frontier:
                for action in ACTIONS:
                    if checked >= max_checks or checked-parent_start >= per_parent:
                        break
                    pose = apply_action(previous, action)
                    key = pose_key(sid, pose, task['init_camera']['intrinsics'])
                    if key in visited:
                        rejects['same_parent_pose_duplicate'] += 1
                        continue
                    visited.add(key)
                    checked += 1
                    prefix = actions + [action]
                    check = check_step(scene, previous, pose)
                    audit = {'parent_task_id': tid, 'pose_key': key, 'extension_depth': layer,
                             'actions_from_initial': prefix}
                    if not check['sample_eligible']:
                        why = (check['runtime_rejections'] or check['sample_rejections'])[-1]['reason']
                        rejects[why] += 1
                        attempts.append({**audit, 'reason': why})
                        continue
                    next_layer.append((pose, prefix, seed))
                    if key in existing_poses or key in accepted_poses:
                        rejects['already_recorded_scene_camera'] += 1
                        attempts.append({**audit, 'reason': 'already_recorded_scene_camera'})
                        continue
                    if not refs_unchanged(scene, task, pose):
                        rejects['reference_changed_or_ambiguous'] += 1
                        attempts.append({**audit, 'reason': 'reference_changed_or_ambiguous'})
                        continue
                    measured = rules.measure(task, pose, occ)
                    drafts, reasons = rules.propose(task, measured)
                    state_id = f'{tid}__ext2_{key[:16]}'
                    rec = {'state_id': state_id, 'task_type': task['task_type'], 'scene_id': sid,
                           'parent_task_id': tid, 'measurement': measured, 'drafts': drafts, 'reasons': reasons,
                           'view_key': view_key(sid, pose, measured['camera']['K_effective'])}
                    first = identifying_prefix(tid, prefix, old_states, old_geometry)
                    candidate = occlusion_geometry_candidate(measured) if first else None
                    if candidate:
                        rec.update(occlusion_geometry_candidate=candidate, occlusion_reference_state_id=first)
                    categories = rec_categories(rec)
                    useful = sorted(k for k in categories if counts[k] < targets[k])
                    if not useful:
                        why = reasons[0] if not categories and reasons else 'semantic_slot_already_sufficient'
                        rejects[why] += 1
                        attempts.append({**audit, 'reason': why, 'candidate_categories': sorted(categories)})
                        continue
                    record = {'state_id': state_id, 'scene_id': sid, 'parent_task_id': tid,
                              'c2w': pose.tolist(), 'actions_from_initial': prefix,
                              'source_kind': 'legal_extension_v2', 'existing_rgb': None,
                              'existing_rgb_sha256': None, 'frame_id': None,
                              'trajectory_parent_id': seed['trajectory_parent_id'], 'pose_key': key,
                              'native_intrinsics': task['init_camera']['intrinsics'], 'split': 'train',
                              **check, 'extension_from': seed['state_id'],
                              'extension_depth': len(prefix)-len(seed['actions_from_initial']),
                              'geometry_only': True, 'training_allowed': False,
                              'fills_semantic_slots': useful}
                    output.append(record); geometry.append(rec); accepted_poses.add(key); counts.update(categories)
                    attempts.append({**audit, 'reason': 'selected_geometry_pending_depth', 'fills_semantic_slots': useful})
                    if len(baseline_poses) + len(output) >= target_total:
                        break
                if checked >= max_checks or checked-parent_start >= per_parent or len(baseline_poses)+len(output) >= target_total:
                    break
            frontier = sorted(next_layer, key=lambda node: digest([tid, layer, node[1]]))[:beam_width]
            if not frontier or checked >= max_checks or checked-parent_start >= per_parent or len(baseline_poses)+len(output) >= target_total:
                break
        parent_ledgers.append({'parent_task_id': tid, 'scene_id': sid, 'task_type': task['task_type'],
                               'checked': checked-parent_start, 'selected_states': len(output)-parent_selected,
                               'rejections': dict(rejects), 'completed_extension_depth': layer if seeds else 0})
        print(json.dumps({'parent': tid, 'checked_states': checked, 'new_geometry_states': len(output),
                          'seconds': round(time.monotonic()-started, 2)}), flush=True)
    if any(sha256(p) != source_hashes[n] for n, p in sources.items()) or sha256(gp) != geometry_hash:
        raise ValueError('source changed while extension was running; no output installed')
    out.mkdir(parents=True, exist_ok=False)
    # Exact bytes, so consumers can independently compare the frozen tasks.
    (out/'tasks.jsonl').write_bytes(sources['tasks.jsonl'].read_bytes())
    write_rows(out/'states.jsonl', output)
    write_rows(out/'geometry.jsonl', geometry)
    write_rows(out/'baseline_recomputed_geometry.jsonl', baseline_rows)
    write_rows(out/'attempt_ledger.jsonl', attempts)
    repo = Path(__file__).resolve().parents[3]
    code = [Path(__file__), Path(rules.__file__), Path(__file__).with_name('sources.py'),
            repo/'data_gen/interiorgs_v10/occlusion.py', repo/'data_gen/interiorgs_v10/contract.py',
            repo/'data_gen/interiorgs_v10/scene.py', repo/'data_gen/interiorgs_v10/start_context.py',
            repo/'vagen/envs/active_spatial/target_visibility.py']
    ledger = {'version': VERSION, 'created_utc': datetime.now(timezone.utc).isoformat(),
              'source_root': str(root), 'source_files': source_hashes,
              'source_geometry': {'path': str(gp), 'sha256': geometry_hash},
              'code_sha256': {str(p): sha256(p) for p in code}, 'config': rules.load_config(),
              'limits': {'max_checks': max_checks, 'per_parent': per_parent, 'depth': depth,
                         'beam_width': beam_width, 'target_total_states': target_total},
              'search': 'Deterministic answer-independent action beam; only final state selection uses semantic deficits.',
              'checked_states': checked, 'baseline_geometry_states': len(baseline_poses),
              'new_geometry_states': len(output), 'elapsed_seconds': time.monotonic()-started,
              'baseline_categories': dict(baseline), 'combined_categories': dict(counts),
              'desired_categories': dict(targets),
              'remaining_deficits': {k: targets[k]-counts[k] for k in targets if counts[k] < targets[k]},
              'parents': parent_ledgers, 'wall_model': 'thick', 'sample_wall_clearance_m': .5,
              'occluder_provider': 'face_occluders_for(scene), target_visibility.OccluderSet.from_detector',
              'two_image_occlusion': 'Geometry-only rendering priorities; earlier identifying source state required; depth/RGB and human checks pending.',
              'human_approved': 0, 'training_released': 0, 'rgb_rendered': False, 'model_run': False,
              'output_sha256': {n: sha256(out/n) for n in ('tasks.jsonl','states.jsonl','geometry.jsonl',
                                                         'baseline_recomputed_geometry.jsonl','attempt_ledger.jsonl')}}
    (out/'generation_ledger.json').write_text(json.dumps(ledger,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    return ledger


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True); p.add_argument('--out', type=Path, required=True)
    p.add_argument('--existing-geometry', type=Path, required=True)
    p.add_argument('--meta-root', type=Path, default=META)
    p.add_argument('--max-checks', type=int, default=20000); p.add_argument('--per-parent', type=int, default=400)
    p.add_argument('--depth', type=int, default=5); p.add_argument('--beam-width', type=int, default=12)
    p.add_argument('--target-total', type=int, default=500)
    a = p.parse_args()
    result = build(a.root, a.out, a.existing_geometry, meta_root=a.meta_root, max_checks=a.max_checks,
                   per_parent=a.per_parent, depth=a.depth, beam_width=a.beam_width, target_total=a.target_total)
    print(json.dumps({k: result[k] for k in ('checked_states','baseline_geometry_states','new_geometry_states','remaining_deficits')}), flush=True)


if __name__ == '__main__':
    main()
