"""Render real depth for reachable states; preserve already rendered SFT RGB.

Run only in the authorised single-GPU SCO job. The wider FOV images are private
diagnostic evidence and are never model inputs. Existing successful cache files
are validated before reuse; failed attempts remain failures.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
from PIL import Image

from data_gen.interiorgs_qa.render import GSRenderer
from vagen.envs.active_spatial.target_visibility import item_camera


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def view_key(scene, pose, K):
    data = {'scene': scene, 'pose': np.asarray(pose).round(12).tolist(), 'K': np.asarray(K).round(12).tolist()}
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()[:24]


def cached_ok(path, key, need_wide):
    try:
        doc = json.loads(path.read_text())
        if not doc.get('ok') or doc['view_key'] != key or (need_wide and not doc.get('wide_depth')):
            return False
        for f, h in [('rgb', 'rgb_sha256'), ('depth', 'depth_sha256')]:
            if sha(doc[f]) != doc[h]:
                return False
        with Image.open(doc['rgb']) as im:
            im.load()
            if im.size != (256, 256):
                return False
        with np.load(doc['depth']) as d:
            if (d['depth'].shape != (256, 256) or d['alpha'].shape != (256, 256)
                    or not np.isfinite(d['depth']).all() or not np.isfinite(d['alpha']).all()
                    or np.any(d['alpha'] < 0) or np.any(d['alpha'] > 1)):
                return False
        # Finalization consumes a stored wide view whenever it is present, even
        # if this particular task does not need cropping evidence.
        if doc.get('wide_depth'):
            if sha(doc['wide_depth']) != doc['wide_depth_sha256']:
                return False
            with np.load(doc['wide_depth']) as d:
                if (d['depth'].shape != (384, 384) or d['alpha'].shape != (384, 384)
                        or not np.isfinite(d['depth']).all() or not np.isfinite(d['alpha']).all()
                        or np.any(d['alpha'] < 0) or np.any(d['alpha'] > 1)):
                    return False
        return True
    except (OSError, ValueError, KeyError):
        return False


def run(out, gs_root):
    out = Path(out).resolve()
    owner = out.stat()
    tasks = {r['task_id']: r for r in read_rows(out / 'tasks.jsonl')}
    states = read_rows(out / 'states.jsonl')
    views = {}
    for state in states:
        if not state.get('sample_eligible', True):
            continue
        task = tasks[state['parent_task_id']]
        # Occupancy remains design-only until the user decides its definition.
        if task['task_type'] == 'screen_occupancy':
            continue
        K, w, h = item_camera(task)
        if (w, h) != (256, 256):
            raise ValueError('Model camera resolution must match Act-SFT')
        pose = np.asarray(state['c2w'], float)
        if abs(pose[2, 2]) > 1e-8 or abs(pose[2, 3] - 1.5) > 1e-8:
            raise ValueError('unreachable pitch/height reached renderer')
        key = view_key(state['scene_id'], pose, K)
        row = views.setdefault(key, {'key': key, 'scene': state['scene_id'], 'pose': pose, 'K': K,
                                    'existing_rgb': None, 'existing_rgb_sha256': None, 'wide': False,
                                    'state_ids': []})
        row['state_ids'].append(state['state_id'])
        row['wide'] |= task['task_type'] == 'fov_inclusion'
        if state.get('existing_rgb'):
            row['existing_rgb'] = state['existing_rgb']
            row['existing_rgb_sha256'] = state.get('existing_rgb_sha256')
    cache = out / 'rendered'
    cache.mkdir(exist_ok=True)
    renderer, current = None, None
    records, failures, reused = [], [], 0
    run_id = str(time.time_ns())
    try:
        for index, row in enumerate(sorted(views.values(), key=lambda r: (r['scene'], r['key']))):
            dest = cache / row['key']
            dest.mkdir(exist_ok=True)
            status = dest / 'status.json'
            if cached_ok(status, row['key'], row['wide']):
                records.append(json.loads(status.read_text()))
                reused += 1
                continue
            attempt = dest / ('attempt_' + str(time.time_ns()))
            attempt.mkdir()
            rec = {'ok': False, 'view_key': row['key'], 'scene_id': row['scene'],
                   'state_ids': row['state_ids'], 'c2w': row['pose'].tolist(),
                   'K_effective': np.asarray(row['K']).tolist(), 'started_at': time.time()}
            try:
                if row['scene'] != current:
                    if renderer is None:
                        renderer = GSRenderer(gs_root)
                    renderer.set_scene(row['scene'])
                    current = row['scene']
                rgb, depth, alpha = renderer.render(row['pose'], 256, row['K'])
                rgb = np.asarray(rgb, np.uint8)
                if rgb.shape != (256, 256, 3) or float(rgb.std()) < 3:
                    raise ValueError('unusable RGB shape or nearly constant image')
                if not np.isfinite(depth).all() or depth.shape != (256, 256):
                    raise ValueError('invalid real depth')
                rgb_path = attempt / 'rgb.png'
                comparison = None
                if row['existing_rgb']:
                    old = Path(row['existing_rgb'])
                    if sha(old) != row['existing_rgb_sha256']:
                        raise ValueError('source RGB SHA changed')
                    with Image.open(old) as im:
                        old_rgb = np.array(im.convert('RGB'))
                    delta = np.abs(old_rgb.astype(float) - rgb.astype(float))
                    comparison = {'max_abs': float(delta.max()), 'mean_abs': float(delta.mean()),
                                  'source': str(old), 'source_sha256': sha(old)}
                    if comparison['max_abs'] > 1:
                        raise ValueError('same-camera source RGB and current depth-render RGB disagree')
                    shutil.copyfile(old, rgb_path)
                    rec['rgb_source'] = 'unchanged_act_sft_image'
                else:
                    Image.fromarray(rgb).save(rgb_path)
                    rec['rgb_source'] = 'rendered_reachable_extension'
                depth_path = attempt / 'depth.npz'
                np.savez_compressed(depth_path, depth=np.asarray(depth, np.float32), alpha=np.asarray(alpha, np.float32))
                if row['wide']:
                    wk = np.array(row['K'], copy=True)
                    wk[0, 2] += 64
                    wk[1, 2] += 64
                    wrgb, wd, wa = renderer.render(row['pose'], 384, wk)
                    wp = attempt / 'inspection_wide_depth.npz'
                    np.savez_compressed(wp, depth=np.asarray(wd, np.float32), alpha=np.asarray(wa, np.float32))
                    Image.fromarray(np.asarray(wrgb, np.uint8)).save(attempt / 'inspection_wide_rgb.png')
                    rec.update(wide_depth=str(wp), wide_depth_sha256=sha(wp), wide_inspection_only=True)
                rec.update(ok=True, rgb=str(rgb_path), rgb_sha256=sha(rgb_path), depth=str(depth_path),
                           depth_sha256=sha(depth_path), same_camera_comparison=comparison, finished_at=time.time())
                (attempt / 'status.json').write_text(json.dumps(rec, indent=2))
                status.write_text(json.dumps(rec, indent=2))
                records.append(rec)
            except Exception:
                rec.update(error=traceback.format_exc(), finished_at=time.time())
                (attempt / 'status.json').write_text(json.dumps(rec, indent=2))
                failures.append(rec)
            if index % 20 == 0:
                print(json.dumps({'view': index+1, 'total': len(views), 'success': len(records),
                                  'failures': len(failures), 'scene': row['scene']}), flush=True)
        summary = {'run_id': run_id, 'requested_unique_views': len(views), 'successful_unique_views': len(records),
                   'failed_unique_views': len(failures), 'reused_unique_views': reused,
                   'renderer': 'actual_gaussian_rgb_depth_alpha', 'same_camera_checks': sum(r.get('same_camera_comparison') is not None for r in records)}
        (out / ('render_run_' + run_id + '.json')).write_text(json.dumps({'summary': summary, 'failures': failures}, indent=2))
        (out / 'render_manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
        (out / 'render_summary.json').write_text(json.dumps(summary, indent=2))
        print(json.dumps(summary), flush=True)
        if failures:
            raise RuntimeError(f'{len(failures)} actual renders failed; saved separately, not marked successful')
    finally:
        if os.geteuid() == 0:
            for p in out.rglob('*'):
                os.chown(p, owner.st_uid, owner.st_gid)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--gs-root', default='/mnt/umm/users/yinbaiqiao/InteriorGS_ply')
    a = p.parse_args()
    run(a.out, a.gs_root)


if __name__ == '__main__':
    main()
