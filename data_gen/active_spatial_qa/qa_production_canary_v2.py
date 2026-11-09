"""Pinned wardrobe canary with identical effective cameras and verified A reuse.

No task labels are accepted. Old requests, runs and reference images are immutable.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/mnt/umm/users/yinbaiqiao/VAGEN-Lite')
ALLOWED_OUTPUT = ROOT / 'data_gen/active_spatial_qa/artifacts/production_v1_20261012'
V6 = ROOT / 'data_gen/active_spatial_qa/artifacts/position_repair_v6_20261002'
SOURCE = V6 / '01_A_upstream/attempt_1790974624513136041/actual_request.json'
SOURCE_SHA = 'a6e4e715b66f2e638e84b7ba83316ad59fe50ed3edb4928ca65fe565efea777b'
REFERENCE_SHA = 'ef2d5413e3661f9837164c5099bdf3f2b84af548e2c7a35b00c169ee3a22723f'


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def effective_matrix(value):
    """Exact little-endian float32 bytes consumed by the local renderer."""
    import numpy as np
    array = np.ascontiguousarray(value, dtype=np.dtype('<f4'))
    if not np.isfinite(array).all():
        raise ValueError('nonfinite effective camera')
    return {'dtype': 'float32', 'byte_order': 'little', 'order': 'C',
            'shape': list(array.shape), 'values': array.tolist(),
            'sha256': hashlib.sha256(array.tobytes(order='C')).hexdigest()}


def preflight(reuse_a_from=None):
    """Validate only CPU metadata, effective matrices and optional existing A."""
    import numpy as np
    from data_gen.active_spatial_qa.render_paired_bank import render_camera

    if sha256(SOURCE) != SOURCE_SHA:
        raise RuntimeError('pinned historical request changed')
    frozen = json.loads(SOURCE.read_text())
    request = frozen['request']
    if len(request['rows']) != 1 or request['scene_id'] != '0003_839989':
        raise RuntimeError('unexpected historical scene or row count')
    reference = Path(request['reference']['image'])
    if request['reference']['sha256'] != REFERENCE_SHA or sha256(reference) != REFERENCE_SHA:
        raise RuntimeError('pinned inspected reference changed')
    hashes = {}
    for name, expected in frozen['asset_identity'].items():
        path = V6 / 'assets' / request['scene_id'] / name
        actual = {'bytes': path.stat().st_size, 'sha256': sha256(path)}
        if actual != expected:
            raise RuntimeError('historical scene identity mismatch: ' + name)
        hashes[str(path)] = actual['sha256']
    hashes[str(SOURCE)] = SOURCE_SHA
    hashes[str(reference)] = REFERENCE_SHA

    original = copy.deepcopy(request['rows'][0])
    row = copy.deepcopy(original)
    width, height = int(request['width']), int(request['height'])
    # Preserve the camera used by the pinned, visually inspected upstream A.
    pose = np.asarray(original['state_pose_c2w'], dtype=np.float32)
    intrinsics = np.asarray(original['camera']['intrinsics'], dtype=np.float32)
    row['state_pose_c2w'] = pose.tolist()
    row['camera']['intrinsics'] = intrinsics.tolist()
    qa_k, qa_w2c = render_camera(row, width, height)
    upstream_w2c = np.linalg.inv(pose)  # SceneRenderer.render_image's exact path.
    camera_a = {'K': effective_matrix(intrinsics), 'w2c': effective_matrix(upstream_w2c)}
    camera_b = {'K': effective_matrix(qa_k), 'w2c': effective_matrix(qa_w2c)}
    if camera_a != camera_b:
        raise RuntimeError('A/B effective float32 K or w2c differ; rendering is forbidden')
    binding = {'source_request': str(SOURCE), 'source_request_sha256': SOURCE_SHA,
               'historical_reference': str(reference), 'historical_reference_sha256': REFERENCE_SHA,
               'input_hashes': hashes, 'source_original_row': original, 'replay_row': row,
               'camera': {'K': intrinsics.tolist(), 'c2w': pose.tolist(),
                          'c2w_input_dtype': 'float32', 'qa_w2c_before_float32': qa_w2c.tolist(),
                          'size': [width, height], 'effective_A': camera_a, 'effective_B': camera_b,
                          'effective_camera_exact_equal': True},
               'A_reuse': None}
    if reuse_a_from is not None:
        binding['A_reuse'] = validate_reused_a(Path(reuse_a_from), binding)
    return row, pose, intrinsics, binding


def validate_reused_a(previous, expected):
    """A failed overall run may contain A that exactly matches the trusted RGB."""
    import numpy as np
    from PIL import Image

    previous = previous.resolve()
    if previous.parent != (ALLOWED_OUTPUT / 'canary').resolve():
        raise ValueError('reused A must belong to an existing canary run in this production directory')
    request_path, status_path = previous / 'actual_request.json', previous / 'status.json'
    recorded = json.loads(request_path.read_text())
    status = json.loads(status_path.read_text())
    if recorded.get('run_id') != previous.name or status.get('run_id') != previous.name:
        raise RuntimeError('old canary run identity mismatch')
    if not recorded.get('script_sha256') or recorded['script_sha256'] != status.get('script_sha256'):
        raise RuntimeError('old request/status script identity mismatch')
    for key in ['source_request', 'source_request_sha256', 'historical_reference', 'historical_reference_sha256', 'input_hashes']:
        if recorded.get(key) != expected[key] or status.get(key) != expected[key]:
            raise RuntimeError('old A input binding mismatch: ' + key)
    for old in [recorded, status]:
        camera = old.get('camera', {})
        if camera.get('size') != expected['camera']['size']:
            raise RuntimeError('old A camera size mismatch')
        for key in ['K', 'c2w']:
            # Compare stored numbers without rounding away a changed request.
            if not np.array_equal(np.asarray(camera.get(key), dtype=np.float64),
                                  np.asarray(expected['camera'][key], dtype=np.float64)):
                raise RuntimeError('old A camera mismatch: ' + key)
    artifact = status.get('artifacts', {}).get('A', {})
    image_path = Path(artifact.get('path', '')).resolve()
    if image_path != (previous / 'A_upstream.png').resolve():
        raise RuntimeError('old A artifact path does not match its run')
    actual_sha = sha256(image_path)
    if actual_sha != artifact.get('sha256') or actual_sha != REFERENCE_SHA:
        raise RuntimeError('reused A must exactly match both its recorded SHA and the trusted reference')
    with Image.open(image_path) as image:
        image.load()
        if image.size != tuple(expected['camera']['size']):
            raise RuntimeError('reused A image dimensions mismatch')
    return {'source_run': str(previous), 'source_request': str(request_path),
            'source_request_sha256': sha256(request_path), 'source_status': str(status_path),
            'source_status_sha256': sha256(status_path), 'source_run_phase': status.get('phase'),
            'image_path': str(image_path), 'image_sha256': actual_sha,
            'input_hashes_match': True, 'camera_exact_equal': True,
            'equals_pinned_reference_sha256': True, 'rendered_again': False}


async def replay(dest, status, prepared):
    import numpy as np
    from PIL import Image
    from data_gen.active_spatial_pipeline.render_utils import RenderConfig, SceneRenderer
    from data_gen.active_spatial_qa.image_contract_v2 import image_stats
    from data_gen.active_spatial_qa.render_paired_bank import render

    row, pose, intrinsics, binding = prepared
    status.update(binding)
    save(dest / 'actual_request.json', status)
    width, height = binding['camera']['size']
    if binding['A_reuse']:
        upstream = Path(binding['A_reuse']['image_path'])
    else:
        upstream = dest / 'A_upstream.png'
        cfg = RenderConfig(gs_root=str(V6 / 'assets'), render_backend='local',
                           image_width=width, image_height=height, gpu_device=0)
        async with SceneRenderer(cfg) as renderer:
            await renderer.set_scene(row['scene_id'])
            image = await renderer.render_image(intrinsics, pose)
            if image is None:
                raise RuntimeError('upstream returned no image')
            image.save(upstream)
    bank = dest / 'qa_input.jsonl'
    bank.write_text(json.dumps(row, allow_nan=False) + '\n')
    qa_out = dest / 'qa'
    summary = await render(argparse.Namespace(bank=str(bank), output_dir=str(qa_out),
        limit=0, backend='local', gs_root=str(V6 / 'assets'), renderer_url='',
        gpu_device=0, width=width, height=height))
    if summary['errors'] != 0 or summary['rendered'] != 1 or summary['candidates'] != 1:
        raise RuntimeError('QA replay did not produce exactly one successful image')
    rows = [json.loads(line) for line in (qa_out / 'manifest.jsonl').read_text().splitlines() if line]
    if len(rows) != 1 or rows[0]['sample_id'] != row['sample_id']:
        raise RuntimeError('QA replay sample identity mismatch')
    actual = rows[0]['render']['request']
    actual_camera = {'K': effective_matrix(actual['intrinsics_K']), 'w2c': effective_matrix(actual['pose_w2c'])}
    if actual_camera != binding['camera']['effective_A'] or [actual['width'], actual['height']] != [width, height]:
        raise RuntimeError('actual QA render request changed the prechecked camera')
    status['actual_qa_effective_camera'] = actual_camera
    qa_path = Path(rows[0]['render']['image_path'])
    arrays, artifacts = {}, {}
    for name, path in [('A', upstream), ('B', qa_path), ('reference', Path(binding['historical_reference']))]:
        with Image.open(path) as image:
            image.load()
            if image.size != (width, height):
                raise RuntimeError('unexpected image size: ' + name)
            image = image.convert('RGB')
            stats = image_stats(image)
            if not stats['nonblank']:
                raise RuntimeError('low-information image: ' + name)
            arrays[name] = np.asarray(image, dtype=np.int16)
            artifacts[name] = {'path': str(path), 'sha256': sha256(path), 'rgb_stats': stats}
    comparisons = {}
    for lhs, rhs in [('A', 'B'), ('A', 'reference'), ('B', 'reference')]:
        delta = np.abs(arrays[lhs] - arrays[rhs])
        comparisons[lhs + '_' + rhs] = {'max_abs': int(delta.max()), 'mean_abs': float(delta.mean())}
    status.update(artifacts=artifacts, comparisons=comparisons)
    if any(value['max_abs'] > 1 for value in comparisons.values()):
        raise RuntimeError('A/B/historical reference pixels differ by more than one')
    if any(sha256(path) != expected for path, expected in binding['input_hashes'].items()):
        raise RuntimeError('historical input changed during replay')
    if binding['A_reuse']:
        fresh = validate_reused_a(Path(binding['A_reuse']['source_run']), binding)
        if fresh != binding['A_reuse']:
            raise RuntimeError('reused A or provenance changed during replay')
    status['normality_evidence'] = 'Identical effective float32 cameras; within one RGB level of the pinned, visually inspected upright wardrobe reference.'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--run-id', default=None)
    parser.add_argument('--reuse-upstream-from', '--reuse-a-from', dest='reuse_a_from', default=None,
                        help='Existing canary run whose A equals the trusted reference exactly.')
    parser.add_argument('--check-only', action='store_true', help='CPU preflight only; no new run or render.')
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output != ALLOWED_OUTPUT.resolve():
        raise ValueError('canary output must be the new production_v1_20261012 directory')
    if args.check_only:
        prepared = preflight(args.reuse_a_from)
        print(json.dumps({'cpu_preflight_passed': True, 'render_performed': False,
                          'canary_verified': False, 'binding': prepared[3]}, allow_nan=False), flush=True)
        return
    run_id = args.run_id or 'run_' + str(time.time_ns())
    if not run_id.startswith('run_') or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in run_id):
        raise ValueError('invalid run ID')
    dest = output / 'canary' / run_id
    dest.mkdir(parents=True, exist_ok=False)
    status = {'ok': False, 'phase': 'running', 'started_at': now(), 'run_id': run_id,
              'script_sha256': sha256(Path(__file__)), 'scope': 'image replay only; no task labels accepted',
              'version': 'qa_production_canary_v2'}
    save(dest / 'status.json', status)
    try:
        prepared = preflight(args.reuse_a_from)
        asyncio.run(replay(dest, status, prepared))
        status.update(ok=True, phase='succeeded')
    except BaseException:
        status.update(ok=False, phase='failed', error=traceback.format_exc())
        raise
    finally:
        status['finished_at'] = now()
        save(dest / 'status.json', status)
    print(json.dumps({'canary_passed': True, 'status': str(dest / 'status.json')}), flush=True)


if __name__ == '__main__':
    main()
