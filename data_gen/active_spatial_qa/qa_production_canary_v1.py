"""Render-only replay of the pinned wardrobe reference through both entries."""
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


async def replay(dest, status):
    import numpy as np
    from PIL import Image
    from data_gen.active_spatial_pipeline.render_utils import RenderConfig, SceneRenderer
    from data_gen.active_spatial_qa.image_contract_v2 import image_stats
    from data_gen.active_spatial_qa.render_paired_bank import render, render_camera

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
    row = copy.deepcopy(request['rows'][0])
    width, height = int(request['width']), int(request['height'])
    pose = np.asarray(row['state_pose_c2w'], dtype=np.float32)
    intrinsics = np.asarray(row['camera']['intrinsics'], dtype=np.float32)
    qa_k, qa_w2c = render_camera(row, width, height)
    if not np.allclose(qa_k, intrinsics, rtol=0, atol=1e-6):
        raise RuntimeError('QA entry changed the trusted intrinsics')
    if not np.allclose(qa_w2c, np.linalg.inv(pose), rtol=0, atol=1e-6):
        raise RuntimeError('QA entry changed the trusted pose')
    status.update(source_request=str(SOURCE), source_request_sha256=SOURCE_SHA,
                  historical_reference=str(reference), historical_reference_sha256=REFERENCE_SHA,
                  input_hashes=hashes, camera={'K': intrinsics.tolist(), 'c2w': pose.tolist(),
                                               'qa_w2c': qa_w2c.tolist(), 'size': [width, height]})
    save(dest / 'actual_request.json', status)
    upstream = dest / 'A_upstream.png'
    cfg = RenderConfig(gs_root=str(V6 / 'assets'), render_backend='local',
                       image_width=width, image_height=height, gpu_device=0)
    async with SceneRenderer(cfg) as renderer:
        await renderer.set_scene(request['scene_id'])
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
    qa_path = Path(rows[0]['render']['image_path'])
    arrays, artifacts = {}, {}
    for name, path in [('A', upstream), ('B', qa_path), ('reference', reference)]:
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
    if any(sha256(path) != expected for path, expected in hashes.items()):
        raise RuntimeError('historical input changed during replay')
    status['normality_evidence'] = 'Within one RGB level of the previously visually inspected upright wardrobe reference.'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True)
    parser.add_argument('--run-id', default=None)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output != ALLOWED_OUTPUT.resolve():
        raise ValueError('canary output must be the new production_v1_20261012 directory')
    run_id = args.run_id or 'run_' + str(time.time_ns())
    if not run_id.startswith('run_') or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in run_id):
        raise ValueError('invalid run ID')
    dest = output / 'canary' / run_id
    dest.mkdir(parents=True, exist_ok=False)
    status = {'ok': False, 'phase': 'running', 'started_at': now(), 'run_id': run_id,
              'script_sha256': sha256(Path(__file__)), 'scope': 'image replay only; no task labels accepted'}
    save(dest / 'status.json', status)
    try:
        asyncio.run(replay(dest, status))
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
