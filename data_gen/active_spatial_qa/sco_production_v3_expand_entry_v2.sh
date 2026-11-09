#!/usr/bin/env bash
set -euo pipefail
umask 002
ROOT=/mnt/umm/users/yinbaiqiao/VAGEN-Lite
RUNTIME_ENV=/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite
OUT=${1:-${ROOT}/data_gen/active_spatial_qa/artifacts/production_v3_expand_20261014}
export QA_PRODUCTION_OUTPUT="${OUT}"
export QA_WORKER_ENTRY="$(readlink -f "${BASH_SOURCE[0]}")"
export PYTHONPATH="${ROOT}:${PYTHONPATH:-}"
export PATH="${RUNTIME_ENV}/bin:${PATH}"
export CUDA_VISIBLE_DEVICES=0
export VAGEN_GSPLAT_PREBUILT=1
export TORCH_EXTENSIONS_DIR=/mnt/umm/users/yinbaiqiao/.cache/torch_extensions_jumpbox_renderer
export CC=${RUNTIME_ENV}/bin/x86_64-conda-linux-gnu-gcc
export CXX=${RUNTIME_ENV}/bin/x86_64-conda-linux-gnu-g++
export CUDA_HOME=${RUNTIME_ENV}
export TORCH_CUDA_ARCH_LIST=9.0
export CPATH="${RUNTIME_ENV}/targets/x86_64-linux/include:${CPATH:-}"
export LIBRARY_PATH="${RUNTIME_ENV}/targets/x86_64-linux/lib:${LIBRARY_PATH:-}"
export OMP_NUM_THREADS=1
export PYTHONUNBUFFERED=1
cd "${ROOT}"
python - <<'PY'
from datetime import datetime, timezone
import fcntl
import ast
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

root = Path.cwd()
artifacts = root / 'data_gen/active_spatial_qa/artifacts'
out = Path(os.environ['QA_PRODUCTION_OUTPUT']).resolve()
previous = artifacts / 'production_v1_20261012'
if out == previous or previous in out.parents or out.parent != artifacts.resolve():
    raise ValueError('new batch must be an independent direct child of the project artifacts directory')
if not out.name.startswith('production_v3_expand_'):
    raise ValueError('refusing to render outside a production_v3_expand batch')
if not (out / 'planned_states.jsonl').is_file() or not (out / 'distribution_config.json').is_file():
    raise ValueError('prepare and inspect the new plan and configuration before submitting this worker')
lock = (out / 'worker.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
run_id = 'run_' + str(time.time_ns())
run = out / 'worker_runs' / run_id
run.mkdir(parents=True, exist_ok=False)

def now():
    return datetime.now(timezone.utc).isoformat()

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)

def strict_loads(text):
    def reject(token):
        raise ValueError('non-finite JSON: ' + token)
    return json.loads(text, parse_constant=reject)

def production_counts():
    counts = {'checked_at': now(), 'counts_are_not_image_review_or_training_approval': True}
    for filename in ('planned_states.jsonl', 'sample_manifest.jsonl', 'failures.jsonl', 'all_failures.jsonl'):
        path = out / filename
        if not path.is_file():
            counts[filename] = {'exists': False}
            continue
        records = [strict_loads(line) for line in path.read_text().splitlines() if line.strip()]
        info = {'exists': True, 'records': len(records), 'sha256': sha256(path)}
        if filename.endswith('failures.jsonl'):
            info['by_reason'] = dict(Counter(str(row.get('reason', 'unspecified')) for row in records))
        if filename == 'sample_manifest.jsonl':
            info['unique_sample_ids'] = len({str(row['sample_id']) for row in records})
            info['by_scene'] = dict(Counter(str(row['scene_id']) for row in records))
            info['by_task'] = dict(Counter(str(row['task_type']) for row in records))
            info['by_state_predicate_answer'] = dict(Counter(str(row.get('automatic_state_predicate_answer')) for row in records))
            info['qa_sft_allowed'] = sum(row.get('qa_sft_allowed') is True for row in records)
            identities = []
            for row in records:
                render = row.get('render', {})
                image_path = Path(render.get('image_path', ''))
                exists = image_path.is_file()
                actual_sha = sha256(image_path) if exists else None
                identities.append({'sample_id': str(row['sample_id']), 'image_path': str(image_path),
                    'exists': exists, 'sha256': actual_sha,
                    'matches_manifest_sha256': exists and actual_sha == render.get('sha256')})
            info['actual_images'] = identities
            info['unique_image_sha256'] = len({image['sha256'] for image in identities if image['exists']})
            info['all_manifest_image_hashes_match'] = all(image['matches_manifest_sha256'] for image in identities)
        counts[filename] = info
    attempts = []
    for path in sorted((out / 'renders').glob('*/attempt_*/status.json')):
        attempt = strict_loads(path.read_text())
        attempts.append({'sample_id': attempt.get('sample_id'), 'ok': attempt.get('ok'),
                         'status': str(path), 'image_path': attempt.get('image_path')})
    counts['render_attempts'] = {'total': len(attempts),
        'succeeded': sum(attempt['ok'] is True for attempt in attempts),
        'failed': sum(attempt['ok'] is False for attempt in attempts),
        'successful_sample_ids': sorted({str(attempt['sample_id']) for attempt in attempts if attempt['ok'] is True})}
    counts['actual_rgb_files'] = len(list((out / 'renders').glob('*/attempt_*/rgb.png')))
    render_runs = []
    for path in sorted((out / 'worker_runs').glob('*.json')):
        record = strict_loads(path.read_text())
        render_runs.append({'path': str(path), 'ok': record.get('ok'), 'phase': record.get('phase'),
            'rendered': len(record.get('rendered', [])), 'reused': len(record.get('reused', []))})
    counts['formal_render_runs'] = render_runs
    return counts

state = {'ok': False, 'phase': 'starting', 'run_id': run_id, 'started_at': now(),
         'scope': 'new v3 QA batch; render and finalize only; isolated real resume check; no model inference or training',
         'phases': []}
save(run / 'status.json', state)

def phase(name, arguments):
    info = {'name': name, 'command': arguments, 'started_at': now(), 'expected_exit_code': 0}
    state['phase'] = name
    state['phases'].append(info)
    save(run / 'status.json', state)
    log_path = run / (name + '.log')
    with log_path.open('w') as log:
        process = subprocess.Popen(arguments, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, bufsize=1)
        for line in process.stdout:
            log.write(line)
            log.flush()
            print(line, end='', flush=True)
        code = process.wait()
    info.update(exit_code=code, finished_at=now(), log=str(log_path))
    save(run / 'status.json', state)
    if code != 0:
        raise RuntimeError(f'{name} failed: exit={code}')

try:
    plan_path = out / 'planned_states.jsonl'
    config_path = out / 'distribution_config.json'
    rows = [strict_loads(line) for line in plan_path.read_text().splitlines() if line.strip()]
    strict_loads(config_path.read_text())
    if not 2 <= len(rows) <= 500 or len({row['sample_id'] for row in rows}) != len(rows):
        raise ValueError('expected 2 to 500 unique new planned samples')
    for row in rows:
        sample_id = str(row['sample_id'])
        if not sample_id or Path(sample_id).name != sample_id or sample_id in ('.', '..'):
            raise ValueError('unsafe sample ID')
    camera_keys = {json.dumps([row['scene_id'], row['state_pose_c2w']], sort_keys=True, allow_nan=False) for row in rows}
    if len(camera_keys) != len(rows):
        raise ValueError('planned samples reuse an identical scene and camera pose')
    input_hashes = {str(path): sha256(path) for path in (plan_path, config_path)}
    save(run / 'input_sha256.json', {'files': input_hashes, 'samples': len(rows),
         'scenes': sorted({row['scene_id'] for row in rows})})
    relative_code = [
        'data_gen/active_spatial_qa/production_v3.py',
        'data_gen/active_spatial_qa/qa_derived_goals_v3.py',
        'data_gen/active_spatial_qa/production_v2.py',
        'data_gen/active_spatial_qa/qa_wall_visibility_v2.py',
        'data_gen/active_spatial_qa/qa_production_resume_check_v3.py',
        'data_gen/active_spatial_qa/production_v1.py',
        'data_gen/active_spatial_qa/production_contract_v1.py',
        'data_gen/active_spatial_qa/render_paired_bank.py',
        'data_gen/active_spatial_qa/generate_paired_qa.py',
        'data_gen/active_spatial_qa/image_contract_v2.py',
        'data_gen/active_spatial_qa/contract.py',
        'data_gen/active_spatial_pipeline/render_utils.py',
        'vagen/envs/active_spatial/env.py',
        'vagen/envs/active_spatial/render/unified_renderer.py',
        'vagen/envs/active_spatial/render/gs_render_local.py',
    ]
    paths = [root / p for p in relative_code] + [Path(os.environ['QA_WORKER_ENTRY'])]
    code_hashes = {str(path): sha256(path) for path in paths}
    save(run / 'code_sha256.json', code_hashes)
    canary_path = previous / 'canary/run_1791882269119395269/status.json'
    canary = strict_loads(canary_path.read_text())
    canary_sha = sha256(canary_path)
    if canary_sha != 'f50abde7cce156c6a2ac73e6f77f5997f915a811efdf2c26ece5a41d3d102310':
        raise RuntimeError('the confirmed historical canary record changed')
    if not canary.get('ok') or any(v['max_abs'] > 1 for v in canary['comparisons'].values()):
        raise RuntimeError('the referenced historical canary did not pass')
    old_code_path = previous / 'worker_runs/run_1791882269119395269/code_sha256.json'
    old_code = strict_loads(old_code_path.read_text())
    renderer_files = [
        'data_gen/active_spatial_qa/render_paired_bank.py',
        'data_gen/active_spatial_pipeline/render_utils.py',
        'vagen/envs/active_spatial/render/unified_renderer.py',
        'vagen/envs/active_spatial/render/gs_render_local.py',
    ]
    renderer_hashes = {str(root / p): sha256(root / p) for p in renderer_files}
    # Reviewed, exact one-file additive delta; all other renderer drift is rejected.
    reviewed_relative = 'vagen/envs/active_spatial/render/gs_render_local.py'
    reviewed_path = root / reviewed_relative
    expected_old = 'eed275dcaa3e31f2622e0ea434cb5a774dc8523f25f70d95cb80afa79bca6bc7'
    expected_new = 'cb5646f8c49a24eecbd3797a783233391e342a2c9800ae079ac3293b45d545dd'
    old_byte_count = 11423
    changed = {path: value for path, value in renderer_hashes.items()
               if old_code.get(path) != value}
    if changed:
        if changed != {str(reviewed_path): expected_new} or old_code.get(str(reviewed_path)) != expected_old:
            raise RuntimeError('renderer drift is not the exact reviewed additive depth-method delta')
        current_bytes = reviewed_path.read_bytes()
        historical_bytes = current_bytes[:old_byte_count]
        if hashlib.sha256(historical_bytes).hexdigest() != expected_old:
            raise RuntimeError('the current renderer no longer preserves the complete historical RGB source prefix')
        historical_tree = ast.parse(historical_bytes)
        current_tree = ast.parse(current_bytes)
        old_class = next(node for node in historical_tree.body
                         if isinstance(node, ast.ClassDef) and node.name == 'GaussianRenderer')
        current_class = next(node for node in current_tree.body
                             if isinstance(node, ast.ClassDef) and node.name == 'GaussianRenderer')
        if len(current_class.body) != len(old_class.body) + 1:
            raise RuntimeError('reviewed renderer does not add exactly one class member')
        added = current_class.body[-1]
        if not isinstance(added, ast.FunctionDef) or added.name != 'render_depth_from_cam_param':
            raise RuntimeError('reviewed renderer added an unexpected class member')
        current_class.body.pop()
        if ast.dump(current_tree, include_attributes=False) != ast.dump(historical_tree, include_attributes=False):
            raise RuntimeError('original RGB renderer AST changed')
        callsites = []
        checked_callsites = [
            'data_gen/active_spatial_qa/production_v3.py',
            'data_gen/active_spatial_qa/render_paired_bank.py',
            'data_gen/active_spatial_pipeline/render_utils.py',
            'vagen/envs/active_spatial/render/unified_renderer.py',
        ]
        for name in checked_callsites:
            tree = ast.parse((root / name).read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and (
                    isinstance(node.func, ast.Attribute) and node.func.attr == added.name
                    or isinstance(node.func, ast.Name) and node.func.id == added.name):
                    callsites.append({'file': name, 'line': node.lineno})
        if callsites:
            raise RuntimeError('QA RGB call chain invokes the newly added depth method')
        save(run / 'renderer_change_review.json', {
            'reviewed_commit': 'a103c48567e8b205905edd29bcc5205b2fd108a1',
            'relative_file': reviewed_relative, 'historical_sha256': expected_old,
            'current_sha256': expected_new, 'historical_prefix_bytes': old_byte_count,
            'historical_full_source_prefix_preserved': True,
            'only_added_ast_method': added.name, 'original_module_ast_unchanged': True,
            'checked_qa_callsite_files': checked_callsites, 'depth_method_call_sites': callsites,
            'historical_canary_reused': True, 'new_gpu_canary_performed': False,
            'scope': 'Exact static additive-method review; original RGB path source unchanged. No claim of a new GPU A/B comparison.',
            'superseded_unrun_entry': str(root / 'data_gen/active_spatial_qa/sco_production_v3_entry.sh'),
            'superseded_unrun_entry_sha256': '30b3753d601ec30f0d500c0377b30182ba55e70f6986b8aa0ba4c6bd2bd1a588'})
    save(run / 'canary_reference.json', {'rerendered': False, 'status': str(canary_path),
         'sha256': canary_sha, 'comparisons': canary['comparisons'],
         'historical_code_record': str(old_code_path), 'historical_code_sha256': sha256(old_code_path),
         'renderer_code_sha256': renderer_hashes,
         'renderer_code_unchanged': not bool(changed),
         'reviewed_additive_delta': str(run / 'renderer_change_review.json') if changed else None})
    names = ['HOSTNAME', 'JOB_ID', 'SCO_JOB_ID', 'TRAINING_JOB_ID', 'RANK', 'LOCAL_RANK',
             'WORLD_SIZE', 'CUDA_VISIBLE_DEVICES', 'VAGEN_GSPLAT_PREBUILT',
             'TORCH_EXTENSIONS_DIR', 'CUDA_HOME', 'TORCH_CUDA_ARCH_LIST', 'OMP_NUM_THREADS']
    environment = {'started_at': now(), 'python': sys.executable, 'python_version': sys.version,
                   'cwd': str(root), 'environment': {k: os.environ[k] for k in names if k in os.environ}}
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv'],
                         capture_output=True, text=True, check=True)
    environment['gpu'] = gpu.stdout
    save(run / 'runtime_environment.json', environment)
    base = [sys.executable, '-m', 'data_gen.active_spatial_qa.production_v3']
    phase('render_missing', base + ['render', '--output', str(out)])
    phase('finalize', base + ['finalize', '--output', str(out)])
    counts = production_counts()
    save(run / 'production_counts.json', counts)
    state['production_counts'] = str(run / 'production_counts.json')
    if not counts['sample_manifest.jsonl']['exists'] or not counts['all_failures.jsonl']['exists']:
        raise RuntimeError('finalize did not produce the manifest and failure records')
    if not counts['sample_manifest.jsonl']['all_manifest_image_hashes_match']:
        raise RuntimeError('a finalized image is missing or does not match its recorded hash')
    resume_out = out.parent / (out.name + '_resume_check') / run_id
    phase('isolated_resume_check', [sys.executable, '-m',
          'data_gen.active_spatial_qa.qa_production_resume_check_v3',
          '--source-output', str(out), '--output', str(resume_out)])
    state['independent_resume_evidence'] = str(resume_out / 'status.json')
    if {str(path): sha256(path) for path in paths} != code_hashes:
        raise RuntimeError('source code changed during the GPU run')
    if any(sha256(path) != value for path, value in input_hashes.items()):
        raise RuntimeError('formal plan or configuration changed during the GPU run')
    state.update(ok=True, phase='succeeded')
except BaseException:
    state.update(ok=False, phase='failed', error=traceback.format_exc())
    raise
finally:
    try:
        save(run / 'production_counts.json', production_counts())
        state['production_counts'] = str(run / 'production_counts.json')
    except BaseException:
        state['production_counts_error'] = traceback.format_exc()
        state.update(ok=False, phase='failed')
        raise
    finally:
        state['finished_at'] = now()
        save(run / 'status.json', state)
        print(json.dumps({'worker_status': str(run / 'status.json'), 'ok': state['ok']}), flush=True)
PY
