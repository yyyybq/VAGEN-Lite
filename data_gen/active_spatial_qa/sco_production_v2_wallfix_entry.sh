#!/usr/bin/env bash
set -euo pipefail
umask 002
ROOT=/mnt/umm/users/yinbaiqiao/VAGEN-Lite
RUNTIME_ENV=/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite
OUT=${1:-${ROOT}/data_gen/active_spatial_qa/artifacts/production_v2_wallfix_20261013}
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
if out == previous or previous in out.parents or out.parent != artifacts:
    raise ValueError('new batch must be an independent direct child of the project artifacts directory')
if not out.name.startswith('production_v2_wallfix_'):
    raise ValueError('refusing to render outside a production_v2_wallfix batch')
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

state = {'ok': False, 'phase': 'starting', 'run_id': run_id, 'started_at': now(),
         'scope': 'new wall-checked QA batch; render and finalize only; isolated real resume check; no model inference or training',
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
    if not 2 <= len(rows) <= 60 or len({row['sample_id'] for row in rows}) != len(rows):
        raise ValueError('expected 2 to 60 unique new planned samples')
    input_hashes = {str(path): sha256(path) for path in (plan_path, config_path)}
    save(run / 'input_sha256.json', {'files': input_hashes, 'samples': len(rows),
         'scenes': sorted({row['scene_id'] for row in rows})})
    relative_code = [
        'data_gen/active_spatial_qa/production_v2.py',
        'data_gen/active_spatial_qa/qa_wall_visibility_v2.py',
        'data_gen/active_spatial_qa/qa_production_resume_check_v2.py',
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
    if any(old_code.get(path) != value for path, value in renderer_hashes.items()):
        raise RuntimeError('renderer code changed since the existing canary; review the reference before submission')
    save(run / 'canary_reference.json', {'rerendered': False, 'status': str(canary_path),
         'sha256': canary_sha, 'comparisons': canary['comparisons'],
         'historical_code_record': str(old_code_path), 'historical_code_sha256': sha256(old_code_path),
         'renderer_code_unchanged': renderer_hashes})
    names = ['HOSTNAME', 'JOB_ID', 'SCO_JOB_ID', 'TRAINING_JOB_ID', 'RANK', 'LOCAL_RANK',
             'WORLD_SIZE', 'CUDA_VISIBLE_DEVICES', 'VAGEN_GSPLAT_PREBUILT',
             'TORCH_EXTENSIONS_DIR', 'CUDA_HOME', 'TORCH_CUDA_ARCH_LIST', 'OMP_NUM_THREADS']
    environment = {'started_at': now(), 'python': sys.executable, 'python_version': sys.version,
                   'cwd': str(root), 'environment': {k: os.environ[k] for k in names if k in os.environ}}
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv'],
                         capture_output=True, text=True, check=True)
    environment['gpu'] = gpu.stdout
    save(run / 'runtime_environment.json', environment)
    base = [sys.executable, '-m', 'data_gen.active_spatial_qa.production_v2']
    phase('render_missing', base + ['render', '--output', str(out)])
    phase('finalize', base + ['finalize', '--output', str(out)])
    resume_out = out.parent / (out.name + '_resume_check') / run_id
    phase('isolated_resume_check', [sys.executable, '-m',
          'data_gen.active_spatial_qa.qa_production_resume_check_v2',
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
    state['finished_at'] = now()
    save(run / 'status.json', state)
    print(json.dumps({'worker_status': str(run / 'status.json'), 'ok': state['ok']}), flush=True)
PY
