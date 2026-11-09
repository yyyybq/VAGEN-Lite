#!/usr/bin/env bash
set -euo pipefail
umask 002
ROOT=/mnt/umm/users/yinbaiqiao/VAGEN-Lite
RUNTIME_ENV=/mnt/umm/users/yinbaiqiao/.conda/envs/vagen-lite
OUT=${ROOT}/data_gen/active_spatial_qa/artifacts/production_v1_20261012
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
out = Path(os.environ['QA_PRODUCTION_OUTPUT'])
out.mkdir(parents=True, exist_ok=True)
lock = (out / 'worker.lock').open('a')
fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
run_id = 'run_' + str(time.time_ns())
run = out / 'worker_runs' / run_id
run.mkdir(parents=True, exist_ok=False)

def now():
    return datetime.now(timezone.utc).isoformat()

def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)

state = {'ok': False, 'phase': 'starting', 'run_id': run_id, 'started_at': now(),
         'scope': 'single GPU render only; no model inference or training', 'phases': []}
save(run / 'status.json', state)

def phase(name, arguments, expected_rc=0, marker=None):
    info = {'name': name, 'command': arguments, 'started_at': now(), 'expected_exit_code': expected_rc}
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
    info['expected_marker_found'] = marker in log_path.read_text() if marker else None
    save(run / 'status.json', state)
    if code != expected_rc or (marker and not info['expected_marker_found']):
        raise RuntimeError(f'{name} failed validation: exit={code}, expected={expected_rc}, marker={info["expected_marker_found"]}')

try:
    relative_code = [
        'data_gen/active_spatial_qa/production_v1.py',
        'data_gen/active_spatial_qa/qa_production_canary_v1.py',
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
    code_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    save(run / 'code_sha256.json', code_hashes)
    names = ['HOSTNAME', 'JOB_ID', 'SCO_JOB_ID', 'TRAINING_JOB_ID', 'RANK', 'LOCAL_RANK',
             'WORLD_SIZE', 'CUDA_VISIBLE_DEVICES', 'VAGEN_GSPLAT_PREBUILT',
             'TORCH_EXTENSIONS_DIR', 'CUDA_HOME', 'TORCH_CUDA_ARCH_LIST', 'OMP_NUM_THREADS']
    environment = {'started_at': now(), 'python': sys.executable, 'python_version': sys.version,
                   'cwd': str(root), 'environment': {k: os.environ[k] for k in names if k in os.environ}}
    gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,driver_version', '--format=csv'],
                         capture_output=True, text=True, check=True)
    environment['gpu'] = gpu.stdout
    save(run / 'runtime_environment.json', environment)
    phase('canary', [sys.executable, '-m', 'data_gen.active_spatial_qa.qa_production_canary_v1',
                    '--output', str(out), '--run-id', run_id])
    base = [sys.executable, '-m', 'data_gen.active_spatial_qa.production_v1']
    phase('render_injected_failure', base + ['render', '--output', str(out), '--inject-failure-after', '1'],
          expected_rc=86, marker='QA_PRODUCTION_INJECTED_FAILURE')
    phase('render_resume', base + ['render', '--output', str(out)])
    phase('finalize', base + ['finalize', '--output', str(out)])
    final_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    if final_hashes != code_hashes:
        raise RuntimeError('source code changed during the GPU run')
    state.update(ok=True, phase='succeeded')
except BaseException:
    state.update(ok=False, phase='failed', error=traceback.format_exc())
    raise
finally:
    state['finished_at'] = now()
    save(run / 'status.json', state)
    print(json.dumps({'worker_status': str(run / 'status.json'), 'ok': state['ok']}), flush=True)
PY
