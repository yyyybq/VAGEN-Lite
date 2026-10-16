#!/usr/bin/env python3
"""Create an immutable evaluation-only overlay on the accepted R1 archive."""
import hashlib
import io
import json
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'exps/vagen_active_spatial/R1-task-context-v1-acceptance-20261003'
BASE = RUN / 'eval_package_v4/source.tar.gz'
OUTPUT = RUN / 'eval_package_v5'
OVERLAYS = (
    'scripts/r1_run_canonical_dev_eval32.py',
    'scripts/r1_compare_task_context_eval.py',
    'scripts/r1_vllm_eval_preflight.py',
    'examples/train/active_spatial/sco_r1_task_context_eval.sh',
)


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


def digest(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def main():
    OUTPUT.mkdir(exist_ok=False)
    with tarfile.open(BASE) as archive:
        files = {m.name: archive.extractfile(m).read() for m in archive.getmembers() if m.isfile()}
    changes = {}
    for name in OVERLAYS:
        value = (ROOT / name).read_bytes()
        if name.endswith('.py'):
            compile(value, name, 'exec')
        changes[name] = {'before': digest_bytes(files[name]) if name in files else None,
                         'after': digest_bytes(value)}
        files[name] = value
    archive_path = OUTPUT / 'source.tar.gz'
    with tarfile.open(archive_path, 'w:gz') as archive:
        for name, value in sorted(files.items()):
            member = tarfile.TarInfo(name)
            member.mode = 0o644
            member.size = len(value)
            archive.addfile(member, io.BytesIO(value))
    owner = RUN / 'package/owner.sh'
    (OUTPUT / 'owner.sh').write_bytes(owner.read_bytes())
    for mode, limit in (('preflight', '2h'), ('renderer', '6h'), ('evaluation', '6h')):
        launch = f'''#!/usr/bin/env bash
set -euo pipefail
PACKAGE={OUTPUT}
RUN={RUN}
if [[ $(id -u) == 0 ]]; then exec bash "${{PACKAGE}}/owner.sh" bash "$0" owner; fi
[[ $(id -u) == 20325 && $(id -g) == 20325 ]] || exit 3
cd "${{PACKAGE}}"; sha256sum -c SHA256SUMS
WORK=$(mktemp -d /tmp/r1_context_eval.XXXXXXXX)
tar -xzf source.tar.gz -C "${{WORK}}"; cd "${{WORK}}"
exec timeout --signal=TERM --kill-after=120s {limit} bash examples/train/active_spatial/sco_r1_task_context_eval.sh {mode} "${{RUN}}"
'''
        (OUTPUT / f'launch_{mode}.sh').write_text(launch)
    provenance = {
        'role': 'evaluation_only', 'base_archive': str(BASE), 'base_sha256': digest(BASE),
        'changes': changes, 'ppo_invoked': False, 'checkpoint_mutated': False,
        'required_acceptance_gate': 'PASS', 'vllm_preflight_models': ['base', 'step1'],
        'eval_tasks': 32, 'paired': True, 'invalid_outputs_remain_in_denominator': True,
        'cuda_contract': 'system toolkit must contain nvcc, curand.h, and libcurand; FlashInfer sampler forced on',
    }
    (OUTPUT / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    (OUTPUT / 'SHA256SUMS').write_text(''.join(
        f'{digest(path)}  {path.name}\n' for path in sorted(OUTPUT.iterdir()) if path.is_file()))
    print(json.dumps({'package': str(OUTPUT), 'archive_sha256': digest(archive_path)}))


if __name__ == '__main__':
    main()
