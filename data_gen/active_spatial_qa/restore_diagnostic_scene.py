"""Restore only this diagnostic's assets with the established R1 AOSS client."""
import configparser
import hashlib
import json
import re
import subprocess
import os
from pathlib import Path

OUT = Path(os.environ.get('POSITION_DIAG_OUT', str(Path(__file__).resolve().parent / 'artifacts/position_repair_v7_20261003')))
SCENE = '0003_839989'

def _credentials(path):
    parser = configparser.RawConfigParser(); parser.read(path)
    access, secret = parser.get('fj', 'access_key', fallback='').strip(), parser.get('fj', 'secret_key', fallback='').strip()
    if not access or not secret: raise RuntimeError('existing AOSS configuration has no fj credentials')
    return access, secret

def _redact(value, access, secret):
    value = value.replace(access, '<redacted>').replace(secret, '<redacted>')
    return re.sub(r's3://[^/@:\\s]+:[^/@\\s]+@', 's3://<redacted>@', value)

def _sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''): digest.update(chunk)
    return digest.hexdigest()

sha256 = _sha256

def _validate_scene(path):
    missing = [n for n in ('structure.json','labels.json','3dgs_compressed.ply') if not (path/n).is_file()]
    errors=[]
    if not missing:
        structure=json.loads((path/'structure.json').read_text()); labels=json.loads((path/'labels.json').read_text())
        if not structure.get('rooms'): errors.append('structure_has_no_rooms')
        if not labels: errors.append('labels_empty')
        if (path/'3dgs_compressed.ply').read_bytes()[:3] != b'ply': errors.append('invalid_ply_header')
    return {'status':'valid' if not missing and not errors else 'invalid', 'missing':missing, 'errors':errors,
            'files':{n:{'bytes':(path/n).stat().st_size, 'sha256':_sha256(path/n)} for n in ('structure.json','labels.json','3dgs_compressed.ply') if (path/n).is_file()}}

def main():
    dest = OUT / 'assets' / SCENE
    dest.mkdir(parents=True, exist_ok=True)
    access, secret = _credentials(Path('/mnt/umm/users/yinbaiqiao/aoss/petreloss.conf'))
    uri = f's3://{access}:{secret}@baiqiao.aoss-internal.cn-fz-01.fjscmsapi-oss.com/InteriorGS/{SCENE}/'
    # Include scene geometry and any exported camera/transform metadata, no other scenes.
    command = ['/mnt/umm/users/yinbaiqiao/aoss/ads-cli', '--quiet', '--threads', '4',
               '--listers', '1', '--conntimeout', '60', '--timeout', '300', '--include',
               r'(3dgs_compressed\.ply|[^/]*\.json|[^/]*(camera|transform)[^/]*\.(npz|npy|txt))$',
               'sync', uri, str(dest.resolve()) + '/']
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=900)
        (OUT / 'restore.log').write_text(_redact(result.stdout + result.stderr, access, secret))
        if result.returncode:
            raise RuntimeError(f'AOSS restore failed rc={result.returncode}; see sanitized restore.log')
    except subprocess.TimeoutExpired:
        raise RuntimeError('AOSS restore timed out; command credentials suppressed') from None
    validation = _validate_scene(dest)
    for file in dest.rglob('*'):
        if file.is_file():
            file.chmod(0o664)
    record = {'source': f's3://baiqiao/InteriorGS/{SCENE}/',
              'asset_root': str(dest.resolve()),
              'restore_command_sha256': hashlib.sha256(' '.join(command).encode()).hexdigest(),
              'source_evidence': ['scripts/r1_aoss_scene_pipeline.py', 'data_gen/active_spatial_qa/sco_act2qa_real_canary_entry.sh'],
              'validation': validation, 'all_files': {str(p.relative_to(dest)): {'sha256': sha256(p), 'bytes': p.stat().st_size}
                                                     for p in dest.rglob('*') if p.is_file()}}
    (OUT / 'asset_provenance.json').write_text(json.dumps(record, indent=2)+'\n')
    if validation['status'] != 'valid':
        raise RuntimeError('restored scene failed validation')
    print(json.dumps({'restored': SCENE, 'files': list(record['all_files'])}), flush=True)

if __name__ == '__main__': main()
