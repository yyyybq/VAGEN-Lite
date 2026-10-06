"""No success-only selection or silent denominator shrinkage in paired eval."""
import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location('context_pair', Path(__file__).resolve().parents[2] / 'scripts/r1_compare_task_context_eval.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def fixture(tmp_path):
    frozen = tmp_path / 'frozen'
    frozen.mkdir()
    audits = [{'source_key': f't{i}', 'scene_id': f's{i % 7}', 'episode_fingerprint': f'f{i}'} for i in range(32)]
    audit_path = frozen / 'audit.jsonl'
    audit_path.write_text(''.join(json.dumps(row) + '\n' for row in audits))
    (frozen / 'eval_protocol.json').write_text(json.dumps({'audit_manifest': {'path': str(audit_path)}}))
    for model in ('base', 'step1'):
        directory = tmp_path / 'paired_eval' / model
        directory.mkdir(parents=True)
        rows = {str(i): {**a, 'status': 'complete', 'paired_seed': i,
                'success': i == (0 if model == 'base' else 1),
                'initial_rgb_evidence': {'passed': True},
                'turns': [{'input': {'task_context_check': {'task_present': True}}, 'done': True}],
                'final_canonical_metric': {'success': True}, 'invalid_turns': int(i == 2), 'model_turns': 1}
                for i, a in enumerate(audits)}
        (directory / 'episode_ledger.json').write_text(json.dumps({'episodes': rows}))
    return tmp_path


def test_taskwise_pairing_retains_invalid_outputs(tmp_path):
    report = MODULE.compare(fixture(tmp_path))
    assert report['pairs'] == 32 and report['success_rate_delta'] == 0
    assert report['gained_tasks'] == report['lost_tasks'] == 1
    assert report['base_invalid_turn_rate'] == 1 / 32
    assert not report['expanded_training_allowed']


@pytest.mark.parametrize('defect', ['missing', 'infra', 'seed', 'fingerprint', 'context'])
def test_pairing_fails_closed(tmp_path, defect):
    run = fixture(tmp_path)
    path = run / 'paired_eval/step1/episode_ledger.json'
    data = json.loads(path.read_text())
    row = data['episodes']['0']
    if defect == 'missing':
        del data['episodes']['0']
    elif defect == 'infra':
        row['status'] = 'infrastructure_error'
    elif defect == 'seed':
        row['paired_seed'] += 1
    elif defect == 'fingerprint':
        row['episode_fingerprint'] = 'wrong episode'
    else:
        row['turns'][0]['input']['task_context_check']['task_present'] = False
    path.write_text(json.dumps(data))
    with pytest.raises(AssertionError):
        MODULE.compare(run)
