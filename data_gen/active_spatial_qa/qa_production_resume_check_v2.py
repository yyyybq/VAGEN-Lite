"""Run a real two-image failure/resume check in a fresh, isolated directory.

The source batch, its code, and its successful images are read-only. This
helper never finalizes a dataset or invokes the formal worker's injection.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path

MARKER = 'QA_ISOLATED_SECOND_RENDER_FAILURE_V2'


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1048576), b''):
            value.update(block)
    return value.hexdigest()


def strict_loads(text):
    def reject(token):
        raise ValueError('non-finite JSON value: ' + token)
    return json.loads(text, parse_constant=reject)


def strict_read(path):
    return strict_loads(Path(path).read_text())


def save(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def file_identity(path):
    path = Path(path)
    return {'path': str(path.resolve()), 'sha256': sha256(path),
            'bytes': path.stat().st_size, 'mtime_ns': path.stat().st_mtime_ns}


def attempt_records(output, sample_id):
    return sorted((output / 'renders' / sample_id).glob('attempt_*/status.json'))


def successful_image(status_path):
    from PIL import Image
    record = strict_read(status_path)
    if record.get('ok') is not True:
        raise AssertionError('expected a successful attempt: ' + str(status_path))
    path = Path(record['image_path'])
    actual = file_identity(path)
    if actual['sha256'] != record['sha256']:
        raise AssertionError('image does not match the recorded hash')
    with Image.open(path) as image:
        image.load()
        if image.size != (256, 256):
            raise AssertionError('unexpected rendered image size')
        actual['decoded_size'] = list(image.size)
    actual['attempt_status'] = file_identity(status_path)
    return actual


def one_new_run(output, previous):
    paths = set((output / 'worker_runs').glob('*.json')) - previous
    if len(paths) != 1:
        raise AssertionError('render_run must write exactly one new worker record')
    path = paths.pop()
    return path, strict_read(path)


async def check(output, rows, evidence):
    import torch
    from data_gen.active_spatial_qa import production_v2
    from vagen.envs.active_spatial.render.unified_renderer import UnifiedRenderGS

    if not torch.cuda.is_available():
        raise RuntimeError('this check requires a real CUDA renderer')
    evidence['gpu'] = {'name': torch.cuda.get_device_name(0),
                       'device_index': 0, 'torch_version': str(torch.__version__)}
    paths = [Path(production_v2.__file__), Path(__file__),
             Path(production_v2.__file__).with_name('render_paired_bank.py'),
             Path(production_v2.__file__).with_name('production_contract_v1.py'),
             Path(production_v2.__file__).with_name('qa_wall_visibility_v2.py'),
             Path(production_v2.__file__).with_name('production_v1.py')]
    evidence['code_sha256'] = {str(path.resolve()): sha256(path) for path in paths}
    first_id, second_id = [str(row['sample_id']) for row in rows]
    args = argparse.Namespace(output=str(output), inject_failure_after=0)
    original = UnifiedRenderGS.render_image_from_cam_param
    calls = []

    async def fail_second_call(renderer, *parameters, **keywords):
        calls.append({'ordinal': len(calls) + 1, 'called_at': now()})
        if len(calls) == 2:
            calls[-1]['injected'] = True
            raise production_v2.InjectedFailure(MARKER)
        image = await original(renderer, *parameters, **keywords)
        calls[-1]['real_render_returned'] = True
        return image

    before_runs = set((output / 'worker_runs').glob('*.json'))
    expected_failure = False
    evidence['phase'] = 'injected_render'
    save(output / 'status.json', evidence)
    UnifiedRenderGS.render_image_from_cam_param = fail_second_call
    try:
        await production_v2.render_run(args)
    except production_v2.InjectedFailure as error:
        if str(error) != MARKER:
            raise
        expected_failure = True
        evidence['caught_injected_failure'] = {'type': type(error).__name__, 'message': str(error)}
    finally:
        UnifiedRenderGS.render_image_from_cam_param = original
        evidence['render_method_restored'] = UnifiedRenderGS.render_image_from_cam_param is original
        evidence['first_run_renderer_calls'] = calls
        save(output / 'status.json', evidence)
    if not expected_failure or len(calls) != 2 or not calls[0].get('real_render_returned'):
        raise AssertionError('did not complete one real render before the requested second-call failure')
    failed_run_path, failed_run = one_new_run(output, before_runs)
    if failed_run.get('ok') is not False or failed_run.get('phase') != 'failed':
        raise AssertionError('injected worker run was incorrectly marked successful')
    if failed_run.get('rendered') != [first_id] or failed_run.get('reused') != [] or MARKER not in failed_run.get('error', ''):
        raise AssertionError('failed worker record does not describe the controlled first/second attempt')
    first_attempts = attempt_records(output, first_id)
    second_attempts = attempt_records(output, second_id)
    if len(first_attempts) != 1 or len(second_attempts) != 1:
        raise AssertionError('first run must create exactly one attempt for each selected sample')
    first_before = successful_image(first_attempts[0])
    second_failure = strict_read(second_attempts[0])
    if second_failure.get('ok') is not False or MARKER not in second_failure.get('error', ''):
        raise AssertionError('second sample must retain an explicit failed attempt')
    if (second_attempts[0].parent / 'rgb.png').exists() or second_failure.get('image_path'):
        raise AssertionError('the deliberately failed second render unexpectedly has a successful image')
    failed_attempt_before = file_identity(second_attempts[0])
    failed_run_before = file_identity(failed_run_path)
    evidence['failure_evidence'] = {'worker_run': failed_run_before, 'worker_state': failed_run,
        'first_image_before_resume': first_before, 'first_attempt_paths': [str(p) for p in first_attempts],
        'second_failed_attempt': failed_attempt_before, 'second_failed_state': second_failure}
    evidence['phase'] = 'resume_render'
    save(output / 'status.json', evidence)

    before_resume_runs = set((output / 'worker_runs').glob('*.json'))
    await production_v2.render_run(args)
    success_run_path, success_run = one_new_run(output, before_resume_runs)
    if success_run.get('ok') is not True or success_run.get('phase') != 'succeeded':
        raise AssertionError('resume did not succeed')
    if success_run.get('reused') != [first_id] or success_run.get('rendered') != [second_id]:
        raise AssertionError('resume did not reuse only the first sample and render only the missing second sample')
    if success_run['started'] <= failed_run['finished']:
        raise AssertionError('resume worker timestamps do not follow the failed run')
    first_after_paths = attempt_records(output, first_id)
    if first_after_paths != first_attempts:
        raise AssertionError('resume added or removed a first-sample attempt')
    first_after = successful_image(first_after_paths[0])
    if first_after != first_before:
        raise AssertionError('reused image path, hash, mtime, size, or attempt record changed')
    second_after_paths = attempt_records(output, second_id)
    second_new_paths = set(second_after_paths) - set(second_attempts)
    if len(second_after_paths) != 2 or len(second_new_paths) != 1:
        raise AssertionError('resume must create exactly one new attempt for the second sample')
    second_success_path = second_new_paths.pop()
    second_success = successful_image(second_success_path)
    if file_identity(second_attempts[0]) != failed_attempt_before or file_identity(failed_run_path) != failed_run_before:
        raise AssertionError('resume modified the preserved failed attempt or failed worker record')
    if any(sha256(path) != value for path, value in evidence['code_sha256'].items()):
        raise AssertionError('code changed during the isolated check')
    evidence['resume_evidence'] = {'worker_run': file_identity(success_run_path), 'worker_state': success_run,
        'first_image_after_resume': first_after, 'second_success_image': second_success,
        'second_attempt_paths': [str(p) for p in second_after_paths]}
    evidence['checks'] = {'first_image_was_really_rendered': True, 'failed_worker_not_successful': True,
        'second_attempt_failed_explicitly': True, 'renderer_restored_before_resume': evidence['render_method_restored'],
        'first_path_sha_mtime_and_attempts_unchanged': True, 'first_reported_reused': True,
        'second_rendered_in_one_new_successful_attempt': True, 'both_success_files_decode_256x256': True,
        'failed_attempt_and_worker_record_preserved': True, 'code_hashes_unchanged': True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source-output', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    source = Path(args.source_output).resolve(strict=True)
    output = Path(args.output).resolve()
    if output == source or source in output.parents or output in source.parents:
        raise ValueError('the check output must be separate from the formal batch directory')
    plan = source / 'planned_states.jsonl'
    config = source / 'distribution_config.json'
    source_identity = {str(path): file_identity(path) for path in (plan, config)}
    records = [strict_loads(line) for line in plan.read_text().splitlines() if line.strip()]
    grouped = {}
    for row in records:
        grouped.setdefault(str(row['scene_id']), []).append(row)
    rows = next((values[:2] for values in grouped.values() if len(values) >= 2), None)
    if rows is None or len({str(row['sample_id']) for row in rows}) != 2:
        raise ValueError('source plan must contain two distinct samples from the same scene')
    for row in rows:
        sample_id = str(row['sample_id'])
        if not sample_id or Path(sample_id).name != sample_id or sample_id in ('.', '..'):
            raise ValueError('unsafe sample ID')
    output.mkdir(parents=True, exist_ok=False)
    evidence = {'schema_version': 'qa_isolated_real_render_resume_check_v2', 'ok': False,
        'phase': 'preparing', 'started_at': now(), 'source_output': str(source), 'output': str(output),
        'source_inputs': source_identity, 'selected_sample_ids': [row['sample_id'] for row in rows],
        'scene_id': rows[0]['scene_id'], 'formal_batch_render_or_injection_called': False,
        'failure_injection_scope': 'only this isolated process, second renderer call; production inject_failure_after=0'}
    save(output / 'status.json', evidence)
    try:
        (output / 'distribution_config.json').write_bytes(config.read_bytes())
        (output / 'planned_states.jsonl').write_text(''.join(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n' for row in rows))
        if any(file_identity(path) != identity for path, identity in source_identity.items()):
            raise AssertionError('formal source plan or configuration changed while preparing the check')
        asyncio.run(check(output, rows, evidence))
        if any(file_identity(path) != identity for path, identity in source_identity.items()):
            raise AssertionError('formal source plan or configuration changed during the isolated check')
        evidence['checks']['formal_source_plan_and_config_unchanged'] = True
        evidence.update(ok=True, phase='succeeded')
    except BaseException:
        evidence.update(ok=False, phase='failed', error=traceback.format_exc())
        raise
    finally:
        evidence['finished_at'] = now()
        save(output / 'status.json', evidence)
    print(json.dumps({'isolated_resume_check_passed': True, 'status': str(output / 'status.json')}), flush=True)


if __name__ == '__main__':
    main()
