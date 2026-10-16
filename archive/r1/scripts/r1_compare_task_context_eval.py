#!/usr/bin/env python3
"""Pair complete fixed-task results; never exclude invalid outputs or errors."""
import argparse
import json
import time
from pathlib import Path


def compare(run, eval_root='paired_eval'):
    protocol = json.loads((run / 'frozen/eval_protocol.json').read_text())
    audits = [json.loads(line) for line in Path(protocol['audit_manifest']['path']).read_text().splitlines() if line.strip()]
    assert len(audits) == 32
    results = {}
    for model in ('base', 'step1'):
        ledger = json.loads((run / eval_root / model / 'episode_ledger.json').read_text())
        rows = list(ledger['episodes'].values())
        assert len(rows) == 32 and all(r['status'] == 'complete' for r in rows), 'incomplete evaluation; do not shrink denominator'
        results[model] = {r['source_key']: r for r in rows}
        assert len(results[model]) == 32
    pairs = []
    for audit in audits:
        key = audit['source_key']
        base, trained = results['base'][key], results['step1'][key]
        assert base['episode_fingerprint'] == trained['episode_fingerprint'] == audit['episode_fingerprint']
        assert base['paired_seed'] == trained['paired_seed']
        for row in (base, trained):
            assert row['initial_rgb_evidence']['passed']
            assert row['turns'] and all(t['input']['task_context_check']['task_present'] for t in row['turns'])
            # Success is canonical geometry plus real environment termination,
            # not reward, action parsing, or the model claiming completion.
            if row['success']:
                assert row['final_canonical_metric']['success'] and row['turns'][-1]['done']
        pairs.append({'source_key': key, 'scene_id': audit['scene_id'],
                      'base_success': bool(base['success']), 'step1_success': bool(trained['success']),
                      'base_invalid_turns': base['invalid_turns'], 'step1_invalid_turns': trained['invalid_turns'],
                      'base_turns': base['model_turns'], 'step1_turns': trained['model_turns']})
    summary = {'status': 'COMPLETE', 'role': 'development_regression', 'pairs': 32,
               'generalization_claim': False, 'one_update_proves_learning': False,
               'expanded_training_allowed': False,
               'base_success': sum(p['base_success'] for p in pairs),
               'step1_success': sum(p['step1_success'] for p in pairs),
               'gained_tasks': sum(not p['base_success'] and p['step1_success'] for p in pairs),
               'lost_tasks': sum(p['base_success'] and not p['step1_success'] for p in pairs)}
    summary['success_rate_delta'] = (summary['step1_success'] - summary['base_success']) / 32
    for model in ('base', 'step1'):
        summary[model + '_invalid_turn_rate'] = sum(p[model + '_invalid_turns'] for p in pairs) / sum(p[model + '_turns'] for p in pairs)
        summary[model + '_episodes_with_invalid'] = sum(p[model + '_invalid_turns'] > 0 for p in pairs)
    output = run / eval_root / 'paired_report.json'
    output.write_text(json.dumps({'summary': summary, 'per_task': pairs}, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    return summary


def watch(run, timeout_seconds, eval_root='paired_eval'):
    """Bounded read-only worker monitoring; only writes this run's reports."""
    status_path = run / 'control_status.json'
    deadline = time.monotonic() + timeout_seconds
    last_phase = None
    while time.monotonic() < deadline:
        status = json.loads(status_path.read_text())
        gate_path = run / 'acceptance_gate.json'
        if gate_path.is_file():
            status['gpu_acceptance'] = json.loads(gate_path.read_text())['status']
            status['phase'] = 'paired_evaluation'
            status['paired_eval'] = 'RUNNING_OR_PENDING'
        exits = sorted((run / 'training_worker/attempts').glob('*/exit.json'))
        if exits:
            result = json.loads(exits[-1].read_text())
            status['trainer_exit'] = result
            if result['exit_status'] == 0:
                try:
                    status['paired_report'] = compare(run, eval_root)
                    status['phase'] = 'complete'
                    status['paired_eval'] = 'COMPLETE'
                except (AssertionError, KeyError, FileNotFoundError) as exc:
                    status['phase'] = 'paired_evaluation_incomplete'
                    status['paired_eval'] = 'FAIL_CLOSED'
                    status['error'] = str(exc)
            else:
                status['phase'] = 'worker_failed_preserve_artifacts'
                status['paired_eval'] = 'NOT_ACCEPTED'
                if status['gpu_acceptance'] != 'PASS':
                    status['gpu_acceptance'] = 'INCOMPLETE_OR_FAILED'
        status['monitor_updated_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        temporary = status_path.with_suffix('.tmp')
        temporary.write_text(json.dumps(status, indent=2) + '\n')
        temporary.replace(status_path)
        if status['phase'] != last_phase:
            print(json.dumps(status), flush=True)
            last_phase = status['phase']
        if exits:
            return
        time.sleep(30)
    raise TimeoutError('bounded monitor ended; do not infer worker completion')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--watch-seconds', type=int, default=0)
    parser.add_argument('--eval-root', default='paired_eval')
    args = parser.parse_args()
    if args.watch_seconds:
        watch(args.run, args.watch_seconds, args.eval_root)
    else:
        compare(args.run, args.eval_root)
