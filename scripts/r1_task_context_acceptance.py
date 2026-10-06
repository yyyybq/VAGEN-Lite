#!/usr/bin/env python3
"""Build an isolated, minimal patch of R1 v7; audit real acceptance artifacts.

Never resumes or writes the historical run. No formal training entry point.
"""
import argparse
import hashlib
import io
import json
import re
import tarfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
OLD = ROOT / "exps/vagen_active_spatial/R1-clean-Projective-v0"
DEV = ROOT / "exps/vagen_active_spatial/r1_h1_aoss_repair_20260905/r1_canonical_dev_eval32_20260915/frozen_input"


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')


def replace(text, old, new, count=1):
    assert text.count(old) == count, (old[:100], text.count(old), count)
    return text.replace(old, new)


def prepare(run):
    assert run.parent == OLD.parent and run != OLD
    run.mkdir(exist_ok=False)
    frozen = run / 'frozen'
    package = run / 'package'
    frozen.mkdir(); package.mkdir()
    source = OLD / 'package_v7/source.tar.gz'
    assert digest(source) == '3c53d0d3473212a11f2df415d7ba07f0cc8e4909d177ead65a034bf6be00c648'
    rows = read(OLD / 'frozen_v1/train.jsonl')
    audits = read(OLD / 'frozen_v1/audit_only.jsonl')
    selected = [i for i, row in enumerate(rows) if row['scene_id'] == '0267_840790'][:12]
    assert len(selected) == 12 and len(rows) == len(audits) == 210
    eval_rows = read(DEV / 'policy_input_rows.jsonl')
    assert len(eval_rows) == 32
    assert not {rows[i]['scene_id'] for i in selected} & {r['scene_id'] for r in eval_rows}
    for name, values in [('train.jsonl', [rows[i] for i in selected]),
                         ('audit_only.jsonl', [audits[i] for i in selected]),
                         ('eval_policy_rows.jsonl', eval_rows)]:
        (frozen / name).write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in values))
    for name in ('development_regression_manifest.jsonl', 'policy_input_rows.jsonl'):
        (frozen / name).write_bytes((DEV / name).read_bytes())
    protocol = json.loads((DEV / 'eval_protocol.json').read_text())
    for key in ('audit_manifest', 'policy_input'):
        path = frozen / Path(protocol[key]['path']).name
        assert digest(path) == protocol[key]['sha256']
        protocol[key]['path'] = str(path)
    protocol['policy_contract']['task_context'] = 'persistent_public_task_v1'
    write_json(frozen / 'eval_protocol.json', protocol)
    for name in ('train.yaml', 'val.yaml'):
        cfg = yaml.safe_load((OLD / 'frozen_v1' / name).read_text())
        env = cfg['envs'][0]
        env['config']['jsonl_path'] = str(frozen / ('train.jsonl' if name == 'train.yaml' else 'eval_policy_rows.jsonl'))
        if name == 'train.yaml':
            env.update(n_envs=12, seed_list=list(range(12)), seed=[0, 12])
            env['config']['train_size'] = 12
        (frozen / name).write_text(yaml.safe_dump(cfg, sort_keys=False))
    smoke = (OLD / 'frozen_v1/smoke.yaml').read_text().replace(str(OLD / 'frozen_v1'), str(frozen)).replace(str(OLD / 'smoke'), str(run / 'smoke'))
    cfg = yaml.safe_load(smoke)
    cfg['trainer']['experiment_name'] = run.name
    assert cfg['trainer']['total_training_steps'] == 700 and cfg['trainer']['critic_warmup'] == 0
    (frozen / 'smoke.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    old_gate = json.loads((OLD / 'frozen_v1/data_gate.json').read_text())
    write_json(frozen / 'data_gate.json', {'model': old_gate['model'], 'episodes': 12,
        'data_gate': 'FROZEN_SUBSET_SCENE_ISOLATED', 'old_indices': selected,
        'source_train_sha256': digest(OLD / 'frozen_v1/train.jsonl'),
        'eval_scene_overlap': [], 'generalization_claim': 'none_development_regression'})
    changes = {}
    with tarfile.open(source) as archive:
        contents = {m.name: archive.extractfile(m).read() for m in archive.getmembers() if m.isfile()}
    def patch(name, value):
        old = contents.get(name)
        data = value.encode() if isinstance(value, str) else value
        if name.endswith('.py'):
            compile(data, name, 'exec')
        changes[name] = {'before': hashlib.sha256(old).hexdigest() if old else None,
                         'after': hashlib.sha256(data).hexdigest()}
        contents[name] = data
    name = 'vagen/envs/active_spatial/prompt.py'
    text = contents[name].decode()
    start = text.index('def action_template(')
    end = text.index('\n\n# Format prompt', start)
    current = (ROOT / name).read_text()
    patch(name, text[:start] + current[current.index('def action_template('):current.index('\n\n# Format prompt', current.index('def action_template('))] + text[end:])
    name = 'vagen/envs/active_spatial/env.py'
    text = contents[name].decode()
    text, count = re.subn(r'(action_template\(\n[^\n]+\n\s+env_feedback=[^\n]+,)', r'\1\n                task_prompt=self._build_task_prompt(self.current_item),', text)
    assert count == 2, count
    patch(name, text)
    name = 'vagen/agent_loop/gym_agent_loop_no_concat.py'
    text = contents[name].decode()
    text = replace(text, '        agent_data.task_type       = _task_type',
        '        agent_data.public_task = str((info or {}).get("task_prompt") or "")\n'
        '        if agent_data.env_name == "ActiveSpatial" and not agent_data.public_task:\n'
        '            raise ValueError("ActiveSpatial reset did not supply a public task")\n'
        '        agent_data.task_type       = _task_type')
    start = text.index('        if len(agent_data.turn_prompt_ids) > self.prompt_length:')
    end = text.index('        return AgentState.GENERATING', start)
    text = text[:start] + '        if len(agent_data.turn_prompt_ids) > self.prompt_length:\n            raise ValueError("current task and image exceed prompt budget; refusing truncation")\n' + text[end:]
    text = replace(text, '            prompt_ids = prompt_ids[-self.prompt_length :]',
        '            raise ValueError("policy prompt exceeds budget; refusing truncation")\n'
        '        if agent_data.public_task:\n'
        '            from vagen.utils.task_context import check_policy_tokens\n'
        '            agent_data.task_context_check = check_policy_tokens(\n'
        '                self.tokenizer, prompt_ids, agent_data.public_task, self.prompt_length)')
    text = replace(text, 'extra_fields={"reward_extra_info": {',
        'extra_fields={"reward_extra_info": {\n'
        '                "task_context_present": float(bool(getattr(agent_data, "task_context_check", {}).get("task_present"))),')
    patch(name, text)
    name = 'scripts/r1_clean_projective_runtime_preflight.py'
    text = contents[name].decode()
    text = replace(text, 'assert len(items)==len(audits)==210', 'assert len(items)==len(audits)==12')
    text = replace(text, '            obs,_=env.reset(seed=i)',
        '            obs,reset_info=env.reset(seed=i)\n'
        '            from vagen.utils.task_context import assert_task_text\n'
        '            public_task=reset_info["task_prompt"]\n'
        '            assert_task_text(obs["obs_str"],public_task)')
    text = replace(text, '                state=env.view_engine.get_pose();',
        '                assert_task_text(obs["obs_str"],public_task)\n'
        '                state=env.view_engine.get_pose();')
    patch(name, text)
    for name in ('vagen/utils/task_context.py', 'scripts/r1_task_context_acceptance.py',
                 'scripts/r1_run_canonical_dev_eval32.py',
                 'examples/train/active_spatial/sco_r1_task_context_acceptance.sh'):
        patch(name, (ROOT / name).read_bytes())
    with tarfile.open(package / 'source.tar.gz', 'w:gz') as archive:
        for name, data in sorted(contents.items()):
            member = tarfile.TarInfo(name)
            member.size = len(data); member.mode = 0o644
            archive.addfile(member, io.BytesIO(data))
    # Bootstrap also comes from this hashed archive; no mutable workspace code.
    owner = contents['examples/train/active_spatial/sco_run_as_artifact_owner.sh']
    (package / 'owner.sh').write_bytes(owner)
    for mode in ('renderer', 'training'):
        launch = f'''#!/usr/bin/env bash
set -euo pipefail
PACKAGE={package}
if [[ $(id -u) == 0 ]]; then
  exec bash "${{PACKAGE}}/owner.sh" bash "$0" owner
fi
[[ $(id -u) == 20325 && $(id -g) == 20325 ]] || exit 3
cd "${{PACKAGE}}"
sha256sum -c SHA256SUMS
WORK=$(mktemp -d /tmp/r1_context.XXXXXXXX)
tar -xzf source.tar.gz -C "${{WORK}}"
cd "${{WORK}}"
exec timeout --signal=TERM --kill-after=120s 6h bash examples/train/active_spatial/sco_r1_task_context_acceptance.sh {mode} {run}
'''
        (package / f'launch_{mode}.sh').write_text(launch)
    write_json(package / 'provenance.json', {'base_archive': str(source), 'base_sha256': digest(source),
        'changes': changes, 'unrelated_worktree_edits_included': False,
        'reward_optimizer_and_verl_unchanged': True, 'max_ppo_updates': 1,
        'renderer_gpus': 1, 'training_gpus': 8, 'worker_timeout_hours': 6})
    for directory in (frozen, package):
        (directory / 'SHA256SUMS').write_text(''.join(f'{digest(p)}  {p.name}\n' for p in sorted(directory.iterdir()) if p.is_file()))
    latest = sorted((OLD / 'formal/rollout_data').glob('*.jsonl'), key=lambda p: int(p.stem))[-1]
    write_json(run / 'historical_run_preserved.json', {'run': str(OLD), 'classification': 'faulty_task_context_baseline',
        'renderer_job': 'pt-t7bb75de', 'trainer_job': 'pt-kbvfmyih', 'scheduler_status': 'FAILED',
        'cause': 'renderer_48h_lease_expired_then_training_connection_failure',
        'latest_rollout_step': int(latest.stem), 'latest_complete_checkpoint': 200,
        'old_checkpoint_complete_sha256': digest(OLD / 'formal/checkpoints/global_step_200/COMPLETE'),
        'old_run_modified': False})
    print(json.dumps({'run': str(run), 'train': 12, 'eval': 32, 'archive_sha256': digest(package / 'source.tar.gz')}))


def gate(run):
    from vagen.utils.task_context import assert_task_text
    rows = read(run / 'smoke/rollout_data/1.jsonl')
    tasks = read(run / 'frozen/train.jsonl')
    assert rows and len(rows) > 48, 'must exercise later-turn policy inputs'
    valid = strict = 0
    for row in rows:
        text = row['input']
        matches = [t['task_description'] for t in tasks if t['task_description'] in text]
        assert len(set(matches)) == 1, 'missing/ambiguous complete task in actual model input'
        assert_task_text(text, matches[0])
        assert float(row['task_context_present']) == 1
        valid += not bool(row['invalid_action'])
        strict += bool(row['strict_format_correct_rate'])
    assert valid / len(rows) >= .95, ('action parsing gate', valid, len(rows))
    assert strict / len(rows) >= .95, ('strict action format gate', strict, len(rows))
    report = {'status': 'PASS', 'real_rollout_turns': len(rows), 'task_retention': 1.0,
              'action_valid_rate': valid / len(rows), 'strict_format_rate': strict / len(rows),
              'ppo_gate': json.loads((run / 'smoke/gate.json').read_text())['status']}
    assert report['ppo_gate'] == 'PASS'
    write_json(run / 'acceptance_gate.json', report)
    print(json.dumps(report))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['prepare', 'gate'])
    parser.add_argument('--run', type=Path, required=True)
    args = parser.parse_args()
    {'prepare': prepare, 'gate': gate}[args.action](args.run.resolve())


if __name__ == '__main__':
    main()
