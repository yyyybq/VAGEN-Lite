"""Replayed QA must keep exact raw-task and pose/image provenance."""
import argparse
import hashlib
import json
import numpy as np
import pytest
from PIL import Image
from data_gen.active_spatial_qa.generate_paired_qa import generate
from data_gen.active_spatial_qa.test_qa_contract import _fixture


def inputs(tmp_path):
    item = dict(_fixture(), task_id='raw-parent', split='train')
    raw = tmp_path / 'tasks.jsonl'
    raw.write_text(json.dumps(item) + '\n')
    image = tmp_path / 'frame.png'
    Image.new('RGB', (32, 32), 'blue').save(image)
    pose = np.eye(4).tolist()
    row = {'source_task_id': item['task_id'], 'source_task_sha256': hashlib.sha256(json.dumps(item, sort_keys=True, allow_nan=False).encode()).hexdigest(), 'scene_id': item['scene_id'], 'split': 'train', 'primitive_trajectory_trace': [{'pose_c2w': pose}], 'primitive_image_paths': [str(image)]}
    bank = tmp_path / 'sft.jsonl'
    bank.write_text(json.dumps(row) + '\n')
    args = argparse.Namespace(input=str(raw), output_dir=str(tmp_path/'qa'), split='train', split_scenes='', task_type='', limit=0, states_from_sft=str(bank))
    return args, row, bank


def test_replayed_pose_image_and_parent_are_preserved(tmp_path):
    args, source, _ = inputs(tmp_path)
    result = generate(args)
    assert result['errors'] == 0 and result['samples_written'] == 1
    row = json.loads((tmp_path/'qa/manifest.jsonl').read_text())
    assert row['parent_task_id'] == source['source_task_id']
    assert row['source_task_sha256'] == source['source_task_sha256']
    assert row['state_pose_c2w'] == source['primitive_trajectory_trace'][0]['pose_c2w']
    assert row['public_observation']['image_path'] == source['primitive_image_paths'][0]
    assert row['source'] == 'real_sft_runtime_replay'


@pytest.mark.parametrize('change', [{'source_task_sha256': 'changed'}, {'split': 'test'}, {'primitive_image_paths': []}])
def test_mismatched_replay_is_excluded(tmp_path, change):
    args, row, bank = inputs(tmp_path)
    bank.write_text(json.dumps(dict(row, **change))+'\n')
    result = generate(args)
    assert result['errors'] == 1 and result['samples_written'] == 0


def test_resume_binds_replay_bank_hash(tmp_path):
    args, row, bank = inputs(tmp_path)
    generate(args)
    bank.write_text(json.dumps(dict(row, source_task_sha256='changed'))+'\n')
    with pytest.raises(ValueError, match='hash mismatch'):
        generate(args)
