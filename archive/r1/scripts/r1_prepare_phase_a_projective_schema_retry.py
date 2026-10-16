#!/usr/bin/env python3
"""Freeze only the 465 schema-failed Phase A requests, preserving their indices."""
import argparse
import copy
import hashlib
import json
from pathlib import Path

from r1_projective_request_schema import half_plane_geometry, validate_projective_params


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    old_scope = a.base / 'r1_fullscale_canonical_expansion_scope_v1_20260928'
    old_run = a.base / 'r1_fullscale_canonical_expansion_phase_a_v1_20260928'
    old = [json.loads(x) for x in (old_scope / 'fresh_sources.jsonl').read_text().splitlines()]
    errors = {}
    for path in sorted((old_run / 'projective_medium_v2/shards').glob('*/selector_results.json')):
        for r in json.loads(path.read_text())['results']:
            assert r['source_row_index'] not in errors
            errors[r['source_row_index']] = r
    indices = [i for i, r in enumerate(old) if r['task_type'] == 'projective_relations']
    assert indices == list(range(465)) and set(indices) == set(errors)
    assert all(r['status'] == 'implementation_error' and r['error'] == "KeyError('boundary_point')" for r in errors.values())
    repaired = []
    for i in indices:
        r = copy.deepcopy(old[i])
        params = r['target_region']['params']
        geometry = half_plane_geometry(params['object_a_center'], params['object_b_center'], params['relation'])
        assert not set(geometry).intersection(params), 'refuse to overwrite existing geometry'
        params.update(geometry)
        validate_projective_params(params)
        check = copy.deepcopy(r)
        for key in geometry:
            del check['target_region']['params'][key]
        assert check == old[i]
        repaired.append(r)
    a.output.mkdir(parents=True, exist_ok=False)
    def put(name, obj):
        (a.output / name).write_text(json.dumps(obj, indent=2, sort_keys=True) + '\n')
    (a.output / 'fresh_sources.jsonl').write_text(''.join(json.dumps(r, sort_keys=True) + '\n' for r in repaired))
    put('sources.json', {'train': str((a.output / 'fresh_sources.jsonl').resolve())})
    selection = json.loads((old_scope / 'projective_medium_selection.json').read_text())
    assert [r['source_row_index'] for r in selection['records']] == indices
    put('projective_medium_selection.json', selection)
    put('fov_source_index_selection.json', {'train': []})
    put('retry_provenance.json', {
        'version': 'r1_phase_a_projective_schema_only_retry_v1_20260930',
        'old_scope': str(old_scope), 'old_scope_sha256': sha(old_scope / 'fresh_sources.jsonl'),
        'old_run': str(old_run), 'requests': len(repaired),
        'original_indices_preserved': True, 'fov_requests': 0,
        'changes': ['target_region.params.boundary_point', 'target_region.params.boundary_direction', 'target_region.params.normal'],
        'budget': {'seed_cap': 12, 'per_seed_expansions': 512, 'candidate_cap_per_seed': 48, 'lower_expansions': 100000},
        'historical_failures': errors,
    })
    (a.output / 'SHA256SUMS').write_text(''.join(f'{sha(f)}  {f.name}\n' for f in sorted(a.output.iterdir()) if f.is_file()))
    print(json.dumps({'requests': len(repaired), 'scenes': len({r['scene_id'] for r in repaired}), 'sha256': sha(a.output / 'fresh_sources.jsonl')}))


if __name__ == '__main__':
    main()
