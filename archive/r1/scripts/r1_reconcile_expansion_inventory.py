#!/usr/bin/env python3
"""Reconcile frozen inventories by scoped source and exact policy task/initial.

Reads existing evidence only. Never promotes FOV or conflates task IDs from
different experiment namespaces. New output directories are immutable snapshots.
"""
import argparse
import hashlib
import json
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

from r1_audit_full_clean_target_inventory import summarize


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


@lru_cache(maxsize=None)
def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@lru_cache(maxsize=None)
def read(path):
    path = Path(path)
    if path.suffix == '.jsonl':
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return json.loads(path.read_text())


def refpath(value):
    return Path(value['path'] if isinstance(value, dict) else value).resolve()


def evidence(value, task_id):
    path = refpath(value)
    if isinstance(value, dict) and value.get('sha256'):
        assert digest(path) == value['sha256'], f'evidence hash mismatch: {path}'
    payload = read(path)
    rows = payload.get('results', []) if isinstance(payload, dict) else payload
    matches = [(i, r) for i, r in enumerate(rows) if r.get('task_id') == task_id]
    assert len(matches) == 1, (path, task_id, len(matches))
    i, record = matches[0]
    return record, {'path': str(path), 'sha256': digest(path), 'record_index': i, 'record_sha256': fingerprint(record)}


def pair_key(raw):
    ids = [str(o.get('id', o.get('ins_id', ''))) for o in raw['target_object']['objects']]
    assert len(ids) == 2 and all(ids) and ids[0] != ids[1]
    return str(raw['scene_id']) + '|' + '--'.join(sorted(ids))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--base', type=Path, required=True)
    p.add_argument('--expansion', type=Path, action='append', default=[])
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    b = a.base.resolve()
    old_inventory = b / 'r1_full_clean_target_inventory_v1_20260928'
    summary = read(old_inventory / 'inventory_summary.json')
    for path, expected in summary['inputs'].items():
        assert digest(Path(path).resolve()) == expected, f'frozen input mismatch: {path}'
    dev = next(Path(x) for x in summary['inputs'] if 'development_regression_manifest' in x)
    local = next(Path(x) for x in summary['inputs'] if 'parent_sources' in x)
    eval_rows = read(dev) + read(local)
    eval_scenes = {r['scene_id'] for r in eval_rows}
    eval_sources = {r.get('source_key') for r in eval_rows}
    base_meta = read(b / 'r1_clean_training_corpus_v1_20260919/verified_repaired_source_inventory.jsonl')
    base_raw = read(b / 'r1_clean_training_corpus_v1_20260919/repaired_canonical_train_manifest.jsonl')
    assert len(base_meta) == len(base_raw)
    base_by_source = {m['source_key']: (m, r) for m, r in zip(base_meta, base_raw)}
    input_files = {str(Path(x).resolve()): h for x, h in summary['inputs'].items()}
    candidates = []

    def add(meta, raw, refs, source_key, origin, source_ref):
        task_id = raw['task_id']
        runtime, rr = evidence(refs['runtime'], task_id)
        rgb, vr = evidence(refs.get('official_rgb', refs.get('observability')), task_id)
        reach, cr = evidence(refs['reachability'], task_id)
        archived, ar = evidence(refs['candidate'], task_id)
        # A manifest may add provenance fields; physical state/task must match.
        for key in ('init_camera', 'target_object', 'target_region', 'scene_id', 'task_type'):
            assert archived[key] == raw[key], (source_key, 'candidate mismatch', key)
        assert runtime['status'] == 'pass' and runtime.get('initial_success') is False
        assert not runtime.get('collisions') and not runtime.get('pose_mismatches')
        assert rgb['passed'] is True and rgb['scene_id'] == raw['scene_id']
        assert rgb['frames_rendered'] == rgb['frames_expected']
        sheet = Path(rgb['contact_sheet'])
        assert sheet.is_file(), sheet
        # Hash every official path frame, not just the heuristic verdict.
        frames = sorted(sheet.parent.glob('step_*.png'))
        assert len(frames) == rgb['frames_expected'], (sheet, len(frames), rgb['frames_expected'])
        for f in frames:
            input_files[str(f.resolve())] = digest(f)
        input_files[str(sheet.resolve())] = digest(sheet)
        steps = runtime.get('steps')
        assert isinstance(steps, int) and 1 <= steps <= 12
        lower = reach.get('certified_lower_bound')
        complete = reach.get('lower_bound_complete', False)
        if not complete:
            audit = reach.get('no_solution_depth_lower_bound') or runtime.get('lower_bound_audit') or {}
            if audit.get('complete') and audit.get('no_success_through_depth'):
                complete, lower = True, audit['max_depth'] + 1
        relation = str(raw['target_region']['params'].get('relation') or 'fov_inclusion')
        physical = {key: raw.get(key) for key in ('scene_id', 'task_type', 'init_camera', 'camera_model_version', 'canonical_task_metric_version')}
        physical.update(target_object=raw['target_object'], relation=relation,
                        object_a_center=raw['target_region']['params'].get('object_a_center'),
                        object_b_center=raw['target_region']['params'].get('object_b_center'))
        episode = fingerprint(physical)
        rec = {
            'source_key': source_key, 'source_row_index': meta['source_row_index'], 'split': 'train',
            'source_identity': source_ref, 'origin': origin, 'scene_id': raw['scene_id'],
            'task_id': task_id, 'task_type': raw['task_type'], 'relation': relation,
            'category_pair': '--'.join(sorted(o['label'] for o in raw['target_object']['objects'])),
            'pair_key': pair_key(raw), 'episode_fingerprint': episode,
            'first_success_step': steps, 'difficulty': {'certified_lower_bound': lower if complete else None,
                'lower_bound_complete': bool(complete), 'certificate_upper_bound': steps,
                'certified_medium': bool(complete and lower >= 4 and 4 <= steps <= 6)},
            'fov_quarantined': raw['task_type'] == 'fov_inclusion',
            'evidence': {'runtime': rr, 'official_rgb': vr, 'reachability': cr, 'candidate': ar,
                         'contact_sheet': str(sheet), 'rgb_frames': [str(f) for f in frames]},
            'versions': {'camera': raw.get('camera_model_version'), 'metric': raw.get('canonical_task_metric_version'),
                         'generator': raw.get('generator_version'), 'observability': rgb.get('version'),
                         'collision': runtime.get('collision_convention')},
            'candidate': raw,
        }
        for ref in (rr, vr, cr, ar):
            input_files[ref['path']] = ref['sha256']
        candidates.append(rec)

    for filename in ('projective_train_ready_deduplicated.jsonl', 'fov_provisional_deduplicated.jsonl'):
        path = old_inventory / filename
        input_files[str(path)] = digest(path)
        for m in read(path):
            if m['origin'] == 'old_v46_target_salvage':
                root = b / 'r1_old_v46_target_salvage_v1_20260919'
                projective = m['task_type'] == 'projective_relations'
                shard = root / ('projective_medium_v2' if projective else 'fov_min4') / 'shards' / m['scene_id']
                cert = shard / ('independent_validation' if projective else 'train')
                refs = {'runtime': cert / 'runtime_replay.json', 'reachability': cert / 'reachability_manifest.jsonl',
                        'official_rgb': shard / ('official_observability_v1' if projective else 'official_observability_v2') / 'observability_manifest.jsonl',
                        'candidate': cert / ('candidate_rows.jsonl' if projective else 'trainable.jsonl')}
                raw, _ = evidence(refs['candidate'], m['task_id'])
            else:
                full_meta, raw = base_by_source[m['source_key']]
                refs = full_meta['evidence']
            add(m, raw, refs, 'old_v46:' + m['source_key'], m['origin'], {'original_source_key': m['source_key']})

    for run in a.expansion:
        run = run.resolve()
        es = read(run / 'expansion_matrix/summary.json')
        assert es['accounting']['closed']
        scope = Path(es['scope']['path'])
        assert digest(scope / 'fresh_sources.jsonl') == es['scope']['fresh_sources_sha256']
        sources = read(scope / 'fresh_sources.jsonl')
        matrix = read(run / 'expansion_matrix/fresh_expansion_matrix.jsonl')
        assert len(matrix) == len(sources) == len({m['source_row_index'] for m in matrix})
        for m in matrix:
            if m['final_class'] not in ('PROJECTIVE_TRAIN_READY', 'FOV_PROVISIONAL_RUNTIME_RGB_V2'):
                continue
            i = m['source_row_index']
            source = sources[i]
            projective = source['task_type'] == 'projective_relations'
            shard = run / ('projective_medium_v2' if projective else 'fov_min4') / 'shards' / m['scene_id']
            cert = shard / ('independent_validation' if projective else 'train')
            refs = {'runtime': cert / 'runtime_replay.json', 'reachability': cert / 'reachability_manifest.jsonl',
                    'official_rgb': shard / ('official_observability_v1' if projective else 'official_observability_v2') / 'observability_manifest.jsonl',
                    'candidate': cert / ('candidate_rows.jsonl' if projective else 'trainable.jsonl')}
            raw, _ = evidence(refs['candidate'], m['candidate_task_id'])
            assert pair_key(raw) == pair_key(source)
            # Source identity survives schema-only changes and rerun namespaces.
            sk = 'fresh:' + fingerprint({'scene': source['scene_id'], 'task': source['task_type'],
                'lineage': source['fresh_task_lineage']})
            add(m, raw, refs, sk, run.name, {'manifest': str(scope / 'fresh_sources.jsonl'),
                'manifest_sha256': digest(scope / 'fresh_sources.jsonl'), 'source_row_index': i,
                'task_id': source['task_id'], 'lineage': source['fresh_task_lineage']})
        for f in (scope / 'fresh_sources.jsonl', run / 'expansion_matrix/summary.json', run / 'expansion_matrix/fresh_expansion_matrix.jsonl'):
            input_files[str(f)] = digest(f)

    selected, excluded, duplicates = [], [], []
    seen_sources, seen_episodes = {}, {}
    for r in candidates:  # Existing audited evidence wins; order is frozen by CLI.
        reasons = []
        if r['scene_id'] in eval_scenes:
            reasons.append('eval_scene_overlap')
        if r['source_identity'].get('original_source_key') in eval_sources:
            reasons.append('eval_source_overlap')
        if reasons:
            excluded.append({'record': r, 'reasons': reasons})
            continue
        other = seen_sources.get(r['source_key']) or seen_episodes.get(r['episode_fingerprint'])
        if other:
            duplicates.append({'source': r['source_key'], 'episode': r['episode_fingerprint'], 'retained_source': other['source_key'],
                               'reason': 'source' if r['source_key'] in seen_sources else 'exact_canonical_task_initial',
                               'original_record': r})
            continue
        seen_sources[r['source_key']] = seen_episodes[r['episode_fingerprint']] = r
        selected.append(r)
    assert len(candidates) == len(selected) + len(excluded) + len(duplicates)
    def stats(rs):
        s = summarize(rs)
        s.update(unique_object_pairs=len({r['pair_key'] for r in rs}),
                 certified_medium=sum(r['difficulty']['certified_medium'] for r in rs),
                 origins=dict(Counter(r['origin'] for r in rs)))
        return s
    proj = [r for r in selected if not r['fov_quarantined']]
    fov = [r for r in selected if r['fov_quarantined']]
    by_pair = defaultdict(list)
    for r in selected:
        by_pair[(r['task_type'], r['pair_key'])].append(r['source_key'])
    summary_out = {'version': 'r1_reconciled_inventory_v1_20260930', 'input_records': len(candidates),
        'unique_source_lineages_before_episode_dedup': len({r['source_key'] for r in candidates}),
        'selected_sources': len(selected), 'selected_episodes': len(seen_episodes), 'duplicates_removed': len(duplicates),
        'new_eval_exclusions': len(excluded), 'projective_train_ready': stats(proj), 'fov_provisional': stats(fov),
        'fov_train_ready': 0, 'old_v46_targets': {'projective': 2003, 'fov': 1503},
        'deficit': {'projective': 2003-len(proj), 'fov_train_ready': 1503, 'fov_provisional_gap':1503-len(fov)},
        'scope': 'completed results only; running retry excluded until final runtime/RGB accounting',
        'dedup_policy': 'scoped source then exact task/pair/relation/initial/camera/metric fingerprint; repeated pair alone is not duplicate episode',
        'evaluation_isolation': {'input_manifests': [str(dev), str(local)], 'excluded_scenes': sorted(eval_scenes),
            'local_action_use': 'exclusion identity only; no states/actions/scores used for generation or selection'},
        'sha256_inputs': input_files}
    a.output.mkdir(parents=True, exist_ok=False)
    def put(name, obj):
        (a.output/name).write_text(json.dumps(obj, indent=2, sort_keys=True)+'\n')
    def putrows(name, rs):
        (a.output/name).write_text(''.join(json.dumps(r, sort_keys=True)+'\n' for r in rs))
    putrows('projective_train_ready_inventory.jsonl', proj)
    putrows('fov_provisional_inventory.jsonl', fov)
    putrows('duplicates.jsonl', duplicates)
    putrows('excluded.jsonl', excluded)
    putrows('all_input_evidence_records.jsonl', candidates)
    put('repeated_pairs.json', [{'task': k[0], 'pair': k[1], 'sources': v} for k,v in sorted(by_pair.items()) if len(v)>1])
    put('inventory_summary.json', summary_out)
    (a.output/'SHA256SUMS').write_text(''.join(f'{digest(f)}  {f.name}\n' for f in sorted(a.output.iterdir()) if f.is_file()))
    print(json.dumps({k:v for k,v in summary_out.items() if k not in ('sha256_inputs','projective_train_ready','fov_provisional')}, indent=2))
    print('Projective', len(proj), 'FOV provisional', len(fov))


if __name__ == '__main__':
    main()
