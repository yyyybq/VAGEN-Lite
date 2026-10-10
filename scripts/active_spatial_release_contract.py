"""Frozen physical-scene partitions and raw parent-task lineage for release v2."""
import json
from pathlib import Path
from vagen.envs.active_spatial.dataset_contract import row_digest

SPLITS = ('train', 'val', 'test')

def scene_partitions(manifest):
    sets = {s: set(map(str, manifest.get(s + '_scenes', []))) for s in SPLITS}
    aliases = manifest.get('aliases', {})
    seen = {}
    for split, scenes in sets.items():
        for scene in scenes:
            physical = aliases.get(scene, scene)
            if physical in seen and seen[physical] != split:
                raise ValueError('Overlapping physical scenes in split manifest')
            seen[physical] = split
    for scene, required in manifest.get('reserved_scenes', {}).items():
        physical = aliases.get(scene, scene)
        if seen.get(physical) != required:
            raise ValueError('Reserved scene reassigned: ' + scene)
    return {scene: split for split, scenes in sets.items() for scene in scenes}


def load_registry(path, split):
    if not path:
        raise ValueError('Release v2 requires --task-registry with raw task hashes')
    lookup = scene_partitions(split)
    registry = {}
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r['task_id'] in registry:
            raise ValueError('Duplicate parent task ID')
        if lookup.get(str(r['scene_id'])) != r['split']:
            raise ValueError('Task registry split disagrees with frozen scenes')
        registry[r['task_id']] = r
    return registry


def bind_derived(row, kind, registry, lookup):
    parent = row.get('source_task_id') if kind == 'trajectory' else row.get('parent_task_id')
    r = registry.get(parent)
    if r is None:
        raise ValueError('Unknown or missing raw parent task ID')
    if row.get('source_task_sha256') != r['source_task_sha256']:
        raise ValueError('Raw parent task hash mismatch')
    if str(row['scene_id']) != str(r['scene_id']) or lookup.get(str(row['scene_id'])) != r['split']:
        raise ValueError('Parent/derived physical scene or split mismatch')
    if row.get('split') != r['split']:
        raise ValueError('Explicit derived split disagrees with raw parent')
    for key in ('camera_model_version', 'canonical_task_metric_version', 'action_protocol_version'):
        if row.get('source_versions', {}).get(key) != r.get('versions', {}).get(key) or not r.get('versions', {}).get(key):
            raise ValueError('Parent/derived version mismatch: ' + key)
    return r


def bind_raw(row, registry, lookup):
    r = registry.get(row.get('task_id'))
    if r is None or row_digest(row) != r['source_task_sha256']:
        raise ValueError('Unregistered or changed raw task')
    if str(row['scene_id']) != r['scene_id'] or lookup.get(r['scene_id']) != r['split'] or row.get('split') != r['split']:
        raise ValueError('Raw task/registry split mismatch')
