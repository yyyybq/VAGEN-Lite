#!/usr/bin/env python3
"""
Object-level train/test split for active_spatial_pipeline outputs.

Each item in the input jsonl belongs to a (scene_id, object_tuple) group.
Within each scene, object_tuples are shuffled and split 80/20 (configurable)
into train / test. This guarantees:
  - both splits cover all scenes
  - no object_tuple appears in both train and test (within the same scene)

object_tuple is taken from item['target_object']['objects'][*]['id'] if
present; otherwise it falls back to the object_label string.

Usage:
    python split_train_test_by_object.py \
        --input  output_100scenes/train_data_all.jsonl \
        --train_out output_100scenes/train.jsonl \
        --test_out  output_100scenes/test.jsonl \
        --test_ratio 0.2 \
        --seed 42
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path


def object_key(item: dict) -> str:
    """Stable identifier for the object tuple used by a sample."""
    tgt = item.get('target_object') or {}
    objs = tgt.get('objects')
    if isinstance(objs, list) and objs:
        ids = []
        for o in objs:
            oid = o.get('id') if isinstance(o, dict) else None
            if oid is None and isinstance(o, dict):
                oid = o.get('label', '?')
            ids.append(str(oid))
        return '|'.join(sorted(ids))
    # Fallback: use the human-readable object_label string.
    return str(item.get('object_label', 'unknown'))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True, type=Path,
                   help='Combined train_data jsonl from all scenes.')
    p.add_argument('--train_out', required=True, type=Path)
    p.add_argument('--test_out',  required=True, type=Path)
    p.add_argument('--test_ratio', type=float, default=0.2,
                   help='Fraction of object_tuples per scene assigned to test set.')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--min_test_per_scene', type=int, default=1,
                   help='If a scene has >=2 object_tuples, force at least this '
                        'many into test (when test_ratio rounds down to 0).')
    args = p.parse_args()

    rng = random.Random(args.seed)

    # 1) Load and group by (scene_id, object_tuple).
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    n_total = 0
    with args.input.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            scene_id = item.get('scene_id', 'unknown')
            key = (scene_id, object_key(item))
            groups[key].append(item)
            n_total += 1

    # 2) For each scene, split its object_tuples.
    scene_to_objs: dict[str, list[str]] = defaultdict(list)
    for (scene_id, obj_key) in groups.keys():
        scene_to_objs[scene_id].append(obj_key)

    train_keys: set[tuple[str, str]] = set()
    test_keys:  set[tuple[str, str]] = set()
    per_scene_stats: list[tuple[str, int, int, int]] = []  # scene, n_obj, n_train_obj, n_test_obj

    for scene_id, obj_list in scene_to_objs.items():
        obj_list_sorted = sorted(set(obj_list))
        n_obj = len(obj_list_sorted)
        shuffled = obj_list_sorted[:]
        rng.shuffle(shuffled)

        n_test = int(round(n_obj * args.test_ratio))
        if n_obj >= 2 and n_test < args.min_test_per_scene:
            n_test = args.min_test_per_scene
        n_test = min(n_test, n_obj)  # safety
        # Ensure at least one in train when possible.
        if n_obj >= 2 and n_test == n_obj:
            n_test = n_obj - 1

        test_objs  = set(shuffled[:n_test])
        train_objs = set(shuffled[n_test:])

        for o in train_objs:
            train_keys.add((scene_id, o))
        for o in test_objs:
            test_keys.add((scene_id, o))

        per_scene_stats.append((scene_id, n_obj, len(train_objs), len(test_objs)))

    # 3) Write outputs.
    args.train_out.parent.mkdir(parents=True, exist_ok=True)
    args.test_out.parent.mkdir(parents=True, exist_ok=True)

    n_train_items = 0
    n_test_items  = 0
    task_counter_train: Counter = Counter()
    task_counter_test:  Counter = Counter()

    with args.train_out.open('w', encoding='utf-8') as ftr, \
         args.test_out.open('w', encoding='utf-8') as fte:
        for key, items in groups.items():
            if key in train_keys:
                out = ftr
                n_train_items += len(items)
                ctr = task_counter_train
            elif key in test_keys:
                out = fte
                n_test_items += len(items)
                ctr = task_counter_test
            else:
                # Single-object scenes where n_obj==1 land here; default to train.
                out = ftr
                n_train_items += len(items)
                ctr = task_counter_train
            for it in items:
                out.write(json.dumps(it, ensure_ascii=False) + '\n')
                ctr[it.get('task_type', 'unknown')] += 1

    # 4) Report.
    n_scenes = len(scene_to_objs)
    n_obj_tuples = len(groups)
    print('=' * 60)
    print('Object-level train/test split')
    print('=' * 60)
    print(f'Input file        : {args.input}')
    print(f'Total items       : {n_total}')
    print(f'Total scenes      : {n_scenes}')
    print(f'Total obj-tuples  : {n_obj_tuples}')
    print(f'Test ratio        : {args.test_ratio}')
    print(f'Seed              : {args.seed}')
    print('-' * 60)
    print(f'Train items       : {n_train_items}  -> {args.train_out}')
    print(f'Test  items       : {n_test_items}  -> {args.test_out}')
    print('-' * 60)
    print('Per-task counts:')
    all_tasks = sorted(set(task_counter_train) | set(task_counter_test))
    print(f'  {"task":34s}  {"train":>8s}  {"test":>8s}')
    for t in all_tasks:
        print(f'  {t:34s}  {task_counter_train[t]:8d}  {task_counter_test[t]:8d}')
    print('-' * 60)
    print('Per-scene object-tuple split (first 20 shown):')
    for scene_id, n_obj, n_tr, n_te in per_scene_stats[:20]:
        print(f'  {scene_id}: {n_obj} obj-tuples -> train {n_tr} / test {n_te}')
    if len(per_scene_stats) > 20:
        print(f'  ... ({len(per_scene_stats) - 20} more scenes)')
    print('=' * 60)


if __name__ == '__main__':
    main()
