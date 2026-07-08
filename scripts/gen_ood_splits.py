#!/usr/bin/env python3
"""
gen_ood_splits.py — 生成 Active Spatial 五轴 OOD 泛化评测集

用法:
    python3 scripts/gen_ood_splits.py
    python3 scripts/gen_ood_splits.py --out_dir data_gen/active_spatial_pipeline/ood_splits
    python3 scripts/gen_ood_splits.py --train_jsonl path/to/train.jsonl --test_jsonl path/to/test.jsonl
    python3 scripts/gen_ood_splits.py --max_per_split 300  # 每个 split 最大条目数（子样本）

输出文件（均为 JSONL，字段与 train_100scenes_7types.jsonl 兼容）:
    ood_scene.jsonl       — Split-1: 未见过的房间场景（8个 OOD scene_id）
    ood_instance.jsonl    — Split-2: 已见类别，未见实例（OOD场景中的已知 object_label）
    ood_category.jsonl    — Split-3: 完全未见过的物体类别（object_label 不在训练集中）
    ood_template.jsonl    — Split-4: 同等语义，不同自然语言表达（task_description 改写）
    ood_geometry.jsonl    — Split-5: 超出训练分布的几何参数（distance 超出 [p10,p90]）
    ood_splits_summary.md — 汇总报告

五个 split 对应五种泛化维度：
    1. 场景泛化  (scene generalization)
    2. 实例泛化  (instance generalization)
    3. 类别泛化  (category generalization)
    4. 语言泛化  (language / instruction template generalization)
    5. 几何泛化  (geometry / parameter generalization)
"""

import json
import os
import re
import random
import argparse
from collections import Counter, defaultdict
from typing import List, Dict, Optional


# ─── Constants ───────────────────────────────────────────────────────────────

TRAIN_TASK_TYPES = {
    'absolute_positioning', 'delta_control', 'equidistance',
    'projective_relations', 'occlusion_alignment', 'fov_inclusion', 'centering'
}

# P10/P90 thresholds per task (computed from train_100scenes_7types.jsonl)
# These are baked in so split-5 is reproducible without recomputing each run.
GEOM_THRESHOLDS = {
    'absolute_positioning': (1.50, 2.58),
    'centering':            (5.63, 10.50),
    'delta_control':        (0.59, 2.11),
    'equidistance':         (3.25, 9.32),
    'fov_inclusion':        (1.35, 3.45),
    'occlusion_alignment':  (2.13, 5.95),
    'projective_relations': (3.19, 6.17),
}

# ─── Instruction template paraphrasers ───────────────────────────────────────

def _rewrite_absolute_positioning(desc: str) -> str:
    """'Move to any position {d}m from {obj}' → alternative phrasing."""
    m = re.match(r'Move to any position ([\d.]+)m from (.+)', desc)
    if m:
        d, obj = m.group(1), m.group(2)
        return f'Navigate to a location {d} meters away from {obj}'
    return desc  # fallback: unchanged


def _rewrite_delta_control(desc: str) -> str:
    """'Move {d}m toward {obj}' → alternative phrasing."""
    m = re.match(r'Move ([\d.]+)m toward (.+)', desc)
    if m:
        d, obj = m.group(1), m.group(2)
        return f'Approach {obj} and stop when you are {d}m closer to it'
    return desc


def _rewrite_equidistance(desc: str) -> str:
    """'Move to any position equidistant from {a} and {b}' → alternative."""
    m = re.match(r'Move to any position equidistant from (.+) and (.+)', desc)
    if m:
        a, b = m.group(1), m.group(2)
        return f'Find a spot that is equally far from both {a} and {b}'
    return desc


def _rewrite_projective_relations(desc: str) -> str:
    """'Position where {a} appears to the left of {b}' → swap subject/object direction."""
    m = re.match(r'Position where (.+) appears to the (left|right) of (.+)', desc)
    if m:
        a, direction, b = m.group(1), m.group(2), m.group(3)
        opposite = 'right' if direction == 'left' else 'left'
        # Semantically equivalent: a left of b  ↔  b right of a
        return f'Position where {b} appears to the {opposite} of {a}'
    return desc


def _rewrite_occlusion_alignment(desc: str) -> str:
    """'Position where {a} is hidden behind {b}' → alternative."""
    m = re.match(r'Position where (.+) is hidden behind (.+)', desc)
    if m:
        a, b = m.group(1), m.group(2)
        return f'Find a viewpoint where {b} fully blocks your view of {a}'
    return desc


def _rewrite_fov_inclusion(desc: str) -> str:
    """'Position where both {a} and {b} are visible' → alternative."""
    m = re.match(r'Position where both (.+) and (.+) are visible', desc)
    if m:
        a, b = m.group(1), m.group(2)
        return f'Locate a spot where you can see both {a} and {b} at the same time'
    return desc


def _rewrite_centering(desc: str) -> str:
    """'Position where {a} is centered between {b} and {c}' → alternative."""
    m = re.match(r'Position where (.+) is centered between (.+) and (.+)', desc)
    if m:
        a, bc = m.group(1), m.group(2) + ' and ' + m.group(3)
        return f'Stand where {a} appears midway between {bc}'
    return desc


TEMPLATE_REWRITERS = {
    'absolute_positioning': _rewrite_absolute_positioning,
    'delta_control':        _rewrite_delta_control,
    'equidistance':         _rewrite_equidistance,
    'projective_relations': _rewrite_projective_relations,
    'occlusion_alignment':  _rewrite_occlusion_alignment,
    'fov_inclusion':        _rewrite_fov_inclusion,
    'centering':            _rewrite_centering,
}


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _percentile(lst: list, p: float) -> float:
    s = sorted(lst)
    idx = int(len(s) * p / 100)
    return float(s[min(idx, len(s) - 1)])


def _balanced_sample(items: List[dict], max_total: int, key: str = 'task_type') -> List[dict]:
    """Sample up to max_total items, balanced across key values."""
    if len(items) <= max_total:
        return items
    groups = defaultdict(list)
    for item in items:
        groups[item.get(key, 'unknown')].append(item)
    n_groups = len(groups)
    per_group = max(1, max_total // n_groups)
    sampled = []
    for g_items in groups.values():
        random.shuffle(g_items)
        sampled.extend(g_items[:per_group])
    # fill up to max_total if we have spare quota
    random.shuffle(sampled)
    return sampled[:max_total]


def _write_jsonl(path: str, items: List[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def _task_dist(items: List[dict]) -> Dict[str, int]:
    return dict(sorted(Counter(i['task_type'] for i in items).items()))


# ─── Split generators ─────────────────────────────────────────────────────────

def make_ood_scene(test_items: List[dict], train_scenes: set,
                   max_total: int) -> List[dict]:
    """Split-1: Items from scenes not in training set."""
    ood_scene_ids = set(i['scene_id'] for i in test_items) - train_scenes
    candidates = [i for i in test_items
                  if i['scene_id'] in ood_scene_ids
                  and i['task_type'] in TRAIN_TASK_TYPES]
    return _balanced_sample(candidates, max_total)


def make_ood_instance(test_items: List[dict], train_scenes: set,
                      train_obj_labels: set, max_total: int) -> List[dict]:
    """Split-2: OOD scene + object_label seen in training (same category, new room)."""
    ood_scene_ids = set(i['scene_id'] for i in test_items) - train_scenes
    candidates = [i for i in test_items
                  if i['scene_id'] in ood_scene_ids
                  and i['task_type'] in TRAIN_TASK_TYPES
                  and i['object_label'] in train_obj_labels]
    return _balanced_sample(candidates, max_total)


def make_ood_category(test_items: List[dict], train_obj_labels: set,
                      max_total: int) -> List[dict]:
    """Split-3: object_label completely absent from training (any scene)."""
    candidates = [i for i in test_items
                  if i['task_type'] in TRAIN_TASK_TYPES
                  and i['object_label'] not in train_obj_labels]
    return _balanced_sample(candidates, max_total)


def make_ood_template(test_items: List[dict], max_total: int) -> List[dict]:
    """Split-4: Paraphrase task_description with alternative natural-language templates."""
    # Filter to 7 training task types; take a balanced sample; rewrite descriptions
    candidates = [i for i in test_items if i['task_type'] in TRAIN_TASK_TYPES]
    sampled = _balanced_sample(candidates, max_total)
    result = []
    unchanged_count = 0
    for item in sampled:
        item = dict(item)  # shallow copy to avoid mutating source
        original_desc = item['task_description']
        rewriter = TEMPLATE_REWRITERS.get(item['task_type'])
        if rewriter:
            new_desc = rewriter(original_desc)
            if new_desc == original_desc:
                unchanged_count += 1
        else:
            new_desc = original_desc
            unchanged_count += 1
        item['task_description_original'] = original_desc
        item['task_description'] = new_desc
        result.append(item)
    if unchanged_count:
        print(f'  [WARNING] {unchanged_count}/{len(result)} items had unchanged descriptions (regex mismatch)')
    return result


def make_ood_geometry(test_items: List[dict], thresholds: dict,
                      max_total: int) -> List[dict]:
    """Split-5: Items with distance outside [p10, p90] of training distribution."""
    candidates = []
    for item in test_items:
        tt = item['task_type']
        if tt not in TRAIN_TASK_TYPES or tt not in thresholds:
            continue
        p10, p90 = thresholds[tt]
        d = item.get('distance', 0.0)
        if d < p10 or d > p90:
            candidates.append(item)
    return _balanced_sample(candidates, max_total)


# ─── Summary report ──────────────────────────────────────────────────────────

def _format_task_dist(d: dict) -> str:
    return ', '.join(f'{k}:{v}' for k, v in sorted(d.items()))


def write_summary(summary_path: str, splits: dict, train_info: dict,
                  geom_thresholds: dict) -> None:
    lines = [
        '# Active Spatial OOD Splits — Summary',
        '',
        '## Training Data Info',
        f"- Train items: {train_info['n_train']}",
        f"- Train scenes: {train_info['n_scenes']}",
        f"- Train object labels: {train_info['n_labels']}",
        f"- Train task types: {', '.join(sorted(TRAIN_TASK_TYPES))}",
        '',
        '## OOD Split Statistics',
        '',
        '| Split | Axis | N items | Scenes | Task distribution |',
        '|-------|------|---------|--------|-------------------|',
    ]
    for split_name, items in splits.items():
        n = len(items)
        scenes = len(set(i['scene_id'] for i in items))
        td = _format_task_dist(_task_dist(items))
        lines.append(f'| {split_name} | — | {n} | {scenes} | {td} |')

    lines += [
        '',
        '## Split Descriptions',
        '',
        '### Split-1: OOD Scene (`ood_scene.jsonl`)',
        '- Items from the **8 held-out scenes** not present in training.',
        f"- Held-out scene IDs: {', '.join(sorted(train_info['ood_scene_ids']))}",
        '- Tests: Can the model navigate spatially in completely unseen room layouts?',
        '',
        '### Split-2: OOD Instance (`ood_instance.jsonl`)',
        '- Items from OOD scenes where `object_label` **was seen in training**.',
        '- Same category of object (e.g., "wardrobe") but in a different room.',
        '- Subset of Split-1; isolates instance-level vs. scene-level generalization.',
        '- Tests: Does the model generalize to new instances of familiar object types?',
        '',
        '### Split-3: OOD Category (`ood_category.jsonl`)',
        '- Items where `object_label` is **entirely absent from the training set**.',
        f"- Unique new labels: {len(set(i['object_label'] for i in splits.get('ood_category', [])))}",
        '- Tests: Can the model navigate to object types it has never seen during training?',
        '',
        '### Split-4: OOD Template (`ood_template.jsonl`)',
        '- Same task content, but `task_description` rewritten with **alternative phrasing**.',
        '- Rewrites per task type:',
        '  - `absolute_positioning`: "Move to any position {d}m from X" → "Navigate to a location {d} meters away from X"',
        '  - `delta_control`: "Move {d}m toward X" → "Approach X and stop when you are {d}m closer to it"',
        '  - `equidistance`: "… equidistant from A and B" → "… equally far from both A and B"',
        '  - `projective_relations`: "A appears to the left of B" → "B appears to the right of A" (semantically equivalent)',
        '  - `occlusion_alignment`: "A is hidden behind B" → "B fully blocks your view of A"',
        '  - `fov_inclusion`: "both A and B are visible" → "you can see both A and B at the same time"',
        '  - `centering`: "A is centered between B and C" → "A appears midway between B and C"',
        '- Original description stored in `task_description_original` field.',
        '- Tests: Does the model parse task instructions by template matching or semantic understanding?',
        '',
        '### Split-5: OOD Geometry (`ood_geometry.jsonl`)',
        '- Items with `distance` **outside [p10, p90]** of training distribution per task type.',
        '- Thresholds (computed from train_100scenes_7types.jsonl):',
    ]
    for tt, (p10, p90) in sorted(geom_thresholds.items()):
        lines.append(f'  - `{tt}`: p10={p10:.2f}m, p90={p90:.2f}m')
    lines += [
        '- Tests: Can the model accurately execute spatial tasks at unusual distances/scales?',
        '',
        '## Usage',
        '',
        '```bash',
        '# Run evaluation on a specific OOD split',
        '# (point your eval config to the split JSONL instead of the main val JSONL)',
        'python3 scripts/gen_ood_splits.py  # regenerate splits',
        '',
        '# Analyze OOD results after running evaluation',
        '# python3 scripts/analyze_experiments.py --exps <exp_name> --ood_eval',
        '```',
    ]
    with open(summary_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Generate 5-axis OOD generalization splits')
    parser.add_argument('--train_jsonl', default=
        'data_gen/active_spatial_pipeline/output_100scenes/train_100scenes_7types.jsonl',
        help='Path to training JSONL')
    parser.add_argument('--test_jsonl', default=
        'data_gen/active_spatial_pipeline/output_100scenes/test.jsonl',
        help='Path to test JSONL (source for OOD items)')
    parser.add_argument('--out_dir', default=
        'data_gen/active_spatial_pipeline/ood_splits',
        help='Output directory for OOD split JSONL files')
    parser.add_argument('--max_per_split', type=int, default=400,
        help='Max items per split (0 = no limit; default: 400)')
    parser.add_argument('--seed', type=int, default=42,
        help='Random seed for balanced sampling')
    args = parser.parse_args()

    random.seed(args.seed)
    max_items = args.max_per_split if args.max_per_split > 0 else 10_000_000

    # ── Load data ──
    print(f'Loading training data: {args.train_jsonl}')
    with open(args.train_jsonl) as f:
        train_items = [json.loads(l) for l in f if l.strip()]

    print(f'Loading test data: {args.test_jsonl}')
    with open(args.test_jsonl) as f:
        test_items = [json.loads(l) for l in f if l.strip()]

    train_scenes = set(i['scene_id'] for i in train_items)
    train_obj_labels = set(i['object_label'] for i in train_items)
    ood_scene_ids = set(i['scene_id'] for i in test_items) - train_scenes

    print(f'\nTraining: {len(train_items)} items, {len(train_scenes)} scenes, '
          f'{len(train_obj_labels)} unique object labels')
    print(f'Test: {len(test_items)} items, {len(set(i["scene_id"] for i in test_items))} scenes')
    print(f'OOD scenes ({len(ood_scene_ids)}): {sorted(ood_scene_ids)}')

    # ── Generate all 5 splits ──
    print('\n--- Generating OOD splits ---')

    print('\n[Split-1] OOD Scene ...')
    split_scene = make_ood_scene(test_items, train_scenes, max_items)
    print(f'  → {len(split_scene)} items | Tasks: {_task_dist(split_scene)}')

    print('\n[Split-2] OOD Instance ...')
    split_instance = make_ood_instance(test_items, train_scenes, train_obj_labels, max_items)
    print(f'  → {len(split_instance)} items | Tasks: {_task_dist(split_instance)}')

    print('\n[Split-3] OOD Category ...')
    split_category = make_ood_category(test_items, train_obj_labels, max_items)
    new_labels = set(i['object_label'] for i in split_category)
    print(f'  → {len(split_category)} items | {len(new_labels)} unique new labels | Tasks: {_task_dist(split_category)}')
    print(f'  Sample new labels: {sorted(new_labels)[:10]}')

    print('\n[Split-4] OOD Template ...')
    split_template = make_ood_template(test_items, max_items)
    # Verify rewriting quality
    unchanged = sum(1 for i in split_template
                    if i['task_description'] == i['task_description_original'])
    print(f'  → {len(split_template)} items | {len(split_template) - unchanged} descriptions rewritten | Tasks: {_task_dist(split_template)}')
    # Show 1 example per task type
    shown = set()
    for item in split_template:
        tt = item['task_type']
        if tt not in shown:
            shown.add(tt)
            orig = item['task_description_original']
            new = item['task_description']
            if orig != new:
                print(f'  {tt}: "{orig}" → "{new}"')

    print('\n[Split-5] OOD Geometry ...')
    split_geometry = make_ood_geometry(test_items, GEOM_THRESHOLDS, max_items)
    by_task_geom = defaultdict(list)
    for item in split_geometry:
        by_task_geom[item['task_type']].append(item['distance'])
    for tt, dists in sorted(by_task_geom.items()):
        p10, p90 = GEOM_THRESHOLDS.get(tt, (0, 999))
        print(f'  {tt}: n={len(dists)}, dist=[{min(dists):.2f},{max(dists):.2f}] (train p10={p10:.2f}, p90={p90:.2f})')

    # ── Write outputs ──
    os.makedirs(args.out_dir, exist_ok=True)
    splits_map = {
        'ood_scene':    split_scene,
        'ood_instance': split_instance,
        'ood_category': split_category,
        'ood_template': split_template,
        'ood_geometry': split_geometry,
    }
    for name, items in splits_map.items():
        path = os.path.join(args.out_dir, f'{name}.jsonl')
        _write_jsonl(path, items)
        print(f'\nWrote {len(items):4d} items → {path}')

    # ── Summary ──
    train_info = {
        'n_train': len(train_items),
        'n_scenes': len(train_scenes),
        'n_labels': len(train_obj_labels),
        'ood_scene_ids': ood_scene_ids,
    }
    summary_path = os.path.join(args.out_dir, 'ood_splits_summary.md')
    write_summary(summary_path, splits_map, train_info, GEOM_THRESHOLDS)
    print(f'\nWrote summary → {summary_path}')

    # ── Print Markdown table ──
    print('\n' + '='*70)
    print('OOD Split Summary')
    print('='*70)
    print(f'{"Split":<18} {"N":>5} {"Scenes":>7}  Task distribution')
    print('-'*70)
    for name, items in splits_map.items():
        n = len(items)
        n_scenes = len(set(i['scene_id'] for i in items))
        td = _format_task_dist(_task_dist(items))
        print(f'{name:<18} {n:>5} {n_scenes:>7}  {td}')


if __name__ == '__main__':
    main()
