"""Summarize actual production stages without counting rewrites as observations."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def load_rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()] if path.exists() else []


def count(values):
    return dict(sorted(Counter(values).items()))


def summarize(rows, stage):
    objects = [(r['scene_id'], str(o['id']), o['label']) for r in rows for o in r['goal']['target_object']['objects']]
    pairs = [(r['scene_id'], tuple(sorted(str(o['id']) for o in r['goal']['target_object']['objects']))) for r in rows]
    image_paths = {r.get('public_observation', {}).get('image_path') for r in rows} - {None}
    images = {hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in image_paths}
    is_plan = stage == 'planned_legal_states'
    answer_key = 'branch_intent' if is_plan else 'private_answer'
    return {
        'records': len(rows),
        'independent_pose_count': len({r['scene_id'] + json.dumps(r['state_pose_c2w']) for r in rows}),
        'independent_visual_inputs': len(images),
        'history_inputs': sum(bool(r.get('public_observation', {}).get('history')) for r in rows),
        'qa_rows': 0 if is_plan else sum(bool(r.get('question')) for r in rows),
        'scene_count': len({r['scene_id'] for r in rows}),
        'category_count': len({o[2] for o in objects}),
        'instance_count': len({(o[0], o[1]) for o in objects}),
        'object_pair_count': len(set(pairs)),
        'goal_count': len({r['goal_id'] for r in rows}),
        'answer_scope': 'sampling_intent_not_labels' if is_plan else 'state_labels',
        'by_answer': count(str(r.get(answer_key)) for r in rows),
        'by_task_answer': count(r['task_type'] + '/' + str(r.get(answer_key)) for r in rows),
        'by_scene': count(r['scene_id'] for r in rows),
        'by_scene_task_answer': count('/'.join([r['scene_id'], r['task_type'], str(r.get(answer_key))]) for r in rows),
        'by_category_mentions': count(o[2] for o in objects),
        'by_instance_mentions': count(o[0] + '/' + o[1] + '/' + o[2] for o in objects),
        'by_pair': count(s + '/' + ','.join(p) for s, p in pairs),
        'by_difficulty': count(r['difficulty'] for r in rows),
        'by_split': count(r['split'] for r in rows),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--reviewed-manifest')
    p.add_argument('--output', required=True)
    a = p.parse_args()
    root = Path(a.root)
    planned = load_rows(root / 'planned_states.jsonl')
    automatic = load_rows(root / 'sample_manifest.jsonl')
    reviewed = load_rows(Path(a.reviewed_manifest)) if a.reviewed_manifest else []
    passed = [r for r in reviewed if r.get('candidate_qa_allowed') and r.get('assistant_review', {}).get('status') == 'passed']
    approved = [r for r in passed if r.get('qa_sft_allowed') and r.get('semantic_review', {}).get('status') == 'passed']
    stages = {'planned_legal_states': planned, 'automatic_rgb_geometry': automatic, 'assistant_visual_pass': passed, 'human_released': approved}
    failures = load_rows(root / 'all_failures.jsonl') or load_rows(root / 'failures.jsonl')
    result = {
        'stages': {k: summarize(v, k) for k, v in stages.items()},
        'assistant_review_statuses': count(r.get('assistant_review', {}).get('status', 'pending') for r in reviewed),
        'assistant_rejection_categories': count(r['assistant_review'].get('category', 'unspecified') for r in reviewed if r.get('assistant_review', {}).get('status') == 'rejected'),
        'generation_failure_records': len(failures),
        'generation_failure_reasons': count(json.dumps(r.get('reason'), ensure_ascii=False, sort_keys=True) for r in failures),
        'planning_statistics': json.loads((root / 'planning_statistics.json').read_text()),
        'counting_notes': [
            'Only existing image bytes with distinct SHA256 count as visual inputs; planned poses are not rendered observations.',
            'Question rewrites are not additional visual inputs. Category counts are unique; category histogram counts both target mentions.',
            'Geometric proposals before planning selection have aggregate rejection counts only; no full scene/object distribution was persisted for that earlier stage.',
            'Automatic checks and assistant visual review do not constitute human approval. Approved training data may be empty.'
        ],
    }
    target = Path(a.output)
    if target.exists():
        raise ValueError('Use a new versioned statistics destination')
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: {f: v[f] for f in ['records', 'independent_visual_inputs', 'scene_count', 'by_task_answer']} for k, v in result['stages'].items()}, ensure_ascii=False))


if __name__ == '__main__':
    main()
