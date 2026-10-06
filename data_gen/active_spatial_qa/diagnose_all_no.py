#!/usr/bin/env python3
"""Read-only audit of frozen QA results; no model loading or rendering."""
import argparse
import csv
import hashlib
import html
import itertools
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from data_gen.active_spatial_qa.qa_eval import evaluate, parse_answer


def read_rows(path):
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def independent_geometry(row):
    """Direct eight-corner pinhole projection, without runtime scoring imports."""
    goal = row['complete_goal']
    obj = goal['target_object']
    lo, hi = obj['bbox_min'], obj['bbox_max']
    corners = np.array(list(itertools.product(*zip(lo, hi))))
    pose = np.array(row['state_pose_c2w'])
    K = np.array(row['camera']['intrinsics'])
    camera = (corners - pose[:3, 3]) @ np.linalg.inv(pose[:3, :3]).T
    front = camera[:, 2] > 1e-6
    width, height = row['render']['resolution']
    params = goal['target_region']['params']
    target = params['occupancy_ratio']
    bbox = None
    occ = 0.0
    if front.any():
        uv = camera[front] @ K.T
        uv = uv[:, :2] / uv[:, 2:3]
        lower, upper = uv.min(axis=0), uv.max(axis=0)
        bbox = [max(0, float(lower[0])), max(0, float(lower[1])),
                min(width - 1, float(upper[0])), min(height - 1, float(upper[1]))]
        occ = max(0, bbox[3] - bbox[1]) / height
    distance = np.linalg.norm(pose[:2, 3] - np.array(params['object_center'])[:2])
    angular = 2 * math.atan(params['object_height'] / (2 * distance)) / math.radians(params['fov_vertical'])
    render_pose = np.array(row['render']['pose_c2w'])
    inverse = np.array(row['render']['extrinsics_w2c'])
    return {'bbox': bbox, 'pixel_height_fraction': occ, 'angular_fov_fraction': angular,
            'target_fraction': target, 'target_percent': target * 100,
            'fov_from_K_vertical_deg': math.degrees(math.atan(K[1, 2] / K[1, 1]) + math.atan((height - K[1, 2]) / K[1, 1])),
            'pose_max_error': float(abs(render_pose - pose).max()),
            'inverse_max_error': float(abs(inverse @ render_pose - np.eye(4)).max()),
            'all_corners_in_front': bool(front.all()),
            'camera_y_world_z': float(pose[2, 1]),
            'pixel_vs_audit_error': abs(occ - row['_audit']['predicate']['details']['visual_bbox_metrics']['visual_occupancy'])}


def processor_trace(row, checkpoint):
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True, local_files_only=True)
    path = Path(row['public_observation']['image_path'])
    image = Image.open(path).convert('RGB')
    messages = [{'role': 'user', 'content': [{'type': 'image', 'image': image}, {'type': 'text', 'text': row['question']}]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[image], padding=True, return_tensors='pt')
    tensors = {}
    for key, value in inputs.items():
        info = {'shape': list(value.shape), 'dtype': str(value.dtype)}
        if key == 'image_grid_thw':
            info['values'] = value.tolist()
        if key == 'pixel_values':
            raw = value.contiguous().numpy().tobytes()
            info.update(sha256=hashlib.sha256(raw).hexdigest(), min=float(value.min()), max=float(value.max()), mean=float(value.float().mean()), std=float(value.float().std()))
        tensors[key] = info
    return {'evidence_type': 'CPU reconstruction of historical HF path, not captured historical forward',
            'checkpoint': checkpoint, 'sample_id': row['sample_id'], 'image_path': str(path), 'image_sha256': digest(path),
            'chat_template': prompt, 'decoded_input': processor.tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=False),
            'image_token_count': int((inputs['input_ids'] == processor.tokenizer.convert_tokens_to_ids('<|image_pad|>')).sum()),
            'processor_class': type(processor.image_processor).__name__, 'tensors': tensors}


def run(args):
    source, out = Path(args.source).resolve(), Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows = read_rows(source / 'frozen_visual_manifest.jsonl')
    eligible = json.loads((source / 'eligible_ids.json').read_text())['eligible_ids']
    models = {name: read_rows(source / 'eval' / f'{name}_predictions.jsonl') for name in ('base', 'act')}
    indices = {name: {r['sample_id']: r for r in records} for name, records in models.items()}
    ids = [r['sample_id'] for r in rows]
    summary = {'n': len(rows), 'ids_match_order': ids == eligible, 'task_types': dict(Counter(r['task_type'] for r in rows)),
               'labels': dict(Counter(r['private_answer'] for r in rows)), 'models': {}, 'source_hashes': {str(p): digest(p) for p in (source/'frozen_visual_manifest.jsonl', source/'eligible_ids.json')}}
    for name, records in models.items():
        parsed = Counter(parse_answer(r.get('prediction', '')) for r in records)
        confusion = Counter((r['private_answer'], parse_answer(indices[name].get(r['sample_id'], {}).get('prediction', ''))) for r in rows)
        recomputed = evaluate(str(source/'frozen_visual_manifest.jsonl'), str(source/'eval'/f'{name}_predictions.jsonl'), str(out/f'{name}_qa_eval_recomputed.json'), True)
        old = json.loads((source/'eval'/f'{name}_qa_eval.json').read_text())
        summary['models'][name] = {'raw_answers': dict(Counter(r.get('prediction', '') for r in records)), 'parsed_answers': dict(parsed),
            'missing_ids': sorted(set(ids)-set(indices[name])), 'extra_ids': sorted(set(indices[name])-set(ids)),
            'duplicate_ids': len(records)-len(indices[name]), 'model_identities': dict(Counter(r['model_identity'] for r in records)),
            'confusion_GT_prediction': {str(k): v for k, v in confusion.items()}, 'qa_eval_matches_recomputation': old == recomputed,
            'file_sha256': digest(source/'eval'/f'{name}_predictions.jsonl')}
    summary['constant_baselines'] = {k: summary['labels'].get(k, 0)/len(rows) for k in ('Yes', 'No')}
    audit, panels, sections = [], [], []
    ordered = sorted(rows, key=lambda r: (r['private_answer'] != 'Yes', r['parent_goal_id'], r['state_id']))
    for row in ordered:
        sid = row['sample_id']
        path = Path(row['public_observation']['image_path'])
        geo = independent_geometry(row)
        image = Image.open(path).convert('RGB')
        pixels = np.asarray(image, dtype=float)
        spatial_std = float(pixels.std(axis=(0, 1)).max())
        source_path = Path(row['source_manifest'])
        if not source_path.exists():
            source_path = Path(__file__).resolve().parents[2] / 'data_gen/active_spatial_pipeline/output_100scenes/scenes' / row['scene_id'] / 'data.jsonl'
        original = read_rows(source_path)[row['parent_source_line']] if source_path.exists() else None
        provenance = {'resolved_source': str(source_path), 'source_line': row['parent_source_line'],
                      'original_fields_match': all(original.get(key) == row['complete_goal'].get(key) for key in ('target_object','target_region','task_description','preset','distance','task_params') if key in original) if original else None,
                      'intrinsics_match': original['init_camera']['intrinsics'] == row['camera']['intrinsics'] if original else None}
        predicate = row['_audit']['predicate']
        rec = {'sample_id': sid, 'parent_goal_id': row['parent_goal_id'], 'state_id': row['state_id'], 'scene_id': row['scene_id'],
            'image_path': str(path), 'image_sha256': digest(path), 'image_size': list(image.size), 'question': row['question'],
            'target_object': row['complete_goal']['target_object']['label'], 'target_condition': row['complete_goal']['task_description'],
            'GT': row['private_answer'], 'base_raw': indices['base'][sid]['prediction'], 'act_raw': indices['act'][sid]['prediction'],
            'image_spatial_channel_std_max': spatial_std, 'flat_image': spatial_std < 5, 'source_provenance': provenance,
            'predicate_version': row['success_predicate_version'], 'score': predicate['score'], 'threshold': predicate['threshold'],
            'geometry': geo, 'predicate': predicate,
            'public_judgment': 'UNDETERMINABLE: precise projected 3D bbox, visibility proxy, tolerance and weighted threshold are not specified in public prompt',
            'raw_label_threshold_consistent': (predicate['score'] >= predicate['threshold']) == (row['private_answer'] == 'Yes')}
        visible_positive_ids = {'ed9440ce5818ebcae771','8d33ee92db1dc5f458e9','6ec70dad4820425d40d6','afefafa5cec7959a4602','d0bcf7a8328874194f02'}
        if sid in visible_positive_ids:
            visual_note = 'Manual image review: pink wardrobe doors are discernible; independent projected outline aligns with the visible door rectangle. Image is upside down. Exact public tolerance/occupancy definition is missing.'
        elif sid == '255f8e714bcb656b93e6':
            visual_note = 'Manual image review: fragmented GS artefacts on black background; no identifiable wardrobe inside projected outline. Geometric visibility is not RGB visibility. GT Yes is not verifiable from public evidence.'
        elif spatial_std < 5:
            visual_note = 'Near-uniform frame: spatial channel standard deviation <5; legacy flattened RGB variance passed because channels have different means. Public target evidence is insufficient.'
        elif sid == 'd09bbbaf17f93fc57802':
            visual_note = 'Manual image review: large pink wardrobe doors are discernible, upside down. Pixel bbox fraction 0.576 differs from requested 0.500, but public tolerance is absent.'
        else:
            visual_note = 'Manual contact-sheet review: cropped/blurred/fragmented room or empty wall; exact designated wardrobe boundary is not reliably recoverable. Precise conjunction cannot be certified from public evidence.'
        rec['visual_review_note'] = visual_note
        rec['qa_sft_eligible_v2'] = False
        rec['review_status_v2'] = 'UNDETERMINABLE'
        audit.append(rec)
        thumb = image.resize((256, 256))
        panel = Image.new('RGB', (320, 330), 'white')
        panel.paste(thumb, (32, 58))
        d = ImageDraw.Draw(panel)
        d.text((5, 5), f"{sid} GT={rec['GT']}", fill='black')
        d.text((5, 20), f"{rec['target_object']} target={geo['target_percent']:.0f}% {rec['state_id']}", fill='black')
        d.text((5, 35), f"pixel={geo['pixel_height_fraction']:.3f} angular={geo['angular_fov_fraction']:.3f}", fill='black')
        d.text((5, 315), f"Base={rec['base_raw']} Act={rec['act_raw']}", fill='black')
        panels.append(panel)
        annotated = image.copy()
        if geo['bbox'] and geo['bbox'][2] >= geo['bbox'][0] and geo['bbox'][3] >= geo['bbox'][1]:
            ImageDraw.Draw(annotated).rectangle(geo['bbox'], outline='red', width=2)
        (out/'overlays').mkdir(exist_ok=True)
        annotated.save(out/'overlays'/f'{sid}.png')
        # Public input images are linked unchanged. The red outline is diagnostic geometry, never a model input.
        rel = __import__('os').path.relpath(path, out)
        sections.append(f'<section id="{sid}"><h2>{sid} | GT={rec["GT"]} | {html.escape(rec["target_object"])}</h2><div class="images"><img src="{html.escape(rel)}"><img src="overlays/{sid}.png"></div><p>{html.escape(row["question"])}</p><pre>{html.escape(json.dumps(rec, indent=2))}</pre></section>')
    sheet = Image.new('RGB', (320*3, 330*math.ceil(len(panels)/3)), 'white')
    for i, panel in enumerate(panels):
        sheet.paste(panel, ((i%3)*320, (i//3)*330))
    sheet.save(out/'contact_sheet.png')
    (out/'review.html').write_text('<!doctype html><meta charset="utf-8"><title>QA Sample Audit</title><style>body{font:14px sans-serif;margin:24px}section{border-top:1px solid #aaa;padding:12px 0}img{width: min(45vw,512px);height:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere}.images{display:flex;gap:12px}</style><h1>Frozen QA Review: Yes First</h1>' + ''.join(sections))
    (out/'sample_audit.json').write_text(json.dumps(audit, indent=2)+'\n')
    fields = ['sample_id','parent_goal_id','state_id','scene_id','target_object','target_condition','GT','base_raw','act_raw','score','threshold','image_path','image_sha256','question','public_judgment','visual_review_note','review_status_v2','qa_sft_eligible_v2']
    with (out/'sample_review.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore'); writer.writeheader(); writer.writerows(audit)
    summary['geometry'] = {'max_pixel_vs_audit_error': max(r['geometry']['pixel_vs_audit_error'] for r in audit),
        'max_pose_error': max(r['geometry']['pose_max_error'] for r in audit), 'max_inverse_error': max(r['geometry']['inverse_max_error'] for r in audit),
        'angular_vs_pixel_label_issues': [{'id': r['sample_id'], 'GT': r['GT'], **r['geometry']} for r in audit if r['GT']=='Yes'],
        'all_label_threshold_consistent': all(r['raw_label_threshold_consistent'] for r in audit)}
    summary['source_provenance_all_match'] = all(r['source_provenance']['original_fields_match'] and r['source_provenance']['intrinsics_match'] for r in audit)
    summary['flat_images'] = [r['sample_id'] for r in audit if r['flat_image']]
    summary['exact_image_duplicates'] = {h: [r['sample_id'] for r in audit if r['image_sha256'] == h] for h, n in Counter(r['image_sha256'] for r in audit).items() if n > 1}
    (out/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    # Preserve all rows and original labels; quarantine is a review decision, not a relabel/drop operation.
    with (out/'review_manifest_v2.jsonl').open('w') as f:
        for row in rows:
            decision = next(r for r in audit if r['sample_id'] == row['sample_id'])
            f.write(json.dumps(dict(row, qa_sft_eligible_v2=False, review_status_v2='UNDETERMINABLE', review_reason_v2=decision['visual_review_note']))+'\n')
    print(json.dumps(summary, indent=2))
    if args.processor:
        for name, records in models.items():
            checkpoint = records[0]['model_identity']
            # All Yes cases plus a same-parent No cover every positive target.
            selected = [r for r in ordered if r['private_answer']=='Yes']
            for parent in sorted(set(r['parent_goal_id'] for r in selected)):
                negatives = [r for r in ordered if r['parent_goal_id']==parent and r['private_answer']=='No']
                if negatives:
                    selected.append(negatives[0])
            traces = [processor_trace(row, checkpoint) for row in selected]
            (out/f'{name}_input_reconstruction.json').write_text(json.dumps(traces, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--processor', action='store_true')
    run(parser.parse_args())
