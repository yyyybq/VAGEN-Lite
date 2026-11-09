#!/usr/bin/env python3
"""Freeze this exact24-image batch after explicit conversation acceptance.

This separate adapter preserves assistant disagreements and all automatic hard
gates. It does not modify the older review application/export utilities. It is
deliberately bound to one manifest and one quoted instruction, not a blanket
human-override switch. No rendering, models, jobs or training are invoked.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import importlib.util
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from PIL import Image

ROOT = Path('/mnt/umm/users/yinbaiqiao/VAGEN-Lite')
BATCH = ROOT / 'data_gen/active_spatial_qa/artifacts/production_v2_wallfix_20261013'
SOURCE = BATCH / 'reviewed/sample_manifest.jsonl'
SOURCE_SHA = '118b0a35e4ff6dad69fff4e8b5644e87437b4538fd2ffa50fc884dbc95b99f83'
USER_QUOTE = '24张都可以记录为放行并固定，然后继续增加'
REVIEWER = 'user (explicit conversation confirmation)'
SCHEMA = 'active_spatial_explicit_batch_acceptance_v2'
CONFIRMATION_SCHEMA = 'active_spatial_explicit_conversation_confirmation_v1'
sys.path.insert(0, str(ROOT))
from data_gen.active_spatial_qa import production_v2 as production
from data_gen.active_spatial_qa.production_contract_v1 import independent_check, question_variants
from data_gen.active_spatial_qa.contract import evaluate_state
from vagen.envs.active_spatial.canonical_camera import build_canonical_camera, CANONICAL_CAMERA_H1_RESIZE_V1


def sha(path):
    return production.sha(Path(path))


def qsha(question):
    return hashlib.sha256(question.encode()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def load_exporter():
    path = ROOT / 'data_gen/active_spatial_qa/qa_production_export_review.py'
    spec = importlib.util.spec_from_file_location('unchanged_qa_export_validation', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, {'path': str(path), 'sha256': sha(path)}


def validate_confirmation(document, rows, source):
    require(Path(source).resolve() == SOURCE.resolve(), 'wrong source manifest path')
    require(sha(source) == SOURCE_SHA, 'source manifest SHA mismatch')
    require(document.get('schema_version') == CONFIRMATION_SCHEMA, 'confirmation schema mismatch')
    require(document.get('source_manifest_sha256') == SOURCE_SHA, 'confirmation manifest SHA mismatch')
    require(document.get('source_manifest') == str(SOURCE), 'confirmation source path mismatch')
    require(document.get('user_quote') == USER_QUOTE, 'explicit user quote mismatch')
    require(document.get('reviewer') == REVIEWER, 'reviewer attribution mismatch')
    require(document.get('reviewer_kind') == 'human', 'confirmation is not human')
    require(document.get('scope') == 'all24_current_images_and_unchanged_questions_answers', 'confirmation scope mismatch')
    require(document.get('override_scope') == 'prior_assistant_target_identity_or_extent_opinions_only', 'override scope mismatch')
    stamp = datetime.fromisoformat(document['confirmation_recorded_at'].replace('Z', '+00:00'))
    require(stamp.utcoffset() is not None, 'confirmation recorded timestamp needs timezone')
    require('reviewed_at' not in document, 'do not invent a past review timestamp')
    source_index = {r['sample_id']: r for r in rows}
    records = document.get('records', [])
    index = {r['sample_id']: r for r in records}
    require(len(rows) == len(source_index) == len(records) == len(index) == 24, 'exact24 unique records required')
    require(set(source_index) == set(index), 'confirmation sample scope mismatch')
    for sid, row in source_index.items():
        c = index[sid]
        require(c.get('status') == 'accepted', sid + ': missing explicit acceptance')
        require(c.get('question_sha256') == qsha(row['question']), sid + ': question binding mismatch')
        require(c.get('image_sha256') == row['render']['sha256'], sid + ': image binding mismatch')
        require(c.get('scene_id') == row['scene_id'] and c.get('goal_id') == row['goal_id'], sid + ': goal identity mismatch')
        require(c.get('accepted_original_answer') == row['automatic_state_predicate_answer'], sid + ': original answer mismatch')
    return index


def validate_row(row, original, planned, cfg, ctx, exporter):
    """No semantic override is examined until immutable inputs and hard gates pass."""
    sid = row['sample_id']
    def check(condition, reason):
        require(condition, sid + ': ' + reason)
    for key in ('sample_id', 'scene_id', 'split', 'goal_id', 'task_type', 'goal',
                'state_pose_c2w', 'source_init_camera', 'source_camera_metadata',
                'source_task_metric_version', 'source_camera_model_version',
                'question', 'question_variants', 'automatic_state_predicate_answer',
                'render', 'metrics', 'geometry', 'visibility_admission'):
        check(row.get(key) == original.get(key), 'source field changed: ' + key)
    check(row['question'] == question_variants(row['goal'])[0], 'original question contract mismatch')
    check(row['split'] == cfg['scene_splits'][row['scene_id']], 'scene split changed')
    check(row['state_pose_c2w'] == planned['state_pose_c2w'] and row['goal'] == planned['goal'], 'render plan binding mismatch')
    render = row['render']
    image_path = Path(row['public_observation']['image_path']).resolve(strict=True)
    check(image_path == Path(render['image_path']).resolve(), 'RGB/render path mismatch')
    check(render.get('sample_id') == sid and render.get('ok') is True, 'render success missing')
    status = json.loads(image_path.with_name('status.json').read_text())
    check(status == render, 'saved render status changed')
    identity = {'record': production.digest(planned), 'assets': row['provenance']['asset_identity'], 'code': production.code_identity()}
    check(render['request_identity'] == identity and render['identity'] == production.digest(identity), 'render request identity mismatch')
    check(sha(image_path) == render['sha256'], 'actual image SHA mismatch')
    with Image.open(image_path) as image:
        image.load()
        check(image.size == (256, 256), 'RGB dimensions changed')
        rgb_stats = production.image_stats(image)
    check(rgb_stats['nonblank'] is True and render['stats']['nonblank'] is True, 'RGB nonblank gate failed')
    check(row['geometry']['status'] == 'legal_indoor' and not row['geometry']['reasons'], 'stored legal geometry failed')
    pose = np.asarray(row['state_pose_c2w'], dtype=float)
    legal = production.legal(cfg, ctx, pose[:3, 3])
    check(legal['status'] == 'legal_indoor' and not legal['reasons'], 'recomputed indoor/collision gate failed')
    visibility = production.visibility_gate(cfg, ctx, row['goal'], pose[:3, 3])
    check(visibility['ok'] is True and row['visibility_admission']['ok'] is True and render['visibility_admission']['ok'] is True,
          'structural/furniture geometric visibility gate failed')
    asset_objects = {str(o['id']): o for o in ctx[3]}
    for obj in row['goal']['target_object']['objects']:
        target = asset_objects.get(str(obj['id']))
        # Frozen source tasks lowercase some asset class names (e.g. Projector).
        # Case normalization does not rename the object or change its scope.
        check(target is not None and target['label'].casefold() == obj['label'].casefold(), 'asset object identity mismatch')
        check(all(np.allclose(obj[k], target[k], atol=1e-5, rtol=0) for k in ('center','bbox_min','bbox_max')), 'asset object geometry mismatch')
    independent = independent_check(row['goal'], pose)
    active = evaluate_state(row['goal'], pose)
    check(independent['admission']['geometry_admissible'] is True, 'independent geometry admission failed')
    native = row['source_init_camera']['intrinsics']
    render_item = {**row['source_camera_metadata'], 'init_camera':row['source_init_camera']}
    camera = build_canonical_camera(K_native=native, render_size=(256,256), transform='resize', c2w=pose,
                                    item=render_item, camera_model_version=CANONICAL_CAMERA_H1_RESIZE_V1)
    check(render['camera']['width'] == render['camera']['height'] == 256, 'request dimensions changed')
    check(np.array_equal(render['camera']['c2w'], pose), 'render c2w mismatch')
    check(np.array_equal(render['camera']['K_effective'], camera.K_effective) and
          np.array_equal(render['camera']['w2c'], camera.w2c), 'actual render camera mismatch')
    check(np.array_equal(independent['camera']['K_effective'], camera.K_effective) and
          np.array_equal(independent['camera']['w2c'], camera.w2c), 'independent camera mismatch')
    answer = row['automatic_state_predicate_answer']
    check(answer in {'Yes','No'}, 'original automatic answer absent')
    desired = answer == 'Yes'
    for name, result in [('recomputed_active',active), ('recomputed_independent',independent),
                         ('stored_active',row['metrics']['active']), ('stored_independent',row['metrics']['independent'])]:
        check(type(result['success']) is bool and result['success'] == desired, name + ' answer disagreement')
    check(row['private_answer'] in (None, answer), 'existing answer differs from original predicate')
    assistant = row.get('assistant_review',{})
    check(assistant.get('image_sha256') == render['sha256'] and assistant.get('question_sha256') == qsha(row['question']),
          'assistant opinion source binding changed')
    check(assistant.get('reviewer_kind') == 'assistant', 'prior opinion reviewer kind changed')
    check(assistant.get('status') == 'passed' or
          (assistant.get('status') == 'rejected' and assistant.get('category') == '目标不明确'),
          'user scope cannot override this kind of assistant rejection')
    proposed = copy.deepcopy(row)
    proposed.update(private_answer=answer, label_validity='valid', observability_validity='valid', candidate_qa_allowed=True)
    # Preserve the actual rejected assistant opinion. Only this specifically
    # named subjective gate is superseded by the explicit human confirmation.
    remaining = [reason for reason in exporter.candidate_reasons(proposed) if reason != 'assistant_review_not_passed']
    check(not remaining, 'other original candidate gates failed: ' + repr(remaining))
    return {'sample_id':sid, 'image_sha256':render['sha256'], 'question_sha256':qsha(row['question']),
            'original_automatic_answer':answer, 'hard_checks_passed':True,
            'fresh_indoor_geometry':legal, 'fresh_geometric_visibility_ok':visibility['ok'],
            'rgb_stats':rgb_stats, 'recomputed_active_success':active['success'],
            'recomputed_independent_success':independent['success'], 'effective_camera':camera.to_dict(),
            'previous_assistant_status':assistant['status'], 'previous_assistant_category':assistant.get('category'),
            'hard_gate_override_used':False}


def apply(source, confirmation_file, output):
    source, confirmation_file, output = Path(source).resolve(strict=True), Path(confirmation_file).resolve(strict=True), Path(output).resolve()
    require(not output.exists(), 'refusing to overwrite existing output directory')
    rows = read_rows(source)
    confirmation = json.loads(confirmation_file.read_text())
    confirmations = validate_confirmation(confirmation, rows, source)
    confirmation_sha = sha(confirmation_file)
    cfg = json.loads((BATCH/'distribution_config.json').read_text())
    raw = {r['sample_id']:r for r in read_rows(BATCH/'sample_manifest.jsonl')}
    plan = {r['sample_id']:r for r in read_rows(BATCH/'planned_states.jsonl')}
    require(set(raw) == set(plan) == {r['sample_id'] for r in rows}, 'raw/planned/source scope mismatch')
    exporter, exporter_identity = load_exporter()
    contexts, asset_audit = {}, {}
    for row in rows:
        scene = row['scene_id']
        if scene not in contexts:
            checks = {}
            for name, expected in row['provenance']['asset_identity']['files'].items():
                p = Path(cfg['scene_root']) / scene / name
                require(sha(p) == expected, scene + ': current asset SHA mismatch: ' + name)
                checks[name] = {'path':str(p), 'sha256':expected}
            asset_audit[scene] = checks
            contexts[scene] = production.context(cfg, scene)
    checks = [validate_row(r, raw[r['sample_id']], plan[r['sample_id']], cfg, contexts[r['scene_id']], exporter) for r in rows]
    require(len({r['render']['sha256'] for r in rows}) == 24, 'duplicate RGB inputs')
    result, datasets, index = [], {s:[] for s in ('train','val','test')}, []
    for row, checked in zip(rows, checks):
        r = copy.deepcopy(row); previous = r['assistant_review']; disagreement = previous['status'] == 'rejected'
        human = {**confirmations[r['sample_id']], 'status':'accepted', 'reviewer_kind':'human', 'reviewer_type':'human',
                 'reviewer':REVIEWER, 'reason':USER_QUOTE, 'confirmation_recorded_at':confirmation['confirmation_recorded_at'],
                 'confirmation_method':'explicit conversation instruction covering the exact24-image batch',
                 'reviewed_at':confirmation['confirmation_recorded_at'],
                 'time_basis':'explicit chat confirmation recorded', 'actual_image_view_time':None,
                 'display_status':'用户已确认', 'individual_ui_click_recorded':False}
        # reviewed_at is a UI compatibility alias for recording time, explicitly
        # not an invented time at which the user actually viewed this image.
        r['human_review'] = human
        r['previous_review_state'] = {k:copy.deepcopy(row.get(k)) for k in
                                     ('private_answer','label_validity','observability_validity','observability_evidence','candidate_qa_allowed','qa_sft_allowed','semantic_review')}
        r.update(private_answer=row['automatic_state_predicate_answer'], label_validity='valid', observability_validity='valid',
                 candidate_qa_allowed=True, qa_sft_allowed=True,
                 observability_evidence='explicit_user_acceptance_after_review; assistant_disagreements_preserved; automatic_hard_gates_rechecked')
        r['semantic_review'] = {'status':'PASS','reviewer_type':'human','reviewer':REVIEWER,'reason':USER_QUOTE,
                                'confirmation_recorded_at':confirmation['confirmation_recorded_at']}
        r['assistant_human_disagreement'] = disagreement
        r['review_disagreements'] = [copy.deepcopy(previous)] if disagreement else []
        r['human_acceptance_application'] = {'schema_version':SCHEMA, 'source_manifest':str(source),
            'source_manifest_sha256':SOURCE_SHA, 'confirmation_file':str(output/'user_acceptance.json'),
            'confirmation_file_sha256':confirmation_sha, 'reviewed_image_sha256':checked['image_sha256'],
            'reviewed_question_sha256':checked['question_sha256'], 'original_automatic_answer':checked['original_automatic_answer'],
            'override_applied':disagreement, 'override_scope':'prior_assistant_target_identity_or_extent_opinions_only' if disagreement else None,
            'automatic_hard_checks_passed':True, 'hard_gate_override_used':False,
            'adapter_sha256':sha(__file__)}
        require(r['assistant_review'] == row['assistant_review'], 'assistant opinion was modified')
        record = {'messages':[{'role':'user','content':'<image>\n'+r['question']},
                              {'role':'assistant','content':r['private_answer']}],
                  'images':[str(Path(r['public_observation']['image_path']).resolve())]}
        exporter.validate_messages(record)
        line = len(datasets[r['split']]); datasets[r['split']].append(record)
        index.append({'sample_id':r['sample_id'],'split':r['split'],'line_zero_based':line,
                      'image_sha256':checked['image_sha256'],'question_sha256':checked['question_sha256'],
                      'original_answer':r['private_answer'],'assistant_human_disagreement':disagreement})
        result.append(r)
    require(sha(source) == SOURCE_SHA and sha(confirmation_file) == confirmation_sha, 'inputs changed during validation')
    for r in rows:
        require(sha(r['public_observation']['image_path']) == r['render']['sha256'], 'image changed during validation')
    def json_text(v):return json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    def jsonl_text(v):return ''.join(json.dumps(r,ensure_ascii=False,allow_nan=False)+'\n' for r in v)
    manifest_text = jsonl_text(result)
    payloads = {'source_manifest.jsonl':source.read_bytes(), 'user_acceptance.json':confirmation_file.read_bytes(),
                'sample_manifest.jsonl':manifest_text.encode(), 'hard_check_results.json':json_text(checks).encode(),
                'export_index.jsonl':jsonl_text(index).encode()}
    for split, records in datasets.items():payloads['approved_qa/'+split+'.jsonl']=jsonl_text(records).encode()
    payloads['approved_qa/all.jsonl']=jsonl_text([r for split in datasets for r in datasets[split]]).encode()
    summary = {'schema_version':SCHEMA,'source_manifest':str(source),'source_manifest_sha256':SOURCE_SHA,
               'output_manifest':str(output/'sample_manifest.jsonl'),
               'output_manifest_sha256':hashlib.sha256(manifest_text.encode()).hexdigest(),
               'confirmation_recorded_at':confirmation['confirmation_recorded_at'], 'user_quote':USER_QUOTE,'reviewer':REVIEWER,
               'human_accepted':24,'approved_qa':24,'preserved_assistant_disagreements':sum(r['assistant_human_disagreement'] for r in result),
               'hard_checks_passed':len(checks),'split_counts':{k:len(v) for k,v in datasets.items()},
               'answer_counts':dict(Counter(r['private_answer'] for r in result)),
               'adapter':{'path':str(Path(__file__).resolve()),'sha256':sha(__file__)},
               'unchanged_exporter_validation':exporter_identity, 'current_asset_checks':asset_audit,
               'bound_inputs':{str(p):sha(p) for p in (BATCH/'sample_manifest.jsonl',BATCH/'planned_states.jsonl',BATCH/'distribution_config.json')},
               'export_files_sha256':{k:hashlib.sha256(v).hexdigest() for k,v in payloads.items()},
               'do_train':False,'no_model_loaded':True,'no_rendering':True,'no_jobs_submitted':True,
               'notes':['Explicit user acceptance supersedes10 earlier assistant target-identity/range opinions; those opinions remain unchanged.',
                        'No target, question, predicate or answer was changed. The10 withheld private answers were restored from the original automatic predicate.',
                        'Human approval is eligibility, not a claim of balanced training coverage; this tiny batch has only4 train examples, all FOV.',
                        'confirmation_recorded_at is the recording time; no actual prior viewing timestamp is invented.',
                        'This versioned adapter is required; older exporters intentionally retain their assistant-pass gate.']}
    payloads['release_audit.json']=json_text(summary).encode()
    output.mkdir(parents=True,exist_ok=False)
    for name, data in payloads.items():
        path=output/name;path.parent.mkdir(parents=True,exist_ok=True)
        with path.open('xb') as stream:stream.write(data)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest',required=True);parser.add_argument('--confirmation',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();summary=apply(args.manifest,args.confirmation,args.output)
    print(json.dumps({k:summary[k] for k in ('output_manifest','human_accepted','approved_qa','preserved_assistant_disagreements','hard_checks_passed','split_counts','do_train')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
