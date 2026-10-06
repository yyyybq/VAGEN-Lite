"""Finalize inspected v6 RGB evidence, preserving all historical records."""
import argparse
import copy
import html
import json
import math
import os
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from .position_evidence import ROOT,read_rows
from .generate_paired_qa import _load_scene_context
from .image_contract_v2 import classify_scene_pose
from .contract import evaluate_state
from .finalize_act2qa import freeze
from .position_render_run import validate_outputs


def _json_default(value):
    """Serialize numpy scalar/array values without weakening type checks."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f'not JSON serializable: {type(value).__name__}')


def main(out):
    results=json.loads((out/'render_results.json').read_text())
    for state in results.values():
        if not validate_outputs(state): raise RuntimeError('RGB artifact hashes/decode invalid')
    evidence=json.loads((out/'render_evidence.json').read_text())
    normal={'A_upstream_init_0003','6ec70dad4820425d40d6_old','6ec70dad4820425d40d6_fixed',
            'ed9440ce5818ebcae771_old','ed9440ce5818ebcae771_fixed','translation_A_upstream_init_0003','translation_6ec70dad4820425d40d6'}
    fragments={'7c2e750524f6c2e2010e_old','7c2e750524f6c2e2010e_fixed','b1c76b6e9659e3310dbd_old','b1c76b6e9659e3310dbd_fixed',
               '255f8e714bcb656b93e6_old','255f8e714bcb656b93e6_fixed'}
    sections=[]
    for row in evidence:
        sid=row['sample_id']; legal=row['classification']['status']=='legal_indoor'
        upright=not sid.endswith('_old')
        row['manual_review']={'normal_room_RGB':sid in normal,'target_wardrobe_identifiable':sid in normal,
                              'upright':upright if sid in normal else None,
                              'artifact_type':'fragmented_GS' if sid in fragments else 'normal_room' if sid in normal else 'empty_black_frame',
                              'task_label_released':False,
                              'reason':'angular/pixel contract remains unresolved' if legal else 'illegal camera, never a training No'}
        image=os.path.relpath(row['artifact']['path'],out)
        sections.append(f'<section><h2>{html.escape(sid)}</h2><img src="{html.escape(image)}"><pre>{html.escape(json.dumps(row,indent=2))}</pre></section>')
    (out/'manual_render_review.json').write_text(json.dumps(evidence,indent=2)+'\n')
    (out/'review.html').write_text('<!doctype html><meta charset="utf-8"><title>Position Repair A/B/C</title><style>body{font:14px sans-serif;margin:24px}section{border-top:1px solid #aaa;padding:12px}img{max-width:512px;width:100%}pre{white-space:pre-wrap;overflow-wrap:anywhere}</style><h1>Real A/B/C RGB and Scene Geometry</h1>'+''.join(sections))
    pairs=[]
    for row in evidence:
        if row['sample_id'].endswith('_old'):
            sid=row['sample_id'][:-4]
            fixed=next(r for r in evidence if r['sample_id']==sid+'_fixed')
            panel=Image.new('RGB',(1024,550),'white'); d=ImageDraw.Draw(panel)
            d.text((5,5),sid+' | old pose axes',fill='black'); d.text((517,5),'fixed pose axes: SAME center, forward, K and dimensions',fill='black')
            for x,rec in [(0,row),(512,fixed)]: panel.paste(Image.open(rec['artifact']['path']).convert('RGB'),(x,38))
            pairs.append(panel)
    sheet=Image.new('RGB',(1024,550*len(pairs)),'white')
    for i,p in enumerate(pairs): sheet.paste(p,(0,550*i))
    sheet.save(out/'orientation_before_after.png')
    old=read_rows(ROOT/'data_gen/active_spatial_qa/artifacts/qa_v2_runtime_canary_20260914/oracle_bank/manifest.jsonl')
    template=old[0]; goal=template['complete_goal']
    item={k:goal[k] for k in ('target_object','target_region','task_params','task_description','preset','distance')}
    item.update(task_type=template['task_type'],init_camera={'intrinsics':template['camera']['intrinsics']})
    context=_load_scene_context(str(out/'assets'),'0003_839989')
    candidates=[]
    for sid in ('A_upstream_init_0003','6ec70dad4820425d40d6_fixed','ed9440ce5818ebcae771_fixed'):
        ev=next(r for r in evidence if r['sample_id']==sid)
        name='01_A_upstream' if sid=='A_upstream_init_0003' else '03_C_controlled'
        request=json.loads((Path(results[name]['attempt'])/'actual_request.json').read_text())['request']
        row=next(r for r in request['rows'] if r['sample_id']==sid)
        own_item=copy.deepcopy(item)
        if row.get('complete_goal'):
            own_item.update(row['complete_goal'])
        metric=evaluate_state(own_item,row['state_pose_c2w'])
        candidates.append({'sample_id':'review_'+sid,'scene_id':'0003_839989','state_pose_c2w':row['state_pose_c2w'],
                           'camera':row['camera'],'public_observation':{'image_path':ev['artifact']['path']},
                           'original_goal':own_item,'geometry':ev['classification'],'manual_review':ev['manual_review'],
                           'runtime_audit':metric,'runtime_audit_decision':'Yes' if metric['success'] else 'No',
                           'private_answer':None,'qa_sft_allowed':False,'supervision_contract_status':'UNVERIFIED_angular_pixel_conflict',
                           'provenance':'new diagnostic pose observation; not a silently repaired original sample'})
    (out/'trusted_observations_not_training.jsonl').write_text(
        ''.join(json.dumps(r, default=_json_default)+'\n' for r in candidates)
    )
    # Exercise production freeze on the actual GPU-rendered C manifest. Keep all
    # rows in the render manifest; geometry and unresolved supervision prevent release.
    audit=out/'actual_freeze_audit_v2'; audit.mkdir(exist_ok=True)
    bank=Path(results['03_C_controlled']['attempt'])/'manifest.jsonl'
    stats=freeze(argparse.Namespace(output_dir=str(audit),rendered_bank=str(bank),errors='',scene_root=str(out/'assets')))
    statistics=json.loads((out/'source_failure_statistics.json').read_text())
    statistics['original_init']['new_RGB']={'n_rendered_unique':1,'n_geometry_checked':6,'normal_room':1,'unrendered':5}
    for group in ('QA_region','QA_offset','QA_flip','original_sample_point'):
        classifications=json.loads((out/'position_classifications.json').read_text())
        categories={r['sample_id']:r['source_category'] for r in classifications}
        selected=[r for r in evidence if r.get('original_sample_id') in categories and categories[r['original_sample_id']]==group and r['control']['factor']=='camera_axis_convention']
        statistics[group]['new_RGB_controls']={'n_renders':len(selected),'n_positions':len({r['original_sample_id'] for r in selected}),
                                                'normal_room':sum(r['manual_review']['normal_room_RGB'] for r in selected),
                                                'upright_normal_room':sum(r['manual_review']['normal_room_RGB'] and r['manual_review']['upright'] for r in selected),
                                                'not_full_branch_coverage':True}
    (out/'source_failure_statistics_reviewed.json').write_text(
        json.dumps(statistics, indent=2, default=_json_default)+'\n'
    )
    print(json.dumps({'validated_RGB':sum(len(s['artifacts']) for s in results.values()),'freeze_eligible':stats['eligible'],
                      'trusted_observations':[(r['sample_id'],r['runtime_audit_decision']) for r in candidates]},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(Path(a.output).resolve())
