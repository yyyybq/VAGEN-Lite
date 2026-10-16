"""Prepare geometry, bind actual rendered evidence, and export pending review QA.

No model prediction is used for selection, labels or review. Training release is
empty until the user supplies a matching human review file.
"""
from __future__ import annotations
import argparse
import base64
from collections import Counter, defaultdict
import hashlib
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image

from data_gen.interiorgs_v10.occlusion import face_occluders_for, face_identifiable
from data_gen.interiorgs_v10.start_context import start_refs
from . import rules, templates, evidence
from .sources import load_task_scene
from .render import read_rows, view_key, sha, cached_ok

META = Path('/mnt/umm/users/yinbaiqiao/InteriorGS_meta')
GROUPS = {'absolute_positioning':'距离估计', 'equidistance':'远近比较',
          'projective_relations':'视角方向', 'occlusion_alignment':'遮挡判断',
          'fov_inclusion':'画面边缘裁切'}
SIZE_EXCLUSION_ZH = {
    'size_comparison_not_uniquely_bound': '较大、较小等描述无法在本任务场景中绑定唯一的真实同类比较对象。',
    'size_comparison_not_identifiable_with_real_depth': '较大、较小等描述所需的比较对象未同时通过几何和实际深度可辨认检查。',
}


def dump(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def write_rows(path, rows):
    Path(path).write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False)+'\n' for r in rows))


def refs_unchanged(scene, task, pose):
    refs = start_refs(scene, task, pose, obs_fn=face_identifiable)
    if refs is None:
        return False
    original = {str(o['id']): o for o in task['target_object']['objects']}
    return all(original[oid]['ref'] == new['ref'] for oid, new in refs.items())


def prepare(root, dest):
    root, dest = Path(root), Path(dest)
    dest.mkdir(parents=True, exist_ok=False)
    tasks = {t['task_id']:t for t in read_rows(root/'tasks.jsonl')}
    scenes, rows = {}, []
    for n, state in enumerate(read_rows(root/'states.jsonl')):
        task = tasks[state['parent_task_id']]
        rec = {'state_id':state['state_id'], 'task_type':task['task_type'], 'scene_id':state['scene_id'],
               'parent_task_id':task['task_id'], 'drafts':[], 'reasons':[]}
        if not state.get('sample_eligible') or not state.get('runtime_legal'):
            rec['reasons'] = ['state_not_eligible']
        elif task['task_type'] not in GROUPS:
            rec['reasons'] = ['family_paused']
        else:
            sid = state['scene_id']
            if sid not in scenes:
                scenes[sid] = load_task_scene(task, META)
            scene = scenes[sid]
            if not refs_unchanged(scene, task, state['c2w']):
                rec['reasons'] = ['reference_changed_or_ambiguous_at_this_pose']
            else:
                measured = rules.measure(task, state['c2w'], face_occluders_for(scene))
                rec['measurement'] = measured
                rec['view_key'] = view_key(sid, state['c2w'], measured['camera']['K_effective'])
                rec['drafts'], rec['reasons'] = rules.propose(task, measured)
        rows.append(rec)
        if n % 100 == 0:
            print(json.dumps({'geometry_states':n+1,'drafts':sum(bool(r['drafts']) for r in rows)}),flush=True)
    write_rows(dest/'geometry.jsonl', rows)
    dump(dest/'geometry_summary.json', {'state_records':len(rows), 'draft_states':sum(bool(r['drafts']) for r in rows),
          'first_reason_counts':dict(Counter(r['reasons'][0] for r in rows if r['reasons'])),
          'config':rules.load_config(), 'source_states_sha256':sha(root/'states.jsonl'),
          'source_tasks_sha256':sha(root/'tasks.jsonl')})


def overlay(path, measured, dest):
    """Separate inspection SVG; raw model image is never changed."""
    encoded=base64.b64encode(Path(path).read_bytes()).decode()
    parts=[f'<svg xmlns="http://www.w3.org/2000/svg" width="256" height="256" viewBox="0 0 256 256"><image width="256" height="256" href="data:image/png;base64,{encoded}"/>']
    for i,obj in enumerate(measured['objects']):
        box=obj['projection'].get('bbox_raw')
        if box is None or len(box)!=4 or not np.isfinite(box).all():
            continue
        x0,y0,x1,y1=box
        color=['#ff367d','#23e4ef'][i%2]
        parts.append(f'<rect x="{x0:.2f}" y="{y0:.2f}" width="{max(x1-x0,0):.2f}" height="{max(y1-y0,0):.2f}" fill="none" stroke="{color}" stroke-width="2"/><text x="{max(2,min(x0,240)):.2f}" y="{max(12,min(y0,250)):.2f}" font-size="9" fill="{color}" stroke="black" stroke-width="0.3">{html.escape(obj.get("ref",obj["role"]))}</text>')
    Path(dest).write_text(''.join(parts)+'</svg>')


def training_row(q):
    if any('overlay' in str(p) or 'inspection_' in str(p) for p in q['images']):
        raise ValueError('inspection image cannot enter model input')
    text='\n'.join(['<image>']*len(q['images']))+'\n'+q['question']+'\n'+'\n'.join(f'{chr(65+i)}. {o}' for i,o in enumerate(q['options']))
    return {'messages':[{'role':'user','content':text}, {'role':'assistant','content':q['answer']}],
            'images':q['images'], 'videos':[]}


def make_question(task, draft, records, states, dest):
    ids=[s['state_id'] for s in states]
    identity={'kind':draft['kind'],'scene':task['scene_id'],'task':task['task_id'],'states':ids}
    qid='cap1_'+hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:20]
    wording=templates.render_draft(draft, identity=identity, seed=rules.load_config()['seed'])
    images=[r['render']['rgb'] for r in records]
    overlays=[]
    for i,(path,rec) in enumerate(zip(images,records)):
        p=dest/'overlays'/f'{qid}_{i}.svg'
        overlay(path,rec['measurement'],p)
        overlays.append(str(p.resolve()))
    return {'qid':qid,'kind':draft['kind'],'group':GROUPS[task['task_type']], 'task_type':task['task_type'],
            'scene_id':task['scene_id'],'parent_task_id':task['task_id'],**wording,
            'images':images,'overlays':overlays,'evidence_zh':draft['evidence_zh'],
            'human_review':'pending','training_allowed':False,
            'source':{'split':'train','formal':True,'original_instruction':task['task_description'],
                      'state_ids':ids,'view_keys':[r['view_key'] for r in records],
                      'source_kinds':[s['source_kind'] for s in states],
                      'actions_from_initial':[s['actions_from_initial'] for s in states],
                      'pose_c2w':[s['c2w'] for s in states],
                      'image_sha256':[r['render']['rgb_sha256'] for r in records],
                      'object_ids':[o['id'] for o in records[-1]['measurement']['objects']],
                      'object_labels':[o['label'] for o in records[-1]['measurement']['objects']],
                      'environment_success':records[-1]['measurement']['metric'].get('success'),
                      'formal_role_answer_key':draft['answer_key'],
                      'mentioned_object_roles':wording.get('mentioned_object_roles',['a','b']),
                      'size_comparison_references':records[-1].get('size_comparison_references',{}),
                      'definition_notes':draft.get('extra',{}),
                      'image_order':'first identifies target; second is judged' if len(images)==2 else 'current view'}}


def distribution(rows):
    by=defaultdict(list)
    for r in rows: by[r['kind']].append(r)
    paths={p for r in rows for p in r['images']}
    return {'qa_rows':len(rows),'image_occurrences':sum(len(r['images']) for r in rows),
            'unique_image_paths':len(paths), 'unique_image_sha256':len({sha(p) for p in paths}),
            'independent_visual_inputs':len({tuple(r['source']['view_keys']) for r in rows}),
            'scene_counts':dict(Counter(r['scene_id'] for r in rows)),
            'parent_family_counts':dict(Counter(r['task_type'] for r in rows)),
            'source_input_occurrences':dict(Counter(s for r in rows for s in r['source']['source_kinds'])),
            'object_category_counts':dict(Counter(s for r in rows for s in r['source']['object_labels'])),
            'object_instances':len({(r['scene_id'],s) for r in rows for s in r['source']['object_ids']}),
            'object_combinations':len({(r['scene_id'],tuple(r['source']['object_ids'])) for r in rows}),
            'by_kind':{k:{'questions':len(v),'semantic_answers':dict(Counter(r['answer_key'] for r in v)),
                           'missing_semantic_answers':sorted({o for r in v for o in r.get('option_keys',[])}-{r['answer_key'] for r in v}),
                           'missing_correct_letters':sorted(set('ABCD'[:len(v[0]['options'])])-{r['answer'] for r in v}),
                           'letter_answers':dict(Counter(r['answer'] for r in v)),
                           'formal_role_answers':dict(Counter(r.get('source_answer_key',r['answer_key']) for r in v)),
                           'object_mention_order':dict(Counter(','.join(r.get('mentioned_object_roles',[])) for r in v)),
                           'templates':dict(Counter(r['template_id'] for r in v)),
                           'scenes':dict(Counter(r['scene_id'] for r in v))} for k,v in by.items()}}


def finalize(root, geometry, dest):
    root, geometry, dest = Path(root), Path(geometry), Path(dest)
    dest.mkdir(parents=True,exist_ok=False); (dest/'overlays').mkdir()
    tasks={r['task_id']:r for r in read_rows(root/'tasks.jsonl')}
    states={r['state_id']:r for r in read_rows(root/'states.jsonl')}
    renders={r['view_key']:r for r in read_rows(root/'render_manifest.jsonl')}
    geometry_rows=read_rows(geometry/'geometry.jsonl')
    if len(geometry_rows)!=len(states) or {r['state_id'] for r in geometry_rows}!=set(states):
        raise ValueError('geometry must bind every source state exactly once')
    scenes, boxes, size_bindings, accepted, ledger = {}, {}, {}, {}, []
    for n,rec in enumerate(geometry_rows):
        task=tasks[rec['parent_task_id']]; state=states[rec['state_id']]; sid=state['scene_id']
        reason=None
        if 'measurement' not in rec or (not rec['drafts'] and task['task_type']!='occlusion_alignment'):
            reason=(rec['reasons'] or ['no_geometry_draft'])[0]
        elif rec['view_key'] not in renders:
            reason='render_missing_or_failed'
        else:
            render=renders[rec['view_key']]
            if not cached_ok(Path(render['rgb']).parent.parent/'status.json', rec['view_key'], task['task_type']=='fov_inclusion'):
                raise ValueError('render evidence changed or incomplete: '+rec['view_key'])
            if sid not in scenes:
                scenes[sid]=load_task_scene(task,META); boxes[sid]=evidence.load_boxes(META/sid)
            if task['task_id'] not in size_bindings:
                size_bindings[task['task_id']]=evidence.bind_size_references(task,scenes[sid])
            size_refs=size_bindings[task['task_id']]
            with np.load(render['depth']) as d:
                wide=None
                if render.get('wide_depth'):
                    with np.load(render['wide_depth']) as w: wide=(w['depth'].copy(),w['alpha'].copy())
                    consistent, report=evidence.wide_center_agrees(d['depth'],d['alpha'],*wide)
                    rec['wide_center_comparison']=report
                    if not consistent:
                        raise ValueError('wide diagnostic and native depth disagree: '+rec['view_key'])
                stat,masks=evidence.analyse(task,np.array(state['c2w']),d['depth'],d['alpha'],face_occluders_for(scenes[sid]),boxes[sid],wide,
                                            size_references=size_refs)
            rec['render']=render; rec['depth_evidence']=stat
            rec['size_comparison_references']=size_refs
            rec['size_reference_exclusions']=[{'target_slot':slot,'reason':why,'reason_zh':SIZE_EXCLUSION_ZH[why]}
                for slot in size_refs if (why:=evidence.size_reference_failure(task,stat,size_refs,slots={slot}))]
            targets=rec['measurement']['objects']
            good=all(stat.get(o['id'],{}).get('visible') for o in targets) and evidence.visible_references(task,stat,size_references=size_refs)
            if rec['drafts'] and not good:
                rec['drafts']=[]; reason=evidence.size_reference_failure(task,stat,size_refs) or 'target_or_reference_not_identifiable_with_real_depth'
            if rec['drafts'] and task['task_type']=='fov_inclusion':
                ok,reason=evidence.fov_depth_agrees(targets,stat,rec['drafts'][0]['extra']['full_flags'])
                if not ok: rec['drafts']=[]
                else:
                    rec['drafts'][0]['evidence_zh'] += [f"{o['ref']}：加宽检查画布中，可归属该标注盒子的实际深度像素有{stat[o['id']]['wide_depth_evidence']['outside_original_frame_fraction']*100:.1f}%落在原图外；原图边缘有{stat[o['id']]['edge_pixels']}个此类像素。" for o in targets]
            if task['task_type']=='occlusion_alignment':
                key,ev,why=evidence.occlusion_class(task,rec['measurement'],stat,masks)
                bslot=1
                if not evidence.visible_references(task,stat,slots={bslot},size_references=size_refs):
                    key=None; why=evidence.size_reference_failure(task,stat,size_refs,slots={bslot}) or 'occluder_reference_not_visible'
                rec['occlusion_state_key']=key; rec['occlusion_evidence']=ev
                rec['occlusion_reason']=why
            identifying = (task['task_type']=='occlusion_alignment' and stat.get(targets[0]['id'],{}).get('visible')
                           and evidence.visible_references(task,stat,slots={0},size_references=size_refs))
            if rec['drafts'] or rec.get('occlusion_state_key') or identifying:
                accepted[rec['state_id']]=rec
            if not rec['drafts'] and not reason: reason=(rec['reasons'] or ['no_single_image_draft'])[0]
        ledger.append({'state_id':state['state_id'],'scene_id':sid,'task_type':task['task_type'],
                       'single_image_drafts':len(rec['drafts']) if rec.get('render') else 0,
                       'occlusion_category':rec.get('occlusion_state_key'), 'first_exclusion':reason,
                       'first_exclusion_zh':SIZE_EXCLUSION_ZH.get(reason),
                       'size_reference_exclusions':rec.get('size_reference_exclusions',[]),
                       'geometry_reasons':rec['reasons'], 'occlusion_exclusion':rec.get('occlusion_reason')})
        if n%100==0: print(json.dumps({'depth_checked_states':n+1,'evidence_states':len(accepted)}),flush=True)
    bank=[]; seen=set()
    def add(task,draft,recs,ss):
        key=(draft['kind'],task['scene_id'],tuple(o['id'] for o in recs[-1]['measurement']['objects']),tuple(r['view_key'] for r in recs))
        if key in seen: return
        seen.add(key); bank.append(make_question(task,draft,recs,ss,dest))
    for stateid,rec in accepted.items():
        task=tasks[rec['parent_task_id']]
        for draft in rec['drafts']: add(task,draft,[rec],[states[stateid]])
        if not rec.get('occlusion_state_key'): continue
        aid=rec['measurement']['objects'][0]['id']
        # The identifying frame must be an earlier prefix of this actual action path.
        current_actions=states[stateid]['actions_from_initial']
        refs=[]
        for otherid,other in accepted.items():
            s=states[otherid]; actions=s['actions_from_initial']
            if (s['parent_task_id']==task['task_id'] and len(actions)<len(current_actions)
                and current_actions[:len(actions)]==actions and other['view_key']!=rec['view_key']
                and other['depth_evidence'].get(aid,{}).get('visible')
                and evidence.visible_references(task,other['depth_evidence'],slots={0},
                                                size_references=size_bindings[task['task_id']])):
                refs.append((len(actions),otherid,other))
        if not refs: continue
        _,firstid,first=min(refs)
        objs=rec['measurement']['objects']; ev=rec['occlusion_evidence']
        draft={'kind':'occlusion_state','answer_key':rec['occlusion_state_key'],
               'option_keys':['fully_hidden','partly_hidden','not_hidden','outside'],
               'fields':{'a':objs[0]['ref'],'b':objs[1]['ref']},
               'evidence_zh':[f"第一张图来自同一动作路径的较早状态，目标{objs[0]['ref']}通过环境和实际深度的可辨认检查。",
                   f"第二张图中，目标几何盒子投影有{ev['target_box_in_frame_fraction']*100:.2f}%在画面内。",
                   f"目标盒子投影区域内，实际深度可归属遮挡物的覆盖比例为{ev['depth_supported_blocker_coverage']*100:.2f}%，可归属目标的比例为{ev['depth_supported_target_visible_fraction']*100:.2f}%。",
                   '深度归属以标注盒子为近似，不是物体真实轮廓的分割标注；仍需人工确认遮挡对象和答案。'],
               'extra':ev}
        add(task,draft,[first,rec],[states[firstid],states[stateid]])
    write_rows(dest/'candidate_bank.jsonl',bank); write_rows(dest/'state_filter_ledger.jsonl',ledger)
    # All private evidence stays separate from the training-format conversation.
    write_rows(dest/'depth_evidence.jsonl',[{k:v for k,v in r.items() if k!='drafts'} for r in accepted.values()])
    from .selection import select_questions, estimate_capacity
    selected,selection=select_questions(bank, families=list(GROUPS.values()))
    write_rows(dest/'questions.jsonl',selected)
    dump(dest/'selection.json',selection)
    dump(dest/'statistics.json',{'candidate_bank':distribution(bank),'review_sample':distribution(selected),
         'state_records':len(ledger),'state_first_exclusions':dict(Counter(r['first_exclusion'] for r in ledger if r['first_exclusion'])),
         'filter_counts_by_parent_family':{f:dict(Counter(r['first_exclusion'] or 'single_image_geometry_depth_candidate' for r in ledger if r['task_type']==f)) for f in GROUPS},
         'human_pending':len(selected),'human_approved':0,'training_released':0})
    dump(dest/'capacity.json',estimate_capacity(bank,selected,scene_ids=sorted({s['scene_id'] for s in states.values()}),families=list(GROUPS.values())))
    write_rows(dest/'pending_training_format.jsonl',[training_row(q) for q in selected])
    write_rows(dest/'approved_training.jsonl',[])
    dump(dest/'training_config.json',{'do_train':False,'human_review_required':True,'approved_count':0,
         'dataset':str((dest/'approved_training.jsonl').resolve()),'pending_review_only':str((dest/'pending_training_format.jsonl').resolve()),
         'train_on_prompt':False,'mask_history':True,'supervision':'assistant answer letter only',
         'image_size':[256,256],'label_and_geometry_in_user_input':False})
    if selected:
        from .review import build_site
        dump(dest/'review_build_receipt.json',build_site(dest/'questions.jsonl',dest/'review_site'))
    print(json.dumps({'bank_questions':len(bank),'selected_questions':len(selected),'destination':str(dest)},ensure_ascii=False),flush=True)


def combine(root, extension, geometry, dest):
    """Join two completed, disjoint render batches without copying or rendering RGB."""
    root, extension, geometry, dest=map(Path,(root,extension,geometry,dest))
    if sha(root/'tasks.jsonl')!=sha(extension/'tasks.jsonl'):
        raise ValueError('extension changed formal tasks')
    states=read_rows(root/'states.jsonl')+read_rows(extension/'states.jsonl')
    measures=read_rows(geometry/'geometry.jsonl')+read_rows(extension/'geometry.jsonl')
    if len({s['state_id'] for s in states})!=len(states):
        raise ValueError('duplicate source state')
    if len(measures)!=len(states) or {s['state_id'] for s in states}!={s['state_id'] for s in measures}:
        raise ValueError('geometry/state join is incomplete')
    renders=read_rows(root/'render_manifest.jsonl')+read_rows(extension/'render_manifest.jsonl')
    if len({r['view_key'] for r in renders})!=len(renders):
        raise ValueError('extension unexpectedly duplicated a rendered camera')
    dest.mkdir(parents=True,exist_ok=False)
    (dest/'tasks.jsonl').write_bytes((root/'tasks.jsonl').read_bytes())
    write_rows(dest/'states.jsonl',states);write_rows(dest/'geometry.jsonl',measures)
    write_rows(dest/'render_manifest.jsonl',renders)
    dump(dest/'combination_receipt.json',{'source_roots':[str(root.resolve()),str(extension.resolve())],
        'states':len(states),'successful_views':len(renders),'new_renders_in_this_step':0,
        'inputs':{str((base/name).resolve()):sha(base/name) for base in (root,extension)
                  for name in ('tasks.jsonl','states.jsonl','render_manifest.jsonl')}})


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('stage',choices=['prepare','combine','finalize'])
    p.add_argument('--root',type=Path,required=True); p.add_argument('--out',type=Path,required=True)
    p.add_argument('--geometry',type=Path);p.add_argument('--extension',type=Path)
    a=p.parse_args()
    if a.stage=='prepare': prepare(a.root,a.out)
    elif a.stage=='combine': combine(a.root,a.extension,a.geometry,a.out)
    else: finalize(a.root,a.geometry,a.out)

if __name__=='__main__': main()
