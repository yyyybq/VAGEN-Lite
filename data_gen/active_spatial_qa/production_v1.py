"""Versioned, legal-state-first QA production. No training or model inference.

The formal Active environment and historical labels are never modified.
Planning uses geometric feasibility only; final labels/questions are created
after RGB validation. Human review is a separate, initially pending gate.
"""
from __future__ import annotations
import argparse, asyncio, copy, hashlib, itertools, json, math, os, time, traceback
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw
from data_gen.active_spatial_qa.image_contract_v2 import classify_scene_pose, pose_from_forward, image_stats
from data_gen.active_spatial_qa.production_contract_v1 import independent_check, question_variants

VERSION = 'active_spatial_qa_production_v1_20261012'
class InjectedFailure(RuntimeError): pass

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''): h.update(b)
    return h.hexdigest()

def digest(x): return hashlib.sha256(json.dumps(x,sort_keys=True,allow_nan=False).encode()).hexdigest()
def readl(p): return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def writej(p,x):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.tmp'); t.write_text(json.dumps(x,ensure_ascii=False,indent=2,default=str,allow_nan=False)+'\n'); t.replace(p)
def writel(p,rows):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.tmp'); t.write_text(''.join(json.dumps(x,ensure_ascii=False,default=str,allow_nan=False)+'\n' for x in rows)); t.replace(p)

def context(cfg,scene):
    from vagen.envs.active_spatial.collision_detector import CollisionDetector
    from data_gen.active_spatial_pipeline.layout_quality import LayoutGeometry
    c=cfg['legality']; d=CollisionDetector(camera_radius=c['camera_radius_m'],safety_margin=c['safety_margin_m'],floor_height=c['min_height_m'],ceiling_height=c['max_height_m'])
    if not d.load_scene(Path(cfg['scene_root'])/scene,scene_id=scene): raise ValueError('scene geometry unavailable')
    l=LayoutGeometry(room_polys=[p.tolist() for p in d.room_profiles],wall_segments=[(a.tolist(),b.tolist()) for a,b in d.wall_segments])
    return d,l

def _finite_geometry_json(value):
    # A collision-free detector reports distance=inf: JSON records use null.
    if isinstance(value, dict): return {k:_finite_geometry_json(v) for k,v in value.items()}
    if isinstance(value, (list, tuple)): return [_finite_geometry_json(v) for v in value]
    if isinstance(value, np.ndarray): return _finite_geometry_json(value.tolist())
    if isinstance(value, np.generic): return _finite_geometry_json(value.item())
    if isinstance(value, float) and not math.isfinite(value): return None
    return value

def legal(cfg,ctx,p):
    c=cfg['legality']; return _finite_geometry_json(classify_scene_pose(p,*ctx,min_height=c['min_height_m'],max_height=c['max_height_m']))

def projection_fields(m):
    # The audit implementation emits canonical-independent bbox measurements.
    return m['projections']

def catalog(scene_path):
    out=[]
    for o in json.loads((scene_path/'labels.json').read_text()):
        pts=np.array([[v['x'],v['y'],v['z']] for v in o.get('bounding_box',[])])
        if pts.shape!=(8,3): continue
        out.append({'id':str(o['ins_id']),'label':o['label'],'bbox_min':pts.min(0).tolist(),'bbox_max':pts.max(0).tolist(),'center':pts.mean(0).tolist()})
    return out

def pose_key(scene,p): return digest({'scene':scene,'pose':np.round(p,5).tolist()})[:20]

def room_svg(ctx,pose,objects):
    polys=ctx[1].room_polys; xy=np.concatenate([np.asarray(p) for p in polys]); lo=xy.min(0)-1; hi=xy.max(0)+1
    def xy2(p): return ((p[0]-lo[0])/(hi[0]-lo[0])*440+20,460-(p[1]-lo[1])/(hi[1]-lo[1])*440)
    parts=['<svg xmlns="http://www.w3.org/2000/svg" width="480" height="480" viewBox="0 0 480 480"><rect width="480" height="480" fill="white"/>']
    for poly in polys: parts.append('<polygon points="'+' '.join('%g,%g'%xy2(p) for p in poly)+'" fill="#ecf4ee" stroke="#243c31"/>')
    for box in ctx[0].object_boxes:
        a=xy2(box.min_point); b=xy2(box.max_point); parts.append(f'<rect x="{min(a[0],b[0])}" y="{min(a[1],b[1])}" width="{abs(a[0]-b[0])}" height="{abs(a[1]-b[1])}" fill="none" stroke="#aaa" stroke-width=".6"/>')
    x,y=xy2(np.asarray(pose)[:3,3]); fx,fy=xy2(np.asarray(pose)[:3,3]+np.asarray(pose)[:3,2]); parts.append(f'<circle cx="{x}" cy="{y}" r="4" fill="blue"/><path d="M{x},{y} L{fx},{fy}" stroke="blue" stroke-width="2"/>')
    for i,obj in enumerate(objects):
        x,y=xy2(obj['center']); parts.append(f'<circle cx="{x}" cy="{y}" r="4" fill="{["#df3d30","#923fe0"][i]}"/><text x="{x+5}" y="{y}">{i+1}</text>')
    return ''.join(parts)+'</svg>'

def plan(args):
    cfg=json.loads(Path(args.config).read_text()); out=Path(args.output); out.mkdir(parents=True,exist_ok=True)
    if (out/'planned_states.jsonl').exists(): raise ValueError('immutable plan exists; use render to resume')
    sources=readl(cfg['source_tasks']); counts=Counter(); failures=[]; selected=[]; used=[]
    source_sha=sha(cfg['source_tasks']); rng=np.random.default_rng(cfg['seed']); provenance={}
    quota=cfg['distribution']['pairs_per_task_per_scene']
    for scene,split in cfg['scene_splits'].items():
        ctx=context(cfg,scene); objs=catalog(Path(cfg['scene_root'])/scene)
        provenance[scene]={'aoss_uri':f's3://baiqiao/InteriorGS/{scene}/','files':{f:sha(Path(cfg['scene_root'])/scene/f) for f in ['labels.json','structure.json','3dgs_compressed.ply']}}
        rawpoints=list(ctx[1].candidate_points(cfg['sampling']['grid_spacing_m'])); rng.shuffle(rawpoints)
        positions=[]
        for xy in rawpoints:
            p=np.array([*xy,cfg['sampling']['camera_height_m']]); g=legal(cfg,ctx,p); counts['sampled_positions']+=1
            if g['status']!='legal_indoor': counts['position_'+g['status']]+=1; continue
            positions.append(p)
        scene_tasks=[x for x in sources if x['scene_id']==scene]
        # Prefer diversity of pairs before additional goals of an existing pair.
        scene_tasks.sort(key=lambda x:digest(x['task_id']))
        pair_seen=Counter(); ordered=[]
        for item in scene_tasks:
            key=tuple(sorted(str(o['id']) for o in item['target_object']['objects']))
            ordered.append((pair_seen[key],item)); pair_seen[key]+=1
        scene_tasks=[x[1] for x in sorted(ordered,key=lambda x:x[0])]
        completed=Counter(); instance_counts=Counter(); pair_counts=Counter()
        for source in scene_tasks:
            objects=source['target_object']['objects']; pair=tuple(sorted(str(o['id']) for o in objects))
            if len(objects)!=2 or objects[0]['label']==objects[1]['label']: continue
            if pair_counts[pair]>=cfg['distribution']['max_observations_per_pair_per_scene']: continue
            if any(instance_counts[o['id']]>=cfg['distribution']['max_observations_per_instance_per_scene'] for o in objects): continue
            center=np.mean([o['center'] for o in objects],axis=0)
            pospool=[np.array(source['init_camera']['extrinsics'])[:3,3],np.array(source['target_region']['sample_point'])]
            for p in pospool[:]:
                for dx,dy in [(0.3,0),(-0.3,0),(0,.3),(0,-.3),(.6,0),(-.6,0),(0,.6),(0,-.6)]: pospool.append(p+np.array([dx,dy,0]))
            pospool+=sorted(positions,key=lambda p:abs(np.linalg.norm(p[:2]-center[:2])-3.5))[:cfg['sampling']['max_grid_positions_per_goal']]
            validpoints=[]; seenpos=set()
            for p in pospool:
                pk=tuple(np.round(p,3))
                if pk in seenpos: continue
                seenpos.add(pk); g=legal(cfg,ctx,p)
                if g['status']!='legal_indoor':
                    counts['goal_position_'+g['status']]+=1
                    failures.append({'scene_id':scene,'source_task_id':source['task_id'],'phase':'position','position':p.tolist(),'reason':g['reasons'],'classification':g})
                    continue
                if not cfg['sampling']['min_target_distance_m']<np.linalg.norm(center-p)<cfg['sampling']['max_target_distance_m']: continue
                validpoints.append((p,g))
            for task in cfg['enabled_tasks']:
                if completed[task]>=quota: continue
                if pair_counts[pair]+2>cfg['distribution']['max_observations_per_pair_per_scene']: continue
                if any(instance_counts[o['id']]+2>cfg['distribution']['max_observations_per_instance_per_scene'] for o in objects): continue
                item=copy.deepcopy(source)
                if task=='fov_inclusion':
                    item.update(task_type=task,task_id='qa1_fov_'+digest({'parent':source['task_id'],'pair':pair})[:20],preset='both_in_frame',task_description='Frame both specified objects with at least 95% of each projected 3D-box rectangle in the image and centres at least 5% of the shorter image side from every border')
                pools={'Yes':[],'No':[]}
                for p,g in validpoints:
                    forward=center-p
                    for angle in cfg['sampling']['yaw_offsets_degrees']:
                        a=math.radians(angle); f=np.array([math.cos(a)*forward[0]-math.sin(a)*forward[1],math.sin(a)*forward[0]+math.cos(a)*forward[1],forward[2]])
                        pose=pose_from_forward(p,f); key=pose_key(scene,pose)
                        if any(k==key for k,_,_ in used): continue
                        m=independent_check(item,pose); counts['geometric_proposals']+=1
                        ps=projection_fields(m)
                        if len(ps)!=2 or any(not z.get('all_corners_in_front',False) for z in ps): counts['projection_not_in_front']+=1; continue
                        if any(z['area_ratio']<cfg['evidence']['min_target_area_ratio'] or z['area_ratio']>cfg['evidence']['max_target_area_ratio'] or min(z['clipped_bbox'][2]-z['clipped_bbox'][0],z['clipped_bbox'][3]-z['clipped_bbox'][1])<cfg['evidence']['min_target_side_px'] for z in ps): counts['target_size']+=1; continue
                        fr=min(z['inside_fraction'] for z in ps); cm=min(min(*z['center_uv'],256-z['center_uv'][0],256-z['center_uv'][1]) for z in ps)
                        if task=='projective_relations':
                            margin=m['relation_margin_px']
                            branch='Yes' if margin>=cfg['evidence']['clear_relation_margin_px'] else 'No' if margin<=cfg['evidence'].get('negative_relation_max_margin_px',-cfg['evidence']['clear_relation_margin_px']) else None
                            if fr<cfg['evidence']['projective_min_inside'] or cm<12: branch=None
                            hard=abs(margin)<cfg['difficulty']['relation_hard_below_px']
                        else:
                            branch='Yes' if fr>=.995 and cm>=20 else 'No' if cfg['evidence']['fov_negative_inside_min']<=fr<=cfg['evidence']['fov_negative_inside_max'] and cm>=5 else None
                            hard=branch=='No' and fr>=cfg['difficulty']['fov_hard_inside_above']
                        if branch is None: counts['not_clear_branch']+=1; continue
                        # Object-space line of sight is only a prefilter, never a mask.
                        blocked=[]
                        for ob in objects:
                            for box in ctx[0].object_boxes:
                                if str(box.ins_id) in pair: continue
                                hit,t=box.intersects_segment(p,np.asarray(ob['center']))
                                if hit and .02<t<.92: blocked.append(str(box.ins_id)); break
                        if blocked: counts['center_ray_occlusion_risk']+=1
                        # Same-label distractors are ambiguous without a grounded descriptor.
                        ambiguous=False; distractors=0
                        for ob in objs:
                            if ob['id'] in pair: continue
                            if ob['label'] not in [z['label'] for z in objects]: continue
                            rel=(np.asarray(ob['center'])-p)@pose[:3,:3]
                            if rel[2]<=0: continue
                            uv=[128*rel[0]/rel[2]+128,(256/480*320)*rel[1]/rel[2]+128]
                            if 0<uv[0]<256 and 0<uv[1]<256:
                                # Conservative: visible-ray test must also be unobstructed.
                                if not any(box.intersects_segment(p,np.asarray(ob['center']))[0] for box in ctx[0].object_boxes if str(box.ins_id) not in [ob['id'],*pair]): ambiguous=True; break
                        if ambiguous: counts['same_category_ambiguous']+=1; continue
                        # Restrict near-duplicate views within this scene.
                        if any(s==scene and np.linalg.norm(p-q[:3,3])<cfg['dedup']['min_translation_m'] and math.degrees(math.acos(float(np.clip(np.dot(pose[:3,2],q[:3,2]),-1,1))))<cfg['dedup']['min_rotation_degrees'] for _,s,q in used): continue
                        m['prefilter_visibility']={'center_ray_blockers':blocked,'note':'Expanded collision boxes are not visibility masks. Actual RGB identity/occlusion must be reviewed.'}
                        pools[branch].append((hard,pose,g,m,key))
                if not all(pools.values()):
                    failures.append({'scene_id':scene,'source_task_id':source['task_id'],'task_type':task,'phase':'sampling','reason':'no_clear_legal_positive_negative_pair','pool_counts':{k:len(v) for k,v in pools.items()}}); continue
                picked=[]; pending_used=[]
                for branch in ['Yes','No']:
                    want_hard=(len(selected)+len(picked))%10 in [0,3,6]
                    pool=sorted(pools[branch],key=lambda x:(bool(x[3]['prefilter_visibility']['center_ray_blockers']),x[0]!=want_hard,-min(z['area_ratio'] for z in x[3]['projections'])))
                    chosen=None
                    for cand in pool:
                        p=cand[1]
                        if not any(s==scene and np.linalg.norm(p[:3,3]-q[:3,3])<cfg['dedup']['min_translation_m'] and math.degrees(math.acos(float(np.clip(np.dot(p[:3,2],q[:3,2]),-1,1))))<cfg['dedup']['min_rotation_degrees'] for _,s,q in used+pending_used): chosen=cand; break
                    if chosen is None: break
                    hard,pose,g,m,key=chosen; pending_used.append((key,scene,pose)); sid='qa1_'+key
                    rec={'schema_version':VERSION,'sample_id':sid,'scene_id':scene,'split':split,'goal_id':item['task_id'],'pair_id':item['task_id'],'task_type':task,'goal':item,'state_pose_c2w':pose.tolist(),'pose_c2w':pose.tolist(),'camera':{'intrinsics':item['init_camera']['intrinsics'],'resolution':[256,256]},'source_init_camera':item['init_camera'],'source_camera_model_version':item['camera_model_version'],'source_task_metric_version':item['canonical_task_metric_version'],'source_camera_metadata':{'native_resolution':[640,480]},'public_observation':{'image_path':None,'observation_config':'single_image','history':[]},'private_answer':None,'branch_intent':branch,'difficulty':'hard' if hard else 'clear','difficulty_basis':{'relation_margin_px':m.get('relation_margin_px'),'min_inside_fraction':min(z['inside_fraction'] for z in m['projections']),'basis':'geometric separation or visible cropping; not model accuracy'},'geometry':g,'provenance':{'source_task_file':cfg['source_tasks'],'source_task_file_sha256':source_sha,'source_task_id':source['task_id'],'source_task_sha256':digest(source),'asset_identity':provenance[scene],'generation':'legal layout positions; target-facing yaw sampling','derived_goal':task!=source['task_type']},'qa_sft_allowed':False,'label_validity':'pending_rgb_evidence','observability_validity':'not_rendered','semantic_review':{'status':'pending','reviewer':None,'reason':None},'assistant_review':{'status':'pending'},'planning_audit':m}
                    picked.append(rec)
                if len(picked)==2:
                    used.extend(pending_used); selected.extend(picked); completed[task]+=1; pair_counts[pair]+=2
                    for ob in objects: instance_counts[ob['id']]+=2
                if all(completed[t]>=quota for t in cfg['enabled_tasks']): break
            if all(completed[t]>=quota for t in cfg['enabled_tasks']): break
        print(json.dumps({'scene':scene,'completed_pairs':completed,'selected':len(selected),'positions':len(positions)}),flush=True)
        # Explicit outside/no-collision probe through the same production gate.
        xy=np.concatenate([np.asarray(p) for p in ctx[1].room_polys]); outside=np.array([xy[:,0].max()+10,xy[:,1].max()+10,1.5]); collision=ctx[0].check_collision(outside); rejected=legal(cfg,ctx,outside)
        if collision.has_collision or rejected['status']=='legal_indoor': raise AssertionError('outside no-collision rejection check failed')
        failures.append({'scene_id':scene,'phase':'actual_production_gate_probe','reason':'outside_room_polygon','collision_type':collision.collision_type,'geometry':rejected,'position':outside.tolist(),'render_requested':False,'label':None})
    writej(out/'distribution_config.json',cfg); writej(out/'asset_provenance.json',provenance)
    writel(out/'planned_states.jsonl',selected); writel(out/'failures.jsonl',failures)
    writej(out/'planning_statistics.json',{'counts':counts,'planned':len(selected),'by_task_answer':dict(Counter(x['task_type']+'/'+x['branch_intent'] for x in selected)),'by_scene':dict(Counter(x['scene_id'] for x in selected)),'requested':cfg['distribution'],'labels_computed':False})
    writej(out/'split_manifest.json',{'version':1,'split_unit':'scene','scene_splits':cfg['scene_splits'],'applies_to':['QA','Action','mixed'],'legacy_trajectory_policy':'Reassign only copied candidate records to this split; forbid every val/test scene in exported Action train; do not alter running environments.'})

async def render_run(args):
    from data_gen.active_spatial_qa.render_paired_bank import render_camera
    from vagen.envs.active_spatial.render.unified_renderer import UnifiedRenderGS
    out=Path(args.output); cfg=json.loads((out/'distribution_config.json').read_text()); rows=readl(out/'planned_states.jsonl')
    runid=str(time.time_ns()); status={'id':runid,'ok':False,'phase':'running','rendered':[],'reused':[],'started':time.time()}; runs=out/'worker_runs'; runs.mkdir(exist_ok=True); renderer=None; contexts={}
    try:
        for row in rows:
            scene=row['scene_id']; contexts.setdefault(scene,None)
            if contexts[scene] is None:
                contexts[scene]=context(cfg,scene)
                for name,expected in row['provenance']['asset_identity']['files'].items():
                    if sha(Path(cfg['scene_root'])/scene/name)!=expected: raise ValueError('scene asset changed since planning')
            g=legal(cfg,contexts[scene],np.array(row['state_pose_c2w'])[:3,3])
            if g['status']!='legal_indoor': raise ValueError('illegal position reached renderer')
            identity={'record':digest(row),'assets':row['provenance']['asset_identity'],'code':{str(Path(__file__).name):sha(__file__),'render_paired_bank.py':sha(Path(__file__).with_name('render_paired_bank.py'))}}
            key=digest(identity); dest=out/'renders'/row['sample_id']; dest.mkdir(parents=True,exist_ok=True); prior=None
            for path in sorted(dest.glob('attempt_*/status.json'),reverse=True):
                try:
                    s=json.loads(path.read_text()); p=Path(s.get('image_path',''))
                    if s.get('ok') and s.get('identity')==key and p.is_file() and sha(p)==s['sha256']:
                        with Image.open(p) as im: im.load(); assert im.size==(256,256)
                        prior=s; break
                except (ValueError,OSError,KeyError,AssertionError): pass
            if prior: status['reused'].append(row['sample_id']); continue
            dest=dest/('attempt_'+str(time.time_ns())); dest.mkdir(); s={'ok':False,'sample_id':row['sample_id'],'identity':key,'request_identity':identity,'started':time.time()}
            try:
                if renderer is None or renderer.cfg.scene_id!=scene:
                    if renderer is not None: await renderer.close()
                    renderer=UnifiedRenderGS(render_backend='local',gs_root=cfg['scene_root'],scene_id=scene,gpu_device=0)
                K,w2c=render_camera(row,256,256); im=await renderer.render_image_from_cam_param(K,w2c,256,256)
                if im.size!=(256,256): raise ValueError('wrong resolution')
                p=dest/'rgb.png'; im.save(p)
                with Image.open(p) as check: check.load()
                s.update(ok=True,image_path=str(p.resolve()),sha256=sha(p),stats=image_stats(im),camera={'K_effective':K.tolist(),'w2c':w2c.tolist(),'c2w':row['state_pose_c2w'],'width':256,'height':256},finished=time.time())
                writej(dest/'status.json',s); status['rendered'].append(row['sample_id'])
            except Exception:
                s.update(ok=False,error=traceback.format_exc(),finished=time.time()); writej(dest/'status.json',s); raise
            if args.inject_failure_after and len(status['rendered'])>=args.inject_failure_after:
                raise InjectedFailure('QA_PRODUCTION_INJECTED_FAILURE: interruption after verified output; resume must reuse it')
        status.update(ok=True,phase='succeeded')
    except BaseException:
        status.update(ok=False,phase='failed',error=traceback.format_exc()); raise
    finally:
        if renderer is not None: await renderer.close()
        status['finished']=time.time(); writej(runs/(runid+'.json'),status); print(json.dumps(status),flush=True)

def finalize(args):
    from data_gen.active_spatial_qa.contract import evaluate_state
    out=Path(args.output); cfg=json.loads((out/'distribution_config.json').read_text()); result=[]; fails=readl(out/'failures.jsonl'); contexts={}; seenhash=set()
    for row in readl(out/'planned_states.jsonl'):
        records=[json.loads(p.read_text()) for p in sorted((out/'renders'/row['sample_id']).glob('attempt_*/status.json'),reverse=True)]
        identity={'record':digest(row),'assets':row['provenance']['asset_identity'],'code':{str(Path(__file__).name):sha(__file__),'render_paired_bank.py':sha(Path(__file__).with_name('render_paired_bank.py'))}}
        s=next((x for x in records if x.get('ok') and x.get('identity')==digest(identity) and Path(x.get('image_path','')).is_file() and sha(x['image_path'])==x.get('sha256')),None)
        if s is None: fails.append({'sample_id':row['sample_id'],'reason':'render_failed_or_missing'}); continue
        with Image.open(s['image_path']) as im: im.load(); stats=image_stats(im)
        if not stats['nonblank']: fails.append({'sample_id':row['sample_id'],'reason':'bad_image_low_information'}); continue
        if s['sha256'] in seenhash: fails.append({'sample_id':row['sample_id'],'reason':'exact_image_duplicate'}); continue
        seenhash.add(s['sha256']); row['public_observation']['image_path']=s['image_path']; row['render']=s; row['observability_validity']='valid'
        pose=np.asarray(row['state_pose_c2w']); independent=independent_check(row['goal'],pose)
        if not np.allclose(np.asarray(s['camera']['K_effective']),independent['camera']['K_effective'],atol=1e-6,rtol=0) or not np.allclose(np.asarray(s['camera']['w2c']),np.linalg.inv(pose),atol=1e-6,rtol=0):
            fails.append({'sample_id':row['sample_id'],'reason':'actual_render_camera_mismatch'}); continue
        scene=row['scene_id']
        if scene not in contexts: contexts[scene]=context(cfg,scene)
        if legal(cfg,contexts[scene],pose[:3,3])['status']!='legal_indoor':
            fails.append({'sample_id':row['sample_id'],'reason':'final_illegal_position'}); continue
        asset_objects={o['id']:o for o in catalog(Path(cfg['scene_root'])/scene)}
        if any(str(o['id']) not in asset_objects or any(not np.allclose(o[k],asset_objects[str(o['id'])][k],atol=1e-5,rtol=0) for k in ['bbox_min','bbox_max','center']) for o in row['goal']['target_object']['objects']):
            fails.append({'sample_id':row['sample_id'],'reason':'target_asset_geometry_mismatch'}); continue
        active=evaluate_state(row['goal'],pose)
        if bool(active['success'])!=bool(independent['success']): fails.append({'sample_id':row['sample_id'],'reason':'independent_predicate_disagreement','active':active,'independent':independent}); continue
        answer='Yes' if active['success'] else 'No'
        if answer!=row['branch_intent']: fails.append({'sample_id':row['sample_id'],'reason':'planned_branch_changed'}); continue
        row.update(private_answer=answer,question=question_variants(row['goal'])[0],question_variants=question_variants(row['goal']),label_validity='valid',qa_sft_allowed=False,metrics={'active':active,'independent':independent},predicate_version=row['goal']['canonical_task_metric_version'])
        scene=row['scene_id']
        if scene not in contexts: contexts[scene]=context(cfg,scene)
        overlay=Image.open(s['image_path']).convert('RGB'); d=ImageDraw.Draw(overlay)
        for i,obj in enumerate(independent['projections']):
            box=obj['clipped_bbox']; d.rectangle(box,outline=['red','cyan'][i],width=2); d.text((box[0]+2,box[1]+2),str(i+1),fill=['red','cyan'][i])
        evidence=out/'inspection_only'; evidence.mkdir(exist_ok=True); op=evidence/(row['sample_id']+'_boxes.png'); overlay.save(op)
        mp=evidence/(row['sample_id']+'_map.svg'); mp.write_text(room_svg(contexts[scene],pose,row['goal']['target_object']['objects']))
        row['inspection']={'overlay_path':str(op.resolve()),'camera_map_path':str(mp.resolve()),'must_not_be_model_input':True}
        result.append(row)
    writel(out/'sample_manifest.jsonl',result); writel(out/'all_failures.jsonl',fails)
    writej(out/'automatic_statistics.json',{'planned':len(readl(out/'planned_states.jsonl')),'rgb_and_geometry_accepted':len(result),'human_approved':0,'by_task_answer':dict(Counter(x['task_type']+'/'+x['private_answer'] for x in result)),'failure_reasons':dict(Counter(str(x['reason']) for x in fails))})
    # Resume audit relies on actual persisted executions and render attempts.
    runs=[json.loads(p.read_text()) for p in sorted((out/'worker_runs').glob('*.json'))]
    failed=[x for x in runs if not x['ok']]; success=[x for x in runs if x['ok']]
    injected=[x for x in failed if 'QA_PRODUCTION_INJECTED_FAILURE' in x.get('error','') and x.get('rendered')]
    writej(out/'resume_verification.json',{'passed':bool(injected and success and success[-1]['started']>injected[0]['finished'] and set(injected[0]['rendered']).issubset(set(success[-1]['reused']))),'failed_run_ids':[x['id'] for x in failed],'success_run_ids':[x['id'] for x in success],'verified_outputs_reused':success[-1]['reused'] if success else [],'render_attempt_counts':{r['sample_id']:len(list((out/'renders'/r['sample_id']).glob('attempt_*'))) for r in result}})
    print(json.dumps({'accepted':len(result),'failures':len(fails)}))

def main():
    p=argparse.ArgumentParser(); p.add_argument('phase',choices=['plan','render','finalize']); p.add_argument('--config'); p.add_argument('--output',required=True); p.add_argument('--inject-failure-after',type=int,default=0); a=p.parse_args()
    if a.phase=='plan': plan(a)
    elif a.phase=='render':
        try: asyncio.run(render_run(a))
        except InjectedFailure:
            print('QA_PRODUCTION_INJECTED_FAILURE',flush=True)
            raise SystemExit(86)
    else: finalize(a)
if __name__=='__main__': main()
