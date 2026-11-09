"""Derive a bounded new furniture-pair goal pool from actual scene labels.

No renderer, model, training or edits to the formal Active environment. A goal
reference is a legal, geometrically successful camera, NOT RGB visibility or a
successful action demonstration. Production must still apply wall/furniture
admission and review the actual image before making QA rows usable.
"""
from __future__ import annotations
import argparse, copy, hashlib, itertools, json, math, sys
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

ROOT=Path('/mnt/umm/users/yinbaiqiao/VAGEN-Lite')
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
from data_gen.active_spatial_qa.production_v2 import context as load_context, legal, catalog
from data_gen.active_spatial_qa.production_contract_v1 import independent_check, question_variants
from data_gen.active_spatial_qa.image_contract_v2 import pose_from_forward

VERSION='active_spatial_qa_derived_furniture_goals_v3_20261014'
DEFAULT_POLICY={
 'allowed_categories':['sofa','multi person sofa','bed','teatable','dining table','table','chair','armchair','stool','desk','side table','bedside table','bench','ottoman','refrigerator','wardrobe','wine cabinet','tv cabinet','shoe cabinet','storage cabinet','bookcase','bookshelf','cupboard'],
 'minimum_extent_m':0.12,'minimum_height_m':0.22,'maximum_extent_m':5.5,
 'minimum_pair_distance_m':0.65,'maximum_pair_distance_m':5.5,
 'max_pairs_per_scene':80,'max_reference_positions':64,
 'relations':['left','right'],
 'excluded_instances':{'0300_840573':{
  '75':'Prior RGB: wardrobe label corresponds to a tall plain panel; identity unconfirmed.',
  '158':'Prior RGB: storage-cabinet label corresponds to a plain panel; identity unconfirmed.',
  '204':'Prior RGB: plant label denotes white branches amid another green plant; ambiguous.',
  '110':'Prior RGB: small visible niche is labelled with a full-height assembly box.',
  '111':'Prior RGB: shower head label ambiguously covers the full pipe/hose assembly.'}},
}

def digest(x): return hashlib.sha256(json.dumps(x,sort_keys=True,allow_nan=False).encode()).hexdigest()
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def norm_label(x): return ' '.join(str(x).replace('_',' ').casefold().split())

def contains(poly,xy):
    """Ray crossing of the actual polygon; no scene-AABB room approximation."""
    poly=np.asarray(poly,dtype=float); x,y=xy; inside=False
    for a,b in zip(poly,np.roll(poly,-1,axis=0)):
        edge=b-a; rel=np.asarray(xy)-a
        if abs(float(edge[0]*rel[1]-edge[1]*rel[0]))<1e-9 and float(rel@edge)>=-1e-9 and float(rel@edge)<=float(edge@edge)+1e-9: return True
        if (a[1]>y)!=(b[1]>y) and x < (b[0]-a[0])*(y-a[1])/(b[1]-a[1])+a[0]: inside=not inside
    return inside

def derive_scene_goals(cfg,scene,templates,*,ctx=None,legal_positions=None,policy=None):
    """Return (new_goal_dicts, audit); input dictionaries/files stay unchanged.

    cfg needs scene_root, legality, sampling, parent_source_tasks (or source_tasks).
    ctx is production_v2/3.context's compatible five-tuple. legal_positions may
    be the caller's precomputed XYZ list; every used seed is rechecked for room
    membership and legal(). policy overrides DEFAULT_POLICY. Only one category
    instance per actual room is selected; another bedroom's bed is not a rival.
    """
    rules=copy.deepcopy(DEFAULT_POLICY); rules.update(copy.deepcopy(cfg.get('derived_goal_policy',{}))); rules.update(copy.deepcopy(policy or {}))
    templates=[t for t in templates if t.get('scene_id')==scene and t.get('task_type')=='projective_relations']
    if not templates: return [],{'scene_id':scene,'goals':0,'reason':'no_frozen_projective_template'}
    templates=sorted(templates,key=lambda t:t['task_id']); template=templates[0]
    sign=template.get('collision_convention',{}).get('structure_y_sign')
    assert sign in (-1,1),'Unconfirmed structure convention'
    signature=lambda t:(t['camera_model_version'],t['canonical_task_metric_version'],digest(t['init_camera']['intrinsics']),t['collision_convention']['structure_y_sign'])
    assert len({signature(t) for t in templates})==1,'Mixed camera/metric/coordinate conventions'
    assert template['canonical_task_metric_version']=='canonical_spatial_task_h1_v1'
    ctx=ctx if ctx is not None else load_context(cfg,scene)
    assert ctx[0].structure_y_sign==sign and ctx[0].structure_convention_status=='frozen'
    scene_path=Path(cfg['scene_root'])/scene; objects=catalog(scene_path)
    assert len({o['id'] for o in objects})==len(objects),'Duplicate label instance IDs'
    room_polys=ctx[1].room_polys
    room_membership={o['id']:[i for i,p in enumerate(room_polys) if contains(p,np.asarray(o['center'])[:2])] for o in objects}
    category_counts=Counter((ri,norm_label(o['label'])) for o in objects for ri in room_membership[o['id']])
    approved=set(map(norm_label,rules['allowed_categories'])); excluded=rules.get('excluded_instances',{}).get(scene,{})
    rejected=Counter(); eligible=[]
    for o in objects:
        cat=norm_label(o['label']); extent=np.asarray(o['bbox_max'])-o['bbox_min']
        if o['id'] in excluded: rejected['known_problem_instance']+=1;continue
        if cat not in approved: rejected['category_not_enabled']+=1;continue
        if len(room_membership[o['id']])!=1:rejected['nonunique_actual_room']+=1;continue
        room=room_membership[o['id']][0]
        if category_counts[room,cat]!=1:rejected['same_room_category_ambiguous']+=1;continue
        if min(extent)<rules['minimum_extent_m'] or extent[2]<rules['minimum_height_m'] or max(extent)>rules['maximum_extent_m']:rejected['object_size']+=1;continue
        eligible.append(o)
    # Only a canonical same-room pair is enumerated; no reversed-duplicate pair.
    pairs=[]
    for a,b in itertools.combinations(sorted(eligible,key=lambda o:(norm_label(o['label']),o['id'])),2):
        ra,rb=room_membership[a['id']][0],room_membership[b['id']][0]
        if ra!=rb:rejected['pair_different_room']+=1;continue
        if norm_label(a['label'])==norm_label(b['label']):rejected['pair_same_category']+=1;continue
        distance=float(np.linalg.norm(np.asarray(a['center'])[:2]-np.asarray(b['center'])[:2]))
        if not rules['minimum_pair_distance_m']<=distance<=rules['maximum_pair_distance_m']:rejected['pair_distance']+=1;continue
        pairs.append((ra,a,b,distance))
    # Interleave rooms and category pairs before truncating; don't fill the pool
    # with all pairs around the first object in the first room.
    pairs.sort(key=lambda p:(p[0],digest([scene,p[1]['id'],p[2]['id']])))
    by_room=defaultdict(list)
    for p in pairs:by_room[p[0]].append(p)
    ordered=[]
    for i in range(max(map(len,by_room.values()),default=0)):
        for room in sorted(by_room):
            if i<len(by_room[room]):ordered.append(by_room[room][i])
    pairs=ordered[:int(rules['max_pairs_per_scene'])]
    h=float(cfg['sampling']['camera_height_m']); legal_cache={}
    def check_position(p):
        key=tuple(np.round(p,8))
        if key not in legal_cache: legal_cache[key]=legal(cfg,ctx,p)
        return legal_cache[key]
    if legal_positions is None:
        positions=[np.array([*xy,h]) for xy in ctx[1].candidate_points(float(cfg['sampling']['grid_spacing_m']))]
    else: positions=[np.asarray(p,dtype=float) for p in legal_positions]
    for t in templates:
        positions += [np.asarray(t['init_camera']['extrinsics'])[:3,3],np.asarray(t['target_region']['sample_point'],dtype=float)]
    positions=list({tuple(np.round(p,8)):p for p in positions if p.shape==(3,) and np.isfinite(p).all() and abs(p[2]-h)<1e-5}.values())
    positions=[p for p in positions if check_position(p)['status']=='legal_indoor']
    positions_by_room={i:[p for p in positions if contains(poly,p[:2])] for i,poly in enumerate(room_polys)}
    parent_file=Path(cfg.get('parent_source_tasks',cfg['source_tasks']))
    parent_identity={'path':str(parent_file),'sha256':sha(parent_file),'task_id':template['task_id'],'record_sha256':digest(template)}
    asset_identity={name:sha(scene_path/name) for name in ('labels.json','structure.json')}
    output=[]; derived_pairs=set(); failures=[]
    for room,a,b,separation in pairs:
        ca,cb=np.asarray(a['center']),np.asarray(b['center']); midpoint=(ca+cb)/2; direction=(cb[:2]-ca[:2])/separation
        minimum=max(.5,float(np.max(np.asarray(a['bbox_max'])-a['bbox_min'])),float(np.max(np.asarray(b['bbox_max'])-b['bbox_min'])))
        candidates=sorted(positions_by_room[room],key=lambda p:(abs(np.linalg.norm(p[:2]-midpoint[:2])-max(2.5,minimum+.3)),tuple(p)))
        for relation in rules['relations']:
            assert relation in ('left','right')
            normal=np.array([direction[1],-direction[0]]) if relation=='left' else np.array([-direction[1],direction[0]])
            goal={k:copy.deepcopy(template[k]) for k in ('camera_model_version','canonical_task_metric_version','collision_convention')}
            for k in ('native_resolution','native_width','native_height','source_resolution','camera_native_resolution'):
                if k in template:goal[k]=copy.deepcopy(template[k])
            goal.update(scene_id=scene,task_type='projective_relations',preset=relation+'_of',object_label=a['label']+'+'+b['label'],target_object={'objects':[copy.deepcopy(a),copy.deepcopy(b)],'primary':copy.deepcopy(a)},init_camera={'intrinsics':copy.deepcopy(template['init_camera']['intrinsics'])},target_region={'type':'half_plane','height':h,'params':{'boundary_point':midpoint[:2].tolist(),'boundary_direction':direction.tolist(),'normal':normal.tolist(),'object_a_center':ca.tolist(),'object_b_center':cb.tolist(),'relation':relation,'min_distance':minimum}})
            goal['task_description']=f"Position where {a['label']} appears to the {relation} of {b['label']}; both projected full 3D-box rectangles must be visible, at least half in frame, with centres separated by at least 12 pixels in the 256x256 view"
            reference=None
            trials=[p for p in candidates if float((p[:2]-midpoint[:2])@normal)>1e-6 and min(np.linalg.norm(p[:2]-ca[:2]),np.linalg.norm(p[:2]-cb[:2]))>=minimum]
            for point in trials[:int(rules['max_reference_positions'])]:
                pose=pose_from_forward(point,midpoint-point); audit=independent_check(goal,pose)
                if audit['success'] and audit['admission']['geometry_admissible']:
                    reference=(point,pose,audit);break
            if reference is None:
                failures.append({'room_index':room,'object_ids':[a['id'],b['id']],'relation':relation,'reason':'no_legal_geometric_reference','reference_trials':min(len(trials),int(rules['max_reference_positions']))});continue
            point,pose,audit=reference; init=None
            for p in candidates[:int(rules['max_reference_positions'])]:
                q=pose_from_forward(p,midpoint-p); result=independent_check(goal,q)
                if not result['success'] and result['admission']['proper_rotation']:
                    init=(q,result,'same_room_legal_view_with_unsatisfied_goal');break
            if init is None:
                # A legal camera may initially look away. This is explicit, not
                # an expert action or a claim that both targets are observable.
                q=pose_from_forward(point,point-midpoint); result=independent_check(goal,q)
                assert not result['success'];init=(q,result,'same_legal_position_looking_away')
            goal['init_camera']['extrinsics']=init[0].tolist()
            goal['target_region'].update(sample_point=point.tolist(),sample_forward=pose[:3,2].tolist())
            goal['target_region']['params']['sample_distance']=float(np.linalg.norm(point[:2]-midpoint[:2]))
            goal['task_id']='qa3_goal_'+digest({'version':VERSION,'scene':scene,'objects':[a['id'],b['id']],'relation':relation,'labels_sha':asset_identity['labels.json'],'template':parent_identity['record_sha256']})[:24]
            goal['goal_derivation']={'version':VERSION,'parent_template':parent_identity,'source_asset_sha256':asset_identity,'raw_object_ids':[a['id'],b['id']],'raw_labels':[a['label'],b['label']],'room_index':room,'room_membership_basis':'actual room polygon contains each object centre and both camera positions; visibility is a separate gate','object_bounds_method':'min/max and mean of the eight unmodified raw labels.json bounding_box vertices; no masks or new box fitting','region_method':'recompute AB midpoint/unit direction and clockwise normal for left, counterclockwise for right; no old region values copied','reference_camera_c2w':pose.tolist(),'reference_legality':check_position(point),'reference_geometric_success':bool(audit['success']),'reference_relation_margin_px':audit['relation_margin_px'],'reference_inside_fractions':[x['inside_fraction'] for x in audit['projections']],'initial_geometric_success':bool(init[1]['success']),'initial_pose_method':init[2],'reference_visibility_verified':False,'requires_production_visibility_admission':True,'requires_rgb_review':True,'action_demonstration':False,'policy_sha256':digest(rules)}
            assert not goal['goal_derivation']['initial_geometric_success']
            question_variants(goal)  # Contract rejects duplicate/empty names.
            output.append(goal);derived_pairs.add(tuple(sorted([a['id'],b['id']])))
    audit={'version':VERSION,'scene_id':scene,'goals':len(output),'derived_pairs':len(derived_pairs),'eligible_objects':len(eligible),'enumerated_pair_candidates':len(ordered),'considered_pair_candidates':len(pairs),'legal_seed_positions':len(positions),'excluded_instance_reasons':excluded,'rejection_counts':dict(rejected),'reference_failures':failures,'category_counts':dict(Counter(norm_label(o['label']) for o in eligible)),'policy':rules,'rendered':False,'rgb_reviewed':False,'formal_environment_modified':False}
    return output,audit

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--audit',type=Path,required=True);p.add_argument('--scenes',nargs='*');a=p.parse_args()
    if a.output.exists() or a.audit.exists():raise ValueError('Use fresh versioned output and audit paths')
    cfg=json.loads(a.config.read_text());parent=Path(cfg.get('parent_source_tasks',cfg['source_tasks']));templates=[json.loads(x) for x in parent.read_text().splitlines() if x.strip()]
    scenes=a.scenes or list(cfg['scene_splits']);goals=[];reports=[]
    for scene in scenes:
        if scene not in cfg['scene_splits']:raise ValueError('Scene has not been assigned a split: '+scene)
        rows,report=derive_scene_goals(cfg,scene,templates);goals.extend(rows);reports.append(report)
        print(json.dumps({'scene_id':scene,'goals':len(rows),'pairs':report.get('derived_pairs',0)},ensure_ascii=False),flush=True)
    assert len({x['task_id'] for x in goals})==len(goals)
    a.output.parent.mkdir(parents=True,exist_ok=True);a.audit.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(''.join(json.dumps(x,ensure_ascii=False,allow_nan=False)+'\n' for x in goals))
    a.audit.write_text(json.dumps({'version':VERSION,'goals':len(goals),'scenes_with_goals':sum(bool(r['goals']) for r in reports),'scene_audits':reports,'source_sha256':sha(parent),'helper_sha256':sha(__file__)},ensure_ascii=False,indent=2)+'\n')
if __name__=='__main__':main()
