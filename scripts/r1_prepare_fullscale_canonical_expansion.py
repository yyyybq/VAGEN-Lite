#!/usr/bin/env python3
"""Freeze fresh target templates from non-target train parents.

The old Projective/FOV rows are deliberately excluded: this is a new-task
request pool with explicit parent lineage, not a further salvage pass.
"""
from __future__ import annotations
import argparse, hashlib, json, math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

VERSION="r1_fullscale_canonical_expansion_scope_v1_20260928"
TARGETS={"projective_relations","fov_inclusion"}
AMBIGUOUS={"0059_839917","0265_840795","0270_840784","0314_840535","0328_840489","0349_840373"}

def rows(p:Path): return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def h(p:Path): return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p:Path,x:Any):
 p.parent.mkdir(parents=True,exist_ok=True); t=p.with_suffix(p.suffix+'.tmp'); t.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n');t.replace(p)
def dumpj(p:Path,x:list[dict]):
 p.parent.mkdir(parents=True,exist_ok=True); t=p.with_suffix(p.suffix+'.tmp');t.write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in x));t.replace(p)
def pair(objects): return '--'.join(sorted(f"{o.get('id')}:{o.get('label')}" for o in objects))
def cats(objects): return '--'.join(sorted(str(o.get('label') or 'unknown') for o in objects))
def center(o):
 c=o.get('center')
 if c and len(c)==3:return [float(v) for v in c]
 lo,hi=o.get('bbox_min'),o.get('bbox_max')
 if lo and hi:return [(float(a)+float(b))/2 for a,b in zip(lo,hi)]
 raise ValueError('object lacks center/bbox')
def finite(o):
 try:return all(math.isfinite(v) for v in center(o))
 except (ValueError,TypeError):return False
def rank(x): return hashlib.sha256('|'.join(map(str,x)).encode()).hexdigest()

def template(parent:dict, index:int, task:str, relation:str|None):
 ob=[dict(o) for o in parent['target_object']['objects'][:2]]; a,b=center(ob[0]),center(ob[1]); mid=[(a[i]+b[i])/2 for i in range(3)]
 params={'object_a_center':a,'object_b_center':b,'min_distance':0.5,'sample_distance':max(1.,math.dist(a,b))}
 if task=='projective_relations': params['relation']=relation
 else: params.update({'fov_horizontal':110.0,'fov_margin':0.05,'min_radius':0.5,'max_radius':8.0})
 out={'task_id':f'r1_fresh_{task}_{index:06d}', 'task_type':task,'scene_id':parent['scene_id'],'init_camera':parent['init_camera'],
      'target_object':{'objects':ob,'primary':ob[0]},'target_region':{'type':'half_plane' if task=='projective_relations' else 'annulus','params':params,'sample_point':[mid[0],mid[1],1.5],'sample_forward':[0.,1.,0.],'height':1.5},
      'sample_target':[mid[0],mid[1],1.5],'camera_params':{'forward':[0.,1.,0.]},'preset':f'{relation}_of' if relation else 'fov_inclusion',
      'object_label':'+'.join(str(o.get('label')) for o in ob),'task_description':('Position where A appears '+relation+' of B' if relation else 'Position where both objects are visible'),
      'canonical_task_metric_version':'canonical_spatial_task_h1_v1','camera_model_version':'canonical_camera_h1_resize_v1',
      'fresh_task_lineage':{'version':VERSION,'parent_source_key':f'train:{parent["_index"]}','parent_task_type':parent.get('task_type'),'parent_task_id':parent.get('task_id'),'object_pair':pair(ob),'relation':relation}}
 return out

def main():
 p=argparse.ArgumentParser();p.add_argument('--old-train',type=Path,required=True);p.add_argument('--inventory-summary',type=Path,required=True);p.add_argument('--projective-inventory',type=Path,required=True);p.add_argument('--canonical-eval',type=Path,required=True);p.add_argument('--local-action-parents',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--projective-request-count',type=int,default=1900);p.add_argument('--fov-request-count',type=int,default=1600);a=p.parse_args()
 old=rows(a.old_train); inv=rows(a.projective_inventory); dev=rows(a.canonical_eval);local=rows(a.local_action_parents)
 eval_scenes={str(x.get('scene_id')) for x in dev+local}; eval_sources={str(x.get('source_key')) for x in dev+local}; existing_pairs=Counter(x.get('category_pair') for x in inv)
 parent=[]; seen=set()
 for i,r in enumerate(old):
  if r.get('task_type') in TARGETS or str(r.get('scene_id')) in eval_scenes or str(r.get('scene_id')) in AMBIGUOUS or f'train:{i}' in eval_sources: continue
  o=(r.get('target_object') or {}).get('objects') or []
  if len(o)!=2 or not all(finite(x) for x in o) or 'init_camera' not in r: continue
  k=(str(r['scene_id']),pair(o))
  if k in seen: continue
  seen.add(k);q=dict(r);q['_index']=i;parent.append(q)
 # Low-frequency category pairs and broad scenes are deterministically first.
 parent.sort(key=lambda r:(existing_pairs.get(cats(r['target_object']['objects']),0),rank((r['scene_id'],pair(r['target_object']['objects']),r['_index']))))
 proj=[]; fov=[]
 for q in parent:
  if len(proj)<a.projective_request_count: proj.append(template(q,len(proj),'projective_relations','left' if len(proj)%2==0 else 'right'))
  if len(fov)<a.fov_request_count: fov.append(template(q,len(fov),'fov_inclusion',None))
  if len(proj)>=a.projective_request_count and len(fov)>=a.fov_request_count:break
 a.output_dir.mkdir(parents=True,exist_ok=True);combined=proj+fov;dumpj(a.output_dir/'fresh_sources.jsonl',combined)
 # Existing canonical runners expect one literal ``train`` split. Offsets
 # preserve source identity across the two requested task families.
 dump(a.output_dir/'sources.json',{'train':str((a.output_dir/'fresh_sources.jsonl').resolve())})
 dump(a.output_dir/'projective_medium_selection.json',{'version':VERSION,'records':[{'split':'train','source_row_index':i,'scene_id':r['scene_id'],'task_type':r['task_type'],'requested_bucket':'medium','fresh_task_id':r['task_id']} for i,r in enumerate(proj)]})
 dump(a.output_dir/'fov_source_index_selection.json',{'train':list(range(len(proj),len(combined)))})
 summary={'version':VERSION,'policy':{'old_target_rows':'excluded_no_salvage_rerun','same_pair':'fresh object pair template from a unique non-target parent','fov':'provisional quarantine until human calibrated gate','local_action':'permanent eval exclusion'},'inputs':{'old_train':{'path':str(a.old_train.resolve()),'sha256':h(a.old_train)},'inventory_summary':{'path':str(a.inventory_summary.resolve()),'sha256':h(a.inventory_summary)},'projective_inventory':{'path':str(a.projective_inventory.resolve()),'sha256':h(a.projective_inventory)}},'counts':{'unique_eligible_parent_pairs':len(parent),'projective_requests':len(proj),'fov_provisional_requests':len(fov),'requested_projective':a.projective_request_count,'requested_fov':a.fov_request_count,'unfilled_projective_request_slots':max(0,a.projective_request_count-len(proj)),'unfilled_fov_request_slots':max(0,a.fov_request_count-len(fov))},'scenes':dict(sorted(Counter(r['scene_id'] for r in combined).items())),'projective_category_pairs':dict(sorted(Counter(cats(r['target_object']['objects']) for r in proj).items(),key=lambda x:(-x[1],x[0])))}
 dump(a.output_dir/'request_summary.json',summary); files=sorted(x for x in a.output_dir.iterdir() if x.is_file() and x.name!='SHA256SUMS');(a.output_dir/'SHA256SUMS').write_text(''.join(f'{h(x)}  {x.name}\n' for x in files));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
