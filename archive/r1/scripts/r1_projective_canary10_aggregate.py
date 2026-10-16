#!/usr/bin/env python3
"""Account and audit split-qualified Projective path-first ten-scene results."""
from __future__ import annotations
import argparse, json, math, statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any
import numpy as np
from r1_canonical_tasks import score_observation

def jsonl(p: Path): return [json.loads(x) for x in p.open() if x.strip()]
def dist(v):
    v=sorted(float(x) for x in v if x is not None)
    return {"count":len(v)} if not v else {"count":len(v),"min":v[0],"median":statistics.median(v),"mean":statistics.mean(v),"max":v[-1]}
def areas(item, pose):
    r=score_observation(item, np.asarray(pose,dtype=float)); o=r["visual_metrics"].get("objects") or []
    return [float(x.get("area_ratio",0.0) or 0.0) for x in o]
def yaw_delta(a,b):
    fa=np.asarray(a,dtype=float)[:3,2]; fb=np.asarray(b,dtype=float)[:3,2]
    aa=math.atan2(float(fa[1]),float(fa[0])); bb=math.atan2(float(fb[1]),float(fb[0]))
    return abs((bb-aa+math.pi)%(2*math.pi)-math.pi)*180/math.pi
def main():
 p=argparse.ArgumentParser(); p.add_argument('--selection',type=Path,required=True); p.add_argument('--prototype',type=Path,required=True); p.add_argument('--runtime',type=Path,required=True); p.add_argument('--observability',type=Path,required=True); p.add_argument('--reachability',type=Path,required=True); p.add_argument('--output',type=Path,required=True); a=p.parse_args()
 sel={(x['split'],int(x['source_row_index'])):x for x in json.loads(a.selection.read_text())['records']}
 gen={(x['split'],int(x['source_row_index'])):x for x in json.loads(a.prototype.read_text())['rows']}
 reach={x['task_id']:x for x in jsonl(a.reachability)}
 runtime={x['task_id']:x for x in json.loads(a.runtime.read_text())['results']}
 rgb={x['task_id']:x for x in jsonl(a.observability)}
 if set(sel)!=set(gen): raise ValueError('selection/generation accounting mismatch')
 rows=[]
 for key,s in sorted(sel.items()):
  g=gen[key]; cand=g.get('status')=='candidate_found'; item=g.get('row',{}); tid=item.get('task_id'); rr=runtime.get(tid,{}); ob=rgb.get(tid,{}); final=bool(cand and rr.get('status')=='pass' and ob.get('passed') is True)
  d=g.get('details',{}); im=d.get('initial_metric',{}); tm=d.get('target_metric',{}); rd=d.get('repaired_difficulty',{})
  difficulty={}
  if cand:
   path=reach[tid]['path']; initial=np.asarray(path[0]['c2w']); terminal=np.asarray(path[-1]['c2w']); ia=areas(item,initial); ta=areas(item,terminal)
   difficulty={'initial_to_terminal_translation_m':float(np.linalg.norm(initial[:2,3]-terminal[:2,3])),'yaw_change_deg':yaw_delta(initial,terminal),'initial_bbox_area_ratios':ia,'initial_min_bbox_area_ratio':min(ia) if ia else None,'terminal_bbox_area_ratios':ta,'terminal_min_bbox_area_ratio':min(ta) if ta else None,'terminal_relation_margin_px':tm.get('relation_margin_px'),'certificate_action_length_upper_bound':reach[tid].get('steps'),'initial_relation_type':'reversed' if not im.get('gates',{}).get('relation') else 'correct_order_margin_insufficient'}
  rows.append({'split':key[0],'source_row_index':key[1],'scene_id':s['scene_id'],'old_status':s.get('old_status'),'old_failure':s.get('old_failure'),'old_failure_taxonomy':s.get('old_failure_taxonomy'),'old_replacement':s.get('old_replacement',False),'generation_status':g.get('status'),'safe_canonical_success_states':g.get('canonical_success_targets',0),'certificate':cand,'runtime_pass':rr.get('status')=='pass' if cand else False,'rgb_pass':ob.get('passed') if cand else None,'rgb_reasons':ob.get('reasons',[]),'final_status':'same_pair_accepted' if final else ('rgb_reject' if cand and rr.get('status')=='pass' else 'unverified'),'task_id':tid,'source_difficulty':g.get('source_difficulty'), 'repaired_difficulty':rd,'pose_difficulty':difficulty,'elapsed_seconds':g.get('elapsed_seconds',0.0)})
 def funnel(rs): return {'source':len(rs),'safe_canonical_success_region':sum(x['safe_canonical_success_states']>0 for x in rs),'certificate':sum(x['certificate'] for x in rs),'runtime_pass':sum(x['runtime_pass'] for x in rs),'rgb_pass':sum(x['rgb_pass'] is True for x in rs),'final_accepted':sum(x['final_status']=='same_pair_accepted' for x in rs)}
 scene={k:funnel([x for x in rows if x['scene_id']==k]) for k in sorted({x['scene_id'] for x in rows})}; split={k:funnel([x for x in rows if x['split']==k]) for k in sorted({x['split'] for x in rows})}
 accepted=[x for x in rows if x['final_status']=='same_pair_accepted']; recover=[x for x in accepted if x['old_status'] not in {'strict_same_pair_repair','count_matched_replacement'}]
 diffkeys=['initial_to_terminal_translation_m','yaw_change_deg','initial_min_bbox_area_ratio','terminal_min_bbox_area_ratio','terminal_relation_margin_px','certificate_action_length_upper_bound']
 payload={'version':'r1_projective_path_first_canary10_aggregate_v1','accounting_closed':len(rows)==len(sel),'funnel':funnel(rows),'per_scene':scene,'per_split':split,'status_counts':dict(Counter(x['final_status'] for x in rows)),'recovered_old_failures':{'count':len(recover),'by_old_status':dict(Counter(x['old_status'] for x in recover)),'by_taxonomy':dict(Counter(x['old_failure_taxonomy'] for x in recover))},'same_pair_vs_replacement':{'same_pair_accepted':len(accepted),'replacement_accepted':0,'old_replacement_to_new_same_pair':sum(x['old_replacement'] for x in accepted)},'difficulty':{'repaired_pose':{k:dist([x['pose_difficulty'].get(k) for x in accepted]) for k in diffkeys},'source':{k:dist([x['source_difficulty'].get(k) for x in accepted if x['source_difficulty']]) for k in ['translation_m','yaw_deg','bbox_area_ratio','relation_margin_px','planner_step_proxy']},'initial_relation_type':dict(Counter(x['pose_difficulty'].get('initial_relation_type') for x in accepted)),'one_step':sum(x['pose_difficulty'].get('certificate_action_length_upper_bound')==1 for x in accepted)},'cost':{'generation_wall_seconds':sum(float(x['elapsed_seconds'] or 0) for x in rows)},'rows':rows}
 a.output.parent.mkdir(parents=True,exist_ok=True); a.output.write_text(json.dumps(payload,indent=2)+'\n'); print(json.dumps({k:payload[k] for k in ['accounting_closed','funnel','status_counts','recovered_old_failures','same_pair_vs_replacement','difficulty','cost']},indent=2))
if __name__=='__main__': main()
