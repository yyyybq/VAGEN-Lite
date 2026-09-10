#!/usr/bin/env python3
"""Medium-only v2: reject every <=3-step shortcut before accepting an initial."""
from __future__ import annotations
import argparse, json, time
from collections import Counter, deque
from pathlib import Path
import numpy as np
from r1_action_graph import ACTIONS, TRANSLATION_ACTIONS, forward_transition
from r1_canonical_tasks import canonical_projective, score_observation
from r1_projective_difficulty_conditioned import (collect_reverse_candidates, read_jsonl, write_json)
from r1_projective_path_first_prototype import make_candidate, pair_midpoint, success_region_points, target_pose
from r1_reachability_audit import state_key
from r1_repair_pipeline import SceneConstraints

VERSION="projective_difficulty_conditioned_medium_selector_v2"

def shortcut_probe(item, start, detector, cap):
    nodes=[(np.asarray(start,float),-1,None)]; q=deque([(0,0)]); seen={state_key(start)}; expansions=0; collisions=Counter()
    while q:
        idx, depth=q.popleft()
        if depth>=3: continue
        if expansions>=cap: return {"complete":False,"reason":"difficulty_unverified_expansion_cap","expansions":expansions,"visited_states":len(seen)}
        expansions+=1; pose=nodes[idx][0]
        for action in ACTIONS:
            nxt=forward_transition(pose,action)
            if action in TRANSLATION_ACTIONS:
                c=detector.check_collision(nxt[:3,3],previous_position=pose[:3,3])
                if c.has_collision: collisions[c.collision_type]+=1; continue
            k=state_key(nxt)
            if k in seen: continue
            seen.add(k); ni=len(nodes); nodes.append((nxt,idx,action))
            metric=canonical_projective(score_observation(item,nxt))
            if metric["success"]:
                chain=[]; cur=ni
                while nodes[cur][1]>=0: chain.append(nodes[cur][2]); cur=nodes[cur][1]
                chain.reverse()
                return {"complete":True,"shortcut_found":True,"shortcut_depth":depth+1,"shortcut_actions":chain,"nearest_discovered_success_state":{"pose_c2w":nxt.tolist(),"metric":metric},"expansions":expansions,"visited_states":len(seen),"collision_rejections":dict(collisions)}
            q.append((ni,depth+1))
    return {"complete":True,"shortcut_found":False,"shortcut_depth":None,"shortcut_actions":[],"expansions":expansions,"visited_states":len(seen),"collision_rejections":dict(collisions)}

def first_success(item, initial):
    pose=np.asarray(initial["pose"],float)
    for n,action in enumerate(initial["actions"],1):
        pose=forward_transition(pose,action)
        if canonical_projective(score_observation(item,pose))["success"]: return n
    return None

def one(item, rec, constraints, cfg):
    t=time.time(); layout,_=constraints.scene(str(item.get("scene_id") or "")); room=constraints.room_index(layout,pair_midpoint(item)[:2])
    if room is None: room=constraints.validate(item,np.asarray(item["init_camera"]["extrinsics"])[:3,3],check_pair_distance=False).get("room_index")
    targets=[]
    for point in success_region_points(item,constraints,room):
        if not constraints.validate(item,point,initial_room_index=room,check_pair_distance=False).get("success"): continue
        for yaw in (0.,-5.,5.,-10.,10.,-15.,15.):
            pose=target_pose(item,point,yaw); metric=canonical_projective(score_observation(item,pose))
            if metric["success"]: targets.append((point,pose,metric))
    targets=targets[:cfg.seed_cap]; pool=[]; rev=[]
    for si,(point,target,metric) in enumerate(targets):
        r=collect_reverse_candidates(item,target,constraints,room,max_steps=12,max_expansions=cfg.per_seed_expansions,candidate_cap=cfg.candidate_cap)
        rev.append({"seed_index":si,"status":r["status"],"expansions":r["expansions"],"visited_states":r["visited_states"],"candidate_count":len(r["candidates"]),"rejection_reasons":r.get("rejection_reasons",{})})
        pool += [(point,target,metric,c) for c in r["candidates"]]
    detector=constraints._collision_cache[str(item.get("scene_id") or "")]
    eligible=[]
    for p,target,metric,c in pool:
        upper=first_success(item,c)
        if upper is not None and 4<=upper<=6: eligible.append((abs(upper-5),tuple(state_key(c["pose"])),p,target,metric,{**c,"actions":c["actions"][:upper],"steps":upper}))
    eligible.sort(); events=[]; accepted=None; unverified=False
    for _,_,p,target,metric,c in eligible:
        probe=shortcut_probe(item,c["pose"],detector,cfg.lower_expansions)
        event={"certificate_upper_bound":c["steps"],"initial_metric":c["metric"],"shortcut_probe":probe}
        events.append(event)
        if not probe["complete"]: unverified=True; continue
        if probe["shortcut_found"]: continue
        accepted=(p,target,metric,c,probe); break
    base={"version":VERSION,"split":rec["split"],"source_row_index":rec["source_row_index"],"scene_id":rec["scene_id"],"requested_bucket":"medium","same_pair_only":True,"success_region_seeds_considered":len(targets),"reverse_seed_cap":cfg.seed_cap,"per_seed_expansion_cap":cfg.per_seed_expansions,"candidate_cap_per_seed":cfg.candidate_cap,"reverse_attempts":rev,"initial_candidates_collected":len(pool),"eligible_medium_candidates":len(eligible),"pre_screen_events":events,"source_action_length":rec["source_action_length"],"baseline_certificate_upper_bound":rec["baseline_certificate_upper_bound"]}
    if accepted:
        p,target,tm,c,probe=accepted; row=make_candidate(int(rec["source_row_index"]),item,p,target,c,split=str(rec["split"])); row["generator_version"]=VERSION; row["reachability_construction"].update({"requested_bucket":"medium","certificate_upper_bound":c["steps"],"certified_lower_bound":4,"lower_bound_complete":True,"first_success_step":c["steps"]})
        out={**base,"status":"difficulty_certified_candidate","certificate_upper_bound":c["steps"],"certified_lower_bound":4,"lower_bound_complete":True,"runtime_shortcut_found":False,"initial_metric":c["metric"],"target_metric":tm,"row":row,"certificate":c}
    elif eligible and not unverified: out={**base,"status":"shortcut_rejected","runtime_shortcut_found":True}
    elif unverified: out={**base,"status":"difficulty_unverified"}
    else: out={**base,"status":"requested_bucket_not_found_within_budget"}
    out["elapsed_seconds"]=time.time()-t; return out

def main():
 p=argparse.ArgumentParser();p.add_argument('--selection',type=Path,required=True);p.add_argument('--sources',type=Path,required=True);p.add_argument('--gs-root',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--seed-cap',type=int,default=12);p.add_argument('--per-seed-expansions',type=int,default=512);p.add_argument('--candidate-cap-per-seed',type=int,default=48);p.add_argument('--lower-expansions',type=int,default=100000);a=p.parse_args();
 sel=json.loads(a.selection.read_text()); raw=json.loads(a.sources.read_text()); sm=raw.get('sources',raw); src={k:read_jsonl(Path(v['path'] if isinstance(v,dict) else v)) for k,v in sm.items()}; a.output_dir.mkdir(parents=True,exist_ok=True); cfg=type('Cfg',(),{'seed_cap':a.seed_cap,'per_seed_expansions':a.per_seed_expansions,'candidate_cap':a.candidate_cap_per_seed,'lower_expansions':a.lower_expansions})(); cons=SceneConstraints(a.gs_root); rows=[]
 for rec in sel['records']:
  try:r=one(src[rec['split']][int(rec['source_row_index'])],rec,cons,cfg)
  except Exception as e:r={"version":VERSION,"split":rec['split'],"source_row_index":rec['source_row_index'],"scene_id":rec['scene_id'],"requested_bucket":"medium","status":"implementation_error","error":repr(e)}
  rows.append(r);write_json(a.output_dir/'checkpoint.json',{'version':VERSION,'results':rows});print(json.dumps({k:r.get(k) for k in ('split','source_row_index','status','eligible_medium_candidates','elapsed_seconds')}),flush=True)
 payload={'version':VERSION,'selection':str(a.selection),'same_pair_only':True,'budgets':{'seed_cap':a.seed_cap,'per_seed_expansions':a.per_seed_expansions,'candidate_cap_per_seed':a.candidate_cap_per_seed,'lower_expansions':a.lower_expansions},'results':rows,'status_counts':dict(Counter(x['status'] for x in rows))};write_json(a.output_dir/'selector_results.json',payload)
if __name__=='__main__': main()
