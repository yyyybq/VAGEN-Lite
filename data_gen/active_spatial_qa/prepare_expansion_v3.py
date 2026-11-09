"""Prepare independently versioned goals and legal camera states; no GPU/model."""
from __future__ import annotations
import argparse, concurrent.futures, hashlib, json, os, subprocess, sys, time, traceback
from collections import Counter
from pathlib import Path

def loadl(p): return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def savej(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def savel(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(''.join(json.dumps(v,ensure_ascii=False,allow_nan=False)+'\n' for v in x));tmp.replace(p)
def derive(payload):
    scene,cfg,out=payload
    from data_gen.active_spatial_qa.qa_derived_goals_v3 import derive_scene_goals
    from data_gen.active_spatial_qa.production_v3 import context,legal
    import numpy as np
    source=loadl(cfg['parent_source_tasks']);templates=[x for x in source if x['scene_id']==scene]
    ctx=context(cfg,scene)
    positions=[]
    for xy in ctx[1].candidate_points(cfg['sampling']['grid_spacing_m']):
        p=np.array([*xy,cfg['sampling']['camera_height_m']])
        if legal(cfg,ctx,p)['status']=='legal_indoor':positions.append(p)
    goals,audit=derive_scene_goals(cfg,scene,templates,ctx=ctx,legal_positions=positions,policy=cfg.get('target_pair_policy'))
    directory=Path(out)/'derived_by_scene';savel(directory/(scene+'.jsonl'),goals);savej(directory/(scene+'.audit.json'),audit)
    return {'scene':scene,'goals':len(goals),'legal_positions':len(positions)}
def plan_scene(payload):
    scene,cfg,out=payload
    directory=Path(out)/'planning'/scene;directory.mkdir(parents=True,exist_ok=True)
    config=dict(cfg,scene_splits={scene:cfg['scene_splits'][scene]})
    savej(directory/'requested_distribution_config.json',config)
    command=[sys.executable,'-m','data_gen.active_spatial_qa.production_v3','plan','--config',str(directory/'requested_distribution_config.json'),'--output',str(directory)]
    started=time.time()
    with (directory/'planning.log').open('w') as log:
        process=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
    status={'scene':scene,'ok':process.returncode==0,'returncode':process.returncode,'command':command,'seconds':time.time()-started}
    if status['ok']:
        report=json.loads((directory/'planning_statistics.json').read_text());status.update(planned=report['planned'],by_task_answer=report['by_task_answer'])
    savej(directory/'status.json',status)
    return status
def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--output',required=True);p.add_argument('--workers',type=int,default=6);a=p.parse_args()
    out=Path(a.output);cfg=json.loads(Path(a.config).read_text());out.mkdir(parents=True,exist_ok=True)
    if (out/'planned_states.jsonl').exists():raise ValueError('Existing immutable plan; use render to resume')
    status={'started':time.time(),'ok':False,'phase':'deriving','derived':[],'planned':[]}
    savej(out/'preparation_status.json',status)
    try:
        pending=[]
        for scene in cfg['scene_splits']:
            destination=out/'derived_by_scene'/(scene+'.jsonl')
            if destination.exists():status['derived'].append({'scene':scene,'goals':len(loadl(destination)),'reused':True})
            else:pending.append((scene,cfg,str(out)))
        with concurrent.futures.ProcessPoolExecutor(max_workers=a.workers) as pool:
            for item in pool.map(derive,pending):
                status['derived'].append(item);savej(out/'preparation_status.json',status);print(json.dumps({'phase':'derived',**item}),flush=True)
        goals=[]
        for scene in cfg['scene_splits']:goals.extend(loadl(out/'derived_by_scene'/(scene+'.jsonl')))
        if len({g['task_id'] for g in goals})!=len(goals):raise ValueError('Duplicate derived goal IDs')
        savel(out/'derived_goals.jsonl',goals)
        cfg['source_tasks']=str((out/'derived_goals.jsonl').resolve());savej(out/'distribution_config.json',cfg)
        status['phase']='planning';savej(out/'preparation_status.json',status)
        pending=[]
        for scene in cfg['scene_splits']:
            directory=out/'planning'/scene
            if (directory/'planned_states.jsonl').exists():
                # Only a previously completed plan is reusable; never a failed process.
                prior=json.loads((directory/'status.json').read_text())
                if not prior.get('ok'):raise ValueError('Incomplete prior scene plan requires a new attempt directory')
                status['planned'].append(dict(prior,reused=True))
            else:pending.append((scene,cfg,str(out)))
        with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
            futures=[pool.submit(plan_scene,item) for item in pending]
            for future in concurrent.futures.as_completed(futures):
                item=future.result();status['planned'].append(item);savej(out/'preparation_status.json',status);print(json.dumps({'phase':'planned',**item}),flush=True)
        if any(not x['ok'] for x in status['planned']):raise RuntimeError('At least one scene preparation failed; no aggregate success')
        rows=[];failures=[];rejections=[];assets={};counts=Counter()
        for scene in cfg['scene_splits']:
            directory=out/'planning'/scene;rows.extend(loadl(directory/'planned_states.jsonl'));failures.extend(loadl(directory/'failures.jsonl'));rejections.extend(loadl(directory/'visibility_rejection_examples.jsonl'))
            assets.update(json.loads((directory/'asset_provenance.json').read_text()));counts.update(json.loads((directory/'planning_statistics.json').read_text())['counts'])
        if len({x['sample_id'] for x in rows})!=len(rows):raise ValueError('Duplicate aggregate observation IDs')
        if len(rows)>cfg['distribution']['independent_inputs_target']:raise ValueError('Plan exceeds explicitly configured target')
        savel(out/'planned_states.jsonl',rows);savel(out/'failures.jsonl',failures);savel(out/'visibility_rejection_examples.jsonl',rejections);savej(out/'asset_provenance.json',assets)
        savej(out/'planning_statistics.json',{'counts':dict(counts),'planned':len(rows),'by_task_answer':dict(Counter(r['task_type']+'/'+r['branch_intent'] for r in rows)),'by_scene':dict(Counter(r['scene_id'] for r in rows)),'requested':cfg['distribution'],'labels_computed':False,'derived_goals':len(goals),'scene_plan_status':status['planned']})
        savej(out/'split_manifest.json',{'version':3,'split_unit':'scene','scene_splits':cfg['scene_splits'],'applies_to':['QA','Action','mixed'],'legacy_trajectory_policy':'No existing Action trajectories are imported. Future exports must exclude every val/test scene from Action train.'})
        status.update(ok=True,phase='succeeded',actual_independent_candidates=len(rows),derived_goals=len(goals))
    except BaseException:
        status.update(ok=False,phase='failed',error=traceback.format_exc());raise
    finally:status['finished']=time.time();savej(out/'preparation_status.json',status)
if __name__=='__main__':main()
