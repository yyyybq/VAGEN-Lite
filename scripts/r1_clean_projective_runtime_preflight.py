#!/usr/bin/env python3
"""Replay audit certificates through the exact policy manifest and PPO env."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
from r1_reconcile_expansion_inventory import read, evidence, digest
from r1_run_canonical_dev_eval32 import env_config
from vagen.envs.active_spatial.env import ActiveSpatialEnv


def main():
    p=argparse.ArgumentParser();p.add_argument('--frozen',type=Path,required=True);p.add_argument('--renderer-url',required=True)
    p.add_argument('--gs-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    items=read(a.frozen/'train.jsonl');audits=read(a.frozen/'audit_only.jsonl')
    assert len(items)==len(audits)==210
    config=env_config(SimpleNamespace(renderer_url=a.renderer_url,gs_root=a.gs_root),a.frozen/'train.jsonl')
    env=ActiveSpatialEnv(config);results=[]
    try:
        for i in sorted(range(len(items)),key=lambda n:(items[n]['scene_id'],n)):
            item,m=items[i],audits[i]
            obs,_=env.reset(seed=i)
            assert env.current_item['task_id']==item['task_id']
            metric=env._calculate_canonical_metric();assert metric and not metric['success']
            images=[im for ims in obs['multi_modal_data'].values() for im in ims]
            assert len(images)==1
            original=Path(m['evidence']['rgb_frames'][0])
            old=np.asarray(Image.open(original).convert('RGB'),dtype=np.int16)
            new=np.asarray(images[0].convert('RGB'),dtype=np.int16)
            assert old.shape==new.shape
            error=np.abs(old-new);mae=float(error.mean());p99=float(np.quantile(error,.99))
            assert mae<=.5 and p99<=2,(i,'initial RGB mismatch',mae,p99)
            assert 'Distance to target:' not in str(obs)
            reach,_=evidence(m['evidence']['reachability'],m['task_id'])
            step_records=[]
            for step,action in enumerate(reach['actions'],1):
                obs,reward,done,info=env.step('<action>'+action+'</action>')
                state=env.view_engine.get_pose();expected=np.asarray(reach['path'][step]['c2w'])
                assert np.allclose(state,expected,atol=1e-6,rtol=0),(i,step,'pose mismatch')
                assert env.collision_count==0
                metric=env._calculate_canonical_metric()
                assert np.isfinite(reward)
                assert int(env._current_step)==step
                assert bool(done)==bool(metric['success'])
                assert bool(metric['success'])==(step==len(reach['actions']))
                step_records.append({'step':step,'action':action,'reward':float(reward),'canonical':metric,'done':bool(done)})
            results.append({'index':i,'source_key':m['source_key'],'policy_task_id':item['task_id'],'status':'PASS',
                            'mae':mae,'p99':p99,'first_success_step':len(reach['actions']),'steps':step_records})
            tmp=a.output.with_suffix('.tmp');tmp.parent.mkdir(parents=True,exist_ok=True)
            tmp.write_text(json.dumps({'status':'RUNNING','completed':len(results),'results':results},indent=2)+'\n');tmp.replace(a.output)
    finally:env.close()
    a.output.write_text(json.dumps({'status':'PASS','completed':len(results),'manifest_sha256':digest(a.frozen/'train.jsonl'),
                                    'policy_input_contract':'current RGB, task, pose; no terminal-distance hint or certificate',
                                    'results':results},indent=2)+'\n')


if __name__=='__main__':main()
