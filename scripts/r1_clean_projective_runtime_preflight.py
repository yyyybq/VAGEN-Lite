#!/usr/bin/env python3
"""Replay audit certificates through the exact policy manifest and PPO env."""
import argparse
import json
import copy
import os
from dataclasses import fields
from pathlib import Path

import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from r1_reconcile_expansion_inventory import read, evidence, digest
from vagen.envs.active_spatial.env import ActiveSpatialEnv
from vagen.envs.active_spatial.env_config import ActiveSpatialEnvConfig


def main():
    p=argparse.ArgumentParser();p.add_argument('--frozen',type=Path,required=True);p.add_argument('--renderer-url',required=True)
    p.add_argument('--gs-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    items=read(a.frozen/'train.jsonl');audits=read(a.frozen/'audit_only.jsonl')
    assert len(items)==len(audits)==210
    os.environ['R1_RENDER_URL']=a.renderer_url
    raw=OmegaConf.to_container(OmegaConf.load(a.frozen/'train.yaml'),resolve=True)['envs'][0]['config']
    assert raw['gs_root']==str(a.gs_root) and raw['jsonl_path']==str(a.frozen/'train.jsonl')
    keys={f.name for f in fields(ActiveSpatialEnvConfig)}
    config=ActiveSpatialEnvConfig(**{k:v for k,v in raw.items() if k in keys})
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
            if not original.is_absolute():original=a.frozen.parents[3]/original
            old=np.asarray(Image.open(original).convert('RGB'),dtype=np.int16)
            new=np.asarray(images[0].convert('RGB'),dtype=np.int16)
            assert old.shape==new.shape
            error=np.abs(old-new);mae=float(error.mean());p99=float(np.quantile(error,.99))
            assert mae<=.5 and p99<=2,(i,'initial RGB mismatch',mae,p99)
            assert 'Distance to target:' not in str(obs)
            # Prove environment-only sampled terminal metadata does not alter
            # the actual policy text. Preserve it for the original reward.
            original_item=env.current_item
            sanitized=copy.deepcopy(original_item)
            for key in ('sample_point','sample_forward','height'):
                sanitized['target_region'].pop(key,None)
            env.current_item=sanitized
            stripped_obs=env._build_observation_from_image(images[0],init_obs=True,task_prompt=env._build_task_prompt(sanitized))
            env.current_item=original_item
            assert stripped_obs['obs_str']==obs['obs_str'],(i,'terminal metadata leaked into policy text')
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
                step_records.append({'step':step,'action':action,'reward':float(reward),'canonical':metric,'done':bool(done)})
                if metric['success']:
                    break
            assert step_records and step_records[-1]['canonical']['success']
            if m['certified_bucket']=='medium_certified':assert 4<=len(step_records)<=6
            results.append({'index':i,'source_key':m['source_key'],'policy_task_id':item['task_id'],'status':'PASS',
                            'mae':mae,'p99':p99,'first_success_step':len(step_records),
                            'historical_certificate_length':len(reach['actions']),
                            'certified_bucket':('easy_certified' if len(step_records)<=3 else m['certified_bucket']),
                            'steps':step_records})
            tmp=a.output.with_suffix('.tmp');tmp.parent.mkdir(parents=True,exist_ok=True)
            tmp.write_text(json.dumps({'status':'RUNNING','completed':len(results),'results':results},indent=2)+'\n');tmp.replace(a.output)
    finally:env.close()
    a.output.write_text(json.dumps({'status':'PASS','completed':len(results),'manifest_sha256':digest(a.frozen/'train.jsonl'),
                                    'policy_input_contract':'current RGB, task, pose; no terminal-distance hint or certificate',
                                    'results':results},indent=2)+'\n')


if __name__=='__main__':main()
