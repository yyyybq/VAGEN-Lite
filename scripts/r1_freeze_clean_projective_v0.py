#!/usr/bin/env python3
"""Freeze the accepted Projective support subset and the historical v46 recipe."""
import argparse
import copy
import json
import os
import tempfile
from collections import Counter
from pathlib import Path

from omegaconf import OmegaConf
from r1_reconcile_expansion_inventory import digest, evidence, fingerprint, read


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    a = p.parse_args()
    root = a.root.resolve()
    b = root/'exps/vagen_active_spatial/r1_h1_aoss_repair_20260905'
    run = root/'exps/vagen_active_spatial/R1-clean-Projective-v0'
    final = run/'frozen'
    assert not final.exists(), 'v0 is immutable; refuse to refreeze'
    inventory = b/'r1_reconciled_inventory_v2_20260930'
    manifest = inventory/'projective_train_ready_inventory.jsonl'
    assert digest(manifest) == '0eec778c2cdff7bab85b7c8d313a00b517d6bcd7d9fa5d30ac973324e6b1311b'
    records = read(manifest)
    assert len(records)==210 and len({r['episode_fingerprint'] for r in records})==210
    protocol = read(b/'r1_canonical_dev_eval32_20260915/frozen_input/eval_protocol.json')
    eval_policy = Path(protocol['policy_input']['path'])
    assert digest(eval_policy)==protocol['policy_input']['sha256']
    eval_audit = Path(protocol['audit_manifest']['path'])
    assert digest(eval_audit)==protocol['audit_manifest']['sha256']
    local = b/'r1_independent_local_action_eval60_v3_20260916/frozen_input/parent_sources.jsonl'
    forbidden = read(eval_audit)+read(local)
    eval_scenes = {r['scene_id'] for r in forbidden}
    eval_sources = {r.get('source_key') for r in forbidden}
    policy, audits = [], []
    for m in records:
        assert not m['fov_quarantined'] and m['task_type']=='projective_relations'
        assert m['scene_id'] not in eval_scenes
        assert m['source_identity'].get('original_source_key') not in eval_sources
        refs=m['evidence']; task_id=m['task_id']
        candidate,_=evidence(refs['candidate'],task_id)
        runtime,_=evidence(refs['runtime'],task_id)
        rgb,_=evidence(refs['official_rgb'],task_id)
        reach,_=evidence(refs['reachability'],task_id)
        assert runtime['status']=='pass' and runtime['initial_success'] is False
        assert not runtime['collisions'] and not runtime['pose_mismatches'] and rgb['passed'] is True
        differing={k for k in candidate.keys()|m['candidate'].keys() if candidate.get(k)!=m['candidate'].get(k)}
        assert differing <= {'collision_convention','source_identity','reachability_construction','camera_model_version','canonical_task_metric_version'}, differing
        assert candidate['init_camera']['extrinsics']==reach['path'][0]['c2w']
        assert len(reach['actions'])==runtime['steps'] and 1<=runtime['steps']<=12
        # Sampled terminal and certificates are audit-only. The formal scorer
        # needs the region definition and object geometry, not a chosen target.
        keep=('task_type','scene_id','init_camera','target_object','target_region','task_description','preset','object_label','canonical_task_metric_version')
        row={k:copy.deepcopy(candidate[k]) for k in keep if k in candidate}
        for k in ('sample_point','sample_forward','height'):
            row['target_region'].pop(k,None)
        row['camera_model_version']='canonical_h1_from_frozen_candidate_intrinsics_and_pose'
        assert rgb['frames'][0]['canonical_metric']['metric_version']=='canonical_spatial_task_h1_v1'
        row['canonical_task_metric_version']='canonical_spatial_task_h1_v1'
        row['collision_convention']={k:runtime['collision_convention'][k] for k in ('version','structure_y_sign')}
        row['task_id']='r1cpv0_'+m['episode_fingerprint'][:24]
        assert row['canonical_task_metric_version']=='canonical_spatial_task_h1_v1'
        policy.append(row)
        lower=m['difficulty']['certified_lower_bound']; upper=runtime['steps']
        if upper<=3:
            bucket='easy_certified'
        elif m['difficulty']['lower_bound_complete'] and lower>=4 and upper<=6:
            bucket='medium_certified'
        else:
            bucket='unknown'
        audits.append({**m,'policy_task_id':row['task_id'],'policy_row_sha256':fingerprint(row),
                       'certified_bucket':bucket,'actions_audit_only':reach['actions']})
    assert len({r['task_id'] for r in policy})==len(policy)
    run.mkdir(parents=True,exist_ok=True)
    staging=Path(tempfile.mkdtemp(prefix='.freeze-',dir=run))
    def js(name,obj): (staging/name).write_text(json.dumps(obj,indent=2,sort_keys=True)+'\n')
    def jl(name,rs): (staging/name).write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rs))
    jl('train.jsonl',policy);jl('audit_only.jsonl',audits)
    (staging/'eval_policy_rows.jsonl').write_bytes(eval_policy.read_bytes())
    js('asset_sources.json',{'train':str(final/'train.jsonl'),'development_regression':str(final/'eval_policy_rows.jsonl')})
    historical=root/'exps/vagen_active_spatial/v46_baseline_qwen25vl_7b'
    train_old=OmegaConf.to_container(OmegaConf.load(historical/'train.yaml'),resolve=True)
    env=copy.deepcopy(train_old['envs'][0]['config'])
    env.update(gs_root=str(run/'assets/ready'),client_url='${oc.env:R1_RENDER_URL}',
               enable_distance_in_obs=False,max_episode_steps=12,turn_budget=12,
               train_size=210,test_size=0,render_fail_fast=True)
    for name,n,file in [('train',210,'train.jsonl'),('val',32,'eval_policy_rows.jsonl')]:
        spec=copy.deepcopy(train_old['envs'][0]);spec.update(n_envs=n,seed_list=list(range(n)),max_turns=12)
        spec['config']=copy.deepcopy(env);spec['config'].update(jsonl_path=str(final/file),train_size=n,test_size=0)
        OmegaConf.save(OmegaConf.create({'envs':[spec]}),staging/f'{name}.yaml')
    old_cfg=OmegaConf.load(historical/'hydra_run/.hydra/config.yaml')
    cfg=copy.deepcopy(old_cfg)
    model=b/'r1_canonical_dev_eval32_20260915/model_restore/qwen25vl7b_pretrained_cc594898'
    model_sha=model.with_name(model.name+'_SHA256SUMS')
    assert digest(model_sha)=='46f05ffcc6127a4caa9a3e8c11ddf298b9a5263c8680afe6b5d017ea91702c8b'
    cfg.data.train_files=str(final/'train.yaml');cfg.data.val_files=str(final/'val.yaml')
    cfg.data.seed=20260930
    cfg.actor_rollout_ref.model.path=str(model);cfg.critic.model.path=str(model)
    cfg.actor_rollout_ref.rollout.agent.agent_loop_config_path='vagen/configs/agent_no_concat_active_spatial.yaml'
    cfg.trainer.experiment_name='R1-clean-Projective-v0'
    cfg.trainer.default_local_dir=str(run/'formal/checkpoints')
    cfg.trainer.validation_data_dir=str(run/'formal/validation')
    cfg.trainer.rollout_data_dir=str(run/'formal/rollout_data')
    cfg.trainer.logger=['console','wandb'];cfg.trainer.resume_mode='disable'
    cfg.trainer.resume_from_path=None
    assert cfg.trainer.total_training_steps==700 and cfg.trainer.critic_warmup==60
    assert cfg.actor_rollout_ref.rollout.tensor_model_parallel_size==4 and cfg.trainer.n_gpus_per_node==8
    OmegaConf.save(cfg,staging/'formal.yaml')
    smoke=copy.deepcopy(cfg)
    smoke.trainer.experiment_name='R1-clean-Projective-v0-preflight'
    smoke.trainer.critic_warmup=0;smoke.trainer.val_before_train=False;smoke.trainer.test_freq=-1
    smoke.trainer.logger=['console'];smoke.trainer.save_freq=1
    for key,suffix in [('default_local_dir','checkpoints'),('validation_data_dir','validation'),('rollout_data_dir','rollout_data')]:
        smoke.trainer[key]=str(run/'smoke'/suffix)
    OmegaConf.save(smoke,staging/'smoke.yaml')
    def leaves(x,prefix=''):
        if isinstance(x,dict):
            return {k:v for key,value in x.items() for k,v in leaves(value,prefix+'.'+key if prefix else key).items()}
        return {prefix:x}
    old_leaves=leaves(OmegaConf.to_container(old_cfg,resolve=False));new_leaves=leaves(OmegaConf.to_container(cfg,resolve=False))
    changes={k:{'old':old_leaves.get(k),'new':v} for k,v in new_leaves.items() if old_leaves.get(k)!=v}
    js('recipe_differences.json',{'resolved_config_changes':changes,
       'environment_changes':{'data':'Projective-only 210 immutable unique episodes','distance_hint':False,'primitive_action_cap':12,
         'camera_metric':'frozen canonical H1 runtime','sampling':'explicit one seed per unique row; shuffled across epochs; n=4 trajectories unchanged'},
       'loop_stop_step':250,'scheduler_horizon':700,'smoke_only_changes':{'critic_warmup':0,'stop_step':1,'save_freq':1,'evaluation':False},
       'formal_actor_updates_expected':191,'interpretation':'support-subset system/data repair experiment, not size-matched multitask v46 reproduction'})
    summary={'experiment':'R1-clean-Projective-v0','episodes':210,'retained_source_lineages':len({r['source_key'] for r in records}),
       'unique_pairs':len({r['pair_key'] for r in records}),'scenes':dict(Counter(r['scene_id'] for r in records)),
       'relations':dict(Counter(r['relation'] for r in records)),'category_pairs':dict(Counter(r['category_pair'] for r in records)),
       'certified_difficulty':dict(Counter(r['certified_bucket'] for r in audits)),
       'certificate_upper_bounds':dict(Counter(str(r['first_success_step']) for r in records)),
       'unknown_difficulty_policy':'upper bounds do not prove shortest paths; no old proxy=12 used',
       'isolation':{'scene_overlap':[],'source_overlap':[],'eval_excluded_scenes':sorted(eval_scenes),'local_action_sha256':digest(local),
                    'eval_policy_sha256':digest(eval_policy),'local_action_role':'permanent evaluation-only'},
       'data_gate':'PASS_EVIDENCE_AND_ISOLATION','training_preflight':'PENDING_LIVE_RUNTIME_AND_PPO',
       'seed':20260930,'stop_step':250,'scheduler_horizon':700,'batch_size':12,'trajectories_per_request':4,
       'independent_episode_count':210,'planned_parent_draws':3000,'planned_policy_trajectories':12000,
       'batches_per_epoch':17,'drop_last_rows_per_epoch':6,'total_epochs':30,
       'dynamic_additions':False,'running_schema_retry':'excluded_from_v0',
       'model':{'path':str(model),'sha256_manifest':str(model_sha),'sha256_manifest_digest':digest(model_sha),'worker_full_hash_check_required':True},
       'inputs':{str(manifest):digest(manifest),str(eval_audit):digest(eval_audit),str(historical/'train.yaml'):digest(historical/'train.yaml'),
                  str(historical/'hydra_run/.hydra/config.yaml'):digest(historical/'hydra_run/.hydra/config.yaml')},
       'legacy_corpus_size_balance_gates':'superseded by explicit user authorization for support-subset RL'}
    js('data_gate.json',summary)
    (staging/'SHA256SUMS').write_text(''.join(f'{digest(f)}  {f.name}\n' for f in sorted(staging.iterdir())))
    os.rename(staging,final)
    print(json.dumps({'frozen':str(final),'episodes':210,'certified_difficulty':summary['certified_difficulty'],'scenes':len(summary['scenes'])}))


if __name__=='__main__': main()
