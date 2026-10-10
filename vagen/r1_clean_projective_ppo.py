"""Original PPO trainer with a fixed experiment endpoint, not a new scheduler.

Workers and optimizers are initialized with the historical 700-step horizon.
Only fit's stopping counter is shortened. The smoke runs in a separate process.
"""
import argparse
import json
import os
from pathlib import Path

import ray
from omegaconf import OmegaConf, open_dict
from vagen import main_ppo
from vagen.ray_trainer import RayPPOTrainer


class FixedEndpointTrainer(RayPPOTrainer):
    def _save_checkpoint(self):
        endpoint = int(os.environ["R1_PPO_ENDPOINT"])
        # The reward pilot needs only the three evaluation snapshots.  Avoid a
        # fourth 7B checkpoint at step 6 when save_freq=2 fires normally.
        if endpoint == 8 and self.global_steps not in (2, 4, 8):
            return
        return super()._save_checkpoint()

    def fit(self):
        endpoint=int(os.environ['R1_PPO_ENDPOINT'])
        resume_from=os.environ.get('R1_PPO_RESUME_FROM')
        assert self.total_training_steps==700
        assert self.config.actor_rollout_ref.actor.optim.total_training_steps==700
        assert self.config.critic.optim.total_training_steps==700
        assert self.config.trainer.resume_mode=='disable'
        assert endpoint in (1, 8, 250)
        if resume_from:
            checkpoint=Path(resume_from).resolve()
            assert endpoint==250, 'infrastructure resume is only valid for the formal endpoint'
            assert checkpoint.name.startswith('global_step_')
            assert (checkpoint/'COMPLETE').is_file(), 'resume checkpoint is not atomically complete'
            assert int((checkpoint/'COMPLETE').read_text().strip()) < endpoint
            with open_dict(self.config):
                self.config.trainer.resume_mode='resume_path'
                self.config.trainer.resume_from_path=str(checkpoint)
        self.total_training_steps=endpoint
        super().fit()
        checkpoint=Path(self.config.trainer.default_local_dir)/f'global_step_{endpoint}'
        assert (checkpoint/'COMPLETE').is_file(), 'checkpoint not atomically complete'
        if endpoint == 8:
            for step in (2, 4, 8):
                assert (
                    Path(self.config.trainer.default_local_dir)
                    / f"global_step_{step}"
                    / "COMPLETE"
                ).is_file(), f"missing evaluation checkpoint at step {step}"
        if endpoint==1:
            # Exercise the actual distributed model/optimizer/dataloader loader.
            # These weights are discarded when this separate process exits.
            with open_dict(self.config):
                self.config.trainer.resume_mode='resume_path'
                self.config.trainer.resume_from_path=str(checkpoint)
            self._load_checkpoint()
            assert self.global_steps==1
            report={'checkpoint_save_and_load':'PASS','checkpoint':str(checkpoint),
                    'scheduler_horizon':700,'updates':1,'formal_initialization':'fresh_pretrained_in_new_process'}
            (checkpoint.parent.parent/'checkpoint_reload.json').write_text(json.dumps(report,indent=2)+'\n')


class CleanTaskRunner(main_ppo.TaskRunner):
    def run(self, config):
        main_ppo.RayPPOTrainer=FixedEndpointTrainer
        return super().run(config)


def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--endpoint',type=int,required=True)
    a=p.parse_args();assert a.endpoint in (1,8,250)
    os.environ['R1_PPO_ENDPOINT']=str(a.endpoint)
    cfg=OmegaConf.load(a.config)
    assert cfg.algorithm.adv_estimator=='no_concat_gae' and not cfg.trainer.concat_multi_turn
    assert cfg.actor_rollout_ref.actor.optim.lr_scheduler_type=='cosine'
    assert cfg.trainer.total_training_steps==700
    with open_dict(cfg):
        cfg.ray_kwargs.ray_init.include_dashboard=False
        cfg.ray_kwargs.ray_init.runtime_env={'env_vars':{
            'R1_PPO_ENDPOINT':str(a.endpoint),'R1_RENDER_URL':os.environ['R1_RENDER_URL'],
            'R1_PPO_RESUME_FROM':os.environ.get('R1_PPO_RESUME_FROM',''),
            'PYTHONPATH':os.environ['PYTHONPATH'],'WANDB_MODE':os.environ.get('WANDB_MODE','offline'),
            'PYTHONDONTWRITEBYTECODE':'1','NO_PROXY':'*','no_proxy':'*',
            'LD_LIBRARY_PATH':os.environ.get('LD_LIBRARY_PATH','')}}
    root=Path(cfg.trainer.default_local_dir).parent;root.mkdir(parents=True,exist_ok=True)
    OmegaConf.save(cfg,root/'launch_resolved.yaml',resolve=True)
    try:
        main_ppo.run_ppo(cfg, task_runner_class=ray.remote(num_cpus=1)(CleanTaskRunner))
    finally:
        ray.shutdown()


if __name__=='__main__':main()
