"""Fixed 150-update endpoint for the S0/S1/S5 reward-only pilot.

The optimizer and scheduler keep the frozen 700-step production horizon.  Only
the fit endpoint is shortened.  Every branch starts in a fresh process from the
same pretrained actor and deterministic critic initialization.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import ray
from omegaconf import OmegaConf, open_dict

from vagen import main_ppo
from vagen.ray_trainer import RayPPOTrainer


ENDPOINT = 150
MILESTONES = (50, 100, 150)


class DenseScorePilotTrainer(RayPPOTrainer):
    def fit(self):
        assert self.total_training_steps == 700
        assert self.config.actor_rollout_ref.actor.optim.total_training_steps == 700
        assert self.config.critic.optim.total_training_steps == 700
        assert self.config.trainer.resume_mode == "disable"
        assert self.config.trainer.critic_warmup == 60
        assert self.config.trainer.save_freq == 50
        assert self.config.trainer.test_freq == 50
        assert self.config.trainer.val_before_train is True
        self.total_training_steps = ENDPOINT
        super().fit()
        root = Path(self.config.trainer.default_local_dir)
        for step in MILESTONES:
            checkpoint = root / f"global_step_{step}"
            assert (checkpoint / "COMPLETE").is_file(), f"missing complete checkpoint {checkpoint}"


class DenseScoreTaskRunner(main_ppo.TaskRunner):
    def run(self, config):
        main_ppo.RayPPOTrainer = DenseScorePilotTrainer
        return super().run(config)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    config = OmegaConf.load(args.config)
    assert config.algorithm.adv_estimator == "no_concat_gae"
    assert not config.trainer.concat_multi_turn
    assert config.algorithm.gamma == 0.95
    assert config.actor_rollout_ref.actor.optim.lr_scheduler_type == "cosine"
    assert config.trainer.total_training_steps == 700
    with open_dict(config):
        config.ray_kwargs.ray_init.include_dashboard = False
        # Keep Ray's Unix sockets on node-local storage (a long shared path can
        # exceed the socket path limit), while allowing the launcher to mirror
        # its logs into the persistent observability directory.
        ray_temp_dir = os.environ.get("DENSE_SCORE_RAY_TEMP_DIR")
        if ray_temp_dir:
            Path(ray_temp_dir).mkdir(parents=True, exist_ok=True)
            config.ray_kwargs.ray_init._temp_dir = ray_temp_dir
        config.ray_kwargs.ray_init.runtime_env = {
            "env_vars": {
                "R1_RENDER_URL": os.environ["R1_RENDER_URL"],
                "PYTHONPATH": os.environ["PYTHONPATH"],
                "WANDB_MODE": os.environ.get("WANDB_MODE", "offline"),
                "PYTHONDONTWRITEBYTECODE": "1",
                "NO_PROXY": "*",
                "no_proxy": "*",
                "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", ""),
            }
        }
    run_root = Path(config.trainer.default_local_dir).parent
    run_root.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, run_root / "launch_resolved.yaml", resolve=True)
    observability_root_text = os.environ.get("DENSE_SCORE_OBSERVABILITY_ROOT")
    if observability_root_text:
        observability_root = Path(observability_root_text)
        observability_root.mkdir(parents=True, exist_ok=True)
        (observability_root / "python_entry.json").write_text(
            json.dumps(
                {
                    "status": "ENTERED",
                    "utc": datetime.now(timezone.utc).isoformat(),
                    "pid": os.getpid(),
                    "config": str(Path(args.config).resolve()),
                    "ray_temp_dir": ray_temp_dir,
                    "optimizer_step_called_by_observer": False,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    try:
        main_ppo.run_ppo(config, task_runner_class=ray.remote(num_cpus=1)(DenseScoreTaskRunner))
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
