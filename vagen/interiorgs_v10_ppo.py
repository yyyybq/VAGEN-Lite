"""PPO for InteriorGS v10 (the dense-score pilot's runner without its R1-only checks).

    R1_RENDER_URL=http://HOST:PORT/render python -m vagen.interiorgs_v10_ppo --config RL/ppo.yaml --endpoint 250

The optimizer and scheduler keep the config's horizon (``trainer.total_training_steps``); ``--endpoint``
only stops ``fit`` after that many updates.  The env keeps its default (v10) success rules: this runner
never sets ``r1_legacy_success``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import ray
from omegaconf import OmegaConf, open_dict

from vagen import main_ppo
from vagen.ray_trainer import RayPPOTrainer


class EndpointTrainer(RayPPOTrainer):
    def fit(self):
        endpoint = int(os.environ["V10_PPO_ENDPOINT"])
        assert 0 < endpoint <= self.total_training_steps
        self.total_training_steps = endpoint
        super().fit()
        checkpoint = Path(self.config.trainer.default_local_dir) / f"global_step_{endpoint}"
        assert checkpoint.is_dir(), f"no checkpoint at the endpoint: {checkpoint}"


class V10TaskRunner(main_ppo.TaskRunner):
    def run(self, config):
        main_ppo.RayPPOTrainer = EndpointTrainer
        return super().run(config)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--endpoint", type=int, required=True)
    args = ap.parse_args()
    os.environ["V10_PPO_ENDPOINT"] = str(args.endpoint)
    cfg = OmegaConf.load(args.config)
    assert cfg.algorithm.adv_estimator == "no_concat_gae" and not cfg.trainer.concat_multi_turn
    with open_dict(cfg):
        cfg.ray_kwargs.ray_init.include_dashboard = False
        temp = os.environ.get("V10_RAY_TEMP_DIR")
        if temp:
            Path(temp).mkdir(parents=True, exist_ok=True)
            cfg.ray_kwargs.ray_init._temp_dir = temp
        cfg.ray_kwargs.ray_init.runtime_env = {"env_vars": {
            "V10_PPO_ENDPOINT": str(args.endpoint), "R1_RENDER_URL": os.environ["R1_RENDER_URL"],
            "PYTHONPATH": os.environ["PYTHONPATH"], "WANDB_MODE": os.environ.get("WANDB_MODE", "offline"),
            "PYTHONDONTWRITEBYTECODE": "1", "NO_PROXY": "*", "no_proxy": "*",
            "LD_LIBRARY_PATH": os.environ.get("LD_LIBRARY_PATH", "")}}
    root = Path(cfg.trainer.default_local_dir).parent
    root.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, root / "launch_resolved.yaml", resolve=True)
    try:
        main_ppo.run_ppo(cfg, task_runner_class=ray.remote(num_cpus=1)(V10TaskRunner))
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
