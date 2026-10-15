"""PPO for InteriorGS v10 (the dense-score pilot's runner without its R1-only checks).

    R1_RENDER_URL=http://HOST:PORT/render python -m vagen.interiorgs_v10_ppo --config RL/ppo.yaml --endpoint 250

Round 10: the config may use verl's concat multi-turn mode (one sequence per episode, ``trainer.concat_multi_turn``
with ``adv_estimator: gae``) or the no-concat mode (``no_concat_gae``); ``history_mode`` checks the pair.

The optimizer and scheduler keep the config's horizon (``trainer.total_training_steps``); ``--endpoint``
only stops ``fit`` after that many updates, so a later run with a larger endpoint and ``resume_mode: auto``
continues from the latest checkpoint.  The env keeps its default (v10) success rules: this runner never sets
``r1_legacy_success``.

Checkpoints (decision of 2026-10-14): with ``trainer.ckpt_keep_every`` = K, every K-th update's checkpoint is
kept and of the others only the latest ``trainer.max_actor_ckpt_to_keep`` (``retained_steps``).
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Iterable, Set

import ray
from omegaconf import OmegaConf, open_dict

from vagen import main_ppo
from vagen.ray_trainer import RayPPOTrainer


def retained_steps(steps: Iterable[int], keep_every: int, keep_latest: int) -> Set[int]:
    """Checkpoints to keep: every ``keep_every``-th step and the latest ``keep_latest`` of the others."""
    steps = sorted(set(int(s) for s in steps))
    milestones = {s for s in steps if keep_every > 0 and s % keep_every == 0}
    others = [s for s in steps if s not in milestones]
    return milestones | set(others[-keep_latest:] if keep_latest > 0 else [])


class EndpointTrainer(RayPPOTrainer):
    def _rotate_complete_checkpoints(self, current_folder: str):
        keep_every = int(self.config.trainer.get("ckpt_keep_every", 0) or 0)
        if keep_every <= 0:
            return super()._rotate_complete_checkpoints(current_folder)
        root = Path(current_folder).parent
        complete = {}
        for folder in root.glob("global_step_*"):
            try:
                step = int(folder.name.split("global_step_")[-1])
            except ValueError:
                continue
            if (folder / "COMPLETE").exists():
                complete[step] = folder
        keep = retained_steps(complete, keep_every, int(self.config.trainer.max_actor_ckpt_to_keep))
        keep.add(int(self.global_steps))
        for step, folder in sorted(complete.items()):
            if step not in keep and folder.resolve() != Path(current_folder).resolve():
                print(f"[ckpt-rotate] removing checkpoint {folder} (keeping {sorted(keep)})")
                shutil.rmtree(folder, ignore_errors=True)

    def fit(self):
        endpoint = int(os.environ["V10_PPO_ENDPOINT"])
        assert 0 < endpoint <= self.total_training_steps
        self.total_training_steps = endpoint
        super().fit()
        checkpoint = Path(self.config.trainer.default_local_dir) / f"global_step_{endpoint}"
        assert checkpoint.is_dir(), f"no checkpoint at the endpoint: {checkpoint}"


# Round 10: the two ways an episode's earlier turns reach the model (training_prep.history_overrides).
# Each turn a separate sample (the no-concat agent loop, its own GAE) or one sequence per episode
# (verl's multi-turn concat mode with plain GAE); the advantage estimator must match the mode.
MODES = {("no_concat_gae", False): "no_concat", ("gae", True): "concat"}


def history_mode(cfg) -> str:
    """The run's mode from (algorithm.adv_estimator, trainer.concat_multi_turn); an unknown pair is an error."""
    key = (str(cfg.algorithm.adv_estimator), bool(cfg.trainer.concat_multi_turn))
    if key not in MODES:
        raise ValueError(f"adv_estimator / concat_multi_turn {key} is not a supported pair: {sorted(MODES)}")
    return MODES[key]


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
    print(f"[interiorgs_v10_ppo] history mode: {history_mode(cfg)}")
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
