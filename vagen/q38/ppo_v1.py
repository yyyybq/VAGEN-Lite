"""Entry point of the q38 runs on verl's V1 trainers (DESIGN_ASYNC 2.1, 2.5.4).

    R1_RENDER_URL=http://HOST:PORT/render python -m vagen.q38.ppo_v1 --config RL/ppo.yaml --endpoint 300

Same command line as vagen.interiorgs_v10_ppo (the 8B entry, which stays as it is); the job script picks the module
with ``V10_PPO_MODULE``.  The config is a complete V1 config (training_prep ``--stack q38``, or the hand-written
vagen/q38/configs/smoke_4b_example.yaml), loaded with OmegaConf like the 8B entry does, then:

1. ``vagen.endpoint`` is set from ``--endpoint`` (recorded in the effective config);
2. the contract (vagen.q38.contract) checks it and writes ``<run>/contract_check.json`` and
   ``<run>/effective_config.yaml``; a failure exits with status 2 before Ray starts;
3. verl's own start-up checks run as in verl.trainer.main_ppo.main (auto_set_device, validate_config);
4. ``run_ppo(cfg, task_runner_class=Q38TaskRunner)``.

Q38TaskRunner is verl's TaskRunnerV1 (verl/verl/trainer/main_ppo.py:102-163) with two differences:

- the trainer class: with ``vagen.trainer_impl: vagen`` (the default) ours from vagen.q38.trainer
  (VagenSyncTrainer / VagenColocateAsyncTrainer, by ``trainer.v1.trainer_mode``); with ``stock`` verl's own class,
  for smoke S0;
- the endpoint, after ``trainer.init()``: the learning-rate horizon was fixed by ``_init_dataloader``
  (trainer_base.py:792-813) and the checkpoint loaded (trainer_base.py:417), so the runner checks
  endpoint <= horizon and total_epochs x steps_per_epoch >= endpoint, returns at once when the loaded checkpoint is
  already at the endpoint (fit would re-issue the restored prompts and prefetch a batch for nothing,
  trainer_base.py:481,490), and otherwise sets ``trainer.total_training_steps = endpoint`` so fit stops there and
  saves (is_last_step, trainer_base.py:493,506-510).  After fit, ``global_step_<endpoint>`` must exist.

The learning-rate schedule keeps the config's horizon (trainer.total_training_steps, 700 on the main line), so a later
run with a larger endpoint and ``resume_mode: auto`` continues the same schedule, as with the 8B entry.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import os
import sys
from pathlib import Path
from pprint import pprint
from typing import Optional

import ray
from omegaconf import DictConfig, OmegaConf, open_dict

from vagen.q38 import contract

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "INFO"))

TRAINER_MODULE = "vagen.q38.trainer"
# trainer.v1.trainer_mode -> our class in TRAINER_MODULE (separate_async is phase 4, DESIGN 2.3)
TRAINER_CLASSES = {
    "sync": "VagenSyncTrainer",
    "colocate_async": "VagenColocateAsyncTrainer",
    "separate_async": "VagenSeparateAsyncTrainer",
}


def select_trainer_cls(config: DictConfig):
    """The trainer class of a run: ours by trainer_mode, or verl's own with ``vagen.trainer_impl: stock``."""
    from verl.trainer.ppo.v1 import get_trainer_cls

    mode = str(config.trainer.v1.trainer_mode)
    stock_cls = get_trainer_cls(mode)
    impl = contract.trainer_impl(config)
    if impl == "stock":
        return stock_cls
    if impl != "vagen":
        raise ValueError(f"vagen.trainer_impl must be one of {contract.TRAINER_IMPLS}, got {impl!r}")
    name = TRAINER_CLASSES.get(mode)
    if name is None:
        raise ValueError(f"no q38 trainer for trainer_mode {mode!r}; known: {sorted(TRAINER_CLASSES)}")
    module = importlib.import_module(TRAINER_MODULE)
    cls = getattr(module, name, None)
    if cls is None:
        raise ValueError(f"{TRAINER_MODULE} has no {name} (trainer_mode {mode!r})")
    if not (isinstance(cls, type) and issubclass(cls, stock_cls)):
        raise TypeError(f"{TRAINER_MODULE}.{name} must subclass verl's {stock_cls.__name__}")
    return cls


def endpoint_checkpoint(config: DictConfig, endpoint: int) -> Path:
    local_dir = str(config.trainer.default_local_dir)
    if not os.path.isabs(local_dir):  # as V1 resolves it (trainer_base.py:889-892)
        local_dir = os.path.join(os.getcwd(), local_dir)
    return Path(local_dir) / f"global_step_{endpoint}"


def prepare_endpoint(trainer, endpoint: int) -> bool:
    """Check the endpoint against an initialised trainer and make fit stop there; False = already there.

    ``trainer.global_steps`` is the loaded checkpoint's step (0 from scratch), ``trainer.total_training_steps`` the
    learning-rate horizon and ``trainer.steps_per_epoch`` len(train_dataset) // train_batch_size (trainer_base.py:792).
    """
    endpoint = int(endpoint)
    horizon = int(trainer.total_training_steps)
    if endpoint <= 0:
        raise ValueError(f"endpoint must be positive, got {endpoint}")
    if endpoint > horizon:
        raise ValueError(f"endpoint {endpoint} is beyond the learning-rate horizon {horizon}")
    total_epochs = int(trainer.config.trainer.total_epochs)
    if total_epochs * int(trainer.steps_per_epoch) < endpoint:
        # fit's loop ends when the epochs run out (trainer_base.py:492) and would save no endpoint checkpoint
        raise ValueError(f"total_epochs {total_epochs} x steps_per_epoch {trainer.steps_per_epoch} "
                         f"< endpoint {endpoint}")
    done = int(trainer.global_steps)
    if done >= endpoint:
        if done > endpoint:
            print(f"[q38] WARNING: the loaded checkpoint (step {done}) is past the endpoint {endpoint}")
        print(f"[q38] already at the endpoint (step {done} >= {endpoint}): nothing to train")
        return False
    trainer.total_training_steps = endpoint
    print(f"[q38] training steps {done + 1}..{endpoint} (learning-rate horizon {horizon})")
    return True


class Q38TaskRunner:
    """verl's TaskRunnerV1 (verl/verl/trainer/main_ppo.py:102-163) with our trainer classes and the endpoint.

    Wrapped with ``ray.remote(num_cpus=1)`` by main, as the 8B entry wraps its runner.
    """

    def __init__(self):
        self.config = None
        self.trainer = None
        self.agent_loop_manager = None

    def init_agent_loop_manager(self):
        # copied from TaskRunnerV1.init_agent_loop_manager (verl/verl/trainer/main_ppo.py:111-131)
        from verl.trainer.ppo.v1 import AgentLoopManagerTQ
        from verl.utils.import_utils import load_class_from_fqn

        manager_class_fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
        if manager_class_fqn:
            agent_loop_manager_cls = load_class_from_fqn(manager_class_fqn, "AgentLoopManager")
        else:
            agent_loop_manager_cls = AgentLoopManagerTQ

        self.agent_loop_manager = agent_loop_manager_cls.create(
            config=self.config,
            llm_client=self.trainer.get_llm_client(),
            teacher_client=self.trainer.get_teacher_client(),
            reward_loop_worker_handles=self.trainer.get_reward_handles(),
        )

    def run(self, config: DictConfig):
        # TaskRunnerV1.run (verl/verl/trainer/main_ppo.py:133-163); differences marked [q38]
        import transfer_queue as tq

        from verl.utils.logging_utils import configure_verl_logging

        configure_verl_logging()
        trainer_cls = select_trainer_cls(config)  # [q38] ours, or verl's own with trainer_impl stock
        endpoint = int(config.vagen.endpoint)  # [q38] set by main from --endpoint
        print(f"[q38] trainer {trainer_cls.__module__}.{trainer_cls.__name__}, endpoint {endpoint}")

        config.transfer_queue.enable = True
        pprint(OmegaConf.to_container(config, resolve=True))
        OmegaConf.resolve(config)
        self.config = config

        # initialize transfer queue
        tq.init(config.transfer_queue)
        succeeded = False
        try:
            self.trainer = trainer_cls(config=config)
            self.trainer.init()
            if not prepare_endpoint(self.trainer, endpoint):  # [q38]
                succeeded = True
                return
            self.init_agent_loop_manager()
            self.trainer.fit(self.agent_loop_manager)
            checkpoint = endpoint_checkpoint(config, endpoint)  # [q38]
            assert checkpoint.is_dir(), f"no checkpoint at the endpoint: {checkpoint}"
            succeeded = True
        finally:
            try:
                tracking = getattr(self.trainer, "logger", None)
                if tracking is not None:
                    tracking.finish(exit_code=0 if succeeded else 1)
            finally:
                tq.close()


def runtime_env_vars(environ=os.environ) -> dict:
    """The environment of every Ray process; as the 8B entry sets it (vagen/interiorgs_v10_ppo.py:140-144)."""
    return {
        "R1_RENDER_URL": environ["R1_RENDER_URL"],
        "PYTHONPATH": environ["PYTHONPATH"],
        "WANDB_MODE": environ.get("WANDB_MODE", "offline"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "NO_PROXY": "*",
        "no_proxy": "*",
        "LD_LIBRARY_PATH": environ.get("LD_LIBRARY_PATH", ""),
    }


def prepare_config(cfg: DictConfig, endpoint: int, environ=os.environ) -> DictConfig:
    """Endpoint, Ray start-up keys and the Ray runtime environment (vagen/interiorgs_v10_ppo.py:134-144)."""
    with open_dict(cfg):
        if OmegaConf.select(cfg, "vagen") is None:
            cfg.vagen = {}
        previous = OmegaConf.select(cfg, "vagen.endpoint", default=None)
        if previous is not None and int(previous) != int(endpoint):
            logger.warning(f"[q38] vagen.endpoint {previous} in the config replaced by --endpoint {endpoint}")
        cfg.vagen.endpoint = int(endpoint)
        cfg.ray_kwargs.ray_init.include_dashboard = False
        temp = environ.get("V10_RAY_TEMP_DIR")
        if temp:
            Path(temp).mkdir(parents=True, exist_ok=True)
            cfg.ray_kwargs.ray_init._temp_dir = temp
        # keep what the config's runtime_env sets (but not null placeholders such as py_executable: null)
        runtime_env = OmegaConf.select(cfg, "ray_kwargs.ray_init.runtime_env", default=None)
        runtime_env = OmegaConf.to_container(runtime_env, resolve=True) if runtime_env is not None else {}
        runtime_env = {k: v for k, v in runtime_env.items() if v is not None}
        runtime_env["env_vars"] = {**(runtime_env.get("env_vars") or {}), **runtime_env_vars(environ)}
        cfg.ray_kwargs.ray_init.runtime_env = runtime_env
    return cfg


def main(argv: Optional[list] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--endpoint", type=int, required=True)
    args = ap.parse_args(argv)
    cfg = OmegaConf.load(args.config)
    prepare_config(cfg, args.endpoint)
    try:
        report = contract.check(cfg, args.endpoint)
    except contract.ContractError as e:
        print(e, flush=True)
        sys.exit(2)
    print(contract.summary(report), flush=True)

    # verl's own start-up checks (verl/verl/trainer/main_ppo.py:173-181)
    from verl.trainer.ppo.utils import need_critic, need_reference_policy
    from verl.utils.config import validate_config
    from verl.utils.device import auto_set_device

    auto_set_device(cfg)
    validate_config(config=cfg, use_reference_policy=need_reference_policy(cfg), use_critic=need_critic(cfg))

    from verl.trainer.main_ppo import run_ppo

    try:
        run_ppo(cfg, task_runner_class=ray.remote(num_cpus=1)(Q38TaskRunner))
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
