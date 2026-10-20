#!/usr/bin/env python3
"""StarVLA training entry of the downstream line: StarVLA's own trainer, with four fixes, StarVLA left unchanged.

Training (8 GPUs, from the StarVLA root so its DeepSpeed config path resolves):
    accelerate launch --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml --num_processes 8 \
        <worktree>/downstream/train/train_entry.py --config_yaml EXP/config.yaml
Seed evidence without training (CPU, one process):
    python downstream/train/train_entry.py --config_yaml EXP/config.yaml --dry-run OUT.json --world 8 --batches 20

What it changes relative to starVLA/training/train_starvla.py (decisions D1 / D2 of the step-1 report):
  1. the backbone is always loaded in bf16 (StarVLA loads with torch_dtype="auto": bf16 or fp32 by the start);
  2. the seed (config `seed`) is set right before build_framework, so the new action head's initial weights depend
     only on the seed (StarVLA seeds after building the model);
  3. the seed is passed to the dataset sampler (StarVLA always uses its default 42), so the data order depends only
     on the seed;
  4. the checkpoint at the last step is not written twice (StarVLA writes steps_<max> and final_model).
It also writes, on rank 0, OUTPUT/downstream_record.json (load dtype, action-head hash, the sample ids of the first
20 global batches, timings) and OUTPUT/metrics.jsonl (one line per logged step).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

STARVLA = Path(os.environ.get("STARVLA_ROOT", "/mnt/umm/users/yinbaiqiao/VAGEN-Lite/third_party/starVLA"))
LOAD_DTYPE = "bfloat16"


# ----------------------------------------------------------------------------------------------------- helpers
def tensor_sha(tensors) -> str:
    import torch

    h = hashlib.sha256()
    for t in tensors:
        h.update(t.detach().to(torch.float32).cpu().contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


def rng_sha() -> str:
    import torch

    return hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()[:16]


def force_bf16_loading(model_cls=None):
    """Every Qwen2.5-VL from_pretrained call in this process loads in bf16 (StarVLA's own call passes "auto")."""
    import torch

    if model_cls is None:
        from transformers import Qwen2_5_VLForConditionalGeneration as model_cls
    if getattr(model_cls, "_downstream_bf16", False):
        return model_cls
    original = model_cls.from_pretrained.__func__

    def from_pretrained(cls, *args, **kwargs):
        kwargs.pop("dtype", None)
        kwargs["torch_dtype"] = torch.bfloat16
        return original(cls, *args, **kwargs)

    model_cls.from_pretrained = classmethod(from_pretrained)
    model_cls._downstream_bf16 = True
    return model_cls


def seed_dataset(seed: int, module=None):
    """StarVLA's build_dataloader calls lerobot_datasets.get_vla_dataset without a seed (default 42); pass ours."""
    if module is None:
        import starVLA.dataloader.lerobot_datasets as module
    original = getattr(module.get_vla_dataset, "_downstream_original", module.get_vla_dataset)

    def get_vla_dataset(*args, **kwargs):
        kwargs["seed"] = seed
        return original(*args, **kwargs)

    get_vla_dataset._downstream_original = original
    get_vla_dataset._downstream_seed = seed
    module.get_vla_dataset = get_vla_dataset
    return module


def seeded_build(build, cfg, seed: int):
    """Set every RNG to `seed`, then build: the action head's init depends on the seed only."""
    from accelerate.utils import set_seed

    set_seed(seed)
    before = rng_sha()
    model = build(cfg)
    return model, before


def sample_ids(dataset, n_batches: int, global_batch: int):
    """The (dataset position, trajectory id, step) StarVLA's mixture sampler picks for the first n_batches global
    batches.  The training DataLoader is sequential (no shuffle) and accelerate gives global step s the contiguous
    index block [s * global_batch, (s + 1) * global_batch), so these are the samples of the first n_batches steps."""
    position = {id(d): i for i, d in enumerate(dataset.datasets)}
    batches = []
    for b in range(n_batches):
        batch = []
        for index in range(b * global_batch, (b + 1) * global_batch):
            ds, traj, step = dataset.sample_step(index)
            batch.append([position[id(ds)], int(traj), int(step)])
        batches.append(batch)
    digest = hashlib.sha256(json.dumps(batches).encode()).hexdigest()[:16]
    return batches, digest


def backbone_dtypes(model) -> dict:
    out = {}
    for p in model.qwen_vl_interface.model.parameters():
        out[str(p.dtype)] = out.get(str(p.dtype), 0) + p.numel()
    return out


def load_config(config_yaml: str, overrides):
    from omegaconf import OmegaConf
    from starVLA.model.framework.share_tools import apply_config_compat

    cfg = OmegaConf.load(config_yaml)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist([o.lstrip("-") for o in overrides]))
    cfg = apply_config_compat(cfg)
    cfg.config_yaml = str(config_yaml)
    return cfg


# ----------------------------------------------------------------------------------------------------- modes
def dry_run(args) -> int:
    """Build (CPU) and record the action-head hash and the first batches' sample ids; no training."""
    os.chdir(STARVLA)
    sys.path.insert(0, str(STARVLA))
    from starVLA.model.framework.base_framework import build_framework
    from starVLA.training.trainer_utils.config_tracker import wrap_config

    cfg = load_config(args.config_yaml, args.overrides)
    seed = int(cfg.seed)
    force_bf16_loading()
    module = seed_dataset(seed)
    cfg = wrap_config(cfg)
    t0 = time.time()
    model, rng_before = seeded_build(build_framework, cfg, seed)
    build_s = time.time() - t0
    data_cfg = cfg.datasets.vla_data
    t0 = time.time()
    dataset = module.get_vla_dataset(data_cfg=data_cfg,
                                     balance_dataset_weights=data_cfg.get("balance_dataset_weights", False),
                                     balance_trajectory_weights=data_cfg.get("balance_trajectory_weights", False))
    data_s = time.time() - t0
    global_batch = int(data_cfg.per_device_batch_size) * args.world
    batches, digest = sample_ids(dataset, args.batches, global_batch)
    record = {"config": str(args.config_yaml), "base_vlm": str(cfg.framework.qwenvl.base_vlm), "seed": seed,
              "load_dtype": LOAD_DTYPE, "backbone_dtypes": backbone_dtypes(model),
              "rng_sha_before_build": rng_before,
              "action_head_init_sha": tensor_sha(model.action_model.parameters()),
              "dataset_seed": int(dataset.seed), "global_batch": global_batch, "n_batches": args.batches,
              "sample_ids_sha": digest, "first_batch": batches[0][:8], "build_seconds": round(build_s, 1),
              "dataset_seconds": round(data_s, 1)}
    Path(args.dry_run).parent.mkdir(parents=True, exist_ok=True)
    Path(args.dry_run).write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({k: record[k] for k in ("base_vlm", "seed", "action_head_init_sha", "sample_ids_sha")}))
    return 0


def train(args) -> int:
    os.chdir(STARVLA)
    sys.path.insert(0, str(STARVLA))
    import torch.distributed as dist
    import starVLA.training.train_starvla as T

    cfg = load_config(args.config_yaml, args.overrides)
    seed = int(cfg.seed)
    force_bf16_loading()
    seed_dataset(seed)
    rank0 = (not dist.is_initialized()) or dist.get_rank() == 0
    record = {"config": str(args.config_yaml), "seed": seed, "load_dtype": LOAD_DTYPE,
              "world_size": T.accelerator.num_processes, "timings": {}}

    original_save = T.VLATrainer._save_checkpoint
    original_finalize = T.VLATrainer._finalize_training
    original_log = T.VLATrainer._log_metrics

    def _save_checkpoint(self):  # the last step is saved once, as final_model
        if self.completed_steps >= int(self.config.trainer.max_train_steps):
            self.accelerator.wait_for_everyone()
            return None
        t = time.time()
        out = original_save(self)
        record["timings"].setdefault("checkpoint_seconds", []).append(round(time.time() - t, 1))
        return out

    def _finalize_training(self):
        t = time.time()
        out = original_finalize(self)
        record["timings"]["final_save_seconds"] = round(time.time() - t, 1)
        return out

    def _log_metrics(self, metrics):
        if self.completed_steps % self.config.trainer.logging_frequency == 0 and self.accelerator.is_main_process:
            row = {"step": self.completed_steps, "time": time.time()}
            row.update({k: float(v) for k, v in metrics.items() if isinstance(v, (int, float))})
            row["lr"] = [float(x) for x in self.lr_scheduler.get_last_lr()]
            with open(Path(self.config.output_dir) / "metrics.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")
        return original_log(self, metrics)

    T.VLATrainer._save_checkpoint = _save_checkpoint
    T.VLATrainer._finalize_training = _finalize_training
    T.VLATrainer._log_metrics = _log_metrics

    cfg = T.wrap_config(cfg)
    output_dir = T.setup_directories(cfg)
    t0 = time.time()
    vla, rng_before = seeded_build(T.build_framework, cfg, seed)
    record["timings"]["build_seconds"] = round(time.time() - t0, 1)
    dtypes = backbone_dtypes(vla)
    if set(dtypes) != {"torch.bfloat16"}:
        raise RuntimeError(f"backbone not loaded in bf16: {dtypes}")
    record.update(backbone_dtypes=dtypes, rng_sha_before_build=rng_before,
                  action_head_init_sha=tensor_sha(vla.action_model.parameters()))
    t0 = time.time()
    dataloader = T.prepare_data(cfg=cfg, accelerator=T.accelerator, output_dir=output_dir)
    record["timings"]["data_seconds"] = round(time.time() - t0, 1)
    dataset = dataloader.dataset
    global_batch = int(cfg.datasets.vla_data.per_device_batch_size) * T.accelerator.num_processes
    if rank0:
        batches, digest = sample_ids(dataset, 20, global_batch)
        record.update(dataset_seed=int(dataset.seed), global_batch=global_batch, sample_ids_sha=digest,
                      first_batch=batches[0][:8])
        (output_dir / "downstream_sample_ids.json").write_text(json.dumps(batches) + "\n")
        (output_dir / "downstream_record.json").write_text(json.dumps(record, indent=1) + "\n")
    optimizer, lr_scheduler = T.setup_optimizer_and_scheduler(model=vla, cfg=cfg)
    trainer = T.VLATrainer(cfg=cfg, model=vla, vla_train_dataloader=dataloader, optimizer=optimizer,
                           lr_scheduler=lr_scheduler, accelerator=T.accelerator)
    t0 = time.time()
    trainer.prepare_training()
    record["timings"]["prepare_seconds"] = round(time.time() - t0, 1)
    t0 = time.time()
    trainer.train()
    record["timings"]["train_and_save_seconds"] = round(time.time() - t0, 1)
    if rank0:
        (output_dir / "downstream_record.json").write_text(json.dumps(record, indent=1) + "\n")
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config_yaml", required=True)
    ap.add_argument("--dry-run", default=None, help="write the seed record to this JSON and exit (CPU)")
    ap.add_argument("--world", type=int, default=8, help="dry run: number of training processes")
    ap.add_argument("--batches", type=int, default=20, help="dry run: global batches whose sample ids are recorded")
    args, overrides = ap.parse_known_args(argv)
    args.overrides = overrides
    return dry_run(args) if args.dry_run else train(args)


if __name__ == "__main__":
    sys.exit(main())
