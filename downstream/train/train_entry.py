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
  4. the checkpoint at the last step is not written twice (StarVLA writes steps_<max> and final_model);
  5. the data / framework config keys are all read before StarVLA saves run/config.yaml, so the policy server
     (which builds its serving contract from that file) sees obs_image_size etc. even in a short run.
It also writes, on rank 0, OUTPUT/downstream_record.json (load dtype, action-head hash, the sample ids of the first
20 global batches, timings) and OUTPUT/metrics.jsonl (one line per logged step).

Full state and segments (stage 1, the user's decisions of 2026-10-22; downstream/train/full_state.py):
  trainer.full_state_interval N   every N steps save the whole training state to OUTPUT/full_state/steps_<step>/
                                  (accelerate save_state: DeepSpeed bf16 weights, fp32 master weights, AdamW moments;
                                  the LR scheduler; random states; plus the data position), keeping only the newest.
                                  A launch that finds one resumes from it: same weights, optimizer, schedule, random
                                  states, and the data loader skips the batches already handed out.
  trainer.eval_points [..]        weights-only checkpoints (checkpoints/steps_<N>_pytorch_model.pt) only at these
                                  steps (without full_state_interval: at every save_interval, as before)
  trainer.stop_at N               leave after step N (its checkpoint and full state saved): one segment of
                                  run_segmented.sh, which evaluates and launches the next segment
  trainer.log_batch_digest true   OUTPUT/batch_digests.jsonl: per batch, a hash of rank 0's actions and instructions
                                  (the resume test compares them with an uninterrupted run)
Each launch appends a line to OUTPUT/launches.jsonl.
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


def batch_digest(batch) -> str:
    """Hash of the actions and instructions of one local batch (a list of examples)."""
    import numpy as np

    h = hashlib.sha256()
    for ex in batch:
        a = ex.get("action") if isinstance(ex, dict) else None
        if a is not None:
            a = a.detach().float().cpu().numpy() if hasattr(a, "detach") else np.asarray(a, dtype=np.float32)
            h.update(np.ascontiguousarray(a, dtype=np.float32).tobytes())
        lang = ex.get("lang") if isinstance(ex, dict) else None
        if lang is not None:
            h.update(str(lang).encode())
    return h.hexdigest()[:16]


class SegmentDone(Exception):
    """Raised on every rank after the stop_at step's checkpoint and full state are saved."""


def segment_settings(trainer_cfg) -> dict:
    """full_state_interval, eval_points, stop_at, log_batch_digest of the trainer section, checked against
    save_interval (StarVLA calls the save hook only at its multiples)."""
    get = trainer_cfg.get
    out = {"full_every": int(get("full_state_interval", 0) or 0),
           "eval_points": sorted(int(x) for x in (get("eval_points", None) or [])),
           "stop_at": int(get("stop_at", 0) or 0), "digests": bool(get("log_batch_digest", False))}
    every = int(get("save_interval", 0) or 0)
    for name, values in (("full_state_interval", [out["full_every"]]), ("eval_points", out["eval_points"]),
                         ("stop_at", [out["stop_at"]])):
        bad = [v for v in values if v and (not every or v % every)]
        if bad:
            raise ValueError(f"{name} {bad} is not a multiple of save_interval {every}")
    return out


SERVING_SECTIONS = (("datasets", "vla_data"), ("framework", "action_model"), ("framework", "qwenvl"))


def mark_for_serving(cfg) -> list:
    """StarVLA saves run/config.yaml with only the config keys training has read, and its policy server builds the
    serving contract from that file (e.g. training_obs_image_size, which a short run never reads).  Read every key of
    the data and framework sections so config.yaml carries them all."""
    read = []
    for path in SERVING_SECTIONS:
        node = cfg
        for part in path:
            node = getattr(node, part)
        for key in list(node.keys()):
            getattr(node, key)
            read.append(".".join(path + (key,)))
    return read


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
    cfg.framework.qwenvl.attn_implementation = "sdpa"  # FlashAttention cannot initialise on CPU; no effect on weights
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
              "mode": "dry run on CPU (attention sdpa)", "load_dtype": LOAD_DTYPE,
              "backbone_dtypes": backbone_dtypes(model),
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

    import shutil

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import full_state as FS

    cfg = load_config(args.config_yaml, args.overrides)
    seed = int(cfg.seed)
    seg = segment_settings(cfg.trainer)
    force_bf16_loading()
    seed_dataset(seed)
    rank0 = (not dist.is_initialized()) or dist.get_rank() == 0
    record = {"config": str(args.config_yaml), "seed": seed, "load_dtype": LOAD_DTYPE,
              "world_size": T.accelerator.num_processes, "timings": {}, "segment": seg,
              "launch_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    data = {"batches": 0, "skip": 0}  # batches handed out by this rank's loader; to skip after a resume

    original_save = T.VLATrainer._save_checkpoint
    original_finalize = T.VLATrainer._finalize_training
    original_log = T.VLATrainer._log_metrics
    original_next = T.VLATrainer._get_next_batch

    def save_full(self, step):
        acc = self.accelerator
        root = Path(self.config.output_dir) / "full_state"
        part = root / f"steps_{step}.partial"
        if acc.is_main_process:
            root.mkdir(parents=True, exist_ok=True)
            shutil.rmtree(part, ignore_errors=True)
        acc.wait_for_everyone()
        t = time.time()
        acc.save_state(str(part))
        acc.wait_for_everyone()
        if acc.is_main_process:
            meta = {"completed_steps": step, "batches_consumed": data["batches"],
                    "epoch": int(getattr(self, "vla_epoch_count", 0)), "batches_per_epoch": len(self.vla_train_dataloader),
                    "max_train_steps": int(self.config.trainer.max_train_steps), "seed": seed,
                    "lr": [float(x) for x in self.lr_scheduler.get_last_lr()],
                    "save_seconds": round(time.time() - t, 1), "bytes": FS.dir_bytes(part),
                    "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
            (part / FS.STATE_FILE).write_text(json.dumps(meta, indent=1) + "\n")
            final = root / f"steps_{step}"
            shutil.rmtree(final, ignore_errors=True)
            part.rename(final)
            meta["removed"] = FS.prune(root, keep=final)
            with open(Path(self.config.output_dir) / "full_state_log.jsonl", "a") as f:
                f.write(json.dumps(meta) + "\n")
            record["timings"].setdefault("full_state", []).append(
                {"step": step, "seconds": meta["save_seconds"], "bytes": meta["bytes"]})
        acc.wait_for_everyone()

    def _save_checkpoint(self):
        """Weights-only checkpoint (eval points, or every save_interval without full state), then the full state,
        then leave if this is the segment's end.  The last step is saved once, as final_model."""
        step = self.completed_steps
        if step >= int(self.config.trainer.max_train_steps):
            self.accelerator.wait_for_everyone()
            return None
        out = None
        if not seg["full_every"] or step in seg["eval_points"]:
            t = time.time()
            out = original_save(self)
            record["timings"].setdefault("checkpoint_seconds", []).append(round(time.time() - t, 1))
        if seg["full_every"] and step % seg["full_every"] == 0:
            save_full(self, step)
        if seg["stop_at"] and step >= seg["stop_at"]:
            raise SegmentDone(step)
        return out

    def _get_next_batch(self):
        batch = original_next(self)
        data["batches"] += 1
        if seg["digests"] and self.accelerator.is_main_process:
            with open(Path(self.config.output_dir) / "batch_digests.jsonl", "a") as f:
                f.write(json.dumps({"batch": data["batches"], "step": self.completed_steps + 1,
                                    "digest": batch_digest(batch)}) + "\n")
        return batch

    def _create_data_iterators(self):
        loader = self.vla_train_dataloader
        if data["skip"]:
            loader = self.accelerator.skip_first_batches(loader, data["skip"])
        self.vla_iter = iter(loader)

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
    T.VLATrainer._get_next_batch = _get_next_batch
    T.VLATrainer._create_data_iterators = _create_data_iterators

    cfg = T.wrap_config(cfg)
    record["serving_keys_read"] = len(mark_for_serving(cfg))
    output_dir = T.setup_directories(cfg)
    resume = FS.latest(output_dir / "full_state") if seg["full_every"] else None
    if resume and int(resume[1]["max_train_steps"]) != int(cfg.trainer.max_train_steps):
        raise RuntimeError(f"full state {resume[0]} was made for {resume[1]['max_train_steps']} steps, this run has "
                           f"{cfg.trainer.max_train_steps} (the cosine schedule would not continue)")
    if rank0 and seg["full_every"]:
        for d in FS.partials(output_dir / "full_state"):
            shutil.rmtree(d)  # a save a kill interrupted
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
    record_name = "downstream_record.json" if not resume else f"downstream_record_resume_{resume[1]['completed_steps']}.json"
    if rank0:
        batches, digest = sample_ids(dataset, 20, global_batch)
        record.update(dataset_seed=int(dataset.seed), global_batch=global_batch, sample_ids_sha=digest,
                      first_batch=batches[0][:8])
        if not resume:
            (output_dir / "downstream_sample_ids.json").write_text(json.dumps(batches) + "\n")
        (output_dir / record_name).write_text(json.dumps(record, indent=1) + "\n")
    optimizer, lr_scheduler = T.setup_optimizer_and_scheduler(model=vla, cfg=cfg)
    trainer = T.VLATrainer(cfg=cfg, model=vla, vla_train_dataloader=dataloader, optimizer=optimizer,
                           lr_scheduler=lr_scheduler, accelerator=T.accelerator)
    t0 = time.time()
    trainer.prepare_training()
    record["timings"]["prepare_seconds"] = round(time.time() - t0, 1)
    T.accelerator.register_for_checkpointing(trainer.lr_scheduler)  # saved and restored with the full state
    if resume:
        t0 = time.time()
        T.accelerator.load_state(str(resume[0]))
        meta = resume[1]
        trainer.completed_steps = int(meta["completed_steps"])
        data["batches"] = int(meta["batches_consumed"])
        epoch, data["skip"] = FS.data_position(data["batches"], len(trainer.vla_train_dataloader))
        if epoch:
            trainer.vla_epoch_count = epoch
        record["resumed_from"] = {"path": str(resume[0]), "step": trainer.completed_steps, "batches": data["batches"],
                                  "skip": data["skip"], "epoch": epoch, "load_seconds": round(time.time() - t0, 1),
                                  "lr_after_load": [float(x) for x in trainer.lr_scheduler.get_last_lr()],
                                  "lr_saved": meta.get("lr")}
        if rank0:
            with open(output_dir / "metrics.jsonl", "a") as f:
                f.write(json.dumps({"resumed_from_step": trainer.completed_steps, "time": time.time()}) + "\n")
    t0 = time.time()
    outcome = "finished"
    try:
        if seg["stop_at"] and trainer.completed_steps >= seg["stop_at"]:
            # the segment's end was reached before (e.g. killed between its two saves): only the missing checkpoint
            if trainer.completed_steps in seg["eval_points"] and not (
                    output_dir / "checkpoints" / f"steps_{trainer.completed_steps}_pytorch_model.pt").is_file():
                original_save(trainer)
            raise SegmentDone(trainer.completed_steps)
        trainer.train()
    except SegmentDone as done:
        outcome = f"segment ended at step {done.args[0]}"
    record["timings"]["train_and_save_seconds"] = round(time.time() - t0, 1)
    record["outcome"] = outcome
    if rank0:
        (output_dir / record_name).write_text(json.dumps(record, indent=1) + "\n")
        with open(output_dir / "launches.jsonl", "a") as f:
            f.write(json.dumps(record) + "\n")
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
