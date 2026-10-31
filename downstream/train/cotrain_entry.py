#!/usr/bin/env python3
"""C1 training entry (the user's decision of 2026-10-30: B with a text loss, lambda 0.1, 8 text samples per step,
the main line's QA mixed 1:1 with the trajectories).  Everything of train_entry.py (segments, full-state resume,
checkpoints, records) is reused unchanged; only StarVLA's VLATrainer._train_step is replaced, when the config has a
trainer.cotrain section:

    total loss = action loss + lambda * text loss, one backward, one gradient clip, one optimiser step

The text loss is the standard next-token cross-entropy on the answer tokens of this process's text sample for the
step (downstream/train/cotrain_text.py: chosen from the step alone, so a resume continues exactly), through the VLM's
own lm_head (which the action loss never reaches: X4).  Logged with every step: action_dit_loss, text_loss,
text_tokens, text_source (0 trajectory, 1 QA).  Every diag_every steps also the gradient norms of each loss alone on
lm_head and on the language model's last layer (torch.autograd.grad, which leaves .grad and DeepSpeed's hooks alone),
for the short run's check of their ratio.  With trainer.cotrain.skip_final_save the last step writes no checkpoint
(the short run).  Launched like train_entry.py (accelerate, DeepSpeed ZeRO-2): cotrain_entry.py --config_yaml ...
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train_entry  # noqa: E402


def grad_norm(loss, params) -> float:
    import torch

    grads = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
    total = sum(float(g.float().pow(2).sum()) for g in grads if g is not None)
    return total ** 0.5


def enable_checkpointing(vlm) -> None:
    """Activation checkpointing on the VLM (its vision blocks and language layers recompute their activations in the
    backward pass): the same gradients for less memory.  2026-10-31: the first short run ran out of memory at its
    third step (rank 7, 70.2 GiB allocated) with the action batch and one text sample in one backward."""
    vlm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if getattr(vlm, "config", None) is not None:
        vlm.config.use_cache = False
    vlm._c1_checkpointing = True


def make_train_step(T, cfg_cotrain: dict, text_batches_for):
    """A replacement for VLATrainer._train_step (T: starVLA.training.train_starvla).  text_batches_for(trainer)
    returns the process's cotrain_text.TextBatches (built once).  With cfg_cotrain gradient_checkpointing, the VLM's
    activation checkpointing is switched on before the first step (enable_checkpointing)."""
    import torch

    lam = float(cfg_cotrain["lambda"])
    diag_every = int(cfg_cotrain.get("diag_every", 0))
    checkpointing = bool(cfg_cotrain.get("gradient_checkpointing", False))

    def _train_step(self, batch_vla, batch_vlm=None):
        batches = text_batches_for(self)
        step = int(self.completed_steps)
        source, index, text_inputs = batches.at(step)
        vla = T._unwrap_model(self.accelerator, self.model)
        vlm = vla.qwen_vl_interface.model
        if checkpointing and not getattr(vlm, "_c1_checkpointing", False):
            enable_checkpointing(vlm)
        text_inputs = {k: v.to(self.accelerator.device) for k, v in text_inputs.items()}
        metrics = {}
        with self.accelerator.accumulate(self.model):
            self.optimizer.zero_grad()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output_dict = self.model.forward(batch_vla)
                action_loss = output_dict["action_loss"]
                text_loss = vlm(**text_inputs, use_cache=False).loss
                total_loss = action_loss + lam * text_loss
            if diag_every and step % diag_every == 0:
                head = [vlm.lm_head.weight]
                last = [p for p in vlm.model.language_model.layers[-1].parameters()] \
                    if hasattr(vlm.model, "language_model") else [p for p in vlm.model.layers[-1].parameters()]
                metrics.update({"grad_lm_head_text": grad_norm(text_loss, head),
                                "grad_lm_head_action": grad_norm(action_loss, head),
                                "grad_last_layer_text": grad_norm(lam * text_loss, last),
                                "grad_last_layer_action": grad_norm(action_loss, last)})
            self.accelerator.backward(total_loss)
            if self.config.trainer.gradient_clipping is not None:
                self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.trainer.gradient_clipping)
            self.optimizer.step()
            if self.accelerator.sync_gradients:
                self.lr_scheduler.step()
        # every process's text loss, so the log has both sources (metrics are written by the main process only)
        mine = torch.tensor([float(text_loss.detach()), 1.0 if source == "qa" else 0.0], device=self.accelerator.device)
        both = self.accelerator.gather(mine).reshape(-1, 2).float().cpu()
        qa = both[:, 1] > 0.5
        metrics.update({"action_dit_loss": action_loss.item(), "text_loss": float(both[:, 0].mean()),
                        "text_loss_traj": float(both[~qa, 0].mean()) if (~qa).any() else float("nan"),
                        "text_loss_qa": float(both[qa, 0].mean()) if qa.any() else float("nan"),
                        "total_loss": total_loss.item(), "text_tokens": int((text_inputs["labels"] != -100).sum()),
                        "text_index": int(index)})
        return metrics

    return _train_step


def install(cfg_path: str, overrides) -> dict:
    """Patch StarVLA for co-training when the config asks for it; returns the cotrain section ({} = plain run)."""
    import starVLA.training.train_starvla as T

    cfg = train_entry.load_config(cfg_path, overrides)
    co = dict(cfg.trainer.get("cotrain") or {})
    if not co:
        return {}
    import cotrain_text as CT

    holder = {}

    def text_batches_for(trainer):
        if "tb" not in holder:
            from transformers import AutoProcessor

            acc = trainer.accelerator
            processor = AutoProcessor.from_pretrained(str(cfg.framework.qwenvl.base_vlm))
            t = time.time()
            holder["tb"] = CT.TextBatches(processor, acc.process_index, acc.num_processes, int(co.get("seed", cfg.seed)),
                                          sources=dict(co.get("sources") or CT.SOURCES),
                                          max_tokens=int(co.get("max_tokens", CT.MAX_TOKENS)))
            if acc.is_main_process:
                print(f"[cotrain] text sources {holder['tb'].sizes} loaded in {time.time() - t:.0f} s", flush=True)
        return holder["tb"]

    T.VLATrainer._train_step = make_train_step(T, co, text_batches_for)
    if co.get("skip_final_save"):
        T.VLATrainer._finalize_training = lambda self: None
    return co


def main(argv=None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config_yaml", required=True)
    args, overrides = ap.parse_known_args(argv)
    train_entry_dir = str(train_entry.STARVLA)
    sys.path.insert(0, train_entry_dir)
    import os

    os.chdir(train_entry_dir)
    install(args.config_yaml, overrides)
    args.overrides = overrides
    return train_entry.train(args)


if __name__ == "__main__":
    sys.exit(main())
