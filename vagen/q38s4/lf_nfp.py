"""C line (q38 s4): LLaMA-Factory SFT of Qwen3.8-27B with the NFP head (Cambrian-S stage 4), without changing LF.

    accelerate launch ... -m vagen.q38s4.lf_nfp RUN_DIR/train.yaml

RUN_DIR/q38s4.json (required, so an arm cannot start with the wrong weights):
    {"nfp_mse": 0.1, "nfp_cos": 0.1}   C-NFP
    {"nfp_mse": 0.0, "nfp_cos": 0.0}   C-noNFP
plus q38_sft.json as for vagen.q38.sft.lf_launcher (bf16 copies, metrics dir), whose main() this runs.

What changes against vagen.q38.sft.lf_launcher (all by replacing names in LF's sft workflow module before run_exp):

- template: the qwen3_8 template's Qwen3-VL plugin becomes lf_plugin.S4VideoPlugin (D4 frame sampling of videos
  marked ``|s4|``; nothing else of the plugin changes);
- load_model: LF's, then ``model.nfp_head`` (nfp.attach_head) when a weight is > 0;
- collator: LF's SFT collator; before it, a sample whose video is an NFP video gets every label IGNORE_INDEX (R7: no
  text loss); after it, ``nfp_video`` (one bool per video of the batch, in video_grid_thw's order);
- trainer: LF's CustomSeq2SeqTrainer with
  * compute_loss: CE exactly as LF computes it (the parent's compute_loss) for a micro-batch with text tokens; for one
    without (only NFP samples) 0 * one logit, so lm_head stays in the graph and nothing is NaN; then
    + nfp_mse * MSE + nfp_cos * (1 - cos), each a token sum scaled to the global token mean of the optimiser step
    (R12, nfp.scale_for_global_mean; the step's NFP token count comes from get_batch_samples, all-reduced);
  * get_batch_samples: also counts the step's NFP target tokens on every rank (one all-reduce, only with the head);
  * log: adds text_loss, nfp_mse, nfp_cos, copy_mse, copy_cos, nfp_pairs, nfp_videos (nfp.summarize; counters
    all-reduced over ranks for the logging window), so they reach LF's trainer_log and the q38 callback's rank files;
  * training_step: on the first micro-batch and at every save step, a freeze check: every parameter of visual.blocks /
    patch_embed / pos_embed has requires_grad False and grad None; the counts go to <metrics dir>/freeze_rank<R>.jsonl
    and the log; a frozen parameter with a gradient stops the run.
- With both weights 0 there is no head; the copy baseline is still logged.  The loss is LF's own on every
  micro-batch with text (test_lf_nfp checks equality bit for bit).
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

from vagen.q38s4 import nfp as N
from vagen.q38s4.lf_video import is_nfp

SETTINGS_FILE = "q38s4.json"
IGNORE_INDEX = -100
FROZEN_PREFIXES = ("visual.blocks.", "visual.patch_embed.", "visual.pos_embed.")


@dataclass
class S4Settings:
    nfp_mse: float = 0.0
    nfp_cos: float = 0.0

    @property
    def head_on(self) -> bool:
        return self.nfp_mse > 0 or self.nfp_cos > 0

    @classmethod
    def from_file(cls, path: Path) -> "S4Settings":
        cfg = json.loads(Path(path).read_text())
        unknown = set(cfg) - {"nfp_mse", "nfp_cos", "note"}
        if unknown:
            raise ValueError(f"{path}: unknown keys {sorted(unknown)}")
        s = cls(float(cfg["nfp_mse"]), float(cfg["nfp_cos"]))
        if s.nfp_mse < 0 or s.nfp_cos < 0:
            raise ValueError(f"{path}: negative weight")
        return s


SETTINGS: Optional[S4Settings] = None


def settings_for(argv: List[str]) -> S4Settings:
    if len(argv) < 2 or not argv[1].endswith((".yaml", ".yml")):
        raise SystemExit("usage: python -m vagen.q38s4.lf_nfp RUN_DIR/train.yaml")
    path = Path(argv[1]).resolve().parent / SETTINGS_FILE
    if not path.is_file():
        raise SystemExit(f"{path} missing: it fixes the arm's NFP weights")
    return S4Settings.from_file(path)


# ---------------------------------------------------------------------------------------------------- collator
def mark_nfp(features: List[Dict[str, Any]]) -> List[bool]:
    """IGNORE every label of an NFP sample (in place); one flag per video, in batch order."""
    flags: List[bool] = []
    for f in features:
        vids = f.get("videos") or []
        on = [is_nfp(v) for v in vids]
        if any(on):
            if not all(on):
                raise ValueError("a sample mixes NFP and QA videos")
            f["labels"] = [IGNORE_INDEX] * len(f["labels"])
        flags.extend(on)
    return flags


def make_collator_class():
    from llamafactory.data import SFTDataCollatorWith4DAttentionMask

    class S4Collator(SFTDataCollatorWith4DAttentionMask):
        def __call__(self, features):
            flags = mark_nfp(features)
            batch = super().__call__(features)
            batch["nfp_video"] = torch.tensor(flags, dtype=torch.bool)
            return batch

    return S4Collator


# ---------------------------------------------------------------------------------------------------- loss
def has_text_tokens(labels: Optional[torch.Tensor]) -> bool:
    """True when the causal-LM loss has at least one target (labels are shifted by one inside the loss)."""
    return labels is None or bool((labels[..., 1:] != IGNORE_INDEX).any())


def s4_loss(model, inputs: Dict[str, Any], tap: N.NFPTap, settings: S4Settings, base_compute, scale: float,
            acc: Dict[str, float]):
    """One micro-batch: (loss, outputs).  ``base_compute(model, inputs)`` is the parent trainer's
    compute_loss(..., return_outputs=True)."""
    nfp_video = inputs.pop("nfp_video", None)
    tap.arm(nfp_video)
    try:
        if has_text_tokens(inputs.get("labels")):
            loss, outputs = base_compute(model, inputs)
            acc["ce_sum"] += float(loss.detach())
            acc["ce_n"] += 1
        else:
            fwd = {k: v for k, v in inputs.items() if k != "labels"}
            outputs = model(**fwd)
            loss = outputs.logits[..., :1, :1].float().sum() * 0.0
        res = tap.collect()
    finally:
        tap.disarm()
    acc["pairs"] += res.n_pairs
    acc["videos"] += res.n_videos
    acc["copy_mse_sum"] += res.copy_mse_sum
    acc["copy_cos_sum"] += res.copy_cos_sum
    if res.mse_sum is not None:
        acc["head_pairs"] += res.n_pairs
        acc["mse_sum"] += float(res.mse_sum.detach())
        acc["cos_sum"] += float(res.cos_sum.detach())
        if settings.head_on:
            extra = settings.nfp_mse * res.mse_sum + settings.nfp_cos * res.cos_sum
            loss = loss + (scale * extra).to(loss.dtype)
    return loss, outputs


def freeze_report(model) -> Dict[str, Any]:
    """Counts over the named parameters (FSDP prefixes removed)."""
    rep = {"frozen": 0, "frozen_with_grad": 0, "encoder": 0, "encoder_trainable": 0, "encoder_with_grad": 0,
           "trainable": 0, "trainable_with_grad": 0, "head_trainable": 0, "merger_trainable": 0, "bad": []}
    for name, p in model.named_parameters():
        clean = name.replace("_fsdp_wrapped_module.", "").replace("_checkpoint_wrapped_module.", "")
        clean = clean[len("model."):] if clean.startswith("model.") else clean
        enc = clean.startswith(FROZEN_PREFIXES)
        if enc:
            rep["encoder"] += 1
            rep["encoder_trainable"] += int(p.requires_grad)
            rep["encoder_with_grad"] += int(p.grad is not None)
        if p.requires_grad:
            rep["trainable"] += 1
            rep["trainable_with_grad"] += int(p.grad is not None)
            rep["head_trainable"] += int(clean.startswith("nfp_head."))
            rep["merger_trainable"] += int(clean.startswith("visual.merger."))
        else:
            rep["frozen"] += 1
            if p.grad is not None:
                rep["frozen_with_grad"] += 1
                rep["bad"].append(clean)
        if enc and (p.requires_grad or p.grad is not None):
            rep["bad"].append(clean)
    rep["bad"] = sorted(set(rep["bad"]))[:20]
    rep["ok"] = not rep["bad"] and rep["encoder"] > 0
    return rep


def freeze_check_due(last: Optional[int], global_step: int, args) -> bool:
    """After the first micro-batch's backward, then once in the step after every save (gradients of that step's
    first micro-batch are there: sync_each_batch reduces every micro-batch)."""
    if last is None:
        return True
    save_steps = int(args.save_steps or 0) if str(args.save_strategy) not in ("no", "SaveStrategy.NO") else 0
    return bool(save_steps) and global_step > 0 and global_step % save_steps == 0 and last != global_step


# ---------------------------------------------------------------------------------------------------- trainer
def make_trainer_class():
    from llamafactory.train.sft.trainer import CustomSeq2SeqTrainer

    class S4Trainer(CustomSeq2SeqTrainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.s4 = SETTINGS or S4Settings()
            head = getattr(self.model, "nfp_head", None)
            if self.s4.head_on and head is None:
                raise ValueError("NFP weights > 0 but the model has no nfp_head")
            self.s4_tap = N.NFPTap(self.model, head)
            self.s4_acc = {k: 0.0 for k in N.ACC_KEYS}
            self.s4_scale = 0.0
            self.s4_checked: Optional[int] = None  # global step of the last freeze check

        def get_batch_samples(self, epoch_iterator, num_batches, device):
            batches, num_items = super().get_batch_samples(epoch_iterator, num_batches, device)
            if self.s4.head_on:
                local = sum(N.count_pairs(b.get("video_grid_thw"), b.get("nfp_video")) for b in batches)
                t = torch.tensor([float(local)], device=device, dtype=torch.float64)
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    torch.distributed.all_reduce(t)
                self.s4_scale = N.scale_for_global_mean(self.accelerator.num_processes, len(batches), t.item())
            return batches, num_items

        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            def base(m, inp):
                return super(S4Trainer, self).compute_loss(m, inp, return_outputs=True,
                                                           num_items_in_batch=num_items_in_batch)

            loss, outputs = s4_loss(model, inputs, self.s4_tap, self.s4, base, self.s4_scale, self.s4_acc)
            return (loss, outputs) if return_outputs else loss

        def training_step(self, model, inputs, num_items_in_batch=None):
            loss = super().training_step(model, inputs, num_items_in_batch)
            if freeze_check_due(self.s4_checked, self.state.global_step, self.args):
                self._s4_freeze_check(model)
            return loss

        def _s4_freeze_check(self, model):
            rep = freeze_report(model)
            rep.update(step=self.state.global_step, rank=self.args.process_index)
            mdir = Path(os.environ.get("VAGEN_Q38_SFT_METRICS_DIR") or Path(self.args.output_dir).resolve().parent
                        / "sft_metrics")
            mdir.mkdir(parents=True, exist_ok=True)
            with open(mdir / f"freeze_rank{self.args.process_index}.jsonl", "a") as f:
                f.write(json.dumps(rep) + "\n")
            print(f"[q38s4] freeze check step {rep['step']} rank {rep['rank']}: encoder params {rep['encoder']}, "
                  f"trainable {rep['encoder_trainable']}, with grad {rep['encoder_with_grad']}; frozen with grad "
                  f"{rep['frozen_with_grad']}; head trainable {rep['head_trainable']}, merger trainable "
                  f"{rep['merger_trainable']}", flush=True)
            self.s4_checked = self.state.global_step
            if not rep["ok"]:
                raise RuntimeError(f"[q38s4] freeze check failed: {rep['bad']}")

        def log(self, logs, *args, **kwargs):
            if "loss" in logs and "train_runtime" not in logs:
                vals = torch.tensor([self.s4_acc[k] for k in N.ACC_KEYS], dtype=torch.float64,
                                    device=self.args.device)
                if torch.distributed.is_available() and torch.distributed.is_initialized():
                    torch.distributed.all_reduce(vals)
                logs.update(N.summarize(dict(zip(N.ACC_KEYS, vals.tolist()))))
                self.s4_acc = {k: 0.0 for k in N.ACC_KEYS}
            return super().log(logs, *args, **kwargs)

    return S4Trainer


# ---------------------------------------------------------------------------------------------------- install
def make_load_model(orig):
    def load_model(tokenizer, model_args, finetuning_args, is_trainable=False, add_valuehead=False):
        model = orig(tokenizer, model_args, finetuning_args, is_trainable, add_valuehead)
        if is_trainable and SETTINGS is not None and SETTINGS.head_on:
            N.attach_head(model)
            print(f"[q38s4] nfp_head attached: {sum(p.numel() for p in model.nfp_head.parameters()):,} params, "
                  f"weights mse {SETTINGS.nfp_mse} cos {SETTINGS.nfp_cos}", flush=True)
        return model

    return load_model


def make_get_template(orig):
    def get_template_and_fix_tokenizer(tokenizer, data_args):
        from vagen.q38s4.lf_plugin import to_s4_plugin

        template = orig(tokenizer, data_args)
        template.mm_plugin = to_s4_plugin(template.mm_plugin)
        return template

    return get_template_and_fix_tokenizer


def install(settings: S4Settings) -> None:
    """Replace the names LF's run_sft looks up at call time."""
    global SETTINGS
    from llamafactory.train.sft import workflow as W

    SETTINGS = settings
    W.get_template_and_fix_tokenizer = make_get_template(W.get_template_and_fix_tokenizer)
    W.load_model = make_load_model(W.load_model)
    W.SFTDataCollatorWith4DAttentionMask = make_collator_class()
    W.CustomSeq2SeqTrainer = make_trainer_class()


def main() -> None:
    settings = settings_for(sys.argv)
    install(settings)
    from vagen.q38.sft import lf_launcher

    lf_launcher.main()


if __name__ == "__main__":
    main()
