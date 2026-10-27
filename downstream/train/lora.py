"""T2 (the user's decision of 2026-10-27): LoRA on the language model of the StarVLA policy, written here instead of
installing peft (StarVLA's VLA trainer has no LoRA switch and the starVLA environment has no peft).

    wrap(model, rank=16, alpha=32, dropout=0.05, seed=1)   every nn.Linear q/k/v/o_proj, gate/up/down_proj of the
                                                           language model's decoder layers becomes a LoRALinear
    freeze(model)                                          trainable: the LoRA matrices and the action head only
    param_groups(model, lora_lr, action_lr)                the optimizer groups while LoRA is on (replaces
                                                           StarVLA's build_param_lr_groups)
    merged(model)                                          a context in which state_dict() gives merged weights

LoRALinear keeps the base layer's weight and bias Parameters (the same objects, nothing copied) and adds A
(rank x in, uniform in +-1/sqrt(in) as peft's Kaiming init with a = sqrt(5), from a generator seeded by `seed`) and
B (out x rank, zeros): y = x W^T + b + dropout(x) A^T B^T * alpha / rank.  With B = 0 the wrapped model computes
exactly what the plain one does.  Inside merged(), each LoRALinear's state_dict has W + alpha / rank * B A (computed
in float32, stored in W's dtype) under the plain name "weight" and no LoRA keys, so the weights-only checkpoints
(steps_*, final_model) are plain StarVLA checkpoints that the policy server loads as any other; outside it (the
DeepSpeed full state used for resuming) the LoRA matrices are saved separately and W stays the start's weight.
"""
from __future__ import annotations

import contextlib
import math
import re

import torch
import torch.nn.functional as F
from torch import nn

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
TRAINABLE_PREFIXES = ("action_model.",)


def layer_pattern(targets=TARGETS):
    """Module names of the language model's projections (the vision tower's blocks also have gate/up/down_proj)."""
    return re.compile(r"(^|\.)language_model\.layers\.\d+\.(self_attn|mlp)\.(" + "|".join(targets) + r")$")


def is_lora(name: str) -> bool:
    return name.endswith((".lora_A", ".lora_B")) or name in ("lora_A", "lora_B")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float = 0.0, generator=None):
        super().__init__()
        self.in_features, self.out_features = base.in_features, base.out_features
        self.weight = base.weight
        self.bias = base.bias
        self.rank, self.alpha, self.scale = int(rank), float(alpha), float(alpha) / int(rank)
        bound = 1.0 / math.sqrt(self.in_features)
        a = (torch.rand(self.rank, self.in_features, generator=generator) * 2 - 1) * bound
        kw = {"dtype": base.weight.dtype, "device": base.weight.device}
        self.lora_A = nn.Parameter(a.to(**kw))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, self.rank, **kw))
        self.dropout = nn.Dropout(dropout) if dropout else nn.Identity()
        self.merge_on_save = False
        self._register_state_dict_hook(LoRALinear._merge_hook)

    def forward(self, x):
        return F.linear(x, self.weight, self.bias) + F.linear(F.linear(self.dropout(x), self.lora_A), self.lora_B) * self.scale

    def delta(self) -> torch.Tensor:
        """alpha / rank * B A in float32."""
        return (self.lora_B.detach().float() @ self.lora_A.detach().float()) * self.scale

    @staticmethod
    def _merge_hook(module, state_dict, prefix, local_metadata):
        if not module.merge_on_save:
            return None
        w = state_dict[prefix + "weight"]
        state_dict[prefix + "weight"] = (w.detach().float() + module.delta().to(w.device)).to(w.dtype)
        del state_dict[prefix + "lora_A"], state_dict[prefix + "lora_B"]
        return None

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, rank={self.rank}, alpha={self.alpha}"


def wrap(model: nn.Module, rank: int = 16, alpha: float = 32, dropout: float = 0.05, seed: int = 0,
         targets=TARGETS) -> dict:
    """Replace the language model's projections by LoRALinear (in module order, one generator: deterministic)."""
    pattern = layer_pattern(targets)
    names = [n for n, m in model.named_modules() if pattern.search(n) and isinstance(m, nn.Linear)]
    if not names:
        raise ValueError("no language-model projection found to wrap (expected ...language_model.layers.N.(self_attn|mlp).q_proj ...)")
    generator = torch.Generator().manual_seed(int(seed))
    for name in names:
        parent_name, child = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        setattr(parent, child, LoRALinear(getattr(parent, child), rank, alpha, dropout, generator))
    lora = sum(p.numel() for n, p in model.named_parameters() if is_lora(n))
    per_target = {t: sum(n.endswith("." + t) for n in names) for t in targets}
    return {"wrapped": len(names), "per_target": per_target, "rank": int(rank), "alpha": float(alpha),
            "dropout": float(dropout), "init_seed": int(seed), "lora_params": lora}


def freeze(model: nn.Module, trainable_prefixes=TRAINABLE_PREFIXES) -> dict:
    """Only the LoRA matrices and the action head are trained; everything else (vision tower, merger, language
    model weights, embeddings, lm_head) is frozen."""
    trainable = total = 0
    for name, p in model.named_parameters():
        p.requires_grad = is_lora(name) or name.startswith(tuple(trainable_prefixes))
        total += p.numel()
        trainable += p.numel() if p.requires_grad else 0
    return {"trainable_params": trainable, "total_params": total}


def param_groups(model: nn.Module, lora_lr: float, action_lr: float) -> list:
    """Optimizer groups of the trainable parameters: LoRA (lora_lr) and the action head (action_lr)."""
    lora, action, other = [], [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (lora if is_lora(name) else action if name.startswith(TRAINABLE_PREFIXES) else other).append((name, p))
    if other:
        raise ValueError(f"unexpected trainable parameters with LoRA on: {[n for n, _ in other][:5]}")
    return [{"params": [p for _, p in lora], "lr": float(lora_lr), "name": "lora"},
            {"params": [p for _, p in action], "lr": float(action_lr), "name": "action_model"}]


@contextlib.contextmanager
def merged(model: nn.Module):
    """state_dict() of the model gives merged plain weights while inside."""
    mods = [m for m in model.modules() if isinstance(m, LoRALinear)]
    for m in mods:
        m.merge_on_save = True
    try:
        yield len(mods)
    finally:
        for m in mods:
            m.merge_on_save = False
