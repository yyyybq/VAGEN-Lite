"""C line (q38 s4): the NFP head and loss of Cambrian-S stage 4 on Qwen3.8-27B (STAGE0_REPORT sections 2, 3, 6).

NFP = next-frame prediction (the paper's LFP): a small head on the language model's last hidden states predicts, at
every video token of time unit t, the frozen vision encoder's feature of the same spatial position in unit t+1.

- Head (R1): ``Linear(H, H) -> GELU -> Linear(H, Dv)`` with bias, H = text hidden size (5,120), Dv = vision width
  (1,152); weights ``trunc_normal_(std 0.02, cut at +-2 std)``, biases 0 (the official code calls an undefined
  ``trunc_normal_``; this is its intent).  Its input is the final-norm output, the same tensor lm_head reads.
- Target (R2, R3): the output of vision ``blocks[-2]`` (blocks[25] of 27, before the merger), averaged over each 2x2
  merge window.  The encoder's patches are already ordered by merge window (the merger views 4 consecutive patches as
  one token), so target k is the mean of patches 4k..4k+3 and lines up one to one with the k-th video token the
  language model sees.  Detached.
- Alignment (R4): one unit = temporal_patch_size (2) frames = 1 s at 2 fps.  Every unit of a video has the same
  h*w/4 tokens, so the token at offset j of unit t predicts the target at offset j of unit t+1; the last unit
  predicts nothing.  Only videos of NFP samples count; QA videos and images never do.
- Loss: per token MSE (mean over the Dv dims) and 1 - cos; summed here, normalised by the caller (lf_nfp: over all
  NFP tokens of the optimiser step on all ranks, R12).  total = CE + 0.1 MSE + 0.1 cos.
- Copy baseline: the same two errors for "prediction = this unit's own target" (target_t against target_{t+1}), on
  the same pairs, to read the head's numbers against.

NFPTap holds the hooks.  It runs the head inside the model's forward (a forward hook on the top module), where an
FSDP1 root unit's parameters are unsharded, exactly as lm_head's are; FSDP1 keeps the root's parameters unsharded
from forward to backward, so the head's backward is on the same footing as lm_head's.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

TARGET_BLOCK = -2  # R3: the vision encoder's second-to-last block
INIT_STD = 0.02


class NFPHead(nn.Module):
    """``Linear(in, hidden) -> GELU -> Linear(hidden, out)``; hidden defaults to in (the official head)."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: Optional[int] = None, std: float = INIT_STD,
                 device=None, dtype=None):
        super().__init__()
        hidden_dim = hidden_dim or in_dim
        self.fc1 = nn.Linear(in_dim, hidden_dim, device=device, dtype=dtype)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim, device=device, dtype=dtype)
        self.std = std
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for lin in (self.fc1, self.fc2):
            nn.init.trunc_normal_(lin.weight, mean=0.0, std=self.std, a=-2 * self.std, b=2 * self.std)
            nn.init.zeros_(lin.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


def merge_targets(block_out: torch.Tensor, merge: int = 2) -> torch.Tensor:
    """(patches, Dv) encoder output -> (patches / merge**2, Dv) fp32, the mean of each merge window."""
    m2 = merge * merge
    if block_out.dim() != 2 or block_out.shape[0] % m2:
        raise ValueError(f"encoder output {tuple(block_out.shape)} is not (k*{m2}, D)")
    return block_out.detach().float().view(-1, m2, block_out.shape[-1]).mean(1)


def unit_pairs(video_grid_thw: torch.Tensor, nfp_video: Sequence[bool], merge: int = 2
               ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Index pairs (src, dst) into the flat list of merged video tokens of a batch (videos in grid order): src runs
    over units 0..t-2 of every NFP video, dst = src + tokens per unit.  Videos with fewer than 2 units give none."""
    grid = video_grid_thw.tolist() if torch.is_tensor(video_grid_thw) else [list(g) for g in video_grid_thw]
    flags = nfp_video.tolist() if torch.is_tensor(nfp_video) else list(nfp_video)
    if len(grid) != len(flags):
        raise ValueError(f"{len(grid)} videos in the grid, {len(flags)} NFP flags")
    src: List[torch.Tensor] = []
    dst: List[torch.Tensor] = []
    off = 0
    for (t, h, w), on in zip(grid, flags):
        if (h * w) % (merge * merge):
            raise ValueError(f"grid {t}x{h}x{w} is not a multiple of the {merge}x{merge} merge")
        n = h * w // (merge * merge)
        if on and t >= 2:
            s = torch.arange(off, off + (t - 1) * n)
            src.append(s)
            dst.append(s + n)
        off += t * n
    if not src:
        return torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long)
    return torch.cat(src), torch.cat(dst)


def count_pairs(video_grid_thw: Optional[torch.Tensor], nfp_video: Optional[torch.Tensor], merge: int = 2) -> int:
    """Number of NFP target tokens of a batch (what the loss is averaged over)."""
    if video_grid_thw is None or nfp_video is None or len(nfp_video) == 0:
        return 0
    n = 0
    for (t, h, w), on in zip(video_grid_thw.tolist(), nfp_video.tolist()):
        if on and t >= 2:
            n += (t - 1) * (h * w // (merge * merge))
    return n


def error_sums(pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sums over tokens of the per-token MSE (mean over dims) and of 1 - cos, in fp32 (official cambrian_qwen2.py
    323-335 before its masked mean)."""
    pred, target = pred.float(), target.float()
    mse = (pred - target).pow(2).mean(-1)
    cos = F.cosine_similarity(pred, target, dim=-1)
    return mse.sum(), (1.0 - cos).sum()


@dataclass
class TapResult:
    n_pairs: int = 0
    n_videos: int = 0
    mse_sum: Optional[torch.Tensor] = None  # with grad, only when the head ran
    cos_sum: Optional[torch.Tensor] = None
    copy_mse_sum: float = 0.0
    copy_cos_sum: float = 0.0


class NFPTap:
    """Hooks on an (unwrapped) Qwen3_5ForConditionalGeneration: the target block's output while the model encodes
    videos, the final norm's output, and the head run on the top module's forward.  ``arm(flags)`` before a forward
    (one bool per video of the batch), ``collect()`` after it."""

    def __init__(self, model: nn.Module, head: Optional[nn.Module] = None, merge: Optional[int] = None):
        self.model = model
        inner = model.model
        self.video_token_id = int(model.config.video_token_id)
        self.merge = int(merge or model.config.vision_config.spatial_merge_size)
        self.head = head
        self._capture = False
        self._armed = False
        self._flags: Optional[torch.Tensor] = None
        self._vision_out: Optional[torch.Tensor] = None
        self._hidden: Optional[torch.Tensor] = None
        self._result: Optional[TapResult] = None
        self.handles = [
            inner.visual.blocks[TARGET_BLOCK].register_forward_hook(self._on_block),
            inner.language_model.norm.register_forward_hook(self._on_norm),
            model.register_forward_hook(self._on_model, with_kwargs=True),
        ]
        orig = inner.get_video_features

        def get_video_features(*args, **kwargs):
            self._capture = True
            try:
                return orig(*args, **kwargs)
            finally:
                self._capture = False

        inner.get_video_features = get_video_features
        self._orig_get_video_features = orig

    def remove(self) -> None:
        for h in self.handles:
            h.remove()
        self.model.model.get_video_features = self._orig_get_video_features

    # hooks
    def _on_block(self, module, args, output):
        if self._armed and self._capture and self._vision_out is None:
            self._vision_out = output.detach()

    def _on_norm(self, module, args, output):
        if self._armed:
            self._hidden = output

    def _on_model(self, module, args, kwargs, output):
        if not self._armed:
            return
        res = TapResult()
        self._result = res
        grid = kwargs.get("video_grid_thw")
        flags = self._flags
        if grid is None or flags is None or len(flags) == 0 or not bool(flags.any()) or self._vision_out is None:
            return
        input_ids = kwargs.get("input_ids")
        if input_ids is None:
            raise ValueError("NFP needs input_ids to find the video tokens")
        targets = merge_targets(self._vision_out, self.merge)
        vid_mask = input_ids == self.video_token_id
        if int(vid_mask.sum()) != targets.shape[0]:
            raise ValueError(f"{int(vid_mask.sum())} video tokens in the batch, {targets.shape[0]} encoder targets")
        src, dst = unit_pairs(grid, flags, self.merge)
        res.n_videos = int(flags.sum())
        res.n_pairs = int(src.numel())
        if res.n_pairs == 0:
            return
        src, dst = src.to(targets.device), dst.to(targets.device)
        y_next = targets[dst]
        with torch.no_grad():
            c_mse, c_cos = error_sums(targets[src], y_next)
        res.copy_mse_sum, res.copy_cos_sum = float(c_mse), float(c_cos)
        if self.head is not None:
            hidden = self._hidden[vid_mask]  # row-major = the order masked_scatter filled the video embeddings
            pred = self.head(hidden[src.to(hidden.device)])
            res.mse_sum, res.cos_sum = error_sums(pred, y_next.to(pred.device))

    # per batch
    def arm(self, nfp_video: Optional[torch.Tensor]) -> None:
        self._armed, self._flags = True, nfp_video
        self._vision_out = self._hidden = self._result = None

    def collect(self) -> TapResult:
        res = self._result if self._result is not None else TapResult()
        self.disarm()
        return res

    def disarm(self) -> None:
        self._armed = False
        self._flags = self._vision_out = self._hidden = self._result = None


def attach_head(model: nn.Module, dtype=torch.float32) -> NFPHead:
    """Add ``model.nfp_head`` (text hidden -> vision width), fp32 like LLaMA-Factory's other trainable weights, on
    lm_head's device (meta on the non-zero ranks of a cpu_ram_efficient_loading FSDP start: rank 0's values are
    broadcast by sync_module_states)."""
    if hasattr(model, "nfp_head"):
        raise ValueError("model already has an nfp_head")
    h = int(model.config.text_config.hidden_size)
    dv = int(model.config.vision_config.hidden_size)
    dev = model.lm_head.weight.device
    head = NFPHead(h, dv, device=dev, dtype=dtype)
    model.nfp_head = head
    return head


def scale_for_global_mean(world_size: int, grad_accum: int, total_pairs: float) -> float:
    """Factor that turns a micro-batch's error sum into its share of the global token mean (R12): the trainer divides
    each micro-batch loss by grad_accum and FSDP averages gradients over world_size ranks, so
    sum * world_size * grad_accum / total_pairs adds up to sum_all / total_pairs over the step."""
    if total_pairs <= 0:
        return 0.0
    return float(world_size) * float(grad_accum) / float(total_pairs)


def summarize(acc: Dict[str, float]) -> Dict[str, Any]:
    """Log values from summed counters (after an all-reduce): text_loss (mean CE over micro-batches with text),
    nfp_mse / nfp_cos (token means of the head's MSE and 1-cos), copy_mse / copy_cos (same for the copy baseline)."""
    out: Dict[str, Any] = {}
    if acc.get("ce_n", 0) > 0:
        out["text_loss"] = acc["ce_sum"] / acc["ce_n"]
    n = acc.get("pairs", 0)
    out["nfp_pairs"] = int(n)
    out["nfp_videos"] = int(acc.get("videos", 0))
    if n > 0:
        if acc.get("head_pairs", 0) > 0:
            out["nfp_mse"] = acc["mse_sum"] / acc["head_pairs"]
            out["nfp_cos"] = acc["cos_sum"] / acc["head_pairs"]
        out["copy_mse"] = acc["copy_mse_sum"] / n
        out["copy_cos"] = acc["copy_cos_sum"] / n
    return out


ACC_KEYS = ("ce_sum", "ce_n", "pairs", "videos", "head_pairs", "mse_sum", "cos_sum", "copy_mse_sum", "copy_cos_sum")
