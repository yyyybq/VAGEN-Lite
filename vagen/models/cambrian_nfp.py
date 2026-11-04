"""Cambrian-S's next-frame prediction (NFP) loss on the concat path (the model line, the user's decision 47 乙 of
2026-11-03: the predictive loss trained together with RL, as the August runs from Cambrian-S-7B-LFP did).

In the concat path one sequence holds an episode: every turn's frame as a block of 756 image positions (27 rows of
27 patches and a newline), the turn's answer after it, the next turn's frame after that.  For every frame but the
last, the head predicts the next frame's SigLIP features from the language model's last hidden states at this
frame's positions -- before the answer, so the next frame is the one the model's own coming actions lead to, as in
the August runs (vagen/agent_loop/agent_loop_no_concat.py, vagen/models/cambrian_register.py _compute_nfp_loss).

Two changes from the LFP checkpoints' design, both because the start model (Cambrian-S-7B, the model line's SFT on
it) has no LFP training:
  - no MIV input: the LFP checkpoints overwrite each frame's first 64 positions with an 8x8 view and predict there;
    here the frame's input is left as the start model saw it, and the hidden states of its 27 x 27 patch positions
    are average-pooled to 8 x 8 for the head;
  - the target is detached: the next frame's SigLIP features (as the sequence's own vision pass computed them) give
    no gradient, so the loss cannot be lowered by making the targets easier.
The target grid is Cambrian-S's: the 27 x 27 SigLIP features bilinearly resized to 8 x 8.  The loss is Cambrian-S's:
mean squared error (over the 1152 features) and cosine distance, averaged over the 64 cells of every pair and
weighted by nfp_mse_loss_weight / nfp_cosine_loss_weight (0.1 / 0.1, Cambrian-S's and the August runs').  The head is
Cambrian-S's two-layer MLP (hidden -> hidden -> 1152, GELU) with its initialisation.

The head runs in every training forward, also on an episode with one frame (weighted 0 there), so every rank's
backward pass meets it (FSDP's gradient reduction needs the same parameters on every rank).

The statistics (frame pairs, the unweighted MSE / cosine distance, the copy baseline's) reach verl's dp_actor beside
the forward's output (keep_stats / take_stats): FSDP (torch 2.8) rebuilds the output from its dataclass fields
before the backward pass (_register_pre_backward_hooks -> _apply_to_tensors) and drops any other attribute, which
lost csnfp's statistics (2026-11-04).
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGE_TOKEN_INDEX = -200
GRID = 8  # Cambrian-S's MIV side (miv_token_len 64)


def build_head(hidden_size: int, mm_hidden_size: int) -> nn.Sequential:
    """Cambrian-S's NFP head (cambrian/model/cambrian_arch.py) with its initialisation."""
    head = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.GELU(), nn.Linear(hidden_size, mm_hidden_size))
    for i in (0, 2):
        nn.init.trunc_normal_(head[i].weight, std=0.02)
        nn.init.zeros_(head[i].bias)
    return head


def patch_offsets(side: int, newline: bool = True) -> torch.Tensor:
    """The offsets of a frame's side x side patch positions inside its block (row-major; with newline every row is
    followed by one newline position, cambrian_miv.append_image_newline)."""
    width = side + 1 if newline else side
    return (torch.arange(side).unsqueeze(1) * width + torch.arange(side).unsqueeze(0)).reshape(-1)


def concat_nfp_pairs(image_ids: torch.Tensor, position_ids: Optional[torch.Tensor], per_image: int,
                     image_index: int = IMAGE_TOKEN_INDEX) -> List[Tuple[int, int, int, int]]:
    """[(row, start of frame k, global index of frame k, global index of frame k + 1)] for consecutive frames of one
    sequence.  Global indices count the frames row by row in order, as the adapter's _embed_multimodal_batch scatters
    them.  A row holds one episode, or several after verl's padding removal: a sequence starts where its positions
    restart at 0, and no pair crosses that."""
    pairs: List[Tuple[int, int, int, int]] = []
    g = 0
    for b in range(image_ids.shape[0]):
        pos = (image_ids[b] == image_index).nonzero(as_tuple=True)[0]
        if pos.numel() % per_image:
            raise ValueError(f"row {b}: {pos.numel()} image positions, not a multiple of {per_image}")
        starts = pos[::per_image].tolist()
        seq = None
        if position_ids is not None and position_ids.dim() == 2 and position_ids.shape == image_ids.shape:
            seq = torch.cumsum((position_ids[b] == 0).long(), 0)  # the sequence each token belongs to
        for k in range(len(starts) - 1):
            if seq is not None and int(seq[starts[k]]) != int(seq[starts[k + 1]]):
                continue
            pairs.append((b, starts[k], g + k, g + k + 1))
        g += len(starts)
    return pairs


def _grid_targets(raw: torch.Tensor, side: int, grid: int = GRID) -> torch.Tensor:
    """(n, side * side, d) SigLIP features -> (n, grid * grid, d), bilinear as Cambrian-S's MIV view."""
    n, _, d = raw.shape
    x = raw.float().view(n, side, side, d).permute(0, 3, 1, 2)
    x = F.interpolate(x, size=(grid, grid), mode="bilinear", align_corners=False)
    return x.permute(0, 2, 3, 1).reshape(n, grid * grid, d)


def _pool_hidden(hidden: torch.Tensor, side: int, grid: int = GRID) -> torch.Tensor:
    """(n, side * side, h) hidden states -> (n, grid * grid, h), average-pooled."""
    n, _, h = hidden.shape
    x = hidden.view(n, side, side, h).permute(0, 3, 1, 2)
    x = F.adaptive_avg_pool2d(x, (grid, grid))
    return x.permute(0, 2, 3, 1).reshape(n, grid * grid, h)


def _losses(pred: torch.Tensor, target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mean squared error over the features and cosine distance, each averaged over every cell."""
    mse = F.mse_loss(pred, target, reduction="none").mean(-1).mean()
    cos = (1.0 - F.cosine_similarity(pred, target, dim=-1)).mean()
    return mse, cos


def concat_nfp_loss(hidden_states: torch.Tensor, pairs: Sequence[Tuple[int, int, int, int]], raw: torch.Tensor,
                    head: nn.Module, *, side: int = 27, newline: bool = True, mse_weight: float = 0.1,
                    cos_weight: float = 0.1) -> Tuple[torch.Tensor, Dict[str, float]]:
    """The weighted NFP loss of ``pairs`` (concat_nfp_pairs) and its statistics.

    hidden_states (bs, seq, h): the language model's last hidden states; raw (n_frames, side * side, d): every
    frame's SigLIP features in global order (detached here).  Statistics: pairs, the unweighted mse / cos, and the
    same two for the copy baseline (the current frame's grid as the prediction of the next one: what a model that
    expects nothing to change would score)."""
    raw = raw.detach()
    if not pairs:  # one frame: the head still runs (every rank's backward meets it), weighted 0
        out = head(hidden_states[:1, :1])
        return out.float().sum() * 0.0, {"pairs": 0.0}
    offs = patch_offsets(side, newline).to(hidden_states.device)
    rows = torch.tensor([p[0] for p in pairs], device=hidden_states.device)
    starts = torch.tensor([p[1] for p in pairs], device=hidden_states.device)
    idx = starts.unsqueeze(1) + offs.unsqueeze(0)  # (P, side * side)
    hidden = hidden_states[rows.unsqueeze(1), idx]  # (P, side * side, h)
    pred = head(_pool_hidden(hidden, side)).float()  # (P, 64, d)
    cur = torch.tensor([p[2] for p in pairs], device=raw.device)
    nxt = torch.tensor([p[3] for p in pairs], device=raw.device)
    target = _grid_targets(raw[nxt], side).to(pred.device)
    mse, cos = _losses(pred, target)
    loss = mse_weight * mse + cos_weight * cos
    with torch.no_grad():
        copy_mse, copy_cos = _losses(_grid_targets(raw[cur], side).to(pred.device), target)
    stats = {"pairs": float(len(pairs)), "mse": float(mse.detach()), "cos": float(cos.detach()),
             "copy_mse": float(copy_mse), "copy_cos": float(copy_cos)}
    return loss, stats


NFP_STATS = ("pairs", "mse", "cos", "copy_mse", "copy_cos")  # concat_nfp_loss's statistics


def nfp_metrics(aux_value: float, stats: Optional[Dict[str, float]], coef: float,
                loss_scale_factor: float) -> Dict[str, float]:
    """The actor's metrics for one micro-batch's auxiliary loss (verl dp_actor).  Without statistics (the LFP path)
    as before.  With them, the same keys for every micro-batch: verl joins each worker's per-key lists and averages
    them with np.mean (DataProto.concat, reduce_metrics), which fails on lists of different lengths -- an earlier
    version left the statistics out for an episode without a pair and stopped csnfp01 three times (2026-11-04).  Such
    an episode (its head pass weighted 0, no loss) gives actor/nfp_loss_active 0 and 0 for the loss and every
    statistic, so the mean over the episodes with a pair is a logged mean divided by actor/nfp_loss_active's."""
    out = {"actor/nfp_loss": aux_value, "actor/nfp_loss_scaled": aux_value * coef * loss_scale_factor,
           "actor/nfp_loss_coef": coef, "actor/image_gen_loss_coef": coef, "actor/nfp_loss_active": 1.0}
    if stats is None:
        return out
    active = bool(stats.get("pairs"))
    out["actor/nfp_loss_active"] = 1.0 if active else 0.0
    for k in NFP_STATS:
        out[f"actor/nfp_{k}"] = float(stats.get(k, 0.0)) if active else 0.0
    return out


_KEPT: Optional[Dict[str, float]] = None  # the last training forward's statistics (keep_stats / take_stats)


def keep_stats(stats: Optional[Dict[str, float]]) -> None:
    """The adapter: keep one training forward's statistics for the actor (take_stats)."""
    global _KEPT
    _KEPT = None if stats is None else dict(stats)


def take_stats(output=None) -> Optional[Dict[str, float]]:
    """The actor, right after its forward: that forward's statistics -- the output's own nfp_stats where it still
    has them (no FSDP), else the kept ones -- and none are kept after, so a forward that keeps none (the LFP path,
    the other models) never reads stale ones."""
    global _KEPT
    kept, _KEPT = _KEPT, None
    own = getattr(output, "nfp_stats", None)
    return own if own is not None else kept
