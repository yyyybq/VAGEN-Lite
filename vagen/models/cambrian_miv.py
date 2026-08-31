"""Shared Cambrian-S visual-token construction helpers.

These functions mirror the FSDP training adapter's Cambrian-S-LFP policy-forward
path: project 27x27 SI features, append image-newline tokens, and optionally
overwrite the first 64 image-token positions with the 8x8 MIV view.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def build_miv_features(
    projected_features: torch.Tensor,
    *,
    si_side_len: int,
    miv_token_len: int,
) -> torch.Tensor:
    """Build the 8x8 MIV view from projected 27x27 SI features."""
    miv_side_len = int(miv_token_len**0.5)
    total_n, _, hidden = projected_features.shape
    feat_bchw = (
        projected_features.view(total_n, si_side_len, si_side_len, hidden)
        .permute(0, 3, 1, 2)
        .float()
    )
    miv_bchw = F.interpolate(
        feat_bchw,
        size=(miv_side_len, miv_side_len),
        mode="bilinear",
        align_corners=False,
    )
    return (
        miv_bchw.permute(0, 2, 3, 1)
        .reshape(total_n, miv_token_len, hidden)
        .to(projected_features.dtype)
    )


def append_image_newline(
    projected_features: torch.Tensor,
    image_newline: torch.Tensor,
    *,
    si_side_len: int,
) -> torch.Tensor:
    """Append one learned newline token after each 27-token image row."""
    total_imgs, _, hidden = projected_features.shape
    features = projected_features.view(total_imgs, si_side_len, si_side_len, hidden)
    newline = image_newline.to(features.dtype)
    newline_exp = newline.view(1, 1, 1, hidden).expand(total_imgs, si_side_len, 1, hidden)
    features = torch.cat([features, newline_exp], dim=2)
    return features.view(total_imgs, -1, hidden)


def build_cambrian_visual_features(
    projected_features: torch.Tensor,
    image_newline: torch.Tensor | None,
    *,
    si_token_len: int,
    mm_use_newline: bool,
    nfp_head: bool,
    miv_token_len: int,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Return final visual tokens plus optional MIV features.

    The returned visual token tensor is the exact tensor that should be scattered
    into image-token positions before the language model sees the prompt.
    """
    si_side_len = int(si_token_len**0.5)
    visual_features = projected_features
    miv_features = None

    if nfp_head and miv_token_len > 0:
        miv_features = build_miv_features(
            projected_features,
            si_side_len=si_side_len,
            miv_token_len=miv_token_len,
        )

    if mm_use_newline:
        if image_newline is None:
            raise ValueError("image_newline is required when mm_use_newline=True")
        visual_features = append_image_newline(
            projected_features,
            image_newline,
            si_side_len=si_side_len,
        )

    if miv_features is not None:
        visual_features = visual_features.clone()
        visual_features[:, :miv_token_len, :] = miv_features

    return visual_features, miv_features
