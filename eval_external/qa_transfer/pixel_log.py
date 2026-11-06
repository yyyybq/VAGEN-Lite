"""What the model actually sees: every call of the Qwen2.5-VL / Qwen2-VL / Qwen3-VL processor appends its image and
video grids (after all resizing) to a JSONL file.  Installed by the backend launchers (vlmeval_launch, lmms_launch)
when EE_PIXEL_LOG names the file; qa_transfer/collect.py summarises it into results.json["input_pixels"].

A grid row (t, h, w) counts patches of `patch` x `patch` pixels (14 for Qwen2/2.5-VL, 16 for Qwen3-VL; each record
carries its patch size, records without one are 14): an image is t = 1, a video has t = frames / 2 (two frames per
temporal patch).  Pixels per image or video frame = h * w * patch^2; visual tokens = t * h * w / 4 (2 x 2 merge).
"""

from __future__ import annotations

import json
import os
import statistics
import threading
from pathlib import Path

PATCH = 14  # records written before 2026-11-06 have no "patch" key
_LOCK = threading.Lock()


def _rows(feature, key: str) -> list[list[int]]:
    grid = feature.get(key) if hasattr(feature, "get") else None
    return [] if grid is None else [[int(x) for x in row] for row in grid.tolist()]


def _processor_classes() -> list[tuple[type, int]]:
    from transformers.models.qwen2_5_vl.processing_qwen2_5_vl import Qwen2_5_VLProcessor
    from transformers.models.qwen2_vl.processing_qwen2_vl import Qwen2VLProcessor

    out = [(Qwen2_5_VLProcessor, 14), (Qwen2VLProcessor, 14)]
    try:  # transformers >= 4.57 (the eval-external-q3 env)
        from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor

        out.append((Qwen3VLProcessor, 16))
    except ImportError:
        pass
    return out


def install(path: str | os.PathLike | None = None) -> bool:
    path = path or os.environ.get("EE_PIXEL_LOG")
    if not path:
        return False
    for cls, patch in _processor_classes():
        if getattr(cls.__call__, "_ee_pixel_log", False):
            continue
        original = cls.__call__

        def call(self, *args, _original=original, _patch=patch, **kwargs):
            out = _original(self, *args, **kwargs)
            rec = {"image_grid_thw": _rows(out, "image_grid_thw"), "video_grid_thw": _rows(out, "video_grid_thw")}
            if rec["image_grid_thw"] or rec["video_grid_thw"]:
                rec["patch"] = _patch
                with _LOCK, open(path, "a") as f:
                    f.write(json.dumps(rec) + "\n")
            return out

        call._ee_pixel_log = True
        cls.__call__ = call
    return True


def summarize(path: Path) -> dict | None:
    """Per model call: images / video frames and visual tokens; per image or frame: pixels (min, median, max)."""
    if not Path(path).exists():
        return None
    calls, pixels, tokens, frames = 0, [], [], []
    for line in open(path):
        rec = json.loads(line)
        patch = rec.get("patch", PATCH)
        calls += 1
        n_frames, n_tokens = 0, 0
        for t, h, w in rec["image_grid_thw"]:
            pixels.append(h * w * patch * patch)
            n_frames += 1
            n_tokens += t * h * w // 4
        for t, h, w in rec["video_grid_thw"]:
            pixels += [h * w * patch * patch] * (2 * t)
            n_frames += 2 * t
            n_tokens += t * h * w // 4
        frames.append(n_frames)
        tokens.append(n_tokens)
    if not calls:
        return None

    def five(v):
        return {"min": min(v), "median": statistics.median(v), "max": max(v)}

    return {"calls": calls, "pixels_per_image_or_frame": five(pixels), "images_or_frames_per_call": five(frames),
            "visual_tokens_per_call": five(tokens)}
