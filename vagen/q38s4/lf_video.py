"""C line (q38 s4): D4 frame sampling for LLaMA-Factory's Qwen3-VL video plugin (STAGE0_REPORT section 3).

A video of the converted data carries how it is sampled in its path: ``<file>|s4|qa`` or ``<file>|s4|nfp:u=<u>``
(LLaMA-Factory keeps a path that is not a file as it is, with one warning).  S4VideoPlugin reads the suffix at both
places LLaMA-Factory samples frames -- the token count at preprocessing (_get_qwen_video_stream_metadata, no
decoding) and the decoding in the collator (_regularize_videos) -- so the two always agree:

- qa: ``video_maxlen`` (256) frames spread uniformly over the whole video (linspace, floor), repeated frames for a
  short video (official utils.py 50-71: exactly 128 frames whatever the length; LF's own sampler drops repeats and
  takes fewer frames for a short video);
- nfp: a contiguous window of up to ``video_maxlen`` frames at ``video_fps`` (2), starting at the fraction u in
  [0, 1) of the possible starts (u is drawn once per sample by the data build, with its seed); a video shorter than
  the window gives all its frames; the count is made even (2 frames = 1 unit) by dropping the last;
- no suffix: LLaMA-Factory's own sampling, unchanged.

The frame size is LLaMA-Factory's (video_max_pixels, 256x256 in the recipe) and the Qwen3-VL processor's
smart_resize, as for every other video.  Timestamps follow LF's Qwen3-VL convention (frames_indices = source index /
source fps * video_fps), so an NFP window's units are stamped with their time in the source video.  Decoding stops
at the last frame needed; a frame the container promised but did not deliver is replaced by the closest earlier one
(counted in ``lf_plugin.MISSING``) so the token count fixed at preprocessing still holds.  The plugin class is in
lf_plugin (it imports LLaMA-Factory; this module does not).
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

SEP = "|s4|"
KINDS = ("qa", "nfp")


@dataclass(frozen=True)
class VideoSpec:
    kind: str  # "qa" | "nfp"
    u: float = 0.0

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"unknown video kind {self.kind!r}")
        if not 0.0 <= self.u < 1.0:
            raise ValueError(f"u={self.u} not in [0, 1)")


def encode(path: str, spec: VideoSpec) -> str:
    if SEP in path:
        raise ValueError(f"{path!r} already has a spec")
    return f"{path}{SEP}qa" if spec.kind == "qa" else f"{path}{SEP}nfp:u={spec.u:.6f}"


def parse(video: Any) -> Tuple[Any, Optional[VideoSpec]]:
    """(path, spec) of a converted video; anything else (no suffix, a frame list) -> (video, None)."""
    if not isinstance(video, str) or SEP not in video:
        return video, None
    path, tail = video.rsplit(SEP, 1)
    if tail == "qa":
        return path, VideoSpec("qa")
    if tail.startswith("nfp:u="):
        return path, VideoSpec("nfp", float(tail[len("nfp:u="):]))
    raise ValueError(f"bad video spec {tail!r} in {video!r}")


def is_nfp(video: Any) -> bool:
    spec = parse(video)[1]
    return spec is not None and spec.kind == "nfp"


def qa_indices(total_frames: int, n: int) -> List[int]:
    if total_frames < 1 or n < 1:
        raise ValueError(f"total_frames={total_frames}, n={n}")
    return np.linspace(0, total_frames - 1, n).astype(np.int64).tolist()


def nfp_indices(total_frames: int, avg_fps: float, u: float, fps: float, n_max: int) -> List[int]:
    """Source frame indices of the NFP window: grid k/fps seconds, k = k0 .. k0+n-1."""
    if avg_fps <= 0:
        raise ValueError(f"avg_fps={avg_fps}")
    n_all = int(math.floor(total_frames / avg_fps * fps + 1e-9))  # frames on the fps grid inside the video
    n = min(n_max, n_all)
    n -= n % 2
    if n < 4:
        raise ValueError(f"video too short for NFP: {total_frames} frames at {avg_fps} fps -> {n} frames at {fps}")
    span = n_all - n
    k0 = min(span, int(math.floor(u * (span + 1))))
    return [min(total_frames - 1, int(math.floor((k0 + k) / fps * avg_fps + 1e-9))) for k in range(n)]


def spec_indices(spec: VideoSpec, total_frames: int, avg_fps: float, video_fps: float, video_maxlen: int) -> List[int]:
    if spec.kind == "qa":
        return qa_indices(total_frames, video_maxlen)
    return nfp_indices(total_frames, avg_fps, spec.u, video_fps, video_maxlen)


def stream_info(container, stream) -> Dict[str, float]:
    """total frames, average fps, duration (s) of a video stream; raises when the container does not say."""
    if stream.average_rate is None or float(stream.average_rate) <= 0:
        raise ValueError("video stream without an average frame rate")
    avg = float(stream.average_rate)
    if stream.duration is not None:
        dur = float(stream.duration * stream.time_base)
    elif container.duration is not None:
        dur = container.duration / 1e6
    else:
        raise ValueError("video stream without a duration")
    total = int(stream.frames) if stream.frames else int(math.floor(dur * avg + 1e-9))
    if total < 1:
        raise ValueError("video stream without frames")
    return {"total_frames": total, "average_fps": avg, "duration": dur}


def decode_at(container, stream, indices: List[int]):
    """The frames at ``indices`` (repeats allowed) as PIL images, decoding from the start up to the last one;
    returns (frames, number of indices the stream did not deliver)."""
    want = set(indices)
    last = max(indices)
    got = {}
    container.seek(0)
    tail = None
    for i, frame in enumerate(container.decode(stream)):
        if i in want:
            got[i] = frame.to_image()
        if i >= last:
            break
        tail = (i, frame)
    else:  # the stream ended before the last index: keep its final frame as the closest earlier one
        if tail is not None and tail[0] not in got:
            got[tail[0]] = tail[1].to_image()
    if not got:
        raise ValueError("no frame decoded")
    missing = 0
    have = sorted(got)
    out = []
    for i in indices:
        if i not in got:
            missing += 1
            j = max((k for k in have if k <= i), default=have[0])
            got[i] = got[j]
        out.append(got[i])
    return out, missing


def video_exists(video: Any) -> bool:
    path, _ = parse(video)
    return isinstance(path, str) and os.path.isfile(path)
