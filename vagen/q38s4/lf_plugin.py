"""C line (q38 s4): S4VideoPlugin, LLaMA-Factory's Qwen3-VL plugin with the D4 frame sampling of lf_video.

Module-level (not built in a function) so that datasets' worker processes can pickle the template that holds it.
Only the two places that pick frames change; a video without the ``|s4|`` suffix goes through LF's own code.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass

from llamafactory.data.mm_plugin import Qwen3VLPlugin  # first: PyAV imported before LLaMA-Factory once a
import av  # noqa: E402  Qwen3.8 processor has loaded (cv2, torchvision) aborts the process (verl env, 2026-11-22)

from vagen.q38s4.lf_video import decode_at, parse, spec_indices, stream_info


def _open_video_stream(path):
    container = av.open(path, "r")
    stream = next((s for s in container.streams if s.type == "video"), None)
    if stream is None:
        container.close()
        raise ValueError(f"{path}: no video stream")
    return container, stream


@dataclass
class S4VideoPlugin(Qwen3VLPlugin):
    def _get_qwen_video_stream_metadata(self, video, video_fps, video_maxlen):
        path, spec = parse(video)
        if spec is None:
            return super()._get_qwen_video_stream_metadata(video, video_fps, video_maxlen)
        container, stream = _open_video_stream(path)
        try:
            info = stream_info(container, stream)
            return {
                "width": stream.width,
                "height": stream.height,
                "average_fps": info["average_fps"],
                "sample_indices": spec_indices(spec, info["total_frames"], info["average_fps"], video_fps,
                                               video_maxlen),
            }
        finally:
            container.close()

    def _regularize_videos(self, videos, **kwargs):
        if not any(parse(v)[1] is not None for v in videos):
            return super()._regularize_videos(videos, **kwargs)
        video_fps = kwargs.get("video_fps", 2.0)
        video_maxlen = kwargs.get("video_maxlen", 128)
        out = {"videos": [], "fps_per_video": [], "durations": [], "frames_indices": []}
        for video in videos:
            path, spec = parse(video)
            if spec is None:
                one = super()._regularize_videos([video], **kwargs)
                for k in out:
                    out[k].extend(one[k])
                continue
            container, stream = _open_video_stream(path)
            try:
                info = stream_info(container, stream)
                idx = spec_indices(spec, info["total_frames"], info["average_fps"], video_fps, video_maxlen)
                frames, missing = decode_at(container, stream, idx)
            finally:
                container.close()
            if missing:
                MISSING["frames"] += missing
                MISSING["videos"] += 1
            if len(frames) % 2:
                raise AssertionError("D4 sampling gives an even frame count")
            out["videos"].append(self._regularize_images(frames, **kwargs)["images"])
            out["fps_per_video"].append(len(idx) / info["duration"])
            out["durations"].append(info["duration"])
            out["frames_indices"].append([i / info["average_fps"] * video_fps for i in idx])
        return out


MISSING = {"frames": 0, "videos": 0}  # frames the containers promised and did not deliver, in this process


def to_s4_plugin(plugin):
    """An S4VideoPlugin with the same fields as LF's Qwen3-VL plugin ``plugin`` (the qwen3_8 template's)."""
    if type(plugin) is not Qwen3VLPlugin:
        raise TypeError(f"expected LF's Qwen3VLPlugin, got {type(plugin).__name__}")
    return S4VideoPlugin(**{f.name: getattr(plugin, f.name) for f in dataclasses.fields(plugin) if f.init})
