"""CPU tests of the D4 frame sampling (vagen.q38s4.lf_video) and its LLaMA-Factory plugin (lf_plugin).

The videos are written here with PyAV (mpeg4, flat grey frames whose level encodes the frame index), so decoding can
be checked frame by frame.  Run as test_nfp.py says.
"""
from __future__ import annotations

import numpy as np
import pytest

from vagen.q38s4 import lf_video as V
from vagen.q38s4.tests.tiny import needs_models, processor

pytest.importorskip("llamafactory.data.mm_plugin")  # before PyAV: the other order aborts (see lf_plugin)
av = pytest.importorskip("av")

LEVEL = 3  # grey level step per frame index (videos here have at most 85 frames)


def write_video(path, n_frames, fps=10, w=64, h=48):
    c = av.open(str(path), "w")
    s = c.add_stream("mpeg4", rate=fps)
    s.width, s.height, s.pix_fmt = w, h, "yuv420p"
    s.bit_rate = 4_000_000
    for i in range(n_frames):
        arr = np.full((h, w, 3), (LEVEL * i) % 256, dtype=np.uint8)
        for p in s.encode(av.VideoFrame.from_ndarray(arr, format="rgb24")):
            c.mux(p)
    for p in s.encode():
        c.mux(p)
    c.close()
    return str(path)


def level_of(img):
    return round(float(np.asarray(img).mean()) / LEVEL)


# ------------------------------------------------------------------------------------------------ specs, indices
def test_encode_parse_roundtrip():
    assert V.encode("/a/b.mp4", V.VideoSpec("qa")) == "/a/b.mp4|s4|qa"
    v = V.encode("/a/b.mp4", V.VideoSpec("nfp", 0.25))
    assert v == "/a/b.mp4|s4|nfp:u=0.250000"
    assert V.parse(v) == ("/a/b.mp4", V.VideoSpec("nfp", 0.25)) and V.is_nfp(v) and not V.is_nfp("/a/b.mp4|s4|qa")
    assert V.parse("/a/b.mp4") == ("/a/b.mp4", None) and V.parse(["frame"])[1] is None
    for bad in ("/a|s4|xyz", "/a|s4|nfp:u=1.0"):
        with pytest.raises(ValueError):
            V.parse(bad)
    with pytest.raises(ValueError):
        V.encode(v, V.VideoSpec("qa"))
    with pytest.raises(ValueError):
        V.VideoSpec("nfp", -0.1)


def test_qa_indices_are_exactly_n_with_repeats():
    assert V.qa_indices(1000, 4) == [0, 333, 666, 999]
    short = V.qa_indices(10, 16)
    assert len(short) == 16 and short[0] == 0 and short[-1] == 9 and len(set(short)) == 10
    assert V.qa_indices(1, 3) == [0, 0, 0]


def test_nfp_window():
    # 300 frames at 30 fps = 10 s -> 20 grid frames at 2 fps (one every 15 source frames)
    assert V.nfp_indices(300, 30.0, 0.0, 2.0, 256) == [15 * k for k in range(20)]
    assert V.nfp_indices(300, 30.0, 0.5, 2.0, 8) == [15 * k for k in range(6, 14)]  # 13 starts, floor(6.5) = 6
    assert V.nfp_indices(300, 30.0, 0.999, 2.0, 8) == [15 * k for k in range(12, 20)]  # the last start
    assert V.nfp_indices(300, 30.0, 0.0, 2.0, 8) == [15 * k for k in range(8)]
    odd = V.nfp_indices(315, 30.0, 0.0, 2.0, 256)  # 10.5 s -> 21 grid frames -> 20 kept
    assert len(odd) == 20
    for u in np.linspace(0, 0.9999, 50):
        idx = V.nfp_indices(317, 29.97, float(u), 2.0, 10)
        assert len(idx) == 10 and idx == sorted(set(idx)) and idx[-1] <= 316
    with pytest.raises(ValueError):
        V.nfp_indices(45, 30.0, 0.0, 2.0, 256)  # 1.5 s -> 2 frames, under the 4-frame minimum


def test_decode_at(tmp_path):
    path = write_video(tmp_path / "v.mp4", 30)
    c = av.open(path)
    s = c.streams.video[0]
    info = V.stream_info(c, s)
    assert info["total_frames"] == 30 and info["average_fps"] == 10.0 and abs(info["duration"] - 3.0) < 0.15
    frames, missing = V.decode_at(c, s, [0, 7, 7, 29, 3])
    assert missing == 0 and [level_of(f) for f in frames] == [0, 7, 7, 29, 3]
    frames, missing = V.decode_at(c, s, [5, 40, 41])  # past the end: the last frame stands in
    assert missing == 2 and [level_of(f) for f in frames] == [5, 29, 29]
    c.close()


# ------------------------------------------------------------------------------------------------ plugin
@pytest.fixture(scope="module")
def proc():
    p = processor()
    p.video_fps, p.video_maxlen, p.video_max_pixels, p.video_min_pixels = 2.0, 8, 64 * 64, 16 * 16
    return p


@pytest.fixture(scope="module")
def plugins():
    from llamafactory.data.mm_plugin import get_mm_plugin

    from vagen.q38s4.lf_plugin import S4VideoPlugin, to_s4_plugin

    base = get_mm_plugin(name="qwen3_vl", image_token="<|image_pad|>", video_token="<|video_pad|>")
    s4 = to_s4_plugin(base)
    assert type(s4) is S4VideoPlugin and s4.video_token == base.video_token
    with pytest.raises(TypeError):
        to_s4_plugin(s4)
    return base, s4


@needs_models
@pytest.mark.parametrize("spec", [V.VideoSpec("qa"), V.VideoSpec("nfp", 0.0), V.VideoSpec("nfp", 0.7)])
def test_plugin_token_count_matches_decoding(tmp_path, proc, plugins, spec):
    _, s4 = plugins
    path = write_video(tmp_path / "v.mp4", 80)  # 8 s at 10 fps -> 16 grid frames at 2 fps
    video = V.encode(path, spec)
    meta = s4._get_qwen_video_grid_metadata([video], proc)
    mm = s4._get_mm_inputs([], [video], [], proc)
    assert meta["video_grid_thw"].tolist() == mm["video_grid_thw"].tolist()
    t = int(meta["video_grid_thw"][0, 0])
    assert t == 4  # 8 frames / 2 per unit
    reg = s4._regularize_videos([video], image_max_pixels=proc.video_max_pixels,
                                image_min_pixels=proc.video_min_pixels, video_fps=2.0, video_maxlen=8)
    idx = V.spec_indices(spec, 80, 10.0, 2.0, 8)
    assert [level_of(f) for f in reg["videos"][0]] == idx
    assert np.allclose(reg["frames_indices"][0], meta["frames_indices"][0])
    msgs = s4.process_messages([{"role": "user", "content": "<video>Q?"}], [], [video], [], proc)
    text = msgs[0]["content"]
    n_tok = int(meta["video_grid_thw"][0].prod()) // 4
    assert text.count("<|video_pad|>") == n_tok and text.count(" seconds>") == t
    if spec.kind == "nfp":
        first = float(text.split("<", 2)[1].split(" ")[0])
        # a unit is stamped with the mean time of its two frames, in the source video's time
        assert abs(first - (idx[0] + idx[1]) / 2 / 10.0) < 0.06


@needs_models
def test_plugin_leaves_unmarked_videos_to_lf(tmp_path, proc, plugins):
    base, s4 = plugins
    path = write_video(tmp_path / "v.mp4", 80)
    a = base._get_mm_inputs([], [path], [], proc)
    b = s4._get_mm_inputs([], [path], [], proc)
    assert a["video_grid_thw"].tolist() == b["video_grid_thw"].tolist()
    assert np.array_equal(a["pixel_values_videos"].numpy(), b["pixel_values_videos"].numpy())
    # mixed: one marked, one not
    mixed = s4._get_mm_inputs([], [V.encode(path, V.VideoSpec("qa")), path], [], proc)
    assert mixed["video_grid_thw"][1].tolist() == a["video_grid_thw"][0].tolist()
