"""CPU tests of the video side of vagen.q38.sft.check_lf_qwen35 (P2_PREP_REPORT: the per-token video check was not
ported; C0 has no video, the D / C cells' P0 has 240 video conversations of 32 frames): the metadata is the P0
evaluation's, the time texts, the block layout, and (when the model folder and llamafactory are readable) a P0-style
video conversation through LLaMA-Factory against Qwen3.8's processor: tokens, pixels (also from the evaluation's uint8
frames), M-RoPE positions, LF's own loader.

Run (login node; the pipeline test runs only with HF_DATASETS_CACHE set, so the datasets cache stays in a folder of
yours):
  cd /mnt/umm/users/yinbaiqiao/VAGEN-Lite-q38 && DISABLE_VERSION_CHECK=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false RAYON_NUM_THREADS=1 \
    PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES= HF_DATASETS_CACHE=DIR PYTHONPATH=$PWD nice -n 19 \
    /mnt/umm/users/yinbaiqiao/q38/envs/verl/bin/python -B -m pytest vagen/q38/tests/test_sft_lf_video.py -q -p no:cacheprovider
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from vagen.q38.sft import check_lf_qwen35 as ck
from vagen.q38.sft import lf_data

MODEL = Path("/mnt/umm/users/yinbaiqiao/q38/models/Qwen3.8-27B")
needs_model = pytest.mark.skipif(not (MODEL / "tokenizer.json").is_file(), reason="Qwen3.8-27B not readable")
OUTER = ("<|vision_start|>", "<|vision_end|>")


def frames(tmp: Path, n: int, size: int = 256):
    from PIL import Image

    out = []
    for i in range(n):
        p = tmp / f"f{i:02d}.png"
        Image.new("RGB", (size, size), ((37 * i) % 256, (11 * i + 90) % 256, 160)).save(p)
        out.append(str(p))
    return out


def video_rec(paths, wrap: bool = False):
    """A P0 video conversation as export_lf.py writes it (the video in the first user turn, one question a turn)."""
    v = OUTER[0] + "<video>" + OUTER[1] if wrap else "<video>"
    return {"messages": [{"role": "user", "content": v + "\nHow many chairs are in the room?\nAnswer with a single number. "},
                         {"role": "assistant", "content": "4"},
                         {"role": "user", "content": "Estimate the floor area of the room in the video (m^2).\nAnswer with a single number."},
                         {"role": "assistant", "content": "21.5\n"}],
            "images": [], "videos": [paths]}


def stripped(rec):
    """lf_data's strip rule (every message content str.strip()), the placeholders left as written."""
    return {**rec, "messages": [{**m, "content": m["content"].strip()} for m in rec["messages"]]}


# ------------------------------------------------------------------------------------------------------ no model
def test_metadata_is_the_evaluations():
    from data_gen.interiorgs_qa.evaluate import video_input

    fr = list(range(32))
    got, meta = video_input(fr, "qwen3_5", 2.0)
    meta = dict(meta)
    assert got is fr and meta.pop("do_sample_frames") is False
    assert ck.video_metadata(32, 2.0) == meta


def test_expected_seconds():
    assert ck.expected_seconds(4, 2.0) == ["0.2", "1.2"]
    assert ck.expected_seconds(3, 2.0) == ["0.2", "1.0"]  # odd: the last index padded
    s = ck.expected_seconds(32, 2.0)
    assert len(s) == 16 and s[0] == "0.2" and s[1] == "1.2" and s[-1] == "15.2"


def test_outer_video_markers():
    blk = lambda t: f"<{t} seconds><|vision_start|>" + "<|video_pad|>" * 3 + "<|vision_end|>"
    inner = blk("0.2") + blk("1.2")
    assert ck.outer_video_markers("<|im_start|>user\n" + inner + "\nq") == [False]
    assert ck.outer_video_markers("<|im_start|>user\n" + OUTER[0] + inner + OUTER[1] + "\nq") == [True]
    assert ck.outer_video_markers(inner + "x" + OUTER[0] + inner + OUTER[1]) == [False, True]
    assert ck.outer_video_markers("<|vision_start|><|image_pad|><|vision_end|>") == []


def test_to_blocks_av_layout(tmp_path):
    paths = frames(tmp_path, 4, 32)
    rec = video_rec(paths)
    out, images, videos, meta = ck.to_blocks_av(rec["messages"], [], [paths], 2.0)
    assert out[0]["content"][0] == {"type": "video"} and out[0]["content"][1]["text"].startswith("\nHow many")
    assert out[2]["content"] == [{"type": "text", "text": rec["messages"][2]["content"]}]
    assert images == [] and [len(v) for v in videos] == [4] and meta == [ck.video_metadata(4, 2.0)]
    with pytest.raises(ValueError, match="without its video"):
        ck.to_blocks(rec["messages"], [])


def test_media_args_and_kind(tmp_path):
    (tmp_path / "train.yaml").write_text(yaml.safe_dump({"image_max_pixels": 65536, "video_max_pixels": 65536,
                                                         "learning_rate": 1e-5}))
    assert ck.lf_media_args(tmp_path) == {"image_max_pixels": 65536, "video_max_pixels": 65536}
    assert ck.lf_media_args(tmp_path / "missing") == {}
    assert [ck.media_kind(r) for r in ({"videos": [["a"]], "images": ["b"]}, {"images": ["b"]}, {})] == ["video", "image", "text"]


# ------------------------------------------------------------------------------------------------------ with the model
@pytest.fixture(scope="module")
def enc():
    pytest.importorskip("llamafactory")
    os.environ.setdefault("DISABLE_VERSION_CHECK", "1")
    # the 8B matrix recipes' media arguments (VAGEN-Lite-data exps/sft_mixture_matrix_v3_qwen3_20261102/*/train.yaml)
    return ck.LFEncoder(str(MODEL), media_args={"image_max_pixels": 65536, "video_max_pixels": 65536})


@needs_model
def test_lf_video_differs_from_the_processor_by_the_outer_markers_only(tmp_path, enc):
    """LF main's qwen3_vl plugin writes the per-frame blocks in place of <video>; transformers 5.10's Qwen3-VL processor
    (and vLLM 0.28, which follows it) expands only the bare <|video_pad|> and keeps the template's
    <|vision_start|> ... <|vision_end|> around them.  Everything else (blocks, times, labels, think block, pixels) agrees."""
    src = video_rec(frames(tmp_path, 32))
    conv = stripped(src)
    r = ck.check_record(enc, conv, src, prefix=True)
    out, hf_ids = r.pop("_hf")
    ids, _ = enc.encode(conv)
    vs, ve = (enc.tokenizer.convert_tokens_to_ids(t) for t in OUTER)
    assert r["outer_video_markers_lf"] == [False] and r["outer_video_markers_processor"] == [True]
    assert r["ids_identical"] is False and len(hf_ids) == len(ids) + 2
    first = hf_ids.index(vs)  # the outer marker comes before the first time text
    last = len(hf_ids) - 1 - hf_ids[::-1].index(ve)
    assert hf_ids[:first] + hf_ids[first + 1:last] + hf_ids[last + 1:] == ids
    assert r["video_counts_ok"] and r["seconds_ok"] and r["video_tokens_lf"] == 1024 and r["video_blocks_lf"] == 16
    assert r["video_grid_processor"] == [[16, 16, 16]]
    assert r["supervised_text_is_answers"] and r["spans_are_answers"] and r["think_block"] and r["supervised_spans"] == 2
    px = ck.check_pixels(enc, conv, out)
    assert px["video_pixels_identical"] and px["video_grid_identical"] and px["video_pixels_eval_identical"]
    assert px["pixels_ok"] and "pixels_identical" not in px  # no images in the record


@needs_model
@pytest.mark.parametrize("n", [32, 5])
def test_wrapped_placeholder_matches_the_processor(tmp_path, enc, n):
    """The record with "<|vision_start|><video><|vision_end|>" (the template's markers around LF's placeholder): LF's
    ids are the processor's one by one, and so are the M-RoPE positions; odd frame counts too (both pad the last frame)."""
    paths = frames(tmp_path, n)
    src = video_rec(list(paths))
    conv = stripped(video_rec(list(paths), wrap=True))
    r = ck.check_record(enc, conv, src, prefix=True)
    out, _ = r.pop("_hf")
    assert r["ok"], json.dumps({k: v for k, v in r.items() if k not in ("seconds_lf",)}, default=str)[:3000]
    assert conv["videos"] == [paths] and src["videos"] == [paths]  # the check leaves the records as they were
    assert r["ids_identical"] and r["generation_prefix"] and r["outer_video_markers_lf"] == [True]
    assert r["seconds_lf"] == ck.expected_seconds(n, 2.0)
    px = ck.check_pixels(enc, conv, out)
    assert px["pixels_ok"], px
    ids, labels = enc.encode(conv)
    pos = ck.check_positions(enc, str(MODEL), ids, labels, conv, out)
    assert pos["positions_identical"], pos


@needs_model
@pytest.mark.skipif(not os.environ.get("HF_DATASETS_CACHE"), reason="set HF_DATASETS_CACHE to a folder of yours")
def test_pipeline_keeps_videos(tmp_path, enc):
    """LF's own loader (get_dataset on a folder's train.yaml) gives the direct encoding and keeps the frame lists."""
    d = tmp_path / "lf"
    d.mkdir()
    paths = frames(tmp_path, 4)
    (tmp_path / "img").mkdir()
    img = frames(tmp_path / "img", 1)[0]
    recs = [stripped(video_rec(paths)),
            {"messages": [{"role": "user", "content": "<image>\nWhat is it?"}, {"role": "assistant", "content": "A"}],
             "images": [img], "videos": []}]
    (d / "vid.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    # the matrix cells' dataset_info entry (VAGEN-Lite-data exps/sft_mixture_matrix_v3_qwen3_20261102/D)
    cols = {"messages": "messages", "images": "images", "videos": "videos"}
    tags = {"role_tag": "role", "content_tag": "content", "user_tag": "user", "assistant_tag": "assistant",
            "system_tag": "system"}
    (d / "dataset_info.json").write_text(json.dumps({"vid": {"file_name": "vid.jsonl", "formatting": "sharegpt",
                                                             "columns": cols, "tags": tags}}))
    cfg = dict(lf_data.RECIPE_8B, model_name_or_path=str(MODEL), template="qwen3_8", enable_thinking=False,
               dataset_dir=str(d), dataset="vid", output_dir=str(d / "model"), image_max_pixels=65536,
               video_max_pixels=65536, report_to="none", do_eval=False)
    (d / "train.yaml").write_text(yaml.safe_dump(cfg))
    got = ck.mode_pipeline(SimpleNamespace(lf_dir=str(d), model=str(MODEL), n=2, scratch=str(tmp_path / "scratch")))
    s = got["splits"]["train_dataset"]
    assert got["ok"], got
    assert s["rows"] == 2 and s["identical_to_direct_encoding"] == 2 and s["media_identical"] == 2 and s["rows_with_video"] == 1
