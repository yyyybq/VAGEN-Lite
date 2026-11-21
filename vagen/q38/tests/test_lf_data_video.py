"""CPU tests of vagen.q38.sft.lf_data's video markers (a NEW DECISION of the q38 line: how a <video> placeholder of
the matrix cells' P0 video conversations is written for Qwen3.8; check_lf_qwen35 tokens / p0_eval show the layouts).

Run (login node): as test_sft_lf.py, with vagen/q38/tests/test_lf_data_video.py
"""
from __future__ import annotations

import json

import pytest
import yaml

from vagen.q38.sft import lf_data


def vrec():
    return {"messages": [{"role": "user", "content": "<video>\nHow many chairs?\nAnswer with a single number. "},
                         {"role": "assistant", "content": "4"},
                         {"role": "user", "content": "And tables?"}, {"role": "assistant", "content": "1\n"}],
            "images": [], "videos": [["/x/0.png", "/x/1.png"]]}


def irec():
    return {"messages": [{"role": "user", "content": " <image>\nWhat is it?\n"}, {"role": "assistant", "content": "A"}],
            "images": ["/x/a.png"]}


def test_a_video_record_needs_a_choice():
    with pytest.raises(ValueError, match="needs video_markers"):
        lf_data.strip_record(vrec(), "t")
    with pytest.raises(ValueError, match="needs video_markers"):
        lf_data.strip_record(vrec(), "t", "both")


def test_outer_and_none():
    out, changed = lf_data.strip_record(vrec(), "t", "outer")
    assert out["messages"][0]["content"] == "<|vision_start|><video><|vision_end|>\nHow many chairs?\nAnswer with a single number."
    assert out["messages"][2]["content"] == "And tables?" and out["messages"][3]["content"] == "1"
    assert changed["user/video_markers"] == 1 and out["videos"] == vrec()["videos"]
    out, changed = lf_data.strip_record(vrec(), "t", "none")
    assert out["messages"][0]["content"] == "<video>\nHow many chairs?\nAnswer with a single number."
    assert "user/video_markers" not in changed


@pytest.mark.parametrize("vm", [None, "outer", "none"])
def test_records_without_video_do_not_change(vm):
    assert lf_data.strip_record(irec(), "t", vm) == lf_data.strip_record(irec(), "t")


def make_src(tmp, recs):
    src = tmp / "src"
    src.mkdir()
    cfg = dict(lf_data.RECIPE_8B, model_name_or_path="/old/8b", dataset_dir=str(src), dataset="P0_train",
               template="qwen3_vl_nothink", output_dir=str(src / "model"), video_max_pixels=65536, report_to="none")
    (src / "train.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    cols = {"messages": "messages", "images": "images", "videos": "videos"}
    (src / "dataset_info.json").write_text(json.dumps({"P0_train": {"file_name": "P0_train.jsonl", "formatting": "sharegpt",
                                                                    "columns": cols}}))
    (src / "P0_train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    return src


def test_build_with_videos(tmp_path):
    src = make_src(tmp_path, [irec(), vrec()])
    with pytest.raises(ValueError, match="needs video_markers"):
        lf_data.build(src, tmp_path / "a", "/m/27b")
    with pytest.raises(ValueError, match="video_markers must be one of"):
        lf_data.build(src, tmp_path / "b", "/m/27b", video_markers="x")
    m = lf_data.build(src, tmp_path / "c", "/m/27b", video_markers="outer")
    got = [json.loads(l) for l in (tmp_path / "c" / "P0_train.jsonl").read_text().splitlines()]
    assert m["video_markers"] == "outer" and got[0]["messages"][0]["content"] == "<image>\nWhat is it?"
    assert got[1]["messages"][0]["content"].startswith("<|vision_start|><video><|vision_end|>\nHow many")
    assert m["datasets"]["P0_train"]["messages_stripped"]["user/video_markers"] == 1


def test_build_without_videos_needs_no_choice(tmp_path):
    src = make_src(tmp_path, [irec()])
    m = lf_data.build(src, tmp_path / "o", "/m/27b")
    assert m["video_markers"] is None
