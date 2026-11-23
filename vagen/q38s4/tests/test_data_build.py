"""CPU tests of the VSI-590K -> LLaMA-Factory build (vagen.q38s4.data_build) on a small synthetic source."""
from __future__ import annotations

import json
from collections import Counter

import pytest

from vagen.q38s4 import data_build as D
from vagen.q38s4 import lf_video as V
from vagen.q38s4.tests.test_lf_video import write_video  # imports LLaMA-Factory before PyAV

Q = "<image>\nThese are frames of a video.\nHow many chairs?\nOptions:\nA. 1\nB. 2\nAnswer with the letter. "


def src_rows():
    rows = []
    for i in range(40):
        v = f"scannet/scene{i % 10:02d}.mp4"
        rows.append({"conversations": [{"from": "human", "value": Q}, {"from": "gpt", "value": " A "}],
                     "video": v, "question_type": "absolute_count"})
    rows.append({"conversations": [{"from": "human", "value": Q}, {"from": "gpt", "value": "B"},
                                   {"from": "human", "value": "And tables?"}, {"from": "gpt", "value": "A"}],
                 "image": "hypersim/a/frame.0001.jpg", "question_type": "absolute_count"})
    rows.append({"conversations": [{"from": "human", "value": Q}, {"from": "gpt", "value": "B"}],
                 "video": "procthor/short.mp4", "question_type": "absolute_count", "id": 7})
    return rows


def write_src(tmp_path, rows):
    p = tmp_path / "vsi.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return p


def args(tmp_path, src, **kw):
    a = ["--src", str(src), "--media-root", str(tmp_path / "media"), "--out", str(tmp_path / "out"),
         "--nfp-rows", "20", "--heldout-frac", "0.2"]
    for k, v in kw.items():
        a += [f"--{k.replace('_', '-')}"] + ([] if v is True else [str(v)])
    return D.parse_args(a)


def read(path):
    return [json.loads(x) for x in open(path)]


def test_convert_row_video_and_image():
    r = D.convert_row(src_rows()[0], "/m", "outer")
    assert r["videos"] == ["/m/scannet/scene00.mp4|s4|qa"] and "images" not in r
    assert r["messages"][0] == {"role": "user", "content": Q.replace("<image>", D.VIDEO_OUTER).strip()}
    assert r["messages"][1] == {"role": "assistant", "content": "A"}
    assert D.convert_row(src_rows()[0], "/m", "none")["messages"][0]["content"].startswith("<video>\nThese")
    img = D.convert_row(src_rows()[40], "/m", "outer")
    assert img["images"] == ["/m/hypersim/a/frame.0001.jpg"] and len(img["messages"]) == 4
    assert img["messages"][0]["content"].startswith("<image>\n") and img["messages"][2]["content"] == "And tables?"


@pytest.mark.parametrize("bad", [
    {"conversations": [{"from": "human", "value": "no placeholder"}, {"from": "gpt", "value": "A"}], "video": "a.mp4"},
    {"conversations": [{"from": "human", "value": Q}, {"from": "gpt", "value": "<image>"}], "video": "a.mp4"},
    {"conversations": [{"from": "gpt", "value": Q}, {"from": "human", "value": "A"}], "video": "a.mp4"},
    {"conversations": [{"from": "human", "value": Q}], "video": "a.mp4"},
    {"conversations": [{"from": "human", "value": Q}, {"from": "gpt", "value": "A"}], "video": "a.mp4", "image": "b"},
    {"conversations": [{"from": "human", "value": Q}, {"from": "gpt", "value": "A"}], "video": "../a.mp4"},
    {"conversations": [{"from": "human", "value": Q}, {"from": "gpt", "value": "  "}], "video": "a.mp4"},
])
def test_bad_rows_stop(bad):
    with pytest.raises(D.BadRow):
        D.convert_row(bad, "/m", "outer")


def test_build_without_probe(tmp_path):
    src = write_src(tmp_path, src_rows())
    m = D.build(args(tmp_path, src, seed=3))
    out = tmp_path / "out"
    qa, nfp, held = read(out / "qa.jsonl"), read(out / "nfp.jsonl"), read(out / "nfp_heldout.jsonl")
    img = read(out / "qa_img.jsonl")
    assert len(qa) == 41 and len(img) == 1 and not (out / "qa_heldout.jsonl").exists()
    assert all(set(r) == {"messages", "videos"} for r in qa) and set(img[0]) == {"messages", "images"}
    media = str(tmp_path / "media")
    assert qa == [D.convert_row(r, media, "outer") for r in src_rows() if "video" in r]  # source order
    assert img == [D.convert_row(src_rows()[40], media, "outer")]
    heldout = (out / "heldout_videos.txt").read_text().split()
    assert len(heldout) == 2 and m["distinct_videos"] == 11 and m["heldout_videos"] == 2  # round(0.2 * 11)
    assert len(nfp) == 20 and len(held) == 2 * 4
    for r in nfp:
        path, spec = V.parse(r["videos"][0])
        assert spec.kind == "nfp" and path[len(str(tmp_path / "media")) + 1:] not in heldout
        assert r["messages"] == [{"role": "user", "content": D.VIDEO_OUTER}, {"role": "assistant", "content": "."}]
    assert sorted({V.parse(r["videos"][0])[1].u for r in held}) == [0.125, 0.375, 0.625, 0.875]
    assert m["rows_out"] == {"qa": 41, "qa_img": 1, "qa_heldout": 0, "nfp": 20, "nfp_heldout": 8}
    assert m["outputs"]["nfp.jsonl"]["lines"] == 20 and m["source"]["rows"] == 42
    assert m["counts"]["video/scannet"] == 40 and m["counts"]["image/hypersim"] == 1 and m["counts"]["turns/4"] == 1
    info = json.loads((out / "dataset_info.json").read_text())
    assert set(info) == {"s4_qa", "s4_qa_img", "s4_nfp", "s4_nfp_heldout"}
    assert info["s4_qa"]["columns"] == {"messages": "messages", "videos": "videos"}
    assert info["s4_qa_img"]["columns"] == {"messages": "messages", "images": "images"}
    assert info["s4_qa_img"]["file_name"] == "qa_img.jsonl"
    # same seed -> same output; another seed -> another draw
    D.build(args(tmp_path, src, seed=3, out=tmp_path / "out2"))
    assert (out / "nfp.jsonl").read_text() == (tmp_path / "out2" / "nfp.jsonl").read_text()
    D.build(args(tmp_path, src, seed=4, out=tmp_path / "out3"))
    assert (out / "nfp.jsonl").read_text() != (tmp_path / "out3" / "nfp.jsonl").read_text()
    with pytest.raises(SystemExit):  # never overwrites
        D.build(args(tmp_path, src, seed=3))


def test_heldout_scope_all_and_too_many_rows(tmp_path):
    src = write_src(tmp_path, src_rows())
    m = D.build(args(tmp_path, src, heldout_scope="all"))
    out = tmp_path / "out"
    held = set((out / "heldout_videos.txt").read_text().split())
    qa_h = read(out / "qa_heldout.jsonl")
    assert m["rows_out"]["qa"] + m["rows_out"]["qa_img"] + m["rows_out"]["qa_heldout"] == 42 and qa_h
    assert m["rows_out"]["qa_img"] == 1 and all("images" not in r for r in qa_h)
    assert all(V.parse(r["videos"][0])[0].split("/media/")[1] in held for r in qa_h)
    info = json.loads((out / "dataset_info.json").read_text())
    assert info["s4_qa_heldout"]["columns"] == {"messages": "messages", "videos": "videos"}
    with pytest.raises(SystemExit):
        D.build(args(tmp_path, src, nfp_rows=1000, out=tmp_path / "o2"))


def test_build_with_probe(tmp_path):
    media = tmp_path / "media"
    (media / "scannet").mkdir(parents=True)
    (media / "procthor").mkdir()
    for i in range(10):
        write_video(media / "scannet" / f"scene{i:02d}.mp4", 40)  # 4 s at 10 fps -> 8 frames at 2 fps
    write_video(media / "procthor" / "short.mp4", 15)  # 1.5 s -> 3 frames: no NFP window
    src = write_src(tmp_path, src_rows())
    m = D.build(args(tmp_path, src, probe=True, heldout_frac=0.0, nfp_rows=40, video_maxlen=8))
    assert m["probe"]["too_short_for_nfp"] == ["procthor/short.mp4"] and m["nfp_pool_rows"] == 40
    nfp = read(tmp_path / "out" / "nfp.jsonl")
    assert len(nfp) == 40 and not any("short.mp4" in r["videos"][0] for r in nfp)
    assert m["rows_out"]["qa"] == 41 and m["rows_out"]["qa_img"] == 1  # the short video's QA row stays
    (media / "scannet" / "scene03.mp4").write_bytes(b"not a video")
    with pytest.raises(SystemExit):
        D.build(args(tmp_path, src, probe=True, out=tmp_path / "o2"))


def test_rows_load_in_llamafactory(tmp_path):
    from llamafactory.data.converter import get_dataset_converter
    from llamafactory.data.parser import get_dataset_list
    from llamafactory.hparams import DataArguments

    src = write_src(tmp_path, src_rows())
    D.build(args(tmp_path, src))
    out = tmp_path / "out"
    data_args = DataArguments(dataset_dir=str(out), media_dir=str(out))
    names = ["s4_qa", "s4_qa_img", "s4_nfp", "s4_nfp_heldout"]
    attrs = dict(zip(names, get_dataset_list(names, str(out))))
    assert all(a.load_from == "file" and a.formatting == "sharegpt" for a in attrs.values())
    assert attrs["s4_qa_img"].images == "images" and attrs["s4_qa_img"].videos is None
    assert attrs["s4_qa"].videos == "videos" and attrs["s4_qa"].images is None
    kinds = Counter()
    for name, fn in (("s4_qa", "qa.jsonl"), ("s4_qa_img", "qa_img.jsonl"), ("s4_nfp", "nfp.jsonl"),
                     ("s4_nfp_heldout", "nfp_heldout.jsonl")):
        conv = get_dataset_converter("sharegpt", attrs[name], data_args)
        for row in read(out / fn):
            got = conv(row)  # each file has its own medium's column only
            assert got["_prompt"] and got["_response"], row
            assert got["_videos"] == row.get("videos") and got["_images"] == row.get("images")
            kinds[(name, "video" if got["_videos"] else "image")] += 1
    assert kinds == {("s4_qa", "video"): 41, ("s4_qa_img", "image"): 1, ("s4_nfp", "video"): 20,
                     ("s4_nfp_heldout", "video"): 8}


def test_large_files_load_and_merge_in_llamafactory(tmp_path):
    """The v1 failure: more than one JSON block (about 10 MB) of video rows before the first image row.  With one
    medium per file LF's loader (datasets' JSON builder) reads every file, aligns and concatenates them in launch's
    order; the same rows in one file (the v1 layout) do not load."""
    from types import SimpleNamespace

    from llamafactory.data.loader import _get_merged_dataset
    from llamafactory.hparams import DataArguments

    from vagen.q38s4 import launch as L

    pad = " Look at the room." * 150  # about 2.7 KB per row
    rows = []
    for i in range(5000):  # about 13 MB of video rows first, as in VSI-590K (image rows from line 305,000 on)
        rows.append({"conversations": [{"from": "human", "value": Q + pad}, {"from": "gpt", "value": "A"}],
                     "video": f"scannet/scene{i % 50:02d}.mp4"})
    rows += src_rows()  # 41 video rows then the image row, then one more video row
    rows += [{"conversations": [{"from": "human", "value": Q}, {"from": "gpt", "value": "B"}],
              "image": f"hypersim/b/frame.{k:04d}.jpg"} for k in range(3)]
    src = write_src(tmp_path, rows)
    D.build(args(tmp_path, src, nfp_rows=100))
    out = tmp_path / "out"
    assert (out / "qa.jsonl").stat().st_size > 12 << 20

    def merged(names):
        data_args = DataArguments(dataset_dir=str(out), media_dir=str(out))
        model_args = SimpleNamespace(cache_dir=str(tmp_path / "hf_cache"), hf_hub_token=None)
        training_args = SimpleNamespace(local_process_index=0, seed=42, dataloader_num_workers=0)
        return _get_merged_dataset(names, model_args, data_args, training_args, stage="sft")

    names = L.train_datasets(out).split(",")
    assert names == ["s4_qa", "s4_qa_img", "s4_nfp"]
    ds = merged(names)
    assert len(ds) == 5041 + 4 + 100
    n_img = sum(1 for x in ds["_images"] if x)
    n_vid = sum(1 for x in ds["_videos"] if x)
    assert n_img == 4 and n_vid == 5041 + 100 and all(bool(a) != bool(b) for a, b in zip(ds["_images"], ds["_videos"]))
    assert ds[5041]["_images"] == [str(tmp_path / "media" / "hypersim/a/frame.0001.jpg")]

    # the v1 layout: the same QA rows in one file with both columns -> the loader fails on the first image row
    with open(out / "qa_v1.jsonl", "w", encoding="utf-8") as f:
        for name in ("qa.jsonl", "qa_img.jsonl"):
            f.write((out / name).read_text(encoding="utf-8"))
    info = json.loads((out / "dataset_info.json").read_text())
    info["s4_qa_v1"] = D.dataset_entry("qa_v1.jsonl", "videos")
    info["s4_qa_v1"]["columns"]["images"] = "images"
    (out / "dataset_info.json").write_text(json.dumps(info))
    with pytest.raises(Exception) as err:  # DatasetGenerationError, raised from the CastError on the next block
        merged(["s4_qa_v1"])
    chain, e = [], err.value
    while e is not None and len(chain) < 10:
        chain.append(f"{type(e).__name__}: {e}")
        e = e.__cause__ or e.__context__
    assert any("cast" in c.lower() for c in chain), chain
