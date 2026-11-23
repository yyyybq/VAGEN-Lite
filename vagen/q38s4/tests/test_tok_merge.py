"""CPU tests of vagen.q38s4.tools.tok_merge (tokenized parts, one per dataset, joined in the training's order)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from datasets import Dataset, DatasetDict, load_from_disk

from vagen.q38s4.tools import tok_merge as T


def lf_rows(n, medium, start=0):
    """Rows shaped like LF's supervised tokenize output; the medium not used is None (typed null when all None)."""
    return [{"input_ids": [1, 2, 300 + i], "attention_mask": [1, 1, 1], "labels": [-100, 2, 300 + i],
             "images": [f"img{i}.jpg"] if medium == "image" else None,
             "videos": [f"vid{i}.mp4"] if medium == "video" else None,
             "audios": None} for i in range(start, start + n)]


def save(path: Path, rows, splits=("train",)) -> Path:
    DatasetDict({s: Dataset.from_list(rows) for s in splits}).save_to_disk(str(path))
    return path


def three_parts(tmp_path):
    qa, img, nfp = lf_rows(5, "video", 0), lf_rows(4, "image", 100), lf_rows(3, "video", 200)
    parts = [save(tmp_path / "qa", qa), save(tmp_path / "img", img), save(tmp_path / "nfp", nfp)]
    return parts, qa + img + nfp


def test_parts_type_the_unused_medium_null(tmp_path):
    parts, _ = three_parts(tmp_path)
    assert T.types(T.load_part(parts[0]))["images"] == "null"
    assert T.types(T.load_part(parts[1]))["videos"] == "null"


def test_merge_order_types_manifest(tmp_path):
    parts, rows = three_parts(tmp_path)
    man = T.merge(parts, tmp_path / "all")
    assert man["ok"] and man["rows"] == 12
    assert not (tmp_path / ("all" + T.PARTIAL)).exists()
    d = load_from_disk(str(tmp_path / "all"))
    assert isinstance(d, DatasetDict) and list(d) == ["train"]
    assert [d["train"][i] for i in range(12)] == rows
    t = T.types(d["train"])
    assert t["images"] == "list<item: string>" and t["videos"] == "list<item: string>"
    m = json.loads((tmp_path / "all" / T.MANIFEST).read_text())
    assert [p["rows"] for p in m["parts"]] == [5, 4, 3]
    assert m["order"] == [str(p.resolve()) for p in parts]
    assert m["check_parts"]["rows_ok"] and m["check_parts"]["n_differ"] == 0
    assert "check_ref" not in m


def test_order_follows_parts(tmp_path):
    parts, _ = three_parts(tmp_path)
    T.merge([parts[1], parts[0], parts[2]], tmp_path / "all")
    d = load_from_disk(str(tmp_path / "all"))["train"]
    assert d[0]["images"] == ["img100.jpg"] and d[3]["images"] == ["img103.jpg"] and d[3]["videos"] is None
    assert d[4]["videos"] == ["vid0.mp4"] and d[4]["images"] is None and d[11]["videos"] == ["vid202.mp4"]


def test_check_ref_equal(tmp_path):
    parts, rows = three_parts(tmp_path)
    ref = save(tmp_path / "ref", rows)  # one dataset of the same rows (what a one-pass tokenize makes)
    man = T.merge(parts, tmp_path / "all", check=ref)
    assert man["ok"] and man["check_ref"]["ok"]
    assert man["check_ref"]["types_ok"] and man["check_ref"]["n_differ"] == 0


@pytest.mark.parametrize("change", ["row", "order", "missing"])
def test_check_ref_detects(tmp_path, change):
    parts, rows = three_parts(tmp_path)
    bad = [dict(r) for r in rows]
    if change == "row":
        bad[7]["input_ids"] = [1, 2, 999]
    elif change == "order":
        bad[4], bad[5] = bad[5], bad[4]
    else:
        bad = bad[:-1]
    man = T.merge(parts, tmp_path / "all", check=save(tmp_path / "ref", bad))
    assert not man["ok"] and not man["check_ref"]["ok"]
    if change == "row":
        assert man["check_ref"]["rows_differ"] == [7]
    if change == "order":
        assert man["check_ref"]["rows_differ"] == [4, 5]
    assert not (tmp_path / "all").exists() and (tmp_path / ("all" + T.PARTIAL) / T.MANIFEST).exists()


def test_check_ref_types(tmp_path):
    parts, rows = three_parts(tmp_path)
    ref = [dict(r, labels=[float(x) for x in r["labels"]]) for r in rows]  # same values, other type
    man = T.merge(parts, tmp_path / "all", check=save(tmp_path / "ref", ref))
    assert not man["check_ref"]["types_ok"] and not man["ok"]


def test_check_parts_detects(tmp_path, monkeypatch):
    parts, _ = three_parts(tmp_path)
    real = T.load_part

    def fake(path):  # the read-back of TOK.partial returns a changed row
        d = real(path)
        if str(path).endswith(T.PARTIAL):
            d = d.map(lambda r, i: dict(r, labels=[0, 0, 0]) if i == 11 else r, with_indices=True)
        return d

    monkeypatch.setattr(T, "load_part", fake)
    man = T.merge(parts, tmp_path / "all")
    assert not man["ok"] and 11 in man["check_parts"]["rows_differ"]
    assert not (tmp_path / "all").exists()


def test_sample_index():
    idx = T.sample_index([5, 0, 4, 3], 3, seed=1)
    assert {0, 4, 5, 8, 9, 11} <= set(idx) and all(0 <= i < 12 for i in idx) and idx == sorted(set(idx))
    assert T.sample_index([0, 0], 5, 0) == []


def test_empty_part_refused(tmp_path):
    parts, rows = three_parts(tmp_path)
    empty = tmp_path / "empty"
    DatasetDict({"train": Dataset.from_list(rows[:1]).select([])}).save_to_disk(str(empty))
    with pytest.raises(SystemExit, match="no rows"):  # datasets cannot even read an empty save back
        T.merge([parts[0], empty, parts[1], parts[2]], tmp_path / "all")
    assert not (tmp_path / "all").exists() and not (tmp_path / ("all" + T.PARTIAL)).exists()


@pytest.mark.parametrize("case", ["out_exists", "partial_exists", "one_part", "twice", "columns", "splits",
                                  "not_dict"])
def test_refusals(tmp_path, case):
    parts, rows = three_parts(tmp_path)
    out = tmp_path / "all"
    if case == "out_exists":
        out.mkdir()
    if case == "partial_exists":
        (tmp_path / ("all" + T.PARTIAL)).mkdir()
    if case == "one_part":
        parts = parts[:1]
    if case == "twice":
        parts = [parts[0], parts[0]]
    if case == "columns":
        parts[2] = save(tmp_path / "other", [{k: v for k, v in r.items() if k != "audios"} for r in rows[:2]])
    if case == "splits":
        parts[2] = save(tmp_path / "two", rows[:2], splits=("train", "validation"))
    if case == "not_dict":
        Dataset.from_list(rows[:2]).save_to_disk(str(tmp_path / "plain"))
        parts[2] = tmp_path / "plain"
    with pytest.raises(SystemExit):
        T.merge(parts, out)
    if case not in ("out_exists", "partial_exists"):
        assert not out.exists() and not (tmp_path / ("all" + T.PARTIAL)).exists()


def test_main_exit_codes(tmp_path, capsys):
    parts, rows = three_parts(tmp_path)
    with pytest.raises(SystemExit) as e:
        T.main(["--parts", *map(str, parts), "--out", str(tmp_path / "a"), "--check",
                str(save(tmp_path / "ref", rows))])
    assert e.value.code == 0 and json.loads(capsys.readouterr().out)["ok"]
    with pytest.raises(SystemExit) as e:
        T.main(["--parts", *map(str, parts), "--out", str(tmp_path / "b"), "--check",
                str(save(tmp_path / "ref2", rows[::-1]))])
    assert e.value.code == 1
