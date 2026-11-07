"""reward.build_und_src on synthetic inputs (temp folders only)."""

import json

import pytest

from reward import build_und_src as b


def _inputs(tmp_path):
    p1 = tmp_path / "p1.jsonl"
    p1.write_text("".join(json.dumps({"messages": [{"role": "user", "content": "<image>q"}, {"role": "assistant", "content": "A"}],
                                      "images": [f"/i/{i}.png"], "sample_id": f"p1_{i}", "task_id": f"t{i}", "scene": f"s{i}",
                                      "qtype": "goal_now", "qclass": "state", "family": "fov_inclusion", "split": "train"}) + "\n"
                          for i in range(3)))
    p0 = tmp_path / "p0.jsonl"
    p0.write_text(json.dumps({"messages": [], "images": ["/a.png"], "videos": [], "source_id": "p0_1"}) + "\n"
                  + json.dumps({"messages": [], "images": [], "videos": ["/v.mp4"], "source_id": "p0_2"}) + "\n")
    info = tmp_path / "dataset_info.json"
    cols = {"formatting": "sharegpt", "columns": {"messages": "messages", "images": "images", "videos": "videos"}}
    info.write_text(json.dumps({"P0_train": dict(cols, file_name="P0_train.jsonl"), "P1_train": dict(cols, file_name="x.jsonl"),
                                "other": {}}))
    recipe = tmp_path / "train.yaml"
    recipe.write_text("template: qwen3_vl_nothink\ncutoff_len: 16384\n")
    return p1, p0, info, recipe


def test_build(tmp_path):
    p1, p0, info, recipe = _inputs(tmp_path)
    out = tmp_path / "src"
    src = b.build(p1, p0, info, recipe, out)
    assert src["p1"]["rows"] == 3 and src["p0"]["rows_kept"] == 1 and src["p0"]["video_rows_dropped"] == 1
    r = json.loads((out / "P1_train.jsonl").read_text().splitlines()[0])
    assert r["source_id"] == "p1_0" and r["videos"] == [] and r["pool"] == "P1" and r["stratum"] == "goal_now"
    assert json.loads(r["meta"]) == {"qtype": "goal_now", "qclass": "state", "family": "fov_inclusion", "task_id": "t0",
                                     "questions": 1}
    d = json.loads((out / "dataset_info.json").read_text())
    assert set(d) == {"P0_train", "P1_train"} and d["P1_train"]["file_name"] == "P1_train.jsonl"
    assert (out / "train.yaml").read_text() == recipe.read_text()
    with pytest.raises(SystemExit):
        b.build(p1, p0, info, recipe, out)
