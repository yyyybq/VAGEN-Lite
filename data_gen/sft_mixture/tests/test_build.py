"""Tests of the mixture builder on small synthetic pools (python -m pytest data_gen/sft_mixture/tests -q)."""

from __future__ import annotations

import collections
import json
from pathlib import Path

import pytest
import yaml

from data_gen.sft_mixture import build as B


def _qa(text: str, answer: str, images=None, videos=None):
    return {"messages": [{"role": "user", "content": "<image>\n" + text}, {"role": "assistant", "content": answer}],
            "images": images or ["/x/a.png"], "videos": videos or []}


def _setup(tmp: Path, p2_val_scene: str = "s9"):
    (tmp / "p0").mkdir()
    p0 = [_qa(f"q{i}", "A") for i in range(6)]
    audit = [{"kind": "image", "qids": [f"id{i}"], "groups": ["S7" if i == 0 else f"S{1 + i % 3}"],
              "image_inputs": 1, "scene_id": "s1"} for i in range(6)]
    B.write_jsonl(tmp / "p0" / "qa_train.jsonl", p0)
    B.write_jsonl(tmp / "p0" / "audit_train.jsonl", audit)
    B.write_jsonl(tmp / "p0" / "qa_val.jsonl", [_qa("v", "B")])
    B.write_jsonl(tmp / "p0" / "audit_val.jsonl", [{"kind": "image", "qids": ["v0"], "groups": ["S1"],
                                                    "image_inputs": 1, "scene_id": "s1"}])
    p1 = [dict(_qa(f"t{i}", "C"), sample_id=f"p1_{i:04d}", scene="s1" if i % 2 else "s2",
               qtype=("goal_now", "closer_now")[i % 2], qclass="state", family="equidistance", task_id=f"v10_{i:04x}")
          for i in range(40)]
    B.write_jsonl(tmp / "p1_train.jsonl", p1)
    B.write_jsonl(tmp / "p1_test.jsonl", [dict(r, sample_id=r["sample_id"] + "v", scene="s9") for r in p1[:10]])
    tasks = [{"task_id": f"v10_{i:024x}", "scene_id": "s1" if i < 20 else p2_val_scene, "task_type": "projective_relations"}
             for i in range(25)]
    B.write_jsonl(tmp / "rows.jsonl", tasks)

    def traj(i):
        return {"messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "<image>obs"},
                             {"role": "assistant", "content": "<action>move_forward|</action>"},
                             {"role": "user", "content": "<image>obs"},
                             {"role": "assistant", "content": "<action>turn_left|</action>"}],
                "images": [f"/f/v10_{i:024x}__s0.png", f"/f/v10_{i:024x}__s1.png"]}
    B.write_jsonl(tmp / "traj_train.jsonl", [traj(i) for i in range(20)])
    B.write_jsonl(tmp / "traj_val.jsonl", [traj(i) for i in range(20, 25)])
    (tmp / "manifest.json").write_text(json.dumps({"train_scenes": ["s1", "s2"], "val_scenes": ["s9"],
                                                   "test_scenes": ["s8"]}))
    (tmp / "base.yaml").write_text(yaml.safe_dump({"stage": "sft", "dataset": "x", "eval_dataset": "y",
                                                   "num_train_epochs": 10}))
    return {
        "name": "t", "out": str(tmp / "out"), "seed": 1, "split_manifest": str(tmp / "manifest.json"),
        "check_media": False,  # synthetic image paths
        "task_rows": str(tmp / "rows.jsonl"), "base_recipe": str(tmp / "base.yaml"),
        "pools": {
            "P0": {"kind": "interiorgs_qa_lf", "dir": str(tmp / "p0"), "train": "qa_train.jsonl",
                   "val": "qa_val.jsonl", "exclude_groups": ["S7"], "val_scenes": "train"},
            "P1": {"kind": "transition_qa", "train": str(tmp / "p1_train.jsonl"), "val": str(tmp / "p1_test.jsonl")},
            "P2": {"kind": "sft_trajectories", "train": str(tmp / "traj_train.jsonl"), "val": str(tmp / "traj_val.jsonl")},
        },
        "val": {"P0": "all", "P1": 4, "P2": 3},
        "recipes": {
            "A": {"pools": {}},
            "D": {"pools": {"P0": "all", "P1": {"times": 1.0, "of": "P2"}, "P2": 10}},
            "E": {"same_as": "D"},
        },
    }


def test_build_writes_tagged_pools_and_per_pool_eval(tmp_path):
    cfg = _setup(tmp_path)
    res = B.build(cfg)
    d = tmp_path / "out" / "D"
    info = json.loads((d / "dataset_info.json").read_text())
    assert set(info) == {"P0_train", "P1_train", "P2_train", "P0_val", "P1_val", "P2_val"}
    counts = {k: len(B.read_jsonl(d / f"{k}.jsonl")) for k in info}
    assert counts == {"P0_train": 5, "P1_train": 10, "P2_train": 10, "P0_val": 1, "P1_val": 4, "P2_val": 3}
    for k in info:
        assert {r["pool"] for r in B.read_jsonl(d / f"{k}.jsonl")} == {k.split("_")[0]}
    p0 = B.read_jsonl(d / "P0_train.jsonl")
    assert all("S7" not in json.loads(r["meta"])["groups"] for r in p0)     # excluded group
    p1 = B.read_jsonl(d / "P1_train.jsonl")
    assert collections.Counter(r["stratum"] for r in p1) == {"goal_now": 5, "closer_now": 5}  # stratified
    y = yaml.safe_load((d / "train.yaml").read_text())
    assert y["dataset"] == "P0_train,P1_train,P2_train" and y["eval_dataset"] == "P0_val,P1_val,P2_val"
    assert y["eval_on_each_dataset"] is True and y["num_train_epochs"] == 10
    table = json.loads((tmp_path / "out" / "recipes.json").read_text())
    assert [t["recipe"] for t in table] == ["A", "D", "E"] and table[2]["counts"] == table[1]["counts"]
    assert not (tmp_path / "out" / "A").exists() and not (tmp_path / "out" / "E").exists()
    assert "P0 的验证行来自 train 场景" in (tmp_path / "out" / "recipes.md").read_text(encoding="utf-8")
    assert res["pools"]["P2"] == {"train": 20, "val": 5}


def test_excluded_scenes_leave_every_pool(tmp_path):
    cfg = _setup(tmp_path)
    (tmp_path / "ex.json").write_text(json.dumps({"excluded": ["s2"]}))
    cfg["exclude_scenes"] = str(tmp_path / "ex.json")
    res = B.build(cfg)
    p1 = B.read_jsonl(tmp_path / "out" / "D" / "P1_train.jsonl")
    assert len(p1) == 10 and {r["scene"] for r in p1} == {"s1"}   # P1 rows of s2 never drawn
    assert res["pools"]["P1"] == {"train": 20, "val": 10}
    assert res["excluded"]["P1"] == {"train": 20, "val": 0} and res["excluded"]["P2"] == {"train": 0, "val": 0}
    assert "排除了 1 个场景" in (tmp_path / "out" / "recipes.md").read_text(encoding="utf-8")


def test_rows_keep_the_source_conversation(tmp_path):
    cfg = _setup(tmp_path)
    B.build(cfg)
    src = {json.dumps(r["messages"]) for r in B.read_jsonl(tmp_path / "traj_train.jsonl")}
    out = B.read_jsonl(tmp_path / "out" / "D" / "P2_train.jsonl")
    assert all(json.dumps(r["messages"]) in src for r in out)
    assert all(r["videos"] == [] and len(r["images"]) == 2 for r in out)


def test_a_val_scene_in_training_rows_is_refused(tmp_path):
    cfg = _setup(tmp_path)
    rows = B.read_jsonl(tmp_path / "p1_train.jsonl")
    rows[0]["scene"] = "s9"
    B.write_jsonl(tmp_path / "p1_train.jsonl", rows)
    with pytest.raises(SystemExit):
        B.build(cfg)


def test_p2_val_rows_must_come_from_val_scenes(tmp_path):
    cfg = _setup(tmp_path, p2_val_scene="s2")   # validation trajectories of a train scene
    with pytest.raises(SystemExit):
        B.build(cfg)


def test_media_must_be_absolute_existing_files(tmp_path):
    img = tmp_path / "a.png"
    img.write_bytes(b"x")
    ok = [{"images": [str(img)], "videos": [[str(img)]]}]
    B.check_media("ok", ok)
    with pytest.raises(SystemExit):
        B.check_media("relative", [{"images": ["exps/a.png"], "videos": []}])
    with pytest.raises(SystemExit):
        B.check_media("missing", [{"images": [str(tmp_path / "b.png")], "videos": []}])


def test_counts():
    avail = {"P0": 230, "P1": 35039, "P2": 9802}
    assert B.resolve_counts({"pools": {"P0": "all", "P1": {"times": 0.5, "of": "P2"}, "P2": "all"}}, avail) == \
        {"P0": 230, "P1": 4901, "P2": 9802}
    assert B.resolve_counts({"total": 1000, "pools": {"P1": {"share": 0.3}, "P2": {"share": 0.7}}}, avail) == \
        {"P1": 300, "P2": 700}
    with pytest.raises(SystemExit):
        B.resolve_counts({"pools": {"P0": 300}}, avail)
