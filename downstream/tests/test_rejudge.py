"""Re-judging N1 / N3 (e) records with the render fix (downstream/n/rejudge.py; RENDER_CAUSE_20261107 item 1): the
set parsing, the scene list, reading the recorded judges, the share tolerance, the holds rule, a whole instance on a
fake scene (judges and image comparison), and collect's counts and lists."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "n"))
import rejudge as R  # noqa: E402


def inst(iid="T_s1_absolute_positioning", task="T", seed=1, tt="absolute_positioning", start_ok=False,
         short_vis=None, best_vis=None):
    return {"instance_id": iid, "task": task, "seed": seed, "scene_id": f"{task}_s{seed}", "task_type": tt,
            "hide_robot": True, "objects": [], "item": {"x": 1}, "start_pose": [0.0, 0.0, 0.0],
            "start_judge": {"success": start_ok, "visible": {"a": 0.0}},
            "shortest": {"pose": [1.0, 0.0, 0.0], "judge": {"success": True, "visible": short_vis or {"a": 0.05}}},
            "best": {"pose": [2.0, 0.0, 0.0], "judge": {"success": True, "visible": best_vis or {"a": 0.08}}},
            "images": {t: f"images/{iid}_{t}.png" for t in R.TAGS}}


class FakeScene:
    """Judges by pose x: success at x >= 1; visible share x * 0.05 (start 0); renders a flat image of value 10 * x."""

    def __init__(self, shift=0.0, img_shift=0):
        self.start, self.entities, self.shift, self.img_shift = (0.0, 0.0, 0.0), [], shift, img_shift

    def judge(self, item, pose):
        x = pose[0]
        return {"success": x >= 1, "visible": {"a": (0.05 * x if x < 1.5 else 0.08) + self.shift}}

    def render(self, pose):
        v = int(10 * pose[0]) + self.img_shift
        return np.full((4, 4, 3), v, dtype=np.uint8), None


def write_images(d: Path, i: dict):
    (d / "images").mkdir(parents=True, exist_ok=True)
    for t in R.TAGS:
        x = R.recorded(i, t)["pose"][0]
        Image.fromarray(np.full((4, 4, 3), int(10 * x), dtype=np.uint8)).save(d / i["images"][t])


def test_parse_sets_and_scene_keys(tmp_path):
    f1, f2 = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    f1.write_text(json.dumps(inst()) + "\n" + json.dumps(inst(iid="x2")) + "\n")
    f2.write_text(json.dumps(inst(task="U", seed=7)) + "\n")
    sets = R.parse_sets([f"n1={f1}", f"e={f2}"])
    assert sets == {"n1": f1, "e": f2}
    assert R.scene_keys(sets) == ["n1:T:1", "e:U:7"]
    with pytest.raises(ValueError):
        R.parse_sets(["nofile"])


def test_recorded_and_share_tolerance():
    i = inst()
    assert R.recorded(i, "start") == {"pose": [0.0, 0.0, 0.0], "success": False, "visible": {"a": 0.0}}
    assert R.recorded(i, "best")["visible"] == {"a": 0.08}
    assert R.shares_agree({"a": 0.0801}, {"a": 0.08})  # within 2% / 1e-4
    assert not R.shares_agree({"a": 0.09}, {"a": 0.08})
    assert not R.shares_agree({}, {"a": 0.08})  # missing counts as 0
    assert R.shares_agree({"a": 0.5}, None)  # nothing recorded


def test_judge_instance_agrees_and_images_equal(tmp_path, monkeypatch):
    import d3_gen as D

    monkeypatch.setattr(D, "scene_check", lambda scene, i: None)
    i = inst()
    write_images(tmp_path, i)
    out_img = tmp_path / "new"
    out_img.mkdir()
    row = R.judge_instance(FakeScene(), i, tmp_path, out_img)
    assert row["scene"] == "ok" and row["holds"] and row["agrees"]
    assert all(row["images"][t]["equal"] for t in R.TAGS)
    assert sorted(p.name for p in out_img.iterdir()) == sorted(f"{i['instance_id']}_{t}.png" for t in R.TAGS)


def test_judge_instance_shares_and_images_differ(tmp_path, monkeypatch):
    import d3_gen as D

    monkeypatch.setattr(D, "scene_check", lambda scene, i: None)
    i = inst()
    write_images(tmp_path, i)
    (tmp_path / "new").mkdir()
    row = R.judge_instance(FakeScene(shift=0.01, img_shift=3), i, tmp_path, tmp_path / "new")
    assert row["holds"] and not row["agrees"]  # still solvable, shares off
    assert row["images"]["best"] == {"equal": False, "mean_abs": 3.0, "max_abs": 3}


def test_holds_fails_when_start_succeeds_or_scene_differs(tmp_path, monkeypatch):
    import d3_gen as D

    i = inst()
    i["start_pose"] = [1.2, 0.0, 0.0]  # the start now succeeds: the record does not hold
    monkeypatch.setattr(D, "scene_check", lambda scene, x: None)
    (tmp_path / "new").mkdir()
    row = R.judge_instance(FakeScene(), i, tmp_path, tmp_path / "new")
    assert not row["holds"] and not row["agrees"]
    assert row["images"]["start"].get("missing")
    monkeypatch.setattr(D, "scene_check", lambda scene, x: "scene_mismatch:box")
    assert R.judge_instance(FakeScene(), i, tmp_path, tmp_path / "new") == {
        "instance_id": i["instance_id"], "task_type": i["task_type"], "scene_id": i["scene_id"],
        "scene": "scene_mismatch:box"}


def test_collect_counts(tmp_path):
    f = tmp_path / "n1.jsonl"
    ids = ["T_s1_a", "T_s1_b", "T_s1_c", "U_s2_d"]
    f.write_text("".join(json.dumps(inst(iid=k, task=k.split("_")[0], seed=int(k.split("_")[1][1:]))) + "\n"
                         for k in ids))
    eq = {t: {"equal": True} for t in R.TAGS}
    rows = [{"instance_id": "T_s1_a", "task_type": "absolute_positioning", "scene_id": "T_s1", "scene": "ok",
             "holds": True, "agrees": True, "images": eq},
            {"instance_id": "T_s1_b", "task_type": "absolute_positioning", "scene_id": "T_s1", "scene": "ok",
             "holds": False, "agrees": False, "images": dict(eq, best={"equal": False})},
            {"instance_id": "T_s1_c", "task_type": "absolute_positioning", "scene_id": "T_s1",
             "scene": "scene_mismatch:start"}]
    d = tmp_path / "out" / "scenes" / "n1"
    d.mkdir(parents=True)
    (d / "T_s1.json").write_text(json.dumps({"rows": rows, "complete": True}))
    s = R.collect({"n1": f}, tmp_path / "out")["n1"]
    c = s["counts"]["all"]
    assert (c["instances"], c["judged"], c["holds"], c["agrees"], c["images_all_equal"]) == (4, 2, 1, 1, 1)
    assert (c["scene_mismatch"], c["missing"], c["best_image_equal"]) == (1, 1, 1)
    assert s["not_holding"] == ["T_s1_b"] and s["images_differ"] == ["T_s1_b"]
    assert s["scene_mismatch"] == ["T_s1_c"] and s["missing"] == ["U_s2_d"]
    assert len((tmp_path / "out" / "rejudge.jsonl").read_text().splitlines()) == 4
