"""Tests of the pointing pool Pg (group G of matrix v3; python -m pytest data_gen/transition_qa/tests -q).  The
integration test reads the round-11 SFT data and InteriorGS scenes on the cluster and is skipped elsewhere."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from data_gen.transition_qa import pointing_qa as PQ

SFT_DIR = Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft")
GS_ROOT = Path("/mnt/umm/users/yinbaiqiao/InteriorGS_ply")


def cand(task, state, slot, family="equidistance", scene="s", split="train"):
    r = {"pool": "Pg", "family": family, "task_id": task, "scene": scene, "split": split, "state_id": state,
         "state_pos": 0, "state_role": "start", "image": "/x.png", "slot": slot, "ref": "the lamp",
         "box": [100.0, 80.0, 140.0, 100.0], "n_slots": 2, "n_in_view": 1}
    r["kind"] = PQ.kind_of(r)
    r["sample_id"] = PQ.sample_id(r)
    return r


def test_kinds_halves_and_probe_spread():
    rs = [cand(f"t{i}", f"s{j}", "A", family=("equidistance", "fov_inclusion")[i % 2]) for i in range(40) for j in range(3)]
    picked = PQ.by_kind(rs, 20, random.Random(0))
    assert len(picked) == 20 and sum(r["kind"] == "point" for r in picked) == 10
    assert len(PQ.by_kind(rs, 1000, random.Random(0))) == len(rs)  # fewer than asked: all
    probe = PQ.probe_select(rs, 10, random.Random(0))
    assert len(probe) == 10 and len({r["task_id"] for r in probe}) == 10
    assert {r["family"] for r in probe} == {"equidistance", "fov_inclusion"}


def test_rows_answer_their_own_truth():
    """Every row's answer is judged right by the probe's judge, in both coordinate formats."""
    from eval.diagnostics import qa_probe as QP
    from vagen.envs.active_spatial.prompt import TARGET_FORMATS

    for kind in PQ.KINDS:
        r = dict(cand("t", "s0", "B"), kind=kind)
        for fmt in TARGET_FORMATS:
            row = PQ.to_lf(r, fmt)
            assert row["qclass"] == "pointing" and row["judge"] == kind and row["options"] == []
            assert row["messages"][0]["content"].startswith("<image>\n") and "the lamp" in row["messages"][0]["content"]
            got = QP.judge_pointing(row, row["messages"][1]["content"])
            assert got["correct"] and got["compliant"], (kind, fmt, row["messages"][1]["content"])
    assert PQ.to_lf(dict(cand("t", "s0", "A"), kind="point"), "targets_pixel")["messages"][1]["content"] == "(120,90)"


@pytest.mark.skipif(not (SFT_DIR / "sft_trajectories.jsonl").exists() or not GS_ROOT.exists(),
                    reason="cluster data only")
def test_raw_and_build_on_two_scenes(tmp_path):
    """One train and one validation scene: candidates only for clearly visible targets, boxes in the frame, the
    split kept, the probe from the validation scene, both formats written and judged right."""
    from eval.diagnostics import qa_probe as QP
    from vagen.envs.active_spatial import pointing as PT
    from data_gen.transition_qa import generate as GEN

    excluded = GEN.read_scene_exclusions({"exclude_scenes": GEN.SCENE_EXCLUSIONS})[1]
    first = {}
    with open(SFT_DIR / "sft_trajectories.jsonl", encoding="utf-8") as fh:
        for line in fh:
            t = json.loads(line)
            if not t.get("reserve") and t["split"] in ("train", "val") and t["scene_id"] not in excluded:
                first.setdefault(t["split"], t["scene_id"])
            if len(first) == 2:
                break
    out = tmp_path / "pg"
    PQ.main(["raw", "--sft-dir", str(SFT_DIR), "--gs-root", str(GS_ROOT), "--out", str(out),
             "--scenes", f"{first['train']},{first['val']}"])
    recs, funnel = PQ.read_raw(out)
    assert recs and funnel["candidates"] == len(recs) and funnel["targets"] > funnel["not_clearly_visible"]
    for r in recs:
        x1, y1, x2, y2 = r["box"]
        assert 0 <= x1 <= x2 <= 255 and 0 <= y1 <= y2 <= 255
        assert PT.hit(PT.box_point(r["box"]), {"in_view": True, "box": r["box"]})
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text(json.dumps({"scene_id": first["val"]}) + "\n")
    PQ.main(["build", "--out", str(out), "--split-manifest", str(SFT_DIR / "split_manifest.json"),
             "--probe-scenes-from", str(tasks), "--train", "40", "--test", "6", "--probe", "4"])
    stats = json.loads((out / "stats.json").read_text())
    for fmt in ("targets_pixel", "targets_norm1000"):
        files = {n: [json.loads(l) for l in (out / fmt / n).read_text().splitlines()]
                 for n in ("pg_train.jsonl", "pg_test.jsonl", "pg_probe.jsonl")}
        assert {r["scene"] for r in files["pg_train.jsonl"]} == {first["train"]}
        assert {r["scene"] for r in files["pg_test.jsonl"] + files["pg_probe.jsonl"]} <= {first["val"]}
        assert not {r["task_id"] for r in files["pg_probe.jsonl"]} & {r["task_id"] for r in files["pg_test.jsonl"]}
        for rows in files.values():
            for row in rows:
                assert QP.judge_pointing(row, row["messages"][1]["content"])["correct"]
        assert stats["files"]["pg_train.jsonl"]["questions"] == len(files["pg_train.jsonl"])
