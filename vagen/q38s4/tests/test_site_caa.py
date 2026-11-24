from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

from vagen.q38s4.tools import site_caa as SC

pytestmark = pytest.mark.skipif(not (SC.EASI_ROOT / "scripts/submissions/backends/lmmseval.py").is_file(),
                                reason="EASI v0.2.2 not at EASI_ROOT (cluster only)")

COUNT, SPATIAL, MOVE = "counting & existence", "spatial relationship reasoning", "movement prediction & navigation"


def part(qa: Path, job: str, name: str, task: str, items, stamp: str = "20261124_120000", done: bool = True,
         model: str = "evalprep__c_nfp_3441_nohead") -> Path:
    """items: (category, chance-adjusted value, its denominator); accuracy = 1 if the value is 1 else 0.  model: the
    folder lmms-eval names after the model path's last two parts (the C line's are not models__*)."""
    d = qa / job / name / model
    d.mkdir(parents=True, exist_ok=True)
    f = d / f"{stamp}_samples_{task}.jsonl"
    f.write_text("".join(json.dumps({"doc_id": i, "accuracy": {"total": 1.0, "overall": float(v == 1), cat: float(v == 1)},
                                     "chance_adjusted_acc": {"total": den, "overall": v, cat: v}}) + "\n"
                         for i, (cat, v, den) in enumerate(items)))
    if done:
        (qa / job / name / "done.json").write_text("{}")
    return f


IMAGE = [(COUNT, 1.0, 1.0), (COUNT, 0.0, 1.0), (SPATIAL, 0.5, 1.0)]
VIDEO = [(SPATIAL, 1.0, 1.0), (MOVE, 1.0, 1.0)]


def test_combined_and_parts(tmp_path):
    qa = tmp_path / "qa"
    img = part(qa, "site_image", "m", "site_bench_image", IMAGE)
    vid = part(qa, "site_video", "m", "site_bench_video_multiimage", VIDEO)
    out = tmp_path / "out" / "m_site.json"
    out.parent.mkdir()
    res = SC.site(qa, "m", out)
    assert res["site_caa"] == round(3.5 / 5 * 100, 4) == 70.0
    assert res["site_image_caa"] == 50.0 and res["site_video_caa"] == 100.0
    subs = res["sub_scores"]
    assert subs["counting_and_existence_caa"] == 50.0 and subs["spatial_relationship_reasoning_caa"] == 75.0
    assert subs["movement_prediction_and_navigation_caa"] == 100.0
    assert res["files"] == {"site_image": str(img), "site_video": str(vid)}
    assert json.loads(out.read_text())["site_caa"] == 70.0
    join = out.parent / "m_site_join"
    assert sorted(p.name for p in join.iterdir()) == sorted([img.name, vid.name])
    assert all(p.is_symlink() for p in join.iterdir())


def test_newest_file_and_unequal_denominators(tmp_path):
    qa = tmp_path / "qa"
    part(qa, "site_image", "m", "site_bench_image", [(COUNT, 0.0, 1.0)] * 3, stamp="20261124_100000")
    part(qa, "site_image", "m", "site_bench_image", [(COUNT, 0.5, 2.0), (COUNT, 1.0, 1.0)], stamp="20261124_110000")
    part(qa, "site_video", "m", "site_bench_video_multiimage", [(MOVE, 1.0, 1.0)])
    res = SC.site(qa, "m", tmp_path / "s.json")
    assert res["files"]["site_image"].endswith("20261124_110000_samples_site_bench_image.jsonl")
    assert res["site_image_caa"] == round(1.5 / 3 * 100, 4) and res["site_caa"] == round(2.5 / 4 * 100, 4)


def test_needs_done_and_fresh_folders(tmp_path):
    qa = tmp_path / "qa"
    part(qa, "site_image", "m", "site_bench_image", IMAGE)
    part(qa, "site_video", "m", "site_bench_video_multiimage", VIDEO, done=False)
    with pytest.raises(SystemExit, match="done.json"):
        SC.site(qa, "m", tmp_path / "s.json")
    (qa / "site_video" / "m" / "done.json").write_text("{}")
    (tmp_path / "s_image").mkdir()
    with pytest.raises(SystemExit, match="exists"):
        SC.site(qa, "m", tmp_path / "s.json")
    with pytest.raises(SystemExit, match="exists"):
        SC.main(["--qa", str(qa), "--name", "m", "--out", str(tmp_path / "s_image")])


def test_no_samples_and_same_named_newest(tmp_path):
    qa = tmp_path / "qa"
    (qa / "site_image" / "m").mkdir(parents=True)
    (qa / "site_image" / "m" / "done.json").write_text("{}")
    part(qa, "site_video", "m", "site_bench_video_multiimage", VIDEO)
    with pytest.raises(SystemExit, match="no samples file"):
        SC.site(qa, "m", tmp_path / "s.json")
    f = part(qa, "site_image", "m", "site_bench_image", IMAGE)
    other = qa / "site_image" / "m" / "models__other" / f.name
    other.parent.mkdir()
    other.write_text(f.read_text())
    with pytest.raises(SystemExit, match="two newest"):
        SC.site(qa, "m", tmp_path / "s.json")


def test_load_easi_leaves_sys_path(tmp_path):
    path, scripts = list(sys.path), sys.modules.get("scripts")
    mod = SC.load_easi()
    assert Path(mod.__file__).resolve() == (SC.EASI_ROOT / "scripts/submissions/backends/lmmseval.py").resolve()
    assert sys.path == path and sys.modules.get("scripts") is scripts
    assert SC.load_easi() is mod and "site" in mod._METRIC_MAP


def test_base_folder_name_and_relative_qa(tmp_path, monkeypatch):
    qa = tmp_path / "qa"
    part(qa, "site_image", "q38base", "site_bench_image", IMAGE, model="models__Qwen3.8-27B")
    part(qa, "site_video", "q38base", "site_bench_video_multiimage", VIDEO, model="models__Qwen3.8-27B")
    (tmp_path / "out").mkdir()
    monkeypatch.chdir(tmp_path)
    assert SC.main(["--qa", "qa", "--name", "q38base", "--out", "out/base.json"]) == 0
    res = json.loads((tmp_path / "out" / "base.json").read_text())
    assert res["site_caa"] == 70.0 and res["site_image_caa"] == 50.0 and res["site_video_caa"] == 100.0
    links = list((tmp_path / "out" / "base_join").iterdir())
    assert len(links) == 2 and all(os.path.isabs(os.readlink(p)) and p.resolve().is_file() for p in links)
