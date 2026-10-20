"""CPU tests of downstream/eval/watch.py (no sco calls), the pool picker, and the spatial-task list."""
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))
import sco_job  # noqa: E402
import watch  # noqa: E402


def fake_exp(root: Path, steps=(25000, 50000), final=True):
    exp = root / "DS-B-100-s1"
    (exp / "run/checkpoints").mkdir(parents=True)
    (exp / "experiment.json").write_text(json.dumps({"steps": 200000}))
    for s in steps:
        (exp / f"run/checkpoints/steps_{s}_pytorch_model.pt").write_bytes(b"x")
    if final:
        (exp / "run/final_model").mkdir()
        (exp / "run/final_model/pytorch_model.pt").write_bytes(b"x")
    (exp / "run/metrics.jsonl").write_text(json.dumps({"step": 50100, "action_dit_loss": 0.0123}) + "\n")
    return exp


def test_checkpoints_in_step_order_with_final_tagged_by_max_steps(tmp_path):
    exp = fake_exp(tmp_path, steps=(50000, 25000))
    assert [t for t, _ in watch.checkpoints(exp)] == ["steps_25000", "steps_50000", "steps_200000"]


def test_tick_dry_waits_for_settled_files_and_never_submits(tmp_path):
    exp = fake_exp(tmp_path)
    old = time.time() - 3600
    os.utime(exp / "run/checkpoints/steps_25000_pytorch_model.pt", (old, old))
    rows = watch.tick(["DS-B-100-s1"], tmp_path, gpus=1, workers=12, settle=600, dry=True)
    states = {e["tag"]: e["state"] for e in rows[0]["evals"]}
    assert states["steps_25000"] == "would submit"
    assert states["steps_50000"].startswith("waiting")
    assert rows[0]["step"] == 50100 and rows[0]["train"] is None
    assert not (exp / "evals").exists()
    text = watch.board(rows)
    assert "DS-B-100-s1" in text and "0.0123" in text


def test_score_groups_and_spatial(tmp_path):
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"per_task": {
        "OpenDrawer": {"n": 20, "successes": 10}, "CloseFridge": {"n": 20, "successes": 5},
        "SearingMeat": {"n": 20, "successes": 1}}}))
    s = watch.score(summary)
    assert s == {"all": "16/60", "atomic_seen": "15/40", "spatial": "11/40", "composite_seen": "1/20"}


def test_waiting_by_pool():
    jobs = [{"resource_pool": {"name": "h800"}, "state": "STARTING"},
            {"resource_pool": "zoetrope", "state": "RUNNING"},
            {"resource_pool": {"name": "zoetrope"}, "state": "QUEUEING"},
            {"resource_pool": {"name": "other"}, "state": "STARTING"}]
    assert sco_job.waiting_by_pool(jobs) == {"h800": 1, "zoetrope": 1}


def test_spatial_list_partitions_the_34_seen_tasks():
    tasks = json.loads((ROOT / "demos/target50_tasks.json").read_text())["tasks"]
    seen = {t["task"] for t in tasks if t["group"] != "composite_unseen"}
    lists = json.loads((ROOT / "demos/spatial_tasks.json").read_text())
    parts = [set(lists["spatial"]), set(lists["borderline_not_in_group"]), set(lists["not_spatial"])]
    assert len(seen) == 34 and set().union(*parts) == seen and sum(len(p) for p in parts) == 34
