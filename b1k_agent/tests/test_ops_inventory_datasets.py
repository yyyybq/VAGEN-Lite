"""scripts/inventory_datasets.py on a SYNTHETIC datasets tree (layout from OmniGibson utils/asset_utils.py and
docs/challenge/baselines.md; file contents are placeholders)."""

import json
import os
import subprocess
import sys

from conftest import ROOT

from scripts import inventory_datasets as inv

SCENE = "house_double_floor_lower"


def touch(path, text="x"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def make_tree(root):
    inst = os.path.join(root, "2026-challenge-task-instances")
    touch(os.path.join(inst, "metadata", "B100_task_misc.csv"),
          "Task ID,Task,Rooms to inlcude\n0,turning_on_radio,living_room\n1,picking_up_trash,kitchen\n2,store_honey,kitchen\n")
    touch(os.path.join(inst, "metadata", "task.jsonl"), '{"task_name": "turning_on_radio", "length": 2149.6}\n')
    sub = os.path.join(inst, "scenes", SCENE, "json", f"{SCENE}_task_turning_on_radio_instances")
    for i in range(3):
        touch(os.path.join(sub, f"{SCENE}_task_turning_on_radio_0_{i}_template-tro_state.json"), "{}")
    pub = os.path.join(inst, "scene_test", "public", SCENE, "json")
    touch(os.path.join(pub, f"{SCENE}_task_turning_on_radio_0_301_template.json"), "{}")
    for i in (301, 302):
        touch(os.path.join(pub, f"{SCENE}_task_picking_up_trash_0_{i}_template.json"), "{}")
    touch(os.path.join(root, "behavior-1k-assets", "VERSION"), "3.9.3\n")
    touch(os.path.join(root, "behavior-1k-assets", "objects", "radio", "radio.usd"), "u" * 100)
    touch(os.path.join(root, "omnigibson.key"), "SECRET-KEY-CONTENT")
    demos = os.path.join(root, "2026-challenge-demos")
    touch(os.path.join(demos, "meta", "info.json"), json.dumps({"codebase_version": "v3.0", "fps": 30, "total_episodes": 4,
                                                                 "features": {"action": {}, "observation.state": {}}}))
    touch(os.path.join(demos, "meta", "tasks.jsonl"), '{"task_index": 0, "task": "turning_on_radio"}\n')
    touch(os.path.join(demos, "data", "chunk-000", "file-000.parquet"))
    touch(os.path.join(demos, "data", "chunk-001", "file-000.parquet"))
    touch(os.path.join(demos, "videos", "observation.rgb.zed_link_camera_0", "chunk-000", "file-000.mp4"))
    touch(os.path.join(demos, "annotations", "task-0000", "episode_00000010.json"), "{}")
    per_task = os.path.join(root, "other_demos", "b1k", "store_honey")
    touch(os.path.join(per_task, "meta", "info.json"), "{}")
    touch(os.path.join(per_task, "data", "chunk-000", "file-000.parquet"))
    task_data = os.path.join(root, "..", "task_data.json")
    touch(task_data, json.dumps({"tasks": [{"id": "turning_on_radio"}, {"id": "picking_up_trash"}, {"id": "store_honey"}]}))
    return os.path.abspath(task_data)


def test_inventory_synthetic_tree(tmp_path):
    root = str(tmp_path / "datasets")
    task_data = make_tree(root)
    out = inv.inventory(root, task_data=task_data)
    items = out["known_items"]
    assert items["B100_task_misc.csv"]["exists"] and not items["instances_hidden_test"]["exists"]
    assert items["behavior-1k-assets"]["version"] == "3.9.3"
    assert items["omnigibson.key"]["exists"] and "bytes" not in items["omnigibson.key"]
    assert out["id_mapping"]["source"] == "B100_task_misc.csv"
    assert out["instances"]["train"]["turning_on_radio"]["ids"] == [0, 1, 2]
    assert out["instances"]["public_test"]["picking_up_trash"]["ids"] == [301, 302]
    rows = {r["task"]: r for r in out["coverage"]}
    assert (rows["turning_on_radio"]["train"], rows["turning_on_radio"]["public_test"]) == (3, 1)
    # chunk-000 of the per-task store_honey root is NOT counted as task 0
    assert rows["turning_on_radio"]["demo_chunks"] == 1 and rows["turning_on_radio"]["annotations"] == 1
    assert rows["picking_up_trash"]["demo_chunks"] == 1 and rows["store_honey"]["demo_chunks"] == 1
    assert out["coverage_summary"]["tasks_with_public_test_instances"] == 2
    demo = next(lr for lr in out["lerobot_roots"] if lr["path"].endswith("2026-challenge-demos"))
    assert demo["info"]["fps"] == 30 and demo["tasks_jsonl"]["known_task_names"] == {"turning_on_radio": 0}
    exts = {e["ext"] for e in out["file_types"]}
    assert {".json", ".usd", ".parquet", ".mp4"} <= exts
    text = inv.render(out)
    assert "| omnigibson.key | omnigibson.key | yes |" in text and "SECRET" not in json.dumps(out)
    assert not out["truncated"]


def test_caps_mark_lower_bounds(tmp_path):
    root = str(tmp_path / "datasets")
    task_data = make_tree(root)
    out = inv.inventory(root, task_data=task_data, max_files=1)
    assert out["truncated"] and ">= " in inv.render(out)


def test_cli_writes_json_and_md(tmp_path):
    root = str(tmp_path / "datasets")
    task_data = make_tree(root)
    js = str(tmp_path / "out" / "datasets.json")
    r = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "inventory_datasets.py"), root,
                        "--task-data", task_data, "--out", js], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    assert json.load(open(js))["coverage_summary"]["tasks_listed"] == 3
    assert os.path.isfile(str(tmp_path / "out" / "datasets.md"))


def test_slow_size_walk_cannot_starve_instance_coverage(tmp_path, monkeypatch):
    """Regression: the size walk ran first and could spend the whole time budget on multi-TB asset trees, after which
    every task showed 0 instances. The instance scan now runs first."""
    root = str(tmp_path / "datasets")
    task_data = make_tree(root)
    real_walk = inv.walk_size

    def exhausting_walk(path, max_files, budget, *a, **kw):
        budget.deadline = 0.0  # this walk used up the whole budget
        return real_walk(path, max_files, budget, *a, **kw)

    monkeypatch.setattr(inv, "walk_size", exhausting_walk)
    out = inv.inventory(root, task_data=task_data)
    assert out["truncated"] and not out["instances_truncated"]
    rows = {r["task"]: r for r in out["coverage"]}
    assert (rows["turning_on_radio"]["train"], rows["picking_up_trash"]["public_test"]) == (3, 2)


def test_truncated_instance_scan_is_reported_as_lower_bound(tmp_path):
    root = str(tmp_path / "datasets")
    task_data = make_tree(root)
    out = inv.inventory(root, task_data=task_data, time_limit_s=0.0)
    assert out["instances_truncated"] and out["coverage_summary"]["instance_scan_truncated"]
    assert "instance counts in the coverage table are lower bounds" in inv.render(out)


def test_symlinked_top_level_dir_is_measured_at_its_target(tmp_path):
    root = str(tmp_path / "datasets")
    task_data = make_tree(root)
    target = str(tmp_path / "other_disk" / "behavior-1k-assets")
    os.makedirs(os.path.dirname(target))
    os.rename(os.path.join(root, "behavior-1k-assets"), target)
    os.symlink(target, os.path.join(root, "behavior-1k-assets"))
    os.symlink(str(tmp_path / "missing"), os.path.join(root, "dangling"))
    out = inv.inventory(root, task_data=task_data)
    top = {t["name"]: t for t in out["top_level"]}
    assert top["behavior-1k-assets"]["symlink"] == target and top["behavior-1k-assets"]["files"] == 2
    assert top["behavior-1k-assets"]["bytes"] >= 100 and out["known_items"]["behavior-1k-assets"]["version"] == "3.9.3"
    assert top["dangling"]["bytes"] == 0 and top["dangling"]["symlink"].endswith("missing")
