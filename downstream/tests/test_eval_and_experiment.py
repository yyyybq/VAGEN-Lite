"""CPU tests of downstream/eval/closed_loop.py helpers and downstream/train/make_experiment.build_config."""
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
sys.path.insert(0, str(ROOT / "train"))
import closed_loop  # noqa: E402
import make_experiment  # noqa: E402


def test_parse_seeds_and_plan_use_task_horizons():
    assert closed_loop.parse_seeds("1000-1002,1005") == [1000, 1001, 1002, 1005]
    plan = closed_loop.plan(["A", "B"], [1, 2], {"A": 450, "B": 1200})
    assert plan == [{"task": "A", "seed": 1, "max_steps": 450}, {"task": "A", "seed": 2, "max_steps": 450},
                    {"task": "B", "seed": 1, "max_steps": 1200}, {"task": "B", "seed": 2, "max_steps": 1200}]


def test_longest_first_is_a_stable_sort_by_horizon():
    plan = closed_loop.plan(["A", "B", "C"], [1, 2], {"A": 450, "B": 1200, "C": 450})
    order = [(e["task"], e["seed"]) for e in closed_loop.longest_first(plan)]
    assert order == [("B", 1), ("B", 2), ("A", 1), ("A", 2), ("C", 1), ("C", 2)]


def state(qpos_sha="q", layout=1, hinge=0.0, obj=(0.0, 0.0, 0.8), images="i"):
    return {"layout_id": layout, "style_id": 1, "instruction": "x", "nq": 8, "qpos_sha": qpos_sha, "images_sha": images,
            "joints": {"door": [False, [hinge]], "obj": [True, [*obj, 1.0, 0.0, 0.0, 0.0]]}, "rep": 0, "seconds": 1.0,
            "task": "T", "seed": 1}


def test_reset_repeat_compare_separates_noise_from_scene_changes():
    import reset_repeat

    same = reset_repeat.compare([state(), state()])
    assert same["distinct_fingerprints"] == 1 and same["max_diff_other"] == 0.0 and not same["scene_differs"]
    noise = reset_repeat.compare([state(), state(qpos_sha="q2", hinge=3e-6, images="i2")])
    assert noise["distinct_fingerprints"] == 2 and not noise["scene_differs"]
    assert noise["max_diff_other"] == 3e-6 and noise["max_diff_free"] == 0.0 and noise["joints_over_1e-4"] == []
    moved = reset_repeat.compare([state(), state(qpos_sha="q3", obj=(0.2, 0.0, 0.8))])
    assert moved["max_diff_free"] == 0.2 and moved["joints_over_1e-4"] == ["obj"]
    assert reset_repeat.compare([state(), state(layout=2)])["scene_differs"]
    assert reset_repeat.compare([state(), {"error": "x", "rep": 1}])["errors"] == 1


class Geom:
    """Hashed by memory address, like the XML elements RoboCasa de-duplicates with list(set(...))."""

    def __init__(self, name, salt):
        self.name, self.salt = name, salt

    def __hash__(self):
        return self.salt


def dedup(items, keep=None):
    """Same shape as Counter.get_reset_regions: list(set(...)) and then rely on the order."""
    return [g.name for g in list(set(items))]


def test_with_ordered_set_keeps_first_seen_order_whatever_the_hashes():
    stable = closed_loop.with_ordered_set(dedup)
    for salts in ([3, 1, 2], [1, 2, 3], [2, 3, 1]):  # a different "memory layout" each time
        a, b, c = Geom("a", salts[0]), Geom("b", salts[1]), Geom("c", salts[2])
        assert stable([a, b, a, c, b]) == ["a", "b", "c"]
    assert [dedup([Geom("a", 3), Geom("b", 1)]), dedup([Geom("a", 1), Geom("b", 3)])] == [["b", "a"], ["a", "b"]]
    assert stable.__name__ == "dedup" and stable.__defaults__ == (None,) and stable.downstream_ordered_set
    assert "set" not in dedup.__globals__  # the original function and its module are untouched


def test_reset_repeat_plan_tasks_are_the_34_seen_tasks():
    import reset_repeat

    tasks = reset_repeat.plan_tasks()
    assert len(tasks) == 34 and "NavigateKitchen" in tasks


def test_registry_horizons_cover_the_50_target_tasks():
    h = closed_loop.horizons()
    assert len(h) == 50 and h["NavigateKitchen"] == 450 and h["SearingMeat"] == 4350


def test_summarize():
    rows = [{"status": "complete", "task": "A", "success": True, "wall_seconds": 10.0, "reset_seconds": 2.0, "steps": 80},
            {"status": "complete", "task": "A", "success": False, "wall_seconds": 30.0, "reset_seconds": 2.0, "steps": 450},
            {"status": "error", "task": "B"}]
    s = closed_loop.summarize(rows, workers=3, wall=40.0)
    assert s["complete"] == 2 and s["errors"] == 1 and s["successes"] == 1
    assert s["per_task"]["A"] == {"n": 2, "successes": 1, "success_rate": 0.5}
    assert s["seconds_per_step_mean"] == round(36.0 / 530, 4)


def write_data(tmp_path, names):
    data = tmp_path / "data"
    data.mkdir()
    (data / "split_manifest.json").write_text(json.dumps({"datasets": [{"name": n} for n in names]}))
    (data / "mixture_spec.json").write_text(json.dumps([[n + "/lerobot", 1.0, "panda_omron_robocasa365"] for n in names]))
    return data


def test_build_config_only_fills_the_per_experiment_fields(tmp_path):
    import yaml

    recipe = yaml.safe_load((ROOT / "train/recipe.yaml").read_text())
    data = write_data(tmp_path, ["NavigateKitchen", "OpenDrawer"])
    kw = dict(exp_dir=tmp_path / "exp", exp_id="DS-B-100-s1", data_dir=data, seed=1, steps=200000,
              save_interval=25000, log_every=100)
    a = make_experiment.build_config(recipe, start_path="/starts/B", **kw)
    b = make_experiment.build_config(recipe, start_path="/starts/RS", **kw)
    assert a["framework"]["qwenvl"]["base_vlm"] == "/starts/B"
    a["framework"]["qwenvl"]["base_vlm"] = b["framework"]["qwenvl"]["base_vlm"]
    assert a == b  # the start is the only difference
    assert a["seed"] == 1 and a["trainer"]["max_train_steps"] == 200000 and a["trainer"]["save_interval"] == 25000
    assert a["datasets"]["vla_data"]["mixture_spec"][1] == ["OpenDrawer/lerobot", 1.0, "panda_omron_robocasa365"]
    assert recipe["framework"]["qwenvl"]["base_vlm"] is None  # the recipe itself is not modified


def test_build_config_freezes_the_named_modules_only_when_asked(tmp_path):
    """X1 (the user's decision 2 of 2026-10-26): only the action head trains = the whole qwen_vl_interface frozen."""
    import yaml

    recipe = yaml.safe_load((ROOT / "train/recipe.yaml").read_text())
    data = write_data(tmp_path, ["OpenDrawer"])
    kw = dict(exp_dir=tmp_path / "exp", exp_id="X1-B-030-s1", start_path="/starts/B", data_dir=data, seed=1,
              steps=50000, save_interval=25000, log_every=100)
    assert make_experiment.build_config(recipe, **kw)["trainer"].get("freeze_modules", "") == ""
    cfg = make_experiment.build_config(recipe, freeze_modules="qwen_vl_interface", **kw)
    assert cfg["trainer"]["freeze_modules"] == "qwen_vl_interface" and cfg["trainer"]["max_train_steps"] == 50000


def test_build_config_rejects_a_mismatched_mixture(tmp_path):
    import yaml

    recipe = yaml.safe_load((ROOT / "train/recipe.yaml").read_text())
    data = write_data(tmp_path, ["A"])
    (data / "mixture_spec.json").write_text(json.dumps([["B/lerobot", 1.0, "panda_omron_robocasa365"]]))
    with pytest.raises(ValueError):
        make_experiment.build_config(recipe, exp_dir=tmp_path / "e", exp_id="x", start_path="/s", data_dir=data,
                                     seed=1, steps=10, save_interval=10, log_every=1)
