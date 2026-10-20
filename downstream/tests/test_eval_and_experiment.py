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


def test_build_config_rejects_a_mismatched_mixture(tmp_path):
    import yaml

    recipe = yaml.safe_load((ROOT / "train/recipe.yaml").read_text())
    data = write_data(tmp_path, ["A"])
    (data / "mixture_spec.json").write_text(json.dumps([["B/lerobot", 1.0, "panda_omron_robocasa365"]]))
    with pytest.raises(ValueError):
        make_experiment.build_config(recipe, exp_dir=tmp_path / "e", exp_id="x", start_path="/s", data_dir=data,
                                     seed=1, steps=10, save_interval=10, log_every=1)
