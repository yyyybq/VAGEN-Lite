"""X3: the target decisions from the two records, the spatial-only plan, the recorder never touching an episode, the
analysis (identity check, table, McNemar) and the run script."""
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "x"))
sys.path.insert(0, str(ROOT / "eval"))
import closed_loop  # noqa: E402
import x3_targets as x3  # noqa: E402


def door(open_, xy=(0.0, 0.0), kind="Drawer"):
    return {"type": kind, "open": open_, "xy": list(xy)}


def test_open_drawer_takes_the_door_opened_most_and_its_side():
    base = [0.0, 0.0, 0.0]  # facing +x: +y is the robot's left
    start = {"base": base, "target": "drawer_L", "side": "left",
             "doors": {"drawer_L": door(0.0, (0.5, 0.4)), "drawer_L_low": door(0.0, (0.5, 0.4)),
                       "drawer_R": door(0.0, (0.5, -0.4)), "cab": door(0.0, (0.6, 0.0), "HingeCabinet")}}
    end = {"doors": {"drawer_L": door(0.30), "drawer_L_low": door(0.0), "drawer_R": door(0.02), "cab": door(0.0)}}
    d = x3.decide("OpenDrawer", start, end)
    assert d["chosen"] == "drawer_L" and d["correct"] and d["chosen_side"] == "left"
    end = {"doors": {"drawer_L": door(0.01), "drawer_L_low": door(0.0), "drawer_R": door(0.96), "cab": door(0.2)}}
    d = x3.decide("OpenDrawer", start, end)
    assert d["chosen"] == "drawer_R" and not d["correct"] and d["chosen_side"] == "right"
    assert not d["same_stack_as_target"]
    end = {"doors": {"drawer_L": door(0.0), "drawer_L_low": door(0.5), "drawer_R": door(0.0), "cab": door(0.0)}}
    d = x3.decide("OpenDrawer", start, end)  # the right side and stack, another drawer of it
    assert d["chosen"] == "drawer_L_low" and not d["correct"] and d["same_stack_as_target"] and d["chosen_side"] == "left"
    end = {"doors": {"drawer_L": door(0.04), "drawer_L_low": door(0.0), "drawer_R": door(0.0), "cab": door(0.0)}}
    d = x3.decide("OpenDrawer", start, end)  # nothing past 5%
    assert d["chosen"] is None and not d["correct"]


def test_lateral_follows_the_heading():
    assert x3.lateral([0, 0, math.pi / 2], [-1.0, 0.5]) > 0  # facing +y: -x is the left
    assert x3.lateral([1, 1, math.pi], [2.0, 2.0]) < 0  # facing -x: +y is the right


def test_turn_off_stove_measures_knobs_on_the_circle():
    start = {"target": "front_left", "knobs": {"front_left": 1.2, "front_right": 0.0, "rear_left": 6.2}}
    end = {"knobs": {"front_left": 0.1, "front_right": 0.05, "rear_left": 0.02}}
    d = x3.decide("TurnOffStove", start, end)
    assert d["chosen"] == "front_left" and d["correct"]
    assert d["turned"]["rear_left"] == pytest.approx(2 * math.pi - 6.18, abs=1e-3)  # 6.2 -> 0.02 is a small turn
    end = {"knobs": {"front_left": 1.2, "front_right": 0.9, "rear_left": 6.2}}
    d = x3.decide("TurnOffStove", start, end)
    assert d["chosen"] == "front_right" and not d["correct"]
    assert x3.angle_change(0.1, 6.2) == pytest.approx(6.2 - 0.1 - 2 * math.pi)


def test_dishwasher_rack_direction():
    out = {"target": "out", "rack": 0.5}
    assert x3.decide("SlideDishwasherRack", out, {"rack": 0.97})["correct"]
    d = x3.decide("SlideDishwasherRack", out, {"rack": 0.2})
    assert d["chosen"] == "in" and not d["correct"]
    assert x3.decide("SlideDishwasherRack", {"target": "in", "rack": 1.0}, {"rack": 0.98})["chosen"] is None


def mug(xyz, under=False, grasped=False, counter=False):
    return {"mug": {"xyz": list(xyz), "under_dispenser": under, "grasped": grasped, "on_counter": counter,
                    "on_machine": False, "xy_to_dispenser": 0.3, "dz_to_dispenser": 0.0}}


def test_mug_places():
    s = {"target": "under_dispenser", "mug": [0.0, 0.0, 1.0]}
    assert x3.decide("CoffeeSetupMug", s, mug([0.3, 0, 1], under=True))["correct"]
    assert x3.decide("CoffeeSetupMug", s, mug([0.3, 0, 1.2], grasped=True))["chosen"] == "in_gripper"
    assert x3.decide("CoffeeSetupMug", s, mug([0.01, 0, 1], counter=True))["chosen"] is None  # untouched
    assert x3.decide("CoffeeSetupMug", s, mug([0.2, 0, 1], counter=True))["chosen"] == "on_counter"
    assert x3.decide("CoffeeSetupMug", s, mug([0.2, 0, 0.1]))["chosen"] == "elsewhere"


def test_bowls_in_the_cabinet_and_their_order():
    s = {"target": "cab_1", "radius": {"bowl1": 0.06, "bowl2": 0.08}}
    end = {"bowls": {"bowl1_in_cabinet": True, "bowl2_in_cabinet": True, "bowl1_in_bowl2": True,
                     "bowl2_in_bowl1": False, "grasped": []}}
    d = x3.decide("StackBowlsCabinet", s, end)
    assert d["correct"] and d["order"] == "smaller_on_top" and d["bowls_in_cabinet"] == 2
    end["bowls"].update(bowl1_in_bowl2=False, bowl2_in_bowl1=True, bowl2_in_cabinet=False)
    d = x3.decide("StackBowlsCabinet", s, end)
    assert not d["correct"] and d["order"] == "larger_on_top" and d["bowls_in_cabinet"] == 1
    s["radius"] = {"bowl1": 0.09, "bowl2": 0.08}  # sizes decide which is smaller, not the names
    assert x3.decide("StackBowlsCabinet", s, end)["order"] == "smaller_on_top"


def test_navigation_takes_the_nearest_placement_point_within_half_a_meter():
    s = {"target": "sink", "src": "stove"}
    fx = {"sink": {"type": "Sink", "dist": 0.15}, "stove": {"type": "Stove", "dist": 2.0},
          "fridge": {"type": "FridgeSideBySide", "error": "x"}}
    d = x3.decide("NavigateKitchen", s, {"fixtures": fx})
    assert d["correct"] and d["target_dist"] == 0.15 and not d["at_start_fixture"] and d["second_dist"] == 2.0
    fx["sink"]["dist"], fx["stove"]["dist"] = 1.5, 0.1
    d = x3.decide("NavigateKitchen", s, {"fixtures": fx})
    assert d["chosen"] == "stove" and d["at_start_fixture"] and not d["correct"]
    fx["stove"]["dist"] = 0.7
    d = x3.decide("NavigateKitchen", s, {"fixtures": fx})
    assert d["chosen"] is None and d["nearest"] == "stove"


def test_stacked_fixtures_share_a_placement_point():
    """X3 B, NavigateKitchen 1008 (a success): the microwave hangs over the oven and their placement points differ by
    0.3 mm, so the nearest fixture alone would name the oven."""
    s = {"target": "microwave", "src": "drawer"}
    fx = {"oven": {"dist": 0.0364}, "microwave": {"dist": 0.0367}, "drawer": {"dist": 3.0}}
    d = x3.decide("NavigateKitchen", s, {"fixtures": fx})
    assert d["correct"] and d["nearest"] == "oven" and d["tied"] == ["oven", "microwave"]
    s["target"] = "sink"  # somewhere else: the nearest is still the choice
    d = x3.decide("NavigateKitchen", s, {"fixtures": dict(fx, sink={"dist": 1.0})})
    assert d["chosen"] == "oven" and not d["correct"]


def test_recorded_only_tasks_decide_too():
    s = {"base": [0, 0, 0], "target": "drawer_F", "doors": {"drawer_F": door(0.0, (0.5, 0.0))}}
    d = x3.decide("DeliverStraw", s, {"doors": {"drawer_F": door(0.6)}, "straw": {"grasped": True}})
    assert d["correct"] and d["straw"] == {"grasped": True}
    s = {"target": "rear_right", "knobs": {"rear_right": 0.0, "front_left": 0.0}}
    d = x3.decide("SearingMeat", s, {"knobs": {"rear_right": 0.0, "front_left": 0.0}, "pan_burner": "front_left"})
    assert d["chosen"] is None and d["pan_burner"] == "front_left"
    with pytest.raises(ValueError):
        x3.decide("PnPCounterToSink", {}, {})


def test_the_x3_plan_is_the_spatial_part_of_the_fixed_evaluation():
    full, _ = closed_loop.eval_plan_episodes(ROOT / "eval/eval_d6.json")
    mine, p = closed_loop.eval_plan_episodes(ROOT / "x/x3_plan.json")
    spatial = set(json.loads((ROOT / "demos/spatial_tasks.json").read_text())["spatial"])
    assert set(spatial) == set(x3.SPATIAL)
    assert mine == [e for e in full if e["task"] in spatial] and len(mine) == 450
    keys = closed_loop.excluded_episodes()["keys"]
    assert len(closed_loop.apply_exclusions(mine, keys)) == 447
    assert p["end_state"] and p["split"] == "target" and p["execute_steps"] == 16 and p["stable_counter_regions"]


class Module:
    def __init__(self, fail_start=False):
        self.fail_start, self.calls = fail_start, []

    def start(self, task, env):
        self.calls.append(("start", task))
        if self.fail_start:
            raise KeyError("drawer")
        return {"target": "t"}

    def end(self, task, env):
        self.calls.append(("end", task))
        return {"doors": {}}


def test_end_state_never_raises():
    m = Module()
    rec = closed_loop.EndState("OpenDrawer", object(), load=lambda: m)
    assert rec.finish(object()) == {"start": {"target": "t"}, "end": {"doors": {}}}
    assert m.calls == [("start", "OpenDrawer"), ("end", "OpenDrawer")]
    rec = closed_loop.EndState("OpenDrawer", object(), load=lambda: Module(fail_start=True))
    out = rec.finish(object())
    assert out["start"] == {"error": "KeyError: 'drawer'"} and out["end"] == {"doors": {}}

    def broken():
        raise ImportError("no robocasa")
    out = closed_loop.EndState("OpenDrawer", object(), load=broken).finish(object())
    assert out["start"]["error"].startswith("ImportError") and out["end"] == out["start"]


def test_the_real_recorder_module_loads_without_the_simulator():
    module = closed_loop._load_x3_targets()
    assert module.MAIN_TASKS == x3.MAIN_TASKS and callable(module.start) and callable(module.end)


def row(task, seed, success, steps=100, sha="a", end_state=None):
    r = {"task": task, "seed": seed, "status": "complete", "success": success, "steps": steps, "init": {"qpos_sha": sha}}
    if end_state:
        r["end_state"] = end_state
    return r


def rack_state(target, before, after):
    return {"start": {"target": target, "rack": before}, "end": {"rack": after}}


def test_analysis_checks_identity_and_pairs_with_b(tmp_path):
    exps, xdir = tmp_path / "experiments", tmp_path / "X3"
    plan = {"B": [(1000, True, 0.97), (1001, False, 0.5), (1002, False, 0.2)],
            "S": [(1000, False, 0.3), (1001, False, 0.5), (1002, True, 0.97)]}
    for s, eps in plan.items():
        orig = exps / f"DS-{s}-100-s1/evals/steps_200000/out/episodes/episodes.jsonl"
        orig.parent.mkdir(parents=True)
        orig.write_text("".join(json.dumps(row("SlideDishwasherRack", seed, ok)) + "\n" for seed, ok, _ in eps))
        rep = xdir / s / "out/episodes/episodes.jsonl"
        rep.parent.mkdir(parents=True)
        lines = [row("SlideDishwasherRack", seed, ok, end_state=rack_state("out", 0.5, after)) for seed, ok, after in eps]
        if s == "S":
            lines[0]["steps"] = 99  # one replay differs from its original
        rep.write_text("".join(json.dumps(r) + "\n" for r in lines) + "not json\n")
    res = x3.analyze(xdir, exps, xdir / "results")
    assert res["identity"]["B"]["same"] == 3 and res["identity"]["B"]["different"] == []
    assert res["identity"]["S"]["different"][0]["seed"] == 1000 and res["identity"]["S"]["same"] == 2
    b = res["starts"]["B"]["SlideDishwasherRack"]
    assert b["correct"]["k"] == 1 and b["failures"] == {"n": 2, "wrong_target": 1, "nothing_chosen": 1,
                                                       "right_target_not_finished": 0}
    assert res["starts"]["S"]["all"]["correct"]["k"] == 1
    assert res["vs_B"]["S"] == {"pairs": 3, "only_this": 1, "only_ref": 1, "p": 1.0}
    md = (xdir / "results/results.md").read_text(encoding="utf-8")
    assert "| S | 3 | 2 | 1 | 0 |" in md and "SlideDishwasherRack" in md
    assert len((xdir / "results/episodes.jsonl").read_text().splitlines()) == 6


def test_run_script_replays_the_e1_final_checkpoint_with_the_x3_plan(tmp_path):
    runs = tmp_path / "experiments"
    ckpt = runs / "DS-RS-100-s1/run/final_model/pytorch_model.pt"
    ckpt.parent.mkdir(parents=True)
    ckpt.write_text("w")
    fake = tmp_path / "eval.sh"
    fake.write_text('echo "$EVAL_PLAN|$1|$2|$3|$4" > "$(dirname "$0")/called"\n')
    env = dict(os.environ, X3_RUNS=str(runs), X3_EVAL=str(fake))
    res = subprocess.run(["bash", str(ROOT / "x/run_x3.sh"), "RS", str(tmp_path / "X3")], capture_output=True,
                         text=True, env=env)
    assert res.returncode == 0, res.stdout + res.stderr
    plan, ck, out, servers, workers = (tmp_path / "called").read_text().strip().split("|")
    assert plan.endswith("downstream/x/x3_plan.json") and ck == str(ckpt) and out == str(tmp_path / "X3/RS/out")
    assert (servers, workers) == ("8", "96")
    res = subprocess.run(["bash", str(ROOT / "x/run_x3.sh"), "RD", str(tmp_path / "X3")], capture_output=True,
                         text=True, env=env)
    assert res.returncode == 6 and "no checkpoint" in res.stdout
