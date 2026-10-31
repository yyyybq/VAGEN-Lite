"""N1 (downstream/n/n_gen.py, n_env.py, n_episode.py, run_n_gen.sh; the user's decision 3 of 2026-10-31): the scene
order, plain object names, the main-checkout loader, the six task types' items and judges (needs the main checkout),
the near-threshold slack, the instance search on a stand-in scene (shortest and most comfortable success), collect's
spread over tasks, the job's per-scene time limit, and the episode runner reading both instance formats."""
import json
import math
import os
import random
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "n"))
import n_env as N  # noqa: E402
import n_episode as E  # noqa: E402
import n_gen as G  # noqa: E402
import n_smoke as S  # noqa: E402

HAS_MAIN = (N.MAIN / "vagen/envs/active_spatial/canonical_task_metrics_v2.py").is_file() and \
    (N.MAIN / "data_gen/interiorgs_v10/families/equidistance.py").is_file()
needs_main = pytest.mark.skipif(not HAS_MAIN, reason="needs the main checkout")
MAIN_PREFIXES = ("vagen", "data_gen")


def isolate_main_modules():
    """N loads the main checkout's vagen / data_gen under their real names (n_env.main_packages); other tests import
    the worktree's own vagen (vagen.envs.robocasa).  Each N test starts without either and leaves things as found."""
    def ours(name):
        return any(name == p or name.startswith(p + ".") for p in MAIN_PREFIXES)

    saved = {k: v for k, v in sys.modules.items() if ours(k)}
    for k in saved:
        del sys.modules[k]
    try:
        yield
    finally:
        for k in [k for k in sys.modules if ours(k)]:
            del sys.modules[k]
        sys.modules.update(saved)


@pytest.fixture(autouse=True)
def isolated():
    yield from isolate_main_modules()


def box(key, name, center, half=(0.2, 0.2, 0.2)):
    c, h = np.asarray(center, float), np.asarray(half, float)
    return {"key": key, "name": name, "kind": "object", "bbox_min": list(c - h), "bbox_max": list(c + h),
            "center": list(c)}


def test_scene_order_takes_the_tasks_in_turn(tmp_path):
    rows = [{"task": t, "seed": s} for t, s in [("B", 1001), ("A", 1003), ("A", 1001), ("B", 1001), ("C", 1000),
                                                ("A", 1002)]]
    x0 = tmp_path / "x0.jsonl"
    x0.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    assert G.scene_order(x0) == [("A", 1001), ("B", 1001), ("C", 1000), ("A", 1002), ("A", 1003)]
    assert G.scene_order(x0, 2) == [("A", 1001), ("B", 1001)]


def test_plain_names():
    assert N.natural_name("kettle non electric") == "kettle" and N.natural_name("jug wide opening") == "jug"
    assert N.natural_name("mug") == "mug" and N.natural_name(" sink ") == "sink"


def test_main_packages_refuse_a_package_imported_from_elsewhere(monkeypatch):
    fake = types.ModuleType("dsfake_pkg")
    fake.__path__ = ["/somewhere/else/dsfake_pkg"]
    monkeypatch.setitem(sys.modules, "dsfake_pkg", fake)
    with pytest.raises(RuntimeError):
        N.main_packages("dsfake_pkg.sub")
    monkeypatch.delitem(sys.modules, "dsfake_pkg")
    N.main_packages("dsfake_pkg.sub")  # stand-ins pointing at the main checkout, no __init__ run
    assert sys.modules["dsfake_pkg.sub"].__path__ == [str(N.MAIN / "dsfake_pkg" / "sub")]
    for name in ("dsfake_pkg.sub", "dsfake_pkg"):
        sys.modules.pop(name, None)


@needs_main
def test_six_task_types_make_upstream_items_and_judge_them():
    mug = box("mug_1", "mug", (3.0, 1.5, 1.0))
    kettle = box("kettle_1", "kettle", (3.0, -1.5, 1.0))
    fridge = box("fridge_1", "fridge", (2.0, 0.0, 0.9), half=(0.3, 0.3, 0.9))
    start = (0.0, 0.0, 0.0)
    cases = {"projective_relations": ([mug, kettle], {"relation": "left"}),
             "equidistance": ([mug, kettle], {}),
             "absolute_positioning": ([mug], {"distance_m": 2.5}),
             "occlusion_alignment": ([box("cup_1", "cup", (4.0, 0.0, 1.0)), fridge], {}),
             "fov_inclusion": ([mug, kettle], {}),
             "screen_occupancy": ([fridge], {"occupancy_ratio": 0.6})}
    texts = {}
    for t, (ents, params) in cases.items():
        item = N.make_item(t, "Task_s1000", ents, params, start)
        assert item["task_type"] == t and [o["id"] for o in item["target_object"]["objects"]] == [e["key"] for e in ents]
        j = N.box_judge(item, start)
        assert isinstance(j["metric"]["gates"], dict) and "success" in j
        texts[t] = item["task_description"]
        m = N.margins(item, j)
        assert all(0.0 <= v <= 1.0 for v in m.values())
    assert texts["equidistance"].startswith("Move to a spot that is equally far from the mug and the kettle")
    assert texts["absolute_positioning"] == ("Move to a spot about 2.5 meters away from the mug (measured to its "
                                             "nearest side), keeping the mug in view.")
    assert texts["occlusion_alignment"] == ("Move so that the cup is hidden behind the fridge from your viewpoint, "
                                            "while the fridge stays in view.")
    assert "about 60% of the image height" in texts["screen_occupancy"]
    assert N.make_item("equidistance", "s", [mug, kettle], {}, start)["canonical_task_metric_version"] == \
        "canonical_spatial_task_h1_v2"
    assert N.required_keys(N.make_item("occlusion_alignment", "s", *cases["occlusion_alignment"], start)) == ["fridge_1"]
    assert N.required_keys(N.make_item("fov_inclusion", "s", [mug, kettle], {}, start)) == ["mug_1", "kettle_1"]
    with pytest.raises(ValueError):  # the upstream tuple rule: equal-distance objects at least 1 m apart
        N.make_item("equidistance", "s", [mug, box("cup_2", "cup", (3.0, 1.0, 1.0))], {}, start)
    fov = N.make_item("fov_inclusion", "s", [mug, kettle], {}, start)
    assert N.box_judge(fov, start)["success"] and not N.box_judge(fov, (0.0, 0.0, math.pi))["success"]


def test_slack_and_near_threshold(monkeypatch):
    monkeypatch.setattr(N, "required_keys", lambda item: ["a", "b"])
    item = {"task_type": "projective_relations"}
    j = {"metric": {"relation_margin_px": 20.0, "inside_frame_fraction_min": 0.9}, "lateral": {"angle_deg": 10.0},
         "visible": {"a": 0.01, "b": 0.0045}}
    m = N.margins(item, j)
    assert m["relation_margin_px"] == pytest.approx(8 / 18) and m["inside_frame"] == 1.0
    assert m["lateral_angle_deg"] == 1.0 and m["visible"] == pytest.approx(0.5)
    assert N.near_threshold(item, j) == ["relation_margin_px", "visible"]
    eq = {"task_type": "equidistance"}
    j2 = {"metric": {"distance_difference_m": 0.22, "equidistance_tolerance_m": 0.25, "inside_frame_fraction_min": 0.7},
          "lateral": {"applies": False}}
    assert N.near_threshold(eq, j2) == ["equidistant"]  # 0.22 > 0.8 x 0.25: the outer fifth of the band
    j2["metric"]["distance_difference_m"] = 0.1
    assert N.near_threshold(eq, j2) == []
    occ = {"task_type": "occlusion_alignment"}
    j3 = {"metric": {"target_hidden_fraction": 0.86, "target_in_frame_fraction": 0.95,
                     "objects": [{"inside_frame_fraction": 0.2}, {"inside_frame_fraction": 0.8}]}, "lateral": {}}
    assert N.near_threshold(occ, j3) == ["target_hidden"]
    assert N.margins({"task_type": "projective_relations"},
                     {"metric": {}, "lateral": {"angle_deg": 43.0}})["lateral_angle_deg"] == pytest.approx(0.4)


class FakeScene:
    """Free floor for x < 0.95; a wall beyond.  Equal distance gets better the further forward."""

    def __init__(self):
        self.start, self.pose, self.task, self.seed, self.hide_robot = (0.0, 0.0, 0.0), (0.0, 0.0, 0.0), "T", 1, True
        self.entities = [box("a", "mug", (2.0, 1.0, 1.0)), box("b", "kettle", (2.0, -1.0, 1.0))]

    def set_pose(self, pose):
        self.pose = pose

    def collision(self):
        return "wall" if self.pose[0] >= 0.95 else None

    def judge(self, item, pose, render=True):
        j = dict(N.box_judge(item, pose))  # the test's stand-in box judge
        j["visible"] = {"a": 0.01, "b": 0.01}
        return j


def fake_box(item, pose):
    """Box success from 0.5 m forward, facing straight; the difference falls from 0.24 m (at 0.6 m) to 0 (0.9 m)."""
    ok = pose[0] > 0.5 and abs(pose[2]) < 1e-6 and abs(pose[1]) < 1e-6
    diff = max(0.0, 0.24 - (pose[0] - 0.6) * 0.8)
    return {"success": ok, "metric": {"gates": {"equidistant": ok}, "distance_difference_m": diff,
                                      "equidistance_tolerance_m": 0.25, "inside_frame_fraction_min": 0.9},
            "lateral": {"applies": False, "ok": True}}


def test_find_keeps_the_shortest_and_the_most_comfortable_success(monkeypatch):
    sc = FakeScene()
    reach = S.reachable(sc, max_depth=4)
    states = reach["states"]
    order = sorted(states, key=lambda k: states[k][1])
    monkeypatch.setattr(N, "make_item", lambda t, sid, ents, params, start: {
        "task_type": t, "task_description": f"equal {ents[0]['name']} {ents[1]['name']}",
        "target_object": {"objects": [{"id": e["key"]} for e in ents]}})
    monkeypatch.setattr(N, "box_judge", fake_box)
    monkeypatch.setattr(N, "required_keys", lambda item: ["a", "b"])
    inst, tally = G.find(sc, states, order, "equidistance", random.Random(0), deadline=1e18)
    assert tally["instance"] == 1 and inst["task_type"] == "equidistance" and inst["hide_robot"] is True
    assert inst["shortest"]["actions"] == 2 and inst["shortest"]["path"] == ["move_forward", "move_forward"]
    assert inst["shortest"]["judge"]["near"] == ["equidistant"]  # 0.24 m of a 0.25 m tolerance
    assert inst["best"]["pose"][0] == pytest.approx(0.9) and inst["best"]["actions"] == 3
    assert inst["near_threshold"] == [] and inst["start_judge"]["success"] is False
    assert inst["instance_id"] == "T_s1_equidistance" and inst["box_success_poses"] >= 2


def test_find_skips_candidates_whose_start_succeeds_and_stops_at_the_deadline(monkeypatch):
    sc = FakeScene()
    states = S.reachable(sc, max_depth=2)["states"]
    order = sorted(states, key=lambda k: states[k][1])
    monkeypatch.setattr(N, "make_item", lambda t, sid, ents, params, start: {
        "task_type": t, "task_description": "", "target_object": {"objects": [{"id": e["key"]} for e in ents]}})
    monkeypatch.setattr(N, "box_judge", lambda item, pose: {"success": True, "metric": {"gates": {}}, "lateral": {}})
    monkeypatch.setattr(N, "required_keys", lambda item: ["a", "b"])
    inst, tally = G.find(sc, states, order, "fov_inclusion", random.Random(0), deadline=1e18)
    assert inst is None and tally["start_succeeds"] == 1
    inst, tally = G.find(sc, states, order, "fov_inclusion", random.Random(0), deadline=0)
    assert inst is None and tally["stopped_time"] == 1


def scene_file(out, task, seed, families):
    insts = {t: {"instance_id": f"{task}_s{seed}_{t}", "task": task, "seed": seed, "task_type": t,
                 "params": {"distance_m": 1.5} if t == "absolute_positioning" else {},
                 "near_threshold": ["visible"] if seed % 2 else [], "shortest": {"actions": 3}}
             for t in families}
    rec = {"task": task, "seed": seed, "scene_id": f"{task}_s{seed}", "complete": True, "instances": insts,
           "tallies": {t: {"instance": 1} for t in families}, "seconds": 100.0, "seconds_setup": 30.0}
    (out / "scenes" / f"{task}_s{seed}.json").write_text(json.dumps(rec))


def test_collect_spreads_each_type_over_the_tasks(tmp_path):
    (tmp_path / "scenes").mkdir()
    for task in ("A", "B", "C"):
        for seed in (1000, 1001, 1002):
            scene_file(tmp_path, task, seed, ["equidistance"] + (["absolute_positioning"] if task == "A" else []))
    (tmp_path / "scenes/D_s1000.timeout.json").write_text(json.dumps({"task": "D", "seed": 1000, "rc": 137,
                                                                      "seconds": 1500}))
    s = G.collect(tmp_path, per_family=4)
    eq = s["families"]["equidistance"]
    assert eq["found"] == 9 and eq["taken"] == 4 and eq["tasks"] == 3
    lines = [json.loads(x) for x in (tmp_path / "instances_eval.jsonl").read_text().splitlines()]
    taken = [x for x in lines if x["task_type"] == "equidistance"]
    assert sorted(x["task"] for x in taken[:3]) == ["A", "B", "C"]  # every task once before any twice
    assert all(x["seed"] == 1000 for x in taken[:3])
    assert s["families"]["absolute_positioning"]["taken"] == 3 and s["families"]["occlusion_alignment"]["taken"] == 0
    assert s["scenes_skipped_for_time"] == [{"scene_id": "D_s1000", "rc": 137, "seconds": 1500}]
    assert s["families"]["absolute_positioning"]["params"] == {'{"distance_m": 1.5}': 3}


STUB = r"""import json, sys, time
from pathlib import Path
cmd = sys.argv[1]
def arg(k):
    return sys.argv[sys.argv.index(k) + 1]
if cmd == "scenes":
    print("Slow:1000\nFast:1001\nDone:1002")
elif cmd == "scene":
    out = Path(arg("--out"))
    with open(out / "calls.jsonl", "a") as f:
        f.write(json.dumps(sys.argv[1:]) + "\n")
    if arg("--task") == "Slow":
        time.sleep(60)
    (out / "scenes" / f"{arg('--task')}_s{arg('--seed')}.json").write_text(json.dumps({"complete": True}))
else:
    out = Path(arg("--out"))
    (out / f"{cmd}.done").write_text(" ".join(sys.argv[1:]))
"""


def test_gen_job_kills_a_slow_scene_records_it_and_skips_done_scenes(tmp_path):
    from test_run_probe_episode import SIM_ENV

    wt = tmp_path / "wt"
    (wt / "downstream/n").mkdir(parents=True)
    (wt / "downstream/eval").mkdir(parents=True)
    (wt / "downstream/n/run_n_gen.sh").write_text((ROOT / "n/run_n_gen.sh").read_text())
    (wt / "downstream/n/n_gen.py").write_text(STUB)
    (wt / "downstream/eval/sim_env.sh").write_text(SIM_ENV)
    out = tmp_path / "out"
    (out / "scenes").mkdir(parents=True)
    (out / "scenes/Done_s1002.json").write_text(json.dumps({"complete": True}))
    env = dict(os.environ, N_X0=str(tmp_path / "x0.jsonl"), SCENE_TIMEOUT="3", SOFT_LIMIT="2")
    res = subprocess.run(["bash", str(wt / "downstream/n/run_n_gen.sh"), str(out), "2", "3", "7"],
                         capture_output=True, text=True, env=env, timeout=60)
    assert res.returncode == 0, res.stdout + res.stderr
    calls = [json.loads(x) for x in (out / "calls.jsonl").read_text().splitlines()]
    assert sorted(c[c.index("--task") + 1] for c in calls) == ["Fast", "Slow"]  # Done was skipped
    assert all(c[c.index("--soft-limit") + 1] == "2" for c in calls)
    skipped = json.loads((out / "scenes/Slow_s1000.timeout.json").read_text())
    assert skipped["rc"] == 137 and skipped["task"] == "Slow"
    assert "--per-family 7" in (out / "collect.done").read_text() and (out / "page.done").is_file()
    again = subprocess.run(["bash", str(wt / "downstream/n/run_n_gen.sh"), str(out), "2", "3", "7"],
                           capture_output=True, text=True, env=env, timeout=60)
    assert again.returncode == 0 and len((out / "calls.jsonl").read_text().splitlines()) == 2  # nothing redone
    out2 = tmp_path / "out2"
    (out2 / "scenes").mkdir(parents=True)
    sliced = subprocess.run(["bash", str(wt / "downstream/n/run_n_gen.sh"), str(out2), "2", "3", "7"],
                            capture_output=True, text=True, env=dict(env, SCENE_SLICE="2/2"), timeout=60)
    assert sliced.returncode == 0
    calls2 = [json.loads(x) for x in (out2 / "calls.jsonl").read_text().splitlines()]
    assert [c[c.index("--task") + 1] for c in calls2] == ["Fast"]  # the 2nd of every 2: Fast only


def test_episode_loader_reads_the_smoke_parts_and_n1_instances(tmp_path):
    smoke = tmp_path / "part1.jsonl"
    smoke.write_text(json.dumps({"task": "T", "seed": 1, "hide_robot": True,
                                 "instance": {"scene_id": "T_s1", "item": {"task_type": "projective_relations"}}})
                     + "\n" + json.dumps({"task": "U", "seed": 2, "instance": None}) + "\n")
    n1 = tmp_path / "instances_eval.jsonl"
    n1.write_text(json.dumps({"instance_id": "T_s1_equidistance", "scene_id": "T_s1", "task": "T", "seed": 1,
                              "hide_robot": True, "item": {"task_type": "equidistance"}}) + "\n")
    got = E.load_instances([smoke, n1], 0)
    assert [E.inst_id(i) for i in got] == ["T_s1", "T_s1_equidistance"]
    assert got[0]["hide_robot"] is True and got[0]["task"] == "T" and got[1]["item"]["task_type"] == "equidistance"
    assert len(E.load_instances([smoke, n1], 1)) == 1
