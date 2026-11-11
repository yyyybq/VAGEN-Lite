"""Tests of package tvrdata (CONTRACT_wave2 5.1, 9): vpd/tvr/{tvr_text, kin_env, tasks, bfs, floors, reach,
demos}.py and the shared helper vpd/tests/tvr_fakes.py.

Hermetic: no Unity, no GPU, no network, no rendering.  TVRBench's REAL thor_env / thor_view_env (read-only checkout
lvp.TVR_REPO; those tests skip without it) run on tvr_fakes' fake ai2thor.  Login node, 1 core:
    cd WT && env -u LP_NUM_THREADS OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 nice -n 19 \
        $PY_Q3 -B -m pytest -q -p no:cacheprovider vpd/tests/test_tvr_data.py
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import random
import shutil
import subprocess
import signal
import sys
import threading
import time
import types
from collections import deque
from pathlib import Path

import pytest

from vpd import paths
from vpd.tests import tvr_fakes as F
from vpd.tvr import bfs, demos, floors, kin_env, lvp, pilot_cfg, reach, tasks as T, tvr_text
from vpd.tvr.kin_env import Pose, ReachSet

needs_repo = pytest.mark.skipif(not F.HAVE_REPO, reason="pinned TVRBench checkout not present")
CATS = pilot_cfg.CATEGORIES
DS_DIFF = {"SR-easy": ("ithor", "easy"), "SR-hard": ("ithor", "hard"), "LR-easy": ("procthor", "easy"),
           "LR-hard": ("procthor", "hard")}


@pytest.fixture
def tvb(tmp_path):
    """make(scenes) -> (thor_env, thor_view_env) of the real checkout on the fake ai2thor (scenes rewritable)."""
    if not F.HAVE_REPO:
        pytest.skip("pinned TVRBench checkout not present")
    root = tmp_path / "fake_ai2thor"
    saved = list(sys.path)
    held = {}

    def make(scenes=None):
        F.write_fake_ai2thor(root, scenes or {})
        if "m" not in held:
            held["m"] = F.import_real_tvrbench(root)
        return held["m"]

    make.root = root
    yield make
    F.purge_tvrbench(root)
    sys.path[:] = saved


def pd(x, z, rot, hor, y=0.9):
    return {"position": {"x": x, "y": y, "z": z}, "rotation_y": float(rot), "horizon": float(hor)}


# ================================================================================================================
# tvr_text
# ================================================================================================================
@needs_repo
def test_source_check_on_pinned_repo():
    r = tvr_text.source_check()
    assert r == {"system_prompt_equal": True, "user_text_equal": True, "parse_action_equal": True,
                 "files_sha256_equal": True}
    d = tvr_text.source_check_detail()
    assert d["parse_action"]["corpus_n"] >= 40 and d["parse_action"]["corpus_equal"]
    assert len(tvr_text.SYSTEM_PROMPT) == 942


@needs_repo
def test_source_check_detects_changes(tmp_path):
    repo = tmp_path / "repo"
    for rel in (tvr_text.VIEW_ENV_REL, tvr_text.EVAL_VA_REL):
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(lvp.TVR_REPO / rel, repo / rel)
    r = tvr_text.source_check(repo)
    assert r["system_prompt_equal"] and r["user_text_equal"] and r["parse_action_equal"]
    assert not r["files_sha256_equal"]  # the other pinned files are absent
    p = repo / tvr_text.VIEW_ENV_REL
    p.write_text(p.read_text(encoding="utf-8").replace("Move forward 0.25m", "Move forward 0.5m"), encoding="utf-8")
    q = repo / tvr_text.EVAL_VA_REL
    q.write_text(q.read_text(encoding="utf-8").replace('text.replace("</think>", "")', "text"), encoding="utf-8")
    r = tvr_text.source_check(repo)
    assert not r["system_prompt_equal"] and r["user_text_equal"] and not r["parse_action_equal"]
    assert tvr_text.main(["check", "--repo", str(repo)]) == 1


def test_parsers_and_user_text():
    assert tvr_text.regex_action("Action: Stop") == "Stop"
    for bad in ("Action: stop", "Action: Stop ", " Action: Stop", "Action:Stop", "Action: Jump", "", None):
        assert tvr_text.regex_action(bad) is None
    assert tvr_text.parse_action("</think>\n\naction:  lookup") == "LookUp"
    assert tvr_text.parse_action("Action: Jump") is None and tvr_text.parse_action("") is None
    v = ["MoveAhead", "RotateRight", "RotateLeft", "LookDown", "Stop"]
    s = tvr_text.build_concat_user_text(v)
    assert s.count("<image>") == 2 and s.endswith("Valid actions at this step: " + ", ".join(v))
    assert tvr_text.valid_from_user_text(s) == v
    with pytest.raises(ValueError):
        tvr_text.valid_from_user_text(s.replace("TARGET", "Target"))
    assert tvr_text.ACTION_NAMES == pilot_cfg.ACTIONS and tvr_text.REGEX == pilot_cfg.REGEX


# ================================================================================================================
# kin_env
# ================================================================================================================
def test_kin_basics():
    assert str(kin_env.r2(-0.0)) == "0.0" and kin_env.r2(1.2500001) == 1.25
    p = kin_env.pose_of(pd(-0.0, 0.25, 359.99, -0.0))
    assert p == Pose(0.0, 0.25, 0, 0) and str(p.x) == "0.0"
    assert kin_env.pose_of(pd(1, 2, -45, 30)).rot == 315
    r = ReachSet.from_positions([{"x": -0.0, "y": 0.9, "z": 0.0}, {"x": 0.25, "y": 0.9, "z": 0.0}])
    want = hashlib.sha256("0.00,0.00\n0.25,0.00".encode()).hexdigest()
    assert r.digest == want and r.n == 2 and (0.0, -0.0) in r and (0.5, 0.0) not in r
    assert ReachSet.from_cells([(-0.0, 0.0), (0.25, 0.0)]) == r and reach.digest(
        [{"x": 0.25, "z": 0.0}, {"x": 0.0, "z": 0.0}]) == want
    with pytest.raises(AttributeError):
        r.n = 3
    # the 8 headings and the 4 moves (thor_env.py _MOVE_DELTAS)
    o = Pose(0.0, 0.0, 0, 0)
    assert kin_env.move_target(o, "MoveAhead") == (0.0, 0.25)
    assert kin_env.move_target(o._replace(rot=45), "MoveAhead") == (0.25, 0.25)
    assert kin_env.move_target(o._replace(rot=45), "MoveLeft") == (-0.25, 0.25)
    assert kin_env.move_target(o._replace(rot=45), "MoveRight") == (0.25, -0.25)
    assert kin_env.move_target(o._replace(rot=270), "MoveBack") == (0.25, 0.0)
    assert kin_env.move_target(o._replace(rot=10), "MoveAhead") is None
    free = None
    assert kin_env.step(o, "LookUp", free) == (Pose(0.0, 0.0, 0, -30), True)
    assert kin_env.step(Pose(0.0, 0.0, 0, -30), "LookUp", free) == (Pose(0.0, 0.0, 0, -30), False)
    assert kin_env.step(Pose(0.0, 0.0, 0, 30), "LookDown", free)[1] is False
    assert kin_env.step(o, "RotateLeft", free)[0].rot == 315
    assert kin_env.step(o, "Stop", free) == (o, None) and kin_env.step(o, "InvalidAction", free) == (o, None)
    assert kin_env.valid_actions(Pose(0.0, 0.0, 0, 30), r) == ["MoveRight", "RotateRight", "RotateLeft", "LookUp",
                                                               "Stop"]
    assert kin_env.step(o, "MoveAhead", r) == (o, False)
    assert kin_env.within_one(o, Pose(0.25, 0.0, 0, 0), r) and not kin_env.within_one(o, Pose(0.0, 0.25, 0, 0), r)
    assert kin_env.within_one(o, Pose(0.0, 0.0, 45, 0), r) and not kin_env.within_one(o, Pose(0.0, 0.0, 90, 0), r)


def test_kin_episode_info():
    t = F.fake_task("ithor_X_easy_t00_s00", "X", "ithor", "easy", (0.0, 0.0, 0, 0), (0.25, 0.0, 0, 0))
    ep = kin_env.KinEpisode(t, F.grid_positions([0.0, 0.25], [0.0]))
    o = ep.reset()
    assert o["valid"] == ["MoveRight", "RotateRight", "RotateLeft", "LookUp", "LookDown", "Stop"]
    assert o["obs_str"] == tvr_text.build_concat_user_text(o["valid"]) and o["state"]["position"]["y"] == 0.9
    s = ep.step("InvalidAction")
    assert s["info"] == {"action": None, "action_executed": False, "action_success": None, "success": None,
                         "stopped": False} and not s["done"]
    s = ep.step("MoveAhead")
    assert s["info"]["action_success"] is False and s["pose"] == Pose(0.0, 0.0, 0, 0)
    s = ep.step("MoveRight")
    assert s["info"]["action_success"] is True and s["pose"] == Pose(0.25, 0.0, 0, 0)
    s = ep.step("Stop")
    assert s["done"] and s["info"]["success"] is True and s["info"]["stopped"] and s["info"]["action_success"] is None
    with pytest.raises(RuntimeError):
        ep.step("Stop")


def _kin_scene(seed=0):
    rng = random.Random(seed)
    xs = [round(-1.0 + 0.25 * i, 2) for i in range(10)]
    blocked = {(rng.choice(xs), rng.choice(xs)) for _ in range(18)}
    return xs, F.grid_positions(xs, xs, blocked)


@needs_repo
def test_kin_env_equals_real_thorenv_2000_sequences(tvb):
    """kin_env vs TVRBench's REAL ThorEnv on the fake ai2thor: poses, success flags, valid lists, user texts."""
    xs, pos = _kin_scene(0)
    te, tv = tvb({"KinGrid": {"positions": pos}})
    thor = lvp.make_thor_env(te, environ={})
    thor.load_scene("KinGrid")
    reach_set = ReachSet.from_positions(pos)
    rng = random.Random(1)
    span = [round(-1.25 + 0.25 * i, 2) for i in range(12)]  # includes cells outside the set
    n_steps = n_fail = n_diag = 0
    for _ in range(2000):
        x, z = rng.choice(span), rng.choice(span)
        rot, hor = rng.choice(range(0, 360, 45)), rng.choice((-30, 0, 30))
        thor.reset(position={"x": x, "y": 0.9, "z": z}, rotation_y=float(rot), horizon=float(hor))
        p = Pose(kin_env.r2(x), kin_env.r2(z), rot, hor)
        assert kin_env.pose_of(thor.get_state()) == p
        for _ in range(rng.randint(3, 20)):
            tv_valid = thor.get_valid_actions(include_stop=True)
            k_valid = kin_env.valid_actions(p, reach_set)
            assert tv_valid == k_valid
            assert tv.build_concat_user_text(tv_valid) == tvr_text.build_concat_user_text(k_valid)
            a = rng.choice(pilot_cfg.NON_STOP)
            _, ok = thor.step(a)
            p, ok2 = kin_env.step(p, a, reach_set)
            assert ok == ok2 and kin_env.pose_of(thor.get_state()) == p
            n_steps += 1
            n_fail += not ok
            n_diag += p.rot % 90 == 45
    assert n_steps > 20000 and n_fail > 1000 and n_diag > 5000


@needs_repo
def test_kin_episode_equals_thorviewenv_with_injection(tvb, tmp_path):
    """Full episodes through ThorViewEnv._sync_reset / _sync_step with the frozen set injected (the worker's reset
    procedure) vs KinEpisode: obs_str, poses, info fields, Stop success, InvalidAction."""
    xs, pos = _kin_scene(3)
    other = F.grid_positions(xs, xs)  # the live set differs from the frozen one
    te, tv = tvb({"FloorPlan901": {"positions": other}})
    env = lvp.make_view_env(tv, procthor_dir=tmp_path, environ={})
    rng = random.Random(2)
    cells = [(p["x"], p["z"]) for p in pos]
    n_stop_ok = 0
    for i in range(150):
        s, g = rng.choice(cells), rng.choice(cells)
        task = F.fake_task(f"ithor_FloorPlan901_easy_t{i:02d}_s00", "FloorPlan901", "ithor", "easy",
                           (s[0], s[1], rng.choice(range(0, 360, 45)), 0), (g[0], g[1], 0, 0))
        env._sync_reset(task)
        thor = env._env
        thor._reachable_positions = list(pos)
        thor._reachable_set = None
        assert ReachSet.from_cells(thor._get_reachable_set()).digest == ReachSet.from_positions(pos).digest
        obs = env._build_obs()
        ep = kin_env.KinEpisode(task, pos)
        k = ep.reset()
        assert obs["obs_str"] == k["obs_str"]
        for t in range(25):
            a = rng.choice(pilot_cfg.ACTIONS[:-1] * 3 + ("InvalidAction", "Stop"))
            if t == 24:
                a = "Stop"
            obs, _, done, info = env._sync_step(f"Action: {a}")
            k = ep.step(a)
            assert obs["obs_str"] == k["obs_str"]
            assert kin_env.pose_of(thor.get_state()) == k["pose"]
            ki = k["info"]
            assert info.get("action") == ki["action"] and info["action_executed"] == ki["action_executed"]
            assert info.get("action_success") == ki["action_success"]
            assert info.get("success") == ki["success"] and bool(info.get("stopped")) == ki["stopped"]
            if a == "Stop":
                assert done and k["done"]
                n_stop_ok += bool(ki["success"])
                break
    env._sync_close()


@needs_repo
def test_reach_is_pose_dependent_and_cached_like_thorenv(tvb, tmp_path):
    """F1/F2 on the fake: the official set is computed at the first task's start after a load and reused for the
    next task of the scene; clearing the cache at a start pose gives that pose's set."""
    full = F.grid_positions([0.0, 0.25, 0.5, 0.75], [0.0, 0.25, 0.5])
    part = [p for p in full if p["x"] <= 0.25]
    te, tv = tvb({"FloorPlan902": {"positions": full, "by_start": {F.start_key(0.0, 0.0): part}}})
    env = lvp.make_view_env(tv, procthor_dir=tmp_path, environ={})
    t1 = F.fake_task("ithor_FloorPlan902_easy_t00_s00", "FloorPlan902", "ithor", "easy", (0.0, 0.0, 0, 0),
                     (0.25, 0.25, 0, 0))
    t2 = F.fake_task("ithor_FloorPlan902_easy_t01_s00", "FloorPlan902", "ithor", "easy", (0.5, 0.5, 0, 0),
                     (0.75, 0.5, 0, 0))
    env._sync_reset(t1)
    assert ReachSet.from_cells(env._env._get_reachable_set()) == ReachSet.from_positions(part)
    env._sync_reset(t2)  # same scene: no reload, ThorEnv keeps the previous task's set
    assert ReachSet.from_cells(env._env._get_reachable_set()) == ReachSet.from_positions(part)
    env._env._reachable_positions = env._env._reachable_set = None
    assert ReachSet.from_positions(env._env.get_reachable_positions()) == ReachSet.from_positions(full)
    env._sync_close()


# ================================================================================================================
# tasks: dev selection, public sequences, 0W / tiers
# ================================================================================================================
def _rl_tasks(n_scenes=6, per_scene=5):
    out = []
    for cat in CATS:
        ds, diff = DS_DIFF[cat]
        for s in range(n_scenes):
            scene = f"procthor_{1000 + s:05d}" if ds == "procthor" else f"FloorPlan{300 + s}"
            for j in range(per_scene):
                out.append(F.fake_task(f"{ds}_{scene}_{diff}_t{j:02d}_s00", scene, ds, diff, (0, 0, 0, 0),
                                       (0.25, 0, 0, 0)))
    return out


def test_select_dev_rule():
    rl = _rl_tasks()
    ids = T.select_dev(rl, per_cat=8, max_per_scene_cat=2, salt="tvr-dev-v1|")
    by = T.by_id(rl)
    assert len(ids) == 32 and len(set(ids)) == 32
    for c, block in zip(CATS, [ids[i:i + 8] for i in range(0, 32, 8)]):
        assert all(pilot_cfg.category_of(by[i]) == c for i in block)
        keys = [T.hash_key("tvr-dev-v1|", i) for i in block]
        assert keys == sorted(keys)
        cnt = {}
        for i in block:
            cnt[by[i]["scene"]] = cnt.get(by[i]["scene"], 0) + 1
        assert max(cnt.values()) <= 2
        # independent recount of the rule: walk the category in hash order with the skip rule
        pool = sorted((t for t in rl if pilot_cfg.category_of(t) == c), key=lambda t: T.hash_key("tvr-dev-v1|",
                                                                                                  t["task_id"]))
        seen, want = {}, []
        for t in pool:
            if len(want) == 8:
                break
            if seen.get(t["scene"], 0) < 2:
                seen[t["scene"]] = seen.get(t["scene"], 0) + 1
                want.append(t["task_id"])
        assert block == want
    with pytest.raises(ValueError):
        T.select_dev(rl, per_cat=13, max_per_scene_cat=2)
    with pytest.raises(ValueError):
        T.select_dev(rl, per_cat=8, forbidden_scenes={by[ids[0]]["scene"]})
    assert T.hash_key("a|", "b") == hashlib.sha256(b"a|b").hexdigest()


def test_dev_select_cli_idempotent_and_refusal(tmp_path):
    rl = _rl_tasks()
    sft = [F.fake_task("ithor_FloorPlan1_easy_t00_s00", "FloorPlan1", "ithor", "easy", (0, 0, 0, 0), (0.25, 0, 0, 0))]
    tree = F.write_fake_tvr_tree(tmp_path / "tree", {"sft": sft, "eval": [], "rl": rl}, [])
    out = tmp_path / "tvr" / "split" / "dev200"
    args = ["dev-select", "--out", str(out), "--tasks-dir", str(tree["tasks_dir"]), "--per-category", "8"]
    assert T.main(args) == 0
    p = out / "selection.json"
    sel = json.loads(p.read_text())
    assert sel["schema"] == "vpd.tvr.dev/1" and len(sel["ids"]) == 32 and sel["salt"] == "tvr-dev-v1|"
    assert sel["rl_json_sha256"] == hashlib.sha256((tree["tasks_dir"] / "rl.json").read_bytes()).hexdigest()
    assert sel["smoke8"] == [i for c in CATS for i in sel["by_category"][c][:2]]
    assert sel["smoke4"] == [sel["by_category"][c][0] for c in CATS]
    assert sum(sel["scenes"].values()) == 32 and set(sel) >= {"rule", "created_utc", "code_sha256",
                                                              "per_category", "max_per_scene_category"}
    before = (p.read_bytes(), p.stat().st_mtime_ns)
    assert T.main(args) == 0
    assert (p.read_bytes(), p.stat().st_mtime_ns) == before
    ts = T.task_set("dev200", root=tmp_path / "tvr", tasks_dir=tree["tasks_dir"])
    assert [t["task_id"] for t in ts] == sel["ids"]
    assert [t["task_id"] for t in T.task_set("smoke4", tmp_path / "tvr", tree["tasks_dir"])] == sel["smoke4"]
    rl2 = [t for t in rl if t["task_id"] != sel["ids"][0]]
    (tree["tasks_dir"] / "rl.json").write_text(json.dumps(rl2))
    assert T.main(args) == 2 and p.read_bytes() == before[0]
    # an eval scene among the chosen ones is refused
    (tree["tasks_dir"] / "rl.json").write_text(json.dumps(rl))
    chosen_scene = T.by_id(rl)[sel["ids"][0]]["scene"]
    (tree["tasks_dir"] / "eval.json").write_text(json.dumps([dict(rl[0], task_id="ithor_e_easy_t00_s00",
                                                                  scene=chosen_scene)]))
    assert T.main(["dev-select", "--out", str(tmp_path / "other"), "--tasks-dir", str(tree["tasks_dir"]),
                   "--per-category", "8"]) == 2
    with pytest.raises(ValueError):
        T.task_set("dev100")
    assert T.parse_sets("eval500,dev200") == ["eval500", "dev200"]


def _sft_one(tid, scene, ds, diff, start, target):
    return F.fake_task(tid, scene, ds, diff, start, target)


def test_public_sequences_assertions(tmp_path):
    t1 = _sft_one("ithor_FloorPlan1_easy_t00_s00", "FloorPlan1", "ithor", "easy", (0, 0, 0, 0), (0.25, 0.25, 45, 0))
    t2 = _sft_one("procthor_procthor_00002_hard_t00_s00", "procthor_00002", "procthor", "hard", (0, 0, 90, 0),
                  (0.5, 0, 90, 30))
    good = [F.public_record(t1, ["RotateRight", "MoveAhead", "Stop"]),
            F.public_record(t2, ["MoveAhead", "MoveAhead", "LookDown", "Stop"])]
    tree = F.write_fake_tvr_tree(tmp_path / "t", {"sft": [t1, t2]}, good)
    pub = T.public_sequences(tree["sft_train"], tasks_dir=tree["tasks_dir"])
    assert list(pub) == [t1["task_id"], t2["task_id"]]
    assert pub[t1["task_id"]]["actions"] == ["RotateRight", "MoveAhead", "Stop"]
    assert pub[t1["task_id"]]["n_turns"] == 3 and pub[t1["task_id"]]["n_images"] == 6
    assert pub[t1["task_id"]]["valid_lines"][0] == kin_env.valid_actions(Pose(0.0, 0.0, 0, 0), None)
    bad_cases = []

    def mut(fn):
        recs = json.loads(json.dumps(good))
        fn(recs)
        bad_cases.append(recs)

    mut(lambda r: r.append(r[0]))                                            # duplicate id
    mut(lambda r: r[0].__setitem__("system", r[0]["system"] + " "))           # system prompt
    mut(lambda r: r[0]["conversations"][1].__setitem__("value", "Action: Jump"))
    mut(lambda r: r[0]["conversations"][1].__setitem__("value", "Action: Stop"))  # two Stops
    mut(lambda r: r[0]["conversations"][5].__setitem__("value", "Action: MoveBack"))  # no Stop
    mut(lambda r: r[0]["images"].pop())                                      # image count
    mut(lambda r: r[0]["images"].reverse())                                  # image order
    mut(lambda r: r[0]["conversations"][0].__setitem__("value", "Valid actions at this step: Stop"))
    mut(lambda r: r[0]["conversations"].reverse())                           # turn order
    mut(lambda r: r[0].__setitem__("images", [im.replace("FloorPlan1_", "FloorPlan9_")
                                               for im in r[0]["images"]]))  # not an sft task
    for k, recs in enumerate(bad_cases):
        p = tmp_path / f"bad{k}.json"
        p.write_text(json.dumps(recs))
        with pytest.raises(ValueError):
            T.public_sequences(p, tasks_dir=tree["tasks_dir"])


def test_expert_medians_median_low():
    tb = {}
    pub = {}
    for c, lens in zip(CATS, ([3, 5, 7, 9], [2], [10, 12], [4, 4, 6])):
        ds, diff = DS_DIFF[c]
        for j, n in enumerate(lens):
            tid = f"{c}_{j}"
            tb[tid] = {"dataset": ds, "difficulty": diff}
            pub[tid] = {"actions": ["MoveAhead"] * n + ["Stop"]}
    assert T.expert_medians(pub, tb) == {"SR-easy": 5, "SR-hard": 2, "LR-easy": 10, "LR-hard": 4}


def _demo_line(tid, cat, n_turns):
    ds, _ = DS_DIFF[cat]
    rec = {"id": tid, "category": cat, "dataset": ds, "scene": "S", "n_turns": n_turns,
           "messages": [{"role": "system", "content": tvr_text.SYSTEM_PROMPT}], "images": []}
    return json.dumps(rec) + "\n"


def _tier_tree(tmp_path, per_cat=30):
    sft, lines = [], []
    for c in CATS:
        ds, diff = DS_DIFF[c]
        for j in range(per_cat):
            tid = f"{ds}_S{c[:2]}_{diff}_t{j:02d}_s00"
            sft.append(F.fake_task(tid, f"S{c[:2]}", ds, diff, (0, 0, 0, 0), (0.25, 0, 0, 0)))
            lines.append(_demo_line(tid, c, 3 + (j * 7) % 11))
    tree = F.write_fake_tvr_tree(tmp_path / "tree", {"sft": sft}, [])
    demos_p = tmp_path / "demos_all.jsonl"
    demos_p.write_text("".join(lines))
    return tree, demos_p, sft, lines


def _mark_demos_done(demos_p, sha=None):
    """What a completed demos build leaves next to demos_all.jsonl: manifest.json (demos_all_sha256) and DONE."""
    sha = hashlib.sha256(demos_p.read_bytes()).hexdigest() if sha is None else sha
    T.write_json(demos_p.parent / "manifest.json", {"schema": "vpd.tvr.demos/1", "demos_all_sha256": sha})
    (demos_p.parent / "DONE").write_text("x\n")
    return hashlib.sha256((demos_p.parent / "manifest.json").read_bytes()).hexdigest()


def test_zero_w_and_tiers(tmp_path):
    tree, demos_p, sft, lines = _tier_tree(tmp_path)
    by = T.by_id(sft)
    kept = [t["task_id"] for t in sft]
    zw = T.zero_w_pool(kept, by)
    assert len(zw) == 20
    for k, c in enumerate(CATS):
        block = zw[5 * k:5 * k + 5]
        mine = sorted((i for i in kept if pilot_cfg.category_of(by[i]) == c),
                      key=lambda i: T.hash_key("tvr-0w-v1|", i))
        assert block == mine[:5]
    pool = [i for i in kept if i not in zw]
    order = T.tier_order(pool, by)
    assert all(len(order[c]) == 25 for c in CATS)
    t8, t40, tall = T.tier_ids(order, 8), T.tier_ids(order, 40), T.tier_ids(order, "ALL")
    assert set(t8) <= set(t40) <= set(tall) and len(tall) == 100 and not set(tall) & set(zw)
    assert [pilot_cfg.category_of(by[i]) for i in t8] == [c for c in CATS for _ in range(2)]
    for bad in (6, 0, "100", True):
        with pytest.raises(ValueError):
            T.tier_ids(order, bad)
    with pytest.raises(ValueError):
        T.tier_ids(order, 104)
    with pytest.raises(ValueError):
        T.zero_w_pool(kept[:3], by)

    out = tmp_path / "tiers"
    args = ["tiers", "--demos", str(demos_p), "--out", str(out), "--tasks-dir", str(tree["tasks_dir"]),
            "--sizes", "8,40"]
    assert T.main(args) == 2 and not out.exists()  # the demos build did not complete (no DONE)
    _mark_demos_done(demos_p, sha="0" * 64)
    assert T.main(args) == 2 and not out.exists()  # DONE, but manifest.json's demos_all_sha256 differs
    dm_sha = _mark_demos_done(demos_p)
    assert T.main(args) == 0
    man = json.loads((out / "manifest.json").read_text())
    assert man["demos_manifest_sha256"] == dm_sha
    assert man["nested"] and man["disjoint_from_0w"] and man["kept"] == 120 and not man["default_sizes"]
    names = ("warmup0w20.jsonl", "demos_8.jsonl", "demos_40.jsonl", "demos_ALL.jsonl", "parity3.jsonl")
    src_lines = set(demos_p.read_bytes().split(b"\n"))
    for n in names:
        b = (out / n).read_bytes()
        assert man[n] == man["files"][n] == hashlib.sha256(b).hexdigest()
        assert all(ln in src_lines for ln in b.split(b"\n"))  # byte-for-byte copies
    assert man["counts"]["demos_40.jsonl"] == {c: 10 for c in CATS}
    assert man["counts"]["warmup0w20.jsonl"] == {c: 5 for c in CATS}
    od = json.loads((out / "order.json").read_text())
    assert od["zero_w"] == zw and od["tiers"]["8"] == t8 and od["tiers"]["ALL"] == tall and od["pool"] == order
    p3 = [json.loads(ln)["id"] for ln in (out / "parity3.jsonl").read_text().splitlines()]
    nt = {json.loads(ln)["id"]: json.loads(ln)["n_turns"] for ln in lines}
    longest = sorted(nt, key=lambda i: (-nt[i], i))[0]
    assert p3 == [longest, order["SR-hard"][0] if order["SR-hard"][0] != longest else order["SR-hard"][1],
                  order["LR-easy"][0] if order["LR-easy"][0] != longest else order["LR-easy"][1]]
    mt = (out / "manifest.json").stat().st_mtime_ns
    assert T.main(args) == 0 and (out / "manifest.json").stat().st_mtime_ns == mt  # identical: no-op
    demos_p.write_text("".join(lines[1:]))
    _mark_demos_done(demos_p)
    assert T.main(args) == 2  # a different tier set is refused


# ================================================================================================================
# bfs
# ================================================================================================================
def _slow_bfs(start, target, r):
    seen, q = {start}, deque([(start, 0)])
    while q:
        p, d = q.popleft()
        if kin_env.is_exact(p, target):
            return d
        for a in pilot_cfg.NON_STOP:
            nq, ok = kin_env.step(p, a, r)
            if ok and nq not in seen:
                seen.add(nq)
                q.append((nq, d + 1))
    return None


def test_bfs_hand_made():
    xs = [0.0, 0.25, 0.5, 0.75, 1.0]
    r = ReachSet.from_positions(F.grid_positions(xs, xs))
    o = Pose(0.0, 0.0, 0, 0)
    assert bfs.bfs_opt(o, o, r) == 0
    assert bfs.bfs_opt(o, Pose(0.25, 0.25, 45, 0), r) == 2          # RotateRight + diagonal MoveAhead
    assert bfs.bfs_opt(o, Pose(0.5, 0.0, 0, 30), r) == 3            # 2 x MoveRight + LookDown
    assert bfs.bfs_opt(o, Pose(1.0, 1.0, 45, 0), r) == 5            # RotateRight + 4 diagonal moves
    assert bfs.bfs_opt(o, Pose(0.0, 0.0, 180, 0), r) == 4
    assert bfs.bfs_opt(o, Pose(2.0, 0.0, 0, 0), r) is None          # target cell outside the set
    wall = ReachSet.from_positions(F.grid_positions(xs, xs, blocked=[(0.5, z) for z in xs]))
    assert bfs.bfs_opt(o, Pose(1.0, 0.0, 0, 0), wall) is None       # disconnected component
    gap = ReachSet.from_positions(F.grid_positions(xs, xs, blocked=[(0.5, z) for z in xs[:-1]]))
    assert bfs.bfs_opt(o, Pose(1.0, 0.0, 0, 0), gap) == _slow_bfs(o, Pose(1.0, 0.0, 0, 0), gap)
    out_start = Pose(-0.25, 0.0, 90, 0)                              # a start outside the set still moves in
    assert bfs.bfs_opt(out_start, Pose(0.0, 0.0, 90, 0), r) == 1
    with pytest.raises(RuntimeError):
        bfs.bfs_opt(o, Pose(1.0, 1.0, 0, 30), r, max_nodes=10)


def test_bfs_equals_slow_kin_bfs_on_random_grids():
    rng = random.Random(7)
    for _ in range(60):
        xs = [round(0.25 * i, 2) for i in range(rng.randint(3, 6))]
        blocked = {(rng.choice(xs), rng.choice(xs)) for _ in range(rng.randint(0, 6))}
        pos = F.grid_positions(xs, xs, blocked)
        if not pos:
            continue
        r = ReachSet.from_positions(pos)
        c = [(p["x"], p["z"]) for p in pos]
        s = Pose(*rng.choice(c), rng.choice(range(0, 360, 45)), rng.choice((-30, 0, 30)))
        g = Pose(*rng.choice(c + [(9.0, 9.0)]), rng.choice(range(0, 360, 45)), rng.choice((-30, 0, 30)))
        assert bfs.bfs_opt(s, g, r) == _slow_bfs(s, g, r)


# ================================================================================================================
# a reach index written directly (for bfs / floors / demos tests)
# ================================================================================================================
def write_index_direct(out: Path, by_set, positions_of):
    by_scene = {}
    for ts in by_set.values():
        for t in ts:
            by_scene.setdefault(t["scene"], {})[t["task_id"]] = t
    for s, tk in by_scene.items():
        ids = sorted(tk)
        sets, tasks_doc = {}, {}
        for i in ids:
            pos = positions_of(tk[i])
            d = reach.digest(pos)
            sets[d] = pos
            e = {"digest": d, "n": ReachSet.from_positions(pos).n, "call_pose": None, "reach_s": 0.0}
            tasks_doc[i] = {"A": e, "B": dict(e), "equal_AB": True}
        doc = {"schema": reach.SCHEMA, "scene": s, "kind": lvp.scene_kind(s),
               "passes": [{"pass": "A", "wid": 0, "load_s": 0.0, "order": ids, "attempt": 1},
                          {"pass": "B", "wid": 1, "load_s": 0.0, "order": ids[::-1], "attempt": 1}],
               "tasks": tasks_doc, "sets": sets, "stable": True}
        T.write_json(out / "scenes" / f"{s}.json", doc)
    return reach.write_index(by_set, out)


def test_bfs_table_cli(tmp_path):
    xs = [0.0, 0.25, 0.5]
    pos = F.grid_positions(xs, xs)
    ev = [F.fake_task("ithor_FloorPlan5_easy_t00_s00", "FloorPlan5", "ithor", "easy", (0, 0, 0, 0), (0.5, 0.5, 45, 0)),
          F.fake_task("ithor_FloorPlan5_easy_t01_s00", "FloorPlan5", "ithor", "easy", (0, 0, 0, 0), (2.0, 0, 0, 0))]
    tree = F.write_fake_tvr_tree(tmp_path / "tree", {"eval": ev}, [])
    write_index_direct(tmp_path / "reach", {"eval500": ev}, lambda t: pos)
    args = ["table", "--task-sets", "eval500", "--reach", str(tmp_path / "reach"), "--out", str(tmp_path / "bfs"),
            "--tasks-dir", str(tree["tasks_dir"])]
    assert bfs.main(args) == 0
    p = tmp_path / "bfs" / "eval500_opt.json"
    doc = json.loads(p.read_text())
    idx = reach.ReachIndex(tmp_path / "reach")
    assert doc["schema"] == "vpd.tvr.bfs/1" and doc["reach_index_sha256"] == idx.sha256 and doc["n_infeasible"] == 1
    assert doc["tasks"]["ithor_FloorPlan5_easy_t00_s00"] == {"feasible": True, "opt": 3, "start_in_reach": True,
                                                            "target_in_reach": True}
    assert doc["tasks"]["ithor_FloorPlan5_easy_t01_s00"]["opt"] is None
    mt = p.stat().st_mtime_ns
    assert bfs.main(args) == 0 and p.stat().st_mtime_ns == mt  # same content: kept
    # another code fingerprint (e.g. after the orchestrator's pilot_cfg fill) does not rewrite identical data
    T.write_json(p, dict(doc, code_sha256="x" * 64))
    b = p.read_bytes()
    assert bfs.main(args) == 0 and p.read_bytes() == b
    T.write_json(p, dict(doc, n_infeasible=7))
    assert bfs.main(args) == 0 and json.loads(p.read_text())["n_infeasible"] == 1  # other data: rewritten


# ================================================================================================================
# floors
# ================================================================================================================
class _Idx:
    def __init__(self, pos_of):
        self.pos_of, self.sha256 = pos_of, "f" * 64

    def reach(self, tid):
        return ReachSet.from_positions(self.pos_of(tid))


def test_fixed_policy_cap_rule_and_ties():
    line = F.grid_positions([0.0], [round(0.25 * i, 2) for i in range(41)])
    far = round(0.25 * 30, 2)
    ti = F.fake_task("ithor_L_easy_t00_s00", "L", "ithor", "easy", (0.0, 0.0, 0, 0), (0.0, far, 0, 0))
    tp = F.fake_task("procthor_procthor_00001_easy_t00_s00", "procthor_00001", "procthor", "easy", (0.0, 0.0, 0, 0),
                     (0.0, far, 0, 0))
    r = ReachSet.from_positions(line)
    assert floors.fixed_hits(ti, r)["MoveAhead"] == []        # 30 moves + Stop = 31 replies > cap 30: never stops
    assert floors.fixed_hits(tp, r)["MoveAhead"] == [30]      # cap 40
    idx = _Idx(lambda tid: line)
    tab = floors.fixed_table([ti, tp], idx)
    assert tab["MoveAhead:30"] == 1 and len(tab) == 8 * 40
    t = {f"{a}:{k}": 0 for a in pilot_cfg.NON_STOP for k in range(40)}
    t["MoveAhead:3"] = t["RotateRight:2"] = 5
    assert floors.choose_fixed(t) == ("RotateRight", 2, 5)    # tie -> smaller k
    t["MoveBack:2"] = 5
    assert floors.choose_fixed(t) == ("MoveBack", 2, 5)       # same k -> ACTIONS order
    zero = {k: 0 for k in t}
    assert floors.choose_fixed(zero) == ("MoveAhead", 0, 0)
    assert floors.fixed_score([ti, tp], idx, "MoveAhead", 30) == 1


def test_random_floor_reproducible_and_independent():
    xs = [0.0, 0.25, 0.5]
    pos = F.grid_positions(xs, xs)
    r = ReachSet.from_positions(pos)
    t = F.fake_task("ithor_R_easy_t00_s00", "R", "ithor", "easy", (0.0, 0.0, 0, 0), (0.25, 0.0, 0, 0))

    def mine(seed, m):  # an independent transcription of the registered policy
        rng = random.Random(f"tvr-floor-rand-v1|{seed}|{t['task_id']}")
        p = Pose(0.0, 0.0, 0, 0)
        for _ in range(m):
            ns = [v for v in kin_env.valid_actions(p, r) if v != "Stop"]
            p = kin_env.step(p, ns[rng.randrange(len(ns))], r)[0]
        return p == Pose(0.25, 0.0, 0, 0)

    for seed in range(40):
        for m in (1, 2, 3):
            assert floors.random_success(t, r, m, seed) == mine(seed, m)
    assert any(floors.random_success(t, r, 1, s) for s in range(40))
    assert not floors.random_success(t, r, 30, 0)              # 30 + Stop > cap 30


def test_floors_doc_and_cli(tmp_path):
    xs = [0.0, 0.25, 0.5]
    pos = F.grid_positions(xs, xs)
    sft, ev = [], []
    for c in CATS:
        ds, diff = DS_DIFF[c]
        sc = "procthor_00007" if ds == "procthor" else "FloorPlan7"
        for j in range(3):
            sft.append(F.fake_task(f"{ds}_{sc}_{diff}_t{j:02d}_s00", sc, ds, diff, (0, 0, 0, 0), (0.25, 0, 90, 0)))
        ev.append(F.fake_task(f"{ds}_{sc}_{diff}_t{50:02d}_s00", sc, ds, diff, (0, 0, 0, 0), (0.0, 0.0, 90, 0)))
    public = [{"task_id": t["task_id"], "actions": ["MoveRight", "RotateRight", "Stop"], "positions": pos}
              for t in sft]
    tree = F.write_fake_tvr_tree(tmp_path / "tree", {"sft": sft, "eval": ev}, public)
    write_index_direct(tmp_path / "reach", {"sft1600": sft, "eval500": ev}, lambda t: pos)
    args = ["--task-sets", "eval500", "--reach", str(tmp_path / "reach"), "--out", str(tmp_path / "floors"),
            "--public", str(tree["sft_train"]), "--tasks-dir", str(tree["tasks_dir"])]
    assert floors.main(args) == 0
    p = tmp_path / "floors" / "floors_eval500.json"
    doc = json.loads(p.read_text())
    assert doc["schema"] == "vpd.tvr.floors/1" and doc["n"] == 4 and doc["warnings"] == []
    assert doc["stop0"] == {"k": 0, "n": 4, "rate": 0.0}
    # sft: "RotateRight^2" never reaches (0.25, 0, 90); the best fixed policy is chosen by the tie rule
    # no fixed policy reaches an sft target (a move AND a rotation are needed): all zero -> MoveAhead^0 (tie rule)
    assert doc["fixed"]["sft"] == {"k": 0, "n": 12} and doc["fixed"]["chosen"] == {"action": "MoveAhead", "k": 0}
    assert doc["fixed"]["eval"] == {"k": 0, "n": 4, "rate": 0.0}
    assert doc["fixed"]["eval"]["k"] == floors.fixed_score(ev, reach.ReachIndex(tmp_path / "reach"),
                                                           doc["fixed"]["chosen"]["action"],
                                                           doc["fixed"]["chosen"]["k"])
    assert doc["random"]["medians"] == {c: 2 for c in CATS} and doc["random"]["seeds"] == [0, 1, 2]
    ks, n = doc["random"]["k"], 4
    assert doc["random"]["mean_rate"] == sum(ks) / (3 * n)
    rates = {"stop0": 0.0, "fixed": doc["fixed"]["eval"]["k"] / n, "random": sum(ks) / (3 * n)}
    assert doc["F"]["rate"] == max(rates.values())
    assert doc["F"]["source"] == next(s for s in ("stop0", "fixed", "random") if rates[s] == doc["F"]["rate"])
    import math
    from fractions import Fraction
    assert doc["F"]["k_equiv"] == math.ceil(max(Fraction(doc["fixed"]["eval"]["k"], n), Fraction(sum(ks), 3 * n)) * n)
    mt = p.stat().st_mtime_ns
    assert floors.main(args) == 0 and p.stat().st_mtime_ns == mt  # same content: kept byte-for-byte
    # STOP0 counts a task that starts at its target, with a warning
    ev2 = ev + [F.fake_task("ithor_FloorPlan7_easy_t99_s00", "FloorPlan7", "ithor", "easy", (0, 0, 0, 0),
                            (0, 0, 0, 0))]
    d2 = floors.floors_doc("eval500", ev2, sft, _Idx(lambda tid: pos), {c: 2 for c in CATS})
    assert d2["stop0"]["k"] == 1 and d2["warnings"] and d2["F"]["rate"] >= 0.2


def test_floor_k_equiv_exact():
    xs = [0.0, 0.25]
    pos = F.grid_positions(xs, [0.0])
    ev = [F.fake_task(f"ithor_K_easy_t{j:02d}_s00", "K", "ithor", "easy", (0, 0, 0, 0),
                      (0.0, 0.0, 0, 0) if j < 3 else (0.25, 0, 180, 0)) for j in range(500)]
    d = floors.floors_doc("eval500", ev, ev[:1], _Idx(lambda tid: pos), {c: 1 for c in CATS})
    assert d["stop0"]["k"] == 3 and d["fixed"]["eval"]["k"] == 3 and d["random"]["k"] == [0, 0, 0]
    # ties between stop0 and fixed -> stop0; k_equiv from exact fractions (3/500 * 500 is exactly 3)
    assert d["F"] == {"rate": 3 / 500, "source": "stop0", "k_equiv": 3}


# ================================================================================================================
# reach: houses, dump (LocalHandle pool on the fake), index, check, CLI
# ================================================================================================================
class MiniWorker:
    """The contract's load / reach_task ops (CONTRACT 5.2.1) on TVRBench's real ThorViewEnv (fake ai2thor)."""

    def __init__(self, tv, wid, houses, procthor_dir, flaky=None, fail=None):
        self.tv, self.wid, self.houses, self.flaky, self.fail = tv, wid, houses, flaky, fail
        self.env = lvp.make_view_env(tv, procthor_dir=procthor_dir, environ={})
        self.calls = []

    def handle(self, cmd):
        self.calls.append((cmd["op"], cmd.get("scene") or cmd.get("task", {}).get("task_id")))
        if self.fail is not None and self.fail(self, cmd):
            raise RuntimeError("fake worker failure")
        if cmd["op"] == "load":
            loaded = bool(cmd.get("force")) or self.env._scene != cmd["scene"]
            s = lvp.load_scene(self.env, cmd["scene"], self.houses) if loaded else 0.0
            return {"scene": self.env._scene, "load_s": s, "loaded": loaded}
        if cmd["op"] == "reach_task":
            task = cmd["task"]
            lvp.resolve_scene(self.env, task["scene"], self.houses)
            before = self.env._scene
            self.env._sync_reset(task)
            thor = self.env._env
            thor._reachable_positions = None
            thor._reachable_set = None
            pos = list(thor.get_reachable_positions())
            if self.flaky is not None:
                pos = self.flaky(self, task, pos)
            return {"scene": self.env._scene, "loaded": before != task["scene"], "load_s": 0.0,
                    "call_pose": thor.get_state(), "positions": pos, "n": len(pos), "digest": reach.digest(pos),
                    "reach_s": 0.0}
        raise ValueError(f"unknown op {cmd['op']}")

    def close(self):
        self.env._sync_close()


class MiniPool:
    """thor_worker.ThorPool's API (CONTRACT 5.2.1): call never raises; restart RETURNS the ready reply (ok False
    when the new start-up failed)."""

    def __init__(self, factory, n):
        from vpd.tvr import probe_lvp

        self.factory, self.LocalHandle = factory, probe_lvp.LocalHandle
        self.w = {}
        for wid in range(n):
            self._spawn(wid)
        self.restarts = []
        self.closed = None
        self.is_closed = False

    def _spawn(self, wid):
        wk = self.factory(wid)
        self.w[wid] = (wk, self.LocalHandle(wid, 2, wk))

    def call(self, wid, cmd, timeout):
        if self.is_closed:
            return {"ok": False, "error": "pool closed", "died": True}
        h = self.w[wid][1]
        return h.wait(h.send(cmd), timeout)

    def ready(self):
        return [{"wid": wid, "ok": True} for wid in sorted(self.w)]

    def restart(self, wid):
        self.restarts.append(wid)
        if self.is_closed:
            return {"ok": False, "error": "pool closed"}
        self._spawn(wid)
        return {"ok": True, "wid": wid}

    def close(self, fast=False):
        self.closed = fast
        self.is_closed = True


def _reach_world(tmp_path, tvb):
    full = F.grid_positions([round(0.25 * i, 2) for i in range(6)], [round(0.25 * i, 2) for i in range(6)])
    part = [p for p in full if p["z"] <= 0.75]
    sft = [F.fake_task(f"ithor_FloorPlan21_easy_t{j:02d}_s00", "FloorPlan21", "ithor", "easy",
                       (0.25 * j, 0.0, 0, 0), (0.25 * j, 0.5, 0, 0)) for j in range(4)]
    sft += [F.fake_task(f"procthor_procthor_00004_hard_t{j:02d}_s00", "procthor_00004", "procthor", "hard",
                        (0.5, 0.25 * j, 90, 0), (1.0, 0.25 * j, 90, 0)) for j in range(3)]
    ev = [F.fake_task("ithor_FloorPlan22_easy_t00_s00", "FloorPlan22", "ithor", "easy", (0.0, 0.0, 0, 0),
                      (0.0, 0.25, 0, 0))]
    scenes = {"FloorPlan21": {"positions": full, "by_start": {F.start_key(0.25, 0.0): part}},
              "procthor_00004": {"positions": full}, "FloorPlan22": {"positions": part}}
    te, tv = tvb(scenes)
    tree = F.write_fake_tvr_tree(tmp_path / "tree", {"sft": sft, "eval": ev}, [])
    hd = tmp_path / "houses"
    doc = reach.houses(["sft1600", "eval500"], hd, src=tree["procthor_gz"], tasks_dir=tree["tasks_dir"])
    return types.SimpleNamespace(te=te, tv=tv, tree=tree, houses=hd, houses_doc=doc, sft=sft, ev=ev, full=full,
                                 part=part)


@needs_repo
def test_houses(tmp_path, tvb):
    w = _reach_world(tmp_path, tvb)
    assert list(w.houses_doc["scenes"]) == ["procthor_00004"] and w.houses_doc["scenes"]["procthor_00004"]["line"] == 4
    assert json.loads((w.houses / "procthor_00004.json").read_text())["id"] == "procthor_00004"
    b = (w.houses / "houses.json").read_bytes()
    reach.houses(["sft1600", "eval500"], w.houses, src=w.tree["procthor_gz"], tasks_dir=w.tree["tasks_dir"])
    assert (w.houses / "houses.json").read_bytes() == b
    assert (w.tree["houses_dir"] / "procthor_00004.json").read_text() == (w.houses / "procthor_00004.json").read_text()


@needs_repo
def test_reach_dump_two_passes_resume_and_index(tmp_path, tvb):
    w = _reach_world(tmp_path, tvb)

    def flaky(wk, task, pos):  # worker 1 loses a cell in one scene: pass A / B disagree there
        if wk.wid == 1 and task["scene"] == "procthor_00004" and task["task_id"].endswith("t01_s00"):
            return pos[1:]
        return pos

    pool = MiniPool(lambda wid: MiniWorker(w.tv, wid, w.houses, w.tree["procthor_dir"], flaky=flaky), 2)
    out = tmp_path / "reach" / "v1"
    logs = []
    res = reach.dump(["sft1600", "eval500"], out, pool, tasks_dir=w.tree["tasks_dir"], log=logs.append)
    assert res["complete"] and not res["failed"] and res["index"] and sorted(res["written"]) == sorted(
        ["FloorPlan21", "FloorPlan22", "procthor_00004"])
    assert reach._scene_order({"FloorPlan21": [1] * 4, "procthor_00004": [1] * 3, "FloorPlan22": [1]}) == [
        "procthor_00004", "FloorPlan21", "FloorPlan22"]
    for s in ("FloorPlan21", "FloorPlan22", "procthor_00004"):
        doc = json.loads((out / "scenes" / f"{s}.json").read_text())
        a, b = doc["passes"]
        assert a["pass"] == "A" and b["pass"] == "B" and a["wid"] != b["wid"]
        assert a["order"] == sorted(a["order"]) and b["order"] == a["order"][::-1]
    idx = reach.ReachIndex(out)
    # F2 on the fake: FloorPlan21 t01 starts at (0.25, 0) whose set is the partial one; the others get the full set
    assert idx.reach("ithor_FloorPlan21_easy_t01_s00") == ReachSet.from_positions(w.part)
    assert idx.reach("ithor_FloorPlan21_easy_t00_s00") == ReachSet.from_positions(w.full)
    assert idx.positions("ithor_FloorPlan22_easy_t00_s00") and idx.digest("ithor_FloorPlan22_easy_t00_s00") == \
        reach.digest(w.part)
    pt = json.loads((out / "scenes" / "procthor_00004.json").read_text())
    e = pt["tasks"]["procthor_procthor_00004_hard_t01_s00"]
    assert not e["equal_AB"] and not pt["stable"]
    wid_a = pt["passes"][0]["wid"]
    assert idx.digest("procthor_procthor_00004_hard_t01_s00") == e["A"]["digest"]  # frozen = pass A
    assert (e["A"]["digest"] == reach.digest(w.full[1:])) == (wid_a == 1)
    ix = json.loads((out / "index.json").read_text())
    assert ix["unstable_tasks"] == ["procthor_procthor_00004_hard_t01_s00"]
    assert ix["task_sets"]["sft1600"]["n"] == 7 and ix["task_sets"]["eval500"]["n"] == 1
    assert ix["task_sets"]["eval500"]["ids_sha256"] == T.ids_sha256(["ithor_FloorPlan22_easy_t00_s00"])
    assert (out / "DONE").is_file() and reach.check(out)["complete"] and reach.check(out)["unstable"] == 1
    assert reach.main(["check", "--out", str(out)]) == 0
    # resume: nothing to do, index.json kept byte-for-byte
    b, mt = (out / "index.json").read_bytes(), (out / "index.json").stat().st_mtime_ns
    pool2 = MiniPool(lambda wid: MiniWorker(w.tv, wid, w.houses, w.tree["procthor_dir"]), 2)
    res2 = reach.dump(["sft1600", "eval500"], out, pool2, tasks_dir=w.tree["tasks_dir"], log=logs.append)
    assert res2["complete"] and res2["written"] == [] and len(res2["skipped"]) == 3
    assert all(not wk.calls for wk, _ in pool2.w.values())
    assert (out / "index.json").read_bytes() == b and (out / "index.json").stat().st_mtime_ns == mt
    # one scene file removed -> only that scene is dumped again
    (out / "scenes" / "FloorPlan22.json").unlink()
    pool3 = MiniPool(lambda wid: MiniWorker(w.tv, wid, w.houses, w.tree["procthor_dir"]), 2)
    res3 = reach.dump(["sft1600", "eval500"], out, pool3, tasks_dir=w.tree["tasks_dir"], log=logs.append)
    assert res3["written"] == ["FloorPlan22"]
    assert {c[1] for wk, _ in pool3.w.values() for c in wk.calls} <= {"FloorPlan22", "ithor_FloorPlan22_easy_t00_s00"}
    # a tampered scene file is detected by the index
    p = out / "scenes" / "FloorPlan21.json"
    p.write_text(p.read_text().replace('"stable"', '"stable" ', 1))
    assert not reach.check(out)["complete"] and reach.main(["check", "--out", str(out)]) == 1


@needs_repo
def test_reach_dump_failures_and_deadline(tmp_path, tvb):
    w = _reach_world(tmp_path, tvb)
    once = {"n": 0}

    def fail_once(wk, cmd):
        if cmd["op"] == "reach_task" and cmd["task"]["scene"] == "FloorPlan22" and once["n"] == 0:
            once["n"] += 1
            return True
        return False

    pool = MiniPool(lambda wid: MiniWorker(w.tv, wid, w.houses, w.tree["procthor_dir"], fail=fail_once), 2)
    out = tmp_path / "r1"
    res = reach.dump(["eval500"], out, pool, tasks_dir=w.tree["tasks_dir"], log=lambda *a: None)
    assert res["complete"] and len(pool.restarts) == 1 and res["errors"][0]["attempt"] == 1
    doc = json.loads((out / "scenes" / "FloorPlan22.json").read_text())
    assert sorted(p["attempt"] for p in doc["passes"]) == [1, 2]
    # a worker that always fails: the scene is incomplete, no index
    pool = MiniPool(lambda wid: MiniWorker(w.tv, wid, w.houses, w.tree["procthor_dir"],
                                           fail=lambda wk, c: wk.wid == 0 and c["op"] == "load"), 2)
    out2 = tmp_path / "r2"
    res = reach.dump(["eval500"], out2, pool, tasks_dir=w.tree["tasks_dir"], log=lambda *a: None)
    assert not res["complete"] and res["failed"] == ["FloorPlan22"] and not (out2 / "index.json").exists()
    assert not (out2 / "DONE").exists() and not res["deadline_hit"]

    # a worker whose restart fails (ThorPool.restart RETURNS ok False) is retired after its first failed pass;
    # that pass goes back to the queue and the other workers finish everything
    class NoRestart(MiniPool):
        def restart(self, wid):
            self.restarts.append(wid)
            return {"ok": False, "id": 0, "error": "cannot restart"}

    def w0_fails_load(first):
        def fail(wk, c):
            if wk.wid == 0:
                first.set()
                return c["op"] == "load"
            assert first.wait(30)  # w0 surely takes a pass before the others start
            return False
        return fail

    class RaisingRestart(MiniPool):
        def restart(self, wid):
            self.restarts.append(wid)
            raise RuntimeError("cannot restart")

    for raising in (False, True):  # ThorPool's API, and a stand-in whose restart raises
        cls = RaisingRestart if raising else NoRestart
        ev = threading.Event()
        pool = cls(lambda wid: MiniWorker(w.tv, wid, w.houses, w.tree["procthor_dir"], fail=w0_fails_load(ev)), 3)
        res = reach.dump(["sft1600", "eval500"], tmp_path / f"r5{int(raising)}", pool,
                         tasks_dir=w.tree["tasks_dir"], log=lambda *a: None)
        assert len(pool.w[0][0].calls) == 1 and pool.restarts == [0]  # exactly one pass, one restart attempt
        assert res["complete"] and res["failed"] == [] and len(res["written"]) == 3 and res["retired"] == [0]
        assert len(res["requeued"]) == 1 and res["requeued"][0]["wid"] == 0
        for s in res["written"]:
            doc = json.loads((tmp_path / f"r5{int(raising)}" / "scenes" / f"{s}.json").read_text())
            assert {p["wid"] for p in doc["passes"]} == {1, 2}  # A and B still on different workers
    # a failed restart after a pass's SECOND failure: the scene fails (no requeue), the worker is retired
    class SecondRestartFails(MiniPool):
        def restart(self, wid):
            self.restarts.append(wid)
            if len(self.restarts) == 1:
                self._spawn(wid)
                return {"ok": True, "wid": wid}
            return {"ok": False, "error": "cannot restart"}

    pool = SecondRestartFails(lambda wid: MiniWorker(w.tv, wid, w.houses, w.tree["procthor_dir"],
                                                     fail=lambda wk, c: wk.wid == 0 and c["op"] == "load"), 2)
    res = reach.dump(["eval500"], tmp_path / "r6", pool, tasks_dir=w.tree["tasks_dir"], log=lambda *a: None)
    assert res["failed"] == ["FloorPlan22"] and res["retired"] == [0] and res["requeued"] == []
    assert pool.restarts == [0, 0] and not res["complete"]
    # deadline already reached: nothing starts
    pool = MiniPool(lambda wid: MiniWorker(w.tv, wid, w.houses, w.tree["procthor_dir"]), 2)
    res = reach.dump(["eval500"], tmp_path / "r3", pool, deadline=time.monotonic() + 1.0,
                     tasks_dir=w.tree["tasks_dir"], log=lambda *a: None)
    assert not res["complete"] and res["deadline_hit"] and not (tmp_path / "r3" / "index.json").exists()
    with pytest.raises(ValueError):
        reach.dump(["eval500"], tmp_path / "r4", MiniPool(lambda wid: MiniWorker(
            w.tv, wid, w.houses, w.tree["procthor_dir"]), 1), tasks_dir=w.tree["tasks_dir"], log=lambda *a: None)


@needs_repo
def test_reach_dump_cli_with_stub_pool(tmp_path, tvb, monkeypatch):
    """The dump CLI drives thor_worker.ThorPool (tvreval); here a stand-in module with the contract's API."""
    w = _reach_world(tmp_path, tvb)
    made = {}

    class StubPool(MiniPool):
        def __init__(self, n_envs, lp_threads, profile, *, out_dir, houses_dir, repo=None, py=None,
                     extra_pythonpath=(), **kw):
            made.update(n=n_envs, lp=lp_threads, profile=profile, out_dir=out_dir, extra=extra_pythonpath,
                        pool=self)
            super().__init__(lambda wid: MiniWorker(w.tv, wid, Path(houses_dir), w.tree["procthor_dir"],
                                                    fail=made.get("fail")), n_envs)

        def start(self):
            made["started"] = made.get("started", 0) + 1
            if made.get("start_error"):
                raise RuntimeError("fake: worker 0 did not start twice")  # as ThorPool's PoolError
            return self.ready()

    stub = types.ModuleType("vpd.tvr.thor_worker")
    stub.ThorPool = StubPool
    stub.TIMEOUTS = dict(reach.DEFAULT_TIMEOUTS)
    monkeypatch.setitem(sys.modules, "vpd.tvr.thor_worker", stub)
    import vpd.tvr as pkg
    monkeypatch.setattr(pkg, "thor_worker", stub, raising=False)
    out = tmp_path / "cli"

    def base(o):
        return ["dump", "--task-sets", "sft1600,eval500", "--out", str(o), "--houses", str(w.houses),
                "--tasks-dir", str(w.tree["tasks_dir"])]

    assert reach.main(base(out) + ["--envs", "6", "--lp", "2", "--profile", "envlib"]) == 1  # E x L > 11
    with monkeypatch.context() as m:  # render.* not filled (whatever the real pilot_cfg holds) and not given
        for k in ("lvp_profile", "lp_threads", "replay_envs"):
            m.setitem(pilot_cfg.FROZEN["render"], k, None)
        assert reach.main(base(out)) == 1 and "pool" not in made
    assert reach.main(base(out) + ["--envs", "2", "--lp", "2", "--profile", "envlib", "--fake-thor",
                                   str(tvb.root), "--deadline-s", "600"]) == 0
    assert made["n"] == 2 and made["lp"] == 2 and made["profile"] == "envlib" and made["extra"] == (str(tvb.root),)
    assert made["pool"].closed is False  # complete: normal close
    assert reach.check(out)["complete"] and json.loads((out / "dump_last.json").read_text())["complete"]
    opts = ["--envs", "2", "--lp", "2", "--profile", "envlib"]
    # a rerun over a complete dump starts no worker, so a failing start-up cannot fail the step: rc 0, index kept
    b = (out / "index.json").read_bytes()
    made.update(start_error=True, started=0)
    assert reach.main(base(out) + opts) == 0 and made["started"] == 0
    assert (out / "index.json").read_bytes() == b
    last = json.loads((out / "dump_last.json").read_text())
    assert last["complete"] and last["written"] == [] and len(last["skipped"]) == 3
    (out / "scenes" / "FloorPlan22.json").unlink()  # one scene left: the pool is started (and here fails: rc 1)
    assert reach.main(base(out) + opts) == 1 and made["started"] == 1
    assert not (out / "scenes" / "FloorPlan22.json").exists()
    made["start_error"] = False
    assert reach.main(base(out) + opts) == 0 and made["started"] == 2 and reach.check(out)["complete"]

    # deadline in the middle of a pass: rc 4, fast close, no restart, dump_last.json written
    monkeypatch.setattr(reach, "NO_NEW_PASS_S", 0.3)

    def slow_fail(wk, c):
        if c["op"] == "reach_task":
            time.sleep(1.2)
            return True
        return False

    made["fail"] = slow_fail
    out4 = tmp_path / "cli4"
    assert reach.main(base(out4) + opts + ["--deadline-s", "1.0"]) == 4
    assert made["pool"].closed is True and made["pool"].restarts == []
    last = json.loads((out4 / "dump_last.json").read_text())
    assert last["deadline_hit"] and not last["complete"] and not (out4 / "index.json").exists()

    # SIGTERM during the dump: rc 143, fast close, no index
    def slow(wk, c):
        if c["op"] == "reach_task":
            time.sleep(0.4)
        return False

    made["fail"] = slow
    out5 = tmp_path / "cli5"
    prev = signal.signal(signal.SIGTERM, lambda *a: None)  # a late signal must never reach pytest's handler
    try:
        timer = threading.Timer(0.5, os.kill, (os.getpid(), signal.SIGTERM))
        timer.start()
        t0 = time.monotonic()
        rc = reach.main(base(out5) + opts)
        dt = time.monotonic() - t0
        timer.join()
    finally:
        signal.signal(signal.SIGTERM, prev)
    assert rc == 143 and dt < 2.0 and made["pool"].closed is True and not (out5 / "index.json").exists()
    end = time.monotonic() + 15  # the worker threads see the closed pool, are retired and end
    while any(t.name.startswith("reach-w") for t in threading.enumerate()) and time.monotonic() < end:
        time.sleep(0.1)
    assert not any(t.name.startswith("reach-w") for t in threading.enumerate())


@needs_repo
def test_reach_dump_deadline_mid_pass_no_restart(tmp_path, tvb, monkeypatch):
    """A pass that fails after the deadline (an ordinary error, not 'deadline') is neither restarted nor retried,
    and deadline_hit is set although it was the last item."""
    w = _reach_world(tmp_path, tvb)
    monkeypatch.setattr(reach, "NO_NEW_PASS_S", 0.3)

    def slow_fail(wk, c):
        if c["op"] == "reach_task":
            time.sleep(1.2)
            return True
        return False

    pool = MiniPool(lambda wid: MiniWorker(w.tv, wid, w.houses, w.tree["procthor_dir"], fail=slow_fail), 2)
    t0 = time.monotonic()
    res = reach.dump(["eval500"], tmp_path / "d1", pool, deadline=t0 + 1.0, tasks_dir=w.tree["tasks_dir"],
                     log=lambda *a: None)
    assert res["deadline_hit"] and not res["complete"] and pool.restarts == [] and res["retired"] == []
    assert len(res["errors"]) == 2 and all(e["attempt"] == 1 for e in res["errors"])
    assert time.monotonic() - t0 < 5 and not (tmp_path / "d1" / "index.json").exists()


@needs_repo
def test_reach_dump_restart_near_the_deadline(tmp_path, tvb, monkeypatch):
    """A restart starts only with NO_NEW_PASS_S + the pool's startup_timeout seconds left (else the worker is retired
    and the scene fails, deadline_hit); a restart that runs into the deadline gives no retry and no requeue, and
    deadline_hit is set by that worker itself (the others may have left take() already: rc 4, not 1)."""
    w = _reach_world(tmp_path, tvb)
    monkeypatch.setattr(reach, "NO_NEW_PASS_S", 0.3)
    w0_no_load = lambda wk, c: wk.wid == 0 and c["op"] == "load"  # noqa: E731

    def run(pool, name, deadline_s):
        t0 = time.monotonic()
        res = reach.dump(["eval500"], tmp_path / name, pool, deadline=t0 + deadline_s, tasks_dir=w.tree["tasks_dir"],
                         log=lambda *a: None)
        assert time.monotonic() - t0 < deadline_s + 5
        assert not res["complete"] and res["failed"] == ["FloorPlan22"] and res["deadline_hit"]
        assert res["requeued"] == [] and not (tmp_path / name / "index.json").exists()
        return res

    # 10 s left < 0.3 + startup_timeout 30: the restart is not even tried; worker 0 is retired
    pool = MiniPool(lambda wid: MiniWorker(w.tv, wid, w.houses, w.tree["procthor_dir"], fail=w0_no_load), 2)
    pool.startup_timeout = 30.0
    res = run(pool, "n1", 10.0)
    assert pool.restarts == [] and res["retired"] == [0]
    assert any(e["error"] == reach.NO_TIME and e["wid"] == 0 for e in res["errors"])

    class SlowRestart(MiniPool):  # the restart takes 1.6 s: it ends inside the last NO_NEW_PASS_S seconds
        ok = False

        def restart(self, wid):
            self.restarts.append(wid)
            time.sleep(1.6)
            if not self.ok:
                return {"ok": False, "error": "cannot restart"}
            self._spawn(wid)
            return {"ok": True, "wid": wid}

    for ok in (False, True):
        pool = SlowRestart(lambda wid: MiniWorker(w.tv, wid, w.houses, w.tree["procthor_dir"], fail=w0_no_load), 2)
        pool.ok, pool.startup_timeout = ok, 0.1
        res = run(pool, f"s{int(ok)}", 1.4)
        assert pool.restarts == [0]
        assert res["retired"] == ([] if ok else [0])
        assert pool.w[0][0].calls == ([] if ok else [("load", "FloorPlan22")])  # no retry on the fresh worker


@needs_repo
def test_reach_dump_on_real_thorpool_failed_restart(tmp_path, tvb):
    """reach.dump on tvreval's REAL thor_worker.ThorPool (local_factory Workers on the fake ai2thor): worker 0
    cannot load and its restart fails -> it is retired after one pass, its pass is requeued, the dump completes."""
    w = _reach_world(tmp_path, tvb)
    from vpd.tvr import thor_worker

    font = lvp.install_font_rule(w.tv, allow_missing=True)
    made = {}
    first = threading.Event()

    class BadW0(thor_worker.Worker):
        def handle(self, cmd):
            if self.wid == 0:
                first.set()
                if cmd.get("op") == "load":
                    raise RuntimeError("fake: worker 0 cannot load")
            elif not first.wait(30):
                raise RuntimeError("worker 0 never took a pass")
            return super().handle(cmd)

    def mk(wid):
        made[wid] = made.get(wid, 0) + 1
        if wid == 0 and made[wid] > 1:
            raise RuntimeError("fake: worker 0 cannot restart")
        return BadW0(w.tv, wid=wid, houses_dir=w.houses, procthor_dir=w.tree["procthor_dir"], font=font,
                     environ={"OMP_NUM_THREADS": "1", "LP_NUM_THREADS": "1", "VPD_LVP_PROFILE": "envlib"})

    pool = thor_worker.ThorPool(3, 1, "envlib", out_dir=tmp_path / "pool", houses_dir=w.houses, local_factory=mk)
    pool.start()
    out = tmp_path / "rp"
    try:
        res = reach.dump(["sft1600", "eval500"], out, pool, tasks_dir=w.tree["tasks_dir"], log=lambda *a: None)
    finally:
        pool.close(fast=True)
    assert res["complete"] and res["failed"] == [] and len(res["written"]) == 3
    assert res["retired"] == [0] and made[0] == 2 and len(res["requeued"]) == 1
    assert reach.check(out)["complete"]
    ix = json.loads((out / "index.json").read_text())
    assert ix["call_pose_mismatch"] == [] and all(e["call_pose_ok"] is True for e in ix["tasks"].values())


@needs_repo
@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")
def test_reach_dump_harness_error_on_real_thorpool(tmp_path, tvb):
    """reach.dump on tvreval's REAL thor_worker.ThorPool (local_factory Workers on the fake ai2thor), 3 workers, no
    deadline.  An error in the harness while a pass runs fails that pass and its scene, and the dump RETURNS (take()
    waits while a pass is 'taken': a dead worker thread used to hold its pass forever): (1) a reply whose positions
    lack x (KeyError in reach.digest), (2) a worker thread that ends on a BaseException (SystemExit raised through
    ThorPool.call), (3) a scene file that cannot be written.  A rerun then dumps only the failed scene."""
    w = _reach_world(tmp_path, tvb)
    from vpd.tvr import thor_worker

    font = lvp.install_font_rule(w.tv, allow_missing=True)
    mode = {}

    class BadReply(thor_worker.Worker):
        def op_reach_task(self, cmd):
            out = super().op_reach_task(cmd)
            if cmd["task"]["scene"] == "FloorPlan22":
                if mode.get("exit"):
                    raise SystemExit(3)
                if mode.get("no_x"):
                    out["positions"] = [{k: v for k, v in q.items() if k != "x"} for q in out["positions"]]
            return out

    def mk(wid):
        return BadReply(w.tv, wid=wid, houses_dir=w.houses, procthor_dir=w.tree["procthor_dir"], font=font,
                        environ={"OMP_NUM_THREADS": "1", "LP_NUM_THREADS": "1", "VPD_LVP_PROFILE": "envlib"})

    def run(out, start=True):
        pool = thor_worker.ThorPool(3, 1, "envlib", out_dir=tmp_path / "pool", houses_dir=w.houses, local_factory=mk)
        if start:
            pool.start()
        box = {}
        th = threading.Thread(target=lambda: box.update(res=reach.dump(
            ["sft1600", "eval500"], out, pool, tasks_dir=w.tree["tasks_dir"], log=lambda *a: None)), daemon=True)
        th.start()
        th.join(60)
        pool.close(fast=True)
        assert not th.is_alive(), "reach.dump did not return"
        return box["res"]

    def failed_fp22(res, out, why):
        assert not res["complete"] and res["failed"] == ["FloorPlan22"] and not res["deadline_hit"], res
        assert res["written"] == ["FloorPlan21", "procthor_00004"] and res["index"] is None
        assert res["retired"] == [] and res["requeued"] == []
        assert any(e["scene"] == "FloorPlan22" and why in e["error"] for e in res["errors"]), res["errors"]
        assert not (out / "index.json").exists() and not (out / "DONE").exists()

    out = tmp_path / "h1"
    mode["no_x"] = True
    failed_fp22(run(out), out, "harness: KeyError('x')")
    mode.clear()
    res = run(out)  # resume: only the failed scene is dumped again
    assert res["complete"] and res["written"] == ["FloorPlan22"] and len(res["skipped"]) == 2
    assert reach.check(out)["complete"]
    res = run(out, start=False)  # nothing left: an unstarted pool is enough (cmd_dump starts none then)
    assert res["complete"] and res["written"] == [] and res["index"]["kept_existing_index"]

    out = tmp_path / "h2"
    mode["exit"] = True
    failed_fp22(run(out), out, "the worker thread ended during the pass")
    mode.clear()

    out = tmp_path / "h3"
    (out / "scenes" / "FloorPlan22.json").mkdir(parents=True)  # os.replace onto a directory fails
    failed_fp22(run(out), out, "finish: IsADirectoryError")


# ================================================================================================================
# demos: a fake replay (TVRBench's real ThorViewEnv with the frozen set injected, as the worker does)
# ================================================================================================================
def _png(img, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="PNG")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fake_replay(tv, run, images, sft_by, public, idx, houses, procthor_dir, doctor=None, after_cap_obs=True):
    """after_cap_obs: eval_va's layout (the observation after the dataset cap's last reply is appended: n + 1
    user turns); False: run_tvr's (it stops at the cap: n user turns)."""
    env = lvp.make_view_env(tv, procthor_dir=procthor_dir, environ={})
    (run / "tasks").mkdir(parents=True, exist_ok=True)
    for tid, seq in public.items():
        task = sft_by[tid]
        cap = pilot_cfg.cap_of(task)
        lvp.resolve_scene(env, task["scene"], houses)
        env._sync_reset(task)
        thor = env._env
        thor._reachable_positions = list(idx.positions(tid))
        thor._reachable_set = None
        obs = env._build_obs()
        cur, tgt = obs["multi_modal_input"][tv.IMAGE_PH]
        imgs = [{"cur_sha256": _png(cur, images / tid / "step_00.png"), "cur_path": str(images / tid / "step_00.png")}]
        tsha = _png(tgt, images / tid / "target.png")
        states, texts = [thor.get_state()], [obs["obs_str"]]
        valid, succ, acts = [tvr_text.valid_from_user_text(obs["obs_str"])], [], []
        result = {"success": False, "stopped": False}
        for k, a in enumerate(seq["actions"]):
            if k >= cap:
                break
            obs, _, done, info = env._sync_step(f"Action: {a}")
            acts.append(a)
            succ.append(info.get("action_success"))
            states.append(thor.get_state())
            if a == "Stop":
                result = {"success": bool(info["success"]), "stopped": True}
                break
            if done:  # the env's own cap (40): eval_va appends no further observation
                break
            if k == cap - 1 and not after_cap_obs:
                break
            # as eval_va: the next observation is appended even after the dataset cap's last reply (iTHOR 30)
            c = obs["multi_modal_input"][tv.IMAGE_PH][0]
            pth = images / tid / f"step_{k + 1:02d}.png"
            imgs.append({"cur_sha256": _png(c, pth), "cur_path": str(pth)})
            texts.append(obs["obs_str"])
            valid.append(tvr_text.valid_from_user_text(obs["obs_str"]))
        traj = {"task_id": tid, "dataset": task["dataset"], "scene": task["scene"], "difficulty": task["difficulty"],
                "result": dict(result, num_steps=len(acts), n_invalid_actions=0, final_pos_error=0.0,
                               final_rot_error=0.0, final_hor_error=0.0),
                "actions": acts, "start": task["start"], "target": task["target"], "raw_first_response": "",
                "raw_last_response": "",
                "vpd": {"schema": "vpd.tvr.trajectory/1", "arm": "replay", "category": pilot_cfg.category_of(task),
                        "cap": cap, "states": states, "replies": [f"Action: {a}" for a in acts],
                        "action_success": succ, "valid": valid, "user_texts": texts, "images": imgs,
                        "target": {"sha256": tsha, "path": str(images / tid / "target.png")},
                        "reach_digest": idx.digest(tid)}}
        if doctor is not None:
            doctor(tid, traj)
        T.write_json(run / "tasks" / tid / "trajectory.json", traj)
    env._sync_close()
    T.write_json(run / "results.json", {"schema": "vpd.tvr.results/1", "valid": True, "n_tasks": len(public)})
    T.write_json(run / "manifest.json", {"pool_ready": [{"wid": 0, "lp": "2", "profile": "envlib",
                                                         "font": {"sha256": "ab" * 32, "mode": "pinned"}}]})


def _demo_world(tmp_path, tvb, doctor=None, after_cap_obs=True):
    xs = [round(0.25 * i, 2) for i in range(5)]
    pos = F.grid_positions(xs, xs, blocked=[(0.5, 0.5)])
    big = F.grid_positions([0.0], [round(0.25 * i, 2) for i in range(5)])
    te, tv = tvb({"FloorPlan31": {"positions": pos}, "procthor_00006": {"positions": pos}})
    mk = F.fake_task
    sft = [mk("ithor_FloorPlan31_easy_t00_s00", "FloorPlan31", "ithor", "easy", (0, 0, 0, 0), (0.25, 0.25, 45, 0)),
           mk("ithor_FloorPlan31_easy_t01_s00", "FloorPlan31", "ithor", "easy", (0.5, 0.25, 0, 0), (0.5, 0.75, 0, 0)),
           mk("ithor_FloorPlan31_hard_t02_s00", "FloorPlan31", "ithor", "hard", (0, 0, 0, 0), (0.0, 0.75, 0, 0)),
           mk("ithor_FloorPlan31_hard_t03_s00", "FloorPlan31", "ithor", "hard", (0, 0, 0, 0), (0.0, 0.25, 0, 30)),
           mk("ithor_FloorPlan31_hard_t04_s00", "FloorPlan31", "ithor", "hard", (0, 0, 0, 0), (0.0, 0.0, 90, 0)),
           mk("procthor_procthor_00006_easy_t05_s00", "procthor_00006", "procthor", "easy", (0, 0, 90, 0),
              (0.5, 0.0, 90, 30)),
           mk("procthor_procthor_00006_hard_t06_s00", "procthor_00006", "procthor", "hard", (0, 0, 0, 0),
              (0.0, 0.25, 0, 0))]
    acts = {"ithor_FloorPlan31_easy_t00_s00": ["RotateRight", "MoveAhead", "Stop"],            # kept
            "ithor_FloorPlan31_easy_t01_s00": ["MoveAhead", "MoveAhead", "Stop"],              # move_failed@0
            "ithor_FloorPlan31_hard_t02_s00": ["MoveAhead", "MoveAhead", "Stop"],              # final_not_target
            "ithor_FloorPlan31_hard_t03_s00": ["MoveAhead", "LookDown", "Stop"],               # pose_mismatch (doctored)
            "ithor_FloorPlan31_hard_t04_s00": ["RotateRight"] * 30 + ["RotateLeft", "Stop"],   # no_stop (cap 30)
            "procthor_procthor_00006_easy_t05_s00": ["MoveAhead", "MoveAhead", "LookDown", "Stop"],  # kept
            "procthor_procthor_00006_hard_t06_s00": ["MoveAhead", "Stop"]}                     # kept, kin mismatch
    public = [{"task_id": tid, "actions": a, "positions": pos} for tid, a in acts.items()]
    tree = F.write_fake_tvr_tree(tmp_path / "tree", {"sft": sft}, public)
    houses = tmp_path / "houses"
    reach.houses(["sft1600"], houses, src=tree["procthor_gz"], tasks_dir=tree["tasks_dir"])
    write_index_direct(tmp_path / "reach", {"sft1600": sft}, lambda t: pos)
    idx = reach.ReachIndex(tmp_path / "reach")
    pub = T.public_sequences(tree["sft_train"], tasks_dir=tree["tasks_dir"])

    def default_doctor(tid, traj):
        v = traj["vpd"]
        if tid == "ithor_FloorPlan31_hard_t03_s00":   # THOR "returned" another horizon after LookDown
            v["states"][2]["horizon"] = 0.0
        if tid == "procthor_procthor_00006_hard_t06_s00":   # broken injection stand-in: another valid list
            v["valid"][0] = [a for a in v["valid"][0] if a != "LookUp"]
            v["user_texts"][0] = tvr_text.build_concat_user_text(v["valid"][0])
        if doctor is not None:
            doctor(tid, traj)

    run, images = tmp_path / "replay", tmp_path / "demos" / "images"
    fake_replay(tv, run, images, T.by_id(sft), pub, idx, houses, tree["procthor_dir"], default_doctor,
                after_cap_obs=after_cap_obs)
    return types.SimpleNamespace(tv=tv, tree=tree, run=run, images=images, sft=sft, acts=acts, pos=pos,
                                 reach=tmp_path / "reach", houses=houses, idx=idx, pub=pub)


def _build_args(w, out):
    return ["build", "--replay", str(w.run), "--images", str(w.images), "--out", str(out), "--reach", str(w.reach),
            "--public", str(w.tree["sft_train"]), "--tasks-dir", str(w.tree["tasks_dir"])]


@needs_repo
def test_demos_build_drop_reasons_records_and_idempotence(tmp_path, tvb):
    w = _demo_world(tmp_path, tvb)
    out = tmp_path / "demos" / "sft1600_lvp"
    assert demos.main(_build_args(w, out)) == 0
    man = json.loads((out / "manifest.json").read_text())
    dropped = {d["task_id"]: d["reason"] for d in json.loads((out / "dropped.json").read_text())}
    assert dropped == {"ithor_FloorPlan31_easy_t01_s00": "move_failed@0",
                       "ithor_FloorPlan31_hard_t02_s00": "final_not_target",
                       "ithor_FloorPlan31_hard_t03_s00": "pose_mismatch@1",
                       "ithor_FloorPlan31_hard_t04_s00": "no_stop"}
    assert man["kept"] == 3 and man["dropped"] == 4 and man["drop_reasons"] == {
        "move_failed": 1, "final_not_target": 1, "pose_mismatch": 1, "no_stop": 1}
    kin = json.loads((out / "kin_check.json").read_text())
    assert not man["kin_equal_all_kept"] and kin["kept_with_mismatch"] == ["procthor_procthor_00006_hard_t06_s00"]
    assert {m["field"] for m in kin["mismatches"] if m["task_id"] == "procthor_procthor_00006_hard_t06_s00"} == {
        "valid", "user_text"}
    assert "ithor_FloorPlan31_hard_t03_s00" in kin["tasks_with_mismatch"]  # the doctored pose, reported
    assert kin["n_checked"] == 7
    lines = (out / "demos_all.jsonl").read_text().splitlines()
    recs = [json.loads(ln) for ln in lines]
    assert [r["id"] for r in recs] == ["ithor_FloorPlan31_easy_t00_s00", "procthor_procthor_00006_easy_t05_s00",
                                       "procthor_procthor_00006_hard_t06_s00"]
    for ln, r in zip(lines, recs):
        assert ln == json.dumps(r)
        tj = json.loads((w.run / "tasks" / r["id"] / "trajectory.json").read_text())
        n = len(w.acts[r["id"]])
        assert r["n_turns"] == n and set(r) == {"id", "category", "dataset", "scene", "n_turns", "messages",
                                                "images"}
        assert r["messages"][0] == {"role": "system", "content": tvr_text.SYSTEM_PROMPT}
        assert [m["role"] for m in r["messages"][1:]] == ["user", "assistant"] * n
        assert [m["content"] for m in r["messages"][1::2]] == tj["vpd"]["user_texts"]  # the replay's own texts
        assert [m["content"] for m in r["messages"][2::2]] == [f"Action: {a}" for a in w.acts[r["id"]]]
        assert r["images"] == [str(w.images / r["id"] / f) for k in range(n)
                               for f in (f"step_{k:02d}.png", "target.png")]
        assert all(os.path.isabs(p) and os.path.isfile(p) for p in r["images"])
        assert all(m["content"].count("<image>") == 2 for m in r["messages"][1::2])
        assert r["category"] == pilot_cfg.category_of(T.by_id(w.sft)[r["id"]])
    assert man["demos_all_sha256"] == hashlib.sha256((out / "demos_all.jsonl").read_bytes()).hexdigest()
    ims = json.loads((out / "images_sha256.json").read_text())
    # every user turn's CURRENT + the target; the capped iTHOR task has 31 user turns (30 replies + the
    # unanswered observation, as eval_va) -> 32 files
    assert ims["n"] == sum(len(a) + 1 if len(a) <= 30 else 32 for a in w.acts.values())
    tj4 = json.loads((w.run / "tasks" / "ithor_FloorPlan31_hard_t04_s00" / "trajectory.json").read_text())
    assert len(tj4["actions"]) == 30 and len(tj4["vpd"]["user_texts"]) == 31
    assert man["images_sha256_file_sha256"] == hashlib.sha256((out / "images_sha256.json").read_bytes()).hexdigest()
    assert man["reach_index_sha256"] == w.idx.sha256 and man["render"]["workers"][0]["profile"] == "envlib"
    assert (out / "DONE").is_file()
    assert demos.main(["check", "--out", str(out)]) == 0
    # rerun: identical -> no-op
    mt = (out / "manifest.json").stat().st_mtime_ns
    assert demos.main(_build_args(w, out)) == 0 and (out / "manifest.json").stat().st_mtime_ns == mt
    # a re-rendered replay (new PNG bytes under the same names, recorded in the new trajectory) leaves
    # demos_all.jsonl identical; images_sha256.json differs -> refused, nothing rewritten
    from PIL import Image

    tp = w.run / "tasks" / "ithor_FloorPlan31_easy_t00_s00" / "trajectory.json"
    img0 = w.images / "ithor_FloorPlan31_easy_t00_s00" / "step_00.png"
    raw_tj, raw_img = tp.read_bytes(), img0.read_bytes()
    with Image.open(img0) as im:
        im.load()
        im.save(img0, format="PNG", compress_level=1)
    assert img0.read_bytes() != raw_img
    tj = json.loads(raw_tj)
    tj["vpd"]["images"][0]["cur_sha256"] = hashlib.sha256(img0.read_bytes()).hexdigest()
    T.write_json(tp, tj)
    ims_b = (out / "images_sha256.json").read_bytes()
    assert demos.main(_build_args(w, out)) == 2 and (out / "manifest.json").stat().st_mtime_ns == mt
    assert (out / "images_sha256.json").read_bytes() == ims_b and demos.main(["check", "--out", str(out)]) == 1
    tp.write_bytes(raw_tj)
    img0.write_bytes(raw_img)
    assert demos.main(_build_args(w, out)) == 0 and demos.main(["check", "--out", str(out)]) == 0
    # a different result over a DONE dir is refused
    tp = w.run / "tasks" / "ithor_FloorPlan31_easy_t00_s00" / "trajectory.json"
    tj = json.loads(tp.read_text())
    tj["vpd"]["states"][1]["rotation_y"] = 90.0
    T.write_json(tp, tj)
    assert demos.main(_build_args(w, out)) == 2 and (out / "manifest.json").stat().st_mtime_ns == mt
    # a tampered image fails the check
    first = recs[0]["images"][0]
    Path(first).write_bytes(Path(first).read_bytes() + b"x")
    assert demos.main(["check", "--out", str(out)]) == 1


@needs_repo
def test_demos_build_refuses_infrastructure_failures(tmp_path, tvb, monkeypatch):
    w = _demo_world(tmp_path, tvb)

    def refused(out):
        assert demos.main(_build_args(w, out)) == 2 and not out.exists()

    (w.run / "results_partial.json").write_text("{}")
    refused(tmp_path / "o1")
    (w.run / "results_partial.json").unlink()
    res = json.loads((w.run / "results.json").read_text())
    T.write_json(w.run / "results.json", dict(res, valid=False))
    refused(tmp_path / "o2")
    T.write_json(w.run / "results.json", res)
    tp = w.run / "tasks" / "ithor_FloorPlan31_easy_t00_s00" / "trajectory.json"
    raw = tp.read_bytes()
    tp.unlink()
    refused(tmp_path / "o3")                       # a missing trajectory
    tp.write_bytes(raw)
    img = w.images / "ithor_FloorPlan31_easy_t00_s00" / "step_01.png"
    ib = img.read_bytes()
    shutil.copyfile(w.images / "ithor_FloorPlan31_easy_t00_s00" / "step_00.png", img)
    refused(tmp_path / "o4")                       # sha256 differs from the trajectory's record
    img.write_bytes(ib)
    tj = json.loads(raw)
    tj["vpd"]["user_texts"][0] = tj["vpd"]["user_texts"][0].replace("Stop", "Stop ")
    T.write_json(tp, tj)
    refused(tmp_path / "o5")                       # a user text that is not build_concat_user_text(valid)
    tj = json.loads(raw)
    tj["actions"][0] = "RotateLeft"
    T.write_json(tp, tj)
    refused(tmp_path / "o6")                       # replayed actions differ from the public ones
    tj = json.loads(raw)
    tj["vpd"]["reach_digest"] = "0" * 64
    T.write_json(tp, tj)
    refused(tmp_path / "o7")                       # the replay used another reach set
    tj = json.loads(raw)
    tj["vpd"]["reach_digest"] = None
    T.write_json(tp, tj)
    refused(tmp_path / "o9")                       # ... or recorded none
    tj = json.loads(raw)
    tj["vpd"]["images"][1]["cur_sha256"] = None
    T.write_json(tp, tj)
    refused(tmp_path / "o10")                      # an image without a recorded sha256
    tp.write_bytes(raw[: len(raw) // 2])
    refused(tmp_path / "o11")                      # a truncated trajectory.json: exit 2, not 1
    tp.write_bytes(raw)
    empty = tmp_path / "no_reach"
    empty.mkdir()
    out12 = tmp_path / "o12"
    args12 = _build_args(w, out12)
    args12[args12.index("--reach") + 1] = str(empty)
    assert demos.main(args12) == 2 and not out12.exists()  # no reach index.json
    # the pinned public file (the default path) must have its registered sha256
    args13 = _build_args(w, tmp_path / "o13")
    i = args13.index("--public")
    del args13[i:i + 2]
    monkeypatch.setattr(T, "PUBLIC_JSON", Path(w.tree["sft_train"]))
    assert demos.main(args13) == 2 and not (tmp_path / "o13").exists()
    assert floors.main(["--task-sets", "sft1600", "--reach", str(w.reach), "--out", str(tmp_path / "f13"),
                        "--tasks-dir", str(w.tree["tasks_dir"])]) == 1 and not (tmp_path / "f13").exists()
    monkeypatch.setitem(pilot_cfg.FROZEN["public"], "sft_train_json_sha256",
                        hashlib.sha256(Path(w.tree["sft_train"]).read_bytes()).hexdigest())
    assert demos.main(args13) == 0
    assert demos.main(_build_args(w, tmp_path / "o8")) == 0


@needs_repo
def test_demos_build_run_tvr_cap_layout(tmp_path, tvb):
    """run_tvr's own layout at the iTHOR cap: no observation after the 30th reply (n user turns, not n + 1)."""
    w = _demo_world(tmp_path, tvb, after_cap_obs=False)
    tid = "ithor_FloorPlan31_hard_t04_s00"
    tj = json.loads((w.run / "tasks" / tid / "trajectory.json").read_text())
    assert len(tj["actions"]) == 30 and len(tj["vpd"]["user_texts"]) == 30 and len(tj["vpd"]["images"]) == 30
    out = tmp_path / "demos" / "cap"
    assert demos.main(_build_args(w, out)) == 0
    dropped = {d["task_id"]: d["reason"] for d in json.loads((out / "dropped.json").read_text())}
    assert dropped[tid] == "no_stop"
    ims = json.loads((out / "images_sha256.json").read_text())
    assert sum(1 for k in ims["files"] if k.startswith(tid + "/")) == len(tj["actions"]) + 1
    kin = json.loads((out / "kin_check.json").read_text())
    assert tid not in kin["tasks_with_mismatch"]


def test_drop_reason_unit():
    t = F.fake_task("ithor_U_easy_t00_s00", "U", "ithor", "easy", (0, 0, 0, 0), (0.0, 0.25, 0, 0))
    st = [pd(0.0, 0.0, 0, 0), pd(0.0, 0.25, 0, 0), pd(0.0, 0.25, 0, 0)]
    assert demos.drop_reason(t, ["MoveAhead", "Stop"], st, [True, None]) is None
    assert demos.drop_reason(t, ["Stop", "MoveAhead"], st, [None, True])[0] == "stop_not_last"
    assert demos.drop_reason(t, ["MoveAhead", "MoveAhead"], st, [True, True])[0] == "pose_mismatch@1"
    assert demos.drop_reason(t, ["MoveAhead"], st[:2], [True])[0] == "no_stop"
    assert demos.drop_reason(t, ["MoveAhead", "Stop"], st, [False, None])[0] == "move_failed@0"
    assert demos.drop_reason(t, ["MoveAhead", "Stop"], [pd(0.25, 0, 0, 0)] + st[1:], [True, None])[0] == \
        "pose_mismatch@start"


# ================================================================================================================
# compatibility
# ================================================================================================================
WORKER_SIDE = ("vpd/tvr/tvr_text.py", "vpd/tvr/kin_env.py", "vpd/tvr/reach.py")
OURS = WORKER_SIDE + ("vpd/tvr/tasks.py", "vpd/tvr/bfs.py", "vpd/tvr/floors.py", "vpd/tvr/demos.py",
                      "vpd/tests/tvr_fakes.py")


def test_py310_syntax_and_stdlib_imports():
    for rel in OURS:
        src = (paths.WORKTREE / rel).read_text(encoding="utf-8")
        tree = ast.parse(src, feature_version=(3, 10))
        if rel in WORKER_SIDE or rel.startswith("vpd/tvr/"):
            top = set()
            for n in tree.body:
                if isinstance(n, ast.Import):
                    top |= {a.name.split(".")[0] for a in n.names}
                elif isinstance(n, ast.ImportFrom) and n.module:
                    top.add(n.module.split(".")[0])
            assert not top & {"numpy", "PIL", "ai2thor", "torch", "skimage"}, (rel, top)
    assert not (paths.WORKTREE / "vpd" / "tvr" / "__init__.py").exists()


def test_worker_side_modules_import_under_tvr_py():
    py = lvp.TVR_PY
    if not Path(py).is_file():
        pytest.skip("TVR env python not present")
    env = dict(os.environ, PYTHONPATH=str(paths.WORKTREE), VPD_WORKTREE=str(paths.WORKTREE), OMP_NUM_THREADS="1")
    r = subprocess.run([str(py), "-B", "-c", "import sys; import vpd.tvr.tvr_text, vpd.tvr.kin_env, vpd.tvr.reach, "
                        "vpd.tvr.tasks; assert sys.version_info[:2] == (3, 10); print('ok')"],
                       env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and r.stdout.strip() == "ok", r.stderr


def test_cli_help():
    for mod in ("tvr_text", "tasks", "bfs", "floors", "reach", "demos"):
        r = subprocess.run([sys.executable, "-B", "-m", f"vpd.tvr.{mod}", "--help"], capture_output=True, text=True,
                           timeout=120, env=dict(os.environ, OMP_NUM_THREADS="1"))
        assert r.returncode == 0 and "usage" in r.stdout.lower(), (mod, r.stderr)


def test_fakes_helpers(tmp_path):
    a = F.frame_of("S", (0.25, 0.5, 90, 30))
    b = F.frame_of("S", F._pose((0.25, 0.5, 90, 30)))
    assert a.shape == (360, 640, 3) and str(a.dtype) == "uint8" and (a == b).all() and a.flags.writeable
    assert not (a == F.frame_of("S", (0.25, 0.5, 45, 30))).all()
    assert F.start_key(-0.0, 0.25) == "0.00,0.25"
    assert len(F.grid_positions([0, 0.25], [0, 0.25], blocked=[(0.25, 0.25)])) == 3
    root = F.write_fake_ai2thor(tmp_path / "f", {"A": {"positions": F.grid_positions([0.0], [0.0])}})
    assert json.loads((root / "fake_scenes.json").read_text())["A"]["positions"][0]["x"] == 0.0
    assert (root / "ai2thor" / "controller.py").is_file() and (root / "ai2thor" / "platform.py").is_file()
