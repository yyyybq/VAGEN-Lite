"""diagnostics/replay_truth.py WITHOUT a simulator: importable and --help-able without omnigibson, simulator
imports confined to main(), input checks, and the replay mechanics driven through a FAKE BatchedEvaluator
that mirrors the official control flow (evaluator.py v3.9.3-post2: load_batch -> policy.reset, _step_fn ->
policy.forward + env step, Timeout fires when episode_steps >= max_steps before the increment).
This is a mock: it says nothing about how the replay behaves in OmniGibson (NOT RUN)."""

import ast
import importlib.util
import json
import os
import subprocess
import sys
import types

import numpy as np
import pytest

from conftest import ROOT
from diagnostics import replay_truth as RT
from diagnostics.truth import compute_q_score, load_truth

TASK = "turning_on_radio"
SIM_MODULES = {"omnigibson", "torch", "omegaconf", "isaacsim", "omni", "pxr", "bddl", "carb", "hydra"}
LIT_ON = "(toggled_on radio_receiver.n.01_1)"
LIT_TABLE = "(ontop radio_receiver.n.01_1 table.n.02_1)"


# ------------------------------------------------------------------------------------ import hygiene


def test_importable_and_help_without_simulator():
    assert importlib.util.find_spec("omnigibson") is None  # this test env really has no simulator
    code = ("import sys, diagnostics.replay_truth as m; "
            f"bad = sorted(set(sys.modules) & set({sorted(SIM_MODULES)!r})); assert not bad, bad")
    env = {**os.environ, "PYTHONPATH": ROOT}
    subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env, check=True)
    out = subprocess.run([sys.executable, "-m", "diagnostics.replay_truth", "--help"], cwd=ROOT, env=env,
                         capture_output=True, text=True, check=True).stdout
    assert "NOT RUN" in out and "--episode-dir" in out


def test_simulator_imports_only_inside_main():
    path = os.path.join(ROOT, "diagnostics", "replay_truth.py")
    tree = ast.parse(open(path).read())
    inside_main = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "main":
            inside_main = {id(n) for n in ast.walk(node)}
    bad = []
    for node in ast.walk(tree):
        names = [a.name for a in node.names] if isinstance(node, ast.Import) else (
            [node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
        for name in names:
            if name.split(".")[0] in SIM_MODULES and id(node) not in inside_main:
                bad.append(f"line {node.lineno}: {name}")
    assert not bad, bad
    main_imports = [n for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)) and id(n) in inside_main]
    assert any(isinstance(n, ast.ImportFrom) and n.module == "omnigibson.eval.evaluator" for n in main_imports)


# ------------------------------------------------------------------------------------ fixtures


def actions(n, on=range(20, 50)):
    a = np.zeros((n, 23), np.float32)
    a[[i for i in on if i < n], 0] = 1.0
    a[:, 3] = np.arange(n) / 1000.0  # distinct per step, to check the order
    return a


def episode_dir(tmp_path, n_log=120, events=((45, "decision"), (77, "provider_call"), (150, "decision")), env=1):
    eid = f"runX-ep001-env{env}"
    d = tmp_path / "local" / "runX" / "episodes" / eid
    d.mkdir(parents=True)
    acts = actions(n_log)
    with open(d / "steps.jsonl", "w") as f:
        for s in range(n_log):
            a = [round(float(x), 5) for x in acts[s]]
            f.write(json.dumps({"step": s, "t": s / 30, "src": "exec", "a": a}) + "\n")
    with open(d / "events.jsonl", "w") as f:
        f.write(json.dumps({"_synthetic": True, "kind": "episode_start", "ctx": {"episode_id": eid}}) + "\n")
        for step, kind in events:
            f.write(json.dumps({"kind": kind, "slot": "check", "step": step}) + "\n")
    tags = {"task": TASK, "mode": "public_test", "instance_ids": "311,312"}
    (d / "summary.json").write_text(json.dumps({"_synthetic": True, "episode_id": eid, "run_id": "runX",
                                                "task_name": TASK, "tags": tags}))
    return str(d)


def official_json(tmp_path, steps=100, q=0.0, iid=312, task=TASK):
    p = tmp_path / f"{task}_{iid}_0.json"
    p.write_text(json.dumps({"_synthetic": True, "task": task, "instance_id": iid, "rollout_id": 0, "steps": steps,
                             "success": q == 1.0, "q_score": {"final": q}}))
    return str(p)


class Head:
    def __init__(self, body):
        self.body = body


class FakeWorld:
    """toggled_on follows the last action's a[0] > 0.5; the radio is on the table from the start."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.on, self.trace = False, []

    def apply(self, a):
        self.trace.append(np.array(a))
        self.on = float(a[0]) > 0.5

    def masks(self):
        return [[self.on, True], [self.on]]


class FakeAccessor:
    def __init__(self, world):
        self.world = world
        self.robot = types.SimpleNamespace(action_dim=23)
        self.success = False

    def get_goal_option_satisfaction(self):
        return self.world.masks()


class FakeBatchedEvaluator:
    last = None

    def __init__(self, cfg):
        self.cfg = cfg
        self.world = FakeWorld()
        task = types.SimpleNamespace(ground_goal_state_options=[
            [Head(["toggled_on", "radio_receiver.n.01_1"]), Head(["ontop", "radio_receiver.n.01_1", "table.n.02_1"])],
            [Head(["toggled_on", "radio_receiver.n.01_1"])]])
        self.env = types.SimpleNamespace(task=task)
        self.instance_eval_states = [types.SimpleNamespace(env_accessor=FakeAccessor(self.world))]
        self.policy = self.load_policy()
        self.exited = False
        FakeBatchedEvaluator.last = self

    def load_policy(self):
        raise AssertionError("the replay subclass must replace the policy")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.exited = True

    def load_batch(self, env_idx_to_instance, write_video=False, video_path=None, rollout_id=0, video_fps=30):
        self.world.reset()
        self.episode_steps = 0
        self.policy.reset()
        self.initial = self.world.masks()

    def _step_fn(self, active_env_indices):
        self.world.apply(np.asarray(self.policy.forward(obs={}))[0])
        truncated = self.episode_steps >= self.cfg["max_steps"]
        self.episode_steps += 1
        return [False], [truncated]

    def run(self, instances, write_video=False, video_path=None, metrics_dir=None, rollout_id=0, video_fps=30):
        self.load_batch({0: instances[0]})
        step = 0
        while True:
            term, trunc = self._step_fn([0])
            step += 1
            if term[0] or trunc[0]:
                break
        result = {"task": self.cfg["task"], "instance_id": instances[0], "rollout_id": rollout_id, "steps": step,
                  "success": False, "q_score": {"final": compute_q_score(False, self.world.masks(), self.initial)}}
        with open(os.path.join(metrics_dir, f"{self.cfg['task']}_{instances[0]}_{rollout_id}.json"), "w") as f:
            json.dump(result, f)
        return {instances[0]: result}


def fake_cfg(plan):
    return {"max_steps": plan["n_actions"] - 1, "task": plan["task"]}  # eval.py cfg, as far as the fake reads it


def make_plan(tmp_path, *extra):
    args = RT.parse_args(["--episode-dir", episode_dir(tmp_path), "--official-json", official_json(tmp_path),
                          "--out", str(tmp_path / "truth" / "ep.truth.jsonl"), *extra])
    p = RT.plan(args)
    p["every"] = args.every
    return p


# ------------------------------------------------------------------------------------ tests


def test_plan_resolves_identity_and_truncates_to_the_official_length(tmp_path):
    p = make_plan(tmp_path)
    assert (p["task"], p["instance_id"], p["mode"], p["episode_id"]) == (TASK, 312, "public_test", "runX-ep001-env1")
    assert p["n_actions"] == 100 and p["steps_in_log"] == 120  # the agent stepped past the official end
    assert p["record_at"] == {0, 30, 45, 60, 77, 90, 100}
    assert p["actions"].shape == (100, 23) and p["actions"].dtype == np.float32


def test_dry_run_prints_the_plan_and_imports_nothing(tmp_path, capsys):
    rc = RT.main(["--episode-dir", episode_dir(tmp_path), "--out", str(tmp_path / "t.jsonl"), "--dry-run",
                  "--every", "50"])
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["n_actions"] == 120 and out["truth_records_planned"] == len({0, 50, 100, 45, 77, 120})
    assert not (set(sys.modules) & SIM_MODULES)


def test_replay_with_fake_evaluator_records_truth(tmp_path):
    p = make_plan(tmp_path)
    footer = RT.run_replay(p, FakeBatchedEvaluator, fake_cfg, meta={"_synthetic": True})
    ev = FakeBatchedEvaluator.last
    assert ev.exited and len(ev.world.trace) == 100  # exactly n actions: max_steps = n - 1
    np.testing.assert_allclose(np.stack(ev.world.trace), p["actions"])  # logged order, nothing skipped
    assert footer["consistent"] is True and footer["replay_steps"] == 100 and footer["overrun_actions"] == 0

    track = load_truth([p["out"]])["runX-ep001-env1"]
    assert track.literals == [LIT_ON, LIT_TABLE] and track.options == [[0, 1], [0]]
    assert track.header["initial_satisfied"] == [1] and track.header["category"] == "training_annotation"
    assert track.header["instance_id"] == 312 and track.header["steps_sha256"] == p["steps_sha256"]
    assert [r["step"] for r in track.records] == [0, 30, 45, 60, 77, 90, 100]
    by_step = {r["step"]: r for r in track.records}
    assert by_step[30]["satisfied"] == [0, 1] and by_step[30]["q_now"] == 1.0  # on after action 20 -> state 21..50
    assert by_step[60]["satisfied"] == [1] and by_step[60]["q_now"] == 0.0
    assert by_step[100]["truncated"] is True and not by_step[90]["truncated"]
    assert track.regressions() == [{"literal": LIT_ON, "true_step": 30, "false_step": 60}]
    replay_json = os.path.join(p["replay_output_dir"], "json", f"{TASK}_312_0.json")
    assert json.load(open(replay_json))["steps"] == 100


def test_divergent_replay_is_flagged(tmp_path):
    args = RT.parse_args(["--episode-dir", episode_dir(tmp_path), "--official-json", official_json(tmp_path, q=0.5),
                          "--out", str(tmp_path / "t.jsonl")])
    p = RT.plan(args)
    p["every"] = args.every
    footer = RT.run_replay(p, FakeBatchedEvaluator, fake_cfg)
    assert footer["consistent"] is False and footer["official_q_final"] == 0.5 and footer["replay_q_final"] == 0.0


def test_replay_policy_order_and_action_dim():
    acts = actions(3)
    pol = RT.ReplayPolicy(acts)
    pol.set_action_dim(23)
    with pytest.raises(RT.ReplayInputError):
        pol.set_action_dim(21)
    got = [pol.forward()[0][3] for _ in range(4)]
    assert got == pytest.approx([0.0, 0.001, 0.002, 0.002]) and pol.overrun == 1
    pol.reset()
    assert pol.forward()[0][3] == 0.0


def test_input_errors(tmp_path):
    d = episode_dir(tmp_path)
    lines = open(os.path.join(d, "steps.jsonl")).read().splitlines()
    gap = tmp_path / "gap.jsonl"
    gap.write_text("\n".join(lines[:5] + lines[6:]) + "\n")
    with pytest.raises(RT.ReplayInputError, match="without gaps"):
        RT.load_actions(str(gap))
    short = tmp_path / "short.jsonl"
    short.write_text(json.dumps({"step": 0, "a": [0.0] * 22}) + "\n")
    with pytest.raises(RT.ReplayInputError, match="23 floats"):
        RT.load_actions(str(short))
    with pytest.raises(RT.ReplayInputError, match="the log only 120"):
        RT.load_actions(os.path.join(d, "steps.jsonl"), limit=200)

    def plan_with(*extra):
        return RT.plan(RT.parse_args(["--episode-dir", d, "--out", str(tmp_path / "t.jsonl"), *extra]))

    with pytest.raises(RT.ReplayInputError, match="not a public_test instance"):
        plan_with("--instance-id", "11")  # an index, not the resolved id
    with pytest.raises(RT.ReplayInputError, match="official JSON is for"):
        plan_with("--official-json", official_json(tmp_path, iid=313))
    assert plan_with("--instance-id", "5", "--mode", "train")["mode"] == "train"
    assert RT.mode_for_instance(305) == "public_test" and RT.mode_for_instance(330) == "hidden_test"
    assert RT.mode_for_instance(7) == "train"


def test_main_refuses_overwrite_and_exits_without_a_free_gpu(tmp_path, monkeypatch, capsys):
    import eval.gpu as gpumod

    d = episode_dir(tmp_path)
    out = tmp_path / "t.jsonl"
    out.write_text("old")
    assert RT.main(["--episode-dir", d, "--out", str(out)]) == 1
    assert "pass --force" in capsys.readouterr().err

    def no_gpu(*a, **k):
        raise gpumod.NoFreeGpuError("need 1 free GPU(s), found []")

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(gpumod, "pick_free_gpus", no_gpu)
    assert RT.main(["--episode-dir", d, "--out", str(out), "--force"]) == 1
    assert "no usable GPU" in capsys.readouterr().err
    assert not (set(sys.modules) & SIM_MODULES)
    assert "CUDA_VISIBLE_DEVICES" not in os.environ
