"""eval/ablation.py: forward ladder and leave-one-out generated from the REAL configs/ files; every
generated config differs from its parent in exactly one modules.<slot> block and still builds and runs."""

import copy
import os

import pytest
import yaml

from conftest import ROOT
from eval import ablation
from helpers import CHECK_REPLIES, FIX, PLAN_REPLY, make_policy, run_episode
from synth import synth_obs

CFG = os.path.join(ROOT, "configs")
L0, L1, L3, L4 = (os.path.join(CFG, f"{n}.yaml") for n in ("L0_exec_only", "L1_plan", "L3_goal", "L4_recover"))
STEPS = ["plan", "check", "goal", "recover"]


def runnable(cfg):
    cfg = copy.deepcopy(cfg)
    cfg["task_metadata"] = {"id_to_name": {0: "turning_on_radio"}, "bddl_root": FIX, "task_misc_csv": None}
    scripted = {"kind": "scripted", "replies": {"plan.next_subtask": [PLAN_REPLY], "check.judge": CHECK_REPLIES}}
    cfg["providers"] = {"astra": dict(scripted), "claude": dict(scripted)}
    return cfg


def test_ladder_steps_switch_exactly_one_slot(tmp_path):
    paths = ablation.ladder(L0, L4, STEPS, str(tmp_path / "ladder"))
    assert [os.path.basename(p) for p in paths] == [f"ladder{k}_{s}.yaml" for k, s in enumerate(STEPS, 1)]
    parents = [L0] + paths[:-1]
    final = ablation.resolved(L4)
    for k, (parent, path, slot) in enumerate(zip(parents, paths, STEPS), 1):
        ok, text, rep = ablation.check(parent, path)
        assert ok and rep["slots"] == [slot], text
        assert ablation.resolved(path)["modules"][slot] == final["modules"][slot]
        raw = open(path, encoding="utf-8").read()
        assert raw.startswith("# GENERATED") and f"switched slot: modules.{slot}" in raw
        assert f"parent {ablation._show(parent)}" in raw
        doc = yaml.safe_load(raw)
        assert set(doc) == {"base", "run", "modules"} and list(doc["modules"]) == [slot]
        assert doc["run"] == {"name": f"ladder{k}_{slot}"}
    # the top of the ladder is the final config, up to its label
    rest = ablation.switch_report(ablation.resolved(paths[-1]), final)
    assert [d[0] for d in rest["diffs"]] == ["run.name"]


def test_loo_disables_exactly_one_enabled_slot(tmp_path):
    paths = ablation.loo(L4, str(tmp_path / "loo"))
    assert [os.path.basename(p) for p in paths] == [f"loo_no_{s}.yaml" for s in STEPS]
    for path, slot in zip(paths, STEPS):
        ok, text, rep = ablation.check(L4, path, slot)
        assert ok, text
        cfg = ablation.resolved(path)
        assert cfg["modules"][slot]["enabled"] is False
        assert all(cfg["modules"][s]["enabled"] for s in STEPS if s != slot)


@pytest.mark.parametrize("kind", ["ladder", "loo"])
def test_generated_configs_build_and_run(tmp_path, kind):
    from agent.config import load_config

    paths = ablation.ladder(L0, L4, STEPS, str(tmp_path)) if kind == "ladder" else ablation.loo(L4, str(tmp_path))
    for path in paths:
        policy, _, _ = make_policy(runnable(load_config(path)), tmp=str(tmp_path / "logs"))
        expected = {s: bool(ablation.resolved(path)["modules"][s]["enabled"]) for s in STEPS}
        described = policy._loop(0).m.describe()
        assert {s: described[s]["enabled"] for s in STEPS} == expected
        assert len(run_episode(policy, lambda t: synth_obs(t), 40)) == 40


def test_check_rejects_multi_slot_and_non_module_diffs(tmp_path):
    ok, text, rep = ablation.check(L0, L4)
    assert not ok and rep["slots"] == sorted(STEPS) and "FAIL" in text
    ok, _, rep = ablation.check(L3, L4)
    assert ok and rep["slots"] == ["recover"]
    ok, _, _ = ablation.check(L0, L1, slot="check")  # wrong expected slot
    assert not ok
    ok, _, rep = ablation.check(os.path.join(CFG, "base.yaml"), L0)  # only run.name differs
    assert not ok and rep["slots"] == [] and rep["labels"]
    sneaky = tmp_path / "sneaky.yaml"
    sneaky.write_text(f"base: {L3}\nrun: {{name: sneaky}}\nloop: {{check_interval_steps: 30}}\n"
                      "modules:\n  recover: {enabled: true, impl: scripted, params: {}}\n")
    ok, text, rep = ablation.check(L3, str(sneaky))
    assert not ok and rep["slots"] == ["recover"] and rep["other"][0][0] == "loop.check_interval_steps", text


def test_cli(tmp_path, capsys):
    assert ablation.main(["check", L3, L4]) == 0
    assert "OK: exactly one module slot switched: recover" in capsys.readouterr().out
    assert ablation.main(["check", L0, L4]) == 1
    out = tmp_path / "ladder"
    assert ablation.main(["ladder", "--base", L0, "--steps", "plan,check", "--from", L4, "--out", str(out)]) == 0
    assert sorted(os.listdir(out)) == ["ladder1_plan.yaml", "ladder2_check.yaml"]
    assert "--steps plan,check" in open(out / "ladder1_plan.yaml", encoding="utf-8").readline()
    capsys.readouterr()
    ablation.ladder(L0, L4, STEPS, str(out))  # a longer ladder, then the short one again
    assert ablation.main(["ladder", "--base", L0, "--steps", "plan,check", "--from", L4, "--out", str(out)]) == 0
    err = capsys.readouterr().err
    assert "ladder3_goal.yaml was not written by this generation" in err and "ladder1_plan.yaml" not in err


@pytest.mark.parametrize("kind", ["ladder", "loo"])
def test_committed_generated_configs_are_current(tmp_path, kind):
    """configs/ladder and configs/loo must match a fresh generation from the current L0 / L4."""
    committed = os.path.join(CFG, kind)
    if not os.path.isdir(committed):
        pytest.skip(f"configs/{kind} not generated")
    fresh = ablation.ladder(L0, L4, STEPS, str(tmp_path)) if kind == "ladder" else ablation.loo(L4, str(tmp_path))
    assert sorted(os.listdir(committed)) == sorted(os.path.basename(p) for p in fresh)
    for path in fresh:
        mine = os.path.join(committed, os.path.basename(path))
        assert ablation.resolved(mine) == ablation.resolved(path), f"{mine} is stale; regenerate it"


def test_ladder_input_errors(tmp_path):
    with pytest.raises(SystemExit):
        ablation.ladder(L0, L1, ["plan", "check"], str(tmp_path))  # check is not enabled in L1
    with pytest.raises(SystemExit):
        ablation.ladder(L1, L4, ["plan"], str(tmp_path))  # plan already enabled in the base
    with pytest.raises(SystemExit):
        ablation.ladder(L0, L4, ["plan", "nonsense"], str(tmp_path))
    assert not os.listdir(tmp_path)


def test_failed_self_check_restores_overwritten_files(tmp_path):
    """A generation that fails its self-check must not delete or alter configs it had overwritten."""
    base = tmp_path / "base_extra.yaml"  # check block with a param L4 lacks -> merged block != L4's block
    base.write_text(f"base: {L0}\nrun: {{name: base_extra}}\n"
                    "modules:\n  check: {enabled: false, params: {extra: 1}}\n")
    out = tmp_path / "ladder"
    out.mkdir()
    (out / "ladder1_plan.yaml").write_text("# committed step 1\n")
    (out / "ladder2_check.yaml").write_text("# committed step 2\n")
    with pytest.raises(SystemExit):
        ablation.ladder(str(base), L4, ["plan", "check", "goal"], str(out))
    assert (out / "ladder1_plan.yaml").read_text() == "# committed step 1\n"
    assert (out / "ladder2_check.yaml").read_text() == "# committed step 2\n"
    assert sorted(os.listdir(out)) == ["ladder1_plan.yaml", "ladder2_check.yaml"]
