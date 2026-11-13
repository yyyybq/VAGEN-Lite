"""data/rounds.py: open / freeze / Dev runs / keep-rollback / resolve, on SYNTHETIC configs and artifacts."""

import dataclasses
import os

import pytest
import yaml

from data import rounds as R


def write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
    return path


@pytest.fixture
def root(tmp_path):
    r = str(tmp_path)
    write(os.path.join(r, "configs", "base.yaml"), "run: {name: base, seed: 0}\nmodules: {check: {enabled: false}}\n")
    write(os.path.join(r, "configs", "r0.yaml"), "base: base.yaml\nrun: {name: r0}\n")
    write(os.path.join(r, "configs", "r1.yaml"),
          "base: base.yaml\nrun: {name: r1}\nmodules: {check: {enabled: true}}\n")
    write(os.path.join(r, "configs", "r2.yaml"), "base: base.yaml\nrun: {name: r2}\n")
    write(os.path.join(r, "artifacts", "checker_r1", "weights.bin"), "synthetic weights v1")
    write(os.path.join(r, "artifacts", "checker_r1", "meta.json"), "{}")
    write(os.path.join(r, "labels", "r1.jsonl"), "{}\n")
    write(os.path.join(r, "artifacts", "memory_r2.jsonl"), "{}\n")
    return r


def round0():
    return R.RoundRecord(round=0, trained_component=None, config=R.RoundConfig(path="configs/r0.yaml"),
                         collection=R.Collection(budget={"episodes": 50}, sources=("step6-dev",)))


def round1():
    return R.RoundRecord(round=0, trained_component="checker",
                         config=R.RoundConfig(path="configs/r1.yaml", overrides=("modules.check.impl=llm",)),
                         collection=R.Collection(budget={"episodes": 100, "provider_usd": 20.0},
                                                 sources=("dev-collect-1",)),
                         labelling=R.Labelling(label_sources=("strong_model", "truth_rule"), files=(
                             R.Artifact(name="labels", path="labels/r1.jsonl", kind="labels"),)),
                         artifacts=(R.Artifact(name="checker", path="artifacts/checker_r1", kind="weights"),))


def kept(reg, k, root, runs=("dev-a",), decision="keep"):
    R.freeze_round(reg, k, root, now="2026-10-22T00:00:00+00:00")
    R.record_dev_runs(reg, k, runs)
    R.decide(reg, k, decision, notes="synthetic")


def test_full_cycle_with_rollback_and_resolve(root, tmp_path):
    reg = R.new_registry()
    R.new_round(reg, round0(), now="t0")
    kept(reg, 0, root)
    r1 = R.new_round(reg, round1(), now="t1")
    assert r1.round == 1 and r1.parent_round == 0
    frozen = R.freeze_round(reg, 1, root)
    assert frozen.frozen and frozen.config.sha256 and all(a.sha256 for a in frozen.all_artifacts())
    R.record_dev_runs(reg, 1, ["dev-r1-a", "dev-r1-b", "dev-r1-a"])
    assert reg.get(1).dev_run_ids == ("dev-r1-a", "dev-r1-b")
    R.decide(reg, 1, "keep")
    r2 = R.new_round(reg, R.RoundRecord(round=0, trained_component="memory",
                                        config=R.RoundConfig(path="configs/r2.yaml"),
                                        artifacts=(R.Artifact(name="library", path="artifacts/memory_r2.jsonl"),)))
    assert r2.parent_round == 1
    kept(reg, 2, root, decision="rollback")

    assert R.effective_round(reg, 2) == 1
    assert R.resolve(reg, 2).round == 1  # rolled back -> the config of round 1
    assert R.resolve(reg, 2, effective=False).round == 2
    assert R.resolve(reg, 0).round == 0
    cfg, rec = R.load_round_config(reg, 2, root)
    assert rec.round == 1 and cfg["run"]["name"] == "r1" and cfg["modules"]["check"]["impl"] == "llm"
    assert R.launch_args(rec, root) == ["--config", os.path.join(root, "configs", "r1.yaml"),
                                        "-o", "modules.check.impl=llm"]

    path = str(tmp_path / "rounds.yaml")
    R.save_registry(reg, path)
    back = R.load_registry(path)
    assert back == reg
    assert R.verify_round(back, 1, root) == []


def test_round_rules(root):
    reg = R.new_registry()
    with pytest.raises(R.RoundError, match="round 0 trains nothing"):
        R.new_round(reg, R.RoundRecord(round=0, trained_component="checker",
                                       config=R.RoundConfig(path="configs/r0.yaml")))
    R.new_round(reg, round0())
    with pytest.raises(R.RoundError, match="not frozen"):
        R.new_round(reg, round1())
    with pytest.raises(R.RoundError, match="Dev evaluation must run the frozen round"):
        R.record_dev_runs(reg, 0, ["x"])
    with pytest.raises(R.RoundError, match="needs a frozen round"):
        R.decide(reg, 0, "keep")
    R.freeze_round(reg, 0, root)
    with pytest.raises(R.RoundError, match="is frozen"):
        R.amend_round(reg, 0, notes="late edit")
    with pytest.raises(R.RoundError, match="already frozen"):
        R.freeze_round(reg, 0, root)
    R.record_dev_runs(reg, 0, ["dev0"])
    with pytest.raises(R.RoundError, match="cannot be rolled back"):
        R.decide(reg, 0, "rollback")
    R.decide(reg, 0, "keep")
    with pytest.raises(R.RoundError, match="exactly one of"):
        R.new_round(reg, R.RoundRecord(round=0, trained_component="planner",
                                       config=R.RoundConfig(path="configs/r1.yaml")))
    with pytest.raises(R.RoundError, match="exactly one of"):
        R.new_round(reg, R.RoundRecord(round=0, trained_component=None, config=R.RoundConfig(path="configs/r1.yaml")))
    R.new_round(reg, round1())
    with pytest.raises(R.RoundError, match="not frozen"):
        R.new_round(reg, round1())
    R.freeze_round(reg, 1, root)
    with pytest.raises(R.RoundError, match="no keep/rollback decision"):
        R.new_round(reg, round1())
    with pytest.raises(R.RoundError, match="is frozen"):
        R.amend_round(reg, 1, frozen=False)


def test_amend_open_round_then_freeze(root):
    reg = R.new_registry()
    R.new_round(reg, round0())
    R.amend_round(reg, 0, notes="collection extended", collection=R.Collection(budget={"episodes": 60}))
    assert reg.get(0).collection.budget == {"episodes": 60}
    with pytest.raises(R.RoundError, match="label_sources"):
        R.amend_round(reg, 0, labelling=R.Labelling(label_sources=("vibes",)))
    with pytest.raises(R.RoundError, match="cannot change"):
        R.amend_round(reg, 0, frozen=True)


def test_freeze_verifies_recorded_hashes_and_verify_detects_changes(root):
    reg = R.new_registry()
    R.new_round(reg, round0())
    kept(reg, 0, root)
    wrong_hash = R.Artifact(name="checker", path="artifacts/checker_r1", sha256="0" * 64)
    rec = dataclasses.replace(round1(), artifacts=(wrong_hash,))
    R.new_round(reg, rec)
    with pytest.raises(R.RoundError, match="recorded sha256"):
        R.freeze_round(reg, 1, root)
    R.amend_round(reg, 1, artifacts=(R.Artifact(name="checker", path="artifacts/checker_r1"),))
    R.freeze_round(reg, 1, root)
    write(os.path.join(root, "artifacts", "checker_r1", "weights.bin"), "retrained after freezing")
    problems = R.verify_round(reg, 1, root)
    assert len(problems) == 1 and "checker" in problems[0] and "changed since freezing" in problems[0]
    with pytest.raises(R.RoundError, match="changed since freezing"):
        R.load_round_config(reg, 1, root, effective=False)
    os.remove(os.path.join(root, "labels", "r1.jsonl"))
    assert any("does not exist" in p for p in R.verify_round(reg, 1, root))


def test_missing_artifact_blocks_freeze(root):
    reg = R.new_registry()
    R.new_round(reg, R.RoundRecord(round=0, trained_component=None, config=R.RoundConfig(path="configs/r0.yaml"),
                                   artifacts=(R.Artifact(name="x", path="artifacts/nope.bin"),)))
    with pytest.raises(R.RoundError, match="does not exist"):
        R.freeze_round(reg, 0, root)
    assert not reg.get(0).frozen


def test_tampered_registry_is_rejected(root, tmp_path):
    reg = R.new_registry()
    R.new_round(reg, round0())
    kept(reg, 0, root)
    path = str(tmp_path / "rounds.yaml")
    R.save_registry(reg, path)
    data = yaml.safe_load(open(path))
    data["invariants"]["report_split"] = "public_test 0-19"
    yaml.safe_dump(data, open(path, "w"))
    with pytest.raises(R.RoundError, match="invariants changed"):
        R.load_registry(path)
    data = yaml.safe_load(open(path))
    data["invariants"]["report_split"] = R.DEFAULT_INVARIANTS["report_split"]
    data["rounds"][0]["decision"] = "maybe"
    yaml.safe_dump(data, open(path, "w"))
    with pytest.raises(R.RoundError, match="decision 'maybe'"):
        R.load_registry(path)


def test_cli(root, tmp_path, capsys):
    path = str(tmp_path / "rounds.yaml")
    spec0 = write(str(tmp_path / "spec0.yaml"),
                  yaml.safe_dump({"trained_component": None, "config": {"path": "configs/r0.yaml"}}))
    spec1 = write(str(tmp_path / "spec1.yaml"), yaml.safe_dump({
        "trained_component": "checker", "config": {"path": "configs/r1.yaml", "overrides": ["modules.check.impl=llm"]},
        "artifacts": [{"name": "checker", "path": "artifacts/checker_r1"}]}))
    run = lambda *a: R.main([*a])  # noqa: E731
    assert run("new", path, "--spec", spec0, "--root", root) == 0
    assert run("freeze", path, "--round", "0", "--root", root) == 0
    assert run("dev-runs", path, "--round", "0", "dev0") == 0
    assert run("decide", path, "--round", "0", "keep") == 0
    assert run("new", path, "--spec", spec1, "--root", root) == 0
    assert run("decide", path, "--round", "1", "keep") == 1  # not frozen
    assert run("freeze", path, "--round", "1", "--root", root) == 0
    assert run("dev-runs", path, "--round", "1", "dev1") == 0
    assert run("decide", path, "--round", "1", "rollback", "--notes", "worse on Dev") == 0
    capsys.readouterr()
    assert run("resolve", path, "--round", "1", "--root", root) == 0
    out = capsys.readouterr()
    assert out.out.strip() == f"--config {os.path.join(root, 'configs', 'r0.yaml')}"
    assert "config of round 0" in out.err
    assert run("resolve", path, "--round", "1", "--own", "--root", root) == 0
    assert "modules.check.impl=llm" in capsys.readouterr().out
    assert run("verify", path, "--root", root) == 0
    assert run("show", path) == 0
    assert "round 1: checker | frozen | dev runs 1 | decision rollback" in capsys.readouterr().out


def test_freeze_covers_the_base_chain(root):
    reg = R.new_registry()
    R.new_round(reg, round0())
    rec = R.freeze_round(reg, 0, root)
    assert set(rec.config.files) == {"configs/r0.yaml", "configs/base.yaml"} and rec.config.resolved_sha256
    write(os.path.join(root, "configs", "base.yaml"), "run: {name: base, seed: 1}\n")
    problems = R.verify_round(reg, 0, root)
    assert any("configs/base.yaml changed" in p for p in problems)
    assert any("resolved config" in p for p in problems)


def test_fingerprint_of_a_repo_config_includes_prompts_and_catalog():
    from conftest import ROOT

    resolved, files = R.config_fingerprint(os.path.join(ROOT, "configs", "S7_min_loop.yaml"), [], ROOT)
    assert {"configs/S7_min_loop.yaml", "configs/L0_exec_only.yaml", "configs/base.yaml",
            "configs/prompts/plan/v1.txt", "configs/prompts/check/v1.txt",
            "configs/skills/pi05_turning_on_radio.yaml"} <= set(files)
    assert "missing" not in files.values() and len(resolved) == 64
