"""CPU tests of the run folders and sco commands (vagen.q38s4.launch) and of the node entry (run_train.sh)."""
from __future__ import annotations

import dataclasses
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

from vagen.q38s4 import launch as L
from vagen.q38s4.lf_nfp import S4Settings

ENTRY = Path(L.__file__).with_name("run_train.sh")


# ------------------------------------------------------------------------------------------------ batch plan
def test_batch_plan():
    p = L.batch_plan(8)
    assert p == {"nodes": 8, "gpus_per_node": 8, "world": 64, "per_device": 1, "grad_accum": 4, "global_batch": 256}
    assert L.batch_plan(2)["grad_accum"] == 16 and L.batch_plan(1)["grad_accum"] == 32
    assert L.batch_plan(2, 2)["grad_accum"] == 64 and L.batch_plan(4, 8, 2)["grad_accum"] == 4
    for bad in ((3,), (5,), (8, 8, 3), (0,), (512,)):
        with pytest.raises(ValueError):
            L.batch_plan(*bad)
    # 590,667 QA + 290,000 NFP rows on 8 nodes: about 3,450 steps
    assert L.estimated_steps(880_667, p) == 3441
    assert L.estimated_steps(256, p) == 1 and L.estimated_steps(257, p) == 2


def test_parse_sets():
    assert L.parse_sets(["save_steps=100", "max_steps=3", "dataset=s4_qa", "x=null"]) == {
        "save_steps": 100, "max_steps": 3, "dataset": "s4_qa", "x": None}
    with pytest.raises(ValueError):
        L.parse_sets(["save_steps"])


# ------------------------------------------------------------------------------------------------ prepare
@pytest.fixture
def repo(tmp_path, monkeypatch):
    """A git repository standing in for the worktree (vagen/ committed)."""
    wt = tmp_path / "wt"
    (wt / "vagen" / "q38s4").mkdir(parents=True)
    (wt / "vagen" / "q38s4" / "x.py").write_text("X = 1\n")
    (wt / "other.txt").write_text("not in the snapshot\n")
    git = ["git", "-C", str(wt), "-c", "user.name=t", "-c", "user.email=t@t"]
    subprocess.run(["git", "init", "-q", str(wt)], check=True)
    subprocess.run(git + ["add", "-A"], check=True)
    subprocess.run(git + ["commit", "-qm", "init"], check=True)
    monkeypatch.setattr(L, "WORKTREE", wt)
    return wt


@pytest.fixture
def data(tmp_path):
    d = tmp_path / "data"
    d.mkdir()
    (d / "dataset_info.json").write_text(json.dumps({"s4_qa": {}, "s4_nfp": {}, "s4_nfp_heldout": {}}))  # v1 build
    (d / "build_manifest.json").write_text(json.dumps({"rows_out": {"qa": 590_667, "nfp": 290_000}}))
    tok = tmp_path / "tok"
    tok.mkdir()
    return d, tok


def prep(tmp_path, data, *extra, run="run"):
    d, tok = data
    a = ["prepare", "--run", str(tmp_path / run), "--data", str(d), "--tokenized", str(tok), *extra]
    return L.prepare(L.parse_args(a))


def test_prepare_train(tmp_path, repo, data):
    plan = prep(tmp_path, data, "--arm", "nfp", "--nodes", "8", "--set", "save_steps=250")
    run = tmp_path / "run"
    head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    assert plan["commit"] == head and (run / "code" / "COMMIT").read_text().strip() == head
    assert (run / "code" / "vagen" / "q38s4" / "x.py").read_text() == "X = 1\n"
    assert not (run / "code" / "other.txt").exists() and (run / "logs").is_dir()
    assert plan["grad_accum"] == 4 and plan["world"] == 64 and plan["estimated_steps"] == 3441
    assert plan["worker_spec"] == "N4lS.Iq.I80.8" and plan["sets"] == {"save_steps": 250}
    assert json.loads((run / "plan.json").read_text()) == plan
    cfg = yaml.safe_load((run / "train.yaml").read_text())
    d, tok = data
    assert cfg["gradient_accumulation_steps"] == 4 and cfg["per_device_train_batch_size"] == 1
    assert cfg["output_dir"] == str(run / "ckpt") and cfg["tokenized_path"] == str(tok)
    assert cfg["dataset"] == "s4_qa,s4_nfp" and cfg["dataset_dir"] == cfg["media_dir"] == str(d)
    assert cfg["model_name_or_path"] == L.MODEL_27B and cfg["save_steps"] == 250
    for k, v in (("learning_rate", 1e-5), ("num_train_epochs", 1), ("lr_scheduler_type", "cosine"),
                 ("warmup_ratio", 0.03), ("weight_decay", 0.0), ("freeze_vision_tower", True),
                 ("freeze_multi_modal_projector", False), ("video_fps", 2.0), ("video_maxlen", 256),
                 ("video_max_pixels", 65536), ("save_total_limit", 1), ("data_shared_file_system", True)):
        assert cfg[k] == v, k
    s = S4Settings.from_file(run / "q38s4.json")  # what lf_nfp reads next to train.yaml
    assert (s.nfp_mse, s.nfp_cos, s.head_on, s.logits) == (0.1, 0.1, True, "targets") and plan["logits"] == "targets"
    assert json.loads((run / "q38_sft.json").read_text()) == {"bf16_copy": "sibling", "host_mem_s": 10}
    # the other arm, with LF's logits path
    plan2 = prep(tmp_path, data, "--arm", "nonfp", "--nodes", "2", "--logits", "all", run="run2")
    s2 = S4Settings.from_file(tmp_path / "run2" / "q38s4.json")
    assert not s2.head_on and s2.logits == "all" and plan2["logits"] == "all"
    assert yaml.safe_load((tmp_path / "run2" / "train.yaml").read_text())["gradient_accumulation_steps"] == 16
    small = prep(tmp_path, data, "--arm", "nfp", "--nodes", "2", "--gpus-per-node", "2", "--global-batch", "8",
                 run="run3")
    assert (small["world"], small["grad_accum"], small["global_batch"]) == (4, 2, 8)


def test_prepare_v2_build_with_image_rows(tmp_path, repo, data):
    """data_build v2: image QA rows in qa_img.jsonl / s4_qa_img; the list and the step estimate include them."""
    d, _ = data
    (d / "dataset_info.json").write_text(json.dumps({"s4_qa": {}, "s4_qa_img": {}, "s4_nfp": {},
                                                     "s4_nfp_heldout": {}}))
    (d / "build_manifest.json").write_text(json.dumps({"rows_out": {"qa": 374_148, "qa_img": 216_519,
                                                                    "nfp": 290_000}}))
    plan = prep(tmp_path, data, "--arm", "nonfp", "--nodes", "8")
    cfg = yaml.safe_load((tmp_path / "run" / "train.yaml").read_text())
    assert cfg["dataset"] == "s4_qa,s4_qa_img,s4_nfp"
    assert plan["rows"] == 880_667 and plan["estimated_steps"] == 3441  # as with the one-file v1 build
    (d / "dataset_info.json").write_text(json.dumps({"s4_qa_img": {}, "s4_nfp": {}}))
    with pytest.raises(SystemExit):  # s4_qa missing
        prep(tmp_path, data, "--arm", "nonfp", run="run_bad")
    assert not (tmp_path / "run_bad").exists()


def test_prepare_refusals(tmp_path, repo, data):
    prep(tmp_path, data, "--arm", "nfp")
    with pytest.raises(SystemExit):  # a run folder is made once
        prep(tmp_path, data, "--arm", "nfp")
    with pytest.raises(ValueError):
        prep(tmp_path, data, "--arm", "nfp", "--nodes", "3", run="r3")
    with pytest.raises(ValueError):
        prep(tmp_path, data, "--arm", "nfp", "--set", "gradient_accumulation_steps=1", run="r4")
    with pytest.raises(SystemExit):  # --arm is required for training
        L.parse_args(["prepare", "--run", "r", "--data", "d", "--tokenized", "t"])
    d, tok = data
    with pytest.raises(SystemExit):  # train before the tokenized data exists
        L.prepare(L.parse_args(["prepare", "--run", str(tmp_path / "r5"), "--arm", "nfp", "--data", str(d),
                                "--tokenized", str(tmp_path / "missing")]))
    (repo / "vagen" / "q38s4" / "x.py").write_text("X = 2\n")
    with pytest.raises(SystemExit):  # uncommitted change under vagen/
        prep(tmp_path, data, "--arm", "nfp", run="r6")
    assert not any((tmp_path / r).exists() for r in ("r3", "r4", "r5", "r6"))  # refused before the folder exists


def test_prepare_tokenize(tmp_path, repo, data):
    d, _ = data
    tok = tmp_path / "tok_new"
    plan = L.prepare(L.parse_args(["prepare", "--run", str(tmp_path / "t"), "--mode", "tokenize", "--data", str(d),
                                   "--tokenized", str(tok)]))
    assert plan["mode"] == "tokenize" and plan["nodes"] == 1 and plan["world"] == 1 and plan["arm"] == "nonfp"
    assert plan["worker_spec"] == "N4lS.Iq.I80.1" and plan["estimated_steps"] is None
    assert yaml.safe_load((tmp_path / "t" / "train.yaml").read_text())["tokenized_path"] == str(tok)
    with pytest.raises(SystemExit):  # already tokenized
        L.prepare(L.parse_args(["prepare", "--run", str(tmp_path / "t2"), "--mode", "tokenize", "--data", str(d),
                                "--tokenized", str(data[1])]))


def test_yaml_keys_are_llamafactory_arguments(tmp_path, repo, data):
    pytest.importorskip("llamafactory.hparams.parser")
    from llamafactory.hparams.parser import _TRAIN_ARGS

    known = {f.name for cls in _TRAIN_ARGS for f in dataclasses.fields(cls)}
    prep(tmp_path, data, "--arm", "nfp")
    cfg = yaml.safe_load((tmp_path / "run" / "train.yaml").read_text())
    assert set(cfg) <= known, sorted(set(cfg) - known)


# ------------------------------------------------------------------------------------------------ submit
def test_sco_command_and_dry_run(tmp_path, repo, data, monkeypatch, capsys):
    prep(tmp_path, data, "--arm", "nfp", "--nodes", "8")
    run = tmp_path / "run"
    plan = json.loads((run / "plan.json").read_text())
    cmd = L.sco_command(run, plan, "zoetrope", "j")
    assert cmd[:4] == [L.SCO, "acp", "jobs", "create"]
    for part in ("--aec2-name=zoetrope", "--worker-nodes=8", "--worker-spec=N4lS.Iq.I80.8", "--priority=NORMAL",
                 f"--container-image-url={L.IMAGE}", "--training-framework=pytorch", "--job-name=j",
                 f"--command=bash {run}/code/vagen/q38s4/run_train.sh {run}"):
        assert part in cmd, part
    assert "--aec2-name=h800" in L.sco_command(run, plan, "h800", "j")
    with pytest.raises(ValueError):
        L.sco_command(run, plan, "auto", "j")
    with pytest.raises(SystemExit):  # no pool, no choice
        L.parse_args(["submit", "--run", str(run)])

    def boom(*a, **k):
        raise AssertionError("dry run ran a command")

    monkeypatch.setattr(L.subprocess, "run", boom)
    L.submit(L.parse_args(["submit", "--run", str(run), "--pool", "h800"]))
    out = capsys.readouterr().out
    assert "dry run" in out and "--aec2-name=h800" in out and "q38s4-train-nfp-run" in out
    assert not (run / "submits.jsonl").exists()


def test_submit_yes_records(tmp_path, repo, data, monkeypatch):
    prep(tmp_path, data, "--arm", "nonfp")
    run = tmp_path / "run"
    seen = []

    def fake(cmd, **k):
        seen.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "job created: pt-xyz\n", "")

    monkeypatch.setattr(L.subprocess, "run", fake)
    L.submit(L.parse_args(["submit", "--run", str(run), "--pool", "zoetrope", "--yes", "--job-name", "jj"]))
    assert len(seen) == 1 and "--job-name=jj" in seen[0]
    rec = [json.loads(x) for x in open(run / "submits.jsonl")]
    assert rec[0]["rc"] == 0 and rec[0]["pool"] == "zoetrope" and "pt-xyz" in rec[0]["stdout"]


# ------------------------------------------------------------------------------------------------ tokenize only
def test_tokenize_only_stops_before_the_model(monkeypatch):
    from vagen.q38s4 import lf_nfp

    calls = []
    load = lf_nfp.make_load_model(lambda *a: calls.append(a) or "model")
    monkeypatch.setenv(lf_nfp.TOKENIZE_ONLY, "1")
    with pytest.raises(SystemExit) as e:
        load(None, None, None, is_trainable=True)
    assert e.value.code == 0 and not calls
    monkeypatch.setenv(lf_nfp.TOKENIZE_ONLY, "0")
    monkeypatch.setattr(lf_nfp, "SETTINGS", S4Settings(0.0, 0.0))
    assert load(None, None, None, is_trainable=True) == "model" and len(calls) == 1


# ------------------------------------------------------------------------------------------------ node entry
def test_entry_syntax():
    subprocess.run(["bash", "-n", str(ENTRY)], check=True)


def test_entry_stops_on_node_count_mismatch(tmp_path, repo, data):
    prep(tmp_path, data, "--arm", "nfp", "--nodes", "8")
    run = tmp_path / "run"
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PET_", "RANK", "WORLD_SIZE"))}
    env.update(PET_NODE_RANK="3", PET_NNODES="2", Q38S4_PYTHON=sys.executable)
    res = subprocess.run(["bash", str(ENTRY), str(run)], env=env, timeout=120)
    assert res.returncode == 2
    log = (run / "logs" / "node3.log").read_text()
    assert "node_rank 3 nnodes 2" in log and "[plan] mode train nodes 8 gpus_per_node 8 grad_accum 4" in log
    assert "the plan 8: stop" in log and "accelerate" not in log.split("[plan]")[1]


def test_entry_opens_checkpoints_and_keeps_the_exit_code(tmp_path, repo, data):
    """Node 0 makes RUN/ckpt and RUN/ckpt_bf16 readable after training (h800: files written 0600 as another uid)."""
    prep(tmp_path, data, "--arm", "nfp", "--nodes", "1")
    run = tmp_path / "run"
    fake = tmp_path / "fakepy.sh"  # plan() reads go to python; the training line writes a 0600 file and fails
    fake.write_text("#!/usr/bin/env bash\n"
                    "if [[ \"$1\" == -m && \"$2\" == accelerate.commands.launch ]]; then\n"
                    f"  mkdir -p {run}/ckpt/checkpoint-1 {run}/ckpt_bf16/checkpoint-1\n"
                    f"  (umask 077; echo w > {run}/ckpt/checkpoint-1/model.safetensors;"
                    f" echo w > {run}/ckpt_bf16/checkpoint-1/model.safetensors)\n"
                    "  exit 5\nfi\n"
                    f"exec {sys.executable} \"$@\"\n")
    fake.chmod(0o755)
    env = {k: v for k, v in os.environ.items() if not k.startswith(("PET_", "RANK", "WORLD_SIZE"))}
    env.update(PET_NODE_RANK="0", PET_NNODES="1", Q38S4_PYTHON=str(fake))
    res = subprocess.run(["bash", str(ENTRY), str(run)], env=env, timeout=120)
    assert res.returncode == 5
    for d in ("ckpt", "ckpt_bf16"):
        f = run / d / "checkpoint-1" / "model.safetensors"
        assert f.stat().st_mode & 0o777 == 0o644, oct(f.stat().st_mode)
        assert (run / d / "checkpoint-1").stat().st_mode & 0o055 == 0o055
    assert "exit 5" in (run / "logs" / "node0.log").read_text()
    env.update(PET_NODE_RANK="1", PET_NNODES="1")  # a node other than 0 leaves the files alone
    (run / "ckpt" / "checkpoint-1" / "model.safetensors").chmod(0o600)
    subprocess.run(["bash", str(ENTRY), str(run)], env=env, timeout=120)
    assert (run / "ckpt" / "checkpoint-1" / "model.safetensors").stat().st_mode & 0o777 == 0o600
