"""CPU tests of the stage-3 helpers (vagen.q38s4.tools.smoke, tools.job, tools.lf_plain)."""
from __future__ import annotations

import json
import subprocess
from collections import Counter
import sys
from pathlib import Path

import pytest

from vagen.q38s4 import launch as L
from vagen.q38s4.tools import job as J
from vagen.q38s4.tools import smoke as S

TOOLS = Path(J.__file__).parent


def write_jsonl(path: Path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def fake_data(root: Path) -> Path:
    d = root / "data"
    d.mkdir()
    write_jsonl(d / "qa.jsonl", [{"i": i} for i in range(30)])
    write_jsonl(d / "nfp.jsonl", [{"j": j} for j in range(20)])
    (d / "nfp_heldout.jsonl").write_text("")
    (d / "dataset_info.json").write_text(json.dumps({"s4_qa": {}, "s4_nfp": {}, "s4_nfp_heldout": {}}))
    return d


# ------------------------------------------------------------------------------------------------------ subset
def test_subset(tmp_path):
    d = fake_data(tmp_path)
    m = S.subset(d, tmp_path / "sub", 7, 3, seed=1)
    assert m["rows_out"] == {"qa": 7, "nfp": 3, "qa_heldout": 0, "nfp_heldout": 0}
    qa = [json.loads(x)["i"] for x in (tmp_path / "sub" / "qa.jsonl").read_text().splitlines()]
    assert len(qa) == 7 and qa == sorted(qa) and len(set(qa)) == 7
    # seeded: the same rows again; another seed other rows
    S.subset(d, tmp_path / "sub2", 7, 3, seed=1)
    assert (tmp_path / "sub2" / "qa.jsonl").read_text() == (tmp_path / "sub" / "qa.jsonl").read_text()
    S.subset(d, tmp_path / "sub3", 7, 3, seed=2)
    assert (tmp_path / "sub3" / "qa.jsonl").read_text() != (tmp_path / "sub" / "qa.jsonl").read_text()
    assert json.loads((tmp_path / "sub" / "dataset_info.json").read_text()).keys() == {"s4_qa", "s4_nfp",
                                                                                      "s4_nfp_heldout"}
    S.subset(d, tmp_path / "qaonly", 5, 0)
    assert (tmp_path / "qaonly" / "nfp.jsonl").read_text() == ""
    with pytest.raises(SystemExit):
        S.subset(d, tmp_path / "sub", 1, 1)  # exists
    with pytest.raises(SystemExit):
        S.subset(d, tmp_path / "big", 31, 0)


# ------------------------------------------------------------------------------------------------------ plain
def make_run(tmp_path: Path, mode: str = "train") -> Path:
    run = tmp_path / "run"
    sh = run / "code" / "vagen" / "q38s4"
    sh.mkdir(parents=True)
    real = Path(L.__file__).with_name("run_train.sh").read_text()
    (sh / "run_train.sh").write_text(real)
    plan = {"mode": mode, "arm": "nonfp", "nodes": 1, "gpus_per_node": 8, "world": 8, "global_batch": 32,
            "image": L.IMAGE, "worker_spec": "N4lS.Iq.I80.8"}
    (run / "plan.json").write_text(json.dumps(plan))
    return run


def test_plain_entry(tmp_path):
    run = make_run(tmp_path)
    dest = S.plain_entry(run)
    src = (run / "code" / "vagen" / "q38s4" / "run_train.sh").read_text()
    out = dest.read_text()
    assert out.count("vagen.q38s4.tools.lf_plain") == 1
    # the tokenize line keeps lf_nfp; nothing else changes
    assert "Q38S4_TOKENIZE_ONLY=1 \"$PY\" -m vagen.q38s4.lf_nfp" in out
    diff = [(a, b) for a, b in zip(src.splitlines(), out.splitlines()[1:]) if a != b]
    assert len(diff) == 1 and "accelerate" not in diff[0][0] and diff[0][1].endswith(S.PLAIN_TO.strip())
    assert subprocess.run(["bash", "-n", str(dest)]).returncode == 0
    with pytest.raises(SystemExit):
        S.plain_entry(run)  # exists


def test_plain_entry_refuses_tokenize_run(tmp_path):
    with pytest.raises(SystemExit):
        S.plain_entry(make_run(tmp_path, "tokenize"))


def test_submit_dry_run(tmp_path, capsys):
    run = make_run(tmp_path)
    with pytest.raises(SystemExit):
        S.submit(run, "zoetrope", plain=True, yes=False)  # run_plain.sh missing
    S.plain_entry(run)
    cmd = S.submit(run, "h800", plain=True, yes=False)
    assert cmd[-1] == f"--command=bash {run.resolve()}/run_plain.sh {run.resolve()}"
    assert "--aec2-name=h800" in cmd and "--priority=NORMAL" in cmd and "--worker-nodes=1" in cmd
    cmd2 = S.submit(run, "zoetrope", plain=False, yes=False)
    assert cmd2[-1].endswith(f"code/vagen/q38s4/run_train.sh {run.resolve()}")
    assert "dry run" in capsys.readouterr().out
    assert not (run / "submits.jsonl").exists()


# ------------------------------------------------------------------------------------------------------ job
def test_job_prepare_and_env(tmp_path, monkeypatch):
    monkeypatch.setattr(L, "snapshot_code", lambda dest, head, worktree=None: dest.mkdir(parents=True) or head)
    spec = J.prepare(tmp_path / "j", "vagen.q38s4.data_build", ["--src", "x"], commit="abc")
    assert spec["module"] == "vagen.q38s4.data_build" and spec["args"] == ["--src", "x"] and spec["threads"] == 4
    assert json.loads((tmp_path / "j" / "job.json").read_text())["commit"] == "abc"
    with pytest.raises(SystemExit):
        J.prepare(tmp_path / "j", "m", [], commit="abc")
    cmd = J.sco_command((tmp_path / "j").resolve(), spec, "zoetrope", "n")
    assert cmd[-1] == f"--command=bash {(tmp_path / 'j').resolve()}/code/vagen/q38s4/tools/job_entry.sh {(tmp_path / 'j').resolve()}"
    assert "--worker-spec=N4lS.Iq.I80.1" in cmd
    with pytest.raises(ValueError):
        J.sco_command(tmp_path, spec, "auto", "n")
    env = J.child_env(tmp_path / "j", 3, {"https_proxy": "p", "OMP_NUM_THREADS": "64", "X": "1"})
    assert "https_proxy" not in env and env["X"] == "1" and env["TOKENIZERS_PARALLELISM"] == "false"
    assert all(env[k] == "3" for k in J.THREAD_VARS) and env["DISABLE_VERSION_CHECK"] == "1"


def test_job_args():
    a = J.parse_args(["prepare", "--job", "J", "--", "mod", "--x", "1"])
    assert a.rest == ["mod", "--x", "1"]
    with pytest.raises(SystemExit):
        J.parse_args(["prepare", "--job", "J"])


def test_job_run_end_to_end(tmp_path):
    """job run execs the module with the thread variables, from JOB/code, through job_entry.sh."""
    job = tmp_path / "j"
    code = job / "code"
    (code / "probe_mod").mkdir(parents=True)
    (code / "COMMIT").write_text("abc\n")
    (code / "probe_mod" / "__init__.py").write_text("")
    (code / "probe_mod" / "__main__.py").write_text(
        "import os, sys, json; print(json.dumps({'argv': sys.argv[1:], 'omp': os.environ['OMP_NUM_THREADS'], "
        "'cwd': os.getcwd()}))")
    # the real job module and launch, copied in (job_entry runs from JOB/code)
    src_root = Path(L.__file__).resolve().parents[2]
    subprocess.run(f"cd {src_root} && tar -cf - vagen/__init__.py vagen/q38s4/__init__.py vagen/q38s4/launch.py "
                   f"vagen/q38s4/tools | tar -xf - -C {code}", shell=True, check=True)
    (job / "job.json").write_text(json.dumps({"module": "probe_mod", "args": ["a", "b c"], "threads": 2,
                                              "commit": "abc"}))
    env = {"Q38S4_PYTHON": sys.executable, "PATH": "/usr/bin:/bin"}
    rc = subprocess.run(["bash", str(code / "vagen" / "q38s4" / "tools" / "job_entry.sh"), str(job)], env=env).returncode
    log = (job / "job.log").read_text()
    assert rc == 0, log
    assert (job / "exit_code").read_text().strip() == "0"
    out = json.loads(next(ln for ln in log.splitlines() if ln.startswith("{")))
    assert out == {"argv": ["a", "b c"], "omp": "2", "cwd": str(code.resolve())}


# ------------------------------------------------------------------------------------------------------ inspect
def test_pad_runs_and_collapse():
    ids = [1, 9, 9, 9, 2, 9, 9, 3, 8, 8]
    assert S.pad_runs(ids, 9) == [3, 2] and S.pad_runs(ids, 8) == [2] and S.pad_runs([], 9) == []
    dec = lambda xs: "".join(f"[{x}]" for x in xs)  # noqa: E731
    assert S.collapse(ids, {9: "<V>", 8: "<I>"}, dec) == "[1]<V>x3[2]<V>x2[3]<I>x2"


def test_describe_row():
    # tokens: 5 = "<0.5 seconds>", 6 = vision_start, 7 = vision_end, 9 = video pad
    vocab = {1: "user ", 5: "<0.5 seconds>", 4: "<2.5 seconds>", 6: "<s>", 7: "</s>", 9: "V", 3: "<think>",
             2: "</think>", 10: "B"}
    dec = lambda xs: "".join(vocab[x] for x in xs)  # noqa: E731
    ids = [1, 5, 6, 9, 9, 9, 7, 4, 6, 9, 9, 9, 7, 3, 2, 10]
    labels = [-100] * 13 + [3, 2, 10]
    d = S.describe_row(ids, labels, {"video_pad": 9, "image_pad": 99}, dec)
    assert d["video_units"] == 2 and d["video_tokens"] == 6 and d["tokens_per_unit"] == {3: 2}
    assert d["timestamps"] == {"n": 2, "first": [0.5, 2.5], "last": [0.5, 2.5]}
    assert d["think_block"] and d["empty_think"] and d["answer"] == "<think></think>B" and d["label_tokens"] == 3
    assert "V" not in d["text"] and "<|video_pad|>x3" in d["text"]


def test_row_kind_and_pick():
    assert S.row_kind({"videos": ["/m/scannet/a.mp4|s4|qa"]}) == ("qa_video", "/m/scannet/a.mp4")
    assert S.row_kind({"videos": ["/m/adt/b.mp4|s4|nfp:u=0.3"]}) == ("nfp", "/m/adt/b.mp4")
    assert S.row_kind({"videos": None, "images": ["/m/hypersim/c.jpg"]}) == ("qa_image", "/m/hypersim/c.jpg")
    assert S.source_of("/m/scannet/x/a.mp4", Path("/m")) == "scannet"
    kinds = ["qa_video"] * 40 + ["nfp"] * 20 + ["qa_image"] * 3
    rows = S.pick_rows(kinds, 20, 0)
    got = Counter(kinds[i] for i in rows)
    assert len(rows) == 20 and len(set(rows)) == 20 and got["qa_image"] == 3 and got["nfp"] >= 5 and got["qa_video"] >= 10
    assert S.pick_rows(kinds, 20, 0) == rows


# ------------------------------------------------------------------------------------------------------ summary
def fake_run(tmp_path: Path, name: str, nodes: int, step_s: float, losses, extra=None) -> Path:
    run = tmp_path / name
    (run / "ckpt").mkdir(parents=True)
    (run / "sft_metrics").mkdir()
    plan = {"mode": "train", "arm": "nfp", "nodes": nodes, "world": nodes * 8, "global_batch": 256}
    (run / "plan.json").write_text(json.dumps(plan))
    hist = []
    for i, x in enumerate(losses, 1):
        h = {"step": i, "loss": x, "grad_norm": 1.0}
        h.update((extra or {}).get(i, {}))
        hist.append(h)
    hist.append({"step": len(losses), "train_loss": 1.0, "train_runtime": 9.0})
    (run / "ckpt" / "trainer_state.json").write_text(json.dumps({"log_history": hist}))
    for r in range(2):
        rows = [{"event": "train_begin"}] + [
            {"event": "step", "step": i, "step_s": (50.0 if i == 1 else step_s) + r, "max_allocated_gib": 60 + r,
             "max_reserved_gib": 70 + r, "rss_gb": 20, **({"used_gb": 900 + i} if r == 0 else {})}
            for i in range(1, len(losses) + 1)]
        (run / "sft_metrics" / f"rank{r}.jsonl").write_text("".join(json.dumps(x) + "\n" for x in rows) + '{"cut')
    (run / "sft_metrics" / "host_mem_sft-x.jsonl").write_text(json.dumps({"peak": {"used_gb": 1234.5,
                                                                                   "cgroup_gb": 1000}}) + "\n")
    return run


def test_summary(tmp_path):
    extra = {i: {"nfp_cos": 0.5 - 0.01 * i, "copy_cos": 0.45} for i in range(1, 11)}
    run = fake_run(tmp_path, "a", 1, 10.0, [3.0 - 0.1 * i for i in range(10)], extra)
    rep = S.summary(run)
    assert rep["steps_logged"] == 10 and rep["losses"]["loss"]["down"]
    assert rep["step_s"]["median"] == 11.0 and rep["step_s"]["first"] == [51.0, 11.0]
    assert rep["samples_per_s"] == round(256 / 11.0, 4)
    assert rep["gpu_peak_gib"] == {"allocated": 61, "reserved": 71}
    assert rep["host_peak_gb"]["machine_used"] == 1234.5 and rep["host_peak_gb"]["under_threshold"]
    v = rep["nfp_vs_copy"]
    assert v["steps"] == 10 and v["steps_below_copy"] == 5 and v["gap_second_half"] > v["gap_first_half"]
    assert json.loads((run / "smoke_summary.json").read_text())["samples_per_s"] == rep["samples_per_s"]


def test_scaling(tmp_path):
    reps = [S.summary(fake_run(tmp_path, f"n{n}", n, t, [1.0] * 6)) for n, t in ((1, 79.0), (2, 39.0), (8, 13.0))]
    res = S.scaling(reps)
    eff = {r["nodes"]: r["efficiency"] for r in res["rows"]}
    assert eff[1] == 1.0 and abs(eff[2] - 80 / 80 * 1.0) < 1e-9 and abs(eff[8] - 80 / (8 * 14)) < 1e-4
    assert res["verdict"] == "8 nodes (>= 0.70)"
    slow = S.summary(fake_run(tmp_path, "slow8", 8, 30.0, [1.0] * 6))
    assert S.scaling([reps[0], slow])["verdict"].startswith("stop")
    with pytest.raises(SystemExit):
        S.scaling(reps[1:])


def test_compare():
    a = [{"step": 1, "loss": 1.5}, {"step": 2, "loss": 1.25}, {"step": 3, "loss": 1.0}]
    assert S.compare(a, [dict(h) for h in a])["bitwise_equal"]
    b = [dict(h) for h in a]
    b[1]["loss"] = 1.25 + 1e-12
    r = S.compare(a, b)
    assert not r["bitwise_equal"] and r["first_different_step"] == 2 and r["equal_steps"] == 2
    assert 0 < r["max_abs_diff"] < 1e-11
    assert not S.compare(a, a[:2])["bitwise_equal"]


def test_lf_plain_installs_only_the_template(monkeypatch):
    from llamafactory.train.sft import workflow as W

    from vagen.q38s4.tools import lf_plain

    before = {k: getattr(W, k) for k in ("load_model", "SFTDataCollatorWith4DAttentionMask", "CustomSeq2SeqTrainer",
                                          "get_template_and_fix_tokenizer")}
    for k, v in before.items():
        monkeypatch.setattr(W, k, v)
    lf_plain.install()
    assert W.load_model is before["load_model"]
    assert W.SFTDataCollatorWith4DAttentionMask is before["SFTDataCollatorWith4DAttentionMask"]
    assert W.CustomSeq2SeqTrainer is before["CustomSeq2SeqTrainer"]
    assert W.get_template_and_fix_tokenizer is not before["get_template_and_fix_tokenizer"]



