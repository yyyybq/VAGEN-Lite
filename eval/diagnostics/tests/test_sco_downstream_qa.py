"""eval/diagnostics/sco_downstream_qa.py and the chain's diagnostics step with plan "downstream_qa" (the reward line,
2026-11-05): the X0 / G0 job asks every model with the downstream line's q0_ask.py, the scores come from its analyze
scripts; everything in temporary folders."""

import json
from pathlib import Path

import pytest

from data_gen.interiorgs_v10 import job_sco
from data_gen.interiorgs_v10 import train_chain as T
from eval.diagnostics import sco_downstream_qa as DQ


def _sets(tmp_path):
    x0 = tmp_path / "x0.jsonl"
    g0 = tmp_path / "g0.jsonl"
    x0.write_text("".join(json.dumps({"id": f"x{i}", "prompt": "?", "image_path": "/i.png"}) + "\n" for i in range(3)))
    g0.write_text("".join(json.dumps({"id": f"g{i}", "prompt": "?", "image_path": "/i.png"}) + "\n" for i in range(2)))
    return {"sets": {"x0": str(x0), "g0": str(g0)},
            "reference": {"x0": str(tmp_path / "x0_Q3B.jsonl"), "g0": str(tmp_path / "g0_Q3B.jsonl")},
            "downstream": str(tmp_path / "downstream"), "python": "/starvla/bin/python"}


def _answer(out: Path, name: str, dq, partial=False):
    out.mkdir(parents=True, exist_ok=True)
    for key, questions in dq["sets"].items():
        ids = [json.loads(l)["id"] for l in Path(questions).read_text().splitlines()]
        if partial:
            ids = ids[:1]
        (out / f"{key}_{name}.jsonl").write_text("".join(json.dumps({"id": i, "reply": "A"}) + "\n" for i in ids))


def _fake_scores(cmd, cwd, log):
    out = Path(cmd[cmd.index("--out") + 1])
    answers = cmd[cmd.index("--answers") + 1:]
    labels = [a.split("=", 1)[0] for a in answers if "=" in a and not a.startswith("--")]
    out.write_text(json.dumps({label: {"all": {"correct": 2, "n": 3, "accuracy": 2 / 3, "unparsed": 0}}
                               for label in labels}))
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(" ".join(cmd))


def test_prepare_answered_and_run_name(tmp_path, monkeypatch):
    dq = _sets(tmp_path)
    monkeypatch.setattr(job_sco, "pick_pool", lambda pools=job_sco.POOLS: "h800")
    out = tmp_path / "out"
    text = DQ.prepare(tmp_path / "run", dq, {"a": "/m/a", "b": "/m b"}, "queue", out).read_text()
    lines = [l for l in text.splitlines() if "q0_ask.py" in l]
    assert len(lines) == 2
    assert lines[0] == (f"/starvla/bin/python downstream/x/q0_ask.py --model Q3B --model-path /m/a --smoke 2 "
                        f"--ask {Path(dq['sets']['x0']).resolve()}={out.resolve() / 'x0_a.jsonl'} "
                        f"--ask {Path(dq['sets']['g0']).resolve()}={out.resolve() / 'g0_a.jsonl'}")
    assert "--model-path '/m b'" in lines[1]  # quoted
    assert f"cd {tmp_path / 'downstream'}" in text and "HF_HUB_OFFLINE=1" in text
    prepared = json.loads((tmp_path / "run" / "job" / "prepared.json").read_text())
    assert (prepared["pool"], prepared["gpus"], prepared["mode_args"][0]) == ("h800", 1, "downstream-qa")
    assert prepared["models"] == {"a": "/m/a", "b": "/m b"}
    DQ.prepare(tmp_path / "run2", dq, {"a": "/m/a"}, "zoetrope", out)
    assert json.loads((tmp_path / "run2" / "job" / "prepared.json").read_text())["pool"] == "zoetrope"
    assert not DQ.answered(out, "a", dq)
    _answer(out, "a", dq, partial=True)
    assert not DQ.answered(out, "a", dq)
    _answer(out, "a", dq)
    assert DQ.answered(out, "a", dq)
    assert DQ.run_name(["b", "a"]) == DQ.run_name(["a", "b"]) != DQ.run_name(["a"])
    with pytest.raises(ValueError):
        DQ.prepare(tmp_path / "bad", dict(dq, sets={"x9": "/q"}, reference={"x9": "/r"}), {"a": "/m"}, "h800", out)
    with pytest.raises(ValueError):
        DQ.prepare(tmp_path / "bad", dict(dq, python=""), {"a": "/m"}, "h800", out)


def test_analyze_uses_the_downstream_scripts(tmp_path):
    dq = _sets(tmp_path)
    out = tmp_path / "out"
    _answer(out, "a", dq)
    calls = []

    def runner(cmd, cwd, log):
        calls.append((cmd, cwd))
        _fake_scores(cmd, cwd, log)

    summary = DQ.analyze(out, ["a"], dq, logs=tmp_path / "logs", runner=runner)
    x0, g0 = [c for c, _ in calls]
    assert x0[:4] == ["/starvla/bin/python", "-B", "downstream/x/x0b_qa.py", "analyze"]
    assert f"Q3B={dq['reference']['x0']}" in x0 and f"Q3_a={out / 'x0_a.jsonl'}" in x0
    assert x0[x0.index("--reference") + 1] == "Q3B" and "--reference" not in g0
    assert g0[2] == "downstream/x/g0_qa.py" and {cwd for _, cwd in calls} == {dq["downstream"]}
    assert sorted(summary["models"]) == ["Q3B (reference)", "a"]
    assert summary["models"]["a"]["x0"] == {"correct": 2, "n": 3, "accuracy": 2 / 3, "unparsed": 0}
    assert json.loads((out / "summary.json").read_text())["models"]["a"]["g0"]["n"] == 3


def test_chain_diagnostics_asks_the_unanswered_models_only(tmp_path, monkeypatch):
    dq = _sets(tmp_path)
    plan = {"tr": str(tmp_path / "tr"), "sft_dir": str(tmp_path / "sft"), "eval_only": True, "eval_sets": {},
            "pool": "zoetrope",
            "eval": {"models": {"a": {"model": "/m/a", "options": []}, "b": {"model": "/m/b", "options": []}}},
            "diagnostics": {"failure_breakdown": False, "probes": {}, "downstream_qa": dq}}
    chain = T.Chain(plan, sleep=lambda s: None)
    out = Path(plan["tr"]) / "diag" / "downstream_qa"
    _answer(out, "b", dq)  # b was answered before: only a is asked
    submitted = []

    def submit(run, job_dir="job"):
        submitted.append(json.loads((run / job_dir / "prepared.json").read_text())["models"])
        (run / job_dir / "submission.json").write_text(json.dumps({"job_id": "pt-fake"}))
        return "pt-fake"

    def wait_all(step, runs, on_retry=None):
        for run in runs.values():
            for name in json.loads((run / "job" / "prepared.json").read_text())["models"]:
                _answer(out, name, dq)

    monkeypatch.setattr(job_sco, "submit", submit)
    monkeypatch.setattr(chain, "wait_all", wait_all)
    monkeypatch.setattr(DQ, "_run", _fake_scores)
    chain.diagnostics()
    assert submitted == [{"a": "/m/a"}]
    assert (Path(plan["tr"]) / "diag" / "downstream_qa_runs" / DQ.run_name(["a"]) / "job" / "launch.sh").is_file()
    summary = json.loads((out / "summary.json").read_text())
    assert sorted(summary["models"]) == ["Q3B (reference)", "a", "b"]
    assert chain.state["steps"]["diagnostics"]["downstream_qa"] == ["g0", "x0"]
    chain.diagnostics()  # everything answered: nothing submitted again
    assert submitted == [{"a": "/m/a"}]
