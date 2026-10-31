"""diagnostics/join.py and diagnostics/truth.py on a SYNTHETIC run directory (official-format result JSONs,
agent summaries / events in agent/recorder.py shapes) and a SYNTHETIC truth file. No simulator."""

import io
import json
import os

import pytest

from diagnostics import join as J
from diagnostics import truth as T

TASK = "turning_on_radio"
RUN = "run1"
LITS = ["(toggled_on radio_receiver.n.01_1)", "(ontop radio_receiver.n.01_1 table.n.02_1)"]


def dump(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f)


def judgment(step, status, source="check.llm@astra"):
    return {"kind": "decision", "slot": "check", "step": step, "seconds": 1.0,
            "out": {"status": status, "confidence": 0.9, "evidence": "", "failure_type": None, "subgoal_updates": {},
                    "holding_updates": {}, "source": source}}


def episode(run_dir, env, events, fallbacks=0):
    eid = f"{RUN}-ep001-env{env}"
    d = os.path.join(run_dir, "episodes", eid)
    summary = {"_synthetic": True, "episode_id": eid, "run_id": RUN, "task_name": TASK,
               "tags": {"task": TASK, "mode": "public_test", "instance_ids": "311,312,313"}, "steps": 900,
               "subtasks": [], "recoveries": 1, "replans": 0, "fallbacks": fallbacks, "provider_calls": 3,
               "provider_errors": 0, "cost_usd": 0.01, "plan_exhausted": False}
    dump(os.path.join(d, "summary.json"), summary)
    with open(os.path.join(d, "events.jsonl"), "w") as f:
        for ev in [{"_synthetic": True, "kind": "episode_start", "t": 0, "ctx": {"episode_id": eid}}] + events:
            f.write(json.dumps(ev) + "\n")
    return eid


@pytest.fixture
def run_dir(tmp_path):
    run = str(tmp_path / RUN)
    dump(os.path.join(run, "status.json"), {"_synthetic": True, "run_id": RUN, "task": TASK, "mode": "public_test",
                                            "instance_ids": [311, 312, 313], "num_rollouts": 1, "state": "done",
                                            "video_fps": 30})
    dump(os.path.join(run, "run_manifest.json"),
         {"_synthetic": True, "run_id": RUN, "config": {"run": {"name": "S7", "seed": 0}}})
    for iid, q in ((311, 1.0), (312, 0.5), (313, 0.0)):
        dump(os.path.join(run, "eval_output", "json", f"{TASK}_{iid}_0.json"),
             {"_synthetic": True, "task": TASK, "instance_id": iid, "rollout_id": 0, "steps": 900, "success": q == 1.0,
              "q_score": {"final": q},
              "time": {"simulator_steps": 900, "simulator_time": 30.0, "normalized_time": 1.0}})
    episode(run, 0, [judgment(150, "in_progress"), judgment(300, "failed"), judgment(450, "done"),
                     judgment(451, "unknown", source="check.null")], fallbacks=1)
    claim = {"kind": "decision", "slot": "plan", "step": 600, "seconds": 1.0, "out": None}
    episode(run, 1, [judgment(150, "done"), claim, judgment(750, "in_progress")], fallbacks=2)
    episode(run, 2, [{"kind": "executor", "step": 0, "instruction": "x", "source": "exec", "infer_ms": 1, "n": 1,
                      "open_loop": 1, "meta": {}}])
    return run


def write_truth(path, eid, sat_by_step, consistent=True, replay_q=0.5):
    buf = io.StringIO()
    w = T.TruthWriter(buf, eid, [LITS], {"_synthetic": True, "task": TASK, "instance_id": 312})
    steps = sorted(sat_by_step)
    w.start([[i in sat_by_step[0] for i in range(len(LITS))]], success=False)
    for s in steps[1:]:
        w.record(s, [[i in sat_by_step[s] for i in range(len(LITS))]], success=False)
    w.finish(replay_q_final=replay_q, official_q_final=0.5, consistent=consistent)
    with open(path, "w") as f:
        f.write(buf.getvalue())
    return path


def test_join_records(run_dir, tmp_path):
    truth = write_truth(str(tmp_path / "t.jsonl"), f"{RUN}-ep001-env1", {0: set(), 150: {0}, 300: {0, 1}, 450: {1}})
    records, warnings = J.join_runs([run_dir], truth_paths=[truth])
    assert not [w for w in warnings if "not found" in w]
    by_iid = {r["instance_id"]: r for r in records}
    assert set(by_iid) == {311, 312, 313} and all(r["synthetic"] for r in records)

    ok = by_iid[311]
    assert ok["success"] is True and ok["n_verdicts"] == 3  # the null checker's verdict is not a model verdict
    assert ok["last_decisive_verdict"]["status"] == "done" and ok["checker_consistent"] is True
    assert ok["fallbacks_before_success"] == 1 and ok["verdict_counts"] == {"in_progress": 1, "failed": 1, "done": 1}
    assert ok["truth"] is None

    partial = by_iid[312]
    assert partial["checker_consistent"] is False  # said done, official failure
    assert partial["last_verdict"]["status"] == "in_progress" and partial["last_decisive_verdict"]["step"] == 150
    assert partial["planner_complete_claims"] == [600] and partial["fallbacks_before_success"] is None
    t = partial["truth"]
    assert t["consistent"] is True and t["n_literals"] == 2 and t["ever_satisfied"] == 2
    assert t["n_regressions"] == 1 and t["regressions"][0] == {"literal": LITS[0], "true_step": 150, "false_step": 450}
    assert t["max_q_now"] == 1.0

    l0 = by_iid[313]
    assert l0["n_verdicts"] == 0 and l0["checker_consistent"] is None and l0["timeline_moments"] == {"instruction": 1}

    md = J.render_markdown(records)
    assert "| turning_on_radio | 3 | 3 | 1/2 | 1/2 | 1 | 3 | 1/1 | 1 |" in md


def test_cli_writes_records_and_markdown(run_dir, tmp_path):
    out, md = str(tmp_path / "d.jsonl"), str(tmp_path / "d.md")
    assert J.main([run_dir, "--out", out, "--md", md]) == 0
    lines = open(out).read().splitlines()
    assert len(lines) == 3 and all(json.loads(x)["run_id"] == RUN for x in lines)
    assert "Diagnostics only" in open(md).read()


def test_local_root_events_take_precedence(run_dir, tmp_path):
    local = tmp_path / "local"
    eid = f"{RUN}-ep001-env2"
    d = local / RUN / "episodes" / eid
    d.mkdir(parents=True)
    (d / "events.jsonl").write_text(json.dumps(judgment(30, "done")) + "\n")
    records, _ = J.join_runs([run_dir], local_root=str(local))
    rec = next(r for r in records if r["episode_id"] == eid)
    assert rec["events_path"] == str(d / "events.jsonl") and rec["checker_consistent"] is False


def test_truth_track_queries(tmp_path):
    path = write_truth(str(tmp_path / "t.jsonl"), "e", {0: {1}, 30: {0, 1}, 60: set(), 90: {0}}, consistent=False)
    track = T.load_truth([path])["e"]
    assert track.consistent is False and track.literals == LITS and track.options == [[0, 1]]
    assert track.at(45) is None  # default: exact step only
    assert track.at(45, max_lag=15)["step"] == 30 and track.at(45, max_lag=14) is None
    assert track.at(30)["step"] == 30 and track.at(-1, max_lag=100) is None
    # newly satisfied only: literal 1 was already true at the start
    assert [r["q_now"] for r in track.records] == [0.0, 0.5, 0.0, 0.5]
    assert track.regressions() == [{"literal": LITS[0], "true_step": 30, "false_step": 60},
                                   {"literal": LITS[1], "true_step": 0, "false_step": 60}]
    assert track.first_true_steps() == {LITS[1]: 0, LITS[0]: 30}
    with pytest.raises(ValueError, match="already loaded"):
        T.load_truth([path, path])


def test_canonical_literals():
    assert T.canonical_from_text("(toggled_on ?radio_receiver.n.01_1)") == LITS[0]
    assert T.canonical_from_text("(not (open ?cabinet.n.01_1))") == "(not (open cabinet.n.01_1))"
    assert T.canonical_literal(["not", ["open", "cabinet.n.01_1"]]) == "(not (open cabinet.n.01_1))"
    for text in ("(forall (?x - can.n.01) (inside ?x ?bin.n.01_1))", "(or (a x) (b y))", "(not (and (a x)))", "",
                 "atom"):
        assert T.canonical_from_text(text) is None


def test_q_mirror_matches_official_semantics():
    assert T.compute_q_score(True, [[False]], [[False]]) == 1.0
    assert T.compute_q_score(False, [[True, True], [True]], [[True, False], [False]]) == 1.0
    assert T.compute_q_score(False, [[True, False]], [[False, False]]) == 0.5
    assert T.compute_q_score(False, [], []) == 0.0
