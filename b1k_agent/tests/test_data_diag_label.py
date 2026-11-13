"""diagnostics/label_distill.py on SYNTHETIC DistillRecords and a SYNTHETIC truth file."""

import io

import pytest

from data import schema as S
from diagnostics import label_distill as L
from diagnostics import truth as T

LITS = ["(toggled_on radio_receiver.n.01_1)", "(ontop radio_receiver.n.01_1 table.n.02_1)"]
SUBGOALS = {"g0": "(toggled_on ?radio_receiver.n.01_1)", "g1": "(ontop ?radio_receiver.n.01_1 ?table.n.02_1)",
            "g2": "(forall (?x - can.n.01) (inside ?x ?bin.n.01_1))"}
EP = "ep-env0"


def truth_file(tmp_path, consistent=True, success_at=None, name="t.jsonl"):
    """g0 true from step 300 on, g1 true from step 600 on; records at 0, 150, 300, 450, 600."""
    buf = io.StringIO()
    w = T.TruthWriter(buf, EP, [LITS], {"_synthetic": True})
    w.start([[False, False]], success=False)
    for s in (150, 300, 450, 600):
        w.record(s, [[s >= 300, s >= 600]], success=(success_at is not None and s >= success_at))
    w.finish(consistent=consistent)
    path = tmp_path / name
    path.write_text(buf.getvalue())
    return str(path)


def rec(step, status=None, targets=("g0",), estimates=None, dtype=None, task_complete=None, error=None, n=0):
    dtype = dtype or ("failure_verdict" if status == "failed" else "continue_switch")
    parsed = None if error else {"status": status}
    return S.DistillRecord(
        record_id=f"{EP}#L{n or step}", call_id="c", run_id="run", episode_id=EP, step=step,
        slot="check" if status else "plan", purpose="check.judge", provider="astra", model="m",
        inputs=S.DistillInputs(user_text="u", subgoal_literals=SUBGOALS, subtask_in_force=None if targets is None
                               else {"skill": "x", "target_subgoals": list(targets)}),
        outputs=S.DistillOutputs(decision_type=dtype, status=status, task_complete=task_complete,
                                 subgoal_estimates=estimates or {}, parsed=parsed, error=error),
        visible_evidence="", source=S.DistillSource(events_path="e", events_sha256="0", line=1), synthetic=True)


@pytest.fixture
def tracks(tmp_path):
    return T.load_truth([truth_file(tmp_path)])


@pytest.mark.parametrize("record,label,correct", [
    (rec(450, "done"), "done", True),
    (rec(150, "done"), "not_done", False),  # premature done
    (rec(150, "in_progress"), "not_done", True),
    (rec(450, "in_progress"), "done", False),  # missed completion
    (rec(150, "failed"), "not_done", True),
    (rec(450, "unknown"), "done", None),  # abstained
    (rec(450, "done", targets=("g0", "g1")), "not_done", False),
    (rec(600, "done", targets=("g0", "g1")), "done", True),
    (rec(600, "done", targets=("g2",)), "unknown", None),  # quantified subgoal: not mappable
    (rec(600, "done", targets=()), "not_done", False),  # no targets: whole goal; never success in this episode
    (rec(600, None, targets=None, dtype="subtask_choice", task_complete=True), "not_done", False),
    (rec(600, None, targets=None, dtype="subtask_choice", task_complete=False), "not_done", True),
])
def test_labels(tracks, record, label, correct):
    out = L.label_record(record, tracks[EP])
    assert (out.truth_label, out.correct) == (label, correct)
    assert out.truth_source.startswith("replay_truth:") and out.errors() == []
    assert out.inputs == record.inputs and out.outputs == record.outputs  # labels never change the data itself


def test_detail_has_targets_and_subgoal_estimates(tracks):
    estimates = {"g0": "satisfied", "g1": "satisfied", "g2": "unknown"}
    out = L.label_record(rec(450, "done", targets=("g0",), estimates=estimates), tracks[EP])
    d = out.truth_detail
    assert d["truth_step"] == 450 and d["lag"] == 0 and d["targets"] == {"g0": {"literal": LITS[0], "truth": True}}
    assert d["subgoal_estimates"] == {"g0": {"estimate": "satisfied", "truth": True, "agree": True},
                                      "g1": {"estimate": "satisfied", "truth": False, "agree": False},
                                      "g2": {"estimate": "unknown", "truth": None, "agree": None}}


def test_lag_unlabelled_and_divergent(tmp_path, tracks):
    assert L.label_record(rec(500, "done"), tracks[EP]).truth_label == "unknown"
    lagged = L.label_record(rec(500, "done"), tracks[EP], max_lag=60)
    assert lagged.truth_label == "done" and lagged.truth_detail["lag"] == 50
    errored = rec(450, error="TimeoutError")
    assert L.label_record(errored, tracks[EP]) == errored
    other = rec(450, "done", dtype="recovery_choice")
    assert L.label_record(other, tracks[EP]) == other
    assert L.label_record(rec(450, "done"), None).truth_label is None

    divergent = T.load_truth([truth_file(tmp_path, consistent=False, name="d.jsonl")])[EP]
    skipped = L.label_record(rec(450, "done"), divergent)
    assert skipped.truth_label is None and "did not reproduce" in skipped.truth_detail["skipped"]
    assert L.label_record(rec(450, "done"), divergent, allow_divergent=True).correct is True


def test_cli(tmp_path, capsys):
    src = tmp_path / "distill.jsonl"
    S.write_jsonl(str(src), [rec(150, "done", n=1), rec(450, "done", n=2), rec(450, "in_progress", n=3)])
    out = tmp_path / "labelled.jsonl"
    assert L.main([str(src), "--truth", truth_file(tmp_path), "--out", str(out)]) == 0
    got = S.read_jsonl(str(out), S.DistillRecord)
    assert [r.correct for r in got] == [False, True, False]
    assert "continue_switch:done: 2 records, 2 labelled, correct 1, wrong 1" in capsys.readouterr().err
