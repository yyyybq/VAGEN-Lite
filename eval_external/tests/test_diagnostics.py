from eval_external.diagnostics.failure_breakdown import breakdown, classify
from eval_external.diagnostics.ownenv_eval import reorder


def _row(mets, end, progress=None):
    progress = progress or [0.1] * len(mets)
    return {"end_reason": end, "trajectory": [{"met": m, "progress": p, "failed_gates": [] if m else ["distance"]}
                                              for m, p in zip(mets, progress)]}


def test_classes():
    assert classify(_row([False, True], "success_auto")) == "reached"
    assert classify(_row([True, False], "max_primitive_steps")) == "reached_then_left"
    assert classify(_row([False, False], "low_info")) == "ended_early"
    assert classify(_row([False, False], "max_primitive_steps")) == "out_of_budget"


def test_breakdown_counts_and_never_met_details():
    rows = [_row([False, True], "success_auto"), _row([False, False], "max_primitive_steps", [0.5, 0.95]),
            _row([False], "low_info", [0.2]), {"index": 3, "error": "RuntimeError: x"}]
    b = breakdown(rows)
    assert b["n"] == 3 and b["errors"] == 1
    assert [b["classes"][c]["n"] for c in ("reached", "reached_then_left", "ended_early", "out_of_budget")] == [1, 0, 1, 1]
    assert b["never_met"]["peak_progress_at_least_0.9"] == 1
    assert b["never_met"]["failed_gates_at_end"] == {"distance": 2}


def test_history_modes():
    earlier = [("t0", "i0", "a0"), ("t1", "i1", "a1"), ("t2", "i2", "a2"), ("t3", "i3", "a3")]
    assert reorder(earlier, "normal", "u") == earlier
    assert reorder(earlier, "drop", "u") == []
    s = reorder(earlier, "shuffle", "u")
    assert sorted(s) == sorted(earlier) and s == reorder(earlier, "shuffle", "u")  # same turns, deterministic
    assert any(reorder(earlier, "shuffle", f"u{k}") != earlier for k in range(5))  # actually reordered


def test_lenient_rewrite_only_wraps_untagged_answer_lines():
    from eval_external.diagnostics.ownenv_eval import lenient_rewrite

    reply = "I observe the cabinet ... then move forward about 1.2 meters.\nAnswer: move_forward|move_right|move_forward"
    assert lenient_rewrite(reply) == "<action>move_forward|move_right|move_forward|</action>"
    assert lenient_rewrite("Answer: <action>turn_left|</action>") is None   # already tagged: untouched
    assert lenient_rewrite("I would go forward.") is None                     # no answer line: untouched
    assert lenient_rewrite("Final answer: turn_left, move_forward.") == "<action>turn_left|move_forward|</action>"


def test_queue_lenient_item():
    from eval_external.sco import queue as Q

    it = Q.item("step2", "SI-7B", "OwnEnv-lenient")
    assert it.run_dir.name == "SI-7B__lenient" and it.command[-4:] == ["SI-7B__lenient", "SI-7B", "normal", "--lenient"]
