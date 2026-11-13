"""data/distill.py: DistillRecords from a SYNTHETIC events.jsonl (agent/recorder.py record shapes) and from
the recorder's real output of a mock episode (scripted providers, fake executor; no model, no simulator)."""

import json
import os

import pytest

from data import distill as D
from data import schema as S
from helpers import make_cfg, make_policy, run_episode
from synth import synth_obs

TASK = "turning_on_radio"
PROMPT = "Turn on the radio receiver that's on the table in the living room."
EP = "synth-ep001-env0"
SUBGOALS = [
    {"subgoal_id": "g0", "literal": "(toggled_on ?radio_receiver.n.01_1)", "predicate": "toggled_on",
     "args": ["radio_receiver.n.01_1"], "negated": False, "option": 0, "quantified": False},
    {"subgoal_id": "g1", "literal": "(ontop ?radio_receiver.n.01_1 ?table.n.02_1)", "predicate": "ontop",
     "args": ["radio_receiver.n.01_1", "table.n.02_1"], "negated": False, "option": 0, "quantified": False},
]
PICK = {"subtask_id": "", "skill": "pick_up", "instruction": "pick up the radio", "args": {"object": "radio"},
        "target_subgoals": ["g1"], "max_steps": 300, "source": "plan.llm@astra", "evidence": "radio visible"}
FULL = {"subtask_id": "", "skill": "__full_task__", "instruction": PROMPT, "args": {}, "target_subgoals": ["g0"],
        "max_steps": None, "source": "plan.llm@astra", "evidence": "go"}


def call(step, slot, parsed, error=None, n=1):
    return {"kind": "provider_call", "call_id": f"{EP}-c{n:05d}", "slot": slot, "step": step,
            "purpose": {"plan": "plan.next_subtask", "check": "check.judge"}[slot], "provider": "astra", "model": "m",
            "system_sha256": "abc", "user_text": f"{slot} prompt at {step}",
            "images": [f"episodes/{EP}/frames/{EP}-c{n:05d}_head.jpg"],
            "response_text": "" if parsed is None else json.dumps(parsed), "parsed": parsed,
            "usage": {"input_tokens": 10},
            "latency_s": 1.5, "cost_usd": 0.001, "error": error, "stop_reason": None}


def plan_reply(skill, target, args=(), complete=False):
    return {"task_complete": complete, "skill": skill, "args": [{"name": k, "value": v} for k, v in args],
            "target_subgoals": [target], "max_steps": 300, "evidence": f"{skill} evidence"}


def check_reply(status, conf=0.9, ft="none", subgoals=()):
    return {"status": status, "confidence": conf, "evidence": f"{status} evidence", "failure_type": ft,
            "subgoals": [{"id": g, "estimate": e} for g, e in subgoals], "holding_left": "", "holding_right": "radio"}


def judgment(status, ft=None, source="check.llm@astra"):
    return {"status": status, "confidence": 0.9, "evidence": "", "failure_type": ft, "subgoal_updates": {},
            "holding_updates": {}, "source": source}


EVENTS = [
    {"_synthetic": True, "kind": "episode_start", "t": 0.0,
     "ctx": {"episode_id": EP, "task_name": TASK, "task_instruction": PROMPT,
             "goal_spec": {"task_name": TASK, "subgoals": SUBGOALS}}, "modules": {},
     "tags": {"task": TASK, "mode": "public_test", "instance_ids": "311"}},
    call(0, "plan", plan_reply("pick_up", "g1", [("object", "radio")]), n=1),
    {"kind": "decision", "slot": "plan", "step": 0, "seconds": 1.0, "out": PICK},
    {"kind": "decision", "slot": "goal", "step": 0, "seconds": 0.0, "out": 0.9},  # vetoed by the goal manager
    call(0, "plan", plan_reply("__full_task__", "g0"), n=2),
    {"kind": "decision", "slot": "plan", "step": 0, "seconds": 1.0, "out": FULL},
    {"kind": "decision", "slot": "goal", "step": 0, "seconds": 0.0, "out": 0.0},
    call(150, "check", check_reply("done", conf=0.55, subgoals=[("g0", "satisfied")]), n=3),
    {"kind": "decision", "slot": "check", "step": 150, "seconds": 2.0, "out": judgment("unknown")},  # downgraded
    call(300, "check", None, error="TimeoutError: read timed out", n=4),
    {"kind": "decision", "slot": "check", "step": 300, "seconds": 60.0,
     "out": judgment("unknown", source="check.llm.fallback")},
    call(450, "check", check_reply("failed", ft="grasp_failed"), n=5),
    {"kind": "decision", "slot": "check", "step": 450, "seconds": 2.0, "out": judgment("failed", "grasp_failed")},
    {"kind": "decision", "slot": "recover", "step": 450, "seconds": 0.0,
     "out": {"kind": "scripted", "primitive": "back_up"}},
    call(600, "check", check_reply("done", subgoals=[("g0", "satisfied")]), n=6),
    {"kind": "decision", "slot": "check", "step": 600, "seconds": 2.0, "out": judgment("done")},
    call(600, "plan", plan_reply("__full_task__", "g0", complete=True), n=7),
    {"kind": "decision", "slot": "plan", "step": 600, "seconds": 1.0, "out": None},
    call(750, "check", check_reply("in_progress", conf=0.7), n=8),
    {"kind": "decision", "slot": "check", "step": 750, "seconds": 2.0, "out": judgment("in_progress")},
    {"kind": "episode_end", "t": 1.0, "summary": {"steps": 900, "subtasks": [
        {"subtask": {**FULL, "subtask_id": "st1"}, "start_step": 0, "end_step": 600, "outcome": "done", "retries": 1},
        {"subtask": {"subtask_id": "st2", "skill": "__full_task__", "instruction": PROMPT, "args": {},
                     "target_subgoals": [], "source": "loop.fallback.exhausted"}, "outcome": "open"}]}},
]


def write_run(root, events=EVENTS, name="run1"):
    ep_dir = os.path.join(root, name, "episodes", EP)
    os.makedirs(ep_dir)
    with open(os.path.join(ep_dir, "events.jsonl"), "w") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")
    with open(os.path.join(root, name, "run_manifest.json"), "w") as f:
        json.dump({"_synthetic": True, "run_id": name}, f)
    return os.path.join(root, name)


def test_export_synthetic_episode(tmp_path):
    run = write_run(str(tmp_path))
    recs, warnings = D.export_run(run)
    assert warnings == []
    assert len(recs) == 8 and all(r.synthetic for r in recs)
    by_call = {r.call_id[-6:]: r for r in recs}
    assert all(r.run_id == "run1" and r.episode_id == EP and r.instance_id == 311 and r.errors() == [] for r in recs)
    assert len({r.record_id for r in recs}) == 8

    first, second = by_call["c00001"], by_call["c00002"]
    assert first.outputs.decision_type == "subtask_choice"
    assert first.outputs.subtask == {"skill": "pick_up", "args": {"object": "radio"}, "target_subgoals": ["g1"],
                                     "max_steps": 300}
    assert first.outputs.module_return == PICK and second.outputs.module_return == FULL  # matched per invocation
    assert first.inputs.subtask_in_force is None and second.inputs.subtask_in_force is None  # planning: no subtask runs
    assert first.inputs.subgoal_literals == {"g0": SUBGOALS[0]["literal"], "g1": SUBGOALS[1]["literal"]}
    assert first.inputs.image_refs == (f"episodes/{EP}/frames/{EP}-c00001_head.jpg",)
    assert first.source.events_path == f"episodes/{EP}/events.jsonl" and first.source.line == 1

    done_low = by_call["c00003"]
    assert done_low.outputs.status == "done" and done_low.outputs.continue_switch == "switch"
    assert done_low.outputs.module_return["status"] == "unknown"  # what the loop actually got
    assert done_low.outputs.subgoal_estimates == {"g0": "satisfied"}
    assert done_low.outputs.holding == {"left": "", "right": "radio"}
    assert done_low.visible_evidence == "done evidence"
    assert done_low.inputs.subtask_in_force["subtask_id"] == "st1"
    assert done_low.inputs.subtask_in_force["inferred_from"] == "episode_summary"

    err = by_call["c00004"]
    assert err.outputs.error.startswith("TimeoutError") and err.outputs.parsed is None and err.outputs.status is None

    failed = by_call["c00005"]
    assert failed.outputs.decision_type == "failure_verdict" and failed.outputs.continue_switch == "recover"
    assert failed.outputs.failure_type == "grasp_failed"

    assert by_call["c00006"].inputs.subtask_in_force["subtask_id"] == "st1"  # the check that ended st1 at 600
    claim = by_call["c00007"]
    assert claim.outputs.task_complete is True and claim.outputs.module_return is None
    late = by_call["c00008"]
    assert late.inputs.subtask_in_force["subtask_id"] == "st2" and late.inputs.subtask_in_force["target_subgoals"] == []
    assert all(r.truth_label is None and r.correct is None and r.truth_source is None for r in recs)


def test_without_summary_the_last_proposal_is_used(tmp_path):
    run = write_run(str(tmp_path), events=EVENTS[:-1])
    recs, _ = D.export_run(run)
    late = {r.call_id[-6:]: r for r in recs}["c00008"]
    assert late.inputs.subtask_in_force["inferred_from"] == "planner returned no subtask"
    early = {r.call_id[-6:]: r for r in recs}["c00003"]
    assert early.inputs.subtask_in_force["inferred_from"] == "last_plan_proposal"
    assert early.inputs.subtask_in_force["skill"] == "__full_task__"


def test_cli_export_skip_errors_and_no_overwrite(tmp_path, capsys):
    run = write_run(str(tmp_path))
    out = str(tmp_path / "distill.jsonl")
    assert D.main(["export", run, "--out", out, "--skip-errors", "--round", "2"]) == 0
    recs = S.read_jsonl(out, S.DistillRecord)
    assert len(recs) == 7 and all(r.round == 2 and not r.outputs.error for r in recs)
    assert "7 records" in capsys.readouterr().err
    assert D.main(["export", run, "--out", out]) == 1
    assert D.main(["export", run, "--out", out, "--force"]) == 0
    assert len(S.read_jsonl(out, S.DistillRecord)) == 8


def test_export_from_mock_episode_recorder_output(tmp_path):
    """The recorder's real event shapes from a mock S7 episode (scripted provider replies)."""
    policy, _, root = make_policy(make_cfg("S7_min_loop"), tmp=str(tmp_path))
    policy.recorder.tags = {"task": "turning_on_radio", "mode": "public_test", "instance_ids": "312"}  # eval/launch.py
    run_episode(policy, lambda t: synth_obs(t), 400)
    recs, warnings = D.export_run(os.path.join(root, "test"))
    assert warnings == [] and {r.instance_id for r in recs} == {312}
    kinds = [(r.step, r.outputs.decision_type, r.outputs.status) for r in recs]
    assert kinds == [(0, "subtask_choice", None), (150, "continue_switch", "in_progress"),
                     (300, "failure_verdict", "failed"), (360, "continue_switch", "done"),
                     (360, "subtask_choice", None)]
    assert all(r.inputs.task_name == "turning_on_radio" and r.inputs.subgoal_literals for r in recs)
    assert all(os.path.isfile(os.path.join(root, "test", p)) for r in recs for p in r.inputs.image_refs)
    checks = [r for r in recs if r.slot == "check"]
    assert all(r.inputs.subtask_in_force["subtask_id"] == "st1" for r in checks)
    assert all(r.outputs.module_return["source"] == "check.llm@astra" for r in checks)
    assert not any(r.synthetic for r in recs)  # mock, but not marked synthetic by the recorder


def test_decision_type_mapping():
    assert D.decision_type("plan", "plan.next_subtask", {}) == "subtask_choice"
    assert D.decision_type("check", "check.judge", {"status": "failed"}) == "failure_verdict"
    assert D.decision_type("check", "check.judge", None) == "continue_switch"
    assert D.decision_type("recover", "recover.propose", {}) == "recovery_choice"
    assert D.decision_type("", "probe", {}) == "other"


def test_missing_run_dir():
    with pytest.raises(FileNotFoundError):
        D.export_run("/nonexistent/run")


def test_malformed_replies_do_not_break_the_export(tmp_path):
    odd_plan = {"skill": 3, "args": "object=radio", "target_subgoals": "g0", "task_complete": "yes", "confidence": True}
    odd_check = {"status": "finished", "subgoals": {"g0": "satisfied"}, "holding_left": 5, "failure_type": None}
    events = EVENTS[:1] + [call(0, "plan", odd_plan, n=1), call(30, "check", odd_check, error=7, n=2)]
    out = str(tmp_path / "d.jsonl")
    assert D.main(["export", write_run(str(tmp_path), events=events), "--out", out]) == 0
    plan, check = S.read_jsonl(out, S.DistillRecord)  # written records pass the strict loader
    assert plan.outputs.subtask["target_subgoals"] == [] and plan.outputs.subtask["args"] == {}
    assert plan.outputs.task_complete is None and plan.outputs.confidence == 1.0
    assert check.outputs.status is None and check.outputs.continue_switch is None
    assert check.outputs.subgoal_estimates == {} and check.outputs.holding == {} and check.outputs.error == "7"
