"""eval/failures.py: failure-record skeletons from a SYNTHETIC events.jsonl, labelling, distribution.

The synthetic events follow agent/recorder.py's record shapes; one test also feeds the timeline
builder the recorder's real output from a mock L4 episode (helpers.make_policy).
"""

import glob
import json
import os

import pytest

from eval import failures as fl
from helpers import TASK_PROMPT, make_cfg, make_policy, run_episode
from synth import synth_obs

TASK = "turning_on_radio"
SUBTASK = {"subtask_id": "", "skill": "pick_up", "instruction": "pick up the radio", "args": {"object": "radio"},
           "target_subgoals": ["g0"], "max_steps": 300, "source": "plan.llm@astra", "evidence": "radio on the table"}


def judgment(status, failure_type=None, evidence=""):
    return {"status": status, "confidence": 0.9, "evidence": evidence, "failure_type": failure_type,
            "subgoal_updates": {}, "holding_updates": {}, "source": "check.llm@astra"}


def executor(step, instruction):
    return {"kind": "executor", "step": step, "instruction": instruction, "source": "exec.fake", "infer_ms": 5.0,
            "n": 1, "open_loop": 1, "meta": {}}


EVENTS = [
    {"_synthetic": True, "kind": "episode_start", "t": 0.0, "ctx": {"task_name": TASK}, "modules": {}, "tags": {}},
    {"kind": "decision", "slot": "goal", "step": 0, "seconds": 0.0, "out": ["g0"]},
    {"kind": "decision", "slot": "plan", "step": 0, "seconds": 1.2, "out": SUBTASK},
    {"kind": "decision", "slot": "goal", "step": 0, "seconds": 0.0, "out": 0.1},
    executor(0, "pick up the radio"),
    executor(1, "pick up the radio"),
    {"kind": "decision", "slot": "check", "step": 150, "seconds": 2.0,
     "out": judgment("in_progress", evidence="arm moving")},
    {"kind": "decision", "slot": "check", "step": 151, "seconds": 0.0,
     "out": {**judgment("unknown"), "source": "check.null"}},  # null checker: not a model judgment
    {"kind": "provider_call", "call_id": "x-c00002", "slot": "check", "step": 300, "latency_s": 60.0,
     "error": "TimeoutError: read timed out", "usage": {}},
    {"kind": "decision", "slot": "check", "step": 300, "seconds": 2.1,
     "out": judgment("failed", "grasp_failed", "gripper closed on nothing")},
    {"kind": "decision", "slot": "recover", "step": 300, "seconds": 0.0,
     "out": {"kind": "scripted", "primitive": "retract_arms", "actions": {"ndarray": [60, 23], "dtype": "float32"},
             "instruction": None, "duration_steps": 0, "reason": "grasp_failed attempt 1",
             "source": "recover.scripted"}},
    executor(360, "pick up the radio"),
    {"kind": "decision", "slot": "check", "step": 360, "seconds": 2.0, "out": judgment("done", evidence="radio held")},
    {"kind": "decision", "slot": "plan", "step": 360, "seconds": 1.0, "out": None},
    executor(360, TASK_PROMPT),
    {"kind": "episode_end", "t": 1.0, "summary": {
        "steps": 600, "subtasks": [
            {"subtask": SUBTASK, "start_step": 0, "end_step": 360, "outcome": "done", "retries": 1},
            {"subtask": {"skill": "__full_task__", "instruction": TASK_PROMPT, "source": "loop.fallback.exhausted"},
             "outcome": "open"}]}},
]


def _write_json(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f)


def make_run(root):
    """One run, two tasks' worth of episodes: radio 301 (q 0.5, events), radio 302 (q 1.0), trash 301 (infra)."""
    run = os.path.join(root, "run1")
    _write_json(os.path.join(run, "run_manifest.json"), {"_synthetic": True, "run_id": "run1", "config_hash": "h",
                                                         "config": {"run": {"name": "L4_recover", "seed": 0}}})
    _write_json(os.path.join(run, "status.json"), {"_synthetic": True, "run_id": "run1", "video_fps": 30})
    for task, iid, q, extra in ((TASK, 301, 0.5, {}), (TASK, 302, 1.0, {}),
                                ("picking_up_trash", 301, 0.0, {"failure_reason": "policy_connection_lost"})):
        _write_json(os.path.join(run, "eval_output", "json", f"{task}_{iid}_0.json"),
                    {"_synthetic": True, "task": task, "instance_id": iid, "rollout_id": 0, "steps": 600,
                     "success": q == 1.0, "q_score": {"final": q},
                     "time": {"simulator_steps": 600, "simulator_time": 20.0, "normalized_time": 1.0}, **extra})
        eid = f"run1-{task}-{iid}"
        _write_json(os.path.join(run, "episodes", eid, "summary.json"),
                    {"_synthetic": True, "episode_id": eid, "task_name": task, "tags": {"task": task, "instance": iid}})
        if (task, iid) == (TASK, 301):
            with open(os.path.join(run, "episodes", eid, "events.jsonl"), "w", encoding="utf-8") as f:
                for ev in EVENTS:
                    f.write(json.dumps(ev) + "\n")
    os.makedirs(os.path.join(run, "eval_output", "videos"))
    open(os.path.join(run, "eval_output", "videos", f"{TASK}_301_0.mp4"), "wb").close()
    return run


def test_categories_are_the_spec_list():
    assert fl.CATEGORIES == ("env_service_error", "target_recognition", "task_order", "reach_and_positioning",
                             "grasp_place", "skill_switch_handover", "recovery_ineffective", "predicate_regression",
                             "uncertain")
    assert fl.CATEGORY_LABELS_ZH["skill_switch_handover"] == "技能切换与交接"
    assert set(fl.CATEGORY_LABELS_ZH) == set(fl.CATEGORIES)


def test_timeline_from_synthetic_events():
    tl = fl.build_timeline(EVENTS, 30.0)
    assert [(m["step"], m["kind"]) for m in tl] == [
        (0, "subtask_start"), (0, "instruction"), (150, "judgment"), (300, "provider_error"), (300, "judgment"),
        (300, "recovery"), (360, "judgment"), (360, "subtask_end"), (360, "plan_exhausted"), (360, "fallback"),
        (360, "instruction")]
    assert tl[0]["evidence"] == "radio on the table" and tl[0]["source"] == "plan.llm@astra"
    failed = tl[4]
    assert failed["model_judgment"]["status"] == "failed" and failed["model_judgment"]["failure_type"] == "grasp_failed"
    assert failed["model_judgment"]["step"] == 300
    assert failed["current_skill"] == "pick_up" and failed["current_instruction"] == "pick up the radio"
    assert failed["video_time_s"] == 10.0
    assert failed["next_action"] == "step 300: recovery: scripted:retract_arms grasp_failed attempt 1"
    assert tl[2]["next_action"].startswith("step 300: recovery")
    # the open full-task fallback after the planner ran out is the skill in force until the episode ends
    assert tl[-1]["current_skill"] == "__full_task__" and tl[-1]["current_instruction"] == TASK_PROMPT
    assert tl[-1]["next_action"] == "episode end (steps 600)"
    assert tl[7]["next_action"] == "step 360: plan_exhausted: planner returned no subtask"


def proposal(step, skill, instruction):
    return {"kind": "decision", "slot": "plan", "step": step, "seconds": 1.0,
            "out": {**SUBTASK, "skill": skill, "instruction": instruction}}


def risk(step, value):
    return {"kind": "decision", "slot": "goal", "step": step, "seconds": 0.0, "out": value}


def test_timeline_rejected_proposals_and_moved_start_step():
    """A proposal the goal manager vetoes never starts; summary start_step (moved by the step-budget window
    in agent/loop.py) is not used for the start."""
    place = {**SUBTASK, "skill": "place", "instruction": "place the radio"}
    events = [proposal(0, "pick_up", "pick up the remote"), risk(0, 0.9),
              proposal(0, "pick_up", "pick up the radio"), risk(0, 0.1), executor(0, "pick up the radio"),
              {"kind": "decision", "slot": "check", "step": 450, "seconds": 1.0, "out": judgment("done")},
              proposal(450, "place", "place the radio"), risk(450, 0.0), executor(450, "place the radio"),
              {"kind": "episode_end", "t": 1.0, "summary": {"steps": 900, "subtasks": [
                  {"subtask": SUBTASK, "start_step": 300, "end_step": 450, "outcome": "done", "retries": 1},
                  {"subtask": place, "outcome": "open"}]}}]
    tl = fl.build_timeline(events)
    assert [(m["step"], m["kind"]) for m in tl] == [
        (0, "plan_rejected"), (0, "subtask_start"), (0, "instruction"), (450, "judgment"), (450, "subtask_end"),
        (450, "subtask_start"), (450, "instruction")]
    assert tl[0]["instruction"] == "pick up the remote" and tl[0]["regression_risk"] == 0.9
    assert tl[0]["current_skill"] is None and tl[0]["next_action"].startswith("step 0: subtask_start: pick_up")
    assert tl[1]["regression_risk"] == 0.1 and tl[3]["current_skill"] == "pick_up"
    assert tl[-1]["current_skill"] == "place"
    # without any summary, the last proposal of a step is taken as started (and says so)
    tl = fl.build_timeline(events[:-1])
    starts = [m for m in tl if m["kind"] == "subtask_start"]
    assert [m["instruction"] for m in starts] == ["pick up the radio", "place the radio"] and starts[0]["inferred"]
    # an open episode (no episode_end event) uses the summary.json snapshot
    tl = fl.build_timeline(events[:-1], summary=events[-1]["summary"])
    assert [(m["step"], m["kind"]) for m in tl if m["kind"] == "subtask_end"] == [(450, "subtask_end")]
    # a snapshot taken before step 450 (the evaluator reconnected afterwards) does not know the later subtask
    old_snapshot = {"open": True, "steps": 400, "subtasks": [{"subtask": SUBTASK, "outcome": "open"}]}
    tl = fl.build_timeline(events[:-1], summary=old_snapshot)
    late = next(m for m in tl if m["step"] == 450 and m["kind"] == "subtask_start")
    assert late["instruction"] == "place the radio" and late["inferred"]
    assert [m["instruction"] for m in tl if m["kind"] == "plan_rejected"] == ["pick up the remote"]


def test_skeleton_never_assigns_a_category(tmp_path):
    run = make_run(str(tmp_path))
    out = tmp_path / "failures.jsonl"
    assert fl.main(["skeleton", run, "--out", str(out)]) == 0
    recs = fl.load_records(str(out))
    # radio 302 (q = 1, budget unknown) gets no record
    assert sorted((r["task"], r["instance_id"]) for r in recs) == [("picking_up_trash", 301), (TASK, 301)]
    for r in recs:
        assert set(fl.RECORD_FIELDS) <= set(r)
        assert r["category"] == "uncertain" and r["category_zh"] == "原因不确定" and r["labeler"] == ""
        assert r["step"] is None and r["video_time_s"] is None and r["model_judgment"] is None
    radio = next(r for r in recs if r["task"] == TASK)
    assert radio["q_final"] == 0.5 and radio["video_exists"]
    assert radio["video_path"].endswith(f"videos/{TASK}_301_0.mp4")
    assert len(radio["timeline"]) == 11 and radio["events_sha256"]
    assert radio["selected_because"] == "official Q 0.5 < 1" and radio["q_budget2026"] == 0.5
    trash = next(r for r in recs if r["task"] != TASK)
    assert trash["infra_error"] and trash["timeline"] == [] and "events.jsonl not found" in trash["evidence_notes"]
    with pytest.raises(SystemExit):  # an existing file may hold labels
        fl.main(["skeleton", run, "--out", str(out)])


def test_skeleton_includes_over_budget_successes(tmp_path):
    """Official Q = 1 but over max_steps x 1 s by the agent's clock: 0 under the 2026 rule, so it gets a record."""
    run = make_run(str(tmp_path))
    _write_json(os.path.join(run, "eval_output", "json", f"{TASK}_303_0.json"),
                {"_synthetic": True, "task": TASK, "instance_id": 303, "rollout_id": 0, "steps": 90, "success": True,
                 "q_score": {"final": 1.0},
                 "time": {"simulator_steps": 90, "simulator_time": 3.0, "normalized_time": 1.0}})
    eid = f"run1-{TASK}-303"
    _write_json(os.path.join(run, "episodes", eid, "summary.json"),
                {"_synthetic": True, "episode_id": eid, "task_name": TASK, "tags": {"task": TASK, "instance": 303},
                 "wallclock_budget_s": 100.0, "elapsed_s": 150.0})
    recs = {(r["task"], r["instance_id"]): r for r in fl.skeleton_records([run])}
    over = recs[(TASK, 303)]
    assert over["budget_ok"] is False and over["q_final"] == 1.0 and over["q_budget2026"] == 0.0
    assert "over the 2026 budget" in over["selected_because"] and over["category"] == "uncertain"
    assert (TASK, 302) not in recs  # Q = 1 within budget (budget unknown here, never zeroed)
    assert recs[("picking_up_trash", 301)]["selected_because"].startswith("infra error")


def test_label_anchors_context_from_the_timeline(tmp_path):
    run = make_run(str(tmp_path))
    out = str(tmp_path / "failures.jsonl")
    fl.write_records(out, fl.skeleton_records([run]))
    rid = next(r["record_id"] for r in fl.load_records(out) if r["task"] == TASK)
    rec = fl.label(out, rid, 310, "grasp_place", "tester", outcome="gripper closed beside the handle",
                   notes="video 10.0-10.5 s")
    assert rec["video_time_s"] == pytest.approx(310 / 30, abs=1e-3)
    assert rec["current_skill"] == "pick_up" and rec["model_judgment"]["status"] == "failed"
    assert rec["next_action"] == "step 360: subtask_end: pick_up -> done (retries 1)"
    assert rec["category_zh"] == "抓放操作" and rec["labeler"] == "tester"
    second = fl.label(out, rid, 365, "skill_switch_handover", "tester", new=True)
    assert second["record_id"] == rid.replace("#0", "#1") and second["timeline_ref"] == rid
    assert second["current_instruction"] == TASK_PROMPT
    recs = fl.load_records(out)
    assert [r["record_id"] for r in recs if r["task"] == TASK] == [rid, second["record_id"]]
    with pytest.raises(SystemExit):
        fl.label(out, rid, 10, "not_a_category", "tester")


def test_summarize_distribution(tmp_path):
    run = make_run(str(tmp_path))
    out = str(tmp_path / "failures.jsonl")
    fl.write_records(out, fl.skeleton_records([run]))
    rid = next(r["record_id"] for r in fl.load_records(out) if r["task"] == TASK)
    fl.label(out, rid, 310, "grasp_place", "tester")
    fl.label(out, rid, 365, "grasp_place", "tester", new=True)
    s = fl.summarize(fl.load_records(out))
    cats = {c["category"]: c for c in s["categories"]}
    assert s["n_records"] == 3 and s["n_episodes"] == 2 and s["n_unlabelled"] == 1
    assert cats["grasp_place"]["count"] == 2 and cats["grasp_place"]["episodes"] == 1
    assert cats["grasp_place"]["share"] == pytest.approx(2 / 3) and cats["uncertain"]["count"] == 1
    assert s["per_task"][TASK]["grasp_place"] == 2 and s["per_task"]["picking_up_trash"]["uncertain"] == 1
    assert s["per_task_share"][TASK]["grasp_place"] == 1.0 and cats["uncertain"]["unlabelled"] == 1
    md = fl.render_summary(s)
    assert "抓放操作" in md and "1 unlabelled" in md and "SYNTHETIC" in md and "2 (100%)" in md
    assert fl.main(["summarize", out, "--out", str(tmp_path / "dist.md")]) == 0
    bad = fl.load_records(out)
    bad[0]["category"] = "other"
    with pytest.raises(ValueError):
        fl.summarize(bad)


def test_timeline_reads_real_recorder_output(tmp_path):
    """The recorder's actual events.jsonl from a mock L4 episode (scripted provider replies)."""
    policy, _, tmp = make_policy(make_cfg("L4_recover"), tmp=str(tmp_path))
    run_episode(policy, lambda t: synth_obs(t), 400)
    paths = glob.glob(os.path.join(tmp, "test", "episodes", "*", "events.jsonl"))
    assert len(paths) == 1
    with open(paths[0], encoding="utf-8") as f:
        tl = fl.build_timeline([json.loads(line) for line in f if line.strip()])
    kinds = [m["kind"] for m in tl]
    assert kinds[0] == "subtask_start" and "judgment" in kinds and "instruction" in kinds
    failed = [i for i, m in enumerate(tl) if m["kind"] == "judgment" and m["model_judgment"]["status"] == "failed"]
    assert failed and tl[failed[0] + 1]["kind"] == "recovery"
    assert all(m["video_time_s"] == pytest.approx(m["step"] / 30.0, abs=1e-3) for m in tl)
