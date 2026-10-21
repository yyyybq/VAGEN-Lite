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
    {"kind": "decision", "slot": "check", "step": 150, "seconds": 2.0, "out": judgment("in_progress", evidence="arm moving")},
    {"kind": "decision", "slot": "check", "step": 151, "seconds": 0.0,
     "out": {**judgment("unknown"), "source": "check.null"}},  # null checker: not a model judgment
    {"kind": "provider_call", "call_id": "x-c00002", "slot": "check", "step": 300, "latency_s": 60.0,
     "error": "TimeoutError: read timed out", "usage": {}},
    {"kind": "decision", "slot": "check", "step": 300, "seconds": 2.1,
     "out": judgment("failed", "grasp_failed", "gripper closed on nothing")},
    {"kind": "decision", "slot": "recover", "step": 300, "seconds": 0.0,
     "out": {"kind": "scripted", "primitive": "retract_arms", "actions": {"ndarray": [60, 23], "dtype": "float32"},
             "instruction": None, "duration_steps": 0, "reason": "grasp_failed attempt 1", "source": "recover.scripted"}},
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
        (300, "recovery"), (360, "judgment"), (360, "subtask_end"), (360, "plan_exhausted"), (360, "instruction")]
    failed = tl[4]
    assert failed["model_judgment"]["status"] == "failed" and failed["model_judgment"]["failure_type"] == "grasp_failed"
    assert failed["current_skill"] == "pick_up" and failed["current_instruction"] == "pick up the radio"
    assert failed["video_time_s"] == 10.0
    assert failed["next_action"] == "step 300: recovery: scripted:retract_arms grasp_failed attempt 1"
    assert tl[2]["next_action"].startswith("step 300: recovery")
    assert tl[-1]["current_instruction"] == TASK_PROMPT and tl[-1]["next_action"] == "episode end (steps 600)"


def test_skeleton_never_assigns_a_category(tmp_path):
    run = make_run(str(tmp_path))
    out = tmp_path / "failures.jsonl"
    assert fl.main(["skeleton", run, "--out", str(out)]) == 0
    recs = fl.load_records(str(out))
    assert sorted((r["task"], r["instance_id"]) for r in recs) == [("picking_up_trash", 301), (TASK, 301)]  # q=1 skipped
    for r in recs:
        assert set(fl.RECORD_FIELDS) <= set(r)
        assert r["category"] == "uncertain" and r["category_zh"] == "原因不确定" and r["labeler"] == ""
        assert r["step"] is None and r["video_time_s"] is None and r["model_judgment"] is None
    radio = next(r for r in recs if r["task"] == TASK)
    assert radio["q_final"] == 0.5 and radio["video_exists"] and radio["video_path"].endswith(f"videos/{TASK}_301_0.mp4")
    assert len(radio["timeline"]) == 10 and radio["events_sha256"]
    trash = next(r for r in recs if r["task"] != TASK)
    assert trash["infra_error"] and trash["timeline"] == [] and "events.jsonl not found" in trash["evidence_notes"]
    with pytest.raises(SystemExit):  # an existing file may hold labels
        fl.main(["skeleton", run, "--out", str(out)])


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
    md = fl.render_summary(s)
    assert "抓放操作" in md and "1 unlabelled" in md and "SYNTHETIC" in md
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
