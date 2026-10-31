"""Regression tests for the adversarial review of data/ and diagnostics/. All inputs are SYNTHETIC or mock
(fake evaluator, fake providers); nothing here says how the code behaves on 105 or in OmniGibson."""

import ast
import contextvars
import dataclasses
import io
import json
import os

import numpy as np
import pytest

from conftest import ROOT
from data import distill as D
from data import rounds as R
from data import schema as S
from diagnostics import join as J
from diagnostics import label_distill as L
from diagnostics import replay_truth as RT
from diagnostics import truth as T
from test_data_diag_replay import FakeBatchedEvaluator, episode_dir, fake_cfg, official_json

TASK = "turning_on_radio"


def dump(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f)


# ------------------------------------------------------------------------------------ join


def check(step, status):
    return {"kind": "decision", "slot": "check", "step": step, "seconds": 1.0,
            "out": {"status": status, "confidence": 0.9, "source": "check.llm@astra"}}


def batched_run(tmp_path):
    """Two envs: env0 timed out at step 400 but the batch ran on (frozen obs) and its checker said done at 2000;
    env1 ran out of the 2026 budget (failure_reason: success / q forced) after a done verdict at 800."""
    run = str(tmp_path / "run1")
    dump(os.path.join(run, "status.json"), {"_synthetic": True, "run_id": "run1", "task": TASK, "mode": "public_test",
                                            "instance_ids": [311, 312], "state": "done"})
    official = {311: {"steps": 401, "q_score": {"final": 0.0}},
                312: {"steps": 900, "q_score": {"final": 0.0}, "failure_reason": "group_time_budget_exceeded"}}
    for iid, rec in official.items():
        dump(os.path.join(run, "eval_output", "json", f"{TASK}_{iid}_0.json"),
             {"_synthetic": True, "task": TASK, "instance_id": iid, "rollout_id": 0, "success": False, **rec})
    events = {0: [check(150, "in_progress"), check(2000, "done"), {"kind": "decision", "slot": "plan", "step": 2100,
                                                                    "out": None}],
              1: [check(800, "done")]}
    for env, evs in events.items():
        eid = f"run1-ep001-env{env}"
        d = os.path.join(run, "episodes", eid)
        dump(os.path.join(d, "summary.json"), {"_synthetic": True, "episode_id": eid, "run_id": "run1",
                                               "task_name": TASK, "tags": {"instance_ids": "311,312"}, "steps": 3000,
                                               "fallbacks": 0})
        with open(os.path.join(d, "events.jsonl"), "w") as f:
            for ev in evs:
                f.write(json.dumps(ev) + "\n")
    return run


def test_join_ignores_decisions_after_the_official_end_and_forced_failure_outcomes(tmp_path):
    records, _ = J.join_runs([batched_run(tmp_path)])
    by_iid = {r["instance_id"]: r for r in records}
    timed_out = by_iid[311]
    assert timed_out["last_decisive_verdict"] is None and timed_out["n_verdicts"] == 1  # step 2000 >= 401
    assert timed_out["planner_complete_claims"] == [] and timed_out["decisions_after_official_end"] == 2
    assert timed_out["checker_consistent"] is None
    budget = by_iid[312]
    assert budget["last_decisive_verdict"]["step"] == 800
    assert budget["checker_consistent"] is None  # success was forced to false by the budget failure
    assert "| turning_on_radio | 2 | 2 | - | - | 0 |" in J.render_markdown(records)


# ------------------------------------------------------------------------------------ replay_truth


def replay_plan(tmp_path, *extra, official=None):
    args = RT.parse_args(["--episode-dir", episode_dir(tmp_path),
                          "--official-json", official or official_json(tmp_path),
                          "--out", str(tmp_path / "truth" / "ep.truth.jsonl"), *extra])
    p = RT.plan(args)
    p["every"] = args.every
    return p


def test_footer_for_a_2026_failure_record_compares_steps_only(tmp_path):
    p = {"official": {"steps": 100, "success": False, "q_final": 0.0, "failure_reason": "group_time_budget_exceeded"}}
    pol = RT.ReplayPolicy(np.zeros((2, 23), np.float32))
    same = RT.footer_fields(p, {"steps": 100, "success": True, "q_score": {"final": 1.0}}, pol)
    assert same["consistent"] is None and same["official_failure_reason"] == "group_time_budget_exceeded"
    assert RT.footer_fields(p, {"steps": 60, "success": True, "q_score": {"final": 1.0}}, pol)["consistent"] is False
    plain = {"official": {"steps": 100, "success": False, "q_final": 0.0}}
    reproduced = RT.footer_fields(plain, {"steps": 100, "success": False, "q_score": {"final": 0.0}}, pol)
    assert reproduced["consistent"] is True


def test_replay_never_writes_over_the_official_result(tmp_path, capsys):
    (tmp_path / "eval_output" / "json").mkdir(parents=True)
    off = official_json(tmp_path / "eval_output" / "json")
    with pytest.raises(RT.ReplayInputError, match="would overwrite the official result"):
        replay_plan(tmp_path, "--replay-output-dir", str(tmp_path / "eval_output"), official=off)
    d = episode_dir(tmp_path / "b")
    out, replay_dir = tmp_path / "t.jsonl", tmp_path / "replay"
    dump(str(replay_dir / "json" / f"{TASK}_312_0.json"), {"_synthetic": True})
    assert RT.main(["--episode-dir", d, "--out", str(out), "--replay-output-dir", str(replay_dir)]) == 1
    assert "pass --force" in capsys.readouterr().err


def test_wrapper_defaults_to_the_episode_launch_tag(tmp_path):
    d = episode_dir(tmp_path)
    summary = json.load(open(os.path.join(d, "summary.json")))
    summary["tags"]["wrapper"] = "rgbd"
    json.dump(summary, open(os.path.join(d, "summary.json"), "w"))
    base = ["--episode-dir", d, "--out", str(tmp_path / "t.jsonl")]
    assert RT.plan(RT.parse_args(base))["env_wrapper"] == "omnigibson.eval.wrappers.RGBDFullResWrapper"
    explicit = RT.plan(RT.parse_args(base + ["--env-wrapper", "omnigibson.eval.wrappers.DefaultWrapper"]))
    assert explicit["env_wrapper"] == "omnigibson.eval.wrappers.DefaultWrapper"
    del summary["tags"]["wrapper"]
    json.dump(summary, open(os.path.join(d, "summary.json"), "w"))
    assert RT.plan(RT.parse_args(base))["env_wrapper"] == RT.DEFAULT_WRAPPER


class CrashingEvaluator(FakeBatchedEvaluator):
    def _step_fn(self, active_env_indices):
        if self.episode_steps == 50:
            raise RuntimeError("simulated simulator crash")
        return super()._step_fn(active_env_indices)


class ShutdownCheckingEvaluator(FakeBatchedEvaluator):
    seen_at_exit = None

    def __exit__(self, *exc):
        out = ShutdownCheckingEvaluator.out
        ShutdownCheckingEvaluator.seen_at_exit = (os.path.exists(out), os.path.exists(out + ".partial"))
        return super().__exit__(*exc)


def test_truth_file_appears_only_complete_and_before_shutdown(tmp_path):
    p = replay_plan(tmp_path)
    with pytest.raises(RuntimeError, match="simulated"):
        RT.run_replay(p, CrashingEvaluator, fake_cfg)
    assert not os.path.exists(p["out"]) and os.path.exists(p["out"] + ".partial")

    reported = []
    ShutdownCheckingEvaluator.out = p["out"]
    footer = RT.run_replay(p, ShutdownCheckingEvaluator, fake_cfg, report=reported.append)
    assert ShutdownCheckingEvaluator.seen_at_exit == (True, False)  # renamed before og.shutdown could end the process
    assert reported == [footer] and T.load_truth([p["out"]])["runX-ep001-env1"].footer is not None


# ------------------------------------------------------------------------------------ label_distill


def test_label_skips_a_truth_track_without_footer(tmp_path):
    buf = io.StringIO()
    w = T.TruthWriter(buf, "ep", [["(toggled_on radio_receiver.n.01_1)"]], {"_synthetic": True})
    w.start([[False]], success=False)
    w.record(150, [[True]], success=False)  # no finish(): the replay was interrupted
    path = tmp_path / "t.jsonl"
    path.write_text(buf.getvalue())
    track = T.load_truth([str(path)])["ep"]
    rec = S.DistillRecord(
        record_id="ep#L1", call_id="c", run_id="run", episode_id="ep", step=150, slot="check", purpose="check.judge",
        provider="astra", model="m",
        inputs=S.DistillInputs(user_text="u", subgoal_literals={"g0": "(toggled_on ?radio_receiver.n.01_1)"},
                               subtask_in_force={"skill": "x", "target_subgoals": ["g0"]}),
        outputs=S.DistillOutputs(decision_type="continue_switch", status="done", parsed={"status": "done"}),
        visible_evidence="", source=S.DistillSource(events_path="e", events_sha256="0", line=1), synthetic=True)
    out = L.label_record(rec, track)
    assert out.truth_label is None and out.correct is None and "no footer" in out.truth_detail["skipped"]
    assert out.errors() == []


# ------------------------------------------------------------------------------------ schema


def test_agent_run_records_must_carry_the_instance_id():
    prov = S.ExperienceProvenance(kind="dev_run", run_id="r", source_trajectory_id="r-ep001-env0")
    assert any("needs instance_id" in e for e in prov.errors("provenance"))
    assert dataclasses.replace(prov, instance_id=312).errors("provenance") == []
    ref = S.DataRef(path="episodes/x/steps.jsonl", key="a", modality="action")
    seg = S.SkillSegment(
        segment_id="s", source_trajectory_id="r-ep001-env0", dataset="b1k_agent_runs", data_version="run1",
        task_name=TASK, skill_instruction="pick up the radio", task_instruction="Turn on the radio.", start_index=0,
        end_index=300, observation_refs=(S.DataRef(path="episodes/x/frames", modality="rgb", camera="head"),),
        action_ref=ref, time_sync=S.TimeSync(fps=30.0, index_unit="step"), completion_label=None, label_source=None,
        episode_outcome=S.EpisodeOutcome(outcome="failure", q_final=0.0), split="train", round=1, synthetic=True)
    assert any("source_instance_id is required" in e for e in seg.errors())
    assert dataclasses.replace(seg, source_instance_id=312).errors() == []
    assert any("Report instance" in e for e in dataclasses.replace(seg, source_instance_id=303).errors())


# ------------------------------------------------------------------------------------ distill


def test_distill_keeps_both_image_refs_with_two_providers(tmp_path):
    """Real RunRecorder + LoggedProvider output, plan on one provider and check on another. call ids used
    to collide per provider ('<ep>-c00001' twice, the second image overwriting the first); providers/base.py
    now numbers calls process-wide, so both records keep their own image. (export_run still drops refs
    from older logs where a later call overwrote the file.)"""
    from agent import context as ctxvars
    from agent.interfaces import EpisodeContext
    from agent.recorder import RunRecorder
    from providers.base import LoggedProvider, ProviderRequest, ProviderResponse

    class Inner:
        def __init__(self, name):
            self.name = self.model = name

        def complete(self, req):
            return ProviderResponse(text="{}", parsed={"status": "in_progress"}, usage={}, latency_s=0.0,
                                    model=self.name, provider=self.name)

    eid = "run2-ep001-env0"
    recorder = RunRecorder("run2", str(tmp_path), save_frames=False)
    recorder.episode_start(EpisodeContext(episode_id=eid, env_index=0, task_name=TASK, task_id=0, task_instruction="t",
                                          goal_spec=None, skills=(), max_steps=None, wallclock_budget_s=None), {})
    plan, chk = LoggedProvider(Inner("astra"), recorder=recorder), LoggedProvider(Inner("claude"), recorder=recorder)
    img = np.zeros((8, 8, 3), np.uint8)

    def calls():  # in a copied context, so the context variables do not leak into other tests
        ctxvars.current_episode.set(eid)
        for slot, provider in (("plan", plan), ("check", chk)):
            ctxvars.current_slot.set(slot)
            provider.complete(ProviderRequest(system="s", user_text=slot, images=[("head", img)]))

    contextvars.copy_context().run(calls)
    recorder._files[eid]["events"].flush()
    recs, warnings = D.export_run(str(tmp_path / "run2"))
    by_slot = {r.slot: r for r in recs}
    assert by_slot["plan"].call_id != by_slot["check"].call_id
    assert len(by_slot["plan"].inputs.image_refs) == 1 and len(by_slot["check"].inputs.image_refs) == 1
    assert by_slot["plan"].inputs.image_refs != by_slot["check"].inputs.image_refs
    assert not any("overwritten by a later call" in w for w in warnings)


# ------------------------------------------------------------------------------------ rounds / guards


def test_round_fingerprint_covers_the_task_prompt_snapshots():
    _, files = R.config_fingerprint(os.path.join(ROOT, "configs", "L0_exec_only.yaml"), [], ROOT)
    assert {"configs/tasks/tasks_2026.jsonl", "configs/tasks/task_catalog_2026.tsv"} <= set(files)
    assert "missing" not in files.values()


def test_data_has_no_dynamic_imports_of_forbidden_packages():
    forbidden = {"omnigibson", "isaacsim", "omni", "pxr", "bddl", "carb", "diagnostics", "eval"}
    bad = []
    for name in sorted(os.listdir(os.path.join(ROOT, "data"))):
        if not name.endswith(".py"):
            continue
        tree = ast.parse(open(os.path.join(ROOT, "data", name), encoding="utf-8").read())
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call) and getattr(node.func, "attr", getattr(node.func, "id", "")) in
                    ("import_module", "__import__") and node.args and isinstance(node.args[0], ast.Constant)
                    and str(node.args[0].value).split(".")[0] in forbidden):
                bad.append(f"{name}:{node.lineno}")
    assert not bad, bad


def test_distill_never_exports_report_or_hidden_episodes(tmp_path):
    def episode(eid, tags):
        d = tmp_path / "run3" / "episodes" / eid
        d.mkdir(parents=True)
        events = [{"_synthetic": True, "kind": "episode_start", "ctx": {"episode_id": eid, "task_name": TASK},
                   "tags": tags},
                  {"kind": "provider_call", "call_id": f"{eid}-c00001", "slot": "check", "step": 150,
                   "purpose": "check.judge", "parsed": {"status": "in_progress"}, "images": []}]
        (d / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))

    launch = {"task": TASK, "mode": "public_test", "instance_ids": "305,311"}
    episode("run3-ep001-env0", launch)  # public_test index 4 -> 305: a Report instance
    episode("run3-ep001-env1", launch)  # index 10 -> 311: Dev
    episode("run3-ep002-env0", {"mode": "public_test", "instance": "4"})  # an index, not a resolved id
    recs, warnings = D.export_run(str(tmp_path / "run3"))
    assert sorted((r.episode_id, r.instance_id) for r in recs) == [("run3-ep001-env1", 311), ("run3-ep002-env0", None)]
    assert any("instance 305 (Report / hidden_test) skipped" in w for w in warnings)
    assert any("instance id unknown" in w for w in warnings)
    assert D.instance_from_tags({"mode": "hidden_test", "instance_ids": "321"}, "r-ep001-env0") == 321
    assert D.instance_from_tags({"mode": "train", "instance_ids": "7,8"}, "r-ep001-env1") == 8
    leaked = dataclasses.replace(recs[0], instance_id=330)
    assert any("hidden_test instance" in e for e in leaked.errors())
    with pytest.raises(S.SchemaError, match="Report instance"):
        S.loads(S.DistillRecord, json.dumps({**json.loads(S.dumps(recs[0])), "instance_id": 301}))
