"""data/schema.py: round trips, validation errors, field categories, source-isolated split.

All records here are SYNTHETIC (built in the test); they check the schema, not any data.
"""

import dataclasses
import json
import random

import pytest

from agent.interfaces import Experience
from data import schema as S


def ref(modality, key="", camera=""):
    return S.DataRef(path="demo/episode_000042.parquet", key=key or modality, modality=modality, camera=camera)


def segment(**kw):
    base = dict(
        segment_id="seg-1", source_trajectory_id="ep42", dataset="b1k_demos", data_version="2025.1",
        task_name="turning_on_radio", skill_instruction="pick up the radio",
        task_instruction="Turn on the radio receiver that's on the table in the living room.",
        start_index=120, end_index=480,
        observation_refs=(ref("rgb", "observation.images.rgb.head", "head"), ref("proprio")),
        action_ref=ref("action"), time_sync=S.TimeSync(fps=30.0, index_unit="frame"),
        completion_label=True, label_source="official_annotation",
        episode_outcome=S.EpisodeOutcome(outcome="success", q_final=1.0, source="demo_dataset"),
        split="train", round=0, skill="pick_up", objects=("radio_receiver.n.01",), synthetic=True)
    base.update(kw)
    return S.SkillSegment(**base)


def experience(**kw):
    base = dict(
        exp_id="exp-1",
        provenance=S.ExperienceProvenance(kind="dev_run", run_id="L2-run", source_trajectory_id="L2-run-ep001-env0",
                                          task_name="turning_on_radio", instance_id=311, start_index=0, end_index=300),
        skill="pick_up", object_categories=("radio_receiver.n.01",), summary="approach from the left, grasp the handle",
        outcome="success",
        obs_refs=(S.DataRef(path="episodes/x/frames/000300_periodic.jpg", modality="rgb", camera="head"),),
        round=1, frozen=True, payload={"approach_side": "left"}, synthetic=True)
    base.update(kw)
    return S.ExperienceRecord(**base)


def distill_record(**kw):
    base = dict(
        record_id="ep#L7", call_id="ep-c00003", run_id="run", episode_id="ep", step=150, slot="check",
        purpose="check.judge", provider="astra", model="m",
        inputs=S.DistillInputs(user_text="TASK ...", image_refs=("episodes/ep/frames/ep-c00003_head.jpg",),
                               subgoal_literals={"g0": "(toggled_on ?radio_receiver.n.01_1)"},
                               subtask_in_force={"skill": "__full_task__", "target_subgoals": ["g0"]}),
        outputs=S.DistillOutputs(decision_type="continue_switch", status="done", confidence=0.9,
                                 continue_switch="switch", subgoal_estimates={"g0": "satisfied"},
                                 parsed={"status": "done"}),
        visible_evidence="display lit",
        source=S.DistillSource(events_path="episodes/ep/events.jsonl", events_sha256="0" * 64, line=7),
        synthetic=True)
    base.update(kw)
    return S.DistillRecord(**base)


# ------------------------------------------------------------------------------------ round trips


@pytest.mark.parametrize("make,cls", [(segment, S.SkillSegment), (experience, S.ExperienceRecord),
                                      (distill_record, S.DistillRecord)])
def test_round_trip(make, cls, tmp_path):
    rec = make()
    assert rec.errors() == []
    text = S.dumps(rec)
    assert S.loads(cls, text) == rec
    path = tmp_path / "x.jsonl"
    assert S.write_jsonl(str(path), [rec, rec]) == 2
    assert S.read_jsonl(str(path), cls) == [rec, rec]


def test_round_trip_keeps_null_and_nested_types():
    rec = segment(completion_label=None, label_source=None, source_instance_id=12,
                  episode_outcome=S.EpisodeOutcome(outcome="failure", q_final=0.0))
    back = S.loads(S.SkillSegment, S.dumps(rec))
    assert back.completion_label is None and isinstance(back.observation_refs, tuple)
    assert isinstance(back.observation_refs[0], S.DataRef) and back == rec


# ------------------------------------------------------------------------------------ validation


def test_missing_unknown_and_wrong_type_fields_are_all_reported():
    d = json.loads(S.dumps(segment()))
    del d["dataset"]
    d["bogus"] = 1
    d["start_index"] = "120"
    d["time_sync"]["fps"] = "fast"
    with pytest.raises(S.SchemaError) as e:
        S.from_dict(S.SkillSegment, d)
    msg = str(e.value)
    for part in ("dataset: missing required field", "bogus: unknown field", "start_index: expected an int",
                 "time_sync.fps: expected a number"):
        assert part in msg


@pytest.mark.parametrize("change,needle", [
    (dict(start_index=500, end_index=480), "0 <= start < end"),
    (dict(label_source="guess"), "label_source 'guess'"),
    (dict(label_source=None), "required with a completion label"),
    (dict(split="test"), "split 'test'"),
    (dict(round=-1), "round must be >= 0"),
    (dict(source_instance_id=305), "Report instance"),
    (dict(source_instance_id=330), "hidden_test instance"),
    (dict(observation_refs=(ref("object_poses"),)), "modality 'object_poses'"),
    (dict(observation_refs=()), "observation_refs is empty"),
    (dict(action_ref=ref("rgb")), "action_ref.modality 'rgb'"),
    (dict(time_sync=S.TimeSync(fps=0, index_unit="second")), "fps must be > 0"),
    (dict(episode_outcome=S.EpisodeOutcome(outcome="success", q_final=0.4)), "contradicts q_final"),
])
def test_segment_validation_errors(change, needle):
    rec = segment(**change)
    assert any(needle in e for e in rec.errors()), rec.errors()
    with pytest.raises(S.SchemaError):
        S.check(rec)
    with pytest.raises(S.SchemaError):
        S.loads(S.SkillSegment, S.dumps(rec))


def test_a_successful_local_segment_of_a_failed_episode_is_kept():
    rec = segment(completion_label=True, episode_outcome=S.EpisodeOutcome(outcome="partial", q_final=0.5))
    assert rec.errors() == []


@pytest.mark.parametrize("change,needle", [
    (dict(outcome="won"), "outcome 'won'"),
    (dict(payload={"sim.object_poses": [1, 2]}), "truth / evaluation field"),
    (dict(payload={"notes": {"q_score": 0.5}}), "payload.notes.q_score"),
    (dict(obs_refs=(S.DataRef(path="x", modality="action"),)), "modality 'action'"),
    (dict(provenance=S.ExperienceProvenance(kind="dev_run", run_id="r", source_trajectory_id="e", instance_id=301)),
     "Report instance"),
    (dict(provenance=S.ExperienceProvenance(kind="demo", source_trajectory_id="e")), "needs dataset and data_version"),
])
def test_experience_validation_errors(change, needle):
    assert any(needle in e for e in experience(**change).errors())


def test_experience_to_runtime_experience():
    exp = experience().to_experience()
    assert isinstance(exp, Experience)
    assert exp.provenance == "dev_run:L2-run/L2-run-ep001-env0[0:300]" and exp.objects == ("radio_receiver.n.01",)


@pytest.mark.parametrize("change,needle", [
    (dict(truth_label="done"), "without truth_source"),
    (dict(correct=True, truth_label="unknown", truth_source="replay"), "truth label is unknown"),
    (dict(truth_label="maybe", truth_source="replay"), "truth_label 'maybe'"),
    (dict(outputs=S.DistillOutputs(decision_type="vibes")), "decision_type 'vibes'"),
    (dict(outputs=S.DistillOutputs(decision_type="continue_switch", continue_switch="stop")), "continue_switch 'stop'"),
    (dict(outputs=S.DistillOutputs(decision_type="continue_switch", confidence=1.5)), "confidence 1.5"),
])
def test_distill_validation_errors(change, needle):
    assert any(needle in e for e in distill_record(**change).errors())


def test_exported_distill_record_has_no_truth():
    rec = distill_record()
    assert (rec.truth_label, rec.truth_source, rec.correct, rec.truth_detail) == (None, None, None, None)
    assert not rec.labelled


# ------------------------------------------------------------------------------------ field categories


@pytest.mark.parametrize("cls", [S.SkillSegment, S.ExperienceRecord, S.DistillRecord])
def test_every_field_has_one_category(cls):
    cats = S.field_categories(cls)
    assert set(cats.values()) <= set(S.CATEGORIES)
    assert all(v is not None for v in cats.values())
    top = {f.name for f in dataclasses.fields(cls)}
    assert top == set(cls.FIELD_CATEGORIES)
    assert {k.split(".")[0] for k in cats} == top  # every top-level field reaches a leaf


def test_category_choices_that_matter():
    seg = S.field_categories(S.SkillSegment)
    assert seg["observation_refs.path"] == S.POLICY_OBSERVATION
    assert seg["completion_label"] == S.TRAINING_ANNOTATION
    assert seg["episode_outcome.q_final"] == S.EVALUATION_RECORD
    dr = S.field_categories(S.DistillRecord)
    assert dr["inputs.user_text"] == S.POLICY_OBSERVATION
    assert dr["truth_label"] == dr["correct"] == S.TRAINING_ANNOTATION
    assert all(v == S.POLICY_OBSERVATION for k, v in dr.items() if k.startswith("inputs."))


def test_undeclared_field_is_rejected():
    @dataclasses.dataclass(frozen=True)
    class Bad:
        a: int
        FIELD_CATEGORIES = {}

    with pytest.raises(S.SchemaError, match="misses"):
        S.field_categories(Bad)


# ------------------------------------------------------------------------------------ split


def many_segments(n_traj, per_traj, datasets=("b1k_demos", "dev_runs")):
    out = []
    for d in datasets:
        for t in range(n_traj):
            for k in range(per_traj):
                out.append(segment(segment_id=f"{d}-{t}-{k}", dataset=d, source_trajectory_id=f"traj{t}",
                                   start_index=100 * k, end_index=100 * k + 90))
    return out


@pytest.mark.parametrize("seed", [0, 1, 7])
@pytest.mark.parametrize("frac", [0.0, 0.2, 0.5, 1.0])
def test_split_by_source_never_splits_a_trajectory(seed, frac):
    segs = many_segments(60, 4)
    random.Random(seed).shuffle(segs)
    out = S.split_by_source(segs, frac, seed=seed)
    S.validate_split(out)
    splits = {}
    for s in out:
        splits.setdefault(s.source_key, set()).add(s.split)
    assert all(len(v) == 1 for v in splits.values())
    dev = sum(1 for v in splits.values() if v == {"dev"}) / len(splits)
    if frac in (0.0, 1.0):
        assert dev == frac
    else:
        assert abs(dev - frac) < 0.15
    assert [s.segment_id for s in out] == [s.segment_id for s in segs]  # order and content otherwise unchanged


def test_split_is_stable_when_trajectories_are_added():
    first = {s.source_key: s.split for s in S.split_by_source(many_segments(30, 2), 0.3, seed=3)}
    later = {s.source_key: s.split for s in S.split_by_source(many_segments(80, 2), 0.3, seed=3)}
    assert all(later[k] == v for k, v in first.items())


def test_validate_split_catches_leaks_duplicates_and_report_instances():
    segs = S.split_by_source(many_segments(5, 3, datasets=("d",)), 0.5, seed=0)
    leaked = list(segs)
    leaked[0] = dataclasses.replace(leaked[0], split="dev" if leaked[0].split == "train" else "train")
    with pytest.raises(S.SchemaError, match="spans splits"):
        S.validate_split(leaked)
    with pytest.raises(S.SchemaError, match="appears 2 times"):
        S.validate_split(segs + [segs[0]])
    with pytest.raises(S.SchemaError, match="Report instance"):
        S.validate_split([dataclasses.replace(segs[0], source_instance_id=301)])


def test_split_rejects_bad_fraction():
    with pytest.raises(ValueError):
        S.split_by_source([segment()], 1.5)
