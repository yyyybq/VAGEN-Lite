"""Record types for training data, memory and distillation (task spec 7.1-7.3), with JSON
(de)serialization, validation and the field-category map.

Three record types:

  SkillSegment      one skill-level slice of a source trajectory (a demo episode or one of our own
                    Dev / train-instance episodes): source and data version, task / skill instruction,
                    raw start / end indices, references to the synchronised observations and actions,
                    completion label + its source, the whole episode's outcome, split and round (7.1).
  ExperienceRecord  one M-memory library entry; ``to_experience()`` gives the runtime
                    agent.interfaces.Experience that M-memory hands to the planner / checker.
  DistillRecord     one strong-model call with its legal inputs, its outputs (subtask choice,
                    continue / switch, failure verdict, recovery choice) and visible evidence (7.3),
                    with the episode's resolved instance id. The verification fields (truth_label,
                    truth_source, correct) stay empty until diagnostics/label_distill.py fills them offline.

Field categories (7.2, agent/fields.py): every leaf field of every record type has exactly one of
POLICY_OBSERVATION (legal at run time), TRAINING_ANNOTATION (labels / metadata, may come from
simulator truth) or EVALUATION_RECORD (official results, read after episodes only); see
``field_categories``. A record type declares its top-level fields in ``FIELD_CATEGORIES``; nested
fields inherit unless a dotted override is given.

Split isolation (7.1): ``split_by_source`` assigns whole source trajectories to train or dev, so
neighbouring segments of one trajectory never land on both sides; ``validate_split`` checks it.
Report (public_test 301-310) and hidden (321-340) instances never enter training data
(docs/PROTOCOL.md; OmniGibson/omnigibson/eval/utils/eval_utils.py TEST_INSTANCE_IDS = range(301, 341),
NUM_PUBLIC_TEST_INSTANCES = 20). The check is by instance id, so records from our own runs (dev_run
experiences, segments indexed by agent step) must carry the resolved instance id.

Status: written and unit-tested (tests/test_data_diag_schema.py); no real data converted yet.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import types
import typing
from typing import Any, ClassVar, Iterable, Optional, Sequence

from agent.fields import EVALUATION_RECORD, POLICY_OBSERVATION, TRAINING_ANNOTATION
from agent.fields import FIELD_CATEGORIES as RUNTIME_FIELD_CATEGORIES
from agent.interfaces import Experience

SCHEMA_VERSION = 1
CATEGORIES = (POLICY_OBSERVATION, TRAINING_ANNOTATION, EVALUATION_RECORD)

LABEL_SOURCES = ("official_annotation", "truth_rule", "strong_model", "human")
SPLITS = ("train", "dev")
OUTCOMES = ("success", "failure", "partial")
PROVENANCE_KINDS = ("demo", "dev_run")
INDEX_UNITS = ("frame", "step")  # demo dataset frame index | agent control step (steps.jsonl "step")
DECISION_TYPES = ("subtask_choice", "continue_switch", "failure_verdict", "recovery_choice", "other")
CONTINUE_SWITCH = ("continue", "switch", "recover", "unknown")
TRUTH_LABELS = ("done", "not_done", "unknown")

REPORT_INSTANCE_IDS = frozenset(range(301, 311))  # public_test indices 0-9
HIDDEN_INSTANCE_IDS = frozenset(range(321, 341))  # hidden_test indices 0-19

# Legal observation modalities = the obs.* policy-observation fields of agent/fields.py.
OBS_MODALITIES = tuple(
    k.split(".", 1)[1] for k, v in RUNTIME_FIELD_CATEGORIES.items() if k.startswith("obs.") and v == POLICY_OBSERVATION
)
ACTION_MODALITY = "action"
# Names of truth / evaluation fields (agent/fields.py sim.* and eval.*) that must never appear in a
# runtime-visible payload.
FORBIDDEN_RUNTIME_KEYS = frozenset(
    k.split(".", 1)[1] for k, v in RUNTIME_FIELD_CATEGORIES.items() if v in (TRAINING_ANNOTATION, EVALUATION_RECORD)
)


class SchemaError(ValueError):
    """A record does not match its schema; the message lists every problem found."""


# ---------------------------------------------------------------------------------------------
# Shared parts
# ---------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class DataRef:
    """Reference to stored data: a file (or dataset) path plus the key / column inside it."""

    path: str  # absolute, or relative to the dataset / run root named by the record
    key: str = ""  # column or key inside the file, e.g. "observation.images.rgb.head" or "a" (steps.jsonl)
    modality: str = ""  # one of OBS_MODALITIES, or "action"
    camera: str = ""  # camera role for images: head | left_wrist | right_wrist
    index: Optional[int] = None  # a single frame / step; None = the enclosing record's index range

    def errors(self, where: str, allowed: Sequence[str]) -> list[str]:
        out = []
        if not self.path:
            out.append(f"{where}.path is empty")
        if self.modality not in allowed:
            out.append(f"{where}.modality {self.modality!r} not in {list(allowed)}")
        if self.index is not None and self.index < 0:
            out.append(f"{where}.index must be >= 0")
        return out


@dataclasses.dataclass(frozen=True)
class TimeSync:
    """How observation and action indices line up in the source."""

    fps: float  # 30 for BEHAVIOR demos and for the official control loop
    index_unit: str  # "frame" | "step"
    action_offset: int = 0  # the action at index i was executed after the observation at i + action_offset
    notes: str = ""

    def errors(self, where: str) -> list[str]:
        out = []
        if not self.fps or self.fps <= 0:
            out.append(f"{where}.fps must be > 0")
        if self.index_unit not in INDEX_UNITS:
            out.append(f"{where}.index_unit {self.index_unit!r} not in {list(INDEX_UNITS)}")
        return out


@dataclasses.dataclass(frozen=True)
class EpisodeOutcome:
    """Outcome of the whole source episode: success / failure / partial, and the official Q when known."""

    outcome: str
    q_final: Optional[float] = None  # official q_score.final (OmniGibson metrics/task_metric.py)
    source: str = ""  # e.g. "official_json:<path>" or "demo_dataset"

    def errors(self, where: str) -> list[str]:
        out = []
        if self.outcome not in OUTCOMES:
            out.append(f"{where}.outcome {self.outcome!r} not in {list(OUTCOMES)}")
        if self.q_final is not None:
            q = self.q_final
            if not 0.0 <= q <= 1.0:
                out.append(f"{where}.q_final {q} outside [0, 1]")
            expected = "success" if q >= 1.0 else ("failure" if q <= 0.0 else "partial")
            if self.outcome in OUTCOMES and self.outcome != expected:
                out.append(f"{where}.outcome {self.outcome!r} contradicts q_final {q} (expected {expected!r})")
        return out


def _instance_errors(where: str, instance_id: Optional[int]) -> list[str]:
    if instance_id in REPORT_INSTANCE_IDS:
        return [f"{where} {instance_id} is a Report instance (public_test 0-9); "
                "it never enters training data or memory"]
    if instance_id in HIDDEN_INSTANCE_IDS:
        return [f"{where} {instance_id} is a hidden_test instance; it never enters training data or memory"]
    return []


# ---------------------------------------------------------------------------------------------
# 7.1 skill segments
# ---------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class SkillSegment:
    """Indices are raw indices in the source episode, half-open: [start_index, end_index)."""

    segment_id: str
    source_trajectory_id: str
    dataset: str
    data_version: str
    task_name: str
    skill_instruction: str
    task_instruction: str
    start_index: int
    end_index: int
    observation_refs: tuple[DataRef, ...]
    action_ref: DataRef
    time_sync: TimeSync
    completion_label: Optional[bool]  # None = not labelled yet
    label_source: Optional[str]  # one of LABEL_SOURCES; required once completion_label is set
    episode_outcome: EpisodeOutcome
    split: str
    round: int
    skill: str = ""  # skill name in the executor catalog (configs/skills/*.yaml)
    objects: tuple[str, ...] = ()  # BDDL object instances / categories the segment acts on
    source_instance_id: Optional[int] = None  # task instance of the source episode, when known
    provenance_notes: tuple[str, ...] = ()  # how boundaries / labels were made, spot checks, fixes
    schema_version: int = SCHEMA_VERSION
    synthetic: bool = False  # True for test fixtures; synthetic data never counts as a result

    FIELD_CATEGORIES: ClassVar[dict[str, str]] = {
        "segment_id": TRAINING_ANNOTATION,
        "source_trajectory_id": TRAINING_ANNOTATION,
        "dataset": TRAINING_ANNOTATION,
        "data_version": TRAINING_ANNOTATION,
        "task_name": POLICY_OBSERVATION,
        "skill_instruction": POLICY_OBSERVATION,
        "task_instruction": POLICY_OBSERVATION,
        "start_index": TRAINING_ANNOTATION,
        "end_index": TRAINING_ANNOTATION,
        "observation_refs": POLICY_OBSERVATION,
        "action_ref": TRAINING_ANNOTATION,  # supervision target
        "time_sync": TRAINING_ANNOTATION,
        "completion_label": TRAINING_ANNOTATION,
        "label_source": TRAINING_ANNOTATION,
        "episode_outcome": EVALUATION_RECORD,
        "split": TRAINING_ANNOTATION,
        "round": TRAINING_ANNOTATION,
        "skill": POLICY_OBSERVATION,
        "objects": POLICY_OBSERVATION,  # from the BDDL task definition
        "source_instance_id": TRAINING_ANNOTATION,
        "provenance_notes": TRAINING_ANNOTATION,
        "schema_version": TRAINING_ANNOTATION,
        "synthetic": TRAINING_ANNOTATION,
    }

    @property
    def source_key(self) -> tuple[str, str]:
        """Isolation unit for splitting: one source trajectory of one dataset (any data version)."""
        return (self.dataset, self.source_trajectory_id)

    def errors(self) -> list[str]:
        out = []
        for name in ("segment_id", "source_trajectory_id", "dataset", "data_version", "task_name", "skill_instruction",
                     "task_instruction"):
            if not getattr(self, name):
                out.append(f"{name} is empty")
        if self.start_index < 0 or self.end_index <= self.start_index:
            out.append(f"indices [{self.start_index}, {self.end_index}) must satisfy 0 <= start < end")
        if not self.observation_refs:
            out.append("observation_refs is empty")
        for i, ref in enumerate(self.observation_refs):
            out += ref.errors(f"observation_refs[{i}]", OBS_MODALITIES)
        out += self.action_ref.errors("action_ref", (ACTION_MODALITY,))
        out += self.time_sync.errors("time_sync")
        if self.time_sync.index_unit == "step" and self.source_instance_id is None:
            out.append("source_instance_id is required for a segment of an agent run (index_unit 'step'): "
                       "Report / hidden instances are excluded by id")
        if self.completion_label is not None and self.label_source not in LABEL_SOURCES:
            out.append(f"label_source {self.label_source!r} not in {list(LABEL_SOURCES)} "
                       "(required with a completion label)")
        if self.completion_label is None and self.label_source is not None and self.label_source not in LABEL_SOURCES:
            out.append(f"label_source {self.label_source!r} not in {list(LABEL_SOURCES)}")
        out += self.episode_outcome.errors("episode_outcome")
        if self.split not in SPLITS:
            out.append(f"split {self.split!r} not in {list(SPLITS)}")
        if self.round < 0:
            out.append("round must be >= 0")
        out += _instance_errors("source_instance_id", self.source_instance_id)
        return out


# ---------------------------------------------------------------------------------------------
# M-memory experiences
# ---------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class ExperienceProvenance:
    kind: str  # "demo" | "dev_run"
    dataset: str = ""
    data_version: str = ""
    source_trajectory_id: str = ""  # demo episode id, or the agent's episode_id
    run_id: str = ""  # dev_run only
    task_name: str = ""
    instance_id: Optional[int] = None
    start_index: Optional[int] = None
    end_index: Optional[int] = None

    def text(self) -> str:
        span = f"[{self.start_index}:{self.end_index}]" if self.start_index is not None else ""
        if self.kind == "dev_run":
            return f"dev_run:{self.run_id}/{self.source_trajectory_id}{span}"
        return f"{self.dataset}@{self.data_version}/{self.source_trajectory_id}{span}"

    def errors(self, where: str) -> list[str]:
        out = []
        if self.kind not in PROVENANCE_KINDS:
            out.append(f"{where}.kind {self.kind!r} not in {list(PROVENANCE_KINDS)}")
        if not self.source_trajectory_id:
            out.append(f"{where}.source_trajectory_id is empty")
        if self.kind == "demo" and not (self.dataset and self.data_version):
            out.append(f"{where}: a demo experience needs dataset and data_version")
        if self.kind == "dev_run" and not self.run_id:
            out.append(f"{where}: a dev_run experience needs run_id")
        if self.kind == "dev_run" and self.instance_id is None:
            out.append(f"{where}: a dev_run experience needs instance_id "
                       "(Report / hidden instances are excluded by id)")
        out += _instance_errors(f"{where}.instance_id", self.instance_id)
        return out


def _forbidden_keys(obj: Any, path: str = "payload") -> list[str]:
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k) in FORBIDDEN_RUNTIME_KEYS or str(k).startswith(("sim.", "eval.", "truth")):
                out.append(f"{path}.{k} is a truth / evaluation field; experiences are runtime-visible")
            out += _forbidden_keys(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            out += _forbidden_keys(v, f"{path}[{i}]")
    return out


@dataclasses.dataclass(frozen=True)
class ExperienceRecord:
    """One memory entry. Everything ``to_experience`` exposes is shown to modules at run time, so it
    must be derived from legal observations; the outcome label may come from a past episode's
    official result (an offline label, frozen with the library)."""

    exp_id: str
    provenance: ExperienceProvenance
    skill: str
    object_categories: tuple[str, ...]
    summary: str
    outcome: str
    obs_refs: tuple[DataRef, ...]
    round: int
    frozen: bool = False
    payload: dict[str, Any] = dataclasses.field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION
    synthetic: bool = False

    FIELD_CATEGORIES: ClassVar[dict[str, str]] = {
        "exp_id": POLICY_OBSERVATION,
        "provenance": TRAINING_ANNOTATION,
        "skill": POLICY_OBSERVATION,
        "object_categories": POLICY_OBSERVATION,
        "summary": POLICY_OBSERVATION,
        "outcome": TRAINING_ANNOTATION,
        "obs_refs": POLICY_OBSERVATION,
        "round": TRAINING_ANNOTATION,
        "frozen": TRAINING_ANNOTATION,
        "payload": POLICY_OBSERVATION,
        "schema_version": TRAINING_ANNOTATION,
        "synthetic": TRAINING_ANNOTATION,
    }

    def errors(self) -> list[str]:
        out = []
        for name in ("exp_id", "skill", "summary"):
            if not getattr(self, name):
                out.append(f"{name} is empty")
        out += self.provenance.errors("provenance")
        if self.outcome not in OUTCOMES:
            out.append(f"outcome {self.outcome!r} not in {list(OUTCOMES)}")
        for i, ref in enumerate(self.obs_refs):
            out += ref.errors(f"obs_refs[{i}]", OBS_MODALITIES)
        if self.round < 0:
            out.append("round must be >= 0")
        out += _forbidden_keys(self.payload)
        try:
            json.dumps(self.payload)
        except (TypeError, ValueError) as e:
            out.append(f"payload is not JSON-serialisable: {e}")
        return out

    def to_experience(self) -> Experience:
        return Experience(exp_id=self.exp_id, provenance=self.provenance.text(), skill=self.skill,
                          objects=tuple(self.object_categories), summary=self.summary, outcome=self.outcome,
                          payload=dict(self.payload))


# ---------------------------------------------------------------------------------------------
# 7.3 distillation records
# ---------------------------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class DistillInputs:
    """What the strong model was given: all legal (red line 4; providers/base.py logs exactly this)."""

    user_text: str
    system_sha256: str = ""
    image_refs: tuple[str, ...] = ()  # JPEGs relative to the run directory (episodes/<ep>/frames/...)
    task_name: str = ""
    task_instruction: str = ""
    subgoal_literals: dict[str, str] = dataclasses.field(default_factory=dict)  # BDDL subgoal id -> literal
    subtask_in_force: Optional[dict[str, Any]] = None  # the loop's current subtask when the call was made


@dataclasses.dataclass(frozen=True)
class DistillOutputs:
    """The model's answer, normalised per decision type, plus what the module returned to the loop."""

    decision_type: str  # one of DECISION_TYPES
    status: Optional[str] = None  # checker: done | in_progress | failed | unknown
    confidence: Optional[float] = None
    failure_type: Optional[str] = None
    continue_switch: Optional[str] = None  # one of CONTINUE_SWITCH
    subtask: Optional[dict[str, Any]] = None  # planner: skill, args, target_subgoals, max_steps
    task_complete: Optional[bool] = None
    recovery: Optional[dict[str, Any]] = None
    subgoal_estimates: dict[str, str] = dataclasses.field(default_factory=dict)
    holding: dict[str, str] = dataclasses.field(default_factory=dict)
    module_return: Any = None  # the module's return value as logged in the matching "decision" event
    parsed: Optional[dict[str, Any]] = None
    response_text: str = ""
    error: Optional[str] = None


@dataclasses.dataclass(frozen=True)
class DistillSource:
    events_path: str  # relative to the run directory
    events_sha256: str
    line: int  # 0-based line of the provider_call event


@dataclasses.dataclass(frozen=True)
class DistillRecord:
    record_id: str  # "<episode_id>#L<line>": unique per run
    call_id: str
    run_id: str
    episode_id: str
    step: int
    slot: str
    purpose: str
    provider: str
    model: str
    inputs: DistillInputs
    outputs: DistillOutputs
    visible_evidence: str
    source: DistillSource
    latency_s: Optional[float] = None
    usage: dict[str, Any] = dataclasses.field(default_factory=dict)
    cost_usd: Optional[float] = None
    round: int = 0
    instance_id: Optional[int] = None  # resolved task instance of the episode (Report / hidden are rejected)
    truth_label: Optional[str] = None  # TRUTH_LABELS; filled offline by diagnostics/label_distill.py
    truth_source: Optional[str] = None
    correct: Optional[bool] = None
    truth_detail: Optional[dict[str, Any]] = None
    schema_version: int = SCHEMA_VERSION
    synthetic: bool = False

    FIELD_CATEGORIES: ClassVar[dict[str, str]] = {
        "record_id": TRAINING_ANNOTATION,
        "call_id": TRAINING_ANNOTATION,
        "run_id": TRAINING_ANNOTATION,
        "episode_id": TRAINING_ANNOTATION,
        "step": TRAINING_ANNOTATION,
        "slot": TRAINING_ANNOTATION,
        "purpose": TRAINING_ANNOTATION,
        "provider": TRAINING_ANNOTATION,
        "model": TRAINING_ANNOTATION,
        "inputs": POLICY_OBSERVATION,
        "outputs": TRAINING_ANNOTATION,  # strong-model labels
        "visible_evidence": TRAINING_ANNOTATION,
        "source": TRAINING_ANNOTATION,
        "latency_s": TRAINING_ANNOTATION,
        "usage": TRAINING_ANNOTATION,
        "cost_usd": TRAINING_ANNOTATION,
        "round": TRAINING_ANNOTATION,
        "instance_id": TRAINING_ANNOTATION,
        "truth_label": TRAINING_ANNOTATION,  # simulator truth, offline only
        "truth_source": TRAINING_ANNOTATION,
        "correct": TRAINING_ANNOTATION,
        "truth_detail": TRAINING_ANNOTATION,
        "schema_version": TRAINING_ANNOTATION,
        "synthetic": TRAINING_ANNOTATION,
    }

    @property
    def labelled(self) -> bool:
        return self.truth_label is not None

    def errors(self) -> list[str]:
        out = []
        for name in ("record_id", "run_id", "episode_id"):
            if not getattr(self, name):
                out.append(f"{name} is empty")
        o = self.outputs
        if o.decision_type not in DECISION_TYPES:
            out.append(f"outputs.decision_type {o.decision_type!r} not in {list(DECISION_TYPES)}")
        if o.continue_switch is not None and o.continue_switch not in CONTINUE_SWITCH:
            out.append(f"outputs.continue_switch {o.continue_switch!r} not in {list(CONTINUE_SWITCH)}")
        if o.confidence is not None and not 0.0 <= o.confidence <= 1.0:
            out.append(f"outputs.confidence {o.confidence} outside [0, 1]")
        if self.truth_label is not None and self.truth_label not in TRUTH_LABELS:
            out.append(f"truth_label {self.truth_label!r} not in {list(TRUTH_LABELS)}")
        if (self.truth_label is not None or self.correct is not None) and not self.truth_source:
            out.append("truth_label / correct set without truth_source")
        if self.correct is not None and self.truth_label in (None, "unknown"):
            out.append("correct is set but the truth label is unknown")
        if self.round < 0:
            out.append("round must be >= 0")
        out += _instance_errors("instance_id", self.instance_id)
        return out


# ---------------------------------------------------------------------------------------------
# Field categories
# ---------------------------------------------------------------------------------------------


def _hints(cls: type) -> dict[str, Any]:
    return typing.get_type_hints(cls)


def _dataclass_in(tp: Any) -> Optional[type]:
    """The dataclass inside ``X``, ``Optional[X]`` or ``tuple[X, ...]``, else None."""
    if dataclasses.is_dataclass(tp):
        return tp
    for arg in typing.get_args(tp):
        if arg is not type(None) and arg is not Ellipsis:
            found = _dataclass_in(arg)
            if found is not None:
                return found
    return None


def field_categories(cls: type) -> dict[str, str]:
    """Dotted leaf field -> category, for every leaf field reachable from ``cls``.

    Top-level fields must all be declared in ``cls.FIELD_CATEGORIES``; nested dataclass fields
    inherit their parent's category unless a dotted key ("episode_outcome.q_final") overrides it."""
    declared = dict(getattr(cls, "FIELD_CATEGORIES", {}))
    missing = [f.name for f in dataclasses.fields(cls) if f.name not in declared]
    if missing:
        raise SchemaError(f"{cls.__name__}.FIELD_CATEGORIES misses {missing}")
    bad = {k: v for k, v in declared.items() if v not in CATEGORIES}
    if bad:
        raise SchemaError(f"{cls.__name__}: unknown categories {bad}")
    out: dict[str, str] = {}

    def walk(c: type, prefix: str, inherited: Optional[str]) -> None:
        hints = _hints(c)
        for f in dataclasses.fields(c):
            path = f"{prefix}{f.name}"
            cat = declared.get(path, inherited)
            sub = _dataclass_in(hints[f.name])
            if sub is not None:
                walk(sub, path + ".", cat)
            else:
                out[path] = cat

    walk(cls, "", None)
    return out


# ---------------------------------------------------------------------------------------------
# JSON (de)serialization
# ---------------------------------------------------------------------------------------------


def to_dict(obj: Any) -> Any:
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: to_dict(getattr(obj, f.name)) for f in dataclasses.fields(obj)}
    if isinstance(obj, (list, tuple)):
        return [to_dict(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): to_dict(v) for k, v in obj.items()}
    return obj


def _convert(tp: Any, value: Any, path: str, problems: list[str]) -> Any:
    if tp is Any:
        return value
    origin, args = typing.get_origin(tp), typing.get_args(tp)
    if origin in (typing.Union, types.UnionType):
        if value is None and type(None) in args:
            return None
        inner = [a for a in args if a is not type(None)]
        return _convert(inner[0], value, path, problems) if len(inner) == 1 else value
    if dataclasses.is_dataclass(tp):
        return _from_dict(tp, value, path, problems)
    if origin is tuple or origin is list:
        if not isinstance(value, (list, tuple)):
            problems.append(f"{path}: expected a list, got {type(value).__name__}")
            return () if origin is tuple else []
        item = args[0] if args else Any
        items = [_convert(item, v, f"{path}[{i}]", problems) for i, v in enumerate(value)]
        return tuple(items) if origin is tuple else items
    if origin is dict:
        if not isinstance(value, dict):
            problems.append(f"{path}: expected an object, got {type(value).__name__}")
            return {}
        vt = args[1] if len(args) == 2 else Any
        return {str(k): _convert(vt, v, f"{path}.{k}", problems) for k, v in value.items()}
    if tp is bool:
        if not isinstance(value, bool):
            problems.append(f"{path}: expected a bool, got {value!r}")
        return value
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            problems.append(f"{path}: expected an int, got {value!r}")
        return value
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            problems.append(f"{path}: expected a number, got {value!r}")
            return value
        return float(value)
    if tp is str:
        if not isinstance(value, str):
            problems.append(f"{path}: expected a string, got {value!r}")
        return value
    return value


def _from_dict(cls: type, data: Any, path: str, problems: list[str]) -> Any:
    if not isinstance(data, dict):
        problems.append(f"{path or cls.__name__}: expected an object, got {type(data).__name__}")
        return None
    hints = _hints(cls)
    known = {f.name for f in dataclasses.fields(cls)}
    for k in sorted(set(data) - known):
        problems.append(f"{path}{'.' if path else ''}{k}: unknown field for {cls.__name__}")
    kwargs = {}
    for f in dataclasses.fields(cls):
        where = f"{path}{'.' if path else ''}{f.name}"
        if f.name not in data:
            if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
                problems.append(f"{where}: missing required field")
            continue
        kwargs[f.name] = _convert(hints[f.name], data[f.name], where, problems)
    try:
        return cls(**kwargs)
    except TypeError as e:
        problems.append(f"{path or cls.__name__}: {e}")
        return None


def from_dict(cls: type, data: Any, *, validate: bool = True) -> Any:
    """Build ``cls`` from a JSON-like dict; raises SchemaError listing every type / field problem
    (and, with ``validate``, every ``errors()`` entry)."""
    problems: list[str] = []
    obj = _from_dict(cls, data, "", problems)
    if not problems and validate and hasattr(obj, "errors"):
        problems = obj.errors()
    if problems:
        raise SchemaError(f"{cls.__name__}: " + "; ".join(problems))
    return obj


def check(obj: Any) -> Any:
    """Raise SchemaError when ``obj.errors()`` is not empty; return ``obj``."""
    problems = obj.errors()
    if problems:
        ident = next((getattr(obj, k) for k in ("record_id", "segment_id", "exp_id") if getattr(obj, k, "")), "")
        raise SchemaError(f"{type(obj).__name__} {ident}: " + "; ".join(problems))
    return obj


def dumps(obj: Any) -> str:
    return json.dumps(to_dict(obj), sort_keys=True, ensure_ascii=False)


def loads(cls: type, text: str, *, validate: bool = True) -> Any:
    return from_dict(cls, json.loads(text), validate=validate)


def write_jsonl(path: str, records: Iterable[Any], *, validate: bool = True) -> int:
    """Write records atomically (tmp file + rename); returns the number written."""
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp{os.getpid()}"
    n = 0
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            for rec in records:
                if validate:
                    check(rec)
                f.write(dumps(rec) + "\n")
                n += 1
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    return n


def read_jsonl(path: str, cls: type, *, validate: bool = True) -> list:
    out = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if not line.strip():
                continue
            try:
                out.append(loads(cls, line, validate=validate))
            except SchemaError as e:
                raise SchemaError(f"{path}:{i + 1}: {e}") from None
    return out


# ---------------------------------------------------------------------------------------------
# Source-isolated split
# ---------------------------------------------------------------------------------------------


def _unit_interval(seed: int, key: tuple[str, str]) -> float:
    h = hashlib.sha256(json.dumps([int(seed), *key]).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") / float(1 << 64)


def split_by_source(segments: Sequence[SkillSegment], dev_fraction: float, seed: int = 0) -> list[SkillSegment]:
    """Assign every segment the split of its source trajectory (``SkillSegment.source_key``).

    A trajectory goes to dev when sha256(seed, dataset, trajectory id) mapped to [0, 1) is below
    ``dev_fraction``. The assignment depends only on (seed, dev_fraction, trajectory), so adding
    trajectories in a later round never moves an earlier one across the split as long as seed and
    dev_fraction stay the same (changing either reassigns trajectories: keep them fixed across rounds);
    the realised dev fraction is approximate for small trajectory counts."""
    if not 0.0 <= dev_fraction <= 1.0:
        raise ValueError(f"dev_fraction must be in [0, 1], got {dev_fraction}")
    return [dataclasses.replace(s, split="dev" if _unit_interval(seed, s.source_key) < dev_fraction else "train")
            for s in segments]


def split_violations(segments: Sequence[SkillSegment]) -> list[str]:
    by_source: dict[tuple[str, str], set[str]] = {}
    ids: dict[str, int] = {}
    for s in segments:
        by_source.setdefault(s.source_key, set()).add(s.split)
        ids[s.segment_id] = ids.get(s.segment_id, 0) + 1
    out = [f"source trajectory {k[0]}/{k[1]} spans splits {sorted(v)}"
           for k, v in sorted(by_source.items()) if len(v) > 1]
    out += [f"split {s.split!r} of segment {s.segment_id} not in {list(SPLITS)}"
            for s in segments if s.split not in SPLITS]
    out += [f"segment_id {k} appears {n} times" for k, n in sorted(ids.items()) if n > 1]
    for s in segments:
        out += [f"segment {s.segment_id}: {e}" for e in _instance_errors("source_instance_id", s.source_instance_id)]
    return out


def validate_split(segments: Sequence[SkillSegment]) -> None:
    """Raise SchemaError when a source trajectory spans train and dev, a split value is invalid, a
    segment id repeats, or a Report / hidden instance appears."""
    problems = split_violations(segments)
    if problems:
        raise SchemaError("; ".join(problems))
