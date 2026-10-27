"""The benchmark tasks this QA reproduces: one entry per task ID of COVERAGE.md that has questions.

A question's task is ``q["subtasks"][0]``; the other entries are tags (SITE's aggregate classes, MindCube's setting
and dynamics labels), never a second task. ``TASKS[task]`` names the benchmark and the benchmark's own category for
the task, spelt exactly as the benchmark's data and its lmms-eval scorer spell it, so that a score on our test set
lines up with the same row of the benchmark's report:

* VSI-Bench: ``question_type`` (direction easy / medium / hard are three types; the benchmark's overall averages them
  into one task, ``main``);
* MMSI-Bench: ``question_type`` as lmms-eval reports it (en dash: ``Positional Relationship (Cam.–Cam.)``);
* MindCube: the prefix of the question id (``rotation`` / ``around`` / ``among``); translation questions are not in
  MindCube-Tiny and are reported on their own;
* ViewSpatial-Bench: ``question_type``;
* EmbSpatial-Bench: ``relation`` (below is ``under`` in the data);
* 3DSRBench: ``category`` (``main``: height / location / multi_object);
* BLINK: the subset (``dataset_name`` of the lmms-eval task).

The values were read on 2026-10-27 from the cached benchmark data (VSI-Bench test.jsonl, EmbSpatial-Bench parquet,
ViewSpatial-Bench.json) and from the lmms-eval tasks in third_party/lmms-eval (mmsi_bench/msr_bench.yaml,
mindcube/utils.py, 3dsrbench/utils.py, blink/*.yaml).

``check(q)`` raises ``TaskMismatch`` for a question that does not belong to exactly one task, or whose benchmark,
group, prompt layout or answer kind differ from its task's.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Tuple

SAME = "same"        # 能做，与原定义一致
DIFF = "differs"     # 能做，定义有差异（逐题标注）

VSI, MMSI, MC, VS, ES, D3, BL = ("VSI-Bench", "MMSI-Bench", "MindCube", "ViewSpatial-Bench", "EmbSpatial-Bench",
                                 "3DSRBench", "BLINK")


@dataclass(frozen=True)
class Task:
    benchmark: str
    field: str            # where the benchmark keeps the category
    category: str         # the benchmark's own value
    main: str             # the benchmark's top-level task the category is averaged into
    groups: Tuple[str, ...]
    styles: Tuple[str, ...]   # prompt layout of the original-format wording (qa.STYLES)
    numeric: bool
    definition: str       # SAME or DIFF (COVERAGE.md)


def _t(b, field, cat, groups, styles, numeric=False, definition=SAME, main=None) -> Task:
    return Task(b, field, cat, main or cat, tuple(groups), tuple(styles), numeric, definition)


_VQT, _MQT, _MID, _BLS = "question_type", "question_type", "id prefix", "subset"
TASKS: Dict[str, Task] = {
    # VSI-Bench (video)
    "VSI-count": _t(VSI, _VQT, "object_counting", ["V1"], ["vsi_na"], True),
    "VSI-distance-absolute": _t(VSI, _VQT, "object_abs_distance", ["V2"], ["vsi_na"], True),
    "VSI-distance-relative": _t(VSI, _VQT, "object_rel_distance", ["V2"], ["vsi_mc"]),
    "VSI-size-object": _t(VSI, _VQT, "object_size_estimation", ["V3"], ["vsi_na"], True),
    "VSI-size-room": _t(VSI, _VQT, "room_size_estimation", ["V3"], ["vsi_na"], True, DIFF),
    "VSI-direction-easy": _t(VSI, _VQT, "object_rel_direction_easy", ["V4"], ["vsi_mc"], main="object_rel_direction"),
    "VSI-direction-medium": _t(VSI, _VQT, "object_rel_direction_medium", ["V4"], ["vsi_mc"], main="object_rel_direction"),
    "VSI-direction-hard": _t(VSI, _VQT, "object_rel_direction_hard", ["V4"], ["vsi_mc"], main="object_rel_direction"),
    "VSI-route": _t(VSI, _VQT, "route_planning", ["V5"], ["vsi_mc"], definition=DIFF),
    "VSI-order": _t(VSI, _VQT, "obj_appearance_order", ["V6"], ["vsi_mc"]),
    # MMSI-Bench (multi-image)
    "MMSI-position-camera-camera": _t(MMSI, _MQT, "Positional Relationship (Cam.–Cam.)", ["M1"], ["mmsi"]),
    "MMSI-motion-camera": _t(MMSI, _MQT, "Motion (Cam.)", ["M1"], ["mmsi"]),
    "MMSI-position-camera-object": _t(MMSI, _MQT, "Positional Relationship (Cam.–Obj.)", ["M2"], ["mmsi"]),
    "MMSI-attribute-measurement": _t(MMSI, _MQT, "Attribute (Meas.)", ["M2"], ["mmsi"]),
    "MMSI-position-camera-region": _t(MMSI, _MQT, "Positional Relationship (Cam.–Reg.)", ["M3"], ["mmsi"], definition=DIFF),
    "MMSI-attribute-appearance": _t(MMSI, _MQT, "Attribute (Appr.)", ["M5"], ["mmsi"], definition=DIFF),
    "MMSI-multistep": _t(MMSI, _MQT, "MSR", ["M6"], ["mmsi"], definition=DIFF),
    # MindCube (multi-image)
    "MC-rotation-current": _t(MC, _MID, "rotation", ["M7"], ["mindcube"]),
    "MC-rotation-mental-turn": _t(MC, _MID, "rotation", ["M7"], ["mindcube"]),
    "MC-around-q1": _t(MC, _MID, "around", ["M8"], ["mindcube"]),
    "MC-around-q2": _t(MC, _MID, "around", ["M8"], ["mindcube"]),
    "MC-around-q3": _t(MC, _MID, "around", ["M8"], ["mindcube"]),
    "MC-among-q1": _t(MC, _MID, "among", ["M9"], ["mindcube"]),
    "MC-among-q2-current": _t(MC, _MID, "among", ["M9"], ["mindcube"]),
    "MC-among-q2-sequence": _t(MC, _MID, "among", ["M9"], ["mindcube"]),
    "MC-among-q5": _t(MC, _MID, "among", ["M9"], ["mindcube"]),
    "MC-translation-transitive": _t(MC, _MID, "translation", ["M10"], ["mindcube"]),
    # ViewSpatial-Bench
    "VS-01": _t(VS, "question_type", "Camera perspective - Relative Direction", ["S1"], ["viewspatial"], definition=DIFF),
    "VS-05": _t(VS, "question_type", "Person perspective - Scene Simulation Relative Direction", ["M11"], ["viewspatial"]),
    # EmbSpatial-Bench (single image)
    "ES-01": _t(ES, "relation", "left", ["S1"], ["embspatial"], definition=DIFF),
    "ES-02": _t(ES, "relation", "right", ["S1"], ["embspatial"], definition=DIFF),
    "ES-03": _t(ES, "relation", "above", ["S2"], ["embspatial"], definition=DIFF),
    "ES-04": _t(ES, "relation", "under", ["S2"], ["embspatial"], definition=DIFF),
    "ES-05": _t(ES, "relation", "close", ["S3"], ["embspatial"]),
    "ES-06": _t(ES, "relation", "far", ["S3"], ["embspatial"]),
    # 3DSRBench (single image)
    "3D-01": _t(D3, "category", "height_higher", ["S2"], ["3dsr"], definition=DIFF, main="height"),
    "3D-02": _t(D3, "category", "location_above", ["S2"], ["3dsr"], definition=DIFF, main="location"),
    "3D-03": _t(D3, "category", "location_closer_to_camera", ["S3"], ["3dsr"], definition=DIFF, main="location"),
    "3D-04": _t(D3, "category", "location_next_to", ["S4"], ["3dsr"], definition=DIFF, main="location"),
    "3D-08": _t(D3, "category", "multi_object_closer_to", ["S4"], ["3dsr"], definition=DIFF, main="multi_object"),
    # BLINK
    "BL-01": _t(BL, _BLS, "Counting", ["S5"], ["blink"]),
    "BL-02": _t(BL, _BLS, "Relative_Depth", ["S3"], ["blink"]),
    "BL-03": _t(BL, _BLS, "Spatial_Relation", ["S1", "S2", "S4"], ["blink"], definition=DIFF),
    "BL-04": _t(BL, _BLS, "Multi-view_Reasoning", ["M1"], ["blink_options"]),
    "BL-05": _t(BL, _BLS, "Object_Localization", ["S6"], ["blink_options"], definition=DIFF),
    "BL-06": _t(BL, _BLS, "Visual_Correspondence", ["M4"], ["blink_options"], definition=DIFF),
}

BENCHMARKS = (VSI, MMSI, MC, VS, ES, D3, BL)


class TaskMismatch(ValueError):
    pass


def check(q: Dict[str, Any]) -> str:
    """The question's task, after checking that the question matches it exactly."""
    task = q["subtasks"][0]
    t = TASKS.get(task)
    qid = q.get("qid", "?")
    if t is None:
        raise TaskMismatch(f"{qid}: task {task!r} is not a benchmark task with questions")
    extra = [s for s in q["subtasks"][1:] if s in TASKS]
    if extra:
        raise TaskMismatch(f"{qid}: second task {extra} besides {task}")
    style = q["variants"]["orig"].get("style")
    problems = []
    if q["benchmark"] != t.benchmark:
        problems.append(f"benchmark {q['benchmark']} != {t.benchmark}")
    if q["group"] not in t.groups:
        problems.append(f"group {q['group']} not in {t.groups}")
    if style not in t.styles:
        problems.append(f"layout {style} not in {t.styles}")
    if (q["answer_type"] == "numeric") != t.numeric:
        problems.append(f"answer type {q['answer_type']}")
    if problems:
        raise TaskMismatch(f"{qid} ({task}): " + "; ".join(problems))
    return task


def record(task: str) -> Dict[str, Any]:
    """The task's fields for an exported question."""
    t = TASKS[task]
    d = asdict(t)
    d.pop("styles")
    d.pop("groups")
    return dict(task=task, **d)
