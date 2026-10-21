"""Failure records and the failure distribution (task spec 6.7).

    python -m eval.failures skeleton RUN_DIR [RUN_DIR ...] --out failures.jsonl [--video-fps 30]
    python -m eval.failures label failures.jsonl --record ID --step S --category grasp_place \
                                  --labeler NAME [--outcome TEXT] [--notes TEXT] [--new]
    python -m eval.failures summarize failures.jsonl [--out failures.md]

skeleton: one record per episode with official Q < 1 (or no valid result). The episode is joined to its
official result exactly as eval/aggregate.py does. The record carries the episode's decision timeline
from the agent's events.jsonl (agent/recorder.py): subtask starts and ends, check judgments, recoveries,
fallbacks, executor instruction switches and provider errors. Each timeline moment has the step, the
video time, the skill and instruction in force, the model's latest judgment and the loop's next action.
The skeleton never assigns a category; every record starts as "uncertain" with no labeler. A human picks
the failure moment from the video around the candidate moments (never from the last frame alone), and
``label`` copies that moment's context into the record.

Video time: the official evaluator writes one frame per executed step to
<output-dir>/videos/<task>_<instance_id>_<rollout_id>.mp4 at --video-fps (default 30)
(OmniGibson/omnigibson/eval/evaluator.py _write_video). The observation the agent sees at step s is the
frame written after step s-1, so video_time_s = step / fps is accurate to one frame.

Status: written and unit-tested on a SYNTHETIC events.jsonl; no real failure has been labelled yet.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from typing import Any, Iterable, Optional, Sequence

from eval.aggregate import InputLog, episode_rows, load_run, md_table

CATEGORIES = (
    "env_service_error",
    "target_recognition",
    "task_order",
    "reach_and_positioning",
    "grasp_place",
    "skill_switch_handover",
    "recovery_ineffective",
    "predicate_regression",
    "uncertain",
)
CATEGORY_LABELS_ZH = {
    "env_service_error": "环境/服务错误",
    "target_recognition": "目标识别",
    "task_order": "任务顺序",
    "reach_and_positioning": "到达与站位",
    "grasp_place": "抓放操作",
    "skill_switch_handover": "技能切换与交接",
    "recovery_ineffective": "恢复无效",
    "predicate_regression": "谓词退化",
    "uncertain": "原因不确定",
}
DEFAULT_CATEGORY = "uncertain"
DEFAULT_VIDEO_FPS = 30.0
SCHEMA_VERSION = 1

RECORD_FIELDS = (
    "schema_version", "record_id", "run_id", "config", "episode_id", "task", "instance_id", "rollout_id", "seed",
    "q_final", "success", "official_failure_reason", "infra_error", "steps",
    "video_path", "video_exists", "video_fps", "video_time_s", "step",
    "current_skill", "current_instruction", "model_judgment", "actual_outcome", "next_action",
    "category", "category_zh", "labeler", "evidence_notes",
    "events_path", "events_sha256", "timeline",
)

# Moment kinds and their order within one step (a check precedes the recovery or plan it triggers).
_PRIO = {"provider_error": 0, "judgment": 1, "recovery": 2, "recovery_none": 2, "subtask_end": 3,
         "plan_exhausted": 4, "fallback": 4, "subtask_start": 4, "adjust": 5, "instruction": 6}
ACTION_KINDS = frozenset({"recovery", "recovery_none", "subtask_end", "plan_exhausted", "fallback", "subtask_start",
                          "adjust", "instruction"})


def _int(x: Any) -> Optional[int]:
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def _moment(step: Any, kind: str, summary: str, **extra) -> dict:
    return {"step": _int(step), "kind": kind, "summary": summary, **extra}


def _judgment(out: dict) -> dict:
    return {k: out.get(k) for k in ("status", "confidence", "failure_type", "evidence", "source")}


def build_timeline(events: Iterable[dict], video_fps: float = DEFAULT_VIDEO_FPS) -> list[dict]:
    """Decision timeline of one episode from its events.jsonl records."""
    moments: list[dict] = []
    exec_instruction = None
    end_summary = None
    for ev in events:
        kind, step = ev.get("kind"), ev.get("step")
        if kind == "decision":
            slot, out = ev.get("slot"), ev.get("out")
            if slot == "plan" and out is None:
                moments.append(_moment(step, "plan_exhausted", "planner returned no subtask"))
            elif slot == "plan" and isinstance(out, dict):
                k = "fallback" if "fallback" in str(out.get("source", "")) else "subtask_start"
                moments.append(_moment(step, k, f"{out.get('skill')}: {out.get('instruction')} [{out.get('source')}]",
                                       skill=out.get("skill"), instruction=out.get("instruction"),
                                       evidence=out.get("evidence")))
            elif slot == "check" and isinstance(out, dict) and out.get("source") != "check.null":
                j = _judgment(out)
                moments.append(_moment(step, "judgment", f"{j['status']} ({j['confidence']}) {j['failure_type'] or ''}".strip(),
                                       judgment=j))
            elif slot == "recover":
                if isinstance(out, dict):
                    moments.append(_moment(step, "recovery", f"{out.get('kind')}:{out.get('primitive')} {out.get('reason') or ''}".strip(),
                                           instruction=out.get("instruction")))
                else:
                    moments.append(_moment(step, "recovery_none", "recovery module proposed nothing"))
            elif slot in ("approach", "memory", "explore") and isinstance(out, dict):
                moments.append(_moment(step, "adjust", f"{slot}: {out.get('reason') or out.get('instruction') or ''}".strip()))
        elif kind == "executor":
            instr = ev.get("instruction")
            if instr != exec_instruction:
                moments.append(_moment(step, "instruction", f"executor instruction -> {instr!r}", instruction=instr))
                exec_instruction = instr
        elif kind == "provider_call" and ev.get("error"):
            moments.append(_moment(step, "provider_error", f"{ev.get('slot')}: {ev.get('error')}"))
        elif kind == "episode_end":
            end_summary = ev.get("summary") or {}

    for rec in (end_summary or {}).get("subtasks") or []:
        st = rec.get("subtask") or {}
        if rec.get("outcome") == "open" or rec.get("end_step") is None:
            continue
        moments.append(_moment(rec.get("end_step"), "subtask_end",
                               f"{st.get('skill')} -> {rec.get('outcome')} (retries {rec.get('retries')})",
                               skill=st.get("skill"), outcome=rec.get("outcome")))
        if str(st.get("source", "")).startswith("loop.fallback"):
            moments.append(_moment(rec.get("start_step"), "fallback", f"loop fallback [{st.get('source')}]: full task prompt",
                                   skill=st.get("skill"), instruction=st.get("instruction")))

    order = sorted(range(len(moments)), key=lambda i: (moments[i]["step"] if moments[i]["step"] is not None else 1 << 30,
                                                       _PRIO.get(moments[i]["kind"], 9), i))
    moments = [moments[i] for i in order]
    skill = instruction = judgment = None
    for m in moments:
        if m["kind"] in ("subtask_start", "fallback"):
            skill, instruction = m.get("skill"), m.get("instruction")
        elif m["kind"] == "instruction":
            instruction = m.get("instruction")
        elif m["kind"] == "judgment":
            judgment = m["judgment"]
        m["video_time_s"] = round(m["step"] / video_fps, 3) if m["step"] is not None else None
        m["current_skill"], m["current_instruction"], m["model_judgment"] = skill, instruction, judgment
    end = (end_summary or {}).get("steps")
    for i, m in enumerate(moments):
        nxt = next((n for n in moments[i + 1:] if n["kind"] in ACTION_KINDS), None)
        m["next_action"] = f"step {nxt['step']}: {nxt['kind']}: {nxt['summary']}" if nxt else f"episode end (steps {end})"
    return moments


def read_events(path: str, inputs: InputLog) -> list[dict]:
    out = []
    for line in inputs.read_bytes(path).decode("utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                out.append({"kind": "unparseable"})
    return out


def _events_path(row: dict, local_root: Optional[str]) -> Optional[str]:
    cands = []
    if row.get("summary_path"):
        cands.append(os.path.join(os.path.dirname(row["summary_path"]), "events.jsonl"))
    if local_root and row.get("episode_id"):
        cands.append(os.path.join(os.path.expanduser(local_root), row["run_id"], "episodes", row["episode_id"], "events.jsonl"))
    return next((p for p in cands if os.path.isfile(p)), None)


def skeleton_records(run_dirs: Sequence[str], video_fps: Optional[float] = None, q_below: float = 1.0,
                     local_root: Optional[str] = None) -> list[dict]:
    inputs = InputLog()
    records = []
    for i, run_dir in enumerate(run_dirs):
        run = load_run(run_dir, inputs, order=i)
        fps = float(video_fps or run.status.get("video_fps") or DEFAULT_VIDEO_FPS)
        rows, _ = episode_rows(run)
        for row in rows:
            if row["q_final"] is not None and row["q_final"] >= q_below and not row["infra_error"]:
                continue
            epath = _events_path(row, local_root)
            timeline = build_timeline(read_events(epath, inputs), fps) if epath else []
            ident = row["episode_id"] or f"{row['run_id']}:{row['task']}_{row['instance_id']}_{row['rollout_id']}"
            rec = {f: None for f in RECORD_FIELDS}
            rec.update(
                schema_version=SCHEMA_VERSION, record_id=f"{ident}#0", run_id=row["run_id"], config=run.config_name,
                episode_id=row["episode_id"], task=row["task"], instance_id=row["instance_id"],
                rollout_id=row["rollout_id"], seed=row["seed"], q_final=row["q_final"], success=row["success"],
                official_failure_reason=row["failure_reason"], infra_error=row["infra_error"], steps=row["steps"],
                video_path=row["video_path"], video_exists=bool(row["video_path"] and os.path.isfile(row["video_path"])),
                video_fps=fps, actual_outcome="", category=DEFAULT_CATEGORY,
                category_zh=CATEGORY_LABELS_ZH[DEFAULT_CATEGORY], labeler="",
                evidence_notes="" if epath else "events.jsonl not found; no decision timeline",
                events_path=epath, events_sha256=inputs.files[os.path.abspath(epath)]["sha256"] if epath else None,
                timeline=timeline)
            records.append(rec)
    return records


def anchor(record: dict, step: int, timeline: Optional[list] = None) -> dict:
    """Copy the context in force at ``step`` (from the decision timeline) into the record."""
    tl = record.get("timeline") if timeline is None else timeline
    tl = tl or []
    idx = max((i for i, m in enumerate(tl) if m["step"] is not None and m["step"] <= step), default=None)
    m = tl[idx] if idx is not None else {}
    nxt = next((n for i, n in enumerate(tl) if n["kind"] in ACTION_KINDS and n["step"] is not None
                and (n["step"] > step or (idx is not None and n["step"] == step and i > idx))), None)
    record.update(step=int(step), video_time_s=round(step / float(record.get("video_fps") or DEFAULT_VIDEO_FPS), 3),
                  current_skill=m.get("current_skill"), current_instruction=m.get("current_instruction"),
                  model_judgment=m.get("model_judgment"),
                  next_action=f"step {nxt['step']}: {nxt['kind']}: {nxt['summary']}" if nxt else "none until episode end")
    return record


def load_records(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_records(path: str, records: Sequence[dict]) -> None:
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)


def label(path: str, record_id: str, step: int, category: str, labeler: str, outcome: str = "", notes: str = "",
          new: bool = False) -> dict:
    if category not in CATEGORIES:
        raise SystemExit(f"category must be one of {list(CATEGORIES)}, got {category!r}")
    if not labeler.strip():
        raise SystemExit("--labeler is required")
    records = load_records(path)
    base = next((r for r in records if r["record_id"] == record_id), None)
    if base is None:
        raise SystemExit(f"no record {record_id!r} in {path}")
    if new:
        stem = record_id.rsplit("#", 1)[0]
        n = 1 + max(int(r["record_id"].rsplit("#", 1)[1]) for r in records if r["record_id"].rsplit("#", 1)[0] == stem)
        rec = {**base, "record_id": f"{stem}#{n}", "timeline": [], "timeline_ref": record_id}
        anchor(rec, step, timeline=base.get("timeline"))
        records.insert(records.index(base) + n, rec)
    else:
        rec = anchor(base, step)
    rec.update(category=category, category_zh=CATEGORY_LABELS_ZH[category], labeler=labeler.strip(),
               actual_outcome=outcome, evidence_notes=notes)
    write_records(path, records)
    return rec


def summarize(records: Sequence[dict]) -> dict:
    bad = sorted({r.get("category") for r in records} - set(CATEGORIES), key=str)
    if bad:
        raise ValueError(f"unknown categories {bad}; allowed: {list(CATEGORIES)}")
    total = len(records)
    counts = collections.Counter(r["category"] for r in records)
    episodes = collections.defaultdict(set)
    per_task = collections.defaultdict(collections.Counter)
    for r in records:
        episodes[r["category"]].add(r["record_id"].rsplit("#", 1)[0])
        per_task[r["task"]][r["category"]] += 1
    return {
        "n_records": total,
        "n_episodes": len({r["record_id"].rsplit("#", 1)[0] for r in records}),
        "n_unlabelled": sum(1 for r in records if not r.get("labeler")),
        "categories": [{"category": c, "label_zh": CATEGORY_LABELS_ZH[c], "count": counts[c],
                        "share": counts[c] / total if total else None, "episodes": len(episodes[c])} for c in CATEGORIES],
        "per_task": {t: {c: per_task[t][c] for c in CATEGORIES} for t in sorted(per_task)},
    }


def render_summary(s: dict) -> str:
    out = ["# Failure distribution (task spec 6.7)", "",
           f"Generated by `eval/failures.py summarize`: {s['n_records']} record(s) from {s['n_episodes']} episode(s); "
           f"{s['n_unlabelled']} unlabelled (category defaults to uncertain until a human labels it).", ""]
    out.append(md_table(["category", "类别", "count", "share", "episodes"],
                        [[c["category"], c["label_zh"], c["count"], c["share"], c["episodes"]] for c in s["categories"]]))
    out += ["", "## Per task", ""]
    out.append(md_table(["task"] + list(CATEGORIES) + ["total"],
                        [[t] + [row[c] for c in CATEGORIES] + [sum(row.values())] for t, row in s["per_task"].items()]))
    return "\n".join(out) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("skeleton", help="unlabelled records for every episode with Q < 1")
    p.add_argument("run_dirs", nargs="+")
    p.add_argument("--out", required=True)
    p.add_argument("--video-fps", type=float, default=None, help="evaluator --video-fps (default: status.json or 30)")
    p.add_argument("--q-below", type=float, default=1.0)
    p.add_argument("--local-root", default=None, help="local log root, searched when the NAS events.jsonl is absent")
    p.add_argument("--overwrite", action="store_true", help="replace an existing file (its labels are lost)")
    p = sub.add_parser("label", help="anchor a record at a step and set its category")
    p.add_argument("path")
    p.add_argument("--record", required=True)
    p.add_argument("--step", type=int, required=True)
    p.add_argument("--category", required=True, choices=CATEGORIES)
    p.add_argument("--labeler", required=True)
    p.add_argument("--outcome", default="", help="what actually happened at that moment (from the video)")
    p.add_argument("--notes", default="", help="evidence: video times, frames, log lines")
    p.add_argument("--new", action="store_true", help="add a further failure record for the same episode")
    p = sub.add_parser("summarize", help="failure distribution table")
    p.add_argument("path")
    p.add_argument("--out", default=None, help="markdown file (default: stdout)")
    p.add_argument("--json", action="store_true", help="print JSON instead of markdown")
    args = ap.parse_args(argv)

    if args.cmd == "skeleton":
        if os.path.exists(args.out) and not args.overwrite:
            raise SystemExit(f"{args.out} exists (it may hold labels); pass --overwrite to replace it")
        records = skeleton_records(args.run_dirs, args.video_fps, args.q_below, args.local_root)
        write_records(args.out, records)
        print(f"wrote {len(records)} unlabelled record(s) to {args.out}")
    elif args.cmd == "label":
        rec = label(args.path, args.record, args.step, args.category, args.labeler, args.outcome, args.notes, args.new)
        print(f"{rec['record_id']}: step {rec['step']} ({rec['video_time_s']} s) -> {rec['category']}")
    else:
        s = summarize(load_records(args.path))
        text = json.dumps(s, indent=2, ensure_ascii=False) if args.json else render_summary(s)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                f.write(text)
        else:
            print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
