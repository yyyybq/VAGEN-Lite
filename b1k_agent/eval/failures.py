"""Failure records and the failure distribution (task spec 6.7).

    python -m eval.failures skeleton RUN_DIR [RUN_DIR ...] --out failures.jsonl [--video-fps 30]
    python -m eval.failures label failures.jsonl --record ID --step S --category grasp_place \
                                  --labeler NAME [--outcome TEXT] [--notes TEXT] [--new]
    python -m eval.failures summarize failures.jsonl [--out failures.md]

skeleton: one record per episode with official Q < 1, no valid result (infra error, missing), or official
Q = 1 but over the 2026 budget (scores 0 under that rule); ``selected_because`` says which. The episode
is joined to its official result exactly as eval/aggregate.py does. The record carries the episode's
decision timeline from the agent's events.jsonl (agent/recorder.py decision / executor / provider_call
records, plus the summary's subtask records): subtask starts and ends, planner proposals the goal manager
rejected, check judgments, recoveries, loop fallbacks, executor instruction switches and provider errors.
Each timeline moment has the step, the video time, the skill and instruction in force, the model's latest
judgment and the loop's next action. The skeleton never assigns a category; every record starts as
"uncertain" with no labeler. A human picks the failure moment from the video around the candidate
moments (never from the last frame alone), and ``label`` copies that moment's context into the record.

Video time: the official evaluator writes no frame at reset and one frame after every executed step to
<output-dir>/videos/<task>_<instance_id>_<rollout_id>.mp4 at --video-fps (default 30)
(OmniGibson/omnigibson/eval/evaluator.py load_batch / step_fn / _write_video, v3.9.3-post2; 2026/eval
also skips frames of failed or non-advanced envs). The agent's observation at step s (s >= 1) is frame s-1,
so video_time_s = step / fps shows the result of the action chosen at step s: accurate to one frame.

Status: written and unit-tested on a SYNTHETIC events.jsonl and on the recorder's output from a mock
episode; no real failure has been labelled yet (nothing has run on server 105).
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from typing import Any, Iterable, Optional, Sequence

from eval.aggregate import InputLog, episode_rows, expand_run_dirs, load_run, md_table

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
    "schema_version", "record_id", "run_id", "config", "condition", "episode_id", "task", "instance_id", "rollout_id",
    "seed", "selected_because", "q_final", "q_budget2026", "budget_ok", "success", "official_failure_reason",
    "infra_error", "steps", "synthetic",
    "video_path", "video_exists", "video_fps", "video_time_s", "step",
    "current_skill", "current_instruction", "model_judgment", "actual_outcome", "next_action",
    "category", "category_zh", "labeler", "evidence_notes",
    "events_path", "events_sha256", "timeline",
)

# Moment kinds and their order within one step: a check precedes the recovery or the end it triggers, an
# end precedes the planning round, rejected proposals precede the subtask that started.
_PRIO = {"provider_error": 0, "judgment": 1, "recovery": 2, "recovery_none": 2, "subtask_end": 3,
         "plan_rejected": 4, "plan_exhausted": 4, "fallback": 5, "subtask_start": 5, "adjust": 6, "instruction": 7}
ACTION_KINDS = frozenset({"recovery", "recovery_none", "subtask_end", "plan_rejected", "plan_exhausted", "fallback",
                          "subtask_start", "adjust", "instruction"})
PROPOSING_SLOTS = ("plan", "explore")  # Planner.next_subtask / Explorer.propose return a Subtask


def _int(x: Any) -> Optional[int]:
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def _moment(step: Any, kind: str, summary: str, **extra) -> dict:
    return {"step": _int(step), "kind": kind, "summary": summary, **extra}


def _decision_moment(slot: str, out: Any, step: Any) -> Optional[dict]:
    """A timeline moment for one recorder ``decision`` record that is not a subtask proposal, or None."""
    if slot == "plan" and out is None:
        return _moment(step, "plan_exhausted", "planner returned no subtask")
    if slot == "check" and isinstance(out, dict) and out.get("source") != "check.null":
        j = {k: out.get(k) for k in ("status", "confidence", "failure_type", "evidence", "source")}
        j["step"] = _int(step)
        return _moment(step, "judgment", f"{j['status']} ({j['confidence']}) {j['failure_type'] or ''}".strip(),
                       judgment=j)
    if slot == "recover" and isinstance(out, dict):
        return _moment(step, "recovery", f"{out.get('kind')}:{out.get('primitive')} {out.get('reason') or ''}".strip(),
                       instruction=out.get("instruction"))
    if slot == "recover":
        return _moment(step, "recovery_none", "recovery module proposed nothing")
    if slot in ("approach", "memory") and isinstance(out, dict):  # AdjustPlan
        return _moment(step, "adjust", f"{slot}: {out.get('reason') or out.get('instruction') or ''}".strip(),
                       instruction=out.get("instruction"))
    return None


def _subtask_text(st: dict) -> str:
    return f"{st.get('skill')}: {st.get('instruction')} [{st.get('source')}]"


def _starts(proposals: list[dict], records: list[dict], instructions: list[dict],
            complete: bool = True) -> tuple[list[dict], list[dict]]:
    """(started-subtask moments, rejected-proposal moments).

    The summary's subtask records (in start order) say what actually ran; their ``start_step`` is not
    used because the loop moves it to the start of the latest step-budget window (agent/loop.py _decide).
    A planner/explorer-started record starts at the step of the first matching proposal after the
    previous match; proposals skipped on the way were vetoed by the goal manager. A loop fallback starts
    when the executor first receives its instruction after the previous subtask ended. With an incomplete
    summary (a snapshot of a still-open episode), proposals after the last recorded start are judged as
    if there were no summary."""
    started, used, pos, boundary = [], set(), 0, 0
    for rec in records:
        st = rec.get("subtask") or {}
        fallback = str(st.get("source", "")).startswith("loop.fallback")
        step, extra = None, {}
        if not fallback:
            for i in range(pos, len(proposals)):
                p = proposals[i]
                if (p["out"].get("skill"), p["out"].get("instruction")) == (st.get("skill"), st.get("instruction")):
                    step, pos = p["step"], i + 1
                    used.add(i)
                    extra = {"evidence": p["out"].get("evidence"), "regression_risk": p.get("risk")}
                    break
        if step is None:
            hit = next((m["step"] for m in instructions if m["step"] is not None and m["step"] >= boundary
                        and m.get("instruction") == st.get("instruction")), None)
            step = hit if hit is not None else boundary
            if not fallback:
                extra["inferred"] = "no matching proposal; start inferred from the executor instruction"
        kind = "fallback" if fallback else "subtask_start"
        started.append(_moment(step, kind, _subtask_text(st), skill=st.get("skill"), instruction=st.get("instruction"),
                               source=st.get("source"), **extra))
        if rec.get("end_step") is not None:
            boundary = max(boundary, int(rec["end_step"]))
    last_start = max((m["step"] for m in started if m["step"] is not None), default=-1)
    unknown = set() if complete else {i for i, p in enumerate(proposals)
                                      if i not in used and p["step"] is not None and p["step"] > last_start}
    rejected = [_moment(p["step"], "plan_rejected", _subtask_text(p["out"]), skill=p["out"].get("skill"),
                        instruction=p["out"].get("instruction"), regression_risk=p.get("risk"))
                for i, p in enumerate(proposals) if i not in used and i not in unknown]
    more_started, more_rejected = _starts_without_summary([proposals[i] for i in sorted(unknown)],
                                                          "newer than the summary snapshot")
    return started + more_started, rejected + more_rejected


def _starts_without_summary(proposals: list[dict], why: str = "no episode summary") -> tuple[list[dict], list[dict]]:
    """No summary (or proposals newer than a snapshot): the last proposal of a step is taken as started."""
    last = {p["step"]: i for i, p in enumerate(proposals)}
    started, rejected = [], []
    for i, p in enumerate(proposals):
        st = p["out"]
        if last[p["step"]] == i:
            started.append(_moment(p["step"], "subtask_start", _subtask_text(st), skill=st.get("skill"),
                                   instruction=st.get("instruction"), source=st.get("source"),
                                   evidence=st.get("evidence"), regression_risk=p.get("risk"),
                                   inferred=f"{why}; last proposal of the step taken as started"))
        else:
            rejected.append(_moment(p["step"], "plan_rejected", _subtask_text(st), skill=st.get("skill"),
                                    instruction=st.get("instruction"), regression_risk=p.get("risk")))
    return started, rejected


def build_timeline(events: Iterable[dict], video_fps: float = DEFAULT_VIDEO_FPS,
                   summary: Optional[dict] = None) -> list[dict]:
    """Decision timeline of one episode from its events.jsonl records.

    Subtask starts, ends and loop fallbacks come from the episode's subtask records: the ``episode_end``
    event, else ``summary`` (summary.json, e.g. the snapshot written when the evaluator's connection
    closed). Each moment carries the step, the video time, the skill and instruction in force, the
    latest model judgment and the loop's next action."""
    moments: list[dict] = []
    proposals: list[dict] = []
    exec_instruction = None
    end_summary = None
    for ev in events:
        kind, step = ev.get("kind"), ev.get("step")
        if kind == "decision":
            slot, out = ev.get("slot"), ev.get("out")
            if slot in PROPOSING_SLOTS and isinstance(out, dict):
                proposals.append({"step": _int(step), "out": out, "risk": None})
            elif slot == "goal" and isinstance(out, (int, float)) and not isinstance(out, bool):
                if proposals and proposals[-1]["step"] == _int(step) and proposals[-1]["risk"] is None:
                    proposals[-1]["risk"] = float(out)  # GoalManager.regression_risk of that proposal
            else:
                m = _decision_moment(slot, out, step)
                if m is not None:
                    moments.append(m)
        elif kind == "executor":
            instr = ev.get("instruction")
            if instr != exec_instruction:
                moments.append(_moment(step, "instruction", f"executor instruction -> {instr!r}", instruction=instr))
                exec_instruction = instr
        elif kind == "provider_call" and ev.get("error"):
            moments.append(_moment(step, "provider_error", f"{ev.get('slot')}: {ev.get('error')}"))
        elif kind == "episode_end":
            end_summary = ev.get("summary") or {}
    complete = end_summary is not None or not (summary or {}).get("open")
    end_summary = end_summary if end_summary is not None else (summary or None)

    records = (end_summary or {}).get("subtasks")
    if isinstance(records, list):
        started, rejected = _starts(proposals, records, [m for m in moments if m["kind"] == "instruction"], complete)
    else:
        started, rejected = _starts_without_summary(proposals)
    moments += rejected + started
    for rec in records or []:
        st = rec.get("subtask") or {}
        if rec.get("outcome") == "open" or rec.get("end_step") is None:
            continue
        moments.append(_moment(rec.get("end_step"), "subtask_end",
                               f"{st.get('skill')} -> {rec.get('outcome')} (retries {rec.get('retries')})",
                               skill=st.get("skill"), outcome=rec.get("outcome")))

    def order(i):
        m = moments[i]
        return (m["step"] if m["step"] is not None else 1 << 30, _PRIO.get(m["kind"], 9), i)

    moments = [moments[i] for i in sorted(range(len(moments)), key=order)]
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
        if m["kind"] == "subtask_end":
            skill = None
    end = (end_summary or {}).get("steps")
    for i, m in enumerate(moments):
        nxt = next((n for n in moments[i + 1:] if n["kind"] in ACTION_KINDS), None)
        m["next_action"] = _describe(nxt) if nxt else f"episode end (steps {end})"
    return moments


def _describe(m: dict) -> str:
    return f"step {m['step']}: {m['kind']}: {m['summary']}"


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
        local = os.path.join(os.path.expanduser(local_root), row["run_id"], "episodes", row["episode_id"])
        cands.append(os.path.join(local, "events.jsonl"))
    return next((p for p in cands if os.path.isfile(p)), None)


def selection_reason(row: dict, q_below: float = 1.0) -> Optional[str]:
    """Why an aggregate.py episode row needs a failure record, or None when it does not."""
    q = row["q_final"]
    if row["infra_error"]:
        return f"infra error: {row['infra_reason']}"
    if row["missing"]:
        return f"no official result: {row['missing_reason']}"
    if q is not None and q < q_below:
        return f"official Q {q:g} < {q_below:g}"
    if row["budget_ok"] is False:
        return f"official Q {q:g} but over the 2026 budget (from {row['budget_source']}): scores 0 under that rule"
    return None


def skeleton_records(run_dirs: Sequence[str], video_fps: Optional[float] = None, q_below: float = 1.0,
                     local_root: Optional[str] = None) -> list[dict]:
    inputs = InputLog()
    records = []
    run_dirs, _ = expand_run_dirs(run_dirs, inputs)
    for i, run_dir in enumerate(run_dirs):
        run = load_run(run_dir, inputs, order=i)
        summaries = dict(run.summaries)
        fps = float(video_fps or run.status.get("video_fps") or DEFAULT_VIDEO_FPS)
        rows, _ = episode_rows(run)
        for row in rows:
            why = selection_reason(row, q_below)
            if why is None:
                continue
            epath = _events_path(row, local_root)
            summary = summaries.get(row["summary_path"])
            timeline = build_timeline(read_events(epath, inputs), fps, summary) if epath else []
            ident = row["episode_id"] or f"{row['run_id']}:{row['task']}_{row['instance_id']}_{row['rollout_id']}"
            q = row["q_final"]
            rec = {f: None for f in RECORD_FIELDS}
            rec.update(
                schema_version=SCHEMA_VERSION, record_id=f"{ident}#0", run_id=row["run_id"], config=run.config_name,
                condition=run.condition, episode_id=row["episode_id"], task=row["task"],
                instance_id=row["instance_id"], rollout_id=row["rollout_id"], seed=row["seed"], selected_because=why,
                q_final=q, q_budget2026=None if q is None else (0.0 if row["budget_ok"] is False else q),
                budget_ok=row["budget_ok"], success=row["success"],
                official_failure_reason=row["failure_reason"], infra_error=row["infra_error"], steps=row["steps"],
                synthetic=row["synthetic"],
                video_path=row["video_path"],
                video_exists=bool(row["video_path"] and os.path.isfile(row["video_path"])),
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
                  next_action=_describe(nxt) if nxt else "none until episode end")
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
    unlabelled = collections.Counter(r["category"] for r in records if not r.get("labeler"))
    episodes = collections.defaultdict(set)
    per_task = collections.defaultdict(collections.Counter)
    for r in records:
        episodes[r["category"]].add(r["record_id"].rsplit("#", 1)[0])
        per_task[r["task"]][r["category"]] += 1
    return {
        "n_records": total,
        "n_episodes": len({r["record_id"].rsplit("#", 1)[0] for r in records}),
        "n_unlabelled": sum(unlabelled.values()),
        "synthetic_inputs": any(r.get("synthetic") for r in records),
        "categories": [{"category": c, "label_zh": CATEGORY_LABELS_ZH[c], "count": counts[c],
                        "share": counts[c] / total if total else None, "episodes": len(episodes[c]),
                        "unlabelled": unlabelled[c]} for c in CATEGORIES],
        "per_task": {t: {c: per_task[t][c] for c in CATEGORIES} for t in sorted(per_task)},
        "per_task_share": {t: {c: per_task[t][c] / sum(per_task[t].values()) for c in CATEGORIES}
                           for t in sorted(per_task)},
    }


def render_summary(s: dict) -> str:
    out = ["# Failure distribution (task spec 6.7)", "",
           f"Generated by `eval/failures.py summarize`: {s['n_records']} record(s) from {s['n_episodes']} episode(s); "
           f"{s['n_unlabelled']} unlabelled (category defaults to uncertain until a human labels it).", ""]
    if s["synthetic_inputs"]:
        out += ["**WARNING: SYNTHETIC inputs present. This distribution is not a result.**", ""]
    out.append(md_table(["category", "类别", "count", "share", "episodes", "of which unlabelled"],
                        [[c["category"], c["label_zh"], c["count"], c["share"], c["episodes"], c["unlabelled"]]
                         for c in s["categories"]]))
    out += ["", "## Per task (count, share within the task)", ""]

    def cell(t, c):
        n = s["per_task"][t][c]
        return f"{n} ({100 * s['per_task_share'][t][c]:.0f}%)" if n else "0"

    out.append(md_table(["task"] + [f"{c} {CATEGORY_LABELS_ZH[c]}" for c in CATEGORIES] + ["total"],
                        [[t] + [cell(t, c) for c in CATEGORIES] + [sum(s["per_task"][t].values())]
                         for t in s["per_task"]]))
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
