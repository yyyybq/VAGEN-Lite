"""Per-episode diagnostic records: official results + agent summaries + agent events (+ replay truth).

    python -m diagnostics.join RUN_DIR [RUN_DIR ...] --out diag.jsonl [--md diag.md]
                               [--local-root ~/b1k_local_logs] [--truth TRUTH.jsonl ...]

Pure Python, offline, after the episodes; no simulator. The official result rows and their matching
to agent episodes come from eval/aggregate.py (load_run / episode_rows: the same join the result
tables use), the decision timeline from eval/failures.py build_timeline. Per episode it adds:

  checker       model verdicts (check decisions not from the null checker): counts, the last one,
                the last decisive one (done / failed), and ``checker_consistent``: whether that last
                decisive verdict agrees with the official outcome (done <-> success). Coarse on
                purpose: a subtask-level verdict vs the episode outcome; None without a decisive
                verdict or an official outcome (no result, an infra error, or a 2026/eval
                failure_reason, whose success / q_score are forced to false / 0);
  planner       steps where the planner claimed the task complete (returned no subtask). The official
                evaluator ends an episode the moment the goal holds, so the agent never observes a
                completed task: every such claim was premature when it was made;
  fallbacks     loop fallbacks (summary counter) and ``fallbacks_before_success`` (= fallbacks of a
                successful episode, None otherwise), recoveries, replans, provider calls / errors;
  truth         with --truth (diagnostics/replay_truth.py output): replay consistency, literals ever
                satisfied, predicate regressions (true at one sample, false at a later one), the
                maximum running Q.

Only decisions made before the official end count (step < the official ``steps``): with num_envs > 1
the evaluator keeps sending a finished env's frozen last observation until the batch ends, and the
agent's later decisions never acted on the scored rollout (``decisions_after_official_end`` counts
them; the summary counters such as ``fallbacks`` cannot be split and include them).

Field categories of the output: official columns are EVALUATION_RECORD, truth columns
TRAINING_ANNOTATION, agent-log columns are the agent's own (legal) records. Nothing here is fed back
into a running agent.

Status: written and unit-tested on SYNTHETIC run directories (tests/test_data_diag_join.py); not run on
real outputs yet.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from typing import Iterable, Optional, Sequence

from diagnostics.truth import TruthTrack, load_truth
from eval.aggregate import InputLog, episode_rows, expand_run_dirs, load_run, md_table
from eval.failures import build_timeline, read_events

OFFICIAL_FIELDS = ("run_id", "config", "task", "instance_id", "rollout_id", "seed", "q_final", "success", "steps",
                   "failure_reason", "infra_error", "missing", "budget_ok", "synthetic", "json_path", "video_path",
                   "episode_id", "join", "summary_path")
SUMMARY_FIELDS = ("recoveries", "replans", "fallbacks", "provider_calls", "provider_errors", "cost_usd",
                  "plan_exhausted")


def find_events(row: dict, local_root: Optional[str]) -> Optional[str]:
    cands = []
    if local_root and row.get("episode_id"):
        cands.append(os.path.join(os.path.expanduser(local_root), str(row["run_id"]), "episodes",
                                  str(row["episode_id"]), "events.jsonl"))
    if row.get("summary_path"):
        cands.append(os.path.join(os.path.dirname(row["summary_path"]), "events.jsonl"))
    return next((p for p in cands if os.path.isfile(p)), None)


def checker_stats(events: Iterable[dict], rollout_steps: Optional[int] = None) -> dict:
    verdicts = []
    claims = []
    after_end = 0
    for ev in events:
        if ev.get("kind") != "decision":
            continue
        if rollout_steps is not None and isinstance(ev.get("step"), int) and ev["step"] >= rollout_steps:
            after_end += 1
            continue
        slot, out = ev.get("slot"), ev.get("out")
        if slot == "check" and isinstance(out, dict) and out.get("source") != "check.null":
            verdicts.append({"step": ev.get("step"), "status": out.get("status"), "confidence": out.get("confidence"),
                             "failure_type": out.get("failure_type"), "source": out.get("source")})
        elif slot == "plan" and out is None:
            claims.append(ev.get("step"))
    decisive = [v for v in verdicts if v["status"] in ("done", "failed")]
    counts = collections.Counter(v["status"] for v in verdicts)
    return {"n_verdicts": len(verdicts), "verdict_counts": dict(counts),
            "last_verdict": verdicts[-1] if verdicts else None,
            "last_decisive_verdict": decisive[-1] if decisive else None,
            "planner_complete_claims": claims, "decisions_after_official_end": after_end}


def truth_stats(track: Optional[TruthTrack], official_q: Optional[float]) -> dict:
    if track is None:
        return {"truth": None}
    regressions = track.regressions()
    first = track.first_true_steps()
    initial = set(track.header.get("initial_satisfied") or [])
    return {"truth": {
        "source": track.source,
        "consistent": track.consistent,
        "replay_q_final": (track.footer or {}).get("replay_q_final"),
        "official_q_final": official_q,
        "n_literals": len(track.literals),
        "initially_satisfied": len(initial),
        "ever_satisfied": len(first),
        "newly_satisfied_ever": sum(1 for lit in first if track.literals.index(lit) not in initial),
        "max_q_now": track.max_q_now(),
        "n_regressions": len(regressions),
        "regressions": regressions[:20],
        "n_records": len(track.records),
    }}


def episode_record(row: dict, events: Optional[list], summary: Optional[dict], truth: Optional[TruthTrack],
                   events_path: Optional[str], video_fps: float) -> dict:
    rec = {k: row.get(k) for k in OFFICIAL_FIELDS}
    rec.update({k: (summary or {}).get(k) for k in SUMMARY_FIELDS})
    rec["events_path"] = events_path
    steps = row.get("steps") if isinstance(row.get("steps"), int) else None
    stats = checker_stats(events or [], steps)
    rec.update(stats)
    outcome_known = row.get("q_final") is not None and not row.get("infra_error") and not row.get("failure_reason")
    success = row.get("success") if outcome_known else None
    last = stats["last_decisive_verdict"]
    consistent = None if (last is None or success is None) else ((last["status"] == "done") == bool(success))
    rec["checker_consistent"] = consistent
    rec["fallbacks_before_success"] = rec["fallbacks"] if success else None
    timeline = build_timeline(events, video_fps, summary, rollout_steps=steps) if events else []
    rec["timeline_moments"] = dict(collections.Counter(m["kind"] for m in timeline))
    rec.update(truth_stats(truth, row.get("q_final")))
    return rec


def join_runs(run_dirs: Sequence[str], *, local_root: Optional[str] = None, truth_paths: Sequence[str] = (),
              video_fps: Optional[float] = None) -> tuple[list[dict], list[str]]:
    inputs = InputLog()
    tracks = load_truth(truth_paths) if truth_paths else {}
    dirs, warnings = expand_run_dirs(run_dirs, inputs)
    records = []
    for i, run_dir in enumerate(dirs):
        run = load_run(run_dir, inputs, order=i)
        fps = float(video_fps or run.status.get("video_fps") or 30.0)
        summaries = dict(run.summaries)
        rows, warn = episode_rows(run)
        warnings += warn
        for row in rows:
            epath = find_events(row, local_root)
            events = read_events(epath, inputs) if epath else None
            if row.get("episode_id") and epath is None:
                warnings.append(f"{row['episode_id']}: events.jsonl not found")
            truth = tracks.get(str(row.get("episode_id"))) if row.get("episode_id") else None
            records.append(episode_record(row, events, summaries.get(row.get("summary_path")), truth, epath, fps))
    used = {r.get("episode_id") for r in records}
    warnings += [f"truth for episode {e} matched no official result" for e in sorted(set(tracks) - used)]
    return records, warnings


def _rate(xs: list) -> str:
    vals = [x for x in xs if x is not None]
    return f"{sum(1 for x in vals if x)}/{len(vals)}" if vals else "-"


def render_markdown(records: Sequence[dict]) -> str:
    by_task: dict[str, list] = collections.defaultdict(list)
    for r in records:
        by_task[str(r.get("task"))].append(r)
    rows = []
    for task, rs in sorted(by_task.items()):
        rows.append([task, len(rs), sum(1 for r in rs if r.get("synthetic")),
                     _rate([r["checker_consistent"] for r in rs]),
                     _rate([None if r["checker_consistent"] is None
                            else (r["last_decisive_verdict"]["status"] == "done" and not r["checker_consistent"])
                            for r in rs]),
                     sum(len(r["planner_complete_claims"]) for r in rs),
                     sum(int(r.get("fallbacks") or 0) for r in rs),
                     _rate([None if r["truth"] is None else r["truth"]["consistent"] for r in rs]),
                     sum(r["truth"]["n_regressions"] for r in rs if r["truth"])])
    head = ["task", "episodes", "synthetic", "checker consistent", "last DONE but failed", "planner 'complete' claims",
            "fallbacks", "replay consistent", "regressions (truth)"]
    note = ("Generated by diagnostics/join.py from official result JSONs, agent summaries / events and replay truth. "
            "Diagnostics only (not a result table); synthetic rows never count as results.")
    return note + "\n\n" + md_table(head, rows) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dirs", nargs="+")
    ap.add_argument("--out", required=True, help="per-episode diagnostic records (JSONL)")
    ap.add_argument("--md", default=None, help="optional per-task markdown summary")
    ap.add_argument("--local-root", default=None, help="the agent's local log root (events.jsonl with every record)")
    ap.add_argument("--truth", action="append", default=[], help="truth JSONL from diagnostics/replay_truth.py")
    ap.add_argument("--video-fps", type=float, default=None)
    args = ap.parse_args(argv)

    records, warnings = join_runs(args.run_dirs, local_root=args.local_root, truth_paths=args.truth,
                                  video_fps=args.video_fps)
    for w in warnings:
        print(f"warning: {w}", file=sys.stderr)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, sort_keys=True, default=str) + "\n")
    if args.md:
        with open(args.md, "w", encoding="utf-8") as f:
            f.write(render_markdown(records))
    print(f"{args.out}: {len(records)} episode records", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
