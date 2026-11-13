"""Fill the verification fields of DistillRecords from replay truth (task spec 7.3: "事后用真值核验").

    python -m diagnostics.label_distill DISTILL.jsonl --truth TRUTH.jsonl [--truth ...] --out LABELLED.jsonl
                                        [--max-lag 0] [--allow-divergent]

Offline only. Input: data/distill.py export (no truth fields) and diagnostics/replay_truth.py output.
Output: the same records with truth_label, truth_source, correct and truth_detail set; the records'
inputs and outputs are unchanged, and the labels never reach a running agent.

What is verified (a "done" claim against the truth of its target subgoals):
  checker verdicts (continue_switch / failure_verdict records): the targets are the target subgoals
      of the subtask in force (inputs.subtask_in_force), mapped through inputs.subgoal_literals to
      canonical goal literals; a subtask without targets (e.g. the full task) is judged against the
      whole goal (task success). truth_label = "done" when every target literal holds at the call's
      step, "not_done" otherwise, "unknown" when a target cannot be mapped (quantified subgoal) or no
      truth record is close enough. A done verdict is correct iff truth says done; in_progress and
      failed verdicts claim "not done" and are correct iff truth says not_done; unknown verdicts get
      correct = null. truth_detail adds the per-subgoal estimates against truth.
  planner records with task_complete set: judged against the whole goal the same way.
  other records: left unlabelled.

Truth at the call's step: the truth record with the same step (replay_truth.py records every decision
and model-call step), or the latest earlier one within --max-lag steps. Episodes whose replay did not
reproduce the official outcome (footer consistent = false) are skipped unless --allow-divergent; a
truth track without a footer (an interrupted replay) is always skipped.

Status: written and unit-tested on SYNTHETIC records and truth (tests/test_data_diag_label.py); no real
labels exist yet.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import sys
from typing import Optional, Sequence

from data.schema import DistillRecord, read_jsonl, write_jsonl
from diagnostics.truth import TruthTrack, canonical_from_text, load_truth

CHECK_TYPES = ("continue_switch", "failure_verdict")
_CLAIMS_DONE = {"done": True, "in_progress": False, "failed": False}


def _targets(rec: DistillRecord) -> tuple[list[str], bool]:
    """(target subgoal ids, whole_goal); whole_goal when the subtask names no targets."""
    sub = rec.inputs.subtask_in_force or {}
    targets = [str(t) for t in sub.get("target_subgoals") or []]
    return targets, not targets


def label_record(rec: DistillRecord, track: Optional[TruthTrack], *, max_lag: int = 0,
                 allow_divergent: bool = False) -> DistillRecord:
    o = rec.outputs
    if o.error or o.parsed is None:
        return rec
    if o.decision_type in CHECK_TYPES:
        claim = _CLAIMS_DONE.get(str(o.status))  # None: abstained (unknown)
        targets, whole = _targets(rec)
    elif o.decision_type == "subtask_choice" and o.task_complete is not None:
        claim, targets, whole = bool(o.task_complete), [], True
    else:
        return rec
    if track is None:
        return rec
    if track.footer is None:
        return dataclasses.replace(rec, truth_detail={"skipped": "truth track has no footer (replay incomplete)",
                                                      "truth_source": track.source})
    if track.consistent is False and not allow_divergent:
        return dataclasses.replace(rec, truth_detail={"skipped": "replay did not reproduce the official outcome",
                                                      "truth_source": track.source})
    state = track.at(rec.step, max_lag)
    source = f"replay_truth:{track.source}"
    if state is None:
        return dataclasses.replace(rec, truth_label="unknown", truth_source=source, correct=None,
                                   truth_detail={"reason": f"no truth record within {max_lag} steps "
                                                           f"before step {rec.step}"})
    truth = track.literal_truth(state)
    detail: dict = {"truth_step": state["step"], "lag": rec.step - state["step"], "success": state.get("success"),
                    "q_now": state.get("q_now"), "whole_goal": whole, "targets": {}}
    if whole:
        done: Optional[bool] = bool(state.get("success"))
    else:
        values = []
        for g in targets:
            lit = canonical_from_text(rec.inputs.subgoal_literals.get(g, ""))
            v = truth.get(lit) if lit else None
            detail["targets"][g] = {"literal": lit, "truth": v}
            values.append(v)
        done = None if any(v is None for v in values) else all(values)
    estimates = {}
    for g, est in (o.subgoal_estimates or {}).items():
        lit = canonical_from_text(rec.inputs.subgoal_literals.get(g, ""))
        v = truth.get(lit) if lit else None
        agree = None if v is None or est not in ("satisfied", "unsatisfied") else ((est == "satisfied") == v)
        estimates[g] = {"estimate": est, "truth": v, "agree": agree}
    detail["subgoal_estimates"] = estimates
    label = "unknown" if done is None else ("done" if done else "not_done")
    correct = None if (claim is None or done is None) else (claim == done)
    return dataclasses.replace(rec, truth_label=label, truth_source=f"{source} step {state['step']}", correct=correct,
                               truth_detail=detail)


def label_records(records: Sequence[DistillRecord], tracks: dict, *, max_lag: int = 0,
                  allow_divergent: bool = False) -> list[DistillRecord]:
    return [label_record(r, tracks.get(r.episode_id), max_lag=max_lag, allow_divergent=allow_divergent)
            for r in records]


def summarize(records: Sequence[DistillRecord]) -> str:
    rows = collections.defaultdict(lambda: collections.Counter())
    for r in records:
        key = f"{r.outputs.decision_type}:{r.outputs.status or ('task_complete' if r.outputs.task_complete else '-')}"
        c = rows[key]
        c["n"] += 1
        c["labelled"] += int(r.truth_label is not None)
        c["correct"] += int(r.correct is True)
        c["wrong"] += int(r.correct is False)
    lines = [f"{k}: {c['n']} records, {c['labelled']} labelled, correct {c['correct']}, wrong {c['wrong']}"
             for k, c in sorted(rows.items())]
    return "\n".join(lines) or "no records"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("distill")
    ap.add_argument("--truth", action="append", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-lag", type=int, default=0, help="accept a truth record up to N steps before the call")
    ap.add_argument("--allow-divergent", action="store_true", help="also label episodes whose replay diverged")
    args = ap.parse_args(argv)

    records = read_jsonl(args.distill, DistillRecord)
    tracks = load_truth(args.truth)
    missing = sorted({r.episode_id for r in records} - set(tracks))
    if missing:
        print(f"warning: no truth for {len(missing)} episode(s): {', '.join(missing[:5])}", file=sys.stderr)
    labelled = label_records(records, tracks, max_lag=args.max_lag, allow_divergent=args.allow_divergent)
    write_jsonl(args.out, labelled)
    print(summarize(labelled), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
