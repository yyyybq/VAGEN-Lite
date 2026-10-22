"""Export strong-model calls from run logs as DistillRecords (task spec 7.3, M-distill data).

    python -m data.distill export RUN_DIR [RUN_DIR ...] --out FILE.jsonl [--round K] [--skip-errors] [--force]

Input: <run>/episodes/<episode_id>/events.jsonl as written by agent/recorder.py. RUN_DIR is the local
run directory (<run.local_log_root>/<run_id>, where the frames referenced by the calls live) or the
NAS copy (<run.log_root>/<run_id>, events.jsonl only). Per episode:

  episode_start  ctx: task name, task prompt, BDDL subgoal ids -> literals (legal task definition)
  provider_call  one record per call (providers/base.py LoggedProvider._log): slot, step, purpose,
                 user text, system-prompt hash, image refs, raw reply, parsed JSON, usage, latency, cost
  decision       the module's return value for (slot, step), attached to the calls of that slot and
                 step as ``outputs.module_return`` (e.g. a "done" the checker downgraded to "unknown")
  plan / explore decisions also set ``inputs.subtask_in_force`` for later calls: the last proposal
                 of a planning round, i.e. the subtask that started unless the goal manager vetoed it
                 (a veto followed by the loop's full-task fallback is not visible here; eval/failures.py
                 reconstructs exact starts from the episode summary when that matters)

Outputs are normalised per decision type: plan -> subtask_choice (skill, args, target subgoals,
task_complete); check -> continue_switch (done = switch, in_progress = continue, unknown) or
failure_verdict (failed = recover, with failure_type); recover -> recovery_choice. Calls that ended
in a provider error are kept (``outputs.error``) unless --skip-errors. Truth fields stay empty: only
diagnostics/label_distill.py fills them, offline. An episode whose events carry "_synthetic" yields
records with ``synthetic=True``.

Status: written and unit-tested on a SYNTHETIC events.jsonl and on the recorder output of a mock
episode (tests/test_data_diag_distill.py); no real run has been exported yet.
"""

from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import os
import sys
from typing import Any, Optional, Sequence

from data.schema import (
    DistillInputs,
    DistillOutputs,
    DistillRecord,
    DistillSource,
    check,
    write_jsonl,
)

PROPOSING_SLOTS = ("plan", "explore")
_CHECK_SWITCH = {"done": "switch", "in_progress": "continue", "failed": "recover", "unknown": "unknown"}


def _float(x: Any) -> Optional[float]:
    try:
        return None if x is None else float(x)
    except (TypeError, ValueError):
        return None


def _int(x: Any, default: int = -1) -> int:
    try:
        return int(x)
    except (TypeError, ValueError):
        return default


def decision_type(slot: str, purpose: str, parsed: Optional[dict]) -> str:
    kind = slot or purpose.split(".", 1)[0]
    if kind in PROPOSING_SLOTS:
        return "subtask_choice"
    if kind == "check":
        return "failure_verdict" if (parsed or {}).get("status") == "failed" else "continue_switch"
    if kind == "recover":
        return "recovery_choice"
    return "other"


def normalise_outputs(slot: str, purpose: str, rec: dict, module_return: Any = None) -> DistillOutputs:
    parsed = rec.get("parsed") if isinstance(rec.get("parsed"), dict) else None
    dtype = decision_type(slot, purpose, parsed)
    p = parsed or {}
    kw: dict[str, Any] = {}
    if dtype == "subtask_choice" and parsed is not None:
        args = {}
        for item in p.get("args") or []:
            if isinstance(item, dict) and "name" in item:
                args[str(item["name"])] = str(item.get("value", ""))
        kw["subtask"] = {"skill": p.get("skill"), "args": args,
                         "target_subgoals": [str(t) for t in p.get("target_subgoals") or []],
                         "max_steps": p.get("max_steps")}
        tc = p.get("task_complete")
        kw["task_complete"] = tc if isinstance(tc, bool) else None
    elif dtype in ("continue_switch", "failure_verdict") and parsed is not None:
        status = p.get("status") if p.get("status") in _CHECK_SWITCH else None
        kw["status"] = status
        kw["continue_switch"] = _CHECK_SWITCH.get(status) if status else None
        ft = p.get("failure_type")
        kw["failure_type"] = ft if (isinstance(ft, str) and ft != "none") else None
        kw["subgoal_estimates"] = {str(it["id"]): str(it["estimate"]) for it in p.get("subgoals") or []
                                   if isinstance(it, dict) and "id" in it and "estimate" in it}
        kw["holding"] = {arm: str(p[f"holding_{arm}"]) for arm in ("left", "right") if isinstance(p.get(f"holding_{arm}"), str)}
    elif dtype == "recovery_choice" and parsed is not None:
        kw["recovery"] = {k: p.get(k) for k in ("kind", "primitive", "instruction", "reason") if k in p}
    conf = _float(p.get("confidence"))
    if conf is not None:
        kw["confidence"] = min(max(conf, 0.0), 1.0)
    return DistillOutputs(decision_type=dtype, module_return=module_return, parsed=parsed,
                          response_text=str(rec.get("response_text") or ""), error=rec.get("error"), **kw)


def _goal_literals(ctx: dict) -> dict[str, str]:
    gs = ctx.get("goal_spec") if isinstance(ctx, dict) else None
    out = {}
    for sg in (gs or {}).get("subgoals") or []:
        if isinstance(sg, dict) and sg.get("subgoal_id"):
            out[str(sg["subgoal_id"])] = str(sg.get("literal", ""))
    return out


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def export_episode(events_path: str, run_id: str, run_dir: str, *, round_: int = 0,
                   skip_errors: bool = False) -> tuple[list[DistillRecord], list[str]]:
    """DistillRecords for every provider_call in one episode's events.jsonl, plus warnings."""
    warnings: list[str] = []
    rel = os.path.relpath(os.path.abspath(events_path), os.path.abspath(run_dir))
    sha = _sha256(events_path)
    ctx: dict = {}
    episode_id = os.path.basename(os.path.dirname(os.path.abspath(events_path)))
    synthetic = False
    subtask: Optional[dict] = None
    calls: list[dict] = []  # {line, rec, subtask, return}
    pending: dict[tuple[str, int], list[dict]] = collections.defaultdict(list)  # calls awaiting their decision
    with open(events_path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    for i, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            warnings.append(f"{rel}:{i + 1}: unparseable line skipped")
            continue
        synthetic = synthetic or bool(ev.get("_synthetic"))
        kind = ev.get("kind")
        if kind == "episode_start":
            ctx = ev.get("ctx") or {}
            episode_id = str(ctx.get("episode_id") or episode_id)
        elif kind == "provider_call":
            call = {"line": i, "rec": ev, "subtask": subtask, "return": None}
            calls.append(call)
            pending[(str(ev.get("slot") or ""), _int(ev.get("step")))].append(call)
        elif kind == "decision":
            slot, step, out = str(ev.get("slot") or ""), _int(ev.get("step")), ev.get("out")
            for call in pending.pop((slot, step), []):  # calls made inside this module invocation
                call["return"] = out
            if slot in PROPOSING_SLOTS and isinstance(out, dict):
                subtask = {k: out.get(k) for k in ("subtask_id", "skill", "instruction", "args", "target_subgoals",
                                                   "max_steps", "source")}
            elif slot == "plan" and out is None:
                subtask = {"skill": "__full_task__", "instruction": ctx.get("task_instruction"), "target_subgoals": [],
                           "source": "loop.fallback.exhausted (inferred: planner returned no subtask)"}

    literals = _goal_literals(ctx)
    records = []
    for c in calls:
        rec = c["rec"]
        if skip_errors and rec.get("error"):
            continue
        slot, purpose, step = str(rec.get("slot") or ""), str(rec.get("purpose") or ""), _int(rec.get("step"))
        outputs = normalise_outputs(slot, purpose, rec, c["return"])
        parsed = outputs.parsed or {}
        record = DistillRecord(
            record_id=f"{episode_id}#L{c['line']}",
            call_id=str(rec.get("call_id") or ""),
            run_id=run_id,
            episode_id=episode_id,
            step=step,
            slot=slot,
            purpose=purpose,
            provider=str(rec.get("provider") or ""),
            model=str(rec.get("model") or ""),
            inputs=DistillInputs(
                user_text=str(rec.get("user_text") or ""),
                system_sha256=str(rec.get("system_sha256") or ""),
                image_refs=tuple(str(p) for p in rec.get("images") or []),
                task_name=str(ctx.get("task_name") or ""),
                task_instruction=str(ctx.get("task_instruction") or ""),
                subgoal_literals=literals,
                subtask_in_force=c["subtask"],
            ),
            outputs=outputs,
            visible_evidence=str(parsed.get("evidence") or ""),
            source=DistillSource(events_path=rel, events_sha256=sha, line=c["line"]),
            latency_s=_float(rec.get("latency_s")),
            usage=dict(rec.get("usage") or {}),
            cost_usd=_float(rec.get("cost_usd")),
            round=round_,
            synthetic=synthetic,
        )
        records.append(check(record))
    return records, warnings


def export_run(run_dir: str, *, round_: int = 0, skip_errors: bool = False) -> tuple[list[DistillRecord], list[str]]:
    run_dir = os.path.abspath(run_dir)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"run dir {run_dir} does not exist")
    run_id = os.path.basename(run_dir)
    manifest = os.path.join(run_dir, "run_manifest.json")
    if os.path.isfile(manifest):
        with open(manifest, encoding="utf-8") as f:
            run_id = str(json.load(f).get("run_id") or run_id)
    records, warnings = [], []
    paths = sorted(glob.glob(os.path.join(run_dir, "episodes", "*", "events.jsonl")))
    if not paths:
        warnings.append(f"{run_dir}: no episodes/*/events.jsonl")
    for p in paths:
        recs, warn = export_episode(p, run_id, run_dir, round_=round_, skip_errors=skip_errors)
        records += recs
        warnings += warn
    return records, warnings


def describe(records: Sequence[DistillRecord]) -> str:
    by_type = collections.Counter(r.outputs.decision_type for r in records)
    errors = sum(1 for r in records if r.outputs.error)
    synthetic = sum(1 for r in records if r.synthetic)
    parts = ", ".join(f"{k} {v}" for k, v in sorted(by_type.items())) or "none"
    return f"{len(records)} records ({parts}); provider errors {errors}; synthetic {synthetic}"


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("export", help="provider calls of one or more runs -> DistillRecord JSONL")
    ex.add_argument("run_dirs", nargs="+")
    ex.add_argument("--out", required=True)
    ex.add_argument("--round", type=int, default=0, dest="round_")
    ex.add_argument("--skip-errors", action="store_true", help="drop calls that ended in a provider error")
    ex.add_argument("--force", action="store_true", help="overwrite an existing output file")
    args = ap.parse_args(argv)

    if os.path.exists(args.out) and not args.force:
        print(f"{args.out} exists; pass --force to overwrite", file=sys.stderr)
        return 1
    records, warnings = [], []
    for d in args.run_dirs:
        recs, warn = export_run(d, round_=args.round_, skip_errors=args.skip_errors)
        records += recs
        warnings += warn
    for w in warnings:
        print(f"warning: {w}", file=sys.stderr)
    write_jsonl(args.out, records)
    print(f"{args.out}: {describe(records)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
