"""Keep up to MAX_ACTIVE one-card jobs running from a list of MODEL:KIND items (login node, background).

    nohup python -m eval_external.sco.queue --stage step2 ours-SFT:OwnEnv base-7B:WalkerBench SI-7B:VSI-Bench ... &

KIND is a QA benchmark of qa_transfer/benchmarks.py, or
  WalkerBench, WalkerBench-blind          closed_loop/walkerbench/run.py (blind: no image, the reference agent's way)
  WalkerBench-lenient                     the same with --lenient (an action named outside the JSON is accepted)
  ...@extra300                            any of the three on the 300 further tasks (closed_loop/walkerbench/extra_set.py)
  OwnEnv, OwnEnv-shuffle, OwnEnv-drop      diagnostics/ownenv_job.sh (our env; dialogue history as is / shuffled / dropped)
  OwnEnv-replace                          history replaced by another same-scene task's turns (after the normal run)
  OwnEnv-lenient                          the same, history as is, untagged "Answer:" lines accepted

Every minute: a finished job (SUCCEEDED, exit 0) is collected into RUN/results.json and deleted; a failed one is
left as it is and marked failed (never resubmitted automatically); a pending item is submitted when fewer than
MAX_ACTIVE of our jobs are unfinished and its inputs are prepared.  Items are submitted in list order.  The state
of every item is written to ROOT/runs/queue_<stage>.json.  Exits when nothing is pending or running.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from eval_external.qa_transfer.benchmarks import BENCHMARKS
from eval_external.sco import job as J

RUNS = J.ROOT / "runs"
QA_RUNS = RUNS / "qa"
OWNENV_MODES = {"OwnEnv": "normal", "OwnEnv-shuffle": "shuffle", "OwnEnv-drop": "drop", "OwnEnv-replace": "replace",
                "OwnEnv-lenient": "lenient"}  # lenient: history normal, an untagged last line "Answer:" accepted


@dataclass
class Item:
    run_dir: Path
    name: str
    command: list[str]
    prepared: Callable[[], bool]
    collect: Callable[[Path], None]
    also_own: tuple[str, ...] = field(default_factory=tuple)


def run_dir(stage: str, model: str, bench: str) -> Path:  # QA run folders (kept for step-1 naming)
    return QA_RUNS / stage / f"{model}__{bench.replace('@', '_')}"


def _qa_collect(rd: Path) -> None:
    from eval_external.qa_transfer.collect import collect

    collect(rd)


def _ownenv_collect(rd: Path) -> None:
    from eval_external.diagnostics.ownenv_eval import summarize

    out = summarize(rd.name)
    (rd / "results.json").write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n")


def _require(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"the job ended without {path}")


ESI_MODELS = ["ours-SFT", "ours-RL-150", "ours-SFT-r11", "ours-RL-r11", "ours-RL-r11-dense", "base-7B", "SI-7B"]


def _esi_collect(rd: Path) -> None:
    from eval_external.closed_loop.esi_format_check import cmd_summary

    cmd_summary(None)
    _require(rd / "summary.json")
    (rd / "results.json").write_text((rd / "summary.json").read_text())


def item(stage: str, model: str, kind: str) -> Item:
    if kind == "ESIFormatCheck":  # "all:ESIFormatCheck": ESI-Bench first-step format check of ESI_MODELS
        rd = RUNS / "esi_format_check"
        return Item(rd, f"{stage}-esi-format",
                    ["python", "-m", "eval_external.closed_loop.esi_format_check", "run", "--models", *ESI_MODELS],
                    lambda: (rd / "items.json").exists(), _esi_collect, also_own=(str(rd),))
    if kind in BENCHMARKS:
        rd = run_dir(stage, model, kind)
        return Item(rd, f"{stage}-{kind}-{model}",
                    ["python", "-m", "eval_external.qa_transfer.run_qa", "--model", model, "--bench", kind,
                     "--run-dir", str(rd)],
                    lambda: (J.ROOT / "data_prep" / f"{kind.replace('@', '_at_')}.json").exists(), _qa_collect)
    base, _, task_set = kind.partition("@")  # "WalkerBench-lenient@extra300": the extra300 task set
    if base in ("WalkerBench", "WalkerBench-blind", "WalkerBench-lenient"):
        variant = base.partition("-")[2]  # "", "blind" or "lenient"
        suffix = "__".join(x for x in (task_set, variant) if x)
        rd = RUNS / "walkerbench" / stage / (f"{model}__{suffix}" if suffix else model)
        ready = J.ROOT / "data_prep" / ("walkerbench_extra300/selection.json" if task_set else "walkerbench.json")
        return Item(rd, f"{stage}-wb{'-' + suffix if suffix else ''}-{model}",
                    ["python", "-m", "eval_external.closed_loop.walkerbench.run", "--model", model, "--run-dir",
                     str(rd)] + ([f"--{variant}"] if variant else []) + (["--task-set", task_set] if task_set else []),
                    lambda: (J.ROOT / "data_prep" / "walkerbench.json").exists() and ready.exists(),
                    lambda d: _require(d / "results.json"))
    if kind in OWNENV_MODES:
        mode = OWNENV_MODES[kind]
        name = model if mode == "normal" else f"{model}__{mode}"
        rd = RUNS / "ownenv" / "eval_test" / name
        command = ["bash", "eval_external/diagnostics/ownenv_job.sh", name, model]
        command += ["normal", "--lenient"] if mode == "lenient" else [mode]
        donor = RUNS / "ownenv" / "eval_test" / model / "results.json"  # replace takes turns from the normal run
        return Item(rd, f"{stage}-own-{mode}-{model}", command,
                    (lambda: donor.exists()) if mode == "replace" else (lambda: True), _ownenv_collect,
                    also_own=(str(RUNS / "ownenv"),))
    raise SystemExit(f"unknown kind {kind}")


def done_label(rd: Path) -> str:
    """'done', or 'done_flagged: ...' when collect's checks flagged the run (QA: run time far below base-7B's)."""
    try:
        res = json.loads((rd / "results.json").read_text())
    except (OSError, ValueError):
        return "done"
    chk = (res.get("checks") if isinstance(res, dict) else None) or {}  # ESI format check: a list of rows
    return "done_flagged: short run" if chk.get("short_run") else "done"


MOVE_AFTER_S = 1800  # 2026-10-20: a job still waiting for a card after this long may change pool ...
MAX_MOVES = 3        # ... to the other pool (h800 <-> zoetrope) when fewer jobs wait there, at most this often


def pool_waiting() -> dict[str, int]:
    """Jobs of the whole workspace waiting for a card, per pool (the latest 300 jobs of `sco acp jobs list`)."""
    import subprocess

    res = subprocess.run([J.SCO, "acp", "jobs", "list", "--workspace-name=aigc", "-o", "json", "--page-size", "300"],
                         capture_output=True, text=True)
    try:
        d = json.loads(res.stdout)
    except ValueError:
        return {}
    out = {p: 0 for p in J.POOLS}
    for j in (d.get("jobs") if isinstance(d, dict) else d) or []:
        pool = (j.get("resource_pool") or {}).get("name")
        if pool in out and j.get("state") in J.WAITING:
            out[pool] += 1
    return out


def move_target(receipt: Path, state: str, waiting: dict[str, int], now: float | None = None) -> str | None:
    """The pool to move a waiting job to, or None: it waited over MOVE_AFTER_S since its last submission, the
    other pool has fewer jobs waiting, and it has moved fewer than MAX_MOVES times."""
    rec = json.loads(receipt.read_text())
    pool = rec.get("pool", J.POOL)
    other = next(p for p in J.POOLS if p != pool)
    started = datetime.datetime.fromisoformat(rec["utc"]).timestamp()
    now = time.time() if now is None else now
    moves = len(list(receipt.parent.glob("submission_*_deleted.json")))
    if (state in J.WAITING and now - started > MOVE_AFTER_S and moves < MAX_MOVES
            and pool in waiting and other in waiting and waiting[other] < waiting[pool]):
        return other
    return None


def step(stage: str, items: list[tuple[str, Item]]) -> dict:
    status = {}
    n_active = len(J.active())
    waiting = None
    for key, it in items:
        receipt = it.run_dir / "job" / "submission.json"
        if receipt.exists():
            st = J.status(it.run_dir)
            target = None
            if st["state"] in J.WAITING:
                waiting = pool_waiting() if waiting is None else waiting
                target = move_target(receipt, st["state"], waiting)
            if target:
                old = J.delete(it.run_dir)
                if old["delete_rc"] == 0:
                    J.submit(it.run_dir, it.name, it.command, {}, also_own=it.also_own, pool=target)
                    status[key] = f"moved to {target} (was waiting as {old['job_id']}; waiting jobs {waiting})"
                    continue
            if st["state"] == "SUCCEEDED" and st["exit_code"] == "0":
                try:
                    it.collect(it.run_dir)
                    J.delete(it.run_dir)
                    status[key] = done_label(it.run_dir)
                except Exception:
                    status[key] = "collect_failed: " + traceback.format_exc(limit=1).strip().splitlines()[-1]
            elif st["state"] in J.FINISHED:
                status[key] = f"failed ({st['state']}, exit {st['exit_code']})"
            else:
                status[key] = f"running ({st['state']})"
            continue
        if (it.run_dir / "results.json").exists():
            status[key] = done_label(it.run_dir)
            continue
        if n_active >= J.MAX_ACTIVE:
            status[key] = "pending"
            continue
        if not it.prepared():
            status[key] = "pending (inputs not prepared)"
            continue
        J.submit(it.run_dir, it.name, it.command, {}, also_own=it.also_own)
        n_active += 1
        status[key] = "submitted"
    return status


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("items", nargs="+", help="MODEL:KIND")
    args = ap.parse_args()
    items = []
    for p in args.items:
        model, kind = p.split(":", 1)
        items.append((p, item(args.stage, model, kind)))
    out = RUNS / f"queue_{args.stage}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            status = step(args.stage, items)
        except Exception:
            status = {"error": traceback.format_exc()}
            print(status["error"], file=sys.stderr, flush=True)  # the queue log
        # written whole and renamed: a queue killed mid-write (2026-10-20 21:03) left an empty file behind
        tmp = out.with_suffix(".json.tmp")
        tmp.write_text(json.dumps({"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                   "items": status}, indent=1) + "\n")
        os.replace(tmp, out)
        live = [v for v in status.values() if v.startswith(("pending", "running", "submitted", "moved"))]
        if "error" not in status and not live:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
