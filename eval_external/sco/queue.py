"""Keep up to MAX_ACTIVE one-card jobs running from a list of MODEL:KIND items (login node, background).

    nohup python -m eval_external.sco.queue --stage step2 ours-SFT:OwnEnv base-7B:WalkerBench SI-7B:VSI-Bench ... &

KIND is a QA benchmark of qa_transfer/benchmarks.py, or
  WalkerBench, WalkerBench-blind          closed_loop/walkerbench/run.py (blind: no image, the reference agent's way)
  OwnEnv, OwnEnv-shuffle, OwnEnv-drop      diagnostics/ownenv_job.sh (our env; dialogue history as is / shuffled / dropped)

Every minute: a finished job (SUCCEEDED, exit 0) is collected into RUN/results.json and deleted; a failed one is
left as it is and marked failed (never resubmitted automatically); a pending item is submitted when fewer than
MAX_ACTIVE of our jobs are unfinished and its inputs are prepared.  Items are submitted in list order.  The state
of every item is written to ROOT/runs/queue_<stage>.json.  Exits when nothing is pending or running.
"""

from __future__ import annotations

import argparse
import datetime
import json
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from eval_external.qa_transfer.benchmarks import BENCHMARKS
from eval_external.sco import job as J

RUNS = J.ROOT / "runs"
QA_RUNS = RUNS / "qa"
OWNENV_MODES = {"OwnEnv": "normal", "OwnEnv-shuffle": "shuffle", "OwnEnv-drop": "drop"}


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


def item(stage: str, model: str, kind: str) -> Item:
    if kind in BENCHMARKS:
        rd = run_dir(stage, model, kind)
        return Item(rd, f"{stage}-{kind}-{model}",
                    ["python", "-m", "eval_external.qa_transfer.run_qa", "--model", model, "--bench", kind,
                     "--run-dir", str(rd)],
                    lambda: (J.ROOT / "data_prep" / f"{kind.replace('@', '_at_')}.json").exists(), _qa_collect)
    if kind in ("WalkerBench", "WalkerBench-blind"):
        blind = kind.endswith("blind")
        rd = RUNS / "walkerbench" / stage / (f"{model}__blind" if blind else model)
        return Item(rd, f"{stage}-wb{'-blind' if blind else ''}-{model}",
                    ["python", "-m", "eval_external.closed_loop.walkerbench.run", "--model", model, "--run-dir",
                     str(rd)] + (["--blind"] if blind else []),
                    lambda: (J.ROOT / "data_prep" / "walkerbench.json").exists(),
                    lambda d: _require(d / "results.json"))
    if kind in OWNENV_MODES:
        mode = OWNENV_MODES[kind]
        name = model if mode == "normal" else f"{model}__{mode}"
        rd = RUNS / "ownenv" / "eval_test" / name
        return Item(rd, f"{stage}-own-{mode}-{model}",
                    ["bash", "eval_external/diagnostics/ownenv_job.sh", name, model, mode],
                    lambda: True, _ownenv_collect, also_own=(str(RUNS / "ownenv"),))
    raise SystemExit(f"unknown kind {kind}")


def step(stage: str, items: list[tuple[str, Item]]) -> dict:
    status = {}
    n_active = len(J.active())
    for key, it in items:
        receipt = it.run_dir / "job" / "submission.json"
        if receipt.exists():
            st = J.status(it.run_dir)
            if st["state"] == "SUCCEEDED" and st["exit_code"] == "0":
                try:
                    it.collect(it.run_dir)
                    J.delete(it.run_dir)
                    status[key] = "done"
                except Exception:
                    status[key] = "collect_failed: " + traceback.format_exc(limit=1).strip().splitlines()[-1]
            elif st["state"] in J.FINISHED:
                status[key] = f"failed ({st['state']}, exit {st['exit_code']})"
            else:
                status[key] = f"running ({st['state']})"
            continue
        if (it.run_dir / "results.json").exists():
            status[key] = "done"
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
        out.write_text(json.dumps({"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                   "items": status}, indent=1) + "\n")
        live = [v for v in status.values() if v.startswith(("pending", "running", "submitted"))]
        if "error" not in status and not live:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
