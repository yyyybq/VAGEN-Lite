"""Keep up to MAX_ACTIVE one-card QA jobs running from a list of (model, benchmark) pairs (login node, background).

    nohup python -m eval_external.sco.queue --stage step2 base-7B:MMSI SI-7B:VSI-Bench ... > queue.log &

Every minute: a finished job (SUCCEEDED) is collected (qa_transfer/collect.py -> results.json) and deleted; a
failed one is left as it is and marked failed (never resubmitted automatically); a pending pair is submitted
when fewer than MAX_ACTIVE of our jobs are unfinished and its benchmark has been prepared
(ROOT/data_prep/<bench>.json).  Run dirs are ROOT/runs/qa/<stage>/<model>__<bench>.  The state of every pair is
written to ROOT/runs/qa/<stage>/queue_status.json.  Exits when no pair is pending or running.
"""

from __future__ import annotations

import argparse
import datetime
import json
import time
import traceback
from pathlib import Path

from eval_external.qa_transfer.benchmarks import BENCHMARKS
from eval_external.sco import job as J

QA_RUNS = J.ROOT / "runs" / "qa"


def run_dir(stage: str, model: str, bench: str) -> Path:
    return QA_RUNS / stage / f"{model}__{bench.replace('@', '_')}"


def prepared(bench: str) -> bool:
    return (J.ROOT / "data_prep" / f"{bench.replace('@', '_at_')}.json").exists()


def step(stage: str, pairs: list[tuple[str, str]]) -> dict:
    status = {}
    n_active = len(J.active())
    for model, bench in pairs:
        rd = run_dir(stage, model, bench)
        key = f"{model}:{bench}"
        receipt = rd / "job" / "submission.json"
        if (rd / "results.json").exists():
            status[key] = "done"
            continue
        if receipt.exists():
            st = J.status(rd)
            if st["state"] == "SUCCEEDED" and st["exit_code"] == "0":
                from eval_external.qa_transfer.collect import collect
                try:
                    collect(rd)
                    J.delete(rd)
                    status[key] = "done"
                except Exception:
                    status[key] = "collect_failed: " + traceback.format_exc(limit=1).strip().splitlines()[-1]
            elif st["state"] in J.FINISHED:
                status[key] = f"failed ({st['state']}, exit {st['exit_code']})"
            else:
                status[key] = f"running ({st['state']})"
            continue
        if n_active >= J.MAX_ACTIVE:
            status[key] = "pending"
            continue
        if not prepared(bench):
            status[key] = "pending (benchmark not prepared)"
            continue
        J.submit(rd, f"{stage}-{bench}-{model}", ["python", "-m", "eval_external.qa_transfer.run_qa", "--model", model,
                                               "--bench", bench, "--run-dir", str(rd)], {})
        n_active += 1
        status[key] = "submitted"
    return status


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--interval", type=int, default=60)
    ap.add_argument("pairs", nargs="+", help="MODEL:BENCH")
    args = ap.parse_args()
    pairs = []
    for p in args.pairs:
        model, bench = p.split(":", 1)
        if bench not in BENCHMARKS:
            raise SystemExit(f"unknown benchmark {bench}")
        pairs.append((model, bench))
    out = QA_RUNS / args.stage / "queue_status.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            status = step(args.stage, pairs)
        except Exception:
            status = {"error": traceback.format_exc()}
        out.write_text(json.dumps({"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                   "pairs": status}, indent=1) + "\n")
        live = [v for v in status.values() if v.startswith(("pending", "running", "submitted"))]
        if "error" not in status and not live:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
