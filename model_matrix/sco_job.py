"""One sco job of the model line outside the training chain (load tests, the StarVLA backbone runs), named
models-<name>.  By default under the same rule as the chain's plan pool "idle" (the user's rule of 2026-10-24): it
waits until a pool has no main-line job waiting that competes with it, a machine for it (8 cards: an idle machine;
1 card: a machine with a free card) and the pause file is absent; then it is submitted there.  ``--pool queue`` (the
user's "开始排队吧" of 2026-10-25) queues it as soon as the pause file is absent, in the pool with the fewest waiting
jobs (job_sco.pick_pool), as the chain's plan pool "queue".

    python -m model_matrix.sco_job submit --run RUN --cards 1 --name backbone-load-q3b -- bash SCRIPT ARGS...
    python -m model_matrix.sco_job submit --run RUN --cards 8 --name backbone-smoke-q3b --pool queue -- bash ...
    python -m model_matrix.sco_job status --run RUN

RUN/job/launch.sh holds the command; RUN/job/submission.json the receipt (an existing receipt is never submitted again).
"""
from __future__ import annotations

import argparse
import json
import shlex
import time
from pathlib import Path

from data_gen.interiorgs_v10 import job_sco

from model_matrix import sco_view

POLL_S = 300


def wait_pool(cards: int, pools=None, sleep=None) -> str:
    sleep = sleep or time.sleep
    while True:
        if not sco_view.PAUSE.exists():
            views = [sco_view.view(p) for p in (pools or job_sco.POOLS)]
            ok = [v for v in views if sco_view.fits(v, cards)]
            if ok:
                return max(ok, key=lambda v: (v["idle_estimate"] or 0, v["machines_with_free_cards"]))["pool"]
        sleep(POLL_S)


def queue_pool(sleep=None) -> str:
    """``--pool queue``: the pool with the fewest waiting jobs, once the pause file is absent."""
    sleep = sleep or time.sleep
    while sco_view.PAUSE.exists():
        sleep(POLL_S)
    return job_sco.pick_pool()


def submit(run: Path, cards: int, name: str, command, pool=None) -> str:
    job = run / "job"
    receipt = job / "submission.json"
    if receipt.exists():
        return json.loads(receipt.read_text())["job_id"]
    job.mkdir(parents=True, exist_ok=True)
    (job / "launch.sh").write_text("#!/usr/bin/env bash\nexec " + " ".join(shlex.quote(c) for c in command) + "\n")
    (job / "launch.sh").chmod(0o755)
    pool = queue_pool() if pool == "queue" else (pool or wait_pool(cards))
    cmd = [job_sco.SCO, "acp", "jobs", "create", "--workspace-name=aigc", f"--aec2-name={pool}",
           f"--job-name=models-{name}"[:60], "--priority=HIGHEST", "--quota-type=reserved",
           f"--container-image-url={job_sco.IMAGE}", f"--storage-mount={job_sco.MOUNT}", "--training-framework=pytorch",
           "--worker-nodes=1", f"--worker-spec={job_sco.SPECS[cards]}", f"--command=bash {job}/launch.sh"]
    res = job_sco.subprocess.run(cmd, capture_output=True, text=True)
    (job / "submit.log").write_text(res.stdout + "\n" + res.stderr)
    ids = set(job_sco.re.findall(r"pt-[a-z0-9]+", res.stdout))
    if res.returncode != 0 or len(ids) != 1:
        raise SystemExit("submission uncertain; read job/submit.log and the job list before any retry")
    job_id = ids.pop()
    receipt.write_text(json.dumps({"utc": job_sco.now(), "job_id": job_id, "pool": pool, "cards": cards,
                                   "command": cmd}, indent=1))
    return job_id


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("submit")
    s.add_argument("--run", type=Path, required=True)
    s.add_argument("--cards", type=int, choices=sorted(job_sco.SPECS), required=True)
    s.add_argument("--name", required=True)
    s.add_argument("--pool", choices=(*job_sco.POOLS, "queue"), default=None,
                   help="skip the idle wait: this pool, or queue (after the pause file, the one with the fewest "
                        "waiting jobs)")
    s.add_argument("command", nargs=argparse.REMAINDER)
    t = sub.add_parser("status")
    t.add_argument("--run", type=Path, required=True)
    a = ap.parse_args()
    if a.cmd == "submit":
        command = a.command[1:] if a.command[:1] == ["--"] else a.command
        if not command:
            raise SystemExit("give the command after --")
        print(submit(a.run, a.cards, a.name, command, a.pool))
    else:
        print(json.dumps(job_sco.status(a.run)))


if __name__ == "__main__":
    main()
