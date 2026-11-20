"""q38 base zero-shot: what the training chain would submit for a plan, without submitting anything or writing a file.

    PYTHONPATH=. python -B vagen/q38/base_zs/dryrun.py /mnt/umm/users/yinbaiqiao/q38/runs/q38-base-zs/plan.json

Runs the chain's own check_code and evaluate step with eval_jobs / tables replaced (no sco call, no progress.json:
the Chain is built from a copy of the plan whose "tr" is a folder that does not exist and is never created), and
prints every job: name, cards, the job script's arguments, the environment job_sco would pass, and the checks
(names start with q38, every result folder is under the chain folder, none exists yet)."""
from __future__ import annotations

import json
import sys
from pathlib import Path

from data_gen.interiorgs_v10 import train_chain as T


def main() -> int:
    plan_path = Path(sys.argv[1])
    plan = json.loads(plan_path.read_text())
    tr = Path(plan["tr"])
    dry = dict(plan, tr=str(tr / "_dryrun_never_written"))
    chain = T.Chain(dry, sleep=lambda s: None)
    chain.save = lambda: None  # nothing written
    chain.mark = lambda step, **kw: None
    problems = []
    try:
        chain.check_code()
        print("check_code: OK (HEAD", plan["commit"], ")")
    except T.ChainError as exc:
        problems.append(f"check_code: {exc}")
    got = {}
    chain.eval_jobs = lambda step, jobs: got.update(jobs)
    chain.tables = lambda: None
    chain.evaluate()
    env = chain.job_env()
    print("job env:", json.dumps(env))
    for name, (eval_dir, result, model, prompt_format, options, *mode) in got.items():
        args = ([mode[0], str(eval_dir), result, str(model), *options] if mode else
                ["eval", str(eval_dir), result, str(model), prompt_format or "-", *options])
        cards = 1 if mode else chain.eval_cards()
        print(f"{name}: {cards} card(s): {' '.join(args)}")
        out = Path(eval_dir) / result
        if not result.startswith("q38"):
            problems.append(f"{name}: result name {result} does not start with q38")
        if tr.resolve() not in out.resolve().parents:
            problems.append(f"{name}: result folder {out} is outside {tr}")
        if out.exists():
            problems.append(f"{name}: result folder {out} exists already")
        if not (Path(model) / "config.json").is_file():
            problems.append(f"{name}: no config.json in {model}")
    print(len(got), "jobs;", sum(1 if j[5:] else chain.eval_cards() for j in got.values()), "cards if all run at once")
    for p in problems:
        print("PROBLEM", p)
    print("DRY RUN OK" if not problems else "DRY RUN NOT OK")
    return 0 if not problems else 1


if __name__ == "__main__":
    raise SystemExit(main())
