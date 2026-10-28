"""The intermediate evaluation points of the model line's full downstream runs (the user's 3.3 of 2026-10-26): the
downstream line's E1 protocol (VAGEN-Lite-downstream/docs/downstream/STATUS.md, the user's decision of 2026-10-23):
a run keeps only its newest full state, so before the next one replaces it the weights of the 150k and 175k states
are copied out with the downstream line's downstream/train/extract_weights.py into EXP/evals_src/steps_<N>/, and the
copy is evaluated by one 8-card job (the downstream line's run_eval_ckpt.sh: 8 policy servers x 96 simulator
processes over the fixed plan; here model_matrix/downstream/eval_ckpt.sh, which runs it or, for a Cambrian-S start,
the same steps through the Cambrian entry) into EXP/evals/steps_<N>/out.  The last step is evaluated inside the
training job (run_full.sh).  The excluded episodes (downstream/eval/excluded_episodes.json) are taken out when the
results are summarized, as the downstream line does.

    python -m model_matrix.downstream.watch_points --exp EXP [--exp EXP ...] [--points 150000,175000] [--every 600] \
        [--pool queue|h800|zoetrope]

Polls until every point is copied and evaluated (or missed: its state replaced before the copy); writes
EXP/evals_src/watch.json and a line per action to stdout.  One evaluation job per point, submitted again once if
it ends without a complete result.
"""
from __future__ import annotations

import argparse
import datetime
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List

DS = Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite-downstream")
EXTRACT = DS / "downstream" / "train" / "extract_weights.py"
STARVLA_PY = "/mnt/umm/users/yinbaiqiao/.conda/envs/starVLA/bin/python"
EVAL = Path(__file__).with_name("eval_ckpt.sh")
POINTS = (150000, 175000)


def now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def complete_states(exp: Path) -> List[int]:
    root = exp / "run" / "full_state"
    return sorted(int(d.name.split("_")[1]) for d in root.glob("steps_*")
                  if d.name.split("_")[1].isdigit() and (d / "downstream_state.json").is_file())


def finished(out: Path) -> bool:
    try:
        return "per_task" in json.loads((out / "episodes" / "summary.json").read_text())
    except (OSError, ValueError):
        return False


class Watcher:
    def __init__(self, exps: List[Path], points=POINTS, extract: Callable = None, submit: Callable = None,
                 status: Callable = None, delete: Callable = None, log: Callable = print, pool: str = "queue"):
        from data_gen.interiorgs_v10 import job_sco

        self.exps, self.points, self.log = [Path(e) for e in exps], tuple(points), log
        self.extract = extract or self._extract
        self.submit = submit or (lambda run, name, cmd: self._submit(run, name, cmd, pool))
        self.status = status or (lambda run: job_sco.status(run))
        self.delete = delete or (lambda run: job_sco.delete(run))

    @staticmethod
    def _submit(run: Path, name: str, cmd: List[str], pool: str, sleep: Callable = time.sleep) -> str:
        """sco_job.submit; a fixed pool first waits for the pause file to go ("queue" waits inside sco_job)."""
        from model_matrix import sco_job, sco_view

        while pool != "queue" and sco_view.PAUSE.exists():
            sleep(sco_job.POLL_S)
        return sco_job.submit(run, 8, name, cmd, pool=pool)

    @staticmethod
    def _extract(state: Path, run: Path, dest: Path, log: Path) -> int:
        with open(log, "a") as f:
            return subprocess.run([STARVLA_PY, str(EXTRACT), str(state), str(run), str(dest)], stdout=f,
                                  stderr=subprocess.STDOUT).returncode

    def record(self, exp: Path) -> Dict:
        path = exp / "evals_src" / "watch.json"
        return json.loads(path.read_text()) if path.is_file() else {}

    def save(self, exp: Path, rec: Dict) -> None:
        (exp / "evals_src").mkdir(parents=True, exist_ok=True)
        (exp / "evals_src" / "watch.json").write_text(json.dumps(rec, indent=1))

    def poll(self) -> bool:
        """One pass over every run and point; True when nothing is left to do."""
        done = True
        for exp in self.exps:
            rec = self.record(exp)
            start = json.loads((exp / "experiment.json").read_text())["start"]["path"]
            states = complete_states(exp)
            for n in self.points:
                p = rec.setdefault(str(n), {})
                if p.get("missed") or p.get("evaluated") or p.get("failed"):
                    continue
                dest = exp / "evals_src" / f"steps_{n}"
                ckpt = dest / "checkpoints" / f"steps_{n}_pytorch_model.pt"
                if not (dest / "source.json").is_file():
                    if n in states:
                        self.log(f"{now()} {exp.name} {n}: copying the weights")
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        rc = self.extract(exp / "run" / "full_state" / f"steps_{n}", exp / "run", dest,
                                          exp / "evals_src" / f"extract_{n}.log")
                        p.setdefault("extract", []).append({"utc": now(), "rc": rc})
                        self.log(f"{now()} {exp.name} {n}: copy rc={rc}")
                    elif states and max(states) > n:
                        p["missed"] = f"the newest full state is steps_{max(states)}, steps_{n} was not copied"
                        self.log(f"{now()} {exp.name} {n}: MISSED ({p['missed']})")
                        continue
                    if not (dest / "source.json").is_file():
                        done = False
                        continue
                ev = exp / "evals" / f"steps_{n}"
                out = ev / "out"
                if finished(out):
                    p.update(evaluated=now())
                    if (ev / "job" / "submission.json").exists():
                        self.delete(ev)
                    self.log(f"{now()} {exp.name} {n}: evaluated")
                    continue
                done = False
                if not (ev / "job" / "submission.json").exists():
                    ev.mkdir(parents=True, exist_ok=True)
                    job = self.submit(ev, f"ds-eval-{exp.name.lower()}-{n // 1000}k",
                                      ["bash", str(EVAL), start, str(ckpt), str(out), "8", "96"])
                    p.setdefault("jobs", []).append({"utc": now(), "job_id": job})
                    self.log(f"{now()} {exp.name} {n}: evaluation submitted {job}")
                    continue
                st = self.status(ev)
                if st.get("finished") or st.get("state") in ("SUSPENDED", "NOT_FOUND"):
                    self.delete(ev)
                    if len(p.get("jobs", [])) >= 2:
                        p["failed"] = f"evaluation ended {st.get('state')} twice without a complete result"
                        self.log(f"{now()} {exp.name} {n}: FAILED ({p['failed']})")
                    else:
                        self.log(f"{now()} {exp.name} {n}: evaluation ended {st.get('state')} without a result; again")
            self.save(exp, rec)
        return done

    def run(self, every: float, sleep: Callable = time.sleep) -> None:
        while not self.poll():
            sleep(every)
        self.log(f"{now()} all points done")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", action="append", type=Path, required=True)
    ap.add_argument("--points", default=",".join(str(p) for p in POINTS))
    ap.add_argument("--every", type=float, default=600.0)
    ap.add_argument("--pool", default="queue", choices=("queue", "h800", "zoetrope"),
                    help="the evaluation jobs' pool: the one with fewer waiting jobs (default), or a fixed one "
                         "(the user's decision 28 of 2026-10-28: h800); a fixed pool also waits for the pause file")
    a = ap.parse_args()
    Watcher(a.exp, [int(p) for p in a.points.split(",") if p.strip()],
            log=lambda m: print(m, flush=True), pool=a.pool).run(a.every)


if __name__ == "__main__":
    sys.exit(main())
