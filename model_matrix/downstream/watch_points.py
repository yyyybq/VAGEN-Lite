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
        [--pool queue|h800|zoetrope] [--extract-on local|sco]

The copy follows the user's login-node rule (2026-10-28): here at nice 19 with 2 threads per numeric library, and
only while the account's cores in use plus these stay within 12 of its 14 (cores_in_use, the rule's own check);
``--extract-on sco`` makes it a 1-card sco job in ``--pool`` instead.

Polls until every point is copied and evaluated (or missed: its state replaced before the copy); writes
EXP/evals_src/watch.json and a line per action to stdout.  One evaluation job per point, submitted again once if
it ends without a complete result.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
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


COPY_THREADS = 2  # the user's login-node rule (2026-10-28): at most 4 cores per task
CORE_BUDGET = 12  # of the account's 14 on the login node, 2 left to monitoring, ssh and the IDE
CPU_STAT = Path("/sys/fs/cgroup/cpu.stat")


def cores_in_use(seconds: float = 5.0, sleep: Callable = time.sleep):
    """The account's cores in use on the login node (cgroup usage over ``seconds``, the rule's own check); None
    where the file is missing."""
    def usage():
        for line in CPU_STAT.read_text().splitlines():
            if line.startswith("usage_usec"):
                return int(line.split()[1])
        return None

    try:
        a = usage()
        sleep(seconds)
        b = usage()
    except OSError:
        return None
    return None if a is None or b is None else (b - a) / (seconds * 1e6)


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
                 status: Callable = None, delete: Callable = None, log: Callable = print, pool: str = "queue",
                 extract_on: str = "local"):
        from data_gen.interiorgs_v10 import job_sco

        self.exps, self.points, self.log = [Path(e) for e in exps], tuple(points), log
        self.extract_on = extract_on
        if extract:
            self.extract = extract
        elif extract_on == "sco":  # the user's login-node rule of 2026-10-28: the copy runs as a 1-card job
            self.extract = lambda state, run, dest, log_path: self._extract_sco(state, run, dest, log_path, pool)
        else:
            self.extract = self._extract
        self.copy_tries = 2 if extract_on == "sco" else None  # local: tried again at every poll, as before
        self.submit = submit or (lambda run, name, cmd: self._submit(run, name, cmd, pool))
        self.status = status or (lambda run: job_sco.status(run))
        self.delete = delete or (lambda run: job_sco.delete(run))

    @staticmethod
    def copy_job(dest: Path) -> Path:
        """The folder of a point's copy job: EXP/evals_src/extract_<N> (next to the copy, steps_<N>)."""
        return dest.parent / f"extract_{dest.name.split('_', 1)[1]}"

    @classmethod
    def _extract_sco(cls, state: Path, run: Path, dest: Path, log: Path, pool: str):
        """The copy as a 1-card sco job (the downstream line's extract_weights.py, the StarVLA Python): None while
        the job is to be submitted (the pause file, for a fixed pool), waits or runs; then 0 when the copy is
        complete (dest/source.json), 1 when the job ended without it.  The job runs as root, so it gives the copy
        and its log back to this user."""
        import shlex

        from data_gen.interiorgs_v10 import job_sco
        from model_matrix import sco_job, sco_view

        job = cls.copy_job(dest)
        if not (job / "job" / "submission.json").is_file():
            if pool != "queue" and sco_view.PAUSE.exists():
                return None
            owner = f"{os.getuid()}:{os.getgid()}"
            script = (f"{shlex.quote(STARVLA_PY)} {shlex.quote(str(EXTRACT))} {shlex.quote(str(state))} "
                      f"{shlex.quote(str(run))} {shlex.quote(str(dest))} >> {shlex.quote(str(log))} 2>&1; rc=$?; "
                      f"chown -R {owner} {shlex.quote(str(dest))} {shlex.quote(str(log))}; exit $rc")
            exp, n = dest.parents[1].name, int(dest.name.split("_", 1)[1])
            sco_job.submit(job, 1, f"ds-copy-{exp.lower()}-{n // 1000}k", ["bash", "-c", script], pool=pool)
            return None
        st = job_sco.status(job)
        if st.get("state") in job_sco.WAITING or st.get("state") == "RUNNING":
            return None
        rc = 0 if (dest / "source.json").is_file() else 1
        job_sco.delete(job)
        return rc

    def drop_copy_job(self, dest: Path) -> None:
        """A point missed while its copy job was still there: the job is deleted."""
        if self.extract_on == "sco" and (self.copy_job(dest) / "job" / "submission.json").is_file():
            from data_gen.interiorgs_v10 import job_sco

            job_sco.delete(self.copy_job(dest))

    @staticmethod
    def _submit(run: Path, name: str, cmd: List[str], pool: str, sleep: Callable = time.sleep) -> str:
        """sco_job.submit; a fixed pool first waits for the pause file to go ("queue" waits inside sco_job)."""
        from model_matrix import sco_job, sco_view

        while pool != "queue" and sco_view.PAUSE.exists():
            sleep(sco_job.POLL_S)
        return sco_job.submit(run, 8, name, cmd, pool=pool)

    @staticmethod
    def _extract(state: Path, run: Path, dest: Path, log: Path, threads: int = COPY_THREADS,
                 in_use: Callable = None):
        """The copy here, under the user's login-node rule (2026-10-28): ``threads`` per process (the numeric
        libraries' variables), at nice 19, and only when the cores in use plus these stay within CORE_BUDGET;
        None (tried at the next poll) otherwise."""
        used = (in_use or cores_in_use)()
        if used is not None and used + threads > CORE_BUDGET:
            return None
        env = dict(os.environ, TOKENIZERS_PARALLELISM="false",
                   **{v: str(threads) for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                                                "NUMEXPR_NUM_THREADS")})
        with open(log, "a") as f:
            return subprocess.run(["nice", "-n", "19", STARVLA_PY, str(EXTRACT), str(state), str(run), str(dest)],
                                  stdout=f, stderr=subprocess.STDOUT, env=env).returncode

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
                args = (exp / "run" / "full_state" / f"steps_{n}", exp / "run", dest,
                        exp / "evals_src" / f"extract_{n}.log")
                if not (dest / "source.json").is_file() and n not in states and states and max(states) > n:
                    p["missed"] = f"the newest full state is steps_{max(states)}, steps_{n} was not copied"
                    self.drop_copy_job(dest)
                    p.pop("copy_pending", None)
                    self.log(f"{now()} {exp.name} {n}: MISSED ({p['missed']})")
                    continue
                pending_job = self.extract_on == "sco" and (self.copy_job(dest) / "job" / "submission.json").is_file()
                if pending_job or (not (dest / "source.json").is_file() and n in states):
                    if not p.get("copy_pending"):
                        self.log(f"{now()} {exp.name} {n}: copying the weights")
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    rc = self.extract(*args)
                    if rc is None:  # --extract-on sco: the copy job is to be submitted, queued or running
                        p["copy_pending"] = True
                        done = False
                        continue
                    p.pop("copy_pending", None)
                    p.setdefault("extract", []).append({"utc": now(), "rc": rc})
                    self.log(f"{now()} {exp.name} {n}: copy rc={rc}")
                    if rc != 0 and self.copy_tries and len(p["extract"]) >= self.copy_tries:
                        p["failed"] = f"the weight copy failed {len(p['extract'])} times"
                        self.log(f"{now()} {exp.name} {n}: FAILED ({p['failed']})")
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
    ap.add_argument("--extract-on", default="local", choices=("local", "sco"),
                    help="where the weight copy runs: here (default) or as a 1-card sco job in --pool (the user's "
                         "login-node rule of 2026-10-28)")
    a = ap.parse_args()
    Watcher(a.exp, [int(p) for p in a.points.split(",") if p.strip()],
            log=lambda m: print(m, flush=True), pool=a.pool, extract_on=a.extract_on).run(a.every)


if __name__ == "__main__":
    sys.exit(main())
