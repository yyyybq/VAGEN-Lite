"""C line (q38 s4): one-card sco jobs that run a python module of this line (data build, subsets, inspection).

  python -m vagen.q38s4.tools.job prepare --job JOB -- vagen.q38s4.data_build --src ... --out ...
  python -m vagen.q38s4.tools.job submit --job JOB --pool zoetrope          # prints the sco command (dry run)
  python -m vagen.q38s4.tools.job submit --job JOB --pool zoetrope --yes    # submits it

prepare writes JOB (which must not exist): code/ (``git archive`` of the committed vagen/, as vagen.q38s4.launch
does, so a running job never reads a file that changes later) and job.json (module, arguments, threads, commit).
The job's entry is JOB/code/vagen/q38s4/tools/job_entry.sh JOB, which runs ``python -m vagen.q38s4.tools.job run
--job JOB``: the thread variables (OMP / MKL / OPENBLAS / NUMEXPR = --threads, TOKENIZERS_PARALLELISM false),
DISABLE_VERSION_CHECK=1, PYTHONPATH JOB/code, the proxy variables unset, then exec of ``python -m MODULE ARGS``.
The log is JOB/job.log.  submit runs nothing without --yes, the pool is always given (zoetrope or h800), NORMAL
priority, the verl image, one node with --gpus cards (default 1); each submission is appended to JOB/submits.jsonl.

Since 2026-11-23 a job on h800 is root in its container but writes /mnt/umm as uid 10174 (root squash; zoetrope
wrote as root and has no worker spec for aigc since about 13:10 UTC that day).  So JOB is made mode 777 and the
job's command starts with ``umask 0000`` (UMASK), so the job can write JOB and what it makes stays deletable by us.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from vagen.q38s4 import launch as L

PROXY_VARS = ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY", "no_proxy",
              "NO_PROXY")
THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
UMASK = "umask 0000; "  # the job writes /mnt/umm as another uid (root squash): keep its files ours to manage


def open_up(*dirs: Path) -> None:
    """Mode 777 on folders the squashed job must write into (existing ones only)."""
    for d in dirs:
        if Path(d).is_dir():
            os.chmod(d, 0o777)


def prepare(job: Path, module: str, argv: List[str], gpus: int = 1, threads: int = 4,
            commit: Optional[str] = None) -> Dict[str, Any]:
    job = Path(job).resolve()
    if job.exists():
        raise SystemExit(f"{job} exists: a job folder is made once")
    if not module or module.startswith("-"):
        raise SystemExit("give the module after --, e.g. -- vagen.q38s4.data_build --src ...")
    if gpus < 1 or threads < 1:
        raise SystemExit("--gpus and --threads must be >= 1")
    commit = commit or L.committed_head()
    job.mkdir(parents=True)
    open_up(job)
    L.snapshot_code(job / "code", commit)
    spec = {"utc": L.utc(), "module": module, "args": list(argv), "gpus": gpus, "threads": threads, "commit": commit,
            "image": L.IMAGE, "worker_spec": f"N4lS.Iq.I80.{gpus}"}
    (job / "job.json").write_text(json.dumps(spec, indent=1) + "\n")
    return spec


def sco_command(job: Path, spec: Dict[str, Any], pool: str, name: str) -> List[str]:
    if pool not in L.POOLS:
        raise ValueError(f"pool {pool!r} not in {L.POOLS}")
    entry = f"{UMASK}bash {job}/code/vagen/q38s4/tools/job_entry.sh {job}"
    return [L.SCO, "acp", "jobs", "create", "--workspace-name=aigc", f"--aec2-name={pool}", f"--job-name={name}",
            "--priority=NORMAL", "--quota-type=reserved", f"--container-image-url={spec['image']}",
            f"--storage-mount={L.STORAGE}", "--training-framework=pytorch", "--worker-nodes=1",
            f"--worker-spec={spec['worker_spec']}", f"--command={entry}"]


def run_sco(cmd: List[str], record: Path, name: str, pool: str, yes: bool) -> None:
    print(" ".join(shlex.quote(c) for c in cmd))
    if not yes:
        print("(dry run: add --yes to submit)")
        return
    res = subprocess.run(cmd, capture_output=True, text=True)
    with open(record, "a") as f:
        f.write(json.dumps({"utc": L.utc(), "pool": pool, "job_name": name, "rc": res.returncode,
                            "stdout": res.stdout, "stderr": res.stderr}) + "\n")
    print(res.stdout + res.stderr)
    if res.returncode:
        raise SystemExit(res.returncode)


def child_env(job: Path, threads: int, base: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    env = dict(os.environ if base is None else base)
    for k in PROXY_VARS:
        env.pop(k, None)
    for k in THREAD_VARS:
        env[k] = str(threads)
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["DISABLE_VERSION_CHECK"] = "1"
    env["PYTHONPATH"] = str(Path(job).resolve() / "code")
    return env


def run(job: Path) -> None:
    job = Path(job).resolve()
    spec = json.loads((job / "job.json").read_text())
    env = child_env(job, spec["threads"])
    argv = [sys.executable, "-m", spec["module"], *spec["args"]]
    print(f"[q38s4 job] {L.utc()} commit {spec['commit']} threads {spec['threads']}: "
          f"{' '.join(shlex.quote(a) for a in argv)}", flush=True)
    os.chdir(job / "code")
    os.execve(sys.executable, argv, env)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    argv = list(sys.argv[1:] if argv is None else argv)
    rest: List[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, rest = argv[:i], argv[i + 1:]
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--job", required=True)
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--threads", type=int, default=4)
    s = sub.add_parser("submit")
    s.add_argument("--job", required=True)
    s.add_argument("--pool", required=True, choices=L.POOLS)
    s.add_argument("--job-name")
    s.add_argument("--yes", action="store_true")
    r = sub.add_parser("run")
    r.add_argument("--job", required=True)
    a = ap.parse_args(argv)
    a.rest = rest
    if a.cmd == "prepare" and not rest:
        ap.error("prepare: give the module and its arguments after --")
    return a


def main(argv: Optional[List[str]] = None) -> None:
    a = parse_args(argv)
    if a.cmd == "prepare":
        print(json.dumps(prepare(Path(a.job), a.rest[0], a.rest[1:], a.gpus, a.threads), indent=1))
    elif a.cmd == "submit":
        job = Path(a.job).resolve()
        spec = json.loads((job / "job.json").read_text())
        name = a.job_name or f"q38s4-job-{job.name}"
        run_sco(sco_command(job, spec, a.pool, name), job / "submits.jsonl", name, a.pool, a.yes)
    else:
        run(Path(a.job))


if __name__ == "__main__":
    sys.exit(main())
