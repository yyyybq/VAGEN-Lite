"""sco jobs of the VPD line: a guarded wrapper around downstream/sco_job.py of THIS worktree.

    python -m vpd.sco prepare --run RUN --name vpd-wb-smoke [--pool zoetrope|h800|zoe-cf-render|auto] [--as-root]
                              [--est-gph 0.8] [--gate wb] [--no-snapshot]
                              -- bash /mnt/umm/users/yinbaiqiao/VAGEN-Lite-vpd/vpd/jobs/wb_smoke.sh ARGS
    python -m vpd.sco submit  --run RUN [--est-gph X]
    python -m vpd.sco status  [--all]
    python -m vpd.sco delete  --run RUN [--force-live]
    python -m vpd.sco resolve --run RUN          # settle an uncertain create (never resubmits)
    python -m vpd.sco budget                     # ledger summary by gate
    python -m vpd.sco queue                      # live downstream-vpd-* jobs of user yinbaiqiao (one list page)
    python -m vpd.sco disk                       # the disk check of submit (a); exit 1 when it refuses

sco_job is imported from VAGEN-Lite-vpd/downstream.  Display names are "downstream-" + NAME with NAME matching
^vpd-[a-z0-9-]+$ (never "ev...": the downstream watcher claims downstream-ev*).  One card per job.

Code snapshots (DECISIONS G1, 2026-11-10; on by default): prepare refuses unless vpd/, downstream/ and
eval_external/ have no uncommitted or untracked (non-ignored) files, then extracts `git archive HEAD` of those trees
into RUN/code with RUN/code/SNAPSHOT.json (vpd.snapshot).  Every payload arg that starts with the worktree path is
rewritten to the same path under RUN/code (an arg that only contains it is refused), and launch.sh exports
VPD_WORKTREE=RUN/code PYTHONPATH=RUN/code and cds into RUN/code, so the job never reads the live worktree's code (the
one exception: the non-root owner wrapper examples/train/active_spatial/sco_run_as_artifact_owner.sh, outside the
snapshot trees).  prepared.json carries the rewritten payload, worktree = RUN/code and code_snapshot; vpd.json the
snapshot summary and the rewrites.  --no-snapshot (tests / emergencies only) keeps the live worktree and is recorded
in vpd.json ("no_snapshot": true).
Pools (G7): sco_job.POOLS (h800, zoetrope) plus EXTRA_POOLS (zoe-cf-render), which only an explicit --pool names.

submit refuses (exit 2) unless, in this order:
  (a) disk: no disk_runs/FLAG_*, no disk_runs/flags/STOP_SUBMIT_DISK_LOW, and the last row of
      VAGEN-Lite-disk/docs/disk/usage.csv has remaining_TB >= 1.5;
  (b) concurrency: at most MAX_LIVE-1 = 3 OTHER live vpd jobs (ledger jobs checked with sco_job.status, plus
      downstream-vpd-* jobs of the user seen in the job list; an unreadable state counts as live);
  (c) budget (DECISIONS B2): vpd.ledger.budget_check for the job's gate, open jobs charged max(est, elapsed).
The whole check + create + ledger append runs under a submit lock, at least 2 s after the previous create.
A create with unknown outcome (sco_job.SubmitUncertain) is logged as "uncertain" (still charged and counted as
live), submit.log is printed, and nothing is retried: use `resolve`.

delete re-checks the state twice (2 s apart) and deletes only a terminal job (SUCCEEDED, FAILED, STOPPED,
CANCELLED) with `sco acp jobs delete` (no stop), or records a NOT_FOUND job whose absence the job list confirms;
a live job only with --force-live (sco_job.delete: stop + delete).  The end time and the actual GPU-h
(sco start_time -> complete_time, else the worker.log [launch] span) go to the ledger.  A force-deleted job that
never started (both checks in a waiting state without start_time, no [launch] line in worker.log) is charged
0.0 GPU-h, gph_source "never_started" (G7); any other force-deleted job without a start time keeps its estimate.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from vpd import ledger, paths, snapshot

WORKTREE = paths.WORKTREE
_DS = str(paths.DOWNSTREAM)
if _DS not in sys.path:
    sys.path.insert(0, _DS)
import sco_job  # noqa: E402  (downstream/sco_job.py of the vpd worktree)

if Path(sco_job.__file__).resolve() != (paths.DOWNSTREAM / "sco_job.py").resolve():
    raise ImportError(f"sco_job came from {sco_job.__file__}, not {paths.DOWNSTREAM / 'sco_job.py'}")
if Path(sco_job.WORKTREE).resolve() != WORKTREE.resolve():
    raise ImportError(f"sco_job.WORKTREE is {sco_job.WORKTREE}, expected {WORKTREE}")

NAME_RE = re.compile(r"^vpd-[a-z0-9-]+$")
DISPLAY_PREFIX = "downstream-vpd-"
USER = "yinbaiqiao"
MAX_LIVE = int(os.environ.get("VPD_MAX_LIVE") or 4)  # live vpd 1-card jobs including the new one
SUBMIT_SPACING_S = 2.0
DELETE_RECHECK_S = 2.0

DISK_RUNS = paths.USER_ROOT / "disk_runs"
STOP_FLAG = DISK_RUNS / "flags" / "STOP_SUBMIT_DISK_LOW"
WARN_FLAG = DISK_RUNS / "WARN_DISK_LOW"
USAGE_CSV = paths.USER_ROOT / "VAGEN-Lite-disk" / "docs" / "disk" / "usage.csv"
MIN_REMAINING_TB = 1.5
USAGE_STALE_H = 6.0
# User decision 2026-11-09 (vpd session, answer to the disk-flag question): exception for the WalkerBench and
# TVR gates only, hard cap 10 GB for everything this line keeps under OUT; E3VS waits for the disk line.
# The exception lifts ONLY the STOP_SUBMIT_DISK_LOW flag and the remaining_TB < 1.5 rule; any FLAG_* file,
# an unreadable usage.csv, or remaining_TB < EXC_HARD_FLOOR_TB still refuses.
DISK_EXCEPTION = paths.OUT / "DISK_EXCEPTION.json"
EXC_HARD_FLOOR_TB = 0.5

TERMINAL = frozenset(sco_job.FINISHED)  # SUCCEEDED FAILED STOPPED CANCELLED
WAITING = frozenset(sco_job.WAITING_STATES)  # created, no cards yet
PAGE_SIZE = 500
META = "vpd.json"
CODE_DIR = "code"  # RUN/code: the job's code snapshot (DECISIONS G1)
# G7 (2026-11-10, ROUND3 section 4: "Vulkan 复查作业，以及为换机器池做的代码改动"): pools besides sco_job.POOLS.
# Explicit only (`--pool auto` never picks one).  sco_job.prepare accepts only its own POOLS, so prepare() rewrites
# prepared.json["pool"], which sco_job.create_command passes as --aec2-name; the spec stays sco_job.SPECS[1]
# (N4lS.Iq.I80.1, listed by `sco aec2 clusters list-workerspec` for zoe-cf-render).  Whether we have quota there is
# unknown until a job schedules (vulkan_recheck 2026-11-10).
EXTRA_POOLS = ("zoe-cf-render",)
POOL_CHOICES = tuple(sco_job.POOLS) + EXTRA_POOLS


class Refused(SystemExit):
    """A guarded action was refused (exit code 2); the message says why."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


def now_utc() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


# --- names --------------------------------------------------------------------------------------------------

def validate_name(name: str) -> str:
    if not isinstance(name, str) or not NAME_RE.fullmatch(name):
        raise ValueError(f"job name {name!r} must match {NAME_RE.pattern}")
    if name.startswith("ev") or ("downstream-" + name).startswith("downstream-ev"):
        raise ValueError(f"job name {name!r}: names starting with 'ev' belong to the downstream watcher")
    if not re.fullmatch(r"[a-z0-9-]{3,40}", name):
        raise ValueError(f"job name {name!r}: 3-40 characters (sco_job rule)")
    if name.endswith("-") or "--" in name:
        raise ValueError(f"job name {name!r}: no trailing or doubled '-'")
    return name


# --- prepare ------------------------------------------------------------------------------------------------

def _git(*args) -> Optional[str]:
    try:
        res = subprocess.run(["git", "-C", str(WORKTREE), *args], capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return res.stdout.strip() if res.returncode == 0 else None


def vpd_tree_digest(root: Path = paths.VPD_PKG) -> Tuple[str, int]:
    """(sha256 over 'relpath\\0sha256(file)\\n' of every file under vpd/, file count); skips caches."""
    h = hashlib.sha256()
    n = 0
    for p in sorted(root.rglob("*")):
        if not p.is_file() or "__pycache__" in p.parts or p.suffix == ".pyc" or ".pytest_cache" in p.parts:
            continue
        h.update(f"{p.relative_to(root).as_posix()}\0{hashlib.sha256(p.read_bytes()).hexdigest()}\n".encode())
        n += 1
    return h.hexdigest(), n


def vpd_code_state() -> dict:
    """HEAD, the porcelain status of vpd/ (untracked included) and a digest of the vpd/ tree at prepare time.
    In a job that runs from a code snapshot (no .git, no git binary) the commit is the snapshot's (SNAPSHOT.json)
    and the status is empty by construction (prepare refused a dirty tree)."""
    snap = snapshot.snapshot_state(WORKTREE)
    if snap["mode"] == "snapshot":
        digest, n = vpd_tree_digest(WORKTREE / "vpd")
        return {"commit": snap["commit"], "vpd_status": [], "vpd_tree_sha256": digest, "vpd_files": n,
                "snapshot": snap}
    dirty = _git("status", "--porcelain", "--", "vpd")
    digest, n = vpd_tree_digest()
    return {"commit": _git("rev-parse", "HEAD"), "vpd_status": dirty.splitlines() if dirty else [],
            "vpd_tree_sha256": digest, "vpd_files": n}


ENV_KEY_RE = re.compile(r"[A-Z_][A-Z0-9_]{0,63}")
ENV_VAL_RE = re.compile(r"[A-Za-z0-9_./=+-]{0,200}")  # no ',' or ':' (sco's --env separators), no spaces
MAX_ENV = 10  # sco: at most 10 envs


def parse_env(items) -> Dict[str, str]:
    """['KEY:VAL', ...] -> {KEY: VAL}; refuses what sco's `--env KEY1:VAL1,KEY2:VAL2` cannot carry."""
    env: Dict[str, str] = {}
    for it in items or []:
        if not isinstance(it, str) or ":" not in it:
            raise ValueError(f"--env wants KEY:VAL, got {it!r}")
        k, v = it.split(":", 1)
        if not ENV_KEY_RE.fullmatch(k) or not ENV_VAL_RE.fullmatch(v):
            raise ValueError(f"--env {it!r}: KEY [A-Z_][A-Z0-9_]*, VAL without ',', ':' or spaces")
        if k in env:
            raise ValueError(f"--env {k} given twice")
        env[k] = v
    if len(env) > MAX_ENV:
        raise ValueError(f"at most {MAX_ENV} --env entries")
    return env


def _wt_prefixes() -> List[str]:
    """The worktree path as given and resolved (longest first)."""
    return sorted({str(WORKTREE), str(WORKTREE.resolve())}, key=len, reverse=True)


def _mentions(text: str, prefix: str) -> bool:
    """True if text contains the path prefix at a path boundary (not e.g. VAGEN-Lite-vpd2)."""
    return re.search(re.escape(prefix) + r"(?=/|$|[\s'\";:,)])", text) is not None


def snapshot_payload(payload: List[str], code: Path) -> Tuple[List[str], List[List[str]]]:
    """(payload with every arg that starts with the worktree path moved under code, [[old, new], ...]).
    An arg that mentions the worktree elsewhere (KEY=/WT/..., a bash -c string) would run live code: refused.
    A rewritten path must exist in the snapshot (else it is uncommitted, ignored or outside the snapshot trees)."""
    out, rewrites = [], []
    for a in payload:
        new = a
        for p in _wt_prefixes():
            if a == p or a.startswith(p + "/"):
                new = str(code) + a[len(p):]
                break
        if new == a:
            hit = [p for p in _wt_prefixes() if _mentions(a, p)]
            if hit:
                raise ValueError(f"payload arg {a!r} mentions the worktree {hit[0]} but does not start with it: "
                                 "give the path as its own argument (prepare rewrites it to RUN/code)")
        else:
            if not os.path.lexists(new):
                raise ValueError(f"payload arg {a!r}: {new} is not in the code snapshot (uncommitted, ignored, or "
                                 f"outside {'/, '.join(snapshot.TREES)}/)")
            rewrites.append([a, new])
        out.append(new)
    return out, rewrites


def snapshot_launch(text: str, code: Path, commit: str) -> str:
    """sco_job's launch.sh with its one `cd WORKTREE` line replaced by the snapshot exports + cd RUN/code."""
    lines = text.splitlines(keepends=True)
    want = f"cd {shlex.quote(str(WORKTREE))}\n"
    cds = [i for i, ln in enumerate(lines) if ln.startswith("cd ")]
    if [lines[i] for i in cds] != [want]:
        raise RuntimeError(f"launch.sh: expected exactly one line {want!r}, found {[lines[i] for i in cds]}")
    q = shlex.quote(str(code))
    lines[cds[0]] = (
        f"# vpd.sco code snapshot of {commit} (DECISIONS G1): this job runs {code}, never the live worktree\n"
        f"export VPD_WORKTREE={q} PYTHONPATH={q}\n"
        f"cd {q} && [[ -f {snapshot.MARKER} ]] || {{ echo \"[launch] no code snapshot at \"{q} >&2; exit 97; }}\n")
    new = "".join(lines)
    rest = new.replace(shlex.quote(str(WORKTREE / sco_job.RUN_AS_OWNER)), "")  # the owner wrapper (non-root)
    left = [p for p in _wt_prefixes() if _mentions(rest, p)]
    if left:
        raise RuntimeError(f"launch.sh still refers to the live worktree {left[0]} after the snapshot rewrite")
    return new


def prepare(run, name: str, payload: List[str], pool: str = "zoetrope", as_root: bool = False,
            est_gph: Optional[float] = None, gate: Optional[str] = None, env=None, snapshot_code: bool = True) -> Path:
    """sco_job.prepare(run, 1, pool, name, payload, as_root) after the vpd checks; writes RUN/job/vpd.json.
    env: optional {KEY: VAL} (or ['KEY:VAL']) for the job container, passed as `sco ... --env` (2026-11-09: the
    TVR debug job asks for NVIDIA_DRIVER_CAPABILITIES:all; whether the platform honours it is measured in-job).
    pool: sco_job.POOLS, EXTRA_POOLS (explicit only) or "auto" (sco_job.pick_pool over sco_job.POOLS).
    snapshot_code (default True; CLI --no-snapshot sets False): the job runs from the code snapshot RUN/code
    (module docstring); refused (Refused) unless the snapshot trees are clean."""
    run = Path(run).resolve()
    validate_name(name)
    env = parse_env(env) if isinstance(env, (list, tuple)) else parse_env([f"{k}:{v}" for k, v in (env or {}).items()])
    inferred = ledger.gate_of(name)
    if gate is not None and gate not in ledger.GATES:
        raise ValueError(f"--gate must be one of {ledger.GATES}")
    if gate is not None and inferred is not None and gate != inferred:
        raise ValueError(f"--gate {gate} contradicts the name {name} (gate {inferred})")
    gate = gate or inferred
    if est_gph is not None and not float(est_gph) > 0:
        raise ValueError("--est-gph must be > 0")
    if not payload:
        raise ValueError("give the payload command after --")
    launch_target = (run / "job" / "launch.sh")

    def no_self_exec(args):
        for a in args:
            if a and ("launch.sh" in a) and Path(a).resolve() == launch_target:
                raise ValueError("the payload must not be RUN/job/launch.sh itself (self-exec trap)")

    no_self_exec(payload)
    if pool == "auto":
        pool = sco_job.pick_pool()  # sco_job.POOLS only: an EXTRA_POOLS pool is never chosen automatically
    if pool not in POOL_CHOICES:
        raise ValueError(f"pool must be one of {POOL_CHOICES} or auto, got {pool!r}")
    if (run / "job" / "submission.json").exists():
        raise FileExistsError(f"{run / 'job'} was already submitted; use a new run directory")
    code = run / CODE_DIR
    snap, rewrites, job_payload = None, [], list(payload)
    if snapshot_code:
        if (run / "job" / "launch.sh").exists() or (run / "job" / META).exists():
            raise ValueError(f"{run} was prepared before: use a new run directory")
        for p in _wt_prefixes():
            if str(run) == p or str(run).startswith(p + "/"):
                raise ValueError(f"RUN {run} is inside the worktree {p}")
        try:
            snap = snapshot.create(WORKTREE, code)
        except snapshot.SnapshotError as e:
            raise Refused(str(e)) from None
    try:
        if snapshot_code:
            job_payload, rewrites = snapshot_payload(list(payload), code)
            no_self_exec(job_payload)
        launch = sco_job.prepare(run, 1, pool if pool in sco_job.POOLS else sco_job.POOLS[0], name, job_payload,
                                 as_root=as_root)
        pp = run / "job" / "prepared.json"
        rec = json.loads(pp.read_text())
        if env:
            rec["env"] = env
        if pool not in sco_job.POOLS:
            rec["pool"] = pool  # -> sco --aec2-name=<pool>; spec unchanged
            rec["pool_via"] = "vpd.sco EXTRA_POOLS"
        text = Path(launch).read_text()
        if snapshot_code:
            Path(launch).write_text(snapshot_launch(text, code, snap["commit"]))
            rec["source_worktree"] = rec.get("worktree")
            rec["worktree"] = str(code)  # what launch.sh cds into
            rec["code_snapshot"] = dict(snapshot.summary(snap), path=str(code / snapshot.MARKER))
            if rewrites:
                rec["payload_as_given"] = list(payload)
        else:
            if f"cd {shlex.quote(str(WORKTREE))}\n" not in text:
                raise RuntimeError(f"{launch} does not cd into {WORKTREE}")
            rec["code_snapshot"] = None
            rec["no_snapshot"] = True
        pp.write_text(json.dumps(rec, indent=1) + "\n")
    except BaseException:
        if snap is not None:
            shutil.rmtree(code, ignore_errors=True)  # the snapshot this call made; the job was never prepared
        raise
    meta = {"utc": ledger.iso(), "name": name, "display_name": "downstream-" + name, "gate": gate,
            "est_gph": float(est_gph) if est_gph is not None else None, "as_root": bool(as_root), "pool": pool,
            "gpus": 1, "worktree": str(WORKTREE), "vpd_code": vpd_code_state(),
            "snapshot": dict(snapshot.summary(snap), path=str(code / snapshot.MARKER)) if snap else None,
            "code_dir": str(code) if snap else None, "payload_rewrites": rewrites, "no_snapshot": not snapshot_code}
    if env:
        meta["env"] = env
    (run / "job" / META).write_text(json.dumps(meta, indent=1) + "\n")
    return Path(launch)


# --- disk ---------------------------------------------------------------------------------------------------

def disk_check() -> Tuple[bool, List[str]]:
    """(ok, messages) per the disk line's stop-flag rule and the last usage.csv row."""
    msgs, ok = [], True
    flags = sorted(DISK_RUNS.glob("FLAG_*")) if DISK_RUNS.is_dir() else []
    for f in flags:
        ok = False
        msgs.append(f"disk flag present: {f}")
    if STOP_FLAG.exists():
        ok = False
        try:
            body = STOP_FLAG.read_text().strip()[:300]
        except OSError:
            body = "?"
        msgs.append(f"disk stop flag present: {STOP_FLAG} {body}")
    if WARN_FLAG.exists():
        msgs.append(f"warning only: {WARN_FLAG} present")
    try:
        lines = [ln for ln in USAGE_CSV.read_text().splitlines() if ln.strip()]
        header = [c.strip() for c in lines[0].split(",")]
        last = [c.strip() for c in lines[-1].split(",")]
        if len(lines) < 2 or len(last) != len(header):
            raise ValueError("no data row")
        row = dict(zip(header, last))
        remaining = float(row["remaining_TB"])
    except (OSError, IndexError, KeyError, ValueError) as e:
        return False, msgs + [f"cannot read remaining_TB from {USAGE_CSV}: {e!r}"]
    if remaining < MIN_REMAINING_TB:
        ok = False
        msgs.append(f"disk remaining {remaining} TB < {MIN_REMAINING_TB} TB ({row.get('utc')})")
    else:
        msgs.append(f"disk remaining {remaining} TB ({row.get('utc')})")
    t = ledger.parse_utc(row.get("utc"))
    if t is not None and (now_utc() - t).total_seconds() > USAGE_STALE_H * 3600:
        msgs.append(f"warning: usage.csv last row is older than {USAGE_STALE_H:.0f} h")
    return ok, msgs


def tree_bytes(root: Path) -> int:
    """Allocated bytes under root (like du; no symlink following; unreadable entries are skipped)."""
    total, stack = 0, [str(root)]
    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                for e in it:
                    try:
                        st = e.stat(follow_symlinks=False)
                    except OSError:
                        continue
                    total += getattr(st, "st_blocks", 0) * 512
                    if e.is_dir(follow_symlinks=False):
                        stack.append(e.path)
        except OSError:
            continue
    return total


def disk_exception_ok(gate: Optional[str]) -> Tuple[bool, str]:
    """Whether the user's recorded disk exception lets a job of this gate pass a refused disk_check."""
    if not DISK_EXCEPTION.exists():
        return False, f"no user disk exception ({DISK_EXCEPTION})"
    try:
        exc = json.loads(DISK_EXCEPTION.read_text())
        gates = list(exc["gates"])
        cap_gb = None if exc["cap_gb"] is None else float(exc["cap_gb"])
    except (OSError, ValueError, KeyError, TypeError) as e:
        return False, f"unreadable disk exception {DISK_EXCEPTION}: {e!r}"
    if gate not in gates:
        return False, f"user disk exception covers gates {gates}, not {gate!r}"
    if DISK_RUNS.is_dir() and sorted(DISK_RUNS.glob("FLAG_*")):
        return False, "user disk exception does not cover disk_runs/FLAG_* flags"
    try:
        lines = [ln for ln in USAGE_CSV.read_text().splitlines() if ln.strip()]
        row = dict(zip([c.strip() for c in lines[0].split(",")], [c.strip() for c in lines[-1].split(",")]))
        remaining = float(row["remaining_TB"])
        if len(lines) < 2:
            raise ValueError("no data row")
    except (OSError, IndexError, KeyError, ValueError) as e:
        return False, f"user disk exception needs a readable usage.csv: {e!r}"
    if remaining < EXC_HARD_FLOOR_TB:
        return False, f"disk remaining {remaining} TB < hard floor {EXC_HARD_FLOOR_TB} TB (exception void)"
    if cap_gb is None:
        return True, (f"user disk exception ({exc.get('by', '?')}): gate {gate}, no size cap, "
                      f"remaining {remaining} TB")
    used_gb = tree_bytes(paths.OUT) / 1e9
    if used_gb >= cap_gb:
        return False, f"vpd OUT uses {used_gb:.2f} GB >= exception cap {cap_gb:g} GB"
    return True, (f"user disk exception ({exc.get('by', '?')}): gate {gate}, vpd OUT uses {used_gb:.2f} GB "
                  f"< cap {cap_gb:g} GB, remaining {remaining} TB")


# --- job list and states ------------------------------------------------------------------------------------

def list_page() -> Optional[list]:
    """The newest PAGE_SIZE jobs of the workspace (every user), None when it cannot be read."""
    try:
        res = subprocess.run([sco_job.SCO, "acp", "jobs", "list", "--workspace-name=aigc", "--page-size",
                              str(PAGE_SIZE), "-o", "json"], capture_output=True, text=True, timeout=180)
        data = json.loads(res.stdout) if res.returncode == 0 else None
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None
    return data if isinstance(data, list) else None


def describe(job_id: str) -> Optional[dict]:
    """`sco acp jobs describe` of one job (None when it cannot be read)."""
    try:
        res = subprocess.run([sco_job.SCO, "acp", "jobs", "describe", "--workspace-name=aigc", "-o", "json", job_id],
                             capture_output=True, text=True, timeout=120)
        d = json.loads(res.stdout) if res.returncode == 0 else None
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None
    return d if isinstance(d, dict) else None


def _owner(j: dict) -> Optional[str]:
    o = j.get("ownership")
    return o.get("user_name") if isinstance(o, dict) else None


def live_vpd_from_list(jobs) -> Dict[str, dict]:
    """{pt_id: job} of the user's downstream-vpd-* jobs that are not terminal."""
    out = {}
    for j in jobs or []:
        if _owner(j) == USER and str(j.get("display_name") or "").startswith(DISPLAY_PREFIX) \
                and j.get("state") not in TERMINAL:
            out[str(j.get("name"))] = j
    return out


def absent_confirmed(job_id: str, submitted_at, page) -> bool:
    """True if the list page provably covers the job's creation time and does not contain it."""
    if page is None or not job_id:
        return False
    if any(str(j.get("name")) == job_id for j in page):
        return False
    if len(page) < PAGE_SIZE:
        return True
    oldest = [ledger.parse_utc(j.get("create_time")) for j in page]
    oldest = [t for t in oldest if t is not None]
    sub = ledger.parse_utc(submitted_at)
    return bool(oldest and sub and min(oldest) < sub - datetime.timedelta(minutes=5))


def observe(job: dict, page=None) -> dict:
    """{state, start_time, end_time, exit_code, absent} of a ledger job; state None when unreadable."""
    run = Path(job["run"])
    if job.get("status") == "uncertain" or not job.get("pt_id"):
        return {"state": "UNCERTAIN", "start_time": None, "end_time": None, "exit_code": None, "absent": False}
    try:
        st = sco_job.status(run)
    except (OSError, ValueError, KeyError) as e:
        return {"state": None, "error": repr(e), "start_time": None, "end_time": None, "exit_code": None,
                "absent": False}
    obs = {"state": st.get("state"), "start_time": st.get("start_time"), "end_time": None,
           "exit_code": st.get("exit_code"), "absent": False}
    if obs["state"] in TERMINAL:
        d = describe(job["pt_id"]) or {}
        obs["end_time"] = d.get("complete_time") or d.get("update_time")
        obs["start_time"] = obs["start_time"] or d.get("start_time")
    elif obs["state"] == "NOT_FOUND":
        obs["absent"] = absent_confirmed(job["pt_id"], job.get("submitted_at"), page)
    return obs


def is_live(obs: dict) -> bool:
    s = obs.get("state")
    if s in TERMINAL:
        return False
    if s == "NOT_FOUND" and obs.get("absent"):
        return False
    return True  # running, queued, suspended, uncertain, unreadable


def elapsed_gph(job: dict, obs: dict, at: Optional[datetime.datetime] = None) -> float:
    h = ledger.hours_between(obs.get("start_time"), obs.get("end_time") or ledger.iso(at or now_utc()))
    return (h or 0.0) * int(job.get("gpus") or 1)


def actual_gph(job: dict, obs: dict) -> Tuple[Optional[float], str]:
    gpus = int(job.get("gpus") or 1)
    h = ledger.hours_between(obs.get("start_time"), obs.get("end_time"))
    if h is not None:
        return round(h * gpus, 4), "sco_start_to_complete"
    span = ledger.worker_log_span(Path(job["run"]) / "job" / "worker.log")
    if span:
        h = ledger.hours_between(*span)
        if h is not None:
            return round(h * gpus, 4), "worker_log"
    return None, "unknown_charged_est"


def record_end(job: dict, obs: dict, state: Optional[str] = None, path=None) -> dict:
    gph, src = actual_gph(job, obs)
    return ledger.append({"event": "end", "run": job["run"], "pt_id": job.get("pt_id"),
                          "state": state or obs.get("state"), "start_time": obs.get("start_time"),
                          "end_time": obs.get("end_time"), "exit_code": obs.get("exit_code"),
                          "gph": gph, "gph_source": src}, path)


def scan(events: List[dict], page, include_ended: bool = False, close: bool = True,
         at: Optional[datetime.datetime] = None) -> Tuple[List[dict], Dict[str, float]]:
    """Observe the ledger's open jobs (and ended ones if asked); record an "end" for each that finished.
    Returns (rows, {run: elapsed GPU-h of still-open jobs})."""
    rows, elapsed = [], {}
    for run, job in ledger.fold(events).items():
        if not ledger.is_open(job):
            if include_ended:
                rows.append({"run": run, "name": job.get("name"), "gate": job.get("gate"),
                             "pt_id": job.get("pt_id"), "ledger": job.get("status"),
                             "gph": ledger.charge(job), "deleted_at": job.get("deleted_at")})
            continue
        obs = observe(job, page)
        row = {"run": run, "name": job.get("name"), "gate": job.get("gate"), "pt_id": job.get("pt_id"),
               "ledger": job.get("status"), "est": job.get("est"), **{k: obs.get(k) for k in
                                                                       ("state", "start_time", "end_time",
                                                                        "exit_code")}}
        if is_live(obs):
            elapsed[run] = elapsed_gph(job, obs, at)
            row["elapsed_gph"] = round(elapsed[run], 4)
            row["live"] = True
        else:
            row["live"] = False
            if close:
                end = record_end(job, obs)
                row["gph"] = end["gph"]
        rows.append(row)
    return rows, elapsed


# --- submit -------------------------------------------------------------------------------------------------

def _print_submit_log(run: Path) -> None:
    log = run / "job" / "submit.log"
    print(f"--- {log} ---", file=sys.stderr)
    try:
        print(log.read_text(errors="replace"), file=sys.stderr)
    except OSError as e:
        print(f"(unreadable: {e})", file=sys.stderr)


def submit(run, est_gph: Optional[float] = None, cap: Optional[float] = None, sleep=time.sleep,
           clock=time.time) -> str:
    run = Path(run).resolve()
    job = run / "job"
    meta_p = job / META
    if not meta_p.exists():
        raise Refused(f"{meta_p} missing: prepare the run with vpd.sco prepare")
    meta = json.loads(meta_p.read_text())
    gate = meta.get("gate")
    est = float(est_gph) if est_gph is not None else meta.get("est_gph")
    with ledger.file_lock(ledger.submit_lock_path()):
        if (job / "submission.json").exists():
            pt = json.loads((job / "submission.json").read_text()).get("job_id")
            raise Refused(f"{run} was already submitted as {pt}; prepare a new run directory")
        if (job / sco_job.UNCERTAIN).exists():
            _print_submit_log(run)
            raise Refused(f"{job / sco_job.UNCERTAIN}: an earlier create is uncertain; run `vpd.sco resolve`")
        events = ledger.read()
        prev = ledger.fold(events).get(str(run))
        if prev is not None and prev.get("status") in ("submitted", "uncertain", "ended"):
            raise Refused(f"{run} is already in the ledger ({prev.get('status')})")
        if gate not in ledger.GATES:
            raise Refused(f"job has no gate ({gate!r}); prepare with --gate wb|tvr|e3vs")
        if est is None or not float(est) > 0:
            raise Refused("no positive GPU-h estimate: pass --est-gph to prepare or submit")
        # (a) disk
        ok, msgs = disk_check()
        for m in msgs:
            print(f"[vpd.sco] {m}", file=sys.stderr)
        if not ok:
            exc_ok, exc_msg = disk_exception_ok(gate)
            print(f"[vpd.sco] {exc_msg}", file=sys.stderr)
            if not exc_ok:
                raise Refused("DISK REFUSED: " + "; ".join(msgs + [exc_msg]))
        # (b) concurrency
        page = list_page()
        if page is None:
            print("[vpd.sco] warning: job list unreadable; counting ledger jobs only", file=sys.stderr)
        rows, elapsed = scan(events, page)
        events = ledger.read()  # includes ends recorded by scan
        live_ids = {r["pt_id"] for r in rows if r.get("live") and r.get("pt_id")}
        n_live = sum(1 for r in rows if r.get("live"))
        foreign = [pid for pid in live_vpd_from_list(page) if pid not in live_ids]
        n_live += len(foreign)
        if foreign:
            print(f"[vpd.sco] live downstream-vpd-* jobs not in the ledger: {foreign}", file=sys.stderr)
        if n_live >= MAX_LIVE:
            raise Refused(f"CONCURRENCY REFUSED: {n_live} other live vpd jobs (at most {MAX_LIVE - 1})")
        # (c) budget
        ok, msg = ledger.budget_check(events, gate, float(est), cap, elapsed)
        print(f"[vpd.sco] {msg}", file=sys.stderr)
        if not ok:
            raise Refused(msg)
        last = ledger.last_submit_ts(events)
        if last is not None:
            wait = SUBMIT_SPACING_S - (clock() - last)
            if wait > 0:
                sleep(wait)
        base = {"name": meta["name"], "run": str(run), "gate": gate, "est": float(est),
                "gpus": int(meta.get("gpus") or 1)}
        try:
            pt = sco_job.submit(run)
        except sco_job.SubmitUncertain:
            ledger.append({"event": "uncertain", **base, "at": ledger.iso(), "ts": clock()})
            _print_submit_log(run)
            raise
        ledger.append({"event": "submit", **base, "pt_id": pt, "submitted_at": ledger.iso(), "ts": clock()})
        return pt


def resolve(run) -> Tuple[str, object]:
    """sco_job.resolve_uncertain on one list page; logs adopt / cleared to the ledger.  Never creates a job."""
    run = Path(run).resolve()
    with ledger.file_lock(ledger.submit_lock_path()):
        res = sco_job.resolve_uncertain(run, list_page())
        if res is None:
            return ("none", None)
        kind, val = res
        if kind == "adopted":
            ledger.append({"event": "adopt", "run": str(run), "pt_id": val, "at": ledger.iso()})
        elif kind == "cleared":
            ledger.append({"event": "cleared", "run": str(run), "at": ledger.iso()})
        return kind, val


# --- status / delete ----------------------------------------------------------------------------------------

def status(include_ended: bool = False) -> dict:
    page = list_page()
    rows, elapsed = scan(ledger.read(), page, include_ended=include_ended)
    known = {r.get("pt_id") for r in rows}
    foreign = [{"pt_id": pid, "display_name": j.get("display_name"), "state": j.get("state")}
               for pid, j in live_vpd_from_list(page).items() if pid not in known]
    return {"jobs": rows, "not_in_ledger": foreign, "list_readable": page is not None,
            "budget": ledger.summarise(ledger.read(), elapsed=elapsed)}


def _sco_delete(job_id: str) -> Tuple[int, str]:
    res = subprocess.run([sco_job.SCO, "acp", "jobs", "delete", "--workspace-name=aigc", job_id],
                         capture_output=True, text=True, timeout=180)
    return res.returncode, (res.stdout.strip() + " " + res.stderr.strip()).strip()


def delete(run, force_live: bool = False, sleep=time.sleep) -> str:
    run = Path(run).resolve()
    receipt = run / "job" / "submission.json"
    if not receipt.exists():
        raise Refused(f"{receipt} missing: nothing was submitted from {run}")
    pt = json.loads(receipt.read_text())["job_id"]
    with ledger.file_lock(ledger.submit_lock_path()):
        job = ledger.fold(ledger.read()).get(str(run)) or {}
        job = {"run": str(run), "gpus": 1, "est": 0.0, **job, "pt_id": pt, "status": "submitted"}
        page = list_page()
        obs1 = observe(job, page)
        sleep(DELETE_RECHECK_S)
        obs2 = observe(job, list_page())
        ended_in_ledger = (ledger.fold(ledger.read()).get(str(run)) or {}).get("status") == "ended"
        s1, s2 = obs1.get("state"), obs2.get("state")
        if s1 in TERMINAL and s2 == s1:
            if not ended_in_ledger:
                record_end(job, obs2)
            rc, out = _sco_delete(pt)
            ledger.append({"event": "delete", "run": str(run), "pt_id": pt, "at": ledger.iso(), "rc": rc,
                           "state": s2})
            return f"{pt}: {s2}; delete rc={rc} {out}"
        if s1 == s2 == "NOT_FOUND" and obs1.get("absent") and obs2.get("absent"):
            if not ended_in_ledger:
                record_end(job, obs2, state="NOT_FOUND")
            ledger.append({"event": "delete", "run": str(run), "pt_id": pt, "at": ledger.iso(), "rc": None,
                           "state": "NOT_FOUND"})
            return f"{pt}: not in the job list; recorded as deleted"
        if not force_live:
            raise Refused(f"{pt}: state {s1} then {s2}; delete only after SUCCEEDED/FAILED/STOPPED/CANCELLED or a "
                          "confirmed NOT_FOUND (pass --force-live to stop a live job)")
        out = sco_job.delete(run)  # stop + delete
        obs2 = dict(obs2, end_time=ledger.iso())
        if not ended_in_ledger:
            started = obs2.get("start_time") or obs1.get("start_time")
            if started:
                obs2["start_time"] = started
                gph, src = round(elapsed_gph(job, obs2), 4), "sco_start_to_force_delete"
            elif s1 in WAITING and s2 in WAITING and not ledger.worker_log_started(run / "job" / "worker.log"):
                gph, src = 0.0, "never_started"  # G7: still waiting for cards, the container never ran
            else:
                gph, src = None, "unknown_charged_est"  # unreadable / ambiguous: charged its estimate
            ledger.append({"event": "end", "run": str(run), "pt_id": pt, "state": f"FORCED_FROM_{s2}",
                           "start_time": obs2.get("start_time"), "end_time": obs2["end_time"],
                           "gph": gph, "gph_source": src})
        ledger.append({"event": "delete", "run": str(run), "pt_id": pt, "at": ledger.iso(), "rc": None,
                       "state": f"FORCED_FROM_{s2}", "detail": out})
        return out


# --- CLI ----------------------------------------------------------------------------------------------------

def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    payload = argv[argv.index("--") + 1:] if "--" in argv else []
    argv = argv[:argv.index("--")] if "--" in argv else argv
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["prepare", "submit", "status", "delete", "resolve", "budget", "queue", "disk"])
    ap.add_argument("--run", type=Path)
    ap.add_argument("--name")
    ap.add_argument("--pool", default="zoetrope", choices=list(POOL_CHOICES) + ["auto"],
                    help="auto picks among sco_job.POOLS only; EXTRA_POOLS must be named")
    ap.add_argument("--as-root", action="store_true")
    ap.add_argument("--no-snapshot", action="store_true",
                    help="prepare: run the live worktree instead of a code snapshot (tests / emergencies only; "
                         "recorded in vpd.json)")
    ap.add_argument("--est-gph", type=float)
    ap.add_argument("--gate", choices=list(ledger.GATES))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--force-live", action="store_true")
    ap.add_argument("--env", action="append", default=[], metavar="KEY:VAL",
                    help="prepare: container env for sco --env (repeatable; no ',' or ':' in VAL)")
    args = ap.parse_args(argv)
    if args.action in ("prepare", "submit", "delete", "resolve") and args.run is None:
        ap.error(f"{args.action} needs --run")
    try:
        if args.action == "prepare":
            if not args.name:
                ap.error("prepare needs --name")
            if args.no_snapshot:
                print("[vpd.sco] --no-snapshot: the job will run the LIVE worktree code (recorded in vpd.json)",
                      file=sys.stderr)
            print(prepare(args.run, args.name, payload, args.pool, args.as_root, args.est_gph, args.gate,
                          env=args.env, snapshot_code=not args.no_snapshot))
        elif args.action == "submit":
            print(submit(args.run, args.est_gph))
        elif args.action == "status":
            print(json.dumps(status(args.all), indent=1))
        elif args.action == "delete":
            print(delete(args.run, args.force_live))
        elif args.action == "resolve":
            print(json.dumps(resolve(args.run)))
        elif args.action == "disk":
            ok, msgs = disk_check()
            for m in msgs:
                print(m)
            print(f"DISK {'ok' if ok else 'REFUSED'}")
            return 0 if ok else 1
        elif args.action == "budget":
            print(json.dumps(ledger.summarise(ledger.read()), indent=1))
        else:
            page = list_page()
            if page is None:
                print("QUEUE unreadable")
                return 1
            live = live_vpd_from_list(page)
            for pid, j in live.items():
                pool = j.get("resource_pool")
                pool = pool.get("name") if isinstance(pool, dict) else pool
                print(f"{pid} {j.get('display_name')} {j.get('state')} {pool} {j.get('start_time')}")
            print(f"QUEUE live_vpd={len(live)} max={MAX_LIVE}")
    except Refused as e:
        print(f"[vpd.sco] {e.message}", file=sys.stderr)
        return 2
    except sco_job.SubmitUncertain as e:
        print(f"[vpd.sco] SUBMIT UNCERTAIN, not retried: {e}", file=sys.stderr)
        return 3
    except ValueError as e:
        print(f"[vpd.sco] error: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
