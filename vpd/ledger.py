"""GPU-hour ledger of the VPD sco jobs (OUT/ledger.jsonl) and the per-gate budget guard (DECISIONS B2).

The ledger is append-only JSON lines, one event per line, written under an exclusive lock file
(<ledger>.lock) in a single write + fsync, so concurrent writers never interleave.  Events:

    {"event": "submit",    "name", "run", "gate", "pt_id", "est", "gpus", "submitted_at"}
    {"event": "uncertain", "name", "run", "gate", "est", "gpus", "at"}       # create outcome unknown
    {"event": "adopt",     "run", "pt_id", "at"}                            # uncertain -> found in the job list
    {"event": "cleared",   "run", "at"}                                     # uncertain -> no job appeared
    {"event": "end",       "run", "pt_id", "state", "start_time", "end_time", "gph", "gph_source"}
    {"event": "delete",    "run", "pt_id", "at", "rc"}

Budget rule (B2, hard): a new job of gate g may be submitted only if
    spent_g(ended jobs, actual GPU-h) + committed_g(open jobs) + est_high(new) <= cap_g
and the same holds summed over all gates against CAP, where an open job (submitted or uncertain, no "end"
yet) is charged max(est_high, elapsed GPU-h so far).  Gate caps: tvr 0.5, e3vs 10.5, wb = CAP - (tvr + e3vs);
CAP = env VPD_GPUH_CAP, default 23 (must not be raised without the user).

CLI:  python -m vpd.ledger summary [--ledger PATH]
"""

from __future__ import annotations

import argparse
import contextlib
import datetime
import errno
import fcntl
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from vpd import paths

GATES = ("wb", "tvr", "e3vs")
CAP_DEFAULT = 23.0
RESERVE = {"tvr": 0.5, "e3vs": 10.5}  # explicit reservations; wb gets the rest of CAP (B2)
EVENTS = ("submit", "uncertain", "adopt", "cleared", "end", "delete")
LOCK_TIMEOUT_S = 120.0


def now_utc() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def iso(t: Optional[datetime.datetime] = None) -> str:
    return (t or now_utc()).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_utc(text) -> Optional[datetime.datetime]:
    if not text:
        return None
    try:
        t = datetime.datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    except ValueError:
        return None
    return t if t.tzinfo else t.replace(tzinfo=datetime.timezone.utc)


def hours_between(start, end) -> Optional[float]:
    a, b = parse_utc(start), parse_utc(end)
    if a is None or b is None:
        return None
    return max(0.0, (b - a).total_seconds() / 3600.0)


# --- caps ---------------------------------------------------------------------------------------------------

def total_cap(env=None) -> float:
    env = os.environ if env is None else env
    raw = env.get("VPD_GPUH_CAP")
    if raw in (None, ""):
        return CAP_DEFAULT
    cap = float(raw)
    if not cap > 0:
        raise ValueError(f"VPD_GPUH_CAP must be > 0, got {raw!r}")
    return cap


def gate_caps(cap: Optional[float] = None) -> Dict[str, float]:
    cap = total_cap() if cap is None else float(cap)
    wb = cap - sum(RESERVE.values())
    if wb < 0:
        raise ValueError(f"CAP {cap} is below the tvr+e3vs reservation {sum(RESERVE.values())}")
    return {"wb": round(wb, 6), **RESERVE}


def gate_of(name: str) -> Optional[str]:
    """Gate implied by a vpd job name: vpd-wb-*, vpd-tvr-*, vpd-e3vs-*; None otherwise."""
    m = re.match(r"^vpd-(wb|tvr|e3vs)(-|$)", name or "")
    return m.group(1) if m else None


# --- file I/O -----------------------------------------------------------------------------------------------

def ledger_path(path=None) -> Path:
    if path is not None:
        return Path(path)
    return Path(os.environ.get("VPD_LEDGER") or paths.LEDGER)


@contextlib.contextmanager
def file_lock(lock: Path, timeout: float = LOCK_TIMEOUT_S):
    """Exclusive flock on `lock` (created if missing), polling until `timeout` seconds."""
    lock.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(lock), os.O_RDWR | os.O_CREAT, 0o644)
    deadline = time.monotonic() + timeout
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as e:
                if e.errno not in (errno.EAGAIN, errno.EACCES, errno.EWOULDBLOCK):
                    raise
                if time.monotonic() > deadline:
                    raise TimeoutError(f"could not lock {lock} within {timeout:.0f} s")
                time.sleep(0.2)
        yield
    finally:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)


def lock_path(path=None) -> Path:
    p = ledger_path(path)
    return p.with_name(p.name + ".lock")


def submit_lock_path(path=None) -> Path:
    """Held by vpd.sco submit across check + create + append, so two submits never pass the guard together."""
    p = ledger_path(path)
    return p.with_name(p.name + ".submit.lock")


def append(event: dict, path=None) -> dict:
    """Append one event line atomically (lock file + one write + fsync).  Returns the stored record."""
    kind = event.get("event")
    if kind not in EVENTS:
        raise ValueError(f"unknown ledger event {kind!r}")
    if not event.get("run"):
        raise ValueError("a ledger event needs 'run'")
    rec = dict(event)
    rec.setdefault("logged_at", iso())
    line = (json.dumps(rec, sort_keys=True) + "\n").encode()
    p = ledger_path(path)
    with file_lock(lock_path(path)):
        fd = os.open(str(p), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            os.write(fd, line)
            os.fsync(fd)
        finally:
            os.close(fd)
    return rec


def read(path=None) -> List[dict]:
    """All events; a missing ledger is empty.  A malformed line raises (never guess a budget)."""
    p = ledger_path(path)
    if not p.exists():
        return []
    out = []
    for i, line in enumerate(p.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except ValueError as e:
            raise ValueError(f"{p}:{i}: malformed ledger line ({e})") from None
        if not isinstance(rec, dict) or rec.get("event") not in EVENTS:
            raise ValueError(f"{p}:{i}: not a ledger event: {line[:120]}")
        out.append(rec)
    return out


# --- folding and accounting ---------------------------------------------------------------------------------

def fold(events: Iterable[dict]) -> Dict[str, dict]:
    """{run: job} in first-seen order.  job: name, run, gate, est, gpus, pt_id, status
    ('submitted' | 'uncertain' | 'cleared' | 'ended'), submitted_at, end (the end event), deleted_at."""
    jobs: Dict[str, dict] = {}
    for ev in events:
        run = str(ev["run"])
        kind = ev["event"]
        j = jobs.get(run)
        if kind in ("submit", "uncertain"):
            j = jobs.setdefault(run, {"run": run})
            j.update({"name": ev.get("name"), "gate": ev.get("gate"), "est": float(ev.get("est") or 0.0),
                      "gpus": int(ev.get("gpus") or 1)})
            if kind == "submit":
                j.update({"status": "submitted", "pt_id": ev.get("pt_id"), "submitted_at": ev.get("submitted_at")})
            else:
                j.update({"status": "uncertain", "pt_id": None, "submitted_at": ev.get("at")})
            j.pop("end", None)
        elif j is None:
            continue  # an end/delete of a run this ledger never submitted: ignored for the budget
        elif kind == "adopt":
            j.update({"status": "submitted", "pt_id": ev.get("pt_id")})
        elif kind == "cleared":
            if j.get("status") == "uncertain":
                j["status"] = "cleared"
        elif kind == "end":
            j["status"] = "ended"
            j["end"] = ev
            j["pt_id"] = j.get("pt_id") or ev.get("pt_id")
        elif kind == "delete":
            j["deleted_at"] = ev.get("at")
    return jobs


def is_open(job: dict) -> bool:
    return job.get("status") in ("submitted", "uncertain")


def charge(job: dict, elapsed_h: Optional[float] = None) -> float:
    """GPU-h a job counts against the budget: actual if ended, max(est_high, elapsed) if open, 0 if cleared."""
    st = job.get("status")
    if st == "ended":
        g = job["end"].get("gph")
        return float(g) if g is not None else float(job.get("est") or 0.0)
    if st in ("submitted", "uncertain"):
        return max(float(job.get("est") or 0.0), float(elapsed_h or 0.0))
    return 0.0


def summarise(events: Iterable[dict], cap: Optional[float] = None,
              elapsed: Optional[Dict[str, float]] = None) -> dict:
    """{"cap", "by_gate": {g: {cap, spent, committed, open_jobs, ended_jobs, remaining}}, "total": {...},
    "ungated": [runs without a known gate]}.  elapsed: {run: GPU-h so far} of open jobs."""
    caps = gate_caps(cap)
    elapsed = elapsed or {}
    by = {g: {"cap": caps[g], "spent": 0.0, "committed": 0.0, "open_jobs": 0, "ended_jobs": 0} for g in GATES}
    ungated = []
    for run, j in fold(events).items():
        g = j.get("gate")
        if g not in by:
            if is_open(j) or j.get("status") == "ended":
                ungated.append(run)
            continue
        c = charge(j, elapsed.get(run))
        if j.get("status") == "ended":
            by[g]["spent"] += c
            by[g]["ended_jobs"] += 1
        elif is_open(j):
            by[g]["committed"] += c
            by[g]["open_jobs"] += 1
    for g, row in by.items():
        row["spent"] = round(row["spent"], 4)
        row["committed"] = round(row["committed"], 4)
        row["remaining"] = round(row["cap"] - row["spent"] - row["committed"], 4)
    tot_cap = sum(caps.values())
    spent = round(sum(r["spent"] for r in by.values()), 4)
    committed = round(sum(r["committed"] for r in by.values()), 4)
    return {"cap": tot_cap, "by_gate": by, "ungated": ungated,
            "total": {"cap": tot_cap, "spent": spent, "committed": committed,
                      "remaining": round(tot_cap - spent - committed, 4)}}


def budget_check(events: Iterable[dict], gate: str, est_high: float, cap: Optional[float] = None,
                 elapsed: Optional[Dict[str, float]] = None) -> Tuple[bool, str]:
    """(ok, message) for submitting a new job of `gate` with estimate `est_high` (B2)."""
    if gate not in GATES:
        return False, f"unknown gate {gate!r}; one of {GATES}"
    if est_high is None or not float(est_high) > 0:
        return False, "a positive --est-gph (high estimate) is required"
    est_high = float(est_high)
    s = summarise(events, cap, elapsed)
    if s["ungated"]:
        return False, f"ledger has jobs without a gate: {s['ungated']}"
    row, tot = s["by_gate"][gate], s["total"]
    need_g = row["spent"] + row["committed"] + est_high
    need_t = tot["spent"] + tot["committed"] + est_high
    detail = (f"gate {gate}: spent {row['spent']:.2f} + committed {row['committed']:.2f} + new {est_high:.2f} = "
              f"{need_g:.2f} vs cap {row['cap']:.2f}; total {need_t:.2f} vs CAP {tot['cap']:.2f}")
    eps = 1e-9
    if need_g > row["cap"] + eps:
        return False, "BUDGET REFUSED: " + detail
    if need_t > tot["cap"] + eps:
        return False, "BUDGET REFUSED (total): " + detail
    return True, "budget ok: " + detail


def last_submit_ts(events: Iterable[dict]) -> Optional[float]:
    """Epoch seconds of the latest create attempt (submit or uncertain): field 'ts', else the ISO time."""
    out = []
    for e in events:
        if e["event"] not in ("submit", "uncertain"):
            continue
        if isinstance(e.get("ts"), (int, float)):
            out.append(float(e["ts"]))
        else:
            t = parse_utc(e.get("submitted_at") or e.get("at"))
            if t is not None:
                out.append(t.timestamp() + 1.0)  # ISO has 1 s resolution: round up
    return max(out) if out else None


# --- actual GPU-h from a worker log -------------------------------------------------------------------------

_LAUNCH_START = re.compile(r"^\[launch\] host=\S* date=(\S+)", re.M)
_LAUNCH_END = re.compile(r"^\[launch\] rc=\d+ date=(\S+)", re.M)


def worker_log_span(log: Path) -> Optional[Tuple[str, str]]:
    """(start, end) of the first '[launch] host=... date=' and last '[launch] rc=... date=' lines of
    RUN/job/worker.log (the downstream line's way of counting GPU-h), None if either is missing."""
    try:
        text = Path(log).read_text(errors="replace")
    except OSError:
        return None
    a, b = _LAUNCH_START.findall(text), _LAUNCH_END.findall(text)
    if not a or not b:
        return None
    return a[0], b[-1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="VPD GPU-h ledger")
    ap.add_argument("action", choices=["summary", "jobs"])
    ap.add_argument("--ledger", type=Path, default=None)
    args = ap.parse_args(argv)
    events = read(args.ledger)
    if args.action == "summary":
        print(json.dumps(summarise(events), indent=1))
    else:
        for run, j in fold(events).items():
            print(json.dumps({k: v for k, v in j.items() if k != "end"} | {"charge": charge(j)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
