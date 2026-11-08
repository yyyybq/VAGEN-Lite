"""WalkerBench smoke + pilot orchestration (WP-WBPILOT; plan 3.6; DECISIONS B2, B3, B5, B7, B9, B17).

    PY_EVAL -m vpd.walkerbench.pilot plan                      # jobs, payloads, estimates, budget, prerequisites
    PY_EVAL -m vpd.walkerbench.pilot prepare JOB [--pool zoetrope|h800|auto] [--new] [--redecide]
    PY_EVAL -m vpd.walkerbench.pilot submit JOB                # stop-and-ask: only after the user's OK
    PY_EVAL -m vpd.walkerbench.pilot status
    PY_EVAL -m vpd.walkerbench.pilot collect [JOB ...]         # verify the expected outputs -> RUN/collect.json
    PY_EVAL -m vpd.walkerbench.pilot delete JOB [--allow-incomplete]   # the sco job record, after collect
    PY_EVAL -m vpd.walkerbench.pilot budget
  in-job helpers (called by vpd/jobs/wb_smoke.sh and wb_pilot_start.sh):
    ... pilot adapter-ok DIR [--init 0W_DIR]                   # exit 0 complete, 1 missing / retrain, 3 other code
    ... pilot eval-ok EVAL_DIR [--adapter ADAPTER_DIR]         # exit 0 fresh results.json, 1 none, 3 stale
    ... pilot render-parity --out F.json                       # B17: one JPEG from PY_EVAL and from PY_Q3
    ... pilot render-probe --copy WB --pano P --heading H --out-dir D   (one interpreter; used by render-parity)
    PY_Q3 -m vpd.walkerbench.pilot vllm-parity --start Q3S --dry-run-json D/dry_run.json --out F.json   # B5
    ... pilot smoke-verdict --smoke SMOKE_DIR                  # -> SMOKE_DIR/smoke_verdict.json

Jobs (one card each, gate wb, RUN = OUT/jobs/<job>-<yyyymmddHHMM>, every sco call through vpd/sco.py):
    wb-demos  vpd-wb-demos  non-root  timeout -k 60 780 bash vpd/jobs/wb_demos.sh RUN                 est 0.2-0.3
    wb-smoke  vpd-wb-smoke  as_root   env VPD_SMOKE_DEADLINE_S=3120 timeout -k 60 3300 bash vpd/jobs/wb_smoke.sh RUN
                                                                                                 est 0.5-1.0
    wb-q3b    vpd-wb-q3b    as_root   env VPD_OPTIONAL=0|1 VPD_JOB_DEADLINE_S=11760 timeout -k 120 11940
                                      bash vpd/jobs/wb_pilot_start.sh Q3B RUN
    wb-q3s / wb-e2 the same for Q3S / E2                                       est 2.05-3.4 (+0.3-0.5 optional)
The pilot estimates are recomputed from the smoke's measurements (smoke_verdict.json "measured") once it exists.
Timeouts (job_timeouts) follow the est_high that is submitted, so the B2 charge is bounded by it: outer =
floor(est_high*3600) - 300 s, kill window 120 s (pilot) or 60 s (smoke, demos), so outer + kill <= est_high*3600
with >= 180 s left for container start / teardown; the payload's soft deadline = outer - 180 s (smoke and pilot),
so the last steps are cut by the job itself and its cleanup (merged weights, df_end, chown) still runs.  The
numbers above are the plan estimates; prepare records the ones it used in RUN/pilot.json "timeouts".

Guards:
- prepare (every job) first creates OUT/walkerbench/runs/{smoke,pilot} as the login user, so the as_root jobs
  never create (and own) these shared parents.
- prepare wb-q3* refuses unless the newest smoke_verdict.json says PASS and vpd/PREREG_WB_PILOT.md is committed and
  clean (`git log -1 -- FILE` non-empty and `git status --porcelain -- FILE` empty; only that file is checked).
- Optional 0W evals (VPD_OPTIONAL): decided once for all three starts (OUT/walkerbench/runs/pilot/
  optional_decision.json) from the post-smoke projection = WB spent + committed (ledger) + est_high of the pilot
  jobs not yet in the ledger.  Optional evals run only if the projection with them is <= the WB cap; they
  are dropped first (DECISIONS B2, PREREG 10.1).  If the mandatory-only projection exceeds the WB cap (12.0) the
  job is not prepared or submitted and nothing mandatory is dropped: stop and hand to the user.
- submit re-checks the prerequisites (and that the PREREG commit is the one recorded at prepare); the B2 budget
  check, disk and concurrency guards are vpd.sco's.
- delete only after collect has verified the expected outputs (RUN/collect.json ok, files re-checked), or with
  --allow-incomplete for a job that ended without them; vpd.sco.delete only deletes terminal jobs.
"""

from __future__ import annotations

import argparse
import base64
import datetime
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from vpd import paths

GATE = "wb"
STARTS = ("Q3B", "Q3S", "E2")
SEED = 1001
PREREG = "vpd/PREREG_WB_PILOT.md"
JOBS_DIR = paths.VPD_PKG / "jobs"
STAMP_RE = re.compile(r"^(?P<job>[a-z0-9-]+)-(?P<stamp>\d{12})$")
# Job time limits from the submitted est_high (audit 2026-11-08: the timeouts must bound the B2 charge):
# outer = floor(est_high*3600) - HEADROOM_S; kill window KILL_S; soft deadline = outer - SOFT_RESERVE_S.
HEADROOM_S = 300
SOFT_RESERVE_S = 180
KILL_S = {"wb-demos": 60, "wb-smoke": 60, "pilot": 120}
MIN_OUTER_S = 600  # an estimate this small cannot hold a job: refuse rather than submit a job that is cut at once
# Extra ceiling for the optional evals below the WB cap; None = the WB cap only, as DECISIONS B2 and PREREG 10.1
# (the WP spec's 9 GPU-h rule is not in the registration, so it is not applied)
OPTIONAL_MAX_PROJECTION: Optional[float] = None
N_TEST = 179
TIERS = ("0W", "50", "200")
PHASE = {"0W": "0W", "50": "tier50", "200": "tier200"}
RES = {"640": "640x400", "1280": "1280x800"}
TOKENS_PER_IMAGE = {"640": 242, "1280": 1002}
# the pilot evals of one start (PREREG 7.2): (tier, res, mode, mandatory)
PILOT_EVALS = (("raw", "640", "empty", True), ("raw", "1280", "empty", True), ("50", "640", "empty", True),
               ("200", "640", "empty", True), ("200", "640", "off", True), ("0W", "640", "empty", False))
# the smoke evals (--limit 6): key -> (model, tier, res, mode, history_turns)
SMOKE_EVALS = {"a": ("Q3B", "raw", "640", "empty", 0), "b": ("Q3B", "raw", "1280", "empty", 0),
               "c": ("Q3S", "0W", "640", "empty", 0), "d": ("Q3B", "raw", "640", "off", 0),
               "e": ("Q3B", "raw", "640", "empty", 2)}
SMOKE_LIMIT = 6
MAX_IMAGES = 80

# plan section 6 (GPU-h low, high), used until a smoke verdict has measurements.  wb-smoke high is 1.0 (plan: 0.8)
# so that its outer timeout (3300 s) and soft deadline (3120 s) keep about the time the smoke was designed for
# (6 vLLM starts, a train probe, a 0W training and a merge); audit 2026-11-08.
PLAN_EST = {"wb-demos": (0.2, 0.3), "wb-smoke": (0.5, 1.0), "pilot": (2.05, 3.4), "optional": (0.3, 0.5)}
# cost model from the smoke measurements (see estimate_from_smoke)
COST = {"overhead_h": (0.05, 0.1),  # job start-up, node facts, imports
        "train_high_factor": 1.25,  # training / merge seconds x this for the high estimate
        "steps_high_factor": 1.5}  # eval steps: 1.5 x the smoke's largest mean steps, at most the step-cap total


class Refused(SystemExit):
    """A pilot action was refused (exit 2); the message says why."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


def utc() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp(now: Optional[datetime.datetime] = None) -> str:
    return (now or datetime.datetime.now(datetime.timezone.utc)).strftime("%Y%m%d%H%M")


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(p) -> Optional[dict]:
    try:
        return json.loads(Path(p).read_text())
    except (OSError, ValueError):
        return None


def write_json(p, obj) -> Path:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str) + "\n")
    os.replace(tmp, p)
    return p


# --- layout -------------------------------------------------------------------------------------------------

class Layout:
    """Every path the pilot reads or writes, under one output root (paths.OUT; tests use a scratch root)."""

    def __init__(self, root: Path = paths.OUT):
        self.root = Path(root)
        self.jobs = self.root / "jobs"
        wb = self.root / "walkerbench"
        self.wb = wb
        self.demos_dir = wb / "demos" / "s2train_640x400"
        self.demos_all = self.demos_dir / "demos_all.jsonl"
        self.demos_done = self.demos_dir / "DONE"
        self.demos_manifest = self.demos_dir / "manifest.json"
        self.warmup = wb / "warmup" / "warmup20.jsonl"
        self.warmup_manifest = wb / "warmup" / "manifest.json"
        self.tiers_root = wb / "tiers"
        self.split = wb / "split" / "split_s2.json"
        self.s2test_tasks = wb / "split" / "s2test" / "tasks"
        self.baselines = wb / "baselines" / "baselines_s2test.json"
        self.smoke_root = wb / "runs" / "smoke"
        self.pilot_root = wb / "runs" / "pilot"
        self.optional_decision = self.pilot_root / "optional_decision.json"

    def tier(self, n, seed: int = SEED) -> Path:
        return self.tiers_root / f"seed{seed}" / f"demos_{n}.jsonl"

    def order(self, seed: int = SEED) -> Path:
        return self.tiers_root / f"order_seed{seed}.json"

    def smoke_dir(self, run) -> Path:
        return self.smoke_root / Path(run).name

    def start_dir(self, start: str) -> Path:
        return self.pilot_root / start

    def adapter(self, start: str, tier: str) -> Path:
        return self.start_dir(start) / "adapters" / tier

    def eval_name(self, start: str, tier: str, res: str, mode: str) -> str:
        return f"pilot__{start}__{tier}__{res}__{mode}"

    def eval_dir(self, start: str, tier: str, res: str, mode: str) -> Path:
        return self.start_dir(start) / "evals" / self.eval_name(start, tier, res, mode)


LAYOUT = Layout()


# --- job table ----------------------------------------------------------------------------------------------

JOBS = {
    "wb-demos": {"name": "vpd-wb-demos", "script": "wb_demos.sh", "as_root": False, "start": None},
    "wb-smoke": {"name": "vpd-wb-smoke", "script": "wb_smoke.sh", "as_root": True, "start": None},
    "wb-q3b": {"name": "vpd-wb-q3b", "script": "wb_pilot_start.sh", "as_root": True, "start": "Q3B"},
    "wb-q3s": {"name": "vpd-wb-q3s", "script": "wb_pilot_start.sh", "as_root": True, "start": "Q3S"},
    "wb-e2": {"name": "vpd-wb-e2", "script": "wb_pilot_start.sh", "as_root": True, "start": "E2"},
}
PILOT_JOBS = ("wb-q3b", "wb-q3s", "wb-e2")


def job_spec(job: str) -> dict:
    if job not in JOBS:
        raise Refused(f"unknown job {job!r}; one of {sorted(JOBS)}")
    return JOBS[job]


def job_timeouts(job: str, est_high: float) -> dict:
    """{est_high, outer_s, kill_s, soft_deadline_s} for a job submitted at est_high GPU-h: the outer timeout plus
    its kill window end HEADROOM_S - kill_s (>= 180 s) before est_high, so the ledger charge stays within the
    estimate; the soft deadline (smoke, pilot) is SOFT_RESERVE_S before the outer timeout."""
    job_spec(job)
    hi = float(est_high)
    if not math.isfinite(hi) or hi <= 0:
        raise Refused(f"{job}: est_high {est_high!r} must be a positive number")
    budget_s = int(math.floor(round(hi * 3600, 6)))
    outer = budget_s - HEADROOM_S
    kill = KILL_S["pilot" if job in PILOT_JOBS else job]
    if outer < MIN_OUTER_S:
        raise Refused(f"{job}: est_high {hi} GPU-h leaves an outer timeout of {outer} s (< {MIN_OUTER_S} s)")
    assert outer + kill <= hi * 3600
    soft = None if job == "wb-demos" else outer - SOFT_RESERVE_S
    return {"est_high": round(hi, 3), "outer_s": outer, "kill_s": kill, "soft_deadline_s": soft,
            "rule": f"outer = floor(est_high*3600) - {HEADROOM_S}; soft = outer - {SOFT_RESERVE_S}"}


def payload_for(job: str, run, optional: Optional[int] = None, est_high: Optional[float] = None) -> List[str]:
    """The sco payload; its time limits come from est_high (job_timeouts), the GPU-h the job is submitted at."""
    spec = job_spec(job)
    script = str(JOBS_DIR / spec["script"])
    run = str(run)
    if est_high is None:
        raise ValueError("payload_for needs the est_high the job is submitted at (its timeouts follow it)")
    t = job_timeouts(job, est_high)
    lim = ["timeout", "-k", str(t["kill_s"]), str(t["outer_s"])]
    if job == "wb-demos":
        return lim + ["bash", script, run]
    if job == "wb-smoke":
        return ["env", f"VPD_SMOKE_DEADLINE_S={t['soft_deadline_s']}"] + lim + ["bash", script, run]
    if optional not in (0, 1):
        raise ValueError("a pilot payload needs optional 0 or 1")
    return (["env", f"VPD_OPTIONAL={optional}", f"VPD_JOB_DEADLINE_S={t['soft_deadline_s']}"] + lim
            + ["bash", script, spec["start"], run])


# --- runs and their state -----------------------------------------------------------------------------------

def runs_of(job: str, L: Layout = None) -> List[Path]:
    L = L or LAYOUT
    if not L.jobs.is_dir():
        return []
    out = []
    for p in L.jobs.iterdir():
        m = STAMP_RE.match(p.name)
        if m and m.group("job") == job and p.is_dir():
            out.append(p)
    return sorted(out, key=lambda p: p.name)


def run_state(run: Path) -> str:
    """'submitted' (job/submission.json), 'uncertain' (an unresolved create), 'prepared', or 'empty'."""
    job = Path(run) / "job"
    if (job / "submission.json").exists():
        return "submitted"
    if (job / "submit_uncertain.json").exists():
        return "uncertain"
    if (job / "prepared.json").exists():
        return "prepared"
    return "empty"


def latest_run(job: str, states=None, L: Layout = None) -> Optional[Path]:
    for r in reversed(runs_of(job, L)):
        if states is None or run_state(r) in states:
            return r
    return None


def ledger_jobs(events=None) -> Dict[str, dict]:
    """{run: folded ledger job} of the wb gate."""
    from vpd import ledger

    events = ledger.read() if events is None else events
    return {r: j for r, j in ledger.fold(events).items() if j.get("gate") == GATE}


def job_in_ledger(job: str, events=None) -> Optional[dict]:
    """The newest ledger entry of this job that is open or ended (a cleared uncertain create does not count)."""
    name = job_spec(job)["name"]
    hit = None
    for j in ledger_jobs(events).values():
        if j.get("name") == name and j.get("status") in ("submitted", "uncertain", "ended"):
            hit = j
    return hit


def outputs_complete(job: str, L: Layout = None) -> bool:
    """The job's products exist: demos DONE; a smoke verdict; a pilot start's expected outputs (latest run)."""
    L = L or LAYOUT
    if job == "wb-demos":
        return L.demos_done.exists()
    if job == "wb-smoke":
        return bool(smoke_verdicts(L))
    r = latest_run(job, ("submitted",), L)
    return r is not None and all(_present(p) for p in expected_outputs(job, r, L))


# --- prerequisites ------------------------------------------------------------------------------------------

def _git(*args) -> Optional[str]:
    try:
        res = subprocess.run(["git", "-c", "safe.directory=*", "-C", str(paths.WORKTREE), *args],
                             capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return res.stdout.strip() if res.returncode == 0 else None


def prereg_state() -> dict:
    """{committed, clean, commit, status}: PREREG_WB_PILOT.md committed (git log -1 non-empty) and without
    uncommitted changes (git status --porcelain -- FILE empty).  Only this file is checked."""
    log = _git("log", "-1", "--format=%H", "--", PREREG)
    st = _git("status", "--porcelain", "--", PREREG)
    return {"file": PREREG, "commit": log or None, "committed": bool(log), "status": st,
            "clean": st == "", "ok": bool(log) and st == ""}


def smoke_verdicts(L: Layout = None) -> List[Path]:
    L = L or LAYOUT
    if not L.smoke_root.is_dir():
        return []
    return sorted((p / "smoke_verdict.json" for p in L.smoke_root.iterdir()
                   if STAMP_RE.match(p.name) and (p / "smoke_verdict.json").is_file()), key=lambda p: p.parent.name)


def latest_verdict(L: Layout = None) -> Tuple[Optional[Path], Optional[dict]]:
    vs = smoke_verdicts(L)
    if not vs:
        return None, None
    return vs[-1], read_json(vs[-1])


def prerequisites(job: str, L: Layout = None) -> List[str]:
    """Problems that block preparing/submitting `job` (empty list = ready)."""
    L = L or LAYOUT
    bad = []
    if job == "wb-demos":
        for p in (L.split, L.order(SEED)):
            if not p.is_file():
                bad.append(f"missing {p} (WP-WBDATA)")
        if L.demos_done.exists():
            bad.append(f"{L.demos_done} exists: the demos are already built")
        return bad
    need = [L.demos_done, L.demos_all, L.warmup, L.tier(50), L.tier(200)]
    for p in need:
        if not p.is_file():
            bad.append(f"missing {p} (job wb-demos)")
    if job == "wb-smoke":
        return bad
    vp, v = latest_verdict(L)
    if v is None:
        bad.append(f"no smoke_verdict.json under {L.smoke_root} (job wb-smoke)")
    elif v.get("verdict") != "PASS":
        bad.append(f"{vp}: verdict {v.get('verdict')!r}, not PASS ({', '.join(v.get('failed') or [])})")
    if not L.baselines.is_file():
        bad.append(f"missing {L.baselines} (WP-WBDATA)")
    ps = prereg_state()
    if not ps["committed"]:
        bad.append(f"{PREREG} is not committed (git log -1 -- {PREREG} is empty)")
    elif not ps["clean"]:
        bad.append(f"{PREREG} has uncommitted changes: {ps['status']!r}")
    return bad


# --- estimates ----------------------------------------------------------------------------------------------

def _num(x) -> Optional[float]:
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) else None


def estimate_from_smoke(m: dict) -> dict:
    """Per-start pilot GPU-h from the smoke measurements m (smoke_verdict.json "measured"):
      train  0W   = the smoke's Q3S 0W load + train seconds (same data and recipe for every start);
             N    = probe load seconds + tokens(tier N, one epoch) x epochs(N) / probe tokens per second;
      merge       = the smoke's merge seconds;
      eval cfg    = server ready seconds + steps x seconds per step of the matching smoke eval, steps low =
                    179 x that eval's mean steps, high = min(step-cap total, 1.5 x 179 x the largest smoke mean);
      high        = training and merges x 1.25, evals at the high step count, plus the job overhead.
    The smoke ran 6 tasks on 8 workers, so its seconds per step are a pessimistic stand-in for 179 tasks."""
    need = ("probe_tokens_per_s", "probe_load_s", "zeroW_s", "tier_tokens", "tier_epochs", "merge_s", "evals",
            "step_cap_total")
    missing = [k for k in need if m.get(k) in (None, {}, [])]
    if missing:
        raise ValueError(f"smoke measurements incomplete: {missing}")
    tps = _num(m["probe_tokens_per_s"])
    if not tps or tps <= 0:
        raise ValueError("probe tokens/s must be > 0")
    ev = m["evals"]
    for k in ("640_empty", "1280_empty", "640_off", "640_merged"):
        e = ev.get(k) or {}
        if _num(e.get("sps")) is None or _num(e.get("mean_steps")) is None:
            raise ValueError(f"smoke eval {k} lacks sps / mean_steps")
    n = int(m.get("n_tasks") or N_TEST)
    max_mean = max(float(ev[k]["mean_steps"]) for k in ("640_empty", "1280_empty", "640_off", "640_merged"))
    steps_high = min(float(m["step_cap_total"]), COST["steps_high_factor"] * n * max_mean)

    def eval_s(k: str) -> Tuple[float, float]:
        e = ev[k]
        ready = float(_num(e.get("ready_s")) or 0.0)
        return ready + n * float(e["mean_steps"]) * float(e["sps"]), ready + steps_high * float(e["sps"])

    train = {"0W": float(m["zeroW_s"])}
    for t in ("50", "200"):
        train[t] = float(m["probe_load_s"]) + float(m["tier_tokens"][t]) * int(m["tier_epochs"][t]) / tps
    merge = float(m["merge_s"])
    f = COST["train_high_factor"]
    mand_evals = [eval_s("640_empty"), eval_s("640_merged"), eval_s("640_off"), eval_s("640_merged"),
                  eval_s("1280_empty")]  # raw640, 200c, 200u, 50c, raw1280
    mand_train = sum(train.values()) + 2 * merge  # merges of 200 and 50
    lo = mand_train + sum(a for a, _ in mand_evals)
    hi = f * mand_train + sum(b for _, b in mand_evals)
    o_lo, o_hi = COST["overhead_h"]
    opt = eval_s("640_merged")
    return {"source": "smoke", "per_start": {
        "mandatory": (round(lo / 3600 + o_lo, 3), round(hi / 3600 + o_hi, 3)),
        "optional": (round((merge + opt[0]) / 3600, 3), round((f * merge + opt[1]) / 3600, 3))},
        "parts_s": {"train": {k: round(v, 1) for k, v in train.items()}, "merge": round(merge, 1),
                    "evals": {k: [round(a, 1), round(b, 1)] for k, (a, b) in
                              zip(("raw640", "200c", "200u", "50c", "raw1280"), mand_evals)},
                    "steps_high": round(steps_high, 1)}, "model": dict(COST)}


def estimates(L: Layout = None) -> dict:
    """{"source": plan|smoke, "jobs": {job: (low, high)}, "optional": {job: (low, high)}, ...}."""
    vp, v = latest_verdict(L)
    out = {"source": "plan", "jobs": {"wb-demos": PLAN_EST["wb-demos"], "wb-smoke": PLAN_EST["wb-smoke"]},
           "optional": {}, "verdict": str(vp) if vp else None}
    per = {"mandatory": PLAN_EST["pilot"], "optional": PLAN_EST["optional"]}
    if v and v.get("measured"):
        try:
            e = estimate_from_smoke(v["measured"])
            per = e["per_start"]
            out.update(source="smoke", parts_s=e["parts_s"], model=e["model"])
        except (ValueError, KeyError, TypeError) as err:
            out["smoke_error"] = str(err)
    for j in PILOT_JOBS:
        out["jobs"][j] = tuple(per["mandatory"])
        out["optional"][j] = tuple(per["optional"])
    return out


def projection(est: dict, events=None, cap: Optional[float] = None, L: Layout = None) -> dict:
    """WB spent + committed (ledger) + est_high of the WB jobs not yet in the ledger (or collected), with and
    without the optional evals of the pending pilot jobs."""
    from vpd import ledger

    events = ledger.read() if events is None else events
    s = ledger.summarise(events, cap)
    wb = s["by_gate"][GATE]
    pending = []
    for j in JOBS:
        led = job_in_ledger(j, events)
        if led is not None and led.get("status") in ("submitted", "uncertain"):
            continue  # already in committed
        if not outputs_complete(j, L):
            pending.append(j)  # never submitted, or ended without its outputs (a rerun is still to pay)
    mand = wb["spent"] + wb["committed"] + sum(est["jobs"][j][1] for j in pending)
    opt = sum(est["optional"].get(j, (0, 0))[1] for j in pending if j in PILOT_JOBS)
    return {"wb_cap": wb["cap"], "spent": wb["spent"], "committed": wb["committed"], "pending": pending,
            "mandatory_high": round(mand, 3), "with_optional_high": round(mand + opt, 3),
            "optional_max": round(wb["cap"] if OPTIONAL_MAX_PROJECTION is None
                                     else min(OPTIONAL_MAX_PROJECTION, wb["cap"]), 3), "estimate_source": est["source"]}


def decide_optional(proj: dict) -> dict:
    """{optional 0|1, stop bool, reason} from a post-smoke projection (B2, PREREG 10.1)."""
    if proj["mandatory_high"] > proj["wb_cap"] + 1e-9:
        return {"optional": 0, "stop": True,
                "reason": (f"STOP, hand to user: the mandatory evals alone project {proj['mandatory_high']:.2f} "
                           f"GPU-h > WB reserve {proj['wb_cap']:.2f}; nothing is submitted and no mandatory eval "
                           f"is dropped (PREREG 10.1)")}
    if proj["with_optional_high"] <= proj["optional_max"] + 1e-9:
        return {"optional": 1, "stop": False,
                "reason": f"with optional {proj['with_optional_high']:.2f} <= {proj['optional_max']:.2f}"}
    return {"optional": 0, "stop": False,
            "reason": (f"VPD_OPTIONAL=0: with optional {proj['with_optional_high']:.2f} > {proj['optional_max']:.2f};"
                       f" mandatory {proj['mandatory_high']:.2f} <= {proj['wb_cap']:.2f}")}


def optional_decision(L: Layout = None, redecide: bool = False, events=None, write: bool = True) -> dict:
    """The one optional decision for all three starts (written once; --redecide recomputes it)."""
    L = L or LAYOUT
    old = read_json(L.optional_decision)
    est = estimates(L)
    proj = projection(est, events, L=L)
    if old and not redecide:
        d = dict(old)
        # the stop rule is re-checked every time; the optional choice itself stays fixed for symmetry
        now = decide_optional(proj)
        d["stop"], d["stop_reason"], d["projection_now"] = now["stop"], now["reason"] if now["stop"] else None, proj
        return d
    if est["source"] != "smoke":
        raise Refused("no smoke measurements yet: the optional decision needs a PASS smoke_verdict.json")
    d = decide_optional(proj)
    d.update(utc=utc(), projection=proj, estimates=est, verdict=est.get("verdict"))
    d["stop_reason"] = d["reason"] if d["stop"] else None
    if write and not d["stop"]:
        write_json(L.optional_decision, d)
    return d


# --- prepare / submit / collect / delete -------------------------------------------------------------------

def prepare(job: str, pool: str = "zoetrope", new: bool = False, redecide: bool = False,
            now: Optional[datetime.datetime] = None, L: Layout = None) -> Path:
    from vpd import sco as vsco

    L = L or LAYOUT
    spec = job_spec(job)
    # The shared parents are created here, as the login user, before any as_root job runs: a root job's mkdir -p
    # would leave runs/ (and runs/smoke, runs/pilot) owned by root, and the login side could then not write
    # runs/pilot/optional_decision.json or GATE.json (the job's chown trap covers only its own dirs).
    for d in (L.smoke_root, L.pilot_root):
        d.mkdir(parents=True, exist_ok=True)
    last = latest_run(job, L=L)
    if last is not None:
        st = run_state(last)
        if st == "prepared" and not new:
            raise Refused(f"{last} is prepared and not submitted: submit it, or pass --new for another run dir")
        if st == "uncertain":
            raise Refused(f"{last}: an earlier create is uncertain; run `python -m vpd.sco resolve --run {last}`")
        led = job_in_ledger(job)
        if led is not None and led.get("status") in ("submitted", "uncertain"):
            raise Refused(f"{job} is open in the ledger ({led.get('pt_id')}, {led['run']})")
    bad = prerequisites(job, L)
    if bad:
        raise Refused(f"{job} not ready:\n  - " + "\n  - ".join(bad))
    est = estimates(L)
    info = {"job": job, "utc": utc(), "estimates": est}
    optional = None
    if job in PILOT_JOBS:
        d = optional_decision(L, redecide=redecide)
        if d.get("stop"):
            raise Refused(d["stop_reason"])
        optional = int(d["optional"])
        info.update(optional=optional, optional_decision=d, prereg=prereg_state())
        vp, v = latest_verdict(L)
        info["smoke_verdict"] = {"path": str(vp), "sha256": sha256_file(vp), "verdict": v.get("verdict")}
    lo, hi = est["jobs"][job]
    if optional:
        hi = round(hi + est["optional"][job][1], 3)
        lo = round(lo + est["optional"][job][0], 3)
    info["est_gph"] = [lo, hi]
    run = L.jobs / f"{job}-{stamp(now)}"
    if run.exists():
        raise Refused(f"{run} exists (prepared twice in one minute?)")
    info["timeouts"] = job_timeouts(job, hi)  # outer + kill window <= the est_high charged at submit (B2)
    payload = payload_for(job, run, optional, hi)
    info["payload"] = payload
    launch = vsco.prepare(run, spec["name"], payload, pool=pool, as_root=spec["as_root"], est_gph=hi, gate=GATE)
    write_json(run / "pilot.json", info)
    return Path(launch)


def submit(job: str, L: Layout = None) -> str:
    """Stop-and-ask in this run of the line: the orchestrator calls it only after the user's OK."""
    from vpd import sco as vsco

    L = L or LAYOUT
    run = latest_run(job, ("prepared",), L)
    if run is None:
        raise Refused(f"no prepared, unsubmitted run of {job}: run `pilot prepare {job}` first")
    newer = latest_run(job, L=L)
    if newer != run:
        raise Refused(f"the newest run of {job} is {newer} ({run_state(newer)}), not the prepared {run}")
    bad = prerequisites(job, L)
    if bad:
        raise Refused(f"{job} not ready:\n  - " + "\n  - ".join(bad))
    info = read_json(run / "pilot.json") or {}
    if job in PILOT_JOBS:
        ps = prereg_state()
        if (info.get("prereg") or {}).get("commit") != ps["commit"]:
            raise Refused(f"{PREREG} commit changed since prepare ({(info.get('prereg') or {}).get('commit')} -> "
                          f"{ps['commit']}): prepare again")
        d = optional_decision(L)
        if d.get("stop"):
            raise Refused(d["stop_reason"])
        if int(d["optional"]) != info.get("optional"):
            raise Refused(f"optional decision {d['optional']} differs from the prepared {info.get('optional')}")
    return vsco.submit(run)


def expected_outputs(job: str, run: Path, L: Layout = None) -> List[Path]:
    L = L or LAYOUT
    if job == "wb-demos":
        out = [L.demos_done, L.demos_all, L.demos_manifest, L.warmup, L.warmup_manifest]
        out += [L.tier(t, s) for s in (SEED, 1, 2, 3) for t in (25, 50, 100, 200, "all")]
        return out + [Path(run) / "DONE"]
    if job == "wb-smoke":
        return [L.smoke_dir(run) / "smoke_verdict.json"]
    start = job_spec(job)["start"]
    info = read_json(Path(run) / "pilot.json") or {}
    out = [L.adapter(start, t) / f for t in TIERS for f in ("adapter.pt", "meta.json")]
    for tier, res, mode, mand in PILOT_EVALS:
        if mand or info.get("optional") == 1:
            out.append(L.eval_dir(start, tier, res, mode) / "results.json")
    return out + [Path(run) / "DONE"]


def _present(p: Path) -> bool:
    if p.name == "results.json":
        return p.is_file() or (p.parent / "results_invalid.json").is_file()
    return p.is_file()


def collect(job: str, L: Layout = None) -> dict:
    L = L or LAYOUT
    run = latest_run(job, ("submitted",), L)
    if run is None:
        raise Refused(f"no submitted run of {job}")
    exp = expected_outputs(job, run, L)
    missing = [str(p) for p in exp if not _present(p)]
    invalid = [str(p.parent) for p in exp if p.name == "results.json" and not p.is_file()
               and (p.parent / "results_invalid.json").is_file()]
    steps = []
    sj = run / "steps.jsonl"
    if sj.is_file():
        for line in sj.read_text().splitlines():
            try:
                steps.append(json.loads(line))
            except ValueError:
                pass
    rep = {"job": job, "run": str(run), "utc": utc(), "ok": not missing, "expected": [str(p) for p in exp],
           "missing": missing, "invalid_evals": invalid,
           "failed_steps": [s.get("name") for s in steps if s.get("rc") not in (0, None)],
           "steps": len(steps), "exit_code": (run / "job" / "exit_code").read_text().strip()
           if (run / "job" / "exit_code").is_file() else None}
    if job == "wb-smoke":
        v = read_json(L.smoke_dir(run) / "smoke_verdict.json")
        rep["verdict"] = v.get("verdict") if v else None
        rep["verdict_failed"] = (v or {}).get("failed")
        rep["estimates"] = estimates(L)
    write_json(run / "collect.json", rep)
    return rep


def delete(job: str, allow_incomplete: bool = False, L: Layout = None) -> str:
    from vpd import sco as vsco

    L = L or LAYOUT
    run = latest_run(job, ("submitted",), L)
    if run is None:
        raise Refused(f"no submitted run of {job}")
    c = read_json(run / "collect.json")
    if c is None:
        raise Refused(f"{run}/collect.json missing: run `pilot collect {job}` first")
    if not c.get("ok") and not allow_incomplete:
        raise Refused(f"collect found {len(c.get('missing') or [])} missing outputs of {job} (first: "
                      f"{(c.get('missing') or ['?'])[0]}); pass --allow-incomplete to delete the record anyway")
    if c.get("ok"):
        gone = [p for p in c.get("expected") or [] if not _present(Path(p))]
        if gone:
            raise Refused(f"outputs vanished since collect: {gone[:3]}; collect again")
    return vsco.delete(run)


def status(L: Layout = None) -> dict:
    from vpd import sco as vsco

    L = L or LAYOUT
    st = vsco.status(include_ended=True)
    rows = [r for r in st["jobs"] if r.get("gate") == GATE]
    runs = {}
    for job in JOBS:
        r = latest_run(job, L=L)
        if r is None:
            runs[job] = None
            continue
        c = read_json(r / "collect.json")
        steps = (r / "steps.jsonl").read_text().splitlines()[-3:] if (r / "steps.jsonl").is_file() else []
        runs[job] = {"run": str(r), "state": run_state(r), "collected_ok": None if c is None else c.get("ok"),
                     "done": (r / "DONE").exists(), "last_steps": [json.loads(s) for s in steps if s.strip()]}
    return {"ledger_jobs": rows, "not_in_ledger": st.get("not_in_ledger"), "runs": runs,
            "budget_wb": st["budget"]["by_gate"][GATE]}


def budget(L: Layout = None) -> dict:
    from vpd import ledger

    L = L or LAYOUT
    events = ledger.read()
    est = estimates(L)
    proj = projection(est, events, L=L)
    return {"ledger": ledger.summarise(events), "estimates": est, "projection": proj,
            "optional_preview": decide_optional(proj) if est["source"] == "smoke" else None,
            "optional_decision": read_json(L.optional_decision)}


def plan(L: Layout = None, out=None) -> dict:
    L = L or LAYOUT
    out = out or sys.stdout
    est = estimates(L)
    try:
        b = budget(L)
    except Exception as e:  # an unreadable ledger must not hide the plan
        b = {"error": repr(e)}
    p = lambda *a: print(*a, file=out)  # noqa: E731
    p(f"VPD WalkerBench pilot plan ({utc()}); estimates from {est['source']}"
      + (f" ({est['verdict']})" if est.get("verdict") else ""))
    for job, spec in JOBS.items():
        run = L.jobs / f"{job}-<yyyymmddHHMM>"
        opt = 0 if job in PILOT_JOBS else None
        lo, hi = est["jobs"][job]
        extra = ""
        if job in PILOT_JOBS:
            olo, ohi = est["optional"][job]
            extra = f" (+{olo:.2f}-{ohi:.2f} optional 0W eval)"
        p(f"\n[{job}] sco name downstream-{spec['name']}, {'as_root' if spec['as_root'] else 'non-root'}, "
          f"gate {GATE}, est {lo:.2f}-{hi:.2f} GPU-h{extra}")
        p("  payload: " + " ".join(payload_for(job, run, opt, hi)).replace("VPD_OPTIONAL=0", "VPD_OPTIONAL=<0|1>")
          + (" (timeouts from the mandatory est_high; with optional evals they follow the larger est_high)"
             if job in PILOT_JOBS else ""))
        last = latest_run(job, L=L)
        p(f"  latest run: {last} ({run_state(last)})" if last else "  latest run: none")
        bad = prerequisites(job, L)
        p("  prerequisites: " + ("ready" if not bad else "; ".join(bad)))
    if "error" in b:
        p(f"\nbudget: unreadable ({b['error']})")
    else:
        wb = b["ledger"]["by_gate"][GATE]
        pr = b["projection"]
        p(f"\nbudget (ledger, gate wb): cap {wb['cap']:.2f}, spent {wb['spent']:.2f}, committed "
          f"{wb['committed']:.2f}, remaining {wb['remaining']:.2f}; total CAP {b['ledger']['total']['cap']:.2f}")
        p(f"projection (spent + committed + est_high of {pr['pending']}): mandatory {pr['mandatory_high']:.2f}, "
          f"with optional {pr['with_optional_high']:.2f} GPU-h vs WB reserve {pr['wb_cap']:.2f} "
          f"(optional only if <= {pr['optional_max']:.2f})")
        if b.get("optional_preview"):
            p(f"optional preview: {b['optional_preview']['reason']}")
    return {"estimates": est, "budget": b}


# --- in-job helpers -----------------------------------------------------------------------------------------

def adapter_check(d: Path, init: Optional[Path] = None) -> Tuple[int, str]:
    """(0, ok) a finished adapter of the current training code (and, with INIT, started from INIT/adapter.pt as it
    is now); (1, why) missing, unfinished, or started from another INIT adapter.pt (retrain: B7, e.g. 0W was
    retrained since); (3, why) finished with other training code (B7: never silently reused; the job stops and a
    person decides)."""
    from vpd.train import n2_train_vpd as T

    d = Path(d)
    meta = read_json(d / "meta.json")
    if meta is None or not (d / "adapter.pt").is_file():
        return 1, f"{d}: no adapter.pt + meta.json"
    steps, planned = meta.get("optimizer_steps"), meta.get("optimizer_steps_planned")
    if not steps or (planned is not None and steps != planned):
        return 1, f"{d}: unfinished ({steps} of {planned} optimizer steps)"
    cur = T.code_fingerprint()["code_sha256"]
    if meta.get("code_sha256") != cur:
        return 3, f"{d}: trained with code {meta.get('code_sha256')}, current {cur}"
    if init is not None:
        want = sha256_file(Path(init) / "adapter.pt") if (Path(init) / "adapter.pt").is_file() else None
        got = (meta.get("init_adapter") or {}).get("sha256")
        if want is None or got != want:
            return 1, f"{d}: started from init adapter {got}, {init}/adapter.pt is now {want}: retrain"
    return 0, f"{d}: complete ({steps} steps, code {cur[:12]})"


def eval_check(d: Path, adapter: Optional[Path] = None) -> Tuple[int, str]:
    """(0, ok) D/results.json exists and, with ADAPTER, served ADAPTER/adapter.pt as it is now; (1, why) no
    results.json; (3, why) stale: the eval served other weights than the adapter now on disk (e.g. the tier was
    retrained since).  A stale eval is not skipped and not overwritten silently: the job flags it."""
    d = Path(d)
    r = read_json(d / "results.json")
    if r is None:
        return 1, f"{d}: no readable results.json"
    if adapter is None:
        return 0, f"{d}: results.json present"
    pt = Path(adapter) / "adapter.pt"
    disk = sha256_file(pt) if pt.is_file() else None
    served = (r.get("adapter") or {}).get("adapter_sha256")
    if disk is None or served != disk:
        return 3, f"{d}: served adapter sha {served}, {pt} is now {disk}: stale"
    return 0, f"{d}: results.json present, adapter {disk[:12]}"


def render_probe(copy: Path, pano: str, heading: float, pitch: float, sizes, out_dir: Path, tag: str) -> dict:
    """Render (pano, heading, pitch) at each size with the copy's ObservationGenerator in THIS interpreter."""
    import importlib

    copy = Path(copy).resolve()
    os.chdir(copy)
    sys.path.insert(0, str(copy))
    try:
        import cv2
        cv2.setNumThreads(1)
    except ImportError:
        cv2 = None
    og = importlib.import_module("engine.observation_generator")
    if not str(og.__file__).startswith(str(copy)):
        raise RuntimeError(f"engine imported from {og.__file__}, not {copy}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i, (w, h) in enumerate(sizes):
        gen = og.ObservationGenerator(output_size=(int(w), int(h)))
        r = gen.generate_observation(pano, float(heading), float(pitch), session_id=f"vpd_render_probe_{tag}",
                                     step=i)
        if not r:
            raise RuntimeError(f"render failed: {pano} {heading} {w}x{h}")
        src = Path(r["image_path"])
        dst = out_dir / f"{pano}_{int(round(float(heading) * 10))}_{w}x{h}.jpg"
        data = src.read_bytes()
        dst.write_bytes(data)
        src.unlink()
        rows.append({"size": [int(w), int(h)], "file": str(dst), "bytes": len(data),
                     "sha256": hashlib.sha256(data).hexdigest()})
    vers = {"python": sys.version.split()[0], "executable": sys.executable}
    for mod in ("numpy", "cv2", "py360convert", "PIL"):
        try:
            vers[mod] = getattr(importlib.import_module(mod), "__version__", "?")
        except ImportError:
            vers[mod] = None
    return {"tag": tag, "pano": pano, "heading": float(heading), "pitch": float(pitch), "renders": rows,
            "versions": vers}


def render_parity(out: Path, work: Optional[Path] = None, interpreters=None, sizes=((640, 400), (1280, 800)),
                  keep: bool = False, tasks_dir: Optional[Path] = None) -> dict:
    """B17: the same pano/heading/size rendered by PY_EVAL and PY_Q3 must give byte-identical JPEGs."""
    from vpd.walkerbench import sandbox

    out = Path(out)
    tasks_dir = Path(tasks_dir or LAYOUT.s2test_tasks)
    interpreters = interpreters or (("PY_EVAL", str(paths.PY_EVAL)), ("PY_Q3", str(paths.PY_Q3)))
    work = Path(work or out.parent / (out.stem + "_work"))
    first = sorted(tasks_dir.glob("*.json"))[0]
    task = json.loads(first.read_text())
    pano, heading = task["spawn_point"], float(task["spawn_heading"])
    copy, _ = sandbox.make_sandbox(work, tasks_dir, render_size=tuple(sizes[0]), exclude=sandbox.LEAN_EXCLUDE)
    env = dict(os.environ, PYTHONPATH=str(paths.WORKTREE), PYTHONDONTWRITEBYTECODE="1",
               TOKENIZERS_PARALLELISM="false")
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[k] = "1"
    probes = []
    try:
        for tag, py in interpreters:
            cmd = [py, "-m", "vpd.walkerbench.pilot", "render-probe", "--copy", str(copy), "--pano", pano,
                   "--heading", repr(heading), "--sizes", ",".join(f"{w}x{h}" for w, h in sizes),
                   "--out-dir", str(out.parent / f"{out.stem}_{tag}"), "--tag", tag]
            res = subprocess.run(cmd, cwd=str(paths.WORKTREE), env=env, capture_output=True, text=True, timeout=900)
            if res.returncode != 0:
                probes.append({"tag": tag, "error": (res.stderr or res.stdout)[-1500:]})
                continue
            probes.append(json.loads(res.stdout.strip().splitlines()[-1]))
    finally:
        if not keep and (copy / sandbox.MARKER).exists():
            shutil.rmtree(work / "WalkerBench", ignore_errors=True)
            try:
                work.rmdir()
            except OSError:
                pass
    ok = len(probes) == len(interpreters) and all("renders" in p for p in probes)
    per_size = []
    if ok:
        for i, (w, h) in enumerate(sizes):
            shas = {p["tag"]: p["renders"][i]["sha256"] for p in probes}
            per_size.append({"size": [w, h], "sha256": shas, "equal": len(set(shas.values())) == 1})
    rep = {"check": "B17 render parity", "task": first.name, "pano": pano, "heading": heading,
           "interpreters": [list(x) for x in interpreters], "probes": probes, "per_size": per_size,
           "equal": bool(ok and per_size and all(s["equal"] for s in per_size)), "utc": utc()}
    write_json(out, rep)
    return rep


def pick_turns(turns: List[int], k: int) -> List[int]:
    """Up to k turns spread over a demo: the first, the second, then evenly up to the last."""
    turns = sorted(turns)
    if len(turns) <= k:
        return turns
    idx = {0, 1, len(turns) - 1}
    for j in range(1, k):
        idx.add(round(j * (len(turns) - 1) / (k - 1)))
    picked = sorted(turns[i] for i in idx)
    keep = [picked[0], picked[1], picked[-1]] + picked[2:-1]
    return sorted(dict.fromkeys(keep[:k]))


def data_url(path) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(Path(path).read_bytes()).decode("ascii")


def oai_request(msgs: list, images: list, served: str) -> dict:
    """The chat request an OpenAI client sends for a training prefix: system/assistant as strings, user content as
    text/image parts in placeholder order (vpd.train.parity_check.oai_messages), each image a JPEG data URL."""
    from vpd.train import parity_check as PC

    out, i = [], 0
    for m in PC.oai_messages(msgs):
        c = m["content"]
        if isinstance(c, list):
            parts = []
            for part in c:
                if part.get("type") == "image":
                    parts.append({"type": "image_url", "image_url": {"url": data_url(images[i])}})
                    i += 1
                else:
                    parts.append({"type": "text", "text": part["text"]})
            out.append({"role": m["role"], "content": parts})
        else:
            out.append({"role": m["role"], "content": c})
    if i != len(images):
        raise ValueError(f"{i} image parts for {len(images)} images")
    return {"model": served, "messages": out, "max_tokens": 1, "temperature": 0.0}


def _post(base_url: str, body: dict, timeout: float = 600) -> dict:
    import urllib.request

    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def parity_cases(dry: dict, records: Dict[str, dict], demos: int = 3, per_demo: int = 4) -> List[dict]:
    """[{id, turn, n_images, train_prompt_tokens, msgs, images}] for the dry run's first `demos` parity demos."""
    from vpd.train import n2_train_vpd as T

    cases = []
    for want in dry["parity"][:demos]:
        rec = records[want["id"]]
        pref = {p["turn"]: p for p in want["prefixes"]}
        chosen = set(pick_turns(list(pref), per_demo))
        for turn, msgs, imgs in T.prefix_contexts(rec, dry["history_k"], dry["blind"]):
            if turn in chosen:
                cases.append({"id": want["id"], "turn": turn, "n_images": len(imgs),
                              "train_prompt_tokens": pref[turn]["prompt_tokens"], "msgs": msgs, "images": imgs})
    return cases


def vllm_parity(start: str, dry_json: Path, out: Path, demos: int = 3, per_demo: int = 4, post=None,
                server=None) -> dict:
    """B5: for 2-3 demos, the live vLLM server's usage.prompt_tokens of the turn-t context equals the training
    tokenization (dry_run.json prefixes, PY_TRAIN) of the same prefix + generation prompt."""
    from vpd import starts
    from vpd.train import n2_train_vpd as T

    dry = json.loads(Path(dry_json).read_text())
    if dry.get("start") != start:
        raise ValueError(f"{dry_json} is a dry run of {dry.get('start')!r}, not {start!r}")
    records = {r.get("id"): r for r in T.load_records(Path(dry["data"]))}
    cases = parity_cases(dry, records, demos, per_demo)
    code = starts.resolve(start)["code"]
    post = post or _post
    rows = []

    def run(base_url):
        for c in cases:
            body = oai_request(c["msgs"], c["images"], code)
            try:
                got = (post(base_url, body).get("usage") or {}).get("prompt_tokens")
                err = None
            except Exception as e:  # recorded; the check fails
                got, err = None, f"{type(e).__name__}: {e}"[:300]
            rows.append({k: c[k] for k in ("id", "turn", "n_images", "train_prompt_tokens")} |
                        {"server_prompt_tokens": got, "equal": got == c["train_prompt_tokens"], "error": err})

    if server is None:
        from vpd.common.vllm_server_vpd import VLLMServer

        server = VLLMServer(code, log=Path(out).parent / (Path(out).stem + "_vllm.log"), max_images=MAX_IMAGES)
    with server as srv:
        run(srv.base_url)
        manifest = srv.manifest() if hasattr(srv, "manifest") else None
    rep = {"check": "B5 live vLLM prompt tokens vs training tokenization", "start": start, "code": code,
           "dry_run": str(dry_json), "demos": sorted({r["id"] for r in rows}), "n": len(rows), "rows": rows,
           "equal": bool(rows) and len({r["id"] for r in rows}) >= min(2, demos) and all(r["equal"] for r in rows),
           "server": manifest, "utc": utc()}
    write_json(out, rep)
    return rep


# --- smoke verdict ------------------------------------------------------------------------------------------

def _chk(name: str, ok, detail) -> dict:
    return {"name": name, "ok": bool(ok), "detail": detail}


def _attn_ok(sites) -> bool:
    return isinstance(sites, dict) and bool(sites) and all(v == "sdpa" for v in sites.values())


def smoke_eval_dir(smoke: Path, key: str) -> Path:
    model, tier, res, mode, k = SMOKE_EVALS[key]
    name = f"smoke__{model}__{tier}__{res}__{mode}" + (f"__k{k}" if k else "")
    return Path(smoke) / "evals" / name


def step_cap_total(tasks_dir: Path) -> int:
    tot = 0
    for p in Path(tasks_dir).glob("*.json"):
        t = json.loads(p.read_text())
        vp = t.get("visual_path") or []
        tot += int(round(len(vp) * 1.5)) if vp else int(t.get("max_steps") or 30)
    return tot


def _jsonl_ids(p: Path) -> List[str]:
    out = []
    for line in Path(p).read_text().splitlines():
        if line.strip():
            out.append(json.loads(line).get("id"))
    return out


def smoke_verdict(smoke: Path, L: Layout = None, local_disk=("/tmp", "/dev/shm")) -> dict:
    """Read every smoke artifact in SMOKE and write SMOKE/smoke_verdict.json (verdict PASS only if every check
    holds); "measured" feeds estimate_from_smoke."""
    from vpd.train import n2_train_vpd as T

    L = L or LAYOUT
    smoke = Path(smoke)
    checks, measured = [], {}
    # HF processor parity (PY_TRAIN dry run vs PY_Q3 parity_check) and the B16 cap in the training processor
    dry = read_json(smoke / "dry_200" / "dry_run.json")
    par = read_json(smoke / "dry_200" / "parity.json")
    checks.append(_chk("hf_parity", par and par.get("verdict") == "PASS",
                       None if not par else {"verdict": par.get("verdict"), "problems": par.get("problems"),
                                             "prefixes_checked": par.get("prefixes_checked")}))
    checks.append(_chk("train_max_pixels", dry and dry.get("max_pixels") == paths.MAX_PIXELS,
                       None if not dry else dry.get("max_pixels")))
    if dry:
        per = {c["id"]: sum(c["tokens"]) for c in dry.get("per_conversation") or []}
        tok = {}
        for n in ("50", "200"):
            try:
                ids = _jsonl_ids(L.tier(n))
                tok[n] = sum(per[i] for i in ids) if all(i in per for i in ids) else None
                measured.setdefault("tier_epochs", {})[n] = T.epochs_for("auto", len(ids))[0]
            except OSError:
                tok[n] = None
        measured["tier_tokens"] = tok if all(v is not None for v in tok.values()) else None
    # B5 live vLLM and B17 render parity
    vp = read_json(smoke / "vllm_parity.json")
    checks.append(_chk("vllm_parity", vp and vp.get("equal") is True,
                       None if not vp else {"n": vp.get("n"), "demos": vp.get("demos"),
                                            "bad": [r for r in vp.get("rows", []) if not r.get("equal")][:5]}))
    rp = read_json(smoke / "render_parity.json")
    checks.append(_chk("render_parity", rp and rp.get("equal") is True,
                       None if not rp else {"per_size": rp.get("per_size"),
                                            "errors": [p.get("error") for p in rp.get("probes", [])
                                                       if p.get("error")]}))
    # train probe (8 longest demos) and the Q3RD load-only check (B10)
    pt = read_json(smoke / "train_probe" / "probe_train.json")
    checks.append(_chk("train_probe", pt and pt.get("rc") == 0 and (pt.get("optimizer_steps") or 0) >= 1
                       and _attn_ok(pt.get("attn_sites")),
                       None if not pt else {k: pt.get(k) for k in ("rc", "optimizer_steps", "tokens", "max_len",
                                                                    "tokens_per_s", "peak_mem_gib", "attn_sites")}))
    pl = read_json(smoke / "train_probe" / "probe_load_Q3RD.json")
    checks.append(_chk("load_Q3RD", pl and pl.get("verdict") == "PASS" and _attn_ok(pl.get("attn_sites")),
                       None if not pl else {k: pl.get(k) for k in ("verdict", "attn_sites", "load_seconds")}))
    if pt:
        measured.update(probe_tokens_per_s=pt.get("tokens_per_s"), probe_load_s=pt.get("load_seconds"),
                        probe_peak_mem_gib=pt.get("peak_mem_gib"))
    # Q3S 0W (the smoke's own copy; every pilot job retrains its 0W, B7)
    zm = read_json(smoke / "adapters" / "Q3S_0W" / "meta.json")
    checks.append(_chk("train_0W", zm and zm.get("optimizer_steps") and
                       zm.get("optimizer_steps") == zm.get("optimizer_steps_planned") and zm.get("attn") == "sdpa",
                       None if not zm else {k: zm.get(k) for k in ("optimizer_steps", "optimizer_steps_planned",
                                                                    "epochs", "tokens_per_s", "peak_mem_gib",
                                                                    "code_sha256", "loss_first", "loss_last")}))
    if zm:
        measured["zeroW_s"] = (zm.get("load_seconds") or 0) + (zm.get("train_seconds") or 0)
        measured["zeroW_peak_mem_gib"] = zm.get("peak_mem_gib")
    # merge round trip: merge.py's own saved-tensor check, then eval c served the merged folder
    mg = read_json(smoke / "merge" / "vpd_merge.json")
    checks.append(_chk("merge", mg and zm and mg.get("merged") == (zm.get("lora") or {}).get("wrapped"),
                       None if not mg else {k: mg.get(k) for k in ("merged", "seconds", "out", "out_facts")}))
    if mg:
        measured["merge_s"] = mg.get("seconds")
    # the five mini evals
    evals = {}
    for key, (model, tier, res, mode, k) in SMOKE_EVALS.items():
        d = smoke_eval_dir(smoke, key)
        r = read_json(d / "results.json")
        problems = []
        if r is None:
            problems.append("no results.json" + (" (results_invalid.json)" if (d / "results_invalid.json").exists()
                                                 else ""))
        else:
            rx = r.get("regex") or {}
            it = r.get("image_tokens") or {}
            srv = r.get("server") or {}
            mm = srv.get("mm_processor_kwargs") or {}
            if r.get("n_tasks") != SMOKE_LIMIT or r.get("tasks_without_log") != 0:
                problems.append(f"n_tasks {r.get('n_tasks')}, tasks_without_log {r.get('tasks_without_log')}")
            if mode == "off":
                if rx.get("constrained_calls") != 0:
                    problems.append(f"unconstrained run sent {rx.get('constrained_calls')} regexes")
            elif not (rx.get("constrained_calls") and rx.get("constrained_calls") == rx.get("answered")
                      and rx.get("regex_mismatch") == 0):
                problems.append(f"regex: {rx}")
            if it.get("tokens_per_image") != [TOKENS_PER_IMAGE[res]]:
                problems.append(f"tokens per image {it.get('tokens_per_image')} != [{TOKENS_PER_IMAGE[res]}]")
            if (it.get("prompt_parity") or {}).get("equal") is not True:
                problems.append(f"eval prompt parity {(it.get('prompt_parity') or {}).get('equal')}")
            if (srv.get("cfg") or {}).get("max_images") != MAX_IMAGES or r.get("max_images") != MAX_IMAGES:
                problems.append(f"max_images {(srv.get('cfg') or {}).get('max_images')}")
            # same rule as pilot_gate.check_eval_settings: max_pixels alone does not cap in vLLM 0.11 / tf 4.57
            if mm.get("max_pixels") != paths.MAX_PIXELS or (mm.get("size") or {}).get("longest_edge") != paths.MAX_PIXELS:
                problems.append(f"mm_processor_kwargs {mm} (B16: max_pixels and size.longest_edge)")
            if list(r.get("render_size") or []) != [int(x) for x in RES[res].split("x")]:
                problems.append(f"render_size {r.get('render_size')}")
            if r.get("history_turns") != k or r.get("constrained") != mode:
                problems.append(f"history_turns {r.get('history_turns')}, constrained {r.get('constrained')}")
            if tier == "0W" and not r.get("model_dir"):
                problems.append("eval c did not serve the merged folder")
        evals[key] = {"dir": str(d), "problems": problems, "ok": not problems,
                      "sps": None if r is None else r.get("seconds_per_step"),
                      "mean_steps": None if r is None else r.get("mean_steps"),
                      "wall_s": None if r is None else r.get("wall_seconds"),
                      "ready_s": None if r is None else (r.get("server") or {}).get("ready_seconds")}
        checks.append(_chk(f"eval_{key}", not problems, {"config": SMOKE_EVALS[key], "problems": problems}))
    checks.append(_chk("merge_reload", evals["c"]["ok"] and mg is not None, "eval c on the merged Q3S 0W"))
    measured["evals"] = {"640_empty": evals["a"], "1280_empty": evals["b"], "640_merged": evals["c"],
                         "640_off": evals["d"], "640_empty_k2": evals["e"]}
    measured["evals"] = {k: {f: v[f] for f in ("sps", "mean_steps", "wall_s", "ready_s")}
                         for k, v in measured["evals"].items()}
    try:
        measured["step_cap_total"] = step_cap_total(L.s2test_tasks)
        measured["n_tasks"] = len(list(L.s2test_tasks.glob("*.json")))
    except OSError:
        measured["step_cap_total"] = None
    disk = {}
    for d in local_disk:
        try:
            u = shutil.disk_usage(d)
            disk[d] = {"free_gb": round(u.free / 1e9, 1), "total_gb": round(u.total / 1e9, 1)}
        except OSError as e:
            disk[d] = {"error": str(e)}
    measured["local_disk"] = disk
    failed = [c["name"] for c in checks if not c["ok"]]
    v = {"schema": "vpd.walkerbench.smoke_verdict/1", "utc": utc(), "smoke": str(smoke),
         "verdict": "PASS" if not failed else "FAIL", "failed": failed, "checks": checks, "measured": measured}
    try:
        v["estimate"] = estimate_from_smoke(measured)
    except (ValueError, KeyError, TypeError) as e:
        v["estimate"] = {"error": str(e)}
    write_json(smoke / "smoke_verdict.json", v)
    return v


# --- CLI ----------------------------------------------------------------------------------------------------

def _sizes(text: str):
    from vpd.walkerbench.sandbox import parse_size

    return [parse_size(s) for s in text.split(",") if s.strip()]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan")
    p = sub.add_parser("prepare")
    p.add_argument("job", choices=sorted(JOBS))
    p.add_argument("--pool", default="zoetrope", choices=("zoetrope", "h800", "auto"))
    p.add_argument("--new", action="store_true", help="another run dir although one is prepared")
    p.add_argument("--redecide", action="store_true", help="recompute the optional-eval decision")
    p = sub.add_parser("submit")
    p.add_argument("job", choices=sorted(JOBS))
    sub.add_parser("status")
    p = sub.add_parser("collect")
    p.add_argument("jobs", nargs="*", help=f"some of {sorted(JOBS)} (default: every job with a submitted run)")
    p = sub.add_parser("delete")
    p.add_argument("job", choices=sorted(JOBS))
    p.add_argument("--allow-incomplete", action="store_true")
    sub.add_parser("budget")
    p = sub.add_parser("adapter-ok")
    p.add_argument("dir", type=Path)
    p.add_argument("--init", type=Path, default=None, help="the 0W dir this adapter must have started from")
    p = sub.add_parser("eval-ok")
    p.add_argument("dir", type=Path)
    p.add_argument("--adapter", type=Path, default=None, help="the adapter dir the merged eval must have served")
    p = sub.add_parser("render-probe")
    p.add_argument("--copy", type=Path, required=True)
    p.add_argument("--pano", required=True)
    p.add_argument("--heading", type=float, required=True)
    p.add_argument("--pitch", type=float, default=0.0)
    p.add_argument("--sizes", default="640x400,1280x800")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--tag", default="probe")
    p = sub.add_parser("render-parity")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--work", type=Path, default=None)
    p.add_argument("--keep", action="store_true")
    p = sub.add_parser("vllm-parity")
    p.add_argument("--start", default="Q3S")
    p.add_argument("--dry-run-json", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--demos", type=int, default=3)
    p.add_argument("--per-demo", type=int, default=4)
    p = sub.add_parser("smoke-verdict")
    p.add_argument("--smoke", type=Path, required=True)
    args = ap.parse_args(argv)
    try:
        if args.cmd == "plan":
            plan()
        elif args.cmd == "prepare":
            print(prepare(args.job, args.pool, args.new, args.redecide))
        elif args.cmd == "submit":
            print(submit(args.job))
        elif args.cmd == "status":
            print(json.dumps(status(), indent=1, default=str))
        elif args.cmd == "collect":
            jobs = args.jobs or [j for j in JOBS if latest_run(j, ("submitted",))]
            for j in jobs:
                job_spec(j)
                rep = collect(j)
                print(json.dumps({k: rep[k] for k in rep if k != "expected"}, indent=1, default=str))
        elif args.cmd == "delete":
            print(delete(args.job, args.allow_incomplete))
        elif args.cmd == "budget":
            print(json.dumps(budget(), indent=1, default=str))
        elif args.cmd == "adapter-ok":
            rc, msg = adapter_check(args.dir, args.init)
            print(f"[adapter-ok] {msg}")
            return rc
        elif args.cmd == "eval-ok":
            rc, msg = eval_check(args.dir, args.adapter)
            print(f"[eval-ok] {msg}")
            return rc
        elif args.cmd == "render-probe":
            print(json.dumps(render_probe(args.copy, args.pano, args.heading, args.pitch, _sizes(args.sizes),
                                          args.out_dir, args.tag)))
        elif args.cmd == "render-parity":
            rep = render_parity(args.out, args.work, keep=args.keep)
            print(json.dumps({k: rep[k] for k in ("equal", "pano", "heading", "per_size")}, indent=1))
            return 0 if rep["equal"] else 1
        elif args.cmd == "vllm-parity":
            rep = vllm_parity(args.start, args.dry_run_json, args.out, args.demos, args.per_demo)
            print(json.dumps({k: rep[k] for k in ("equal", "n", "demos")}))
            return 0 if rep["equal"] else 1
        elif args.cmd == "smoke-verdict":
            v = smoke_verdict(args.smoke)
            print(json.dumps({"verdict": v["verdict"], "failed": v["failed"], "estimate": v.get("estimate")},
                             indent=1, default=str))
    except Refused as e:
        print(f"[pilot] REFUSED: {e.message}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
