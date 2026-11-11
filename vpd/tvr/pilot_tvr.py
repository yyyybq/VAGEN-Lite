"""TVR pilot orchestration (CONTRACT_wave2 sections 5.3.1, 7; DECISIONS G3, G10; mirrors vpd/walkerbench/pilot.py).

    PY_EVAL -m vpd.tvr.pilot_tvr plan
    PY_EVAL -m vpd.tvr.pilot_tvr prepare JOB [--pool zoetrope|h800|auto] [--new]
    PY_EVAL -m vpd.tvr.pilot_tvr submit JOB
    PY_EVAL -m vpd.tvr.pilot_tvr status
    PY_EVAL -m vpd.tvr.pilot_tvr collect JOB
    PY_EVAL -m vpd.tvr.pilot_tvr delete JOB [--allow-incomplete]
    PY_EVAL -m vpd.tvr.pilot_tvr budget
    PY_EVAL -m vpd.tvr.pilot_tvr data-fill        # the 13 data.* values from the data job's files (read only)
  in-job helpers (pure; called by vpd/jobs/tvr_*.sh):
    ... pilot_tvr expect JOB --run RUN            # RUN/expect.json; exit 0 iff every expected item holds
    ... pilot_tvr smoke-verdict --smoke DIR       # DIR/smoke_verdict.json (mechanism only, never an SR)
    ... pilot_tvr cal-decide --cal-root DIR       # DIR/cal_decision.json (EXPLORATORY, Q3B only, dev200 only)
    ... pilot_tvr guard JOB --run RUN             # the job's guard step (RUN/guard.json); exit 0 ok, 2 refused
    ... pilot_tvr cfg KEY [KEY ...]               # pilot_cfg values for the payload; exit 2 if one is None
    ... pilot_tvr served START                    # the served model name (the start's code)
    ... pilot_tvr parity-ok FILE --adapter DIR    # 0 current + pass, 4 current + failed, 1 missing / other sha

Jobs (one card each, gate tvr, display name downstream-vpd-<job>, RUN = OUT/jobs/<job>-<yyyymmddHHMM>; every sco
call goes through vpd.sco, which runs the job from a code snapshot RUN/code, DECISIONS G1):
    tvr-data         non-root  tvr_data.sh RUN                 est_high 2.5 (constant, CONTRACT 7.1)
    tvr-cal          as_root   tvr_cal.sh RUN                  est_high 5.5 (7.2)
    tvr-cal400       as_root   tvr_cal400.sh RUN               est_high 1.5 (7.2b; only if cal_decision.train_400)
    tvr-msmoke       as_root   tvr_mech_smoke.sh RUN           est_high 1.0 (7.3)
    tvr-train-<s>    as_root   tvr_train.sh START RUN          est_high DERIVED from the calibration (5.3.1)
    tvr-eval-<s>-top as_root   tvr_eval.sh START top RUN       est_high DERIVED (5.3.1; always split, O20)
    tvr-eval-<s>-low as_root   tvr_eval.sh START low RUN       est_high DERIVED
Payload: `env VPD_JOB_DEADLINE_S=<soft> timeout -k <kill> <outer> bash WT/vpd/jobs/<script> [START [top|low]] RUN`;
outer = floor(est_high*3600) - HEADROOM_S, soft = outer - SOFT_RESERVE_S (vpd.walkerbench.pilot's constants; the WB
job_timeouts itself refuses non-WB jobs, so the rule is re-implemented here), kill 60 (data, msmoke) or 120.
Refused: outer < MIN_OUTER_S, est_high > budget.job_est_high_max_h (5.5; never silently raised).

Derived est_high (start-independent: Q3B calibration measurements only; frozen in PREREG_TVR_PILOT.md section 16):
  train-<s>: 1.3 x sum over the job's tiers of (cal adapter meta train_seconds + load_seconds) / 3600 + 0.1; tiers
             0W, 100, N_top (+ Q3B: 0W_s1002, 100_s1002, taking the cal 0W / 100 values); N_top 400 -> cal400 meta.
  eval-<s>-top|low: sum over the job's evals (top: N_top c, u, blind; low: 0W c, 100 c [+ Q3B 100s1002 c]) of
             [vllm_ready_s_max + steps_high / R + merge_s + parity_s] / 3600 + 0.15, each eval counted in full (the
             literal formula: conservative, a top session's shared start-up is counted three times); R = min over the
             cal image-arm dev evals of results.perf.steps_per_s (the blind arm: the shakedown's blind steps/s);
             merge_s / parity_s = the max seconds of the cal jobs' merge_* / merge_parity_* steps (steps.jsonl);
             steps_high = min(17,500 (the eval500 cap total), 1.5 x 500 x the cal dev200 mean_steps_per_task of the
             same tier: 0W -> 0W, 100 -> 100, N_top (and its u / blind arms) -> N_top, 100s1002 -> 100).
Round budget (brief 2C / 5: 200 GPU-h ends the round; ledger.budget_check is uncapped by BUDGET_POLICY.json):
  round_used = ledger total spent - budget.round_baseline_gpuh; prepare / submit refuse (exit 2) when round_used + the
  est of the open (submitted / uncertain) ledger jobs + the new est_high > budget.round_cap_gpuh; the three numbers
  are printed and stored in RUN/prepared.json["round_budget"].
Guards (prepare and submit; in-job again by `guard`, where the prereg's committed state comes from the snapshot):
  every job: pilot_cfg.missing(job) == []; the values the payloads spell out (PAYLOAD_CONSTANTS: seeds, replicate,
  merge-parity n, grid, workers) == pilot_cfg; the trainer fingerprint prefix (n2_train_vpd.code_fingerprint()) ==
  recipe.trainer_code_sha256_prefix.  tvr-cal: the data job DONE, every data.* filled (pilot_cfg REQUIRED) and each
  sha == the file on disk.  tvr-cal400: the cal job DONE and cal_decision.train_400.  tvr-msmoke / train / eval:
  vpd/PREREG_TVR_PILOT.md committed and clean (git log -1 / status --porcelain on that file only) and
  pilot_cfg.check_prereg(<it>) == []; data shas; tiers.n_top == CAL/cal_decision.json n_top (with_400 true for 400;
  n_top_problems).  train / eval: the newest smoke_verdict.json PASS.  msmoke / train: cal_decision.futility false, or
  OUT/tvr/runs/explore/cal/Q3B/futility_override.json exists (the orchestrator writes it only after the user
  decided, section 8.1).  eval-<s>-*: the train-<s> adapters adapter-ok with the right init.
Shared parents: before vsco.prepare of ANY job, prepare creates OUT/tvr/runs, runs/explore/cal/Q3B, runs/smoke,
runs/pilot and runs/pilot/<S> as the login user, so an as_root job never creates (and owns) them.
"""

from __future__ import annotations

import argparse
import datetime
import fractions
import json
import math
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from vpd import paths
from vpd.tvr import pilot_cfg
from vpd.walkerbench.pilot import (HEADROOM_S, MIN_OUTER_S, SOFT_RESERVE_S, adapter_check, eval_check, read_json,
                                   sha256_file, stamp, utc, write_json)

GATE = "tvr"
STARTS = tuple(pilot_cfg.STARTS)
PREREG = "vpd/PREREG_TVR_PILOT.md"
JOBS_DIR = paths.VPD_PKG / "jobs"
STAMP_RE = re.compile(r"^(?P<job>[a-z0-9-]+)-(?P<stamp>\d{12})$")
EST_HIGH = {"tvr-data": 2.5, "tvr-cal": 5.5, "tvr-cal400": 1.5, "tvr-msmoke": 1.0}  # CONTRACT 7.1-7.3
KILL_S = {"tvr-data": 60, "tvr-msmoke": 60}  # every other job 120
KILL_DEFAULT = 120
TRAIN_FACTOR, TRAIN_PLUS_H = 1.3, 0.1
EVAL_PLUS_H, STEPS_FACTOR = 0.15, 1.5
EVAL500_N = 500
PUBLIC_N = 1600  # the public sft_train.json sequences = sft1600 (the replay's and the demos' denominator)
EVAL500_CAP_TOTAL = 17500  # 250 iTHOR tasks x 30 + 250 ProcTHOR tasks x 40 (eval.json; protocol.max_steps)
# values the bash payloads spell out literally (the contract-7 golden command lists); every job's guard (and prepare)
# refuses when pilot_cfg disagrees, so a change to pilot_cfg cannot silently diverge from the payloads
PAYLOAD_CONSTANTS = {"seed": 1001, "tiers.replicate.start": "Q3B", "tiers.replicate.tier": "100",
                     "tiers.replicate.seed": 1002, "gates.merge_parity.n_demos": 20, "protocol.image_grid": [1, 22, 40],
                     "recipe.num_workers": 3, "recipe.threads": 2, "starts": list(STARTS)}


def smoke_n(cfg=None) -> int:
    """Tasks of smoke8: dev.smoke_per_category dev200 tasks per category."""
    return len(pilot_cfg.CATEGORIES) * int(pilot_cfg.get("dev.smoke_per_category", cfg))


SHAKE_N = len(pilot_cfg.CATEGORIES)  # smoke4: the first dev200 task of each category
ARMS_TOP = ("c", "u", "blind")
IMAGE_ARMS = ("c", "u")
PHASE = {"0W": "0W", "100": "tier100", "400": "tier400", "ALL": "tierALL", "0W_s1002": "0W_s1002",
         "100_s1002": "tier100_s1002"}
TIER_FILE = {"0W": "warmup0w20.jsonl", "100": "demos_100.jsonl", "400": "demos_400.jsonl", "ALL": "demos_ALL.jsonl",
             "0W_s1002": "warmup0w20.jsonl", "100_s1002": "demos_100.jsonl"}
# eval tier label -> adapter dir name (pilot); the replicate's eval label is 100s1002
EVAL_ADAPTER = {"0W": "0W", "100": "100", "400": "400", "ALL": "ALL", "100s1002": "100_s1002"}
DATA_FILES = {  # pilot_cfg data.<key> -> the file it is the sha256 of (relative to OUT/tvr)
    "dev_selection_sha256": "split/dev200/selection.json", "houses_sha256": "houses/houses.json",
    "reach_index_sha256": "reach/v1/index.json", "replay_manifest_sha256": "replay/sft1600_lvp/manifest.json",
    "demos_all_sha256": "demos/sft1600_lvp/demos_all.jsonl", "tiers_manifest_sha256": "tiers/v1/manifest.json",
    "bfs_eval500_sha256": "bfs/eval500_opt.json", "bfs_dev200_sha256": "bfs/dev200_opt.json",
    "floors_eval500_sha256": "floors/floors_eval500.json"}
DROP_REASONS = ("move_failed", "pose_mismatch", "no_stop", "stop_not_last", "final_not_target")


class Refused(SystemExit):
    """A pilot action was refused (exit 2); the message says why."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


def _cfg(cfg):
    return pilot_cfg.FROZEN if cfg is None else cfg


def replicate(cfg=None) -> dict:
    return pilot_cfg.get("tiers.replicate", cfg)


# --- layout -------------------------------------------------------------------------------------------------

class Layout:
    """Every path the TVR pilot reads or writes, under one output root (paths.OUT; tests use a scratch root)."""

    def __init__(self, root: Path = paths.OUT):
        self.root = Path(root)
        self.jobs = self.root / "jobs"
        t = self.root / "tvr"
        self.tvr = t
        self.selection = t / "split" / "dev200" / "selection.json"
        self.houses = t / "houses" / "houses.json"
        self.reach = t / "reach" / "v1"
        self.replay = t / "replay" / "sft1600_lvp"
        self.demos = t / "demos" / "sft1600_lvp"
        self.tiers = t / "tiers" / "v1"
        self.bfs = t / "bfs"
        self.floors = t / "floors"
        self.runs = t / "runs"
        self.cal = self.runs / "explore" / "cal" / "Q3B"
        self.smoke_root = self.runs / "smoke"
        self.pilot_root = self.runs / "pilot"
        self.cal_decision = self.cal / "cal_decision.json"
        self.futility_override = self.cal / "futility_override.json"
        self.fallback = self.cal / "shake" / "fallback_used.json"

    def data_file(self, key: str) -> Path:
        return self.tvr / DATA_FILES[key]

    def shared_parents(self, starts=STARTS) -> List[Path]:
        return [self.runs, self.cal, self.smoke_root, self.pilot_root] + [self.pilot_root / s for s in starts]

    # calibration (EXPLORATORY, Q3B)
    def cal_adapter(self, tier: str) -> Path:
        return self.cal / "adapters" / tier

    def cal_eval(self, tier: str, arm: str = "c") -> Path:
        return self.cal / "evals" / f"cal__Q3B__{tier}__{arm}"

    def cal_shake(self, arm: str) -> Path:
        return self.cal / "shake" / f"calshake__Q3B__raw__{arm}"

    def smoke_dir(self, run) -> Path:
        return self.smoke_root / Path(run).name

    # pilot
    def start_dir(self, s: str) -> Path:
        return self.pilot_root / s

    def adapter(self, s: str, tier: str) -> Path:
        return self.start_dir(s) / "adapters" / tier

    def eval_name(self, s: str, tier: str, arm: str) -> str:
        return f"pilot__{s}__{tier}__{arm}"

    def eval_dir(self, s: str, tier: str, arm: str) -> Path:
        return self.start_dir(s) / "evals" / self.eval_name(s, tier, arm)

    def parity(self, s: str, tier: str) -> Path:
        return self.start_dir(s) / "parity" / f"merge_{tier}.json"

    def canary(self, s: str, part: str) -> Path:
        return self.start_dir(s) / "canary" / part / "render_parity.json"


LAYOUT = Layout()


# --- job table ----------------------------------------------------------------------------------------------

def _jobs() -> Dict[str, dict]:
    j = {"tvr-data": {"name": "vpd-tvr-data", "script": "tvr_data.sh", "as_root": False, "kind": "data"},
         "tvr-cal": {"name": "vpd-tvr-cal", "script": "tvr_cal.sh", "as_root": True, "kind": "cal"},
         "tvr-cal400": {"name": "vpd-tvr-cal400", "script": "tvr_cal400.sh", "as_root": True, "kind": "cal400"},
         "tvr-msmoke": {"name": "vpd-tvr-msmoke", "script": "tvr_mech_smoke.sh", "as_root": True, "kind": "msmoke"}}
    for s in STARTS:
        sl = s.lower()
        j[f"tvr-train-{sl}"] = {"name": f"vpd-tvr-train-{sl}", "script": "tvr_train.sh", "as_root": True,
                                "kind": "train", "start": s}
        for part in ("top", "low"):
            j[f"tvr-eval-{sl}-{part}"] = {"name": f"vpd-tvr-eval-{sl}-{part}", "script": "tvr_eval.sh",
                                          "as_root": True, "kind": "eval", "start": s, "part": part}
    for v in j.values():
        v.setdefault("start", None)
        v.setdefault("part", None)
    return j


JOBS = _jobs()


def job_spec(job: str) -> dict:
    if job not in JOBS:
        raise Refused(f"unknown job {job!r}; one of {sorted(JOBS)}")
    return JOBS[job]


def job_timeouts(job: str, est_high: float, cfg=None) -> dict:
    """{est_high, outer_s, kill_s, soft_deadline_s}: the WB rule (pilot.job_timeouts) with the TVR kill windows and
    the 5.5 h per-job cap."""
    job_spec(job)
    hi = float(est_high)
    if not math.isfinite(hi) or hi <= 0:
        raise Refused(f"{job}: est_high {est_high!r} must be a positive number")
    cap = float(pilot_cfg.get("budget.job_est_high_max_h", cfg))
    if hi > cap + 1e-9:
        raise Refused(f"{job}: est_high {hi:.3f} GPU-h > the {cap} h job cap (CONTRACT 5.3.1): stop and ask; the "
                      "outer timeout is never raised silently")
    outer = int(math.floor(round(hi * 3600, 6))) - HEADROOM_S
    kill = KILL_S.get(job, KILL_DEFAULT)
    if outer < MIN_OUTER_S:
        raise Refused(f"{job}: est_high {hi} GPU-h leaves an outer timeout of {outer} s (< {MIN_OUTER_S} s)")
    assert outer + kill <= hi * 3600
    return {"est_high": round(hi, 3), "outer_s": outer, "kill_s": kill, "soft_deadline_s": outer - SOFT_RESERVE_S,
            "rule": f"outer = floor(est_high*3600) - {HEADROOM_S}; soft = outer - {SOFT_RESERVE_S}"}


def payload_for(job: str, run, est_high: float, cfg=None) -> List[str]:
    spec = job_spec(job)
    t = job_timeouts(job, est_high, cfg)
    args = []
    if spec["start"]:
        args.append(spec["start"])
    if spec["part"]:
        args.append(spec["part"])
    return (["env", f"VPD_JOB_DEADLINE_S={t['soft_deadline_s']}", "timeout", "-k", str(t["kill_s"]),
             str(t["outer_s"]), "bash", str(JOBS_DIR / spec["script"])] + args + [str(run)])


# --- runs ---------------------------------------------------------------------------------------------------

def runs_of(job: str, L: Layout = None) -> List[Path]:
    L = L or LAYOUT
    if not L.jobs.is_dir():
        return []
    return sorted((p for p in L.jobs.iterdir() if p.is_dir() and (m := STAMP_RE.match(p.name))
                   and m.group("job") == job), key=lambda p: p.name)


def run_state(run: Path) -> str:
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


def done_run(job: str, L: Layout = None) -> Optional[Path]:
    """The newest run of `job` with RUN/DONE."""
    for r in reversed(runs_of(job, L)):
        if (r / "DONE").is_file():
            return r
    return None


def ledger_jobs(events=None) -> Dict[str, dict]:
    from vpd import ledger

    events = ledger.read() if events is None else events
    return {r: j for r, j in ledger.fold(events).items() if j.get("gate") == GATE}


def job_in_ledger(job: str, events=None) -> Optional[dict]:
    name = job_spec(job)["name"]
    hit = None
    for j in ledger_jobs(events).values():
        if j.get("name") == name and j.get("status") in ("submitted", "uncertain", "ended"):
            hit = j
    return hit


# --- prereg, smoke, futility, data ----------------------------------------------------------------------------

def _git(*args) -> Optional[str]:
    try:
        res = subprocess.run(["git", "-c", "safe.directory=*", "-C", str(paths.WORKTREE), *args],
                             capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return res.stdout.strip() if res.returncode == 0 else None


def prereg_path() -> Path:
    return paths.WORKTREE / PREREG


def prereg_state() -> dict:
    """PREREG_TVR_PILOT.md committed (git log -1 non-empty) and clean (git status --porcelain -- FILE empty)."""
    log = _git("log", "-1", "--format=%H", "--", PREREG)
    st = _git("status", "--porcelain", "--", PREREG)
    return {"file": PREREG, "commit": log or None, "committed": bool(log), "status": st, "clean": st == "",
            "ok": bool(log) and st == ""}


def prereg_text_at(commit: str) -> Optional[str]:
    """The prereg's text at a git commit of this tree (None if git cannot show it)."""
    return _git("show", f"{commit}:{PREREG}")


def payload_constant_problems(cfg=None) -> List[str]:
    """pilot_cfg values the payloads spell out literally (PAYLOAD_CONSTANTS) must still be those values."""
    out = []
    for key, want in PAYLOAD_CONSTANTS.items():
        got = json.loads(json.dumps(pilot_cfg.get(key, cfg)))
        if got != want:
            out.append(f"pilot_cfg {key} = {got!r}, but the payloads use {want!r}: update vpd/jobs/tvr_*.sh first")
    return out


def prereg_problems(in_job: bool = False, cfg=None) -> List[str]:
    bad = []
    if not in_job:  # in a job the committed state is checked by vpd_snapshot_file (the snapshot has no git)
        ps = prereg_state()
        if not ps["committed"]:
            bad.append(f"{PREREG} is not committed (git log -1 -- {PREREG} is empty)")
        elif not ps["clean"]:
            bad.append(f"{PREREG} has uncommitted changes: {ps['status']!r}")
    diff = check_prereg(cfg)
    if diff:
        bad.append(f"{PREREG} block != pilot_cfg.FROZEN ({len(diff)} differences, first: {diff[0]})")
    return bad


def check_prereg(cfg=None) -> List[str]:
    """pilot_cfg.check_prereg on the prereg of this code tree (the single parser and comparison test_tvr_prereg
    uses).  With a test cfg the comparison is against that cfg."""
    return pilot_cfg.check_prereg(prereg_path(), cfg)


def smoke_verdicts(L: Layout = None) -> List[Path]:
    L = L or LAYOUT
    if not L.smoke_root.is_dir():
        return []
    return sorted((p / "smoke_verdict.json" for p in L.smoke_root.iterdir()
                   if STAMP_RE.match(p.name) and (p / "smoke_verdict.json").is_file()), key=lambda p: p.parent.name)


def latest_verdict(L: Layout = None) -> Tuple[Optional[Path], Optional[dict]]:
    vs = smoke_verdicts(L)
    return (vs[-1], read_json(vs[-1])) if vs else (None, None)


def futility_problems(L: Layout) -> List[str]:
    if L.futility_override.is_file():
        return []
    d = read_json(L.cal_decision)
    if d is None:
        return [f"no {L.cal_decision} (job tvr-cal; cal-decide)"]
    if d.get("futility") is not False:
        return [f"{L.cal_decision}: futility {d.get('futility')!r} (Q3B top-tier dev Wilson LB <= the dev200 floor, "
                f"or not decided yet): stop and ask the user; only {L.futility_override.name} (written after the "
                "user's decision) lifts this"]
    return []


def n_top_problems(L: Layout, cfg=None) -> List[str]:
    """pilot_cfg tiers.n_top (filled by hand) == CAL/cal_decision.json n_top, the registered calibration rule's
    output; for 400 also with_400 true (cal-decide after tvr-cal400).  [] while n_top is unfilled (missing() says
    so)."""
    top = pilot_cfg.get("tiers.n_top", cfg)
    if top not in ("ALL", "400"):
        return []
    d = read_json(L.cal_decision)
    if d is None:
        return [f"no {L.cal_decision} (job tvr-cal; cal-decide): tiers.n_top {top!r} has no calibration decision"]
    if d.get("n_top") != top:
        return [f"tiers.n_top {top!r} != {L.cal_decision} n_top {d.get('n_top')!r}"]
    if top == "400" and d.get("with_400") is not True:
        return [f"tiers.n_top '400' but {L.cal_decision} with_400 {d.get('with_400')!r} (job tvr-cal400, cal-decide)"]
    return []


def data_problems(L: Layout, cfg=None) -> List[str]:
    """Every filled data.* sha256 equals its file on disk; F_eval500 / F_source / kept / dropped equal the files."""
    bad = []
    for key in sorted(DATA_FILES):
        want = pilot_cfg.get(f"data.{key}", cfg)
        if want is None:
            continue
        p = L.data_file(key)
        got = sha256_file(p) if p.is_file() else None
        if got != want:
            bad.append(f"data.{key} {want[:12]}... != {p} ({None if got is None else got[:12]})")
    fl = read_json(L.floors / "floors_eval500.json") or {}
    for key, val in (("F_eval500", (fl.get("F") or {}).get("rate")), ("F_source", (fl.get("F") or {}).get("source"))):
        want = pilot_cfg.get(f"data.{key}", cfg)
        if want is not None and want != val:
            bad.append(f"data.{key} {want!r} != floors_eval500.json {val!r}")
    man = read_json(L.demos / "manifest.json") or {}
    for key in ("kept", "dropped"):
        want = pilot_cfg.get(f"data.{key}", cfg)
        if want is not None and want != man.get(key):
            bad.append(f"data.{key} {want!r} != demos manifest {man.get(key)!r}")
    return bad


def data_fill(L: Layout = None) -> dict:
    """The 13 data.* values the orchestrator fills (pilot_cfg FROZEN and the prereg block, one commit) from the data
    job's files: the sha256 of each DATA_FILES file, kept / dropped from the demos manifest, F_eval500 / F_source
    from floors_eval500.json F.  Read only.  Refused unless the newest tvr-data run has DONE and an ok expect.json;
    "problems" = data_problems with these values (expected [])."""
    L = L or LAYOUT
    run = latest_run("tvr-data", L=L)
    if run is None or not (run / "DONE").is_file():
        raise Refused(f"the newest tvr-data run {run} has no DONE")
    exp = read_json(run / "expect.json") or {}
    if exp.get("ok") is not True:
        raise Refused(f"{run}/expect.json is not ok (failed {exp.get('failed')})")
    vals = {}
    for key in DATA_FILES:
        p = L.data_file(key)
        if not p.is_file():
            raise Refused(f"{p} missing")
        vals[key] = sha256_file(p)
    man = read_json(L.demos / "manifest.json") or {}
    F = (read_json(L.floors / "floors_eval500.json") or {}).get("F") or {}
    vals.update(kept=man.get("kept"), dropped=man.get("dropped"), F_eval500=F.get("rate"), F_source=F.get("source"))
    data = {k: vals.get(k) for k in pilot_cfg.FROZEN["data"]}  # pilot_cfg's key order
    absent = [k for k, v in data.items() if v is None]
    if absent:
        raise Refused(f"data.* not derivable from the data job's files: {absent}")
    cfg = json.loads(json.dumps(pilot_cfg.FROZEN))
    cfg["data"].update(data)
    return {"run": str(run), "data": data, "problems": data_problems(L, cfg)}


def trainer_prefix() -> str:
    from vpd.train import n2_train_vpd as T

    return T.code_fingerprint()["code_sha256"]


def trainer_problems(cfg=None) -> List[str]:
    want = pilot_cfg.get("recipe.trainer_code_sha256_prefix", cfg)
    got = trainer_prefix()
    return [] if got.startswith(want) else [f"trainer code_sha256 {got[:12]} != frozen prefix {want}"]


def train_tiers(start: str, cfg=None) -> List[Tuple[str, Optional[str]]]:
    """[(adapter dir, init dir or None)] of a start's pilot training job."""
    top = pilot_cfg.n_top(cfg)
    out = [("0W", None), ("100", "0W"), (top, "0W")]
    if start == replicate(cfg)["start"]:
        out += [("0W_s1002", None), ("100_s1002", "0W_s1002")]
    return out


def eval_plan(start: str, part: str, cfg=None) -> List[Tuple[str, str]]:
    """[(eval tier label, arm)] of one eval job (CONTRACT 7.5)."""
    if part == "top":
        top = pilot_cfg.n_top(cfg)
        return [(top, a) for a in ARMS_TOP]
    out = [("0W", "c"), ("100", "c")]
    rep = replicate(cfg)
    if start == rep["start"]:
        out.append((f"{rep['tier']}s{rep['seed']}", "c"))
    return out


def adapters_problems(start: str, L: Layout, cfg=None) -> List[str]:
    bad = []
    for tier, init in train_tiers(start, cfg):
        rc, msg = adapter_check(L.adapter(start, tier), None if init is None else L.adapter(start, init))
        if rc != 0:
            bad.append(f"adapter {start}/{tier}: {msg}")
    return bad


def prerequisites(job: str, L: Layout = None, cfg=None, in_job: bool = False) -> List[str]:
    """Problems that block preparing / submitting `job` (in_job: the job's own guard step); [] = ready."""
    L = L or LAYOUT
    spec = job_spec(job)
    kind = spec["kind"]
    bad = [f"pilot_cfg value missing for {job}: {k}" for k in pilot_cfg.missing(job, cfg)]
    bad += payload_constant_problems(cfg)
    bad += trainer_problems(cfg)
    if kind == "cal":
        if done_run("tvr-data", L) is None:
            bad.append("no tvr-data run with DONE (job tvr-data)")
        bad += data_problems(L, cfg)
    elif kind == "cal400":
        if done_run("tvr-cal", L) is None:
            bad.append("no tvr-cal run with DONE (job tvr-cal)")
        d = read_json(L.cal_decision)
        if not d or d.get("train_400") is not True:
            bad.append(f"{L.cal_decision}: train_400 is {None if not d else d.get('train_400')!r}, not true")
        bad += data_problems(L, cfg)
    elif kind in ("msmoke", "train", "eval"):
        bad += prereg_problems(in_job, cfg)
        bad += data_problems(L, cfg)
        bad += n_top_problems(L, cfg)
        if kind in ("msmoke", "train"):
            bad += futility_problems(L)
        if kind in ("train", "eval"):
            vp, v = latest_verdict(L)
            if v is None:
                bad.append(f"no smoke_verdict.json under {L.smoke_root} (job tvr-msmoke)")
            elif v.get("verdict") != "PASS":
                bad.append(f"{vp}: verdict {v.get('verdict')!r}, not PASS ({', '.join(v.get('failed') or [])})")
        if kind == "eval" and not any(b.startswith("pilot_cfg value missing") for b in bad):
            bad += adapters_problems(spec["start"], L, cfg)
    return bad


# --- estimates ----------------------------------------------------------------------------------------------

def _num(x) -> Optional[float]:
    return float(x) if isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) else None


def _ceil3(x: float) -> float:
    return math.ceil(round(x * 1000, 6)) / 1000


def derived_train(start: str, L: Layout = None, cfg=None) -> dict:
    L = L or LAYOUT
    src = {"0W": "0W", "100": "100", "ALL": "ALL", "400": "400", "0W_s1002": "0W", "100_s1002": "100"}
    parts, total = {}, 0.0
    for tier, _ in train_tiers(start, cfg):
        m = read_json(L.cal_adapter(src[tier]) / "meta.json")
        ts, ls = (None, None) if m is None else (_num(m.get("train_seconds")), _num(m.get("load_seconds")))
        if ts is None or ls is None:
            raise Refused(f"cannot derive the train est_high of {start}: {L.cal_adapter(src[tier])}/meta.json lacks "
                          "train_seconds / load_seconds (job tvr-cal / tvr-cal400)")
        parts[tier] = {"cal_tier": src[tier], "train_s": ts, "load_s": ls}
        total += ts + ls
    est = _ceil3(TRAIN_FACTOR * total / 3600 + TRAIN_PLUS_H)
    return {"est_high": est, "rule": f"{TRAIN_FACTOR} x sum(train_seconds + load_seconds) / 3600 + {TRAIN_PLUS_H}",
            "parts": parts}


def _cal_step_seconds(L: Layout, prefix_re: str) -> List[float]:
    out = []
    rx = re.compile(prefix_re)
    for job in ("tvr-cal", "tvr-cal400"):
        for r in runs_of(job, L):
            sj = r / "steps.jsonl"
            if not sj.is_file():
                continue
            for line in sj.read_text().splitlines():
                try:
                    s = json.loads(line)
                except ValueError:
                    continue
                if s.get("rc") == 0 and rx.fullmatch(str(s.get("name"))) and _num(s.get("secs")) is not None:
                    out.append(float(s["secs"]))
    return out


def cal_measurements(L: Layout = None, cfg=None) -> dict:
    """The Q3B calibration numbers the derived eval est_high uses (start-independent)."""
    L = L or LAYOUT
    top = pilot_cfg.n_top(cfg)
    evals, missing = {}, []
    for tier in ("0W", "100", "ALL", "400"):
        r = read_json(L.cal_eval(tier) / "results.json")
        if r is not None:
            evals[tier] = r.get("perf") or {}
    for tier in ("0W", "100", top):
        if tier not in evals:
            missing.append(str(L.cal_eval(tier) / "results.json"))
    shake = {a: (read_json(L.cal_shake(a) / "results.json") or {}).get("perf") for a in ("c", "blind")}
    if not shake["blind"]:
        missing.append(str(L.cal_shake("blind") / "results.json"))
    rates = [_num(p.get("steps_per_s")) for p in evals.values()]
    ready = [_num(p.get("vllm_ready_s")) for p in list(evals.values()) + [s for s in shake.values() if s]]
    merge = _cal_step_seconds(L, r"merge_(0W|100|ALL|400)")
    parity = _cal_step_seconds(L, r"merge_parity_(0W|100|ALL|400)")
    if missing or None in rates or not rates or None in ready or not merge or not parity:
        raise Refused("cannot derive the eval est_high from the calibration: "
                      + "; ".join(missing + ([] if rates and None not in rates else ["perf.steps_per_s"])
                                  + ([] if ready and None not in ready else ["perf.vllm_ready_s"])
                                  + ([] if merge else ["merge_* step seconds"])
                                  + ([] if parity else ["merge_parity_* step seconds"])))
    rb = _num((shake["blind"] or {}).get("steps_per_s"))
    means = {t: _num(p.get("mean_steps_per_task")) for t, p in evals.items()}
    if rb is None or rb <= 0 or min(rates) <= 0 or any(means.get(t) is None for t in ("0W", "100", top)):
        raise Refused("calibration perf incomplete (blind steps_per_s, steps_per_s > 0, mean_steps_per_task)")
    return {"R": min(rates), "R_blind": rb, "vllm_ready_s_max": max(ready), "merge_s": max(merge),
            "parity_s": max(parity), "mean_steps": means, "n_top": top}


def derived_eval(start: str, part: str, L: Layout = None, cfg=None) -> dict:
    L = L or LAYOUT
    m = cal_measurements(L, cfg)
    cal_tier = {"0W": "0W", "100": "100", "ALL": "ALL", "400": "400"}
    rep = replicate(cfg)
    total, parts = 0.0, []
    for tier, arm in eval_plan(start, part, cfg):
        src = cal_tier.get(tier, rep["tier"])  # the seed replicate (100s1002) uses the cal 100 values
        steps_high = min(float(EVAL500_CAP_TOTAL), STEPS_FACTOR * EVAL500_N * m["mean_steps"][src])
        rate = m["R_blind"] if arm == "blind" else m["R"]
        s = m["vllm_ready_s_max"] + steps_high / rate + m["merge_s"] + m["parity_s"]
        parts.append({"tier": tier, "arm": arm, "cal_tier": src, "steps_high": round(steps_high, 1),
                      "rate": rate, "seconds": round(s, 1)})
        total += s
    est = _ceil3(total / 3600 + EVAL_PLUS_H)
    return {"est_high": est, "rule": "sum over evals [vllm_ready_s_max + steps_high / R + merge_s + parity_s] / 3600"
            f" + {EVAL_PLUS_H}; steps_high = min({EVAL500_CAP_TOTAL}, {STEPS_FACTOR} x {EVAL500_N} x cal mean steps)",
            "measurements": m, "parts": parts}


def est_high(job: str, L: Layout = None, cfg=None) -> dict:
    spec = job_spec(job)
    if job in EST_HIGH:
        return {"est_high": EST_HIGH[job], "rule": "constant (CONTRACT 7)"}
    if spec["kind"] == "train":
        return derived_train(spec["start"], L, cfg)
    return derived_eval(spec["start"], spec["part"], L, cfg)


def round_budget(new_est: float, events=None, cfg=None) -> dict:
    """round_used (= ledger total spent - baseline) + est of the open ledger jobs + new <= round cap."""
    from vpd import ledger

    events = ledger.read() if events is None else events
    base = float(pilot_cfg.get("budget.round_baseline_gpuh", cfg))
    cap = float(pilot_cfg.get("budget.round_cap_gpuh", cfg))
    spent = float(ledger.summarise(events)["total"]["spent"])
    open_est = sum(float(j.get("est") or 0.0) for j in ledger.fold(events).values() if ledger.is_open(j))
    used = spent - base
    total = used + open_est + float(new_est)
    return {"ledger_spent": round(spent, 4), "baseline": base, "round_used": round(used, 4),
            "open_est": round(open_est, 4), "new_est_high": round(float(new_est), 4), "total": round(total, 4),
            "cap": cap, "ok": total <= cap + 1e-9}


# --- prepare / submit / collect / delete -------------------------------------------------------------------

def make_shared_parents(L: Layout) -> List[str]:
    out = []
    for d in L.shared_parents():
        d.mkdir(parents=True, exist_ok=True)
        out.append(str(d))
    return out


def prepare(job: str, pool: str = "zoetrope", new: bool = False, now=None, L: Layout = None, cfg=None,
            events=None) -> Path:
    from vpd import sco as vsco

    L = L or LAYOUT
    spec = job_spec(job)
    parents = make_shared_parents(L)  # as the login user, before anything else (CONTRACT 5.3.1, E-M3)
    last = latest_run(job, L=L)
    if last is not None:
        st = run_state(last)
        if st == "prepared" and not new:
            raise Refused(f"{last} is prepared and not submitted: submit it, or pass --new for another run dir")
        if st == "uncertain":
            raise Refused(f"{last}: an earlier create is uncertain; run `python -m vpd.sco resolve --run {last}`")
    led = job_in_ledger(job, events)
    if led is not None and led.get("status") in ("submitted", "uncertain"):
        raise Refused(f"{job} is open in the ledger ({led.get('pt_id')}, {led['run']})")
    bad = prerequisites(job, L, cfg)
    if bad:
        raise Refused(f"{job} not ready:\n  - " + "\n  - ".join(bad))
    est = est_high(job, L, cfg)
    hi = est["est_high"]
    t = job_timeouts(job, hi, cfg)
    rb = round_budget(hi, events, cfg)
    print(f"[pilot_tvr] round budget: used {rb['round_used']:.2f} + open {rb['open_est']:.2f} + new "
          f"{rb['new_est_high']:.2f} = {rb['total']:.2f} GPU-h vs cap {rb['cap']:.0f}", file=sys.stderr)
    if not rb["ok"]:
        raise Refused(f"ROUND BUDGET: round_used {rb['round_used']:.2f} + open {rb['open_est']:.2f} + new "
                      f"{rb['new_est_high']:.2f} = {rb['total']:.2f} > {rb['cap']} GPU-h (brief 2C: the round stops)")
    run = L.jobs / f"{job}-{stamp(now)}"
    if run.exists():
        raise Refused(f"{run} exists (prepared twice in one minute?)")
    payload = payload_for(job, run, hi, cfg)
    info = {"job": job, "utc": utc(), "est": est, "est_high": hi, "timeouts": t, "payload": payload,
            "round_budget": rb, "shared_parents": parents, "as_root": spec["as_root"], "pool": pool}
    if spec["kind"] in ("msmoke", "train", "eval"):
        info["prereg"] = prereg_state()
    if spec["kind"] in ("train", "eval"):
        vp, v = latest_verdict(L)
        info["smoke_verdict"] = {"path": str(vp), "sha256": sha256_file(vp), "verdict": v.get("verdict")}
    if L.cal_decision.is_file():
        info["cal_decision"] = {"path": str(L.cal_decision), "sha256": sha256_file(L.cal_decision)}
    launch = vsco.prepare(run, spec["name"], payload, pool=pool, as_root=spec["as_root"], est_gph=hi, gate=GATE)
    write_json(run / "prepared.json", info)
    return Path(launch)


def submit(job: str, L: Layout = None, cfg=None, events=None) -> str:
    from vpd import sco as vsco

    L = L or LAYOUT
    spec = job_spec(job)
    run = latest_run(job, ("prepared",), L)
    if run is None:
        raise Refused(f"no prepared, unsubmitted run of {job}: run `pilot_tvr prepare {job}` first")
    newer = latest_run(job, L=L)
    if newer != run:
        raise Refused(f"the newest run of {job} is {newer} ({run_state(newer)}), not the prepared {run}")
    bad = prerequisites(job, L, cfg)
    if bad:
        raise Refused(f"{job} not ready:\n  - " + "\n  - ".join(bad))
    info = read_json(run / "prepared.json") or {}
    if spec["kind"] in ("msmoke", "train", "eval"):
        ps = prereg_state()
        if (info.get("prereg") or {}).get("commit") != ps["commit"]:
            raise Refused(f"{PREREG} commit changed since prepare ({(info.get('prereg') or {}).get('commit')} -> "
                          f"{ps['commit']}): prepare again")
    rb = round_budget(float(info.get("est_high") or 0.0), events, cfg)
    if not rb["ok"]:
        raise Refused(f"ROUND BUDGET at submit: {rb['total']:.2f} > {rb['cap']} GPU-h")
    return vsco.submit(run)


def expected_outputs(job: str, run: Path, L: Layout = None, cfg=None) -> List[Path]:
    """The files collect requires (the expect items' files plus RUN/DONE and RUN/expect.json)."""
    L = L or LAYOUT
    return [Path(c["path"]) for c in expect_items(job, run, L, cfg, files_only=True)] + [
        Path(run) / "expect.json", Path(run) / "DONE"]


def collect(job: str, L: Layout = None, cfg=None) -> dict:
    L = L or LAYOUT
    run = latest_run(job, ("submitted",), L)
    if run is None:
        raise Refused(f"no submitted run of {job}")
    items = expect_items(job, run, L, cfg)
    exp = expected_outputs(job, run, L, cfg)
    missing = [str(p) for p in exp if not p.exists()]
    rep = {"job": job, "run": str(run), "utc": utc(), "ok": not missing and all(c["ok"] for c in items),
           "expected": [str(p) for p in exp], "missing": missing, "failed": [c["name"] for c in items if not c["ok"]],
           "exit_code": (run / "job" / "exit_code").read_text().strip() if (run / "job" / "exit_code").is_file()
           else None}
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
        raise Refused(f"{run}/collect.json missing: run `pilot_tvr collect {job}` first")
    if not c.get("ok") and not allow_incomplete:
        raise Refused(f"collect found problems in {job} (missing {len(c.get('missing') or [])}, failed "
                      f"{c.get('failed')}); pass --allow-incomplete to delete the record anyway")
    if c.get("ok"):
        gone = [p for p in c.get("expected") or [] if not Path(p).exists()]
        if gone:
            raise Refused(f"outputs vanished since collect: {gone[:3]}; collect again")
    return vsco.delete(run)


def status(L: Layout = None) -> dict:
    from vpd import sco as vsco

    L = L or LAYOUT
    st = vsco.status(include_ended=True)
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
    return {"ledger_jobs": [r for r in st["jobs"] if r.get("gate") == GATE], "not_in_ledger": st.get("not_in_ledger"),
            "runs": runs, "budget_tvr": st["budget"]["by_gate"][GATE]}


def budget(L: Layout = None, cfg=None, events=None) -> dict:
    from vpd import ledger

    L = L or LAYOUT
    events = ledger.read() if events is None else events
    rows = {}
    for job in JOBS:
        try:
            rows[job] = est_high(job, L, cfg)["est_high"]
        except (Refused, ValueError, KeyError) as e:
            rows[job] = f"not derivable yet: {getattr(e, 'message', e)}"[:200]
    return {"round_budget": round_budget(0.0, events, cfg), "est_high": rows, "ledger": ledger.summarise(events)}


def plan(L: Layout = None, cfg=None, out=None) -> dict:
    L = L or LAYOUT
    out = out or sys.stdout
    p = lambda *a: print(*a, file=out)  # noqa: E731
    p(f"VPD TVR pilot plan ({utc()}); n_top {pilot_cfg.get('tiers.n_top', cfg)!r}")
    rows = {}
    for job, spec in JOBS.items():
        try:
            e = est_high(job, L, cfg)
            pay = " ".join(payload_for(job, L.jobs / f"{job}-<yyyymmddHHMM>", e["est_high"], cfg))
            est = f"est_high {e['est_high']:.2f} GPU-h ({e['rule'][:60]})"
        except (Refused, ValueError, KeyError) as err:
            pay, est = "-", f"est_high not derivable yet ({str(getattr(err, 'message', err))[:120]})"
        last = latest_run(job, L=L)
        try:
            bad = prerequisites(job, L, cfg)
        except (ValueError, KeyError) as err:  # e.g. tiers.n_top not frozen yet
            bad = [f"{type(err).__name__}: {err}"]
        rows[job] = {"est": est, "prerequisites": bad}
        p(f"\n[{job}] sco name downstream-{spec['name']}, {'as_root' if spec['as_root'] else 'non-root'}, gate {GATE}, "
          f"{est}")
        p(f"  payload: {pay}")
        p(f"  latest run: {last} ({run_state(last)})" if last else "  latest run: none")
        p("  prerequisites: " + ("ready" if not bad else "; ".join(bad)[:600]))
    try:
        rb = round_budget(0.0, None, cfg)
        p(f"\nround budget: used {rb['round_used']:.2f} + open {rb['open_est']:.2f} = {rb['total']:.2f} of "
          f"{rb['cap']:.0f} GPU-h")
    except Exception as e:  # noqa: BLE001  (an unreadable ledger must not hide the plan)
        p(f"\nround budget: unreadable ({e!r})")
    return rows


# --- in-job helpers: steps, expect -----------------------------------------------------------------------------

def _chk(name: str, ok, detail=None, path=None) -> dict:
    c = {"name": name, "ok": bool(ok), "detail": detail}
    if path is not None:
        c["path"] = str(path)
    return c


def read_steps(run: Path) -> List[dict]:
    p = Path(run) / "steps.jsonl"
    out = []
    if p.is_file():
        for line in p.read_text().splitlines():
            try:
                out.append(json.loads(line))
            except ValueError:
                out.append({"name": "?unparseable", "rc": None, "line": line[:200]})
    return out


def required_steps(job: str, cfg=None) -> List[str]:
    spec = job_spec(job)
    kind, s = spec["kind"], spec["start"]
    if kind == "data":
        return ["node", "guard", "dev_select", "houses", "reach_dump", "replay", "demos_build", "tiers", "bfs",
                "floors"]
    if kind == "cal":
        return ["guard", "merge_disk", "dry", "hfparity", "shake", "canary_replay", "canary_parity", "train_0W",
                "train_100", "train_ALL", "merge_ALL", "merge_parity_ALL", "eval_ALL", "cal_decide", "merge_0W",
                "eval_0W", "merge_100", "eval_100"]
    if kind == "cal400":
        return ["guard", "merge_disk", "canary_replay", "canary_parity", "train_400", "merge_400", "merge_parity_400",
                "eval_400", "cal_decide"]
    if kind == "msmoke":
        out = ["guard", "prereg_committed"]
        for st in STARTS:
            out += [f"dry_{st}", f"hfparity_{st}", f"session_{st}"]
        return out + ["replay2", "render_parity", "verdict"]
    if kind == "train":
        return ["guard", "prereg_committed"] + [f"train_{t}" for t, _ in train_tiers(s, cfg)]
    out = ["guard", "prereg_committed", "merge_disk", "canary_replay", "canary_parity"]
    tiers = []
    for tier, _arm in eval_plan(s, spec["part"], cfg):
        if tier not in tiers:
            tiers.append(tier)
    for tier in tiers:
        out += [f"merge_{tier}", f"merge_parity_{tier}", f"eval_{tier}"]
    return out + [f"official_{tier}_{arm}" for tier, arm in eval_plan(s, spec["part"], cfg)]


def steps_check(job: str, run: Path, cfg=None) -> dict:
    """Every recorded step rc 0 (a failed X is excused by a later X_fallback with rc 0, the cal's O9 retry) and every
    required step recorded (NAME or NAME:<suffix>, e.g. NAME:skip)."""
    steps = read_steps(run)
    names = [str(s.get("name")) for s in steps]
    failed = []
    for i, s in enumerate(steps):
        if s.get("rc") != 0:
            n = str(s.get("name"))
            if any(str(t.get("name")) == n + "_fallback" and t.get("rc") == 0 for t in steps[i + 1:]):
                continue
            failed.append(n)
    absent = [r for r in required_steps(job, cfg)
              if not any(n == r or n.startswith(r + ":") for n in names)]
    return {"ok": bool(steps) and not failed and not absent, "failed": failed, "absent": absent, "n": len(steps)}


def _results(d: Path) -> Optional[dict]:
    return read_json(Path(d) / "results.json")


def _valid_results(d: Path, n: Optional[int] = None) -> Tuple[bool, dict]:
    r = _results(d)
    if r is None:
        return False, {"dir": str(d), "results": None, "invalid": (Path(d) / "results_invalid.json").is_file(),
                       "partial": (Path(d) / "results_partial.json").is_file()}
    ok = r.get("valid") is True and not r.get("tasks_missing") and (n is None or r.get("n_tasks") == n)
    return ok, {"dir": str(d), "valid": r.get("valid"), "n_tasks": r.get("n_tasks"),
                "tasks_missing": (r.get("tasks_missing") or [])[:5]}


def parity_state(path: Path, adapter_dir: Path) -> Tuple[int, str]:
    """0 current + pass; 4 current (same adapter sha) + failed (not rerun); 1 missing / other adapter sha."""
    rep = read_json(path)
    pt = Path(adapter_dir) / "adapter.pt"
    disk = sha256_file(pt) if pt.is_file() else None
    if rep is None:
        return 1, f"{path}: missing"
    if disk is None or rep.get("adapter_sha256") != disk:
        return 1, f"{path}: adapter sha {str(rep.get('adapter_sha256'))[:12]} != {pt} {str(disk)[:12]}"
    if rep.get("pass") is not True:
        return 4, f"{path}: current but FAILED (mean |dCE| {rep.get('mean_abs_dce')}, agreement " \
                  f"{rep.get('reply_agreement')})"
    return 0, f"{path}: pass, adapter {disk[:12]}"


def _rp_pass(p: Path) -> Tuple[bool, dict]:
    r = read_json(p)
    return bool(r and r.get("pass") is True), None if r is None else {
        "pass": r.get("pass"), "n_frames": r.get("n_frames"), "n_pass": r.get("n_pass"),
        "min_psnr_db": r.get("min_psnr_db"), "min_ssim": r.get("min_ssim"), "exact": r.get("exact")}


def expect_items(job: str, run: Path, L: Layout = None, cfg=None, files_only: bool = False) -> List[dict]:
    """The job's expect list (CONTRACT 7.x) as checks [{name, ok, detail, path}]; files_only: just the checks that
    name a file (for collect)."""
    L = L or LAYOUT
    run = Path(run)
    spec = job_spec(job)
    kind, s, part = spec["kind"], spec["start"], spec["part"]
    out: List[dict] = []

    def fexists(name, p):
        out.append(_chk(name, Path(p).is_file(), None, p))

    if kind == "data":
        sel = read_json(L.selection) or {}
        out.append(_chk("selection", sel.get("schema") == "vpd.tvr.dev/1" and len(sel.get("ids") or []) ==
                        pilot_cfg.get("dev.n", cfg), {"n": len(sel.get("ids") or [])}, L.selection))
        fexists("houses", L.houses)
        fexists("reach_index", L.reach / "index.json")
        fexists("reach_done", L.reach / "DONE")
        ok, det = _valid_results(L.replay, PUBLIC_N)
        out.append(_chk("replay_results", ok, det, L.replay / "results.json"))
        for f in ("demos_all.jsonl", "dropped.json", "kin_check.json", "DONE"):
            fexists(f"demos_{f}", L.demos / f)
        man = read_json(L.demos / "manifest.json") or {}
        kept = man.get("kept")
        out.append(_chk("demos_kept", isinstance(kept, int) and kept >= pilot_cfg.get("gates.min_kept", cfg),
                        {"kept": kept, "min_kept": pilot_cfg.get("gates.min_kept", cfg)}, L.demos / "manifest.json"))
        out.append(_chk("kin_equal_all_kept", man.get("kin_equal_all_kept") is True,
                        man.get("kin_equal_all_kept"), L.demos / "manifest.json"))
        for f in ("warmup0w20.jsonl", "demos_100.jsonl", "demos_400.jsonl", "demos_ALL.jsonl", "parity3.jsonl",
                  "order.json", "manifest.json"):
            fexists(f"tiers_{f}", L.tiers / f)
        for st in ("eval500", "dev200", "sft1600"):
            fexists(f"bfs_{st}", L.bfs / f"{st}_opt.json")
        for st in ("eval500", "dev200"):
            fl = read_json(L.floors / f"floors_{st}.json")
            out.append(_chk(f"floors_{st}", fl is not None and fl.get("warnings") == [],
                            None if fl is None else {"warnings": fl.get("warnings"), "F": fl.get("F")},
                            L.floors / f"floors_{st}.json"))
    elif kind in ("cal", "cal400"):
        C = L.cal
        if kind == "cal":
            dry = read_json(C / "dry" / "dry_run.json")
            out.append(_chk("dry_run", dry is not None and dry_grid_ok(dry, cfg)[0],
                            None if dry is None else dry_grid_ok(dry, cfg)[1], C / "dry" / "dry_run.json"))
            par = read_json(C / "dry" / "parity.json")
            out.append(_chk("hf_parity", bool(par) and par.get("verdict") == "PASS",
                            None if not par else par.get("verdict"), C / "dry" / "parity.json"))
            for arm in ("c", "blind"):
                ok, det = _valid_results(L.cal_shake(arm), SHAKE_N)
                out.append(_chk(f"shake_{arm}", ok, det, L.cal_shake(arm) / "results.json"))
            ok, det = _rp_pass(C / "canary" / "render_parity.json")
            out.append(_chk("canary", ok, det, C / "canary" / "render_parity.json"))
            tiers = [("0W", None), ("100", "0W"), ("ALL", "0W")]
            parity_tiers = ["ALL"]
            evals = ["0W", "100", "ALL"]
        else:
            ok, det = _rp_pass(C / "canary400" / "render_parity.json")
            out.append(_chk("canary400", ok, det, C / "canary400" / "render_parity.json"))
            tiers, parity_tiers, evals = [("400", "0W")], ["400"], ["400"]
        if not files_only:
            for tier, init in tiers:
                rc, msg = adapter_check(L.cal_adapter(tier), None if init is None else L.cal_adapter(init))
                out.append(_chk(f"adapter_{tier}", rc == 0, msg))
        for tier in parity_tiers:
            p = C / "parity" / f"merge_{tier}.json"
            rc, msg = parity_state(p, L.cal_adapter(tier))
            out.append(_chk(f"merge_parity_{tier}", rc == 0, msg, p))
        for tier in evals:
            ok, det = _valid_results(L.cal_eval(tier), pilot_cfg.get("dev.n", cfg))
            out.append(_chk(f"eval_{tier}", ok, det, L.cal_eval(tier) / "results.json"))
        d = read_json(L.cal_decision)
        ok = d is not None and d.get("n_top") in ("ALL", "400") and (kind == "cal" or d.get("with_400") is True)
        out.append(_chk("cal_decision", ok, None if d is None else {k: d.get(k) for k in (
            "n_top", "train_400", "futility", "with_400")}, L.cal_decision))
    elif kind == "msmoke":
        v = read_json(L.smoke_dir(run) / "smoke_verdict.json")
        out.append(_chk("smoke_verdict", bool(v) and v.get("verdict") == "PASS",
                        None if not v else {"verdict": v.get("verdict"), "failed": v.get("failed")},
                        L.smoke_dir(run) / "smoke_verdict.json"))
    elif kind == "train":
        for tier, init in train_tiers(s, cfg):
            if files_only:
                out.append(_chk(f"adapter_{tier}", True, None, L.adapter(s, tier) / "adapter.pt"))
                continue
            rc, msg = adapter_check(L.adapter(s, tier), None if init is None else L.adapter(s, init))
            out.append(_chk(f"adapter_{tier}", rc == 0, msg, L.adapter(s, tier) / "adapter.pt"))
    else:  # eval
        ok, det = _rp_pass(L.canary(s, part))
        out.append(_chk(f"canary_{part}", ok, det, L.canary(s, part)))
        seen = []
        for tier, arm in eval_plan(s, part, cfg):
            ad = L.adapter(s, EVAL_ADAPTER[tier])
            if tier not in seen:
                seen.append(tier)
                rc, msg = parity_state(L.parity(s, tier), ad)
                out.append(_chk(f"merge_parity_{tier}", rc == 0, msg, L.parity(s, tier)))
            d = L.eval_dir(s, tier, arm)
            ok, det = _valid_results(d, EVAL500_N)
            if ok and not files_only:
                rc, msg = eval_check(d, ad)
                ok, det = rc == 0, dict(det, adapter=msg)
            out.append(_chk(f"eval_{tier}_{arm}", ok, det, d / "results.json"))
            oc = read_json(d / "official_check.json")
            out.append(_chk(f"official_{tier}_{arm}", bool(oc) and oc.get("equal") is True,
                            oc, d / "official_check.json"))
    if files_only:
        return [c for c in out if c.get("path")]
    return out


def expect(job: str, run: Path, L: Layout = None, cfg=None) -> dict:
    L = L or LAYOUT
    items = expect_items(job, run, L, cfg)
    st = steps_check(job, run, cfg)
    items.append(_chk("steps", st["ok"], st))
    rep = {"schema": "vpd.tvr.expect/1", "job": job, "run": str(run), "utc": utc(),
           "ok": all(c["ok"] for c in items), "failed": [c["name"] for c in items if not c["ok"]], "items": items}
    write_json(Path(run) / "expect.json", rep)
    return rep


# --- in-job helpers: guard -----------------------------------------------------------------------------------

def guard(job: str, run: Path, L: Layout = None, cfg=None) -> dict:
    """The job's in-job guard: prerequisites(in_job=True) (the prereg's committed state comes from the snapshot,
    checked by the payload's prereg_committed step) plus, for tvr-data, the pinned TVRBench file shas, the badge
    font sha and the probe compare verdict."""
    L = L or LAYOUT
    bad = prerequisites(job, L, cfg, in_job=True)
    facts = {}
    if job_spec(job)["kind"] == "data":
        repo = paths.TVR_REPO
        for rel, want in pilot_cfg.get("tvrbench.files_sha256", cfg).items():
            p = repo / rel
            got = sha256_file(p) if p.is_file() else None
            if got != want:
                bad.append(f"{p}: sha256 {str(got)[:12]} != pinned {want[:12]}")
        from vpd.tvr import lvp

        font = Path(lvp.BADGE_FONT)
        got = sha256_file(font) if font.is_file() else None
        if got != pilot_cfg.get("render.badge_font_sha256", cfg):
            bad.append(f"badge font {font}: sha256 {str(got)[:12]} != render.badge_font_sha256")
        verdict = pilot_cfg.get("render.compare_verdict", cfg)
        facts["compare_verdict"] = verdict
        if verdict is None:
            bad.append("render.compare_verdict is not recorded (probe fill)")
        pr = pilot_cfg.get("render.probe_run", cfg)
        want = pilot_cfg.get("render.probe_summary_sha256", cfg)
        if pr and want:
            sp = Path(str(pr).replace("OUT/", str(L.root) + "/", 1)) / "out" / "summary.json"
            got = sha256_file(sp) if sp.is_file() else None
            facts["probe_summary"] = {"path": str(sp), "sha256": got}
            if got is None:
                bad.append(f"{sp}: missing (render.* was filled from it)")
            elif got != want:
                bad.append(f"{sp}: sha256 {got[:12]} != render.probe_summary_sha256 {want[:12]}")
    rep = {"job": job, "run": str(run), "utc": utc(), "ok": not bad, "problems": bad, "facts": facts}
    write_json(Path(run) / "guard.json", rep)
    return rep


# --- in-job helpers: smoke verdict -----------------------------------------------------------------------------

def dry_grid_ok(dry: dict, cfg=None) -> Tuple[bool, dict]:
    """Every conversation's image grids == protocol.image_grid and its <|image_pad|> count == image_pad_tokens per
    image; tokens per image = prod(grid)/4 + 2 (run_vpd's B4 convention) == protocol.image_tokens."""
    grid = list(pilot_cfg.get("protocol.image_grid", cfg))
    pad = int(pilot_cfg.get("protocol.image_pad_tokens", cfg))
    want_tok = int(pilot_cfg.get("protocol.image_tokens", cfg))
    per = dry.get("per_conversation") or []
    bad = []
    for c in per:
        gs = [list(g) for g in c.get("grid_set") or []]
        if gs != [grid]:
            bad.append({"id": c.get("id"), "grid_set": gs})
        elif not c.get("images") or c.get("image_tokens") != pad * c["images"]:
            bad.append({"id": c.get("id"), "image_tokens": c.get("image_tokens"), "images": c.get("images")})
    tok = grid[0] * grid[1] * grid[2] // 4 + 2
    ok = bool(per) and not bad and tok == want_tok
    return ok, {"conversations": len(per), "grid": grid, "tokens_per_image": tok, "bad": bad[:5]}


def worker_problems(rr, cfg=None) -> List[dict]:
    """[] iff run_tvr's renderer block has workers_ok true and EVERY worker's own ready reply (renderer.workers) has
    profile == render.lvp_profile, L == render.lp_threads, the pinned font (mode "pinned", sha ==
    render.badge_font_sha256), the pinned system prompt and the pinned TVRBench files (system_prompt_ok,
    tvrbench_files_ok); else what is wrong, per worker.  The block's own profile / lp are the harness's request,
    not what the workers ran (gate G0(4), smoke-verdict renderer_<S>)."""
    rr = rr if isinstance(rr, dict) else {}
    want = {"profile": pilot_cfg.get("render.lvp_profile", cfg), "lp": str(pilot_cfg.get("render.lp_threads", cfg)),
            "font_mode": "pinned", "font_sha256": pilot_cfg.get("render.badge_font_sha256", cfg),
            "system_prompt_ok": True, "tvrbench_files_ok": True}
    ws = rr.get("workers")
    out = [] if isinstance(ws, list) and ws else [{"workers": "no worker ready reply"}]
    for w in ws if isinstance(ws, list) else []:
        w = w if isinstance(w, dict) else {}
        got = dict({k: w.get(k) for k in want}, lp=str(w.get("lp")))
        wrong = {k: got[k] for k in want if got[k] != want[k]}
        if wrong:
            out.append({"wid": w.get("wid"), **wrong})
    if rr.get("workers_ok") is not True and not out:
        out.append({"workers_ok": rr.get("workers_ok")})
    return out


def smoke_verdict(smoke: Path, L: Layout = None, cfg=None) -> dict:
    """SMOKE/smoke_verdict.json (CONTRACT 7.3): mechanism only.  Reads the dry runs, parity files, run validity and
    mechanism fields of results.json (valid, n_tasks, tasks_missing, invalid, audit, renderer, perf) and
    render_parity.json; never a success / SR field."""
    L = L or LAYOUT
    smoke = Path(smoke)
    checks, measured = [], {}
    prof = pilot_cfg.get("render.lvp_profile", cfg)
    font = pilot_cfg.get("render.badge_font_sha256", cfg)
    lp = pilot_cfg.get("render.lp_threads", cfg)
    grid = list(pilot_cfg.get("protocol.image_grid", cfg))
    tok = int(pilot_cfg.get("protocol.image_tokens", cfg))
    for s in STARTS:
        dry = read_json(smoke / s / "dry" / "dry_run.json")
        ok, det = (False, "missing") if dry is None else dry_grid_ok(dry, cfg)
        checks.append(_chk(f"dry_{s}", ok, det))
        par = read_json(smoke / s / "dry" / "parity.json")
        checks.append(_chk(f"hfparity_{s}", bool(par) and par.get("verdict") == "PASS",
                           None if not par else {"verdict": par.get("verdict"), "problems": par.get("problems")}))
        ev = smoke / s / "evals"
        vp = read_json(ev / f"smoke__{s}__raw__vllm_parity.json")
        checks.append(_chk(f"vllm_parity_{s}", bool(vp) and vp.get("equal") is True,
                           None if not vp else {"equal": vp.get("equal"), "n": vp.get("n")}))
        res = {a: _results(ev / f"smoke__{s}__raw__{a}") for a in ARMS_TOP}
        bad = []
        for a, r in res.items():
            if r is None:
                bad.append(f"{a}: no results.json")
            elif r.get("valid") is not True or r.get("tasks_missing") or r.get("n_tasks") != smoke_n(cfg):
                bad.append(f"{a}: valid {r.get('valid')}, n_tasks {r.get('n_tasks')}, missing "
                           f"{(r.get('tasks_missing') or [])[:3]}")
        checks.append(_chk(f"runs_{s}", not bad, bad))
        bad = []
        for a in ("c", "blind"):
            inv = (res[a] or {}).get("invalid") or {}
            if not (inv.get("regex_mismatch") == 0 and (inv.get("replies") or 0) > 0):
                bad.append({a: {"regex_mismatch": inv.get("regex_mismatch"), "replies": inv.get("replies")}})
        checks.append(_chk(f"regex_{s}", not bad, bad))
        inv_u = (res["u"] or {}).get("invalid") or {}
        rate = None
        if inv_u.get("replies"):
            rate = round(1 - (inv_u.get("unparseable_steps") or 0) / inv_u["replies"], 4)
        checks.append(_chk(f"u_arm_{s}", res["u"] is not None, {"parse_rate": rate, "judged": False}))
        bad = []
        for a in IMAGE_ARMS:
            au = (res[a] or {}).get("audit") or {}
            pp = (au.get("prompt_parity") or {}).get("equal")
            it = au.get("image_tokens") or {}
            tpi = it.get("tokens_per_image")
            tpi = list(tpi) if isinstance(tpi, (list, tuple)) else [tpi]
            gs = [list(g) for g in it.get("grid_set") or []]
            if pp is not True or tpi != [tok] or gs != [grid]:
                bad.append({a: {"prompt_parity": pp, "tokens_per_image": tpi, "grid_set": gs}})
        checks.append(_chk(f"audit_{s}", not bad, bad))
        bad = []
        for a in IMAGE_ARMS:
            rr = (res[a] or {}).get("renderer") or {}
            got = {"profile": rr.get("profile"), "font_sha256": rr.get("font_sha256"), "lp": rr.get("lp")}
            wp = worker_problems(rr, cfg)
            if got != {"profile": prof, "font_sha256": font, "lp": lp} or prof is None or lp is None or wp:
                bad.append({a: dict(got, workers_ok=rr.get("workers_ok"), workers=wp[:5])})
        checks.append(_chk(f"renderer_{s}", not bad, bad))
        measured[s] = {a: {k: ((res[a] or {}).get("perf") or {}).get(k) for k in (
            "steps_per_s", "vllm_ready_s", "max_images_per_request", "max_prompt_tokens", "cgroup_cores_max")}
            for a in ARMS_TOP}
    ok, det = _rp_pass(smoke / "replay2" / "render_parity.json")
    checks.append(_chk("render_parity", ok, det))
    failed = [c["name"] for c in checks if not c["ok"]]
    v = {"schema": "vpd.tvr.smoke_verdict/1", "utc": utc(), "smoke": str(smoke),
         "verdict": "PASS" if not failed else "FAIL", "failed": failed, "checks": checks, "measured": measured,
         "note": "mechanism only: no success rate is read or reported"}
    write_json(smoke / "smoke_verdict.json", v)
    return v


# --- in-job helpers: calibration decision ----------------------------------------------------------------------

def _wilson(k: int, n: int, cfg=None) -> List[float]:
    from vpd.walkerbench.pilot_gate import wilson_lb, wilson_ub

    z = float(pilot_cfg.get("gates.z", cfg))
    return [wilson_lb(k, n, z), wilson_ub(k, n, z)]


def cal_decide(cal_root: Path, L: Layout = None, cfg=None, floors_path: Optional[Path] = None) -> dict:
    """CAL/cal_decision.json (CONTRACT 6.10, G10(e); EXPLORATORY, Q3B only): n_top = ALL iff the ALL dev200 c rate
    <= tiers.all_dev_sr_max (equality -> ALL), else 400 (train_400); futility = the Wilson LB of the chosen top tier's
    dev200 c eval <= the dev200 floor F (null until the 400 eval exists when n_top is 400)."""
    L = L or LAYOUT
    cal_root = Path(cal_root)
    p_all = cal_root / "evals" / "cal__Q3B__ALL__c" / "results.json"
    r = read_json(p_all)
    if r is None or r.get("valid") is not True:
        raise Refused(f"{p_all}: missing or not valid")
    k, n = int(r["sr"]["k"]), int(r["sr"]["n"])
    if n != pilot_cfg.get("dev.n", cfg):
        raise Refused(f"{p_all}: n {n} != dev.n {pilot_cfg.get('dev.n', cfg)}")
    thr = pilot_cfg.get("tiers.all_dev_sr_max", cfg)
    top = "ALL" if fractions.Fraction(k, n) <= fractions.Fraction(str(thr)) else "400"
    fpath = Path(floors_path or L.floors / "floors_dev200.json")
    fl = read_json(fpath)
    if fl is None or not isinstance((fl.get("F") or {}).get("rate"), (int, float)):
        raise Refused(f"{fpath}: no floor F")
    F = float(fl["F"]["rate"])
    out = {"schema": "vpd.tvr.cal_decision/1", "exploratory": True, "start": "Q3B", "task_set": "dev200",
           "all_dev": {"k": k, "n": n, "rate": k / n, "wilson95": _wilson(k, n, cfg)},
           "rule": f"n_top = ALL if rate <= {thr} else 400", "n_top": top, "train_400": top == "400",
           "floor_dev200": {"F": F, "source": fl["F"].get("source"), "file": str(fpath),
                            "sha256": sha256_file(fpath)},
           "results_sha256": sha256_file(p_all), "utc": utc()}
    if top == "ALL":
        tk, tn, tsrc = k, n, p_all
    else:
        p400 = cal_root / "evals" / "cal__Q3B__400__c" / "results.json"
        r4 = read_json(p400)
        tk = tn = tsrc = None
        if r4 is not None and r4.get("valid") is True:
            tk, tn, tsrc = int(r4["sr"]["k"]), int(r4["sr"]["n"]), p400
            out["results400_sha256"] = sha256_file(p400)
    out["with_400"] = top == "400" and tk is not None
    if tk is None:
        out.update(futility=None, futility_pending="n_top 400: re-decided by cal-decide after job tvr-cal400")
    else:
        lb = _wilson(tk, tn, cfg)[0]
        out.update(futility=bool(lb <= F), top_dev={"tier": top, "k": tk, "n": tn, "rate": tk / tn,
                                                    "wilson95": _wilson(tk, tn, cfg), "results": str(tsrc)},
                   futility_rule="Wilson LB of the top tier's dev200 c <= floor_dev200 F -> stop and ask the user")
    old = read_json(cal_root / "cal_decision.json")
    if old is not None and old.get("n_top") not in (None, top):
        raise Refused(f"cal_decision.json says n_top {old.get('n_top')!r}, now {top!r}: refusing to change it")
    write_json(cal_root / "cal_decision.json", out)
    return out


# --- CLI ----------------------------------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan")
    p = sub.add_parser("prepare")
    p.add_argument("job", choices=sorted(JOBS))
    p.add_argument("--pool", default="zoetrope", choices=("zoetrope", "h800", "auto"))
    p.add_argument("--new", action="store_true")
    for name in ("submit", "collect"):
        sub.add_parser(name).add_argument("job", choices=sorted(JOBS))
    sub.add_parser("status")
    p = sub.add_parser("delete")
    p.add_argument("job", choices=sorted(JOBS))
    p.add_argument("--allow-incomplete", action="store_true")
    sub.add_parser("budget")
    sub.add_parser("data-fill")
    for name in ("expect", "guard"):
        p = sub.add_parser(name)
        p.add_argument("job", choices=sorted(JOBS))
        p.add_argument("--run", type=Path, required=True)
    p = sub.add_parser("smoke-verdict")
    p.add_argument("--smoke", type=Path, required=True)
    p = sub.add_parser("cal-decide")
    p.add_argument("--cal-root", type=Path, required=True)
    p.add_argument("--floors", type=Path, default=None, help="default OUT/tvr/floors/floors_dev200.json")
    p = sub.add_parser("cfg")
    p.add_argument("keys", nargs="+")
    p = sub.add_parser("served")
    p.add_argument("start", choices=STARTS)
    p = sub.add_parser("parity-ok")
    p.add_argument("file", type=Path)
    p.add_argument("--adapter", type=Path, required=True)
    args = ap.parse_args(argv)
    try:
        if args.cmd == "plan":
            plan()
        elif args.cmd == "prepare":
            print(prepare(args.job, args.pool, args.new))
        elif args.cmd == "submit":
            print(submit(args.job))
        elif args.cmd == "status":
            print(json.dumps(status(), indent=1, default=str))
        elif args.cmd == "collect":
            rep = collect(args.job)
            print(json.dumps({k: rep[k] for k in rep if k != "expected"}, indent=1, default=str))
            return 0 if rep["ok"] else 1
        elif args.cmd == "delete":
            print(delete(args.job, args.allow_incomplete))
        elif args.cmd == "budget":
            print(json.dumps(budget(), indent=1, default=str))
        elif args.cmd == "data-fill":
            rep = data_fill()
            print(json.dumps(rep, indent=1))
            return 0 if not rep["problems"] else 1
        elif args.cmd == "expect":
            rep = expect(args.job, args.run)
            print(json.dumps({"ok": rep["ok"], "failed": rep["failed"]}))
            return 0 if rep["ok"] else 1
        elif args.cmd == "guard":
            rep = guard(args.job, args.run)
            for b in rep["problems"]:
                print(f"[pilot_tvr] guard: {b}", file=sys.stderr)
            print(json.dumps({"ok": rep["ok"], "problems": len(rep["problems"])}))
            return 0 if rep["ok"] else 2
        elif args.cmd == "smoke-verdict":
            v = smoke_verdict(args.smoke)
            print(json.dumps({"verdict": v["verdict"], "failed": v["failed"]}))
            return 0 if v["verdict"] == "PASS" else 1
        elif args.cmd == "cal-decide":
            d = cal_decide(args.cal_root, floors_path=args.floors)
            print(json.dumps({k: d.get(k) for k in ("n_top", "train_400", "futility", "with_400")}))
        elif args.cmd == "cfg":
            vals = [pilot_cfg.get(k) for k in args.keys]
            print(" ".join("None" if v is None else str(v) for v in vals))
            if any(v is None for v in vals):
                print(f"[pilot_tvr] cfg: {[k for k, v in zip(args.keys, vals) if v is None]} not filled",
                      file=sys.stderr)
                return 2
        elif args.cmd == "served":
            from vpd import starts

            print(starts.resolve(args.start)["code"])
        elif args.cmd == "parity-ok":
            rc, msg = parity_state(args.file, args.adapter)
            print(f"[parity-ok] {msg}")
            return rc
    except Refused as e:
        print(f"[pilot_tvr] REFUSED: {e.message}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
