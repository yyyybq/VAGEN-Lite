"""TVR pilot orchestration (CONTRACT_wave2 sections 5.3.1, 7; DECISIONS G3, G10; mirrors vpd/walkerbench/pilot.py).

    PY_EVAL -m vpd.tvr.pilot_tvr plan [--explore]   # --explore: the EXPLORATORY jobs (XJOBS) and the block budget
    PY_EVAL -m vpd.tvr.pilot_tvr prepare JOB [--pool zoetrope|h800|auto] [--new]
    PY_EVAL -m vpd.tvr.pilot_tvr submit JOB
    PY_EVAL -m vpd.tvr.pilot_tvr status [--explore]
    PY_EVAL -m vpd.tvr.pilot_tvr collect JOB
    PY_EVAL -m vpd.tvr.pilot_tvr delete JOB [--allow-incomplete]
    PY_EVAL -m vpd.tvr.pilot_tvr budget [--explore]
    PY_EVAL -m vpd.tvr.pilot_tvr data-fill        # the 13 data.* values from the data job's files (read only)
  in-job helpers (pure; called by vpd/jobs/tvr_*.sh):
    ... pilot_tvr expect JOB --run RUN            # RUN/expect.json; exit 0 iff every expected item holds
    ... pilot_tvr smoke-verdict --smoke DIR       # DIR/smoke_verdict.json (mechanism only, never an SR)
    ... pilot_tvr xsmoke-verdict --smoke DIR [--job tvr-xsmoke-q3sg]  # an explore smoke's DIR/smoke_verdict.json
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

EXPLORATORY RL starts (PREREG_TVR_PILOT.md section 17, row 2026-11-12; DECISIONS G18): RL_STARTS = (Q3RD,
Q3RS150) through the same recipe (seed 1001, tiers/v1, 0W warm-up, N_top, eval500, frozen protocol, the section-16
est_high and timeout rules, as_root, zoetrope), outputs under OUT/tvr/runs/explore/rl/<S> (adapters/, evals/
xrl__<S>__<tier>__<arm>, parity/, canary/), never runs/pilot.  Not gated: outside the decision table, A4 and the
confirmatory round.  Separate job table XJOBS (JOBS, and every pilot job's payload, guard and expect, are unchanged;
`plan`, `status` and `budget` show the explore jobs only with --explore):
    tvr-xsmoke            as_root   tvr_xsmoke.sh RUN                       est_high 1.2 (constant, XEST_HIGH)
    tvr-xtrain-<s>        as_root   VPD_TVR_ROOT=explore/rl tvr_train.sh START RUN       est_high DERIVED (as train)
    tvr-xeval-<s>-top|low as_root   VPD_TVR_ROOT=explore/rl tvr_eval.sh START top|low RUN est_high DERIVED (as eval)
Explore payload: `env VPD_JOB_DEADLINE_S=<soft> [VPD_TVR_ROOT=explore/rl] timeout -k <kill> <outer> bash ...`.
Explore guards (prepare, submit, in-job `guard`): as the pilot's (missing(), PAYLOAD_CONSTANTS, the trainer prefix,
the prereg committed and clean and its block == pilot_cfg, the data shas incl. the tiers manifest, n_top, futility)
plus: the start in EXPLORE_STARTS with a registry code, its checkpoint dir present and its processor read from that
dir (no eval view, DECISIONS G18); the run_tvr harness code_sha256 starts with HARNESS_SHA256_PREFIX; exactly one
section-17 row of 2026-11-12 in the prereg (XROW_NEEDLE); the tier files the job reads == tiers/v1/manifest.json
files; train / eval: the newest explore smoke verdict (runs/explore/rl/smoke/<RUN>/smoke_verdict.json) PASS; eval:
the xtrain adapters adapter-ok with the right init, meta start / seed 1001 / phase / data_sha256 (= the manifest's).
Block budget (DECISIONS G18, the user's cap of this block): prepare / submit refuse (exit 2) when (ledger total
spent - BLOCK_BASELINE_GPUH) + the est of the open ledger jobs + the new est_high > BLOCK_CAP_GPUH, printing the
three numbers (RUN/prepared.json["block_budget"]); the round-200 check above applies too.  Merge parity in the
explore evals is report only (DECISIONS G16 (1)): its json must exist for the served adapter; its pass is recorded,
not judged (steps_check excuses merge_parity_<x> followed by merge_parity_<x>:report_only).

EXPLORATORY third start Q3SG (genc0, registry code sft_q3genc0_e8; PREREG_TVR_PILOT.md section 17, row 2026-11-13;
DECISIONS G20): EXPLORE_STARTS = RL_STARTS + Q3SG_STARTS.  Q3SG uses the same explore root (outputs under
OUT/tvr/runs/explore/rl/Q3SG: adapters/, evals/xrl__Q3SG__<tier>__<arm>, parity/, canary/; "rl" is only the directory
name), the same recipe, guards, est_high rules and block budget (BLOCK_BASELINE_GPUH 48.5136, BLOCK_CAP_GPUH 25.0)
as the RL starts.  Its jobs (XJOBS; the RL jobs and their payloads are unchanged):
    tvr-xsmoke-q3sg       as_root   tvr_xsmoke.sh Q3SG RUN (Q3SG only)            est_high 0.7 (constant, XEST_HIGH)
    tvr-xtrain-q3sg       as_root   VPD_TVR_ROOT=explore/rl tvr_train.sh Q3SG RUN        est_high DERIVED (as train)
    tvr-xeval-q3sg-top|low as_root  VPD_TVR_ROOT=explore/rl tvr_eval.sh Q3SG top|low RUN est_high DERIVED (as eval)
Per-start rules (only the Q3SG jobs differ):
  registration row: Q3RD / Q3RS150 need exactly one section-17 row of 2026-11-12 (XROW_NEEDLE), Q3SG exactly one of
  2026-11-13 (Q3SG_ROW_NEEDLE).
  trainer fingerprint: vpd/starts.json gained the Q3SG entry and no trainer byte changed, so the fingerprint of this
  tree is Q3SG_TRAINER_SHA256 (119da3c84b8d), not the pilot's PILOT_TRAINER_SHA256 (0865dd75f1ee).  The Q3SG jobs
  (prepare, submit, the in-job guard, xsmoke-verdict, expect, the xeval adapter checks) require code_sha256 starting
  with Q3SG_TRAINER_PREFIX and the invariant: the 5 .py entries of code_files == TRAINER_PY_SHA256 (the pilot / RL
  adapters' recorded shas) and vpd/starts.json without the Q3SG entry, re-serialised as json.dumps(table, indent=1)
  + "\n", has the pilot's sha STARTS_JSON_SHA256_PILOT (an adapter's code_files: starts.json ==
  STARTS_JSON_SHA256_Q3SG).  Every other job (pilot and RL) still requires recipe.trainer_code_sha256_prefix
  0865dd75f1ee (pilot_cfg, unchanged), so on this tree they refuse to run.  Intended: the TVR pilot is closed and the
  RL block is done; their adapters and results are unaffected.  x_adapter_check judges an explore adapter by its
  start's registered trainer code (RL: the recipe prefix; Q3SG: the Q3SG prefix and the invariant), not by the
  current tree, so compare_rl still validates the RL adapters.
  smoke verdict: tvr-xtrain-q3sg / tvr-xeval-q3sg-* need the newest smoke_verdict.json under runs/explore/rl/smoke/
  whose "starts" include Q3SG to be PASS (the RL starts: the newest tvr-xsmoke verdict, unchanged).
tvr-xsmoke-q3sg est_high: one start ~1,815 s (XEST_HIGH's per-start figure with the calibration's slowest vLLM start
and merge) + 300 + 180 s = 2,295 s = 0.64 h -> 0.7 GPU-h (rounded up to 0.1 h as for tvr-xsmoke's 1.2): outer 2220 s,
soft 2040 s, kill 60 s.
"""

from __future__ import annotations

import argparse
import datetime
import fractions
import hashlib
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

# EXPLORATORY starts (PREREG section 17, rows 2026-11-12 and 2026-11-13; DECISIONS G18, G20); see the module docstring
RL_STARTS = ("Q3RD", "Q3RS150")   # the RL starts (row 2026-11-12)
Q3SG_STARTS = ("Q3SG",)           # the third explore start, genc0 (row 2026-11-13)
EXPLORE_STARTS = RL_STARTS + Q3SG_STARTS
XRL_ROOT = "explore/rl"      # the explore payloads' VPD_TVR_ROOT: outputs under OUT/tvr/runs/explore/rl/<S>
XRL_PREFIX = "xrl"           # explore eval dirs <root>/<S>/evals/xrl__<S>__<tier>__<arm>
HARNESS_SHA256_PREFIX = "64edd575f36b"   # run_tvr.code_sha256() of the pilot evals (the registered harness)
XROW_DATE = "2026-11-12"
XROW_NEEDLE = "两个 RL 起点按试点完全相同的配方训练和评测"
Q3SG_ROW_DATE = "2026-11-13"
Q3SG_ROW_NEEDLE = "第三个探索起点 Q3SG 按试点完全相同的配方训练和评测"
BLOCK_BASELINE_GPUH = 48.5136   # ledger total spent when this block started (DECISIONS G18)
BLOCK_CAP_GPUH = 25.0           # the user's cap of this block (DECISIONS G18): above it, stop and ask
# the trainer fingerprint per start (row 2026-11-13; DECISIONS G20).  n2_train_vpd.CODE_FILES = the 5 .py files below
# + vpd/starts.json; the pilot and RL adapters recorded PILOT_TRAINER_SHA256 (meta.code_files: TRAINER_PY_SHA256 and
# STARTS_JSON_SHA256_PILOT, e.g. OUT/tvr/runs/explore/rl/Q3RD/adapters/0W/meta.json); starts.json + the Q3SG entry
# (STARTS_JSON_SHA256_Q3SG) gives Q3SG_TRAINER_SHA256
PILOT_TRAINER_SHA256 = "0865dd75f1ee51b889cce3c5b9f9e8d4df62fd4a70343532ea47eb9120e12a76"
Q3SG_TRAINER_SHA256 = "119da3c84b8dfd79aa22b2c7fd5c5641a1cf390b47deae3e8e852268dc46f4bf"
Q3SG_TRAINER_PREFIX = Q3SG_TRAINER_SHA256[:12]
STARTS_JSON = "vpd/starts.json"
STARTS_JSON_SHA256_PILOT = "86bb356e61d64f9f15ab8a8e25bc334eb3e6eaf37dfaa369bbc2243378d39cfb"
STARTS_JSON_SHA256_Q3SG = "8cd21cfa60769614cd86bcd18c42ca4d8e4e1cdf6ba6b1db589cda65225f9cbc"
TRAINER_PY_SHA256 = {
    "vpd/train/n2_train_vpd.py": "6c2b4056b337b0bcabb1d731976bec4d31f08baafe2ba8e0f0d841c337be6a97",
    "downstream/train/lora.py": "54e9a52b1cab63d23902d53dc30dacd11a44c62d3be6ba6a46af4cf03d4afce4",
    "vpd/starts.py": "727701fc58771fba9144f6f80910091de72756f2156769a59e8c3e1e7c982fcd",
    "vpd/paths.py": "4d16d91dd789d5d726341af0ef358e99ea3c4f798aff1724508ea8b49df1e91e",
    "vpd/walkerbench/history.py": "178c0b889fa78383bf204cd0d3152365fde6d4cdd9c4f148fe5eb79dab563af8"}
# tvr-xsmoke (both starts, one job): per start ~1,290 s typical (bf16 load + save 150, dry 45, HF parity 45, raw
# session 330, train 110, merge 150, merge parity 160, merged smoke4 session 300; tvr-msmoke / tvr-cal step times),
# ~1,815 s with the calibration's slowest vLLM start (505 s) and merge (186 s): 2 x 1,815 s + 300 + 180 -> 1.2 GPU-h.
# tvr-xsmoke-q3sg (Q3SG only): 1 x 1,815 s + 300 + 180 = 2,295 s = 0.64 h -> 0.7 GPU-h (rounded up to 0.1 h)
XEST_HIGH = {"tvr-xsmoke": 1.2, "tvr-xsmoke-q3sg": 0.7}
XSMOKE_LIMIT, XSMOKE_EPOCHS = 16, 1   # the smoke trains the first 16 0W demos for one epoch (the trainer has no step
XSMOKE_STEPS = 2                      # cap): ceil(16 x 1 / 8) = 2 optimizer steps
XSMOKE_RAW_SET, XSMOKE_MERGED_SET = "smoke8", "smoke4"


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

    # EXPLORATORY RL starts (runs/explore/rl; never runs/pilot)
    @property
    def xroot(self) -> Path:
        return self.runs / "explore" / "rl"

    @property
    def xsmoke_root(self) -> Path:
        return self.xroot / "smoke"

    def x_shared_parents(self) -> List[Path]:
        return [self.runs, self.runs / "explore", self.xroot, self.xsmoke_root] + [self.xroot / s
                                                                                   for s in EXPLORE_STARTS]

    def xsmoke_dir(self, run) -> Path:
        return self.xsmoke_root / Path(run).name

    def xstart_dir(self, s: str) -> Path:
        return self.xroot / s

    def xadapter(self, s: str, tier: str) -> Path:
        return self.xstart_dir(s) / "adapters" / tier

    def xeval_name(self, s: str, tier: str, arm: str) -> str:
        return f"{XRL_PREFIX}__{s}__{tier}__{arm}"

    def xeval_dir(self, s: str, tier: str, arm: str) -> Path:
        return self.xstart_dir(s) / "evals" / self.xeval_name(s, tier, arm)

    def xparity(self, s: str, tier: str) -> Path:
        return self.xstart_dir(s) / "parity" / f"merge_{tier}.json"

    def xcanary(self, s: str, part: str) -> Path:
        return self.xstart_dir(s) / "canary" / part / "render_parity.json"


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


def _xjobs() -> Dict[str, dict]:
    """The EXPLORATORY jobs (DECISIONS G18, G20); "base" = the pilot_cfg REQUIRED entry missing() checks; "starts" =
    the explore starts the job covers; "xlist" = tvr_xsmoke.sh's start-list argument (None: its default, the RL
    starts)."""
    j = {}
    for smoke, starts, xlist in (("tvr-xsmoke", RL_STARTS, None), ("tvr-xsmoke-q3sg", Q3SG_STARTS, "Q3SG")):
        j[smoke] = {"name": f"vpd-{smoke}", "script": "tvr_xsmoke.sh", "as_root": True, "kind": "xsmoke",
                    "start": None, "part": None, "base": "tvr-msmoke", "starts": starts, "xlist": xlist}
        for s in starts:
            sl = s.lower()
            j[f"tvr-xtrain-{sl}"] = {"name": f"vpd-tvr-xtrain-{sl}", "script": "tvr_train.sh", "as_root": True,
                                     "kind": "xtrain", "start": s, "part": None, "base": "tvr-train", "starts": (s,),
                                     "xlist": None}
            for part in ("top", "low"):
                j[f"tvr-xeval-{sl}-{part}"] = {"name": f"vpd-tvr-xeval-{sl}-{part}", "script": "tvr_eval.sh",
                                               "as_root": True, "kind": "xeval", "start": s, "part": part,
                                               "base": "tvr-eval", "starts": (s,), "xlist": None}
    return j


XJOBS = _xjobs()
# the smokes (PREREG section 16: kill window 60 s for data and smoke jobs)
KILL_S.update({j: 60 for j, spec in XJOBS.items() if spec["kind"] == "xsmoke"})


def job_spec(job: str) -> dict:
    if job in XJOBS:
        return XJOBS[job]
    if job not in JOBS:
        raise Refused(f"unknown job {job!r}; one of {sorted(JOBS)} or the explore jobs {sorted(XJOBS)}")
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
    if spec.get("xlist"):  # tvr_xsmoke.sh's start list (tvr-xsmoke: none, the script's default)
        args.append(spec["xlist"])
    if spec["start"]:
        args.append(spec["start"])
    if spec["part"]:
        args.append(spec["part"])
    env = ["env", f"VPD_JOB_DEADLINE_S={t['soft_deadline_s']}"]
    if spec["kind"] in ("xtrain", "xeval"):  # the explore output root of tvr_train.sh / tvr_eval.sh
        env.append(f"VPD_TVR_ROOT={XRL_ROOT}")
    return (env + ["timeout", "-k", str(t["kill_s"]),
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
    """The pilot's and the RL starts' rule: this tree's trainer code_sha256 starts with
    recipe.trainer_code_sha256_prefix (0865dd75f1ee; the Q3SG jobs use q3sg_trainer_problems instead)."""
    want = pilot_cfg.get("recipe.trainer_code_sha256_prefix", cfg)
    got = trainer_prefix()
    return [] if got.startswith(want) else [f"trainer code_sha256 {got[:12]} != frozen prefix {want}"]


# --- the Q3SG trainer rule (PREREG section 17, row 2026-11-13; DECISIONS G20) ------------------------------------

def trainer_fingerprint(root: Path = None) -> dict:
    """n2_train_vpd.code_fingerprint of ROOT (default: the trainer's own tree, as the trainer records it)."""
    from vpd.train import n2_train_vpd as T

    return T.code_fingerprint(T.WORKTREE if root is None else Path(root))


def starts_json_without(path: Path, drop=Q3SG_STARTS) -> bytes:
    """starts.json (PATH) without the DROP entries, re-serialised as json.dumps(table, indent=1) + "\\n" (the
    pilot's starts.json bytes when DROP = the Q3SG entry; the key order of the file is kept)."""
    table = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(table, dict):
        raise ValueError(f"{path}: not a json object")
    return (json.dumps({k: v for k, v in table.items() if k not in drop}, indent=1) + "\n").encode()


def code_files_problems(files, starts_sha256: str = STARTS_JSON_SHA256_Q3SG) -> List[str]:
    """A trainer fingerprint's code_files (an adapter's meta.code_files) under the Q3SG invariant: exactly the 6
    CODE_FILES, the 5 .py shas == TRAINER_PY_SHA256 (the pilot / RL adapters' records), starts.json == starts_sha256."""
    if not isinstance(files, dict):
        return [f"code_files {type(files).__name__}, not the trainer's per-file shas"]
    bad = [f"code_files {rel} {str(files.get(rel))[:12]} != the pilot's {want[:12]}"
           for rel, want in TRAINER_PY_SHA256.items() if files.get(rel) != want]
    if files.get(STARTS_JSON) != starts_sha256:
        bad.append(f"code_files {STARTS_JSON} {str(files.get(STARTS_JSON))[:12]} != {starts_sha256[:12]}")
    other = sorted(set(files) - set(TRAINER_PY_SHA256) - {STARTS_JSON})
    if other:
        bad.append(f"code_files has other files {other}")
    return bad


def q3sg_trainer_problems(fp: dict = None, root: Path = None) -> List[str]:
    """The Q3SG jobs' trainer rule on this tree (ROOT: default the trainer's own tree): code_sha256 starts with
    Q3SG_TRAINER_PREFIX; code_files_problems(code_files) == []; vpd/starts.json without the Q3SG entry, re-serialised,
    has the pilot's sha STARTS_JSON_SHA256_PILOT ("5 个 .py 文件 sha 不变、starts.json 只多这一条")."""
    from vpd.train import n2_train_vpd as T

    root = Path(T.WORKTREE if root is None else root)
    fp = trainer_fingerprint(root) if fp is None else fp
    got = str(fp.get("code_sha256"))
    bad = [] if got.startswith(Q3SG_TRAINER_PREFIX) else [
        f"trainer code_sha256 {got[:12]} != the Q3SG prefix {Q3SG_TRAINER_PREFIX} (PREREG section 17, 2026-11-13)"]
    bad += [f"trainer {b}" for b in code_files_problems(fp.get("code_files"))]
    try:
        old = hashlib.sha256(starts_json_without(root / STARTS_JSON)).hexdigest()
    except (OSError, ValueError) as e:
        bad.append(f"{root / STARTS_JSON}: unreadable ({e})")
    else:
        if old != STARTS_JSON_SHA256_PILOT:
            bad.append(f"{root / STARTS_JSON} without {list(Q3SG_STARTS)} re-serialised: sha256 {old[:12]} != the "
                       f"pilot's {STARTS_JSON_SHA256_PILOT[:12]} (starts.json changed beyond the Q3SG entry)")
    return bad


def x_code_problems(s: str, meta: dict, cfg=None) -> List[str]:
    """An explore adapter's (or the explore smoke's) recorded trainer code, by its start's registered rule: Q3SG
    code_sha256 starting with Q3SG_TRAINER_PREFIX and code_files_problems(meta.code_files) == []; the RL starts
    code_sha256 starting with recipe.trainer_code_sha256_prefix (0865dd75f1ee)."""
    got = str((meta or {}).get("code_sha256") or "")
    if s in Q3SG_STARTS:
        bad = [] if got.startswith(Q3SG_TRAINER_PREFIX) else [
            f"trained with code {got[:12] or None}, not the Q3SG prefix {Q3SG_TRAINER_PREFIX}"]
        return bad + code_files_problems((meta or {}).get("code_files"))
    want = pilot_cfg.get("recipe.trainer_code_sha256_prefix", cfg)
    return [] if want and got.startswith(want) else [f"trained with code {got[:12] or None}, not the registered "
                                                     f"prefix {want}"]


def x_trainer_problems(job: str, cfg=None) -> List[str]:
    """An explore job's trainer rule on this tree: the Q3SG jobs q3sg_trainer_problems(); the RL jobs
    trainer_problems(cfg) (0865dd75f1ee, so they refuse on a tree whose starts.json has the Q3SG entry)."""
    if set(XJOBS[job]["starts"]) & set(Q3SG_STARTS):
        return q3sg_trainer_problems()
    return trainer_problems(cfg)


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
    if job in XJOBS:
        return x_prerequisites(job, L, cfg, in_job)
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
    if job in XEST_HIGH:
        return {"est_high": XEST_HIGH[job], "rule": "constant (explore smoke; pilot_tvr XEST_HIGH)"}
    if spec["kind"] in ("train", "xtrain"):  # the explore starts: the same start-independent section-16 rules
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
    if job in XJOBS:
        return x_prepare(job, pool, new, now, L, cfg, events)
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
    if job in XJOBS:
        return x_submit(job, L, cfg, events)
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


def status(L: Layout = None, explore: bool = False) -> dict:
    from vpd import sco as vsco

    L = L or LAYOUT
    st = vsco.status(include_ended=True)
    runs = {}
    for job in (XJOBS if explore else JOBS):
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


def budget(L: Layout = None, cfg=None, events=None, explore: bool = False) -> dict:
    from vpd import ledger

    L = L or LAYOUT
    events = ledger.read() if events is None else events
    rows = {}
    for job in (XJOBS if explore else JOBS):
        try:
            rows[job] = est_high(job, L, cfg)["est_high"]
        except (Refused, ValueError, KeyError) as e:
            rows[job] = f"not derivable yet: {getattr(e, 'message', e)}"[:200]
    out = {"round_budget": round_budget(0.0, events, cfg), "est_high": rows, "ledger": ledger.summarise(events)}
    if explore:
        out["block_budget"] = block_budget(0.0, events)
        out["est_high_total"] = round(sum(v for v in rows.values() if isinstance(v, float)), 3)
    return out


def plan(L: Layout = None, cfg=None, out=None, explore: bool = False) -> dict:
    L = L or LAYOUT
    out = out or sys.stdout
    p = lambda *a: print(*a, file=out)  # noqa: E731
    if explore:
        p(f"VPD TVR EXPLORATORY RL plan ({utc()}; PREREG section 17, {XROW_DATE}; DECISIONS G18); starts "
          f"{list(EXPLORE_STARTS)} (Q3SG: row {Q3SG_ROW_DATE}, DECISIONS G20), outputs {L.xroot}; n_top "
          f"{pilot_cfg.get('tiers.n_top', cfg)!r}")
    else:
        p(f"VPD TVR pilot plan ({utc()}); n_top {pilot_cfg.get('tiers.n_top', cfg)!r}")
    rows = {}
    for job, spec in (XJOBS if explore else JOBS).items():
        try:
            e = est_high(job, L, cfg)
            pay = " ".join(payload_for(job, L.jobs / f"{job}-<yyyymmddHHMM>", e["est_high"], cfg))
            est = f"est_high {e['est_high']:.2f} GPU-h ({e['rule'][:60]})"
            if explore:
                t = job_timeouts(job, e["est_high"], cfg)
                est += f", outer {t['outer_s']} s, soft {t['soft_deadline_s']} s, kill {t['kill_s']} s"
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
        if explore:
            bb = block_budget(0.0, None)
            p(f"block budget: used {bb['block_used']:.4f} (ledger {bb['ledger_spent']:.4f} - {bb['baseline']}) + open "
              f"{bb['open_est']:.2f} = {bb['total']:.2f} of {bb['cap']} GPU-h")
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
    if job in XJOBS:
        return x_required_steps(job, cfg)
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
    """Every recorded step rc 0 (a failed X is excused by a later X_fallback with rc 0, the cal's O9 retry; in an
    explore job a failed merge_parity_X by a later merge_parity_X:report_only with rc 0, the report-only merge parity
    of DECISIONS G16 (1)) and every required step recorded (NAME or NAME:<suffix>, e.g. NAME:skip)."""
    steps = read_steps(run)
    names = [str(s.get("name")) for s in steps]
    failed = []
    explore = job in XJOBS
    for i, s in enumerate(steps):
        if s.get("rc") != 0:
            n = str(s.get("name"))
            if any(str(t.get("name")) == n + "_fallback" and t.get("rc") == 0 for t in steps[i + 1:]):
                continue
            if explore and n.startswith("merge_parity_") and ":" not in n and any(
                    str(t.get("name")) == n + ":report_only" and t.get("rc") == 0 for t in steps[i + 1:]):
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
    if job in XJOBS:
        return x_expect_items(job, run, L, cfg, files_only)
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
    if job in XJOBS:
        facts.update(x_facts(job))
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


# --- EXPLORATORY RL starts (PREREG section 17, row 2026-11-12; DECISIONS G18) ------------------------------------

def harness_sha256() -> str:
    from vpd.tvr import run_tvr

    return run_tvr.code_sha256()["code_sha256"]


def harness_problems() -> List[str]:
    got = harness_sha256()
    return [] if got.startswith(HARNESS_SHA256_PREFIX) else [
        f"run_tvr harness code_sha256 {got[:12]} != the pilot evals' {HARNESS_SHA256_PREFIX} (a run_tvr CODE_FILE "
        "changed: the explore evals would not use the registered harness)"]


def x_start_problems(s: str) -> List[str]:
    """An explore start: in EXPLORE_STARTS (never a pilot start), a registry code, its checkpoint dir, and the
    processor read from that dir (no eval view: DECISIONS G18 registered the checkpoint's processor files)."""
    from vpd import starts

    if s not in EXPLORE_STARTS or s in STARTS:
        return [f"start {s!r} is not one of the explore starts {list(EXPLORE_STARTS)}"]
    try:
        e = starts.resolve(s)
    except KeyError as err:
        return [f"start {s}: {err}"]
    bad = []
    if not e.get("code"):
        bad.append(f"start {s}: no registry code in vpd/starts.json (the served model name)")
    ck = Path(e["path"])
    if not os.path.lexists(ck / "config.json"):
        bad.append(f"start {s}: {ck}/config.json missing")
    pdir = Path(starts.processor_dir(s))
    if pdir != ck:
        bad.append(f"start {s}: the processor would be read from {pdir}, not the checkpoint dir {ck} (DECISIONS G18 "
                   "registered the checkpoint's processor files; an eval view was built since?): stop and ask")
    return bad


def xrow(text: str, date: str = XROW_DATE, needle: str = XROW_NEEDLE) -> Optional[str]:
    """The single section-17 row of DATE containing NEEDLE (None unless exactly one); default the 2026-11-12 row
    registering the explore RL starts."""
    rows = [ln for ln in text.splitlines() if ln.startswith(f"| {date} |") and needle in ln]
    return rows[0] if len(rows) == 1 else None


def xrow_key(s: str) -> Tuple[str, str]:
    """(date, needle) of the section-17 row registering explore start S: Q3SG 2026-11-13, the RL starts 2026-11-12."""
    return (Q3SG_ROW_DATE, Q3SG_ROW_NEEDLE) if s in Q3SG_STARTS else (XROW_DATE, XROW_NEEDLE)


def xrow_problems(starts=RL_STARTS) -> List[str]:
    """Exactly one registration row per start of STARTS (default the RL starts: the 2026-11-12 row)."""
    try:
        text = prereg_path().read_text(encoding="utf-8")
    except OSError:
        text = ""
    bad = []
    for date, needle in dict.fromkeys(xrow_key(s) for s in starts):
        if xrow(text, date, needle) is None:
            what = "the third explore start Q3SG" if date == Q3SG_ROW_DATE else "the explore RL starts"
            bad.append(f"{PREREG} has no single section-17 row '| {date} | ... {needle} ...' (the registration of "
                       f"{what})")
    return bad


def tier_manifest_files(L: Layout) -> dict:
    return (read_json(L.tiers / "manifest.json") or {}).get("files") or {}


def tier_file_problems(L: Layout, names) -> List[str]:
    """Every tier file the job reads == its sha256 in tiers/v1/manifest.json (whose own sha256 is
    data.tiers_manifest_sha256, checked by data_problems)."""
    files = tier_manifest_files(L)
    bad = []
    for name in names:
        p = L.tiers / name
        want, got = files.get(name), (sha256_file(p) if p.is_file() else None)
        if want is None or got != want:
            bad.append(f"{p}: sha256 {str(got)[:12]} != tiers manifest {str(want)[:12]}")
    return bad


def x_adapter_check(s: str, tier: str, init: Optional[str], L: Layout, cfg=None,
                    adapter_root: Optional[Path] = None) -> Tuple[int, str]:
    """An explore adapter: adapter_check's rules (complete; from INIT/adapter.pt as it is now) with the trainer code of
    its start's registered rule (x_code_problems: RL the recipe prefix 0865dd75f1ee; Q3SG the Q3SG prefix and the
    code_files invariant; DECISIONS G20) instead of this tree's current code, plus its meta: start, seed 1001, phase,
    data_sha256 == the tiers manifest's (L.tiers); 0W from the fresh LoRA, 100 / TOP from this start's 0W.  Adapters
    under adapter_root/<S>/adapters (default L.xroot).  (0, ok) or (rc, why): 1 missing / unfinished / other init,
    3 other trainer code (adapter_check's codes)."""
    ar = Path(adapter_root) if adapter_root is not None else L.xroot
    d = ar / s / "adapters" / tier
    m = read_json(d / "meta.json")
    if m is None or not (d / "adapter.pt").is_file():
        return 1, f"{d}: no adapter.pt + meta.json"
    steps, planned = m.get("optimizer_steps"), m.get("optimizer_steps_planned")
    if not steps or (planned is not None and steps != planned):
        return 1, f"{d}: unfinished ({steps} of {planned} optimizer steps)"
    code = x_code_problems(s, m, cfg)
    if code:
        return 3, f"{d}: " + "; ".join(code)
    if init is not None:
        ip = ar / s / "adapters" / init / "adapter.pt"
        want = sha256_file(ip) if ip.is_file() else None
        got = (m.get("init_adapter") or {}).get("sha256")
        if want is None or got != want:
            return 1, f"{d}: started from init adapter {got}, {ip} is now {want}: retrain"
    msg = f"{d}: complete ({steps} steps, code {str(m.get('code_sha256'))[:12]})"
    want = {"start": s, "seed": int(pilot_cfg.SEED), "phase": PHASE[tier],
            "data_sha256": tier_manifest_files(L).get(TIER_FILE[tier])}
    bad = [f"meta {k} {m.get(k)!r} != {v!r}" for k, v in want.items() if m.get(k) != v or v is None]
    ia = m.get("init_adapter")
    if init is None and ia is not None:
        bad.append(f"init_adapter {(ia or {}).get('dir')!r}, but {tier} starts from the fresh LoRA")
    if init is not None and (ia or {}).get("start") != s:
        bad.append(f"init_adapter start {(ia or {}).get('start')!r} != {s!r}")
    return (1, f"{d}: " + "; ".join(bad)) if bad else (0, msg)


def x_adapters_problems(s: str, L: Layout, cfg=None) -> List[str]:
    bad = []
    for tier, init in train_tiers(s, cfg):
        rc, msg = x_adapter_check(s, tier, init, L, cfg)
        if rc != 0:
            bad.append(f"adapter {s}/{tier}: {msg}")
    return bad


def xsmoke_verdicts(L: Layout = None) -> List[Path]:
    L = L or LAYOUT
    if not L.xsmoke_root.is_dir():
        return []
    return sorted((p / "smoke_verdict.json" for p in L.xsmoke_root.iterdir()
                   if (m := STAMP_RE.match(p.name)) and m.group("job") == "tvr-xsmoke"
                   and (p / "smoke_verdict.json").is_file()), key=lambda p: p.parent.name)


def xsmoke_verdicts_with(s: str, L: Layout = None) -> List[Path]:
    """Every smoke_verdict.json under runs/explore/rl/smoke/<RUN>/ (any job's RUN name) whose "starts" include S,
    oldest first (by the RUN name's stamp, then name)."""
    L = L or LAYOUT
    if not L.xsmoke_root.is_dir():
        return []
    out = []
    for p in L.xsmoke_root.iterdir():
        m = STAMP_RE.match(p.name)
        v = read_json(p / "smoke_verdict.json") if m else None
        if isinstance(v, dict) and isinstance(v.get("starts"), list) and s in v["starts"]:
            out.append((m.group("stamp"), p.name, p / "smoke_verdict.json"))
    return [x[2] for x in sorted(out)]


def latest_xverdict(L: Layout = None, s: Optional[str] = None) -> Tuple[Optional[Path], Optional[dict]]:
    """The explore smoke verdict an xtrain / xeval job of start S needs: Q3SG the newest one whose "starts" include
    Q3SG (xsmoke_verdicts_with); otherwise (the RL starts) the newest tvr-xsmoke verdict."""
    vs = xsmoke_verdicts_with(s, L) if s in Q3SG_STARTS else xsmoke_verdicts(L)
    return (vs[-1], read_json(vs[-1])) if vs else (None, None)


def x_prerequisites(job: str, L: Layout = None, cfg=None, in_job: bool = False) -> List[str]:
    """Problems that block an explore job (see the module docstring); [] = ready."""
    L = L or LAYOUT
    spec = XJOBS[job]
    kind, s = spec["kind"], spec["start"]
    bad = [f"pilot_cfg value missing for {job}: {k}" for k in pilot_cfg.missing(spec["base"], cfg)]
    bad += payload_constant_problems(cfg)
    bad += x_trainer_problems(job, cfg)
    bad += harness_problems()
    for st in spec["starts"]:
        bad += x_start_problems(st)
    bad += prereg_problems(in_job, cfg)
    bad += xrow_problems(spec["starts"])
    bad += data_problems(L, cfg)
    bad += n_top_problems(L, cfg)
    if kind in ("xsmoke", "xtrain"):
        bad += futility_problems(L)
    if any(b.startswith("pilot_cfg value missing") for b in bad):
        return bad
    if kind == "xsmoke":
        bad += tier_file_problems(L, ("parity3.jsonl", TIER_FILE["0W"]))
    else:
        bad += tier_file_problems(L, [TIER_FILE[t] for t, _ in train_tiers(s, cfg)])
    if kind in ("xtrain", "xeval"):
        vp, v = latest_xverdict(L, s)
        if v is None:
            bad.append(f"no smoke_verdict.json under {L.xsmoke_root} " + (
                f"whose starts include {s} (job tvr-xsmoke-{s.lower()})" if s in Q3SG_STARTS else "(job tvr-xsmoke)"))
        elif v.get("verdict") != "PASS":
            bad.append(f"{vp}: verdict {v.get('verdict')!r}, not PASS ({', '.join(v.get('failed') or [])})")
    if kind == "xeval":
        bad += x_adapters_problems(s, L, cfg)
    return bad


def block_budget(new_est: float, events=None) -> dict:
    """This block's spend (DECISIONS G18): (ledger total spent - BLOCK_BASELINE_GPUH) + est of the open ledger jobs +
    new <= BLOCK_CAP_GPUH."""
    from vpd import ledger

    events = ledger.read() if events is None else events
    spent = float(ledger.summarise(events)["total"]["spent"])
    open_est = sum(float(j.get("est") or 0.0) for j in ledger.fold(events).values() if ledger.is_open(j))
    used = spent - BLOCK_BASELINE_GPUH
    total = used + open_est + float(new_est)
    return {"ledger_spent": round(spent, 4), "baseline": BLOCK_BASELINE_GPUH, "block_used": round(used, 4),
            "open_est": round(open_est, 4), "new_est_high": round(float(new_est), 4), "total": round(total, 4),
            "cap": BLOCK_CAP_GPUH, "ok": total <= BLOCK_CAP_GPUH + 1e-9}


def make_x_shared_parents(L: Layout) -> List[str]:
    out = []
    for d in L.x_shared_parents():
        d.mkdir(parents=True, exist_ok=True)
        out.append(str(d))
    return out


def x_facts(job: str = "tvr-xsmoke") -> dict:
    """The facts an explore job's prepared.json / guard.json record: the harness and trainer fingerprints, the sha256
    of the job's registration row; the Q3SG jobs also the trainer rule and its problems (q3sg_trainer_problems)."""
    starts = XJOBS[job]["starts"]
    date, needle = xrow_key(starts[0])
    try:
        row = xrow(prereg_path().read_text(encoding="utf-8"), date, needle)
    except OSError:
        row = None
    fp = trainer_fingerprint()
    out = {"explore": True, "harness_code_sha256": harness_sha256(), "trainer_code_sha256": fp["code_sha256"],
           "prereg_row_sha256": None if row is None else hashlib.sha256(row.encode()).hexdigest()}
    if set(starts) & set(Q3SG_STARTS):
        out.update(prereg_row_date=date, trainer_rule=f"Q3SG: prefix {Q3SG_TRAINER_PREFIX}, 5 .py shas == the "
                   "pilot's, starts.json == the pilot's + the Q3SG entry",
                   trainer_code_files=fp.get("code_files"), trainer_problems=q3sg_trainer_problems(fp))
    return out


def _budget_checks(job: str, hi: float, events, cfg) -> Tuple[dict, dict]:
    from vpd import ledger

    events = ledger.read() if events is None else events
    rb = round_budget(hi, events, cfg)
    print(f"[pilot_tvr] round budget: used {rb['round_used']:.2f} + open {rb['open_est']:.2f} + new "
          f"{rb['new_est_high']:.2f} = {rb['total']:.2f} GPU-h vs cap {rb['cap']:.0f}", file=sys.stderr)
    bb = block_budget(hi, events)
    print(f"[pilot_tvr] block budget: used {bb['block_used']:.4f} + open {bb['open_est']:.4f} + new "
          f"{bb['new_est_high']:.4f} = {bb['total']:.4f} GPU-h vs cap {bb['cap']}", file=sys.stderr)
    if not rb["ok"]:
        raise Refused(f"ROUND BUDGET: round_used {rb['round_used']:.2f} + open {rb['open_est']:.2f} + new "
                      f"{rb['new_est_high']:.2f} = {rb['total']:.2f} > {rb['cap']} GPU-h (brief 2C: the round stops)")
    if not bb["ok"]:
        raise Refused(f"BLOCK BUDGET: block_used {bb['block_used']:.4f} + open {bb['open_est']:.4f} + new "
                      f"{bb['new_est_high']:.4f} = {bb['total']:.4f} > {bb['cap']} GPU-h (DECISIONS G18: the "
                      f"user's cap of this block; {job} not prepared: stop and ask)")
    return rb, bb


def x_prepare(job: str, pool: str = "zoetrope", new: bool = False, now=None, L: Layout = None, cfg=None,
              events=None) -> Path:
    from vpd import sco as vsco

    L = L or LAYOUT
    spec = XJOBS[job]
    parents = make_x_shared_parents(L)  # as the login user, before anything else (never runs/pilot)
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
    bad = x_prerequisites(job, L, cfg)
    if bad:
        raise Refused(f"{job} not ready:\n  - " + "\n  - ".join(bad))
    est = est_high(job, L, cfg)
    hi = est["est_high"]
    t = job_timeouts(job, hi, cfg)
    rb, bb = _budget_checks(job, hi, events, cfg)
    run = L.jobs / f"{job}-{stamp(now)}"
    if run.exists():
        raise Refused(f"{run} exists (prepared twice in one minute?)")
    payload = payload_for(job, run, hi, cfg)
    info = {"job": job, "utc": utc(), "exploratory": True, "est": est, "est_high": hi, "timeouts": t,
            "payload": payload, "round_budget": rb, "block_budget": bb, "shared_parents": parents,
            "as_root": spec["as_root"], "pool": pool, "prereg": prereg_state(), "facts": x_facts(job)}
    if spec["kind"] in ("xtrain", "xeval"):
        vp, v = latest_xverdict(L, spec["start"])
        info["smoke_verdict"] = {"path": str(vp), "sha256": sha256_file(vp), "verdict": v.get("verdict")}
    if L.cal_decision.is_file():
        info["cal_decision"] = {"path": str(L.cal_decision), "sha256": sha256_file(L.cal_decision)}
    launch = vsco.prepare(run, spec["name"], payload, pool=pool, as_root=spec["as_root"], est_gph=hi, gate=GATE)
    write_json(run / "prepared.json", info)
    return Path(launch)


def x_submit(job: str, L: Layout = None, cfg=None, events=None) -> str:
    from vpd import sco as vsco

    L = L or LAYOUT
    run = latest_run(job, ("prepared",), L)
    if run is None:
        raise Refused(f"no prepared, unsubmitted run of {job}: run `pilot_tvr prepare {job}` first")
    newer = latest_run(job, L=L)
    if newer != run:
        raise Refused(f"the newest run of {job} is {newer} ({run_state(newer)}), not the prepared {run}")
    bad = x_prerequisites(job, L, cfg)
    if bad:
        raise Refused(f"{job} not ready:\n  - " + "\n  - ".join(bad))
    info = read_json(run / "prepared.json") or {}
    ps = prereg_state()
    if (info.get("prereg") or {}).get("commit") != ps["commit"]:
        raise Refused(f"{PREREG} commit changed since prepare ({(info.get('prereg') or {}).get('commit')} -> "
                      f"{ps['commit']}): prepare again")
    _budget_checks(job, float(info.get("est_high") or 0.0), events, cfg)
    return vsco.submit(run)


def x_required_steps(job: str, cfg=None) -> List[str]:
    spec = XJOBS[job]
    kind, s = spec["kind"], spec["start"]
    if kind == "xsmoke":
        out = ["guard", "prereg_committed", "merge_disk"]
        for st in spec["starts"]:
            out += [f"{x}_{st}" for x in ("load", "dry", "hfparity", "session", "train", "merge", "merge_parity",
                                          "eval")]
        return out + ["verdict"]
    if kind == "xtrain":
        return ["guard", "prereg_committed"] + [f"train_{t}" for t, _ in train_tiers(s, cfg)]
    out = ["guard", "prereg_committed", "merge_disk", "canary_replay", "canary_parity"]
    tiers = []
    for tier, _arm in eval_plan(s, spec["part"], cfg):
        if tier not in tiers:
            tiers.append(tier)
    for tier in tiers:
        out += [f"merge_{tier}", f"merge_parity_{tier}", f"eval_{tier}"]
    return out + [f"official_{tier}_{arm}" for tier, arm in eval_plan(s, spec["part"], cfg)]


def x_parity_reported(path: Path, adapter_dir: Path) -> Tuple[bool, dict]:
    """The explore merge parity (report only, DECISIONS G16 (1)): ok iff its json exists for the adapter as it is now
    (parity_state 0 or 4); its pass and numbers are recorded, not judged."""
    rc, msg = parity_state(path, adapter_dir)
    rep = read_json(path) or {}
    return rc in (0, 4), {"parity_ok_rc": rc, "msg": msg, "pass_reported_not_judged": rep.get("pass"),
                          "mean_abs_dce": rep.get("mean_abs_dce"), "reply_agreement": rep.get("reply_agreement"),
                          "n_replies": rep.get("n_replies"), "n_demos": rep.get("n_demos")}


def x_expect_items(job: str, run: Path, L: Layout, cfg=None, files_only: bool = False) -> List[dict]:
    spec = XJOBS[job]
    kind, s, part = spec["kind"], spec["start"], spec["part"]
    out: List[dict] = []
    if kind == "xsmoke":
        v = read_json(L.xsmoke_dir(run) / "smoke_verdict.json")
        out.append(_chk("smoke_verdict", bool(v) and v.get("verdict") == "PASS"
                        and v.get("starts") == list(spec["starts"]),
                        None if not v else {"verdict": v.get("verdict"), "failed": v.get("failed"),
                                            "starts": v.get("starts")},
                        L.xsmoke_dir(run) / "smoke_verdict.json"))
    elif kind == "xtrain":
        for tier, init in train_tiers(s, cfg):
            if files_only:
                out.append(_chk(f"adapter_{tier}", True, None, L.xadapter(s, tier) / "adapter.pt"))
                continue
            rc, msg = x_adapter_check(s, tier, init, L, cfg)
            out.append(_chk(f"adapter_{tier}", rc == 0, msg, L.xadapter(s, tier) / "adapter.pt"))
    else:  # xeval
        ok, det = _rp_pass(L.xcanary(s, part))
        out.append(_chk(f"canary_{part}", ok, det, L.xcanary(s, part)))
        seen = []
        for tier, arm in eval_plan(s, part, cfg):
            ad = L.xadapter(s, EVAL_ADAPTER[tier])
            if tier not in seen:
                seen.append(tier)
                ok, det = x_parity_reported(L.xparity(s, tier), ad)
                out.append(_chk(f"merge_parity_{tier}", ok, det, L.xparity(s, tier)))
            d = L.xeval_dir(s, tier, arm)
            ok, det = _valid_results(d, EVAL500_N)
            if ok and not files_only:
                rc, msg = eval_check(d, ad)
                ok, det = rc == 0, dict(det, adapter=msg)
            out.append(_chk(f"eval_{tier}_{arm}", ok, det, d / "results.json"))
            oc = read_json(d / "official_check.json")
            out.append(_chk(f"official_{tier}_{arm}", bool(oc) and oc.get("equal") is True, oc,
                            d / "official_check.json"))
    if files_only:
        return [c for c in out if c.get("path")]
    return out


# --- explore smoke verdict (in-job; mechanism only, never a success field) --------------------------------------

def _session_checks(ev: Path, name: str, arms, n: int, cfg=None) -> Tuple[List[Tuple[str, bool, object]], dict]:
    """The pilot smoke's mechanism checks of one run_tvr session (smoke_verdict's runs / regex / u_arm / audit /
    renderer) on EV/<name>__<arm>; [(check, ok, detail)], measured perf.  Reads only valid, n_tasks, tasks_missing,
    invalid, audit, renderer, perf."""
    grid = list(pilot_cfg.get("protocol.image_grid", cfg))
    tok = int(pilot_cfg.get("protocol.image_tokens", cfg))
    prof = pilot_cfg.get("render.lvp_profile", cfg)
    font = pilot_cfg.get("render.badge_font_sha256", cfg)
    lp = pilot_cfg.get("render.lp_threads", cfg)
    res = {a: _results(ev / f"{name}__{a}") for a in arms}
    out = []
    bad = []
    for a, r in res.items():
        if r is None:
            bad.append(f"{a}: no results.json")
        elif r.get("valid") is not True or r.get("tasks_missing") or r.get("n_tasks") != n:
            bad.append(f"{a}: valid {r.get('valid')}, n_tasks {r.get('n_tasks')}, missing "
                       f"{(r.get('tasks_missing') or [])[:3]}")
    out.append(("runs", not bad, bad))
    bad = []
    for a in [x for x in arms if x in ("c", "blind")]:
        inv = (res[a] or {}).get("invalid") or {}
        if not (inv.get("regex_mismatch") == 0 and (inv.get("replies") or 0) > 0):
            bad.append({a: {"regex_mismatch": inv.get("regex_mismatch"), "replies": inv.get("replies")}})
    out.append(("regex", not bad, bad))
    if "u" in arms:
        inv_u = (res["u"] or {}).get("invalid") or {}
        rate = None
        if inv_u.get("replies"):
            rate = round(1 - (inv_u.get("unparseable_steps") or 0) / inv_u["replies"], 4)
        out.append(("u_arm", res["u"] is not None, {"parse_rate": rate, "judged": False}))
    bad = []
    for a in [x for x in arms if x in IMAGE_ARMS]:
        au = (res[a] or {}).get("audit") or {}
        pp = (au.get("prompt_parity") or {}).get("equal")
        it = au.get("image_tokens") or {}
        tpi = it.get("tokens_per_image")
        tpi = list(tpi) if isinstance(tpi, (list, tuple)) else [tpi]
        gs = [list(g) for g in it.get("grid_set") or []]
        if pp is not True or tpi != [tok] or gs != [grid]:
            bad.append({a: {"prompt_parity": pp, "tokens_per_image": tpi, "grid_set": gs}})
    out.append(("audit", not bad, bad))
    bad = []
    for a in [x for x in arms if x in IMAGE_ARMS]:
        rr = (res[a] or {}).get("renderer") or {}
        got = {"profile": rr.get("profile"), "font_sha256": rr.get("font_sha256"), "lp": rr.get("lp")}
        wp = worker_problems(rr, cfg)
        if got != {"profile": prof, "font_sha256": font, "lp": lp} or prof is None or lp is None or wp:
            bad.append({a: dict(got, workers_ok=rr.get("workers_ok"), workers=wp[:5])})
    out.append(("renderer", not bad, bad))
    bad = [{a: str((r or {}).get("code_sha256"))[:12]} for a, r in res.items()
           if not str((r or {}).get("code_sha256") or "").startswith(HARNESS_SHA256_PREFIX)]
    out.append(("harness", not bad, bad or HARNESS_SHA256_PREFIX))
    measured = {a: {k: ((res[a] or {}).get("perf") or {}).get(k) for k in (
        "steps_per_s", "vllm_ready_s", "max_images_per_request", "max_prompt_tokens", "cgroup_cores_max")}
        for a in arms}
    return out, {"res": res, "measured": measured}


def _finite(x) -> bool:
    return _num(x) is not None


def xsmoke_verdict(smoke: Path, L: Layout = None, cfg=None, job: str = "tvr-xsmoke") -> dict:
    """SMOKE/smoke_verdict.json of an explore smoke JOB (tvr-xsmoke: the RL starts; tvr-xsmoke-q3sg: Q3SG; EXPLORATORY;
    DECISIONS G18, G20): PASS iff every mechanism check passes.
    Per explore start S of the job (SMOKE/S/): load (the bf16 conversion of the trainer's loader, load/load.json), dry
    run (grid [1,22,40], 222 tokens per image), HF parity, vLLM prompt parity and the pilot smoke's session checks on
    the raw start (smoke8: c, u, blind), the smoke training (meta: XSMOKE_STEPS optimizer steps of the first
    XSMOKE_LIMIT 0W demos, seed 1001, the start's registered trainer code (x_code_problems: RL the recipe prefix;
    Q3SG the Q3SG prefix and the code_files invariant), the manifest's 0W sha), the merge (merge_parity's record of
    vpd_merge.json: every wrapped module merged, from this adapter), merge parity (report only: it ran on this
    adapter; its pass is recorded, never judged), and the merged smoke4 c session (vLLM prompt parity, session
    checks, the served adapter).  Never reads a success / SR field (run_tvr --mechanism-only also nulls them)."""
    L = L or LAYOUT
    smoke = Path(smoke)
    if job not in XJOBS or XJOBS[job]["kind"] != "xsmoke":
        raise Refused(f"{job!r} is not an explore smoke job")
    starts = XJOBS[job]["starts"]
    checks, measured = [], {}
    raw_n = len(pilot_cfg.CATEGORIES) * int(pilot_cfg.get("dev.smoke_per_category", cfg))
    merged_n = SHAKE_N
    w0 = tier_manifest_files(L).get(TIER_FILE["0W"])
    for s in starts:
        sd = smoke / s
        ld = read_json(sd / "load" / "load.json")
        checks.append(_chk(f"load_{s}", bool(ld) and ld.get("ok") is True, None if not ld else {
            k: ld.get(k) for k in ("ok", "problems", "dtype", "attn_sites", "processor_dir", "checkpoint_dtypes",
                                   "tensor_checks_ok", "seconds")}))
        dry = read_json(sd / "dry" / "dry_run.json")
        ok, det = (False, "missing") if dry is None else dry_grid_ok(dry, cfg)
        checks.append(_chk(f"dry_{s}", ok, det))
        par = read_json(sd / "dry" / "parity.json")
        checks.append(_chk(f"hfparity_{s}", bool(par) and par.get("verdict") == "PASS",
                           None if not par else {"verdict": par.get("verdict"), "problems": par.get("problems")}))
        ev = sd / "evals"
        vp = read_json(ev / f"smoke__{s}__raw__vllm_parity.json")
        checks.append(_chk(f"vllm_parity_{s}", bool(vp) and vp.get("equal") is True,
                           None if not vp else {"equal": vp.get("equal"), "n": vp.get("n")}))
        raw, rinfo = _session_checks(ev, f"smoke__{s}__raw", ARMS_TOP, raw_n, cfg)
        for name, ok, det in raw:
            checks.append(_chk(f"{name}_{s}", ok, det))
        ad = sd / "adapters" / "0W"
        m = read_json(ad / "meta.json")
        sha = sha256_file(ad / "adapter.pt") if (ad / "adapter.pt").is_file() else None
        want = {"start": s, "seed": int(pilot_cfg.SEED), "phase": "0W", "optimizer_steps": XSMOKE_STEPS,
                "optimizer_steps_planned": XSMOKE_STEPS, "conversations": XSMOKE_LIMIT, "epochs": XSMOKE_EPOCHS,
                "data_sha256": w0, "attn": "sdpa", "init_adapter": None}
        got = {k: (m or {}).get(k) for k in want}
        code = x_code_problems(s, m or {}, cfg)
        tok = bool(m) and sha is not None and got == want and w0 is not None and not code and _finite(
            (m or {}).get("loss_first")) and _finite((m or {}).get("loss_last"))
        checks.append(_chk(f"train_{s}", tok, dict(got, adapter_sha256=sha, code_sha256=str(
            (m or {}).get("code_sha256"))[:12], code_problems=code, loss_finite=bool(m) and _finite(
                m.get("loss_first")) and _finite(m.get("loss_last")))))
        pj = sd / "parity" / "merge_0W.json"
        pr = read_json(pj) or {}
        mg = pr.get("merged") if isinstance(pr.get("merged"), dict) else {}
        wrapped = ((m or {}).get("lora") or {}).get("wrapped")
        mok = bool(mg) and sha is not None and mg.get("adapter_sha256") == sha and mg.get("start") == s and \
            isinstance(wrapped, int) and mg.get("merged") == wrapped
        checks.append(_chk(f"merge_{s}", mok, {"merged": mg.get("merged"), "wrapped": wrapped,
                                                "adapter_sha256_equal": mg.get("adapter_sha256") == sha if mg else None,
                                                "processor_src": mg.get("processor_src")}))
        pok, pdet = x_parity_reported(pj, ad)
        pok = pok and pr.get("n_demos") == int(pilot_cfg.get("gates.merge_parity.n_demos", cfg)) and \
            _finite(pr.get("mean_abs_dce"))
        checks.append(_chk(f"merge_parity_{s}", pok, pdet))
        vm = read_json(ev / f"xsmoke__{s}__0W__vllm_parity.json")
        checks.append(_chk(f"merged_vllm_parity_{s}", bool(vm) and vm.get("equal") is True,
                           None if not vm else {"equal": vm.get("equal"), "n": vm.get("n")}))
        mc, minfo = _session_checks(ev, f"xsmoke__{s}__0W", ("c",), merged_n, cfg)
        for name, ok, det in mc:
            checks.append(_chk(f"merged_{name}_{s}", ok, det))
        served = ((minfo["res"].get("c") or {}).get("adapter") or {}).get("adapter_sha256")
        checks.append(_chk(f"merged_served_{s}", sha is not None and served == sha,
                           {"served_adapter_sha256": served, "adapter_sha256": sha}))
        measured[s] = {"raw": rinfo["measured"], "merged": minfo["measured"]}
    failed = [c["name"] for c in checks if not c["ok"]]
    v = {"schema": "vpd.tvr.xsmoke_verdict/1", "exploratory": True, "utc": utc(), "smoke": str(smoke), "job": job,
         "starts": list(starts), "verdict": "PASS" if not failed else "FAIL", "failed": failed,
         "checks": checks, "measured": measured,
         "note": "mechanism only: no success rate is read or reported; merge parity is reported, not judged"}
    write_json(smoke / "smoke_verdict.json", v)
    return v


# --- CLI ----------------------------------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    all_jobs = sorted(JOBS) + sorted(XJOBS)  # the explore jobs (DECISIONS G18) after the pilot's
    sub.add_parser("plan").add_argument("--explore", action="store_true", help="the EXPLORATORY jobs (XJOBS)")
    p = sub.add_parser("prepare")
    p.add_argument("job", choices=all_jobs)
    p.add_argument("--pool", default="zoetrope", choices=("zoetrope", "h800", "auto"))
    p.add_argument("--new", action="store_true")
    for name in ("submit", "collect"):
        sub.add_parser(name).add_argument("job", choices=all_jobs)
    sub.add_parser("status").add_argument("--explore", action="store_true")
    p = sub.add_parser("delete")
    p.add_argument("job", choices=all_jobs)
    p.add_argument("--allow-incomplete", action="store_true")
    sub.add_parser("budget").add_argument("--explore", action="store_true")
    sub.add_parser("data-fill")
    for name in ("expect", "guard"):
        p = sub.add_parser(name)
        p.add_argument("job", choices=all_jobs)
        p.add_argument("--run", type=Path, required=True)
    p = sub.add_parser("smoke-verdict")
    p.add_argument("--smoke", type=Path, required=True)
    p = sub.add_parser("xsmoke-verdict")
    p.add_argument("--smoke", type=Path, required=True)
    p.add_argument("--job", default="tvr-xsmoke", choices=[j for j, s in XJOBS.items() if s["kind"] == "xsmoke"],
                   help="the explore smoke job (default tvr-xsmoke: the RL starts)")
    p = sub.add_parser("cal-decide")
    p.add_argument("--cal-root", type=Path, required=True)
    p.add_argument("--floors", type=Path, default=None, help="default OUT/tvr/floors/floors_dev200.json")
    p = sub.add_parser("cfg")
    p.add_argument("keys", nargs="+")
    p = sub.add_parser("served")
    p.add_argument("start", choices=STARTS + EXPLORE_STARTS)
    p = sub.add_parser("parity-ok")
    p.add_argument("file", type=Path)
    p.add_argument("--adapter", type=Path, required=True)
    args = ap.parse_args(argv)
    try:
        if args.cmd == "plan":
            plan(explore=args.explore)
        elif args.cmd == "prepare":
            print(prepare(args.job, args.pool, args.new))
        elif args.cmd == "submit":
            print(submit(args.job))
        elif args.cmd == "status":
            print(json.dumps(status(explore=args.explore), indent=1, default=str))
        elif args.cmd == "collect":
            rep = collect(args.job)
            print(json.dumps({k: rep[k] for k in rep if k != "expected"}, indent=1, default=str))
            return 0 if rep["ok"] else 1
        elif args.cmd == "delete":
            print(delete(args.job, args.allow_incomplete))
        elif args.cmd == "budget":
            print(json.dumps(budget(explore=args.explore), indent=1, default=str))
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
        elif args.cmd == "xsmoke-verdict":
            v = xsmoke_verdict(args.smoke, job=args.job)
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
