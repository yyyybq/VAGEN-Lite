"""C line (q38 s4): the guards of a formal training run (stage 4), run on the login node.

  python -m vagen.q38s4.tools.guard check --run RUN [--job JOB]
  python -m vagen.q38s4.tools.guard watch --run RUN --job JOB --out DIR [--job-name NAME] [--interval 120] [--stop]

``check`` reads the run once and prints its status (JSON).  ``watch`` does so every --interval seconds and writes
DIR/status.json after each pass and DIR/guard.jsonl (one line per pass).  JOB must be one of the run's own
submissions (RUN/submits.jsonl, written by launch submit); its recorded job name is the name a stop checks, and
--job-name, if given, must equal it.  When a rule fires it writes DIR/ALERT.json, stops the job (only with --stop,
and only after describing the job again before each try: it must still be RUNNING under that name; up to
STOP_TRIES tries, then describes until the job has left RUNNING), writes ALERT.json again with the stop record and
exits 1.  When the job ends it writes DIR/END.json (with the final checks) and exits 0, or 1 if they fail.  A pass
that fails (a stale file handle, sco not answering) is retried with a growing pause and the guard keeps watching;
from GUARD_FAILS failed passes in a row it keeps DIR/GUARD_ERROR.json up to date, renamed to
GUARD_ERROR_<utc>.json once a pass works again.  The run folder is only read.

A resubmitted run (a new job resuming from ckpt/ in the same folder) is a new attempt: the node-0 header of
run_train.sh ("[q38s4 train] <utc> host ...") in node0.log clears the logged steps, the head history and the NaN
finding, rank 0's train_begin event clears the step times, and host-memory files and step events older than the
job's start_time are not counted, so an earlier attempt's steps neither stall nor alert the new one.

Rules (the brief's stage 4: "损失 NaN；nfp_cos 连续 200 步不低于复制基线（头没学到）；吞吐掉到基准一半以下（机器故障）；
主机内存 > 1,600 GB；存盘失败"):

  nan         a logged step (RUN/logs/node0.log, the trainer's log dict) with a non-finite loss, grad_norm,
              text_loss, nfp_mse or nfp_cos.
  head        C-NFP only (q38s4.json weights > 0): the last HEAD_WINDOW (200) logged steps with NFP pairs all have
              nfp_cos >= copy_cos (the head never beat "predict = the current unit's features" in 200 steps).
  throughput  from step THROUGHPUT_FROM (21): the median step_s of the last 20 steps (rank 0's step events; equal to
              the slowest rank's within 0.1% in the smoke runs) above 2 x BASE_STEP_S (25.76 s, S3's 8-node median),
              i.e. samples per second below half of S3's (at least 20 step times of this attempt); or, while the job
              is RUNNING, no new step for STALL_MIN (30) minutes, SAVE_STALL_MIN (60) after a save step,
              FINAL_STALL_MIN (120) after the last step (the final full save, the export, the last bf16 copy and the
              chmod all come after it), FIRST_STEP_MIN (75) after the job's start before its first step,
              RESUME_FIRST_STEP_MIN (120) when ckpt/ already holds a checkpoint (a resume loads the optimizer state
              too); a hang is throughput 0 and NCCL's timeout is 3 hours.
  host_mem    a node's used memory (MemTotal - MemAvailable: used_gb in the host-memory records, current and peak,
              and in rank 0's step events) above HOST_MEM_GB (1,600).  The page cache (cgroup current_gb) is not
              counted: it grows with every save and the kernel gives it back.
  save        a bf16 copy (sft_metrics/bf16_copies.json and its log) with exit_status != 0 or ok false; a step after
              save step N logged while ckpt/checkpoint-N has no trainer_state.json and no later checkpoint exists;
              the job leaving RUNNING other than SUCCEEDED; at SUCCEEDED, fewer steps than the progress bar's total,
              no trainer_state.json in ckpt/, or no successful bf16 copy of the last checkpoint.
              A bf16 copy running longer than BF16_WARN_MIN (45) minutes is a warning only (31 min in S1 v2).

Warnings only (status.json "warnings"): a long bf16 copy; the job SUSPENDED (a waiting state here, not an end);
sco describe failing DESCRIBE_WARN passes in a row (the stall and end rules need the job's state); rank 0's step
events more than LOG_LAG_WARN steps ahead of node0.log's log dicts (dicts lost, so the NaN and head rules would be
blind); log dicts that could not be parsed are counted in "log_dicts_dropped".

The step of a log dict is the progress bar's "N/TOTAL" printed before it on the same line.  Files are read
incrementally; a line not yet ended by a newline waits for the next pass; an unreadable JSON line (all nodes append
to one host-memory file) is skipped and counted.  sco runs in its own process group, killed whole on a timeout
(sco's sco-acp child outlives a killed sco).
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
import signal
import statistics
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

SCO = "/mnt/umm/users/yinbaiqiao/.sco/bin/sco"
WORKSPACE = "aigc"

HEAD_WINDOW = 200
BASE_STEP_S = 25.76
THROUGHPUT_FROM = 21
THROUGHPUT_WINDOW = 20
STALL_MIN, SAVE_STALL_MIN, FINAL_STALL_MIN = 30.0, 60.0, 120.0
FIRST_STEP_MIN, RESUME_FIRST_STEP_MIN = 75.0, 120.0
HOST_MEM_GB = 1600.0
BF16_WARN_MIN = 45.0
GUARD_FAILS = 5
MAX_PAUSE_S = 900.0
DESCRIBE_WARN = 5
LOG_LAG_WARN = 20
STOP_TRIES, STOP_VERIFY, STOP_WAIT_S = 3, 6, 30.0
FINITE_KEYS = ("loss", "grad_norm", "text_loss", "nfp_mse", "nfp_cos")
END_STATES = ("SUCCEEDED", "FAILED", "STOPPED", "DELETED", "CANCELLED", "KILLED")
WAIT_WARN_STATES = ("SUSPENDED",)

STEP_RE = re.compile(r"(\d+)/(\d+) \[")
HEADER = "[q38s4 train] "  # run_train.sh's first line on each node; node 0's starts an attempt
JOB_ID_RE = re.compile(r"job id\s*:\s*(\S+)")


def utc(t: Optional[float] = None) -> str:
    return datetime.fromtimestamp(time.time() if t is None else t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_utc(s: str) -> float:
    """ISO time as sco and the metric files write it ("2026-11-24T06:39:52Z"); no zone = UTC."""
    d = datetime.fromisoformat(s.strip().replace("Z", "+00:00"))
    return (d if d.tzinfo else d.replace(tzinfo=timezone.utc)).timestamp()


def try_utc(s: Any) -> Optional[float]:
    try:
        return parse_utc(s)
    except (AttributeError, TypeError, ValueError):
        return None


# ------------------------------------------------------------------------------------------------ reading
class Tail:
    """Incremental reader of a growing text file: the complete lines added since the last call.  A file that
    cannot be read now (missing, a stale handle) gives no lines and is read from the same place next time."""

    def __init__(self, path: Path):
        self.path, self.pos, self.rest = Path(path), 0, b""

    def lines(self) -> List[str]:
        try:
            size = self.path.stat().st_size
            if size < self.pos:  # truncated or replaced: start over
                self.pos, self.rest = 0, b""
            if size == self.pos:
                return []
            with open(self.path, "rb") as f:
                f.seek(self.pos)
                data = f.read(size - self.pos)
        except OSError:
            return []
        self.pos += len(data)
        *done, self.rest = (self.rest + data).split(b"\n")
        return [d.decode("utf-8", errors="replace") for d in done]


def log_steps(lines: List[str], stats: Optional[Dict[str, int]] = None) -> List[Tuple[int, int, Dict[str, Any]]]:
    """(step, total, log dict) for every loss dict in the trainer's console log; stats["dropped"] counts the dicts
    that could not be read (no progress bar before them, or not a literal)."""
    stats = {} if stats is None else stats
    out = []
    for line in lines:
        start = 0
        while True:
            i = line.find("{'loss'", start)
            if i < 0:
                break
            j = line.find("}", i)
            if j < 0:
                stats["dropped"] = stats.get("dropped", 0) + 1
                break
            steps = STEP_RE.findall(line[:i])
            start = j + 1
            if not steps:
                stats["dropped"] = stats.get("dropped", 0) + 1
                continue
            try:
                d = ast.literal_eval(line[i:j + 1])
            except (ValueError, SyntaxError):
                stats["dropped"] = stats.get("dropped", 0) + 1
                continue
            out.append((int(steps[-1][0]), int(steps[-1][1]), d))
    return out


def num(v: Any) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def json_lines(lines: List[str]) -> Tuple[List[Dict[str, Any]], int]:
    good, bad = [], 0
    for line in lines:
        if not line.strip():
            continue
        try:
            r = json.loads(line)
        except ValueError:
            bad += 1
            continue
        if isinstance(r, dict):
            good.append(r)
        else:
            bad += 1
    return good, bad


# ------------------------------------------------------------------------------------------------ rules
def rule_nan(step: int, d: Dict[str, Any]) -> Optional[str]:
    for k in FINITE_KEYS:
        if k in d:
            v = num(d[k])
            if v is None or not math.isfinite(v):
                return f"step {step}: {k} = {d[k]!r}"
    return None


def rule_head(history: List[Tuple[int, float, float]], window: int = HEAD_WINDOW) -> Optional[str]:
    """history: (step, nfp_cos, copy_cos) of the logged steps with NFP pairs, in step order."""
    if len(history) < window:
        return None
    last = history[-window:]
    if all(n >= c for _, n, c in last):
        return (f"steps {last[0][0]}-{last[-1][0]}: nfp_cos >= copy_cos at all {window} steps with NFP pairs "
                f"(last {last[-1][1]:.4g} vs {last[-1][2]:.4g})")
    return None


def rule_throughput(step_s: Dict[int, float], base: float = BASE_STEP_S, start: int = THROUGHPUT_FROM,
                    window: int = THROUGHPUT_WINDOW) -> Optional[str]:
    """step_s: this attempt's step times; at least `window` of them (a resume's first step is slow too)."""
    if len(step_s) < window or max(step_s) < start:
        return None
    ks = sorted(step_s)[-window:]
    med = statistics.median(step_s[k] for k in ks)
    if med > 2 * base:
        return f"median step time {med:.1f} s over steps {ks[0]}-{ks[-1]} > 2 x {base} s"
    return None


def stall_limit(last_step: int, save_steps: int, total: Optional[int]) -> float:
    if total and last_step >= total:
        return FINAL_STALL_MIN
    return SAVE_STALL_MIN if save_steps and last_step % save_steps == 0 else STALL_MIN


def rule_stall(now: float, job_start: Optional[float], last_step: Optional[int], last_step_t: Optional[float],
               save_steps: int, total: Optional[int] = None, resumed: bool = False) -> Optional[str]:
    """last_step / last_step_t: the latest step event; one older than the job's start is an earlier attempt's."""
    if last_step_t is not None and job_start is not None and last_step_t < job_start:
        last_step = last_step_t = None
    if last_step is None or last_step_t is None:
        first = RESUME_FIRST_STEP_MIN if resumed else FIRST_STEP_MIN
        if job_start is not None and now - job_start > first * 60:
            return f"no step {(now - job_start) / 60:.0f} min after the job started (limit {first:.0f})"
        return None
    limit = stall_limit(last_step, save_steps, total)
    idle = (now - last_step_t) / 60
    if idle > limit:
        return f"no new step for {idle:.0f} min after step {last_step} (limit {limit:.0f})"
    return None


def rule_host_mem(used: Dict[str, float], limit: float = HOST_MEM_GB) -> Optional[str]:
    over = {k: v for k, v in used.items() if v > limit}
    if over:
        k = max(over, key=over.get)
        return f"host used memory {over[k]:.0f} GB > {limit:.0f} ({k})"
    return None


def checkpoint_ok(ckpt: Path, n: int) -> bool:
    if (ckpt / f"checkpoint-{n}" / "trainer_state.json").is_file():
        return True
    later = [p.name.split("-", 1)[1] for p in ckpt.glob("checkpoint-*")]
    return any(m.isdigit() and int(m) > n for m in later)


def rule_save(ckpt: Path, last_step: Optional[int], save_steps: int, copies: List[Dict[str, Any]],
              logs_ok: Dict[str, Optional[bool]], now: float) -> Tuple[Optional[str], List[str]]:
    warns = []
    for c in copies:
        name = Path(c.get("checkpoint", "?")).name
        st = c.get("exit_status")
        if st is not None and st != 0:
            return f"bf16 copy of {name}: exit_status {st}", warns
        if st == 0 and logs_ok.get(name) is False:
            return f"bf16 copy of {name}: its log says ok false", warns
        started = try_utc(c.get("started"))
        if st is None and started is not None:
            mins = (now - started) / 60
            if mins > BF16_WARN_MIN:
                warns.append(f"bf16 copy of {name} running {mins:.0f} min")
    if last_step and save_steps:
        n = ((last_step - 1) // save_steps) * save_steps  # the last save step before the last logged step
        if n > 0 and not checkpoint_ok(ckpt, n):
            return f"step {last_step} logged but {ckpt}/checkpoint-{n} has no trainer_state.json", warns
    return None, warns


def final_check(ckpt: Path, last_step: Optional[int], total: Optional[int], copies: List[Dict[str, Any]],
                logs_ok: Dict[str, Optional[bool]], want_copy: bool = True) -> Optional[str]:
    """At SUCCEEDED: all steps logged, the final save there, the last checkpoint's bf16 copy done and verified
    (want_copy: the run saves checkpoints and q38_sft.json asks for bf16 copies)."""
    if total and (last_step or 0) < total:
        return f"SUCCEEDED at step {last_step} of {total}"
    if not (ckpt / "trainer_state.json").is_file():
        return f"SUCCEEDED but {ckpt}/trainer_state.json is missing (no final save)"
    if total and want_copy:
        name = f"checkpoint-{total}"
        mine = [c for c in copies if Path(c.get("checkpoint", "")).name == name]
        if not mine or mine[-1].get("exit_status") != 0 or logs_ok.get(name) is not True:
            return f"SUCCEEDED but the bf16 copy of {name} is missing, failed or not verified"
    return None


# ------------------------------------------------------------------------------------------------ the guard
def run_sco(cmd: List[str], timeout: float) -> subprocess.CompletedProcess:
    """Run sco in its own process group; on a timeout kill the whole group (sco-acp outlives a killed sco)."""
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True)
    try:
        out, err = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except OSError:
            pass
        p.communicate()
        raise
    return subprocess.CompletedProcess(cmd, p.returncode, out, err)


def describe(job: str, runner: Callable = run_sco) -> Dict[str, Any]:
    try:
        res = runner([SCO, "acp", "jobs", "describe", f"--workspace-name={WORKSPACE}", job, "-o", "json"],
                     timeout=120)
    except subprocess.TimeoutExpired:
        return {"state": None, "error": "sco describe timed out"}
    except OSError as e:
        return {"state": None, "error": f"sco describe: {e}"}
    try:
        d = json.loads(res.stdout)
    except ValueError:
        return {"state": None, "error": (res.stdout + res.stderr)[-300:]}
    return d.get("job", d) if isinstance(d, dict) else {"state": None, "error": "not an object"}


def submitted_jobs(run: Path) -> Dict[str, Optional[str]]:
    """job id -> job name of the run's successful submissions (RUN/submits.jsonl, launch submit's records)."""
    p = Path(run) / "submits.jsonl"
    if not p.is_file():
        return {}
    out: Dict[str, Optional[str]] = {}
    for r in json_lines(p.read_text().splitlines())[0]:
        m = JOB_ID_RE.search(r.get("stdout") or "")
        if m and r.get("rc") == 0:
            out[m.group(1)] = r.get("job_name")
    return out


class Guard:
    def __init__(self, run: Path, job: Optional[str] = None, base_step_s: float = BASE_STEP_S,
                 host_mem_gb: float = HOST_MEM_GB, head_window: int = HEAD_WINDOW,
                 describer: Callable[[str], Dict[str, Any]] = describe):
        import yaml  # in the verl env; train.yaml fixes save_steps

        self.run, self.job = Path(run), job
        self.base, self.mem_limit, self.window = base_step_s, host_mem_gb, head_window
        self.describer = describer
        s4 = json.loads((self.run / "q38s4.json").read_text())
        self.head_on = float(s4.get("nfp_mse", 0)) > 0 or float(s4.get("nfp_cos", 0)) > 0
        cfg = yaml.safe_load((self.run / "train.yaml").read_text())
        self.save_steps = int(cfg.get("save_steps") or 0) if cfg.get("save_strategy", "steps") == "steps" else 0
        plan = json.loads((self.run / "plan.json").read_text())
        self.total: Optional[int] = int(cfg.get("max_steps") or 0) or plan.get("estimated_steps")
        sft = self.run / "q38_sft.json"
        self.want_copy = bool(self.save_steps) and sft.is_file() and bool(json.loads(sft.read_text()).get("bf16_copy"))
        self.log = Tail(self.run / "logs" / "node0.log")
        self.rank0 = Tail(self.run / "sft_metrics" / "rank0.jsonl")
        self.host: Dict[str, Tail] = {}
        self.host_first: Dict[str, Optional[float]] = {}  # each host-memory file's first record time
        self.steps: Dict[int, Dict[str, Any]] = {}
        self.head: Dict[int, Tuple[int, float, float]] = {}
        self.step_s: Dict[int, float] = {}
        self.step_t: Dict[int, float] = {}
        self.used: Dict[str, float] = {}
        self.bad_json = 0
        self.log_stats: Dict[str, int] = {}
        self.attempts = 0  # node-0 headers seen in node0.log
        self.nan: Optional[str] = None
        self.describe_fails = 0

    def read(self) -> None:
        for line in self.log.lines():
            if line.startswith(HEADER) and " host " in line and " node_rank 0 " in line:
                self.attempts += 1  # a new attempt: its steps alone count from here
                self.steps.clear()
                self.head.clear()
                self.nan = None
                continue
            for step, total, d in log_steps([line], self.log_stats):
                self.total = total or self.total  # the progress bar's total is the trainer's own count
                self.steps[step] = d
                self.nan = self.nan or rule_nan(step, d)
                n, c = num(d.get("nfp_cos")), num(d.get("copy_cos"))
                if num(d.get("nfp_pairs")) and n is not None and c is not None:
                    self.head[step] = (step, n, c)
                else:
                    self.head.pop(step, None)
        ev, bad = json_lines(self.rank0.lines())
        self.bad_json += bad
        for e in ev:
            if e.get("event") == "train_begin":
                self.step_s.clear()
                self.step_t.clear()
                self.used.pop("rank0 step events", None)
                continue
            if e.get("event") != "step" or "step" not in e:
                continue
            k = int(e["step"])
            if num(e.get("step_s")) is not None:
                self.step_s[k] = float(e["step_s"])
            t = try_utc(e.get("utc"))
            if t is not None:
                self.step_t[k] = t
            u = num(e.get("used_gb"))
            if u is not None:
                self.used["rank0 step events"] = max(self.used.get("rank0 step events", 0.0), u)
        for p in sorted((self.run / "sft_metrics").glob("host_mem_*.jsonl")):
            recs, bad = json_lines(self.host.setdefault(p.name, Tail(p)).lines())
            self.bad_json += bad
            for r in recs:
                if p.name not in self.host_first:
                    self.host_first[p.name] = try_utc(r.get("utc"))
                for part in ("mem", "peak"):
                    u = num((r.get(part) or {}).get("used_gb"))
                    if u is not None:
                        self.used[p.name] = max(self.used.get(p.name, 0.0), u)

    def used_now(self, job_start: Optional[float]) -> Dict[str, float]:
        """Used memory of this attempt: host-memory files begun before the job's start are an earlier job's."""
        if job_start is None:
            return dict(self.used)
        return {k: v for k, v in self.used.items()
                if self.host_first.get(k) is None or self.host_first[k] >= job_start}

    def bf16(self) -> Tuple[List[Dict[str, Any]], Dict[str, Optional[bool]]]:
        p = self.run / "sft_metrics" / "bf16_copies.json"
        try:
            copies = json.loads(p.read_text())
        except (OSError, ValueError):
            return [], {}
        if not isinstance(copies, list):
            return [], {}
        ok: Dict[str, Optional[bool]] = {}
        for c in copies:
            name = Path(c.get("checkpoint", "?")).name
            try:
                ok[name] = json.loads(Path(c["log"]).read_text()).get("ok")
            except (KeyError, OSError, ValueError, TypeError, AttributeError):
                ok[name] = None
        return copies, ok

    def resumed(self) -> bool:
        return any(p.name.split("-", 1)[-1].isdigit() for p in (self.run / "ckpt").glob("checkpoint-*"))

    def check(self, now: Optional[float] = None) -> Dict[str, Any]:
        now = time.time() if now is None else now
        self.read()
        job = self.describer(self.job) if self.job else {}
        state = job.get("state")
        if self.job:
            self.describe_fails = self.describe_fails + 1 if state is None else 0
        job_start = try_utc(job.get("start_time"))
        last = max(self.steps) if self.steps else None
        last_ev = max(self.step_t) if self.step_t else None
        last_t = max(self.step_t.values()) if self.step_t else None
        copies, logs_ok = self.bf16()
        head = [self.head[k] for k in sorted(self.head)]
        used = self.used_now(job_start)
        alerts: Dict[str, str] = {}
        if self.nan:
            alerts["nan"] = self.nan
        if self.head_on:
            r = rule_head(head, self.window)
            if r:
                alerts["head"] = r
        r = rule_throughput(self.step_s, self.base)
        if state == "RUNNING":
            r = r or rule_stall(now, job_start, last_ev, last_t, self.save_steps, self.total,
                                resumed=self.resumed())
        if r:
            alerts["throughput"] = r
        r = rule_host_mem(used, self.mem_limit)
        if r:
            alerts["host_mem"] = r
        r, warns = rule_save(self.run / "ckpt", last, self.save_steps, copies, logs_ok, now)
        if r:
            alerts["save"] = r
        ended = state in END_STATES
        if ended:
            r = (final_check(self.run / "ckpt", last, self.total, copies, logs_ok, self.want_copy)
                 if state == "SUCCEEDED"
                 else f"job {self.job} is {state} at step {last} of {self.total}")
            if r:
                alerts.setdefault("save", r)
        if state in WAIT_WARN_STATES:
            warns.append(f"job {self.job} is {state} (no stall rule while it waits)")
        if self.describe_fails >= DESCRIBE_WARN:
            warns.append(f"sco describe failed {self.describe_fails} passes in a row: no stall or end rules meanwhile")
        if last_ev is not None and last_ev - (last or 0) > LOG_LAG_WARN:
            warns.append(f"rank 0 is at step {last_ev} but node0.log's last log dict is step {last}")
        recent = [self.step_s[k] for k in sorted(self.step_s)[-THROUGHPUT_WINDOW:]]
        tail = [self.steps[k] for k in sorted(self.steps)[-10:]]
        h = head[-self.window:]
        return {
            "utc": utc(now), "run": str(self.run), "job": self.job, "state": state, "head_on": self.head_on,
            "describe_error": job.get("error"), "attempts": self.attempts, "last_step": last,
            "last_step_event": last_ev, "total_steps": self.total,
            "last_step_utc": utc(last_t) if last_t is not None else None,
            "median_step_s_20": round(statistics.median(recent), 2) if recent else None,
            "loss_last10": [num(d.get("loss")) for d in tail],
            "nfp_cos_below_copy_last200": [sum(1 for _, n, c in h if n < c), len(h)] if self.head_on else None,
            "max_host_used_gb": round(max(used.values()), 1) if used else None,
            "bad_json_lines": self.bad_json, "log_dicts_dropped": self.log_stats.get("dropped", 0),
            "bf16_copies": [[Path(c.get("checkpoint", "?")).name, c.get("exit_status")] for c in copies],
            "alerts": alerts, "warnings": warns, "ended": ended,
        }


def stop_job(job: str, expected_name: Optional[str], describer: Callable = describe, runner: Callable = run_sco,
             sleep: Callable[[float], None] = time.sleep, tries: int = STOP_TRIES, verify: int = STOP_VERIFY,
             wait_s: float = STOP_WAIT_S) -> Dict[str, Any]:
    """Stop a RUNNING job with the expected name.  Before each try (up to `tries`) the job is described again
    and only a RUNNING job under that name is stopped; after sco stop returns 0 the job is described until it
    has left RUNNING (up to `verify` times, `wait_s` apart).  stopped is True only when a describe after a stop
    command shows another state; a job not RUNNING under the name at the first describe is left alone."""
    rec: Dict[str, Any] = {"utc": utc(), "tries": [], "stopped": False}
    for i in range(tries):
        d = describer(job)
        if i == 0:
            rec.update(state_before=d.get("state"), display_name=d.get("display_name"))
        if d.get("state") != "RUNNING" or not expected_name or d.get("display_name") != expected_name:
            rec["why"] = f"not RUNNING under the expected name (state {d.get('state')})"
            if i and d.get("state") not in (None, "RUNNING"):
                rec.update(stopped=True, state_after=d.get("state"))  # it left RUNNING after an earlier try
            return rec
        try:
            res = runner([SCO, "acp", "jobs", "stop", f"--workspace-name={WORKSPACE}", job], timeout=180)
            rec["tries"].append({"utc": utc(), "rc": res.returncode, "out": (res.stdout + res.stderr)[-300:]})
            if res.returncode == 0:
                break
        except subprocess.TimeoutExpired:
            rec["tries"].append({"utc": utc(), "rc": None, "out": "sco stop timed out"})
        except OSError as e:
            rec["tries"].append({"utc": utc(), "rc": None, "out": f"sco stop: {e}"})
        if i + 1 < tries:
            sleep(wait_s * (i + 1))
    if not rec["tries"] or rec["tries"][-1]["rc"] != 0:
        rec["why"] = "sco stop did not succeed; check the job"
        return rec
    for _ in range(verify):
        sleep(wait_s)
        s = describer(job).get("state")
        rec["state_after"] = s
        if s is not None and s != "RUNNING":
            break
    rec["stopped"] = rec.get("state_after") not in (None, "RUNNING")
    if not rec["stopped"]:
        rec["why"] = "sco stop returned 0 but the job has not left RUNNING; check the job"
    return rec


def write_json(path: Path, obj: Any) -> bool:
    try:
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(obj, indent=1) + "\n")
        os.replace(tmp, path)
        return True
    except OSError:
        return False


def append_line(path: Path, obj: Any) -> None:
    try:
        with open(path, "a") as f:
            f.write(json.dumps(obj) + "\n")
    except OSError:
        pass


def watch(guard: Guard, out: Path, interval: float, stop: bool, expected_name: Optional[str],
          sleep: Callable[[float], None] = time.sleep, max_passes: Optional[int] = None) -> Tuple[int, Dict]:
    out.mkdir(parents=True, exist_ok=True)
    passes = fails = 0
    st: Dict[str, Any] = {}
    while True:
        pause = interval
        try:
            st = guard.check()
            if fails >= GUARD_FAILS and (out / "GUARD_ERROR.json").exists():
                try:
                    os.replace(out / "GUARD_ERROR.json", out / f"GUARD_ERROR_{utc().replace(':', '')}.json")
                except OSError:
                    pass
            fails = 0
        except Exception:  # noqa: BLE001 - the guard must keep watching through a bad read
            fails += 1
            st = {"utc": utc(), "guard_error": traceback.format_exc()[-1500:], "fails_in_a_row": fails}
            if fails >= GUARD_FAILS:
                write_json(out / "GUARD_ERROR.json", st)
            pause = min(interval * 2 ** min(fails - 1, 3), max(interval, MAX_PAUSE_S))
        write_json(out / "status.json", st)
        append_line(out / "guard.jsonl", st)
        if st.get("alerts"):
            write_json(out / "ALERT.json", {**st, "stop": "pending" if (stop and guard.job) else None})
            st["stop"] = stop_job(guard.job, expected_name, guard.describer) if (stop and guard.job) else None
            write_json(out / "ALERT.json", st)
            return 1, st
        if st.get("ended"):
            write_json(out / "END.json", st)
            return 0, st
        passes += 1
        if max_passes is not None and passes >= max_passes:
            return 0, st
        sleep(pause)


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("check", "watch"):
        p = sub.add_parser(name)
        p.add_argument("--run", required=True)
        p.add_argument("--job")
        p.add_argument("--base-step-s", type=float, default=BASE_STEP_S)
        if name == "watch":
            p.add_argument("--out", required=True)
            p.add_argument("--interval", type=float, default=120.0)
            p.add_argument("--stop", action="store_true", help="stop the job when a rule fires")
            p.add_argument("--job-name", help="if given, must equal the job name RUN/submits.jsonl records")
    a = ap.parse_args(argv)
    name = None
    if a.cmd == "watch":
        if not a.job:
            ap.error("watch needs --job")
        jobs = submitted_jobs(Path(a.run))
        if a.job not in jobs:
            ap.error(f"job {a.job} is not among the run's submissions (RUN/submits.jsonl): {sorted(jobs)}")
        name = jobs[a.job]
        if a.job_name and a.job_name != name:
            ap.error(f"--job-name {a.job_name} but RUN/submits.jsonl records {a.job} as {name}")
        if a.stop and not name:
            ap.error(f"RUN/submits.jsonl records no job name for {a.job}: no stop possible")
    g = Guard(Path(a.run), a.job, a.base_step_s)
    if a.cmd == "check":
        print(json.dumps(g.check(), indent=1))
        return
    rc, st = watch(g, Path(a.out), a.interval, a.stop, name)
    print(json.dumps(st, indent=1))
    sys.exit(rc)


if __name__ == "__main__":
    main()
