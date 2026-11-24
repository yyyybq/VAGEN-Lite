"""C line (q38 s4): the guards of a formal training run (stage 4), run on the login node.

  python -m vagen.q38s4.tools.guard check --run RUN [--job JOB]
  python -m vagen.q38s4.tools.guard watch --run RUN --job JOB --job-name NAME --out DIR [--interval 120] [--stop]

``check`` reads the run once and prints its status (JSON).  ``watch`` does so every --interval seconds and writes
DIR/status.json after each pass and DIR/guard.jsonl (one line per pass).  When a rule fires it writes DIR/ALERT.json,
stops the job (only with --stop, and only after describing the job again: it must still be RUNNING under NAME) and
exits 1.  When the job ends it writes DIR/END.json (with the final checks) and exits 0, or 1 if they fail.  When
the guard itself fails GUARD_FAILS passes in a row it writes DIR/GUARD_ERROR.json and exits 2 without touching the
job.  The run folder is only read.

Rules (the brief's stage 4: "损失 NaN；nfp_cos 连续 200 步不低于复制基线（头没学到）；吞吐掉到基准一半以下（机器故障）；
主机内存 > 1,600 GB；存盘失败"):

  nan         a logged step (RUN/logs/node0.log, the trainer's log dict) with a non-finite loss, grad_norm,
              text_loss, nfp_mse or nfp_cos.
  head        C-NFP only (q38s4.json weights > 0): the last HEAD_WINDOW (200) logged steps with NFP pairs all have
              nfp_cos >= copy_cos (the head never beat "predict = the current unit's features" in 200 steps).
  throughput  from step THROUGHPUT_FROM (21): the median step_s of the last 20 steps (rank 0's step events; equal to
              the slowest rank's within 0.1% in the smoke runs) above 2 x BASE_STEP_S (25.76 s, S3's 8-node median),
              i.e. samples per second below half of S3's; or, while the job is RUNNING, no new step for STALL_MIN (30)
              minutes, SAVE_STALL_MIN (60) after a save step, FIRST_STEP_MIN (75) after the job's start before step 1
              (a hang is throughput 0; NCCL's timeout is 3 hours).
  host_mem    a node's used memory (MemTotal - MemAvailable: used_gb in the host-memory records, current and peak,
              and in rank 0's step events) above HOST_MEM_GB (1,600).  The page cache (cgroup current_gb) is not
              counted: it grows with every save and the kernel gives it back.
  save        a bf16 copy (sft_metrics/bf16_copies.json and its log) with exit_status != 0 or ok false; a step after
              save step N logged while ckpt/checkpoint-N has no trainer_state.json and no later checkpoint exists;
              the job leaving RUNNING other than SUCCEEDED; at SUCCEEDED, fewer steps than the progress bar's total,
              no trainer_state.json in ckpt/, or no successful bf16 copy of the last checkpoint.
              A bf16 copy running longer than BF16_WARN_MIN (45) minutes is a warning only (31 min in S1 v2).

The step of a log dict is the progress bar's "N/TOTAL" printed before it on the same line.  Files are read
incrementally; a line not yet ended by a newline waits for the next pass; an unreadable JSON line (all nodes append
to one host-memory file) is skipped and counted.
"""
from __future__ import annotations

import argparse
import ast
import json
import math
import os
import re
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
STALL_MIN, SAVE_STALL_MIN, FIRST_STEP_MIN = 30.0, 60.0, 75.0
HOST_MEM_GB = 1600.0
BF16_WARN_MIN = 45.0
GUARD_FAILS = 5
FINITE_KEYS = ("loss", "grad_norm", "text_loss", "nfp_mse", "nfp_cos")
END_STATES = ("SUCCEEDED", "FAILED", "STOPPED", "SUSPENDED", "DELETED", "CANCELLED", "KILLED")

STEP_RE = re.compile(r"(\d+)/(\d+) \[")


def utc(t: Optional[float] = None) -> str:
    return datetime.fromtimestamp(time.time() if t is None else t, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_utc(s: str) -> float:
    """ISO time as sco and the metric files write it ("2026-11-24T06:39:52Z"); no zone = UTC."""
    d = datetime.fromisoformat(s.strip().replace("Z", "+00:00"))
    return (d if d.tzinfo else d.replace(tzinfo=timezone.utc)).timestamp()


# ------------------------------------------------------------------------------------------------ reading
class Tail:
    """Incremental reader of a growing text file: the complete lines added since the last call."""

    def __init__(self, path: Path):
        self.path, self.pos, self.rest = Path(path), 0, b""

    def lines(self) -> List[str]:
        try:
            size = self.path.stat().st_size
        except FileNotFoundError:
            return []
        if size < self.pos:  # truncated or replaced: start over
            self.pos, self.rest = 0, b""
        if size == self.pos:
            return []
        with open(self.path, "rb") as f:
            f.seek(self.pos)
            data = f.read(size - self.pos)
        self.pos += len(data)
        *done, self.rest = (self.rest + data).split(b"\n")
        return [d.decode("utf-8", errors="replace") for d in done]


def log_steps(lines: List[str]) -> List[Tuple[int, int, Dict[str, Any]]]:
    """(step, total, log dict) for every loss dict in the trainer's console log."""
    out = []
    for line in lines:
        start = 0
        while True:
            i = line.find("{'loss'", start)
            if i < 0:
                break
            j = line.find("}", i)
            if j < 0:
                break
            steps = STEP_RE.findall(line[:i])
            start = j + 1
            if not steps:
                continue
            try:
                d = ast.literal_eval(line[i:j + 1])
            except (ValueError, SyntaxError):
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
    if not step_s or max(step_s) < start:
        return None
    ks = sorted(step_s)[-window:]
    med = statistics.median(step_s[k] for k in ks)
    if med > 2 * base:
        return f"median step time {med:.1f} s over steps {ks[0]}-{ks[-1]} > 2 x {base} s"
    return None


def rule_stall(now: float, job_start: Optional[float], last_step: Optional[int], last_step_t: Optional[float],
               save_steps: int) -> Optional[str]:
    if last_step is None or last_step_t is None:
        if job_start is not None and now - job_start > FIRST_STEP_MIN * 60:
            return f"no step {(now - job_start) / 60:.0f} min after the job started (limit {FIRST_STEP_MIN:.0f})"
        return None
    limit = SAVE_STALL_MIN if save_steps and last_step % save_steps == 0 else STALL_MIN
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
        if st is None and c.get("started"):
            mins = (now - parse_utc(c["started"])) / 60
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
def describe(job: str, runner: Callable = subprocess.run) -> Dict[str, Any]:
    try:
        res = runner([SCO, "acp", "jobs", "describe", f"--workspace-name={WORKSPACE}", job, "-o", "json"],
                     capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return {"state": None, "error": "sco describe timed out"}
    try:
        d = json.loads(res.stdout)
    except ValueError:
        return {"state": None, "error": (res.stdout + res.stderr)[-300:]}
    return d.get("job", d) if isinstance(d, dict) else {"state": None, "error": "not an object"}


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
        self.steps: Dict[int, Dict[str, Any]] = {}
        self.head: Dict[int, Tuple[int, float, float]] = {}
        self.step_s: Dict[int, float] = {}
        self.step_t: Dict[int, float] = {}
        self.used: Dict[str, float] = {}
        self.bad_json = 0
        self.nan: Optional[str] = None

    def read(self) -> None:
        for step, total, d in log_steps(self.log.lines()):
            self.total = total or self.total  # the progress bar's total is the trainer's own count
            self.steps[step] = d  # a resumed run logs steps again: the latest wins
            self.nan = self.nan or rule_nan(step, d)
            n, c = num(d.get("nfp_cos")), num(d.get("copy_cos"))
            if num(d.get("nfp_pairs")) and n is not None and c is not None:
                self.head[step] = (step, n, c)
            else:
                self.head.pop(step, None)
        ev, bad = json_lines(self.rank0.lines())
        self.bad_json += bad
        for e in ev:
            if e.get("event") != "step" or "step" not in e:
                continue
            k = int(e["step"])
            if num(e.get("step_s")) is not None:
                self.step_s[k] = float(e["step_s"])
            if e.get("utc"):
                self.step_t[k] = parse_utc(e["utc"])
            u = num(e.get("used_gb"))
            if u is not None:
                self.used["rank0 step events"] = max(self.used.get("rank0 step events", 0.0), u)
        for p in sorted((self.run / "sft_metrics").glob("host_mem_*.jsonl")):
            recs, bad = json_lines(self.host.setdefault(p.name, Tail(p)).lines())
            self.bad_json += bad
            for r in recs:
                for part in ("mem", "peak"):
                    u = num((r.get(part) or {}).get("used_gb"))
                    if u is not None:
                        self.used[p.name] = max(self.used.get(p.name, 0.0), u)

    def bf16(self) -> Tuple[List[Dict[str, Any]], Dict[str, Optional[bool]]]:
        p = self.run / "sft_metrics" / "bf16_copies.json"
        try:
            copies = json.loads(p.read_text())
        except (FileNotFoundError, ValueError):
            return [], {}
        ok: Dict[str, Optional[bool]] = {}
        for c in copies:
            name = Path(c.get("checkpoint", "?")).name
            try:
                ok[name] = json.loads(Path(c["log"]).read_text()).get("ok")
            except (KeyError, OSError, ValueError, TypeError, AttributeError):
                ok[name] = None
        return copies, ok

    def check(self, now: Optional[float] = None) -> Dict[str, Any]:
        now = time.time() if now is None else now
        self.read()
        job = self.describer(self.job) if self.job else {}
        state = job.get("state")
        job_start = parse_utc(job["start_time"]) if job.get("start_time") else None
        last = max(self.steps) if self.steps else None
        last_ev = max(self.step_t) if self.step_t else None
        copies, logs_ok = self.bf16()
        head = [self.head[k] for k in sorted(self.head)]
        alerts: Dict[str, str] = {}
        if self.nan:
            alerts["nan"] = self.nan
        if self.head_on:
            r = rule_head(head, self.window)
            if r:
                alerts["head"] = r
        r = rule_throughput(self.step_s, self.base)
        if state == "RUNNING":
            r = r or rule_stall(now, job_start, last_ev, self.step_t.get(last_ev) if last_ev else None,
                                self.save_steps)
        if r:
            alerts["throughput"] = r
        r = rule_host_mem(self.used, self.mem_limit)
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
        recent = [self.step_s[k] for k in sorted(self.step_s)[-THROUGHPUT_WINDOW:]]
        tail = [self.steps[k] for k in sorted(self.steps)[-10:]]
        h = head[-self.window:]
        return {
            "utc": utc(now), "run": str(self.run), "job": self.job, "state": state, "head_on": self.head_on,
            "describe_error": job.get("error"), "last_step": last, "total_steps": self.total,
            "last_step_utc": utc(self.step_t[last_ev]) if last_ev else None,
            "median_step_s_20": round(statistics.median(recent), 2) if recent else None,
            "loss_last10": [num(d.get("loss")) for d in tail],
            "nfp_cos_below_copy_last200": [sum(1 for _, n, c in h if n < c), len(h)] if self.head_on else None,
            "max_host_used_gb": round(max(self.used.values()), 1) if self.used else None,
            "bad_json_lines": self.bad_json,
            "bf16_copies": [[Path(c.get("checkpoint", "?")).name, c.get("exit_status")] for c in copies],
            "alerts": alerts, "warnings": warns, "ended": ended,
        }


def stop_job(job: str, expected_name: Optional[str], describer: Callable = describe,
             runner: Callable = subprocess.run) -> Dict[str, Any]:
    """Describe the job again right before stopping it; stop only a RUNNING job with the expected name."""
    d = describer(job)
    rec: Dict[str, Any] = {"utc": utc(), "state_before": d.get("state"), "display_name": d.get("display_name")}
    if d.get("state") != "RUNNING" or not expected_name or d.get("display_name") != expected_name:
        rec.update(stopped=False, why="not RUNNING under the expected name")
        return rec
    try:
        res = runner([SCO, "acp", "jobs", "stop", f"--workspace-name={WORKSPACE}", job], capture_output=True,
                     text=True, timeout=180)
        rec.update(stopped=res.returncode == 0, rc=res.returncode, out=(res.stdout + res.stderr)[-500:])
    except subprocess.TimeoutExpired:
        rec.update(stopped=None, why="sco stop timed out; check the job")
    return rec


def write_json(path: Path, obj: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1) + "\n")
    os.replace(tmp, path)


def watch(guard: Guard, out: Path, interval: float, stop: bool, expected_name: Optional[str],
          sleep: Callable[[float], None] = time.sleep, max_passes: Optional[int] = None) -> Tuple[int, Dict]:
    out.mkdir(parents=True, exist_ok=True)
    passes = fails = 0
    st: Dict[str, Any] = {}
    while True:
        try:
            st = guard.check()
            fails = 0
        except Exception:  # noqa: BLE001 - the guard must keep watching through a bad read
            fails += 1
            st = {"utc": utc(), "guard_error": traceback.format_exc()[-1500:], "fails_in_a_row": fails}
            if fails >= GUARD_FAILS:
                write_json(out / "GUARD_ERROR.json", st)
                return 2, st
        write_json(out / "status.json", st)
        with open(out / "guard.jsonl", "a") as f:
            f.write(json.dumps(st) + "\n")
        if st.get("alerts"):
            st["stop"] = stop_job(guard.job, expected_name, guard.describer) if (stop and guard.job) else None
            write_json(out / "ALERT.json", st)
            return 1, st
        if st.get("ended"):
            write_json(out / "END.json", st)
            return 0, st
        passes += 1
        if max_passes is not None and passes >= max_passes:
            return 0, st
        sleep(interval)


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
            p.add_argument("--job-name", help="the job's display name, checked before a stop")
    a = ap.parse_args(argv)
    g = Guard(Path(a.run), a.job, a.base_step_s)
    if a.cmd == "check":
        print(json.dumps(g.check(), indent=1))
        return
    if not a.job or (a.stop and not a.job_name):
        ap.error("watch needs --job, and --stop needs --job-name")
    rc, st = watch(g, Path(a.out), a.interval, a.stop, a.job_name)
    print(json.dumps(st, indent=1))
    sys.exit(rc)


if __name__ == "__main__":
    main()
