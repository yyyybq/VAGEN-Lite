"""Reward line, phase 3, experiment D (docs/reward/PREREG_P3_D.md): drift by the brief's definition, from the
training log, for any RL run of this line or another.

The measure is the one the chain's guard reads (train_chain.guard_reason): the trainer's console lines
"step:N - ... - transition/invalid_action_rate/mean:V - ...", V = the share of unreadable turns per episode, averaged
over the update's episodes.  The 10-update mean at update k is the mean of the values of the last 10 logged updates
ending at k, defined once 10 exist (as the guard: ``len(inv) >= w``).  A run's logs may be several files (a trainer
retried from a checkpoint): they are read in name order (the start time is in the name) and a later file's value of an
update replaces an earlier one, and a later file's first logged update also drops the earlier file's values from
there on (a retry resumes from a checkpoint; the abandoned attempt's tail is not the run).

Definitions (PREREG_P3_D section 1, the user's brief of 2026-11-06):
* stopped: the chain's guard stopped the run for unreadable answers (progress.json guard_stops, "unparseable answers
  ..."), or, without a progress file, the 10-update mean went above ``stop`` (0.10); a run ended by another stop
  (the greedy-validation stop line, KL, a success drop, a failure) is not "stopped" here: it is judged by the next
  clause on the updates it reached, and the reason is reported;
* drift: stopped, or the 10-update mean went above ``warn`` (0.01) and from then on never fell below ``recover``
  (0.005) up to the run's last logged update.  Equivalently: there is an update above 1% after the last update below
  0.5% (or with no update below 0.5% at all after it);
* warnings: the times the 10-update mean went from at or below 1% to above 1% (the first defined value above 1% counts
  as one), the report-only 1% warning of the brief;
* also: the first update above 1%, the peak and its update, the first update back below 0.5% after each warning, and
  whether the last warning came late (after ``late``, 250, so the run had little room to fall back).

    python -m reward.drift_p3 --run NAME=RUN_DIR[@PROGRESS_JSON] [...] --out FILE.json

RUN_DIR is the run's folder with logs/rl-*worker-0.log (e.g. reward_runs/runs/rw_b_w1/rl_run_rw_s0); PROGRESS_JSON
is the chain's progress.json (its guard_stops[NAME] is read).  Writes FILE.json and FILE.md.  Read only, standard
library only, one core.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

KEY = "transition/invalid_action_rate/mean"
_STEP = re.compile(r"step:(\d+) - (.*)")
_VAL = re.compile(re.escape(KEY) + r":(?:np\.float64\()?([-+0-9.eE]+|nan)")


def series_from_text(text: str, out: Optional[Dict[int, float]] = None) -> Dict[int, float]:
    """{update: value} from one log's text; a later line of the same update replaces an earlier one."""
    out = {} if out is None else out
    for line in text.splitlines():
        m = _STEP.search(line)
        if not m:
            continue
        v = _VAL.search(m.group(2))
        if v and v.group(1) != "nan":
            out[int(m.group(1))] = float(v.group(1))
    return out


def series(logs: Sequence[Path]) -> Dict[int, float]:
    """A run's values over its log files in name (start-time) order.  A later file is a retry that resumed from a
    checkpoint: the updates it logs from its first logged update on replace everything earlier from there (an
    abandoned attempt's tail past the resume point is dropped); a file without step lines changes nothing."""
    out: Dict[int, float] = {}
    for p in sorted(logs, key=lambda q: q.name):
        with open(p, errors="replace") as f:
            part = series_from_text(f.read())
        if not part:
            continue
        s0 = min(part)
        out = {k: v for k, v in out.items() if k < s0}
        out.update(part)
    return {k: out[k] for k in sorted(out) if k > 0}


def rolling(values: Dict[int, float], window: int = 10) -> Dict[int, float]:
    """The guard's mean: at each logged update, the mean of the last ``window`` logged updates, once that many exist."""
    steps = sorted(values)
    out = {}
    for i, s in enumerate(steps):
        if i + 1 >= window:
            w = [values[t] for t in steps[i + 1 - window:i + 1]]
            out[s] = sum(w) / len(w)
    return out


def classify(values: Dict[int, float], guard_stop: Optional[str] = None, window: int = 10, warn: float = 0.01,
             recover: float = 0.005, stop: float = 0.10, late: int = 250, have_progress: bool = False) -> Dict[str, Any]:
    """The brief's definition on one run's values.  ``guard_stop``: the chain's guard_stops reason for the run (None:
    not stopped, or no progress file); ``have_progress``: whether a progress file was read.  Only the training-log
    unreadable guard ("unparseable answers ...") counts as the first clause; a run ended by any other stop (the
    greedy-validation stop line, KL, a success drop, a failure) is judged by the second clause on the updates it
    reached, and its reason is reported (PREREG_P3_D section 1).  Without a progress file, a 10-update mean above
    ``stop`` counts as stopped."""
    r = rolling(values, window)
    steps = sorted(r)
    res: Dict[str, Any] = {"updates": len(values), "last_update": max(values) if values else None,
                           "first_defined": steps[0] if steps else None}
    if not steps:
        res.update(drift=None, reason="fewer than %d logged updates" % window)
        return res
    warnings: List[Dict[str, Any]] = []
    prev = None
    for s in steps:
        above = r[s] > warn
        if above and (prev is None or prev <= warn):
            warnings.append({"update": s, "mean": r[s], "back_below_recover": None})
        if r[s] < recover:
            for w in warnings:
                if w["back_below_recover"] is None and s > w["update"]:
                    w["back_below_recover"] = s
        prev = r[s]
    peak_s = max(steps, key=lambda s: r[s])
    over_stop = [s for s in steps if r[s] > stop]
    stopped_guard = bool(guard_stop and "unparseable" in guard_stop)
    stopped = stopped_guard or (not have_progress and bool(over_stop))
    other_stop = guard_stop if guard_stop and not stopped_guard else None
    below = [s for s in steps if r[s] < recover]
    last_below = below[-1] if below else None
    never_back = any(r[s] > warn for s in steps if last_below is None or s > last_below)
    drift = stopped or never_back
    res.update({
        "drift": drift,
        "stopped": stopped, "guard_stop": guard_stop, "other_stop": other_stop,
        "first_over_stop": over_stop[0] if over_stop else None,
        "never_back_below_recover": never_back,
        "warnings": len(warnings), "warning_list": warnings,
        "first_over_warn": warnings[0]["update"] if warnings else None,
        "peak": r[peak_s], "peak_update": peak_s,
        "last_below_recover": last_below,
        "late_last_warning": bool(warnings and warnings[-1]["update"] > late),
        "max_per_update": max(values.values()),
    })
    if drift:
        res["reason"] = ("stopped by the guard" if stopped_guard else "10-update mean above %g" % stop if stopped
                         else "above %g and never back below %g" % (warn, recover))
    else:
        res["reason"] = "never above %g" % warn if not warnings else "back below %g after each warning" % recover
    return res


def run_dir_logs(d: Path) -> List[Path]:
    return sorted((d / "logs").glob("rl-*worker-0.log"))


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--run", action="append", required=True, help="NAME=RUN_DIR[@PROGRESS_JSON]")
    ap.add_argument("--out", required=True)
    ap.add_argument("--window", type=int, default=10)
    ap.add_argument("--warn", type=float, default=0.01)
    ap.add_argument("--recover", type=float, default=0.005)
    ap.add_argument("--stop", type=float, default=0.10)
    ap.add_argument("--late", type=int, default=250)
    a = ap.parse_args(argv)
    out: Dict[str, Any] = {"definition": {k: getattr(a, k) for k in ("window", "warn", "recover", "stop", "late")},
                           "metric": KEY, "runs": {}}
    for spec in a.run:
        name, rest = spec.split("=", 1)
        d, _, prog = rest.partition("@")
        logs = run_dir_logs(Path(d))
        guard = None
        if prog:
            with open(prog) as f:
                guard = ((json.load(f).get("guard_stops") or {}).get(name) or {}).get("reason")
        res = classify(series(logs), guard, a.window, a.warn, a.recover, a.stop, a.late, have_progress=bool(prog))
        res["logs"] = [str(p) for p in logs]
        out["runs"][name] = res
    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    pct = lambda v: "-" if v is None else "%.2f%%" % (100 * v)  # noqa: E731
    lines = ["| 运行 | 走样 | 理由 | 1% 预警（第几次过 1% → 第几次回落到 0.5% 以下） | 峰值（第几次） | 末次 | 停下的原因 |",
             "| --- | --- | --- | --- | --- | --- | --- |"]
    for name, r in out["runs"].items():
        ws = "；".join(f"{w['update']} → {w['back_below_recover'] or '没回落'}" for w in r.get("warning_list") or []) or "无"
        lines.append(f"| {name} | {'是' if r.get('drift') else '否' if r.get('drift') is False else '?'} | "
                     f"{r.get('reason')} | {r.get('warnings', 0)} 次：{ws} | "
                     f"{pct(r.get('peak'))}（{r.get('peak_update', '-')}） | {r.get('last_update')} | "
                     f"{r.get('guard_stop') or '-'} |")
    Path(a.out).with_suffix(".md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
