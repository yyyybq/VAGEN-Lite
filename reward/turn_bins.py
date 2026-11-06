"""Reward line (the user's follow-ups of 2026-11-05): reward.turn_metrics' rates again, per band of tasks, and what
the failed episodes of each band did.

Bands are by the start's minimum number of actions d* (the offline metrics: 1-3, 4-5, 6-8) or by the margin
LIMIT - d* (the perturbed-start evaluation, where a row's d_star is already d* + k: margin = 12 - (d* + k)).  Reads the
per-episode records of a turn_metrics output (FILE.json "episodes": {LABEL|SET: [episode]}, each with the remaining
numbers "d" before the first turn and after every turn), merges each label's sets, and per label (or group of labels)
and band gives reward.turn_metrics.summarize plus:

* episode error share: the share of episodes with at least one error (a turn after which the remaining number grew);
* the failed episodes split by their remaining numbers:
  - no_error: the number never grew; the episode failed by stalling (turns that did not shorten it);
  - recovered_ran_out: it erred, but its last number is the smallest of the episode: it came back from its errors and
    was still on its way when the actions ran out;
  - not_recovered: it ended farther than the closest it came; not_recovered_late counts those of them whose last
    error came in one of the last two turns (no turns left to come back);
  - unknown: the last number (or every number) missing;
* failed_beyond_start: failed episodes whose last number is above their start's (they ended farther than they began);
* where failed episodes end (mean and median remaining number) and the closest they came (mean of their smallest).

    python -m reward.turn_bins --metrics FILE.json --by dstar --bands 1-3,4-5,6-8 \
        [--group NAME=PATTERN[,PATTERN ...]] [--label PATTERN ...] --out OUT.json
    python -m reward.turn_bins --metrics FILE.json --by margin --limit 12 --bands 0,1-2,3-4,5- ...

PATTERN is an fnmatch pattern on the merged labels (RUN@STEP, e.g. rw_s0@150 or rw_*@*).  OUT.json and OUT.md.
"""

import argparse
import fnmatch
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from reward.turn_metrics import summarize

INF = 10 ** 9
SPLIT = ("no_error", "recovered_ran_out", "not_recovered", "not_recovered_late", "unknown")


def parse_bands(spec: str) -> List[Tuple[str, int, int]]:
    """'1-3,4-5,6-' -> [(name, low, high)] (inclusive; an open end runs to infinity)."""
    out = []
    for part in spec.split(","):
        part = part.strip()
        lo, sep, hi = part.partition("-")
        low = int(lo)
        high = (INF if hi == "" else int(hi)) if sep else low
        if high < low:
            raise ValueError(f"band {part}: high below low")
        out.append((part, low, high))
    names = [b[0] for b in out]
    if len(set(names)) != len(names):
        raise ValueError(f"bands repeat: {spec}")
    return out


def band_of(value: int, bands: Sequence[Tuple[str, int, int]]) -> Optional[str]:
    for name, low, high in bands:
        if low <= value <= high:
            return name
    return None


def failure_kind(e: Dict[str, Any]) -> Optional[str]:
    """The failed episode's kind (see the module docstring); None for a successful one."""
    if e["success"]:
        return None
    d = e["d"]
    known = [x for x in d if x is not None]
    if not known or d[-1] is None:
        return "unknown"
    errors = [t for t in range(1, len(d)) if d[t] is not None and d[t - 1] is not None and d[t] > d[t - 1]]
    if not errors:
        return "no_error"
    if d[-1] <= min(known):
        return "recovered_ran_out"
    return "not_recovered_late" if errors[-1] >= len(d) - 2 else "not_recovered"


def band_summary(eps: List[Dict[str, Any]]) -> Dict[str, Any]:
    ok = [e for e in eps if "d" in e]
    s = summarize(eps)
    s["episode_error_share"] = round(s["episodes_with_error"] / len(ok), 4) if ok else None
    failed = [e for e in ok if not e["success"]]
    kinds = defaultdict(int)
    for e in failed:
        kinds[failure_kind(e)] += 1
    s["failed"] = len(failed)
    s["failed_split"] = {k: kinds.get(k, 0) for k in SPLIT}
    s["failed_split"]["not_recovered_total"] = kinds.get("not_recovered", 0) + kinds.get("not_recovered_late", 0)
    s["failed_share"] = {k: (round(v / len(failed), 4) if failed else None) for k, v in s["failed_split"].items()}
    s["failed_beyond_start"] = sum(1 for e in failed if e["d"] and e["d"][0] is not None and e["d"][-1] is not None
                                   and e["d"][-1] > e["d"][0])
    s["failed_beyond_start_share"] = round(s["failed_beyond_start"] / len(failed), 4) if failed else None
    ends = [e["d"][-1] for e in failed if e["d"] and e["d"][-1] is not None]
    closest = [min(x for x in e["d"] if x is not None) for e in failed if any(x is not None for x in e["d"])]
    s["fail_end_mean"] = round(statistics.mean(ends), 3) if ends else None
    s["fail_end_median"] = statistics.median(ends) if ends else None
    s["fail_closest_mean"] = round(statistics.mean(closest), 3) if closest else None
    return s


def merged_episodes(metrics: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """{RUN@STEP: every episode of its sets} from a turn_metrics output."""
    out: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for lab, eps in metrics["episodes"].items():
        out[lab.split("|")[0]] += eps
    return dict(out)


def run(metrics: Dict[str, Any], by: str, bands: Sequence[Tuple[str, int, int]], limit: int,
        groups: Dict[str, Sequence[str]], labels: Sequence[str]) -> Dict[str, Any]:
    merged = merged_episodes(metrics)
    key = (lambda e: int(e["d_star"])) if by == "dstar" else (lambda e: limit - int(e["d_star"]))  # noqa: E731

    def per_band(eps: List[Dict[str, Any]]) -> Dict[str, Any]:
        split = defaultdict(list)
        outside = 0
        for e in eps:
            if "d_star" not in e:  # a replay error: kept for summarize's count
                split["(replay error)"].append(e)
                continue
            b = band_of(key(e), bands)
            if b is None:
                outside += 1
            else:
                split[b].append(e)
        res = {b: band_summary(split[b]) for b, _, _ in bands}
        res["all"] = band_summary([e for b, _, _ in bands for e in split[b]])
        res["outside_bands"] = outside
        res["replay_errors"] = len(split.get("(replay error)", []))
        return res

    def pick(patterns: Sequence[str]) -> List[str]:
        found = sorted({lab for p in patterns for lab in merged if fnmatch.fnmatchcase(lab, p)})
        missing = [p for p in patterns if not any(fnmatch.fnmatchcase(lab, p) for lab in merged)]
        if missing:
            raise SystemExit(f"no label matches {missing}")
        return found

    out: Dict[str, Any] = {"by": by, "limit": limit if by == "margin" else None, "bands": [b[0] for b in bands],
                           "groups": {}, "labels": {}}
    for name, patterns in groups.items():
        labs = pick(patterns)
        out["groups"][name] = {"labels": labs, **per_band([e for lab in labs for e in merged[lab]])}
    for lab in pick(labels) if labels else []:
        out["labels"][lab] = per_band(merged[lab])
    return out


def table(res: Dict[str, Any]) -> str:
    title = "最少步数 d*" if res["by"] == "dstar" else f"余量 {res['limit']} − d*"
    cols = ["组 / 模型", title, "局数", "成功率", "有失误的局", "逐回合失误率", "找回率", "有失误的成功率",
            "无失误的成功率", "失败局", "失败：没失误", "失败：找回了、步数用完", "失败：没找回（其中最后两回合才失误）",
            "失败：终点比起点还远", "失败终点剩余（均值）", "失败局离目标最近（均值）"]
    pct = lambda x: "—" if x is None else f"{100 * x:.1f}%"  # noqa: E731
    num = lambda x: "—" if x is None else f"{x}"  # noqa: E731
    L = ["| " + " | ".join(cols) + " |", "|" + " --- |" * len(cols)]
    rows = [(f"{n}（{len(g['labels'])} 个）", g) for n, g in res["groups"].items()] + list(res["labels"].items())
    for name, per in rows:
        for b in list(res["bands"]) + ["all"]:
            s = per[b]
            sh = s["failed_share"]
            nr = (f"{pct(sh['not_recovered_total'])}（{pct(sh['not_recovered_late'])}）"
                  if s["failed"] else "—")
            L.append("| " + " | ".join([
                name, "合计" if b == "all" else b, str(s["replayed"]), pct(s["success_rate"]),
                pct(s["episode_error_share"]), pct(s["error_rate"]), pct(s["recovery_rate"]),
                pct(s["success_with_error"]), pct(s["success_without_error"]), str(s["failed"]),
                pct(sh["no_error"]), pct(sh["recovered_ran_out"]), nr, pct(s["failed_beyond_start_share"]),
                num(s["fail_end_mean"]), num(s["fail_closest_mean"])]) + " |")
    return "\n".join(L) + "\n"


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--metrics", required=True, help="a reward.turn_metrics output (FILE.json)")
    ap.add_argument("--by", choices=["dstar", "margin"], required=True)
    ap.add_argument("--bands", required=True, help="e.g. 1-3,4-5,6-8 or 0,1-2,3-4,5-")
    ap.add_argument("--limit", type=int, default=12, help="the action limit (margin = limit - d_star)")
    ap.add_argument("--group", action="append", default=[], help="NAME=PATTERN[,PATTERN ...]")
    ap.add_argument("--label", action="append", default=[], help="PATTERN (one row per matching label)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    groups = {}
    for g in a.group:
        name, _, pats = g.partition("=")
        groups[name] = [p for p in pats.split(",") if p]
    res = run(json.loads(Path(a.metrics).read_text()), a.by, parse_bands(a.bands), a.limit, groups, a.label)
    res["metrics"] = a.metrics
    Path(a.out).write_text(json.dumps(res, indent=1, ensure_ascii=False))
    md = table(res)
    Path(a.out).with_suffix(".md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
