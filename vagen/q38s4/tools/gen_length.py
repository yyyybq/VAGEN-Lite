"""q38s4 stage 5 (PREREG_v1 section 8): answer lengths and seconds per question of an EASI run, and each benchmark's
projected time, so a generation cap can be decided (and registered) before any accuracy is seen.  Reads no score:
from the samples files only each line's doc_id and token_counts[0].output_tokens; from the lmms-eval log only the
"Metric summary" line (generation seconds, generated tokens) and the last finished "Model Responding" progress bar
(its wall time); never results.json, done.json or easi_results.json.

    python -m vagen.q38s4.tools.gen_length --qa TR/qa --name c_nfp_500 [--base-qa q38/runs/q38-base-zs/qa]
        [--base-name q38base] [--out FILE.json]

A run's samples files are every <timestamp>_samples_<task>.jsonl with the newest timestamp, in any model folder
(lmms-eval names it after the model path's last two parts: models__Qwen3.8-27B for the base, evalprep__... or
ckpt_bf16__... for the C line).  One EASI job can be several lmms-eval tasks (blink: 14 sub-tasks, one file each) and
--limit applies to each task, so a limit-50 blink run answers 700 questions; they are pooled.

Per EASI job (qa_eval.EASI_8_JOBS): n (the questions answered), mean / median / 90th percentile / max output tokens,
the share at the task's cap (TASK_CAPS: the task yamls' max_new_tokens); the base's mean tokens on the same questions
(same task and doc_id; --limit takes each task's first docs, which need not be typical of the whole benchmark) and on
the whole benchmark, and their ratio; seconds per question: wall (the progress bar's time over its count: generation
plus each question's preparation, e.g. video decoding, 35-50% of a video benchmark's time) and generation only (the
Metric summary's elapsed over n); the warm-up (the bar's time at its first question: 45-65 s of start-up in every
run, the base's full runs too, so most of a 50-question run) and the steady rate after it ((wall - warm-up) / (n -
1)); the full benchmark's question count (the base run's lines, a count only), the projected hours (projected_from:
warm-up + steady rate x the other questions; without a first-question bar, mean wall seconds per question x full
count; generation-only when a log has no finished bar) and the base's own (its log's wall time), and whether the
projection is over the 3-hour line (PROJECT_LIMIT_H).  Loading and scoring (minutes) are in neither.  summary_tokens is the Metric summary's token total (a cross-check of the samples
files' sum).

A job whose scoring failed after generation (vsi_bench at --limit 50: the first 50 questions lack some question
types and its aggregation raises KeyError) has no samples file; when its log has a Metric summary and a finished
bar, its row is "log only" (source): n from the bar, the mean from the summary's tokens over n, no per-question
statistics and no same-question base mean.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

EASI_JOBS = ("vsi_bench", "mmsi_bench", "mindcube_tiny", "viewspatial", "site_image", "site_video", "blink",
             "3dsrbench", "embspatial")
# max_new_tokens in the EASI v0.2.2 lmms-eval task yamls (read 2026-11-24; reported, not enforced); viewspatial's yaml
# leaves it out, so the model class default (128) applies
TASK_CAPS = {"vsi_bench": 16, "mmsi_bench": 2048, "mindcube_tiny": 4096, "viewspatial": 128, "site_image": 1024,
             "site_video": 1024, "blink": 1024, "3dsrbench": 4096, "embspatial": 1024}
PROJECT_LIMIT_H = 3.0
BASE_QA = Path("/mnt/umm/users/yinbaiqiao/q38/runs/q38-base-zs/qa")
SUMMARY = re.compile(r"Total elapsed time: ([0-9.]+)s, Total gen tokens: ([0-9]+)")
BAR = re.compile(r"Model Responding: 100%\|[^|]*\| (\d+)/(\d+) \[([0-9:]+)<")
FIRST = re.compile(r"Model Responding:\s+\d+%\|[^|]*\| 1/(\d+) \[([0-9:]+)<")
SEP = "_samples_"
LOG_ONLY = "log only: no samples file (scoring failed after generation), mean from the Metric summary over the bar's n"


def stamp_of(p: Path) -> str:
    return p.name.split(SEP, 1)[0]


def task_of(p: Path) -> str:
    return p.name.split(SEP, 1)[1][:-len(".jsonl")]


def run_samples(folder: Path) -> List[Path]:
    """The samples files of a result folder's newest run (all files with the newest timestamp), in one model folder."""
    found = list(folder.glob(f"*/*{SEP}*.jsonl"))
    if not found:
        return []
    stamp = max(stamp_of(p) for p in found)
    files = sorted(p for p in found if stamp_of(p) == stamp)
    parents = sorted({str(p.parent) for p in files})
    if len(parents) > 1:
        raise SystemExit(f"{folder}: run {stamp} has samples files in {len(parents)} model folders: {parents}")
    return files


def records(path: Path) -> List[Tuple[Any, int]]:
    """(doc_id, output tokens) per answered line."""
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            counts = rec.get("token_counts") or [{}]
            out.append((rec.get("doc_id"), int(counts[0].get("output_tokens", 0))))
    return out


def hms(text: str) -> int:
    secs = 0
    for part in text.split(":"):
        secs = secs * 60 + int(part)
    return secs


def elapsed(folder: Path) -> Optional[Dict[str, Any]]:
    """The newest log that has a "Metric summary": its last summary {"seconds", "tokens"}, its last finished
    progress bar {"wall_seconds", "wall_n"} and that bar's time at its first question {"first_seconds"} (each None
    when the log has none)."""
    for log in sorted(folder.glob("eval_*.log"), key=lambda p: p.name, reverse=True):
        text = log.read_text(encoding="utf-8", errors="replace")
        hits = SUMMARY.findall(text)
        if hits:
            s, t = hits[-1]
            bars = [b for b in BAR.findall(text) if b[0] == b[1]]
            wall = {"wall_seconds": float(hms(bars[-1][2])), "wall_n": int(bars[-1][1])} if bars else \
                {"wall_seconds": None, "wall_n": None}
            firsts = [f for f in FIRST.findall(text) if wall["wall_n"] is not None and int(f[0]) == wall["wall_n"]]
            wall["first_seconds"] = float(hms(firsts[-1][1])) if firsts else None
            return {"seconds": float(s), "tokens": float(t), "log": str(log), **wall}
    return None


def pct(xs: List[int], q: float) -> float:
    s = sorted(xs)
    return float(s[min(len(s) - 1, int(q * len(s)))]) if s else 0.0


def mean(xs: List[int]) -> Optional[float]:
    return round(statistics.mean(xs), 1) if xs else None


def job_row(qa: Path, name: str, bench: str, base_qa: Path, base_name: str) -> Dict[str, Any]:
    folder = qa / bench / name
    row: Dict[str, Any] = {"benchmark": bench, "cap": TASK_CAPS.get(bench)}
    files = run_samples(folder)
    el = elapsed(folder)
    log_only = not files and el is not None and bool(el["wall_n"])
    if not (files or log_only) or el is None:
        return {**row, "missing": "no samples file" if not files else "no Metric summary in the logs"}
    cap = TASK_CAPS.get(bench)
    if files:
        recs = {task_of(f): records(f) for f in files}
        toks = [t for rs in recs.values() for _, t in rs]
        n = len(toks)
        row.update({"n": n, "tasks": len(files), "samples": [str(f) for f in files], "tokens_mean": mean(toks) or 0.0,
                    "tokens_median": statistics.median(toks) if toks else 0, "tokens_p90": pct(toks, 0.9),
                    "tokens_max": max(toks) if toks else 0,
                    "at_cap": round(sum(t >= cap for t in toks) / n, 3) if cap and n else None})
    else:
        recs, n = {}, el["wall_n"]
        row.update({"n": n, "tasks": None, "samples": [], "tokens_mean": round(el["tokens"] / n, 1),
                    "tokens_median": None, "tokens_p90": None, "tokens_max": None, "at_cap": None,
                    "source": LOG_ONLY})
    gen_spq = round(el["seconds"] / n, 2) if n else None
    wall_spq = round(el["wall_seconds"] / n, 2) if n and el["wall_seconds"] is not None else None
    warm = el["first_seconds"] if el["wall_seconds"] is not None and n and n > 1 else None
    steady = (el["wall_seconds"] - warm) / (n - 1) if warm is not None else None
    row.update({"summary_tokens": el["tokens"],
                "gen_seconds": el["seconds"], "gen_s_per_q": gen_spq,
                "tok_per_s": round(el["tokens"] / el["seconds"], 1) if el["seconds"] else None,
                "wall_seconds": el["wall_seconds"], "wall_n": el["wall_n"], "wall_s_per_q": wall_spq,
                "s_per_q": wall_spq if wall_spq is not None else gen_spq,
                "s_per_q_from": "wall" if wall_spq is not None else "generation only",
                "warmup_seconds": warm, "steady_s_per_q": round(steady, 2) if steady is not None else None})
    base = base_qa / bench / base_name
    bfiles, bel = run_samples(base), elapsed(base)
    if bfiles:
        brecs = {task_of(f): records(f) for f in bfiles}
        same, unmatched = [], []
        for task, rs in recs.items():
            ids = {d for d, _ in rs}
            if task in brecs:
                same += [t for d, t in brecs[task] if d in ids]
            else:
                unmatched.append(task)
        full = sum(len(rs) for rs in brecs.values())
        bsame = mean(same)
        if steady is not None:  # the first question's start-up once, then the steady rate
            proj, proj_from = (warm + steady * (full - 1)) / 3600, "warm-up + steady"
        elif row["s_per_q"] is not None:
            proj, proj_from = row["s_per_q"] * full / 3600, row["s_per_q_from"]
        else:
            proj, proj_from = None, None
        row.update({"full_n": full, "base_tokens_mean": mean([t for rs in brecs.values() for _, t in rs]),
                    "base_same_n": len(same) if recs else None, "base_same_tokens_mean": bsame,
                    "len_ratio": round(row["tokens_mean"] / bsame, 2) if bsame else None,
                    "projected_h": round(proj, 2) if proj is not None else None, "projected_from": proj_from})
        if unmatched:
            row["tasks_not_in_base"] = unmatched
        row["over_limit"] = row["projected_h"] is not None and row["projected_h"] > PROJECT_LIMIT_H
    if bel is not None:
        row["base_gen_h"] = round(bel["seconds"] / 3600, 2)
        if bel["wall_seconds"] is not None:
            row["base_h"] = round(bel["wall_seconds"] / 3600, 2)
    return row


COLUMNS = ("benchmark", "n", "mean tok", "base mean tok, same qs", "ratio", "p90", "max", "at cap", "s/q",
           "warm-up s", "steady s/q", "full n", "projected h", "base h", "> 3 h")


def table(rows: List[Dict[str, Any]]) -> str:
    lines = ["| " + " | ".join(COLUMNS) + " |", "|" + "---|" * len(COLUMNS)]
    for r in rows:
        if "missing" in r:
            lines.append(f"| {r['benchmark']} | {r['missing']} |" + " |" * (len(COLUMNS) - 2))
            continue
        spq = "" if r["s_per_q"] is None else f"{r['s_per_q']}" + ("" if r["s_per_q_from"] == "wall" else " (gen)")
        cells = [r["benchmark"] + (" (log only)" if r.get("source") == LOG_ONLY else ""), r["n"], r["tokens_mean"], r.get("base_same_tokens_mean", ""), r.get("len_ratio", ""),
                 r["tokens_p90"], r["tokens_max"],
                 "" if r["at_cap"] is None else f"{100 * r['at_cap']:.0f}% of {r['cap']}", spq,
                 r["warmup_seconds"], r["steady_s_per_q"], r.get("full_n", ""), r.get("projected_h", ""), r.get("base_h", r.get("base_gen_h", "")),
                 "yes" if r.get("over_limit") else "no"]
        lines.append("| " + " | ".join("" if c is None else str(c) for c in cells) + " |")
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qa", required=True, type=Path)
    ap.add_argument("--name", required=True)
    ap.add_argument("--base-qa", type=Path, default=BASE_QA)
    ap.add_argument("--base-name", default="q38base")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)
    rows = [job_row(a.qa, a.name, b, a.base_qa, a.base_name) for b in EASI_JOBS]
    if a.out:
        a.out.write_text(json.dumps({"qa": str(a.qa), "name": a.name, "limit_h": PROJECT_LIMIT_H, "rows": rows},
                                    indent=1) + "\n")
    print(table(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
