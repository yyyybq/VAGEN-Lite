"""q38s4 stage 5 (PREREG_v1 section 8): answer lengths and seconds per question of an EASI run, and each benchmark's
projected time, so a generation cap can be decided (and registered) before any accuracy is seen.  Reads no score:
from the samples files only each line's token_counts[0].output_tokens; from the lmms-eval log only the "Metric
summary" line (total elapsed generation time, total generated tokens); never results.json, done.json or
easi_results.json.

    python -m vagen.q38s4.tools.gen_length --qa TR/qa --name c_nfp_500 [--base-qa q38/runs/q38-base-zs/qa]
        [--base-name q38base] [--out FILE.json]

Per EASI job (qa_eval.EASI_8_JOBS): n (the questions answered: lines of the newest samples file), mean / median /
90th percentile / max output tokens, the share at the task's cap (TASK_CAPS: the task yamls' max_new_tokens), seconds
per question (elapsed / n) and tokens per second; the full benchmark's question count (lines of the base's newest
samples file, a count only), the projected generation hours (seconds per question x full count) and the base's own
(its log's elapsed), and whether the projection is over the 3-hour line (PROJECT_LIMIT_H).  The elapsed time is the
model's generation only (lmms-eval's gen_metrics), without loading and scoring (minutes).
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

EASI_JOBS = ("vsi_bench", "mmsi_bench", "mindcube_tiny", "viewspatial", "site_image", "site_video", "blink",
             "3dsrbench", "embspatial")
# max_new_tokens in the EASI v0.2.2 lmms-eval task yamls (read 2026-11-24; reported, not enforced)
TASK_CAPS = {"vsi_bench": 16, "mmsi_bench": 2048, "mindcube_tiny": 4096, "viewspatial": 128, "site_image": 1024,
             "site_video": 1024, "blink": 1024, "3dsrbench": 4096, "embspatial": 1024}
PROJECT_LIMIT_H = 3.0
BASE_QA = Path("/mnt/umm/users/yinbaiqiao/q38/runs/q38-base-zs/qa")
SUMMARY = re.compile(r"Total elapsed time: ([0-9.]+)s, Total gen tokens: ([0-9]+)")


def newest_samples(folder: Path) -> Optional[Path]:
    """The newest samples file of a result folder (lmms-eval names them <timestamp>_samples_<task>.jsonl)."""
    found = sorted(folder.glob("models__*/*_samples_*.jsonl"), key=lambda p: p.name, reverse=True)
    return found[0] if found else None


def output_tokens(path: Path) -> List[int]:
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            counts = json.loads(line).get("token_counts") or [{}]
            out.append(int(counts[0].get("output_tokens", 0)))
    return out


def count_lines(path: Path) -> int:
    with open(path, "rb") as f:
        return sum(1 for line in f if line.strip())


def elapsed(folder: Path) -> Optional[Dict[str, float]]:
    """The last "Metric summary" of the newest log that has one: {"seconds", "tokens"}."""
    for log in sorted(folder.glob("eval_*.log"), key=lambda p: p.name, reverse=True):
        hits = SUMMARY.findall(log.read_text(encoding="utf-8", errors="replace"))
        if hits:
            s, t = hits[-1]
            return {"seconds": float(s), "tokens": float(t), "log": str(log)}
    return None


def pct(xs: List[int], q: float) -> float:
    s = sorted(xs)
    return float(s[min(len(s) - 1, int(q * len(s)))]) if s else 0.0


def job_row(qa: Path, name: str, bench: str, base_qa: Path, base_name: str) -> Dict[str, Any]:
    folder = qa / bench / name
    row: Dict[str, Any] = {"benchmark": bench, "cap": TASK_CAPS.get(bench)}
    sam = newest_samples(folder)
    el = elapsed(folder)
    if sam is None or el is None:
        return {**row, "missing": "no samples file" if sam is None else "no Metric summary in the logs"}
    toks = output_tokens(sam)
    n = len(toks)
    cap = TASK_CAPS.get(bench)
    row.update({"n": n, "samples": str(sam), "tokens_mean": round(statistics.mean(toks), 1) if toks else 0.0,
                "tokens_median": statistics.median(toks) if toks else 0, "tokens_p90": pct(toks, 0.9),
                "tokens_max": max(toks) if toks else 0,
                "at_cap": round(sum(t >= cap for t in toks) / n, 3) if cap and n else None,
                "gen_seconds": el["seconds"], "s_per_q": round(el["seconds"] / n, 2) if n else None,
                "tok_per_s": round(el["tokens"] / el["seconds"], 1) if el["seconds"] else None})
    base = base_qa / bench / base_name
    bsam, bel = newest_samples(base), elapsed(base)
    if bsam is not None:
        full = count_lines(bsam)
        btoks = output_tokens(bsam)
        row.update({"full_n": full, "base_tokens_mean": round(statistics.mean(btoks), 1) if btoks else None,
                    "projected_h": round(row["s_per_q"] * full / 3600, 2) if row["s_per_q"] is not None else None})
        row["over_limit"] = row["projected_h"] is not None and row["projected_h"] > PROJECT_LIMIT_H
    if bel is not None:
        row["base_h"] = round(bel["seconds"] / 3600, 2)
    return row


def table(rows: List[Dict[str, Any]]) -> str:
    head = ("| benchmark | n | mean tok | p90 | max | at cap | s/q | tok/s | full n | projected h | base h | "
            "base mean tok | > 3 h |")
    lines = [head, "|" + "---|" * 13]
    for r in rows:
        if "missing" in r:
            lines.append(f"| {r['benchmark']} | {r['missing']} |" + " |" * 11)
            continue
        cells = [r["benchmark"], r["n"], r["tokens_mean"], r["tokens_p90"], r["tokens_max"],
                 "" if r["at_cap"] is None else f"{100 * r['at_cap']:.0f}% of {r['cap']}", r["s_per_q"],
                 r["tok_per_s"], r.get("full_n", ""), r.get("projected_h", ""), r.get("base_h", ""),
                 r.get("base_tokens_mean", ""), "yes" if r.get("over_limit") else "no"]
        lines.append("| " + " | ".join(str(c) for c in cells) + " |")
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
