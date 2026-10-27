#!/usr/bin/env python3
"""T5 (the user's decision of 2026-10-27): NavigateKitchen-only training, its 100-episode evaluation.

Reads, per start, the in-job evaluation of the final checkpoint (EXPERIMENTS/T5-<start>-nav-s1/evals/steps_<last>/
out/episodes/episodes.jsonl, plan downstream/x/t5_plan.json: seeds 1000-1099 with X3's read-only recorder) and
writes OUT/results.json and results.md, per start:
  success          RoboCasa's own check: base within 0.20 m of the instructed fixture's placement point and heading
                   cosine >= 0.98
  near_target      the base within NEAR_M (0.5 m) of the instructed fixture's placement point, any heading
  at_other         the base at another fixture (its placement point nearest and within NEAR_M; stacked fixtures that
                   share the instructed one's spot count as the instructed one)
with Wilson 95% intervals, and per start against B, episode by episode, McNemar exact tests on success and on
near_target.

    python downstream/x/t5_analyze.py --experiments downstream_runs/experiments --out downstream_runs/t/T5/results
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import x3_targets as x3  # noqa: E402
from x0_qa import mcnemar_exact, wilson  # noqa: E402

STARTS = ("B", "S", "RS", "RD")


def exp_id(start: str) -> str:
    return f"T5-{start}-nav-s1"


def final_eval(exp: Path):
    """episodes.jsonl of the evaluation of the latest step, or None."""
    evals = sorted((p for p in (exp / "evals").glob("steps_*") if p.name[6:].isdigit()), key=lambda p: int(p.name[6:]))
    path = evals[-1] / "out/episodes/episodes.jsonl" if evals else None
    return path if path and path.is_file() else None


def episode_table(rows: dict) -> dict:
    """{seed: {success, near_target, correct, at_other}} from the NavigateKitchen rows that carry both records."""
    out = {}
    for (task, seed), r in rows.items():
        es = r.get("end_state") or {}
        if task != "NavigateKitchen" or "start" not in es or "end" not in es or "error" in es["end"]:
            continue
        d = x3.decide(task, es["start"], es["end"])
        out[seed] = {"success": bool(r["success"]), "correct": d["correct"],
                     "near_target": d["target_dist"] is not None and d["target_dist"] <= x3.NEAR_M,
                     "at_other": d["chosen"] is not None and not d["correct"]}
    return out


def rate(k: int, n: int) -> dict:
    lo, hi = wilson(k, n) if n else (None, None)
    return {"k": k, "n": n, "rate": round(k / n, 4) if n else None, "ci95": [lo, hi]}


def mcnemar(a: dict, b: dict, key: str) -> dict:
    seeds = sorted(set(a) & set(b))
    only_a = sum(a[s][key] and not b[s][key] for s in seeds)
    only_b = sum(b[s][key] and not a[s][key] for s in seeds)
    return {"pairs": len(seeds), "only_this": only_a, "only_B": only_b, "p": round(mcnemar_exact(only_a, only_b), 4)}


def analyze(experiments: Path, out: Path) -> dict:
    tables, res = {}, {"starts": {}, "vs_B": {}, "missing": []}
    for s in STARTS:
        path = final_eval(experiments / exp_id(s))
        if path is None:
            res["missing"].append(s)
            continue
        tables[s] = episode_table(x3.read_rows(path))
        t = tables[s]
        res["starts"][s] = {"source": str(path), "episodes": len(t),
                            **{k: rate(sum(v[k] for v in t.values()), len(t))
                               for k in ("success", "near_target", "correct", "at_other")}}
    for s in tables:
        if s != "B" and "B" in tables:
            res["vs_B"][s] = {k: mcnemar(tables[s], tables["B"], k) for k in ("success", "near_target")}
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "results.md").write_text(markdown(res), encoding="utf-8")
    return res


def _pct(r) -> str:
    if not r or not r["n"]:
        return "—"
    lo, hi = r["ci95"]
    return f"{100 * r['rate']:.1f}%（{r['k']}/{r['n']}，{100 * lo:.1f}–{100 * hi:.1f}）"


def markdown(res: dict) -> str:
    lines = ["# T5 NavigateKitchen 单任务（自动生成）", "",
             "成功 = RoboCasa 的判定（离指令设施的放置点 ≤ 0.20 米且朝向余弦 ≥ 0.98）；到达附近 = 离指令设施的放置点 ≤ 0.5 米，"
             "不管朝向；停在别的设施 = 离另一个设施的放置点最近且 ≤ 0.5 米。括号里是回合数和 95% 区间。", "",
             "| 起点 | 回合 | 成功 | 到达指令设施附近 | 停在别的设施 | 和 B 比 p（成功 / 到达附近） |",
             "| --- | ---: | ---: | ---: | ---: | --- |"]
    for s, st in res["starts"].items():
        v = res["vs_B"].get(s)
        p = f"{v['success']['p']} / {v['near_target']['p']}" if v else "—"
        lines.append(f"| {s} | {st['episodes']} | {_pct(st['success'])} | {_pct(st['near_target'])} | "
                     f"{_pct(st['at_other'])} | {p} |")
    if res["missing"]:
        lines += ["", f"还没有评测结果：{'、'.join(res['missing'])}"]
    lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--experiments", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    res = analyze(args.experiments, args.out)
    print(json.dumps({s: v["success"]["rate"] for s, v in res["starts"].items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
