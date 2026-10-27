#!/usr/bin/env python3
"""X4 (the user's decision 6 of 2026-10-26): how much of the main-line ability the E1 final models keep.

Each of the 8 models (the VLM of the four E1 final checkpoints, exported to HF, and the four original starts) was
evaluated once by the main line's own evaluation (sco_interiorgs_v10_train.sh eval, prompt format no_think, greedy)
on the v2 validation set (413 tasks) and test set (292 tasks); the results are X4/eval_{val,test}_v2/<name>/
results.jsonl (one row per task: task_id, task_type, success, ...).  This writes X4/results/results.json and
results.md: per model and set the success rate (Wilson 95% interval) and per task type; per start the change from
the original start to its E1 model, task by task (McNemar exact test; a bootstrap 95% interval of the difference over
tasks, 2,000 resamples, seed 0).  "Kept" when the drop is not significant (p >= 0.05).

    python downstream/x/x4_analyze.py --x4 downstream_runs/x/X4
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from x0_qa import mcnemar_exact, wilson  # noqa: E402

STARTS = ("B", "S", "RS", "RD")
SETS = {"val": 413, "test": 292}


def read(path: Path) -> dict:
    """{task_id: row} of one model's results (a later row of a task wins)."""
    out = {}
    for line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            r = json.loads(line)
        except ValueError:
            continue
        out[r["task_id"]] = r
    return out


def rate(rows) -> dict:
    k, n = sum(bool(r["success"]) for r in rows), len(rows)
    lo, hi = wilson(k, n) if n else (None, None)
    return {"k": k, "n": n, "rate": round(k / n, 4) if n else None, "ci95": [lo, hi]}


def by_type(rows: dict) -> dict:
    types = sorted({r.get("task_type", "?") for r in rows.values()})
    return {t: rate([r for r in rows.values() if r.get("task_type", "?") == t]) for t in types}


def paired(after: dict, before: dict, resamples: int = 2000, seed: int = 0) -> dict:
    """E1 model (after) against its original start (before), on the tasks both have."""
    keys = sorted(set(after) & set(before))
    a = [int(bool(after[k]["success"])) for k in keys]
    b = [int(bool(before[k]["success"])) for k in keys]
    gain = sum(x and not y for x, y in zip(a, b))
    loss = sum(y and not x for x, y in zip(a, b))
    diffs = [x - y for x, y in zip(a, b)]
    rng = random.Random(seed)
    boot = sorted(sum(rng.choice(diffs) for _ in diffs) / len(diffs) for _ in range(resamples)) if diffs else []
    ci = [round(boot[int(0.025 * resamples)], 4), round(boot[int(0.975 * resamples) - 1], 4)] if boot else [None, None]
    p = mcnemar_exact(gain, loss)
    diff = round(sum(diffs) / len(diffs), 4) if diffs else None
    return {"tasks": len(keys), "diff": diff, "ci95": ci, "only_e1": gain, "only_start": loss, "p": round(p, 6),
            "kept": not (diff is not None and diff < 0 and p < 0.05)}


def analyze(x4: Path) -> dict:
    res = {"models": {}, "retention": {}}
    rows = {}
    for s, n in SETS.items():
        for d in sorted((x4 / f"eval_{s}_v2").iterdir()) if (x4 / f"eval_{s}_v2").is_dir() else []:
            f = d / "results.jsonl"
            if d.is_dir() and f.is_file():
                rows[(s, d.name)] = read(f)
                done = d / "done.json"
                res["models"].setdefault(d.name, {})[s] = dict(rate(rows[(s, d.name)].values()), expected=n,
                                                               finished=done.is_file(),
                                                               by_type=by_type(rows[(s, d.name)]))
    for st in STARTS:
        for s in SETS:
            a, b = rows.get((s, f"E1-{st}-200k")), rows.get((s, f"orig-{st}"))
            if a is not None and b is not None:
                res["retention"].setdefault(st, {})[s] = paired(a, b)
    out = x4 / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "results.md").write_text(markdown(res), encoding="utf-8")
    return res


def _pct(r) -> str:
    if not r or not r.get("n"):
        return "—"
    lo, hi = r["ci95"]
    return f"{100 * r['rate']:.1f}%（{100 * lo:.1f}–{100 * hi:.1f}）"


def markdown(res: dict) -> str:
    lines = ["# X4 能力保留（自动生成）", "", "## 成功率（括号里是 95% 区间）", "",
             "| 模型 | 验证集（413） | 评测集（292） |", "| --- | ---: | ---: |"]
    for st in STARTS:
        for name in (f"orig-{st}", f"E1-{st}-200k"):
            m = res["models"].get(name, {})
            lines.append(f"| {name} | {_pct(m.get('val'))} | {_pct(m.get('test'))} |")
    lines += ["", "## E1 模型和自己的原始起点逐题比（E1 − 起点，百分点；括号里是 95% 区间）", "",
              "| 起点 | 题集 | 题数 | 差 | 只有 E1 对 / 只有起点对 | McNemar p | 保留 |",
              "| --- | --- | ---: | ---: | --- | ---: | --- |"]
    for st, sets in res["retention"].items():
        for s, r in sets.items():
            lo, hi = r["ci95"]
            lines.append(f"| {st} | {s} | {r['tasks']} | {100 * r['diff']:+.1f}（{100 * lo:+.1f}–{100 * hi:+.1f}） | "
                         f"{r['only_e1']} / {r['only_start']} | {r['p']:.3g} | {'是' if r['kept'] else '否'} |")
    lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--x4", type=Path, required=True)
    res = analyze(ap.parse_args(argv).x4)
    print(json.dumps({st: {s: (r["diff"], r["p"]) for s, r in v.items()} for st, v in res["retention"].items()}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
