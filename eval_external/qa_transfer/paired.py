"""Paired comparisons of QA runs on the same items: model B minus model A on a benchmark's headline score, in
points, with a 95% interval from a paired bootstrap over items, and for benchmarks scored right / wrong per item
the exact McNemar p.

    python -m eval_external.qa_transfer.paired --stage step2 --pairs base-7B:ours-SFT ours-SFT:ours-RL-150 \
        --out eval_external/results/qa_paired_step2

The bootstrap resamples items within each question type (same draws for both models; SEED, N_BOOT draws) and
recomputes the headline the benchmark's way.  The headline rule is the first of HEADLINE_RULES[bench] (default
"items") that reproduces the run's results.json overall within 0.05 points: "items" averages the items, "types"
averages the question types, "vsi8" averages VSI-Bench's eight tasks (the three relative-direction difficulties
averaged into one), "vsti5" averages VLM-3R's five VSTI categories (collect.VSTI_GROUPS).  Per question type (VSTI:
per category) the same difference and interval are written too.  Writes OUT.md and OUT.json.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from eval_external.diagnostics.compare_ownenv import mcnemar_exact
from eval_external.qa_transfer.benchmarks import BENCHMARKS
from eval_external.qa_transfer.collect import VSTI_GROUPS, _item_score
from eval_external.qa_transfer.summary import COLUMNS, load_results

N_BOOT = 2000
SEED = 20261021
CHUNK = 200
VSI_MERGE = {"object_rel_direction": ("object_rel_direction_easy", "object_rel_direction_medium",
                                      "object_rel_direction_hard")}
HEADLINE_RULES = {"VSI-Bench": ("vsi8", "types", "items"), "VSTI-Bench": ("vsti5",),
                  "VSTI-Bench-prefix": ("vsti5",)}
BENCH_ORDER = list(dict.fromkeys(b for _, b, _ in COLUMNS))


def item_scores(result: dict) -> dict[str, tuple[str, float]]:
    """{item id: (question type, score in [0, 1])} of one run, read from the per-item files it lists."""
    out = {}
    if result["backend"] == "vlmevalkit":
        import pandas as pd

        f = [p for p in result["files"] if p.endswith(".xlsx")][0]
        for _, x in pd.read_excel(f).iterrows():
            rec = x.to_dict()
            s = _item_score(rec)
            if s is not None:
                out[str(rec["index"])] = (str(rec.get("question_type", rec.get("category", "all"))), s)
        return out
    task = BENCHMARKS[result["bench"]]["task"]
    res_file, sam_file = result["files"][:2]
    res = json.load(open(res_file))["results"][task]
    metric = next(k.split(",")[0] for k in res if not k.startswith("alias") and "stderr" not in k)
    for line in open(sam_file):
        r = json.loads(line)
        if task.startswith("sat"):
            out[str(r["doc_id"])] = ("all", float(r["exact_match"]))
            continue
        doc = r.get(metric)
        if isinstance(doc, str):
            doc = json.loads(doc) if doc.startswith("{") else None
        s = _item_score(doc) if isinstance(doc, dict) else None
        if s is not None:
            out[str(r["doc_id"])] = (doc["question_type"], s)
    return out


def headline(rule: str, sums: dict, ns: dict):
    """Headline as a fraction from per-type score sums (floats or arrays of bootstrap draws) and item counts."""
    if rule == "items":
        return sum(sums.values()) / sum(ns.values())
    means = {t: sums[t] / ns[t] for t in sums}
    if rule == "vsi8":
        for g, parts in VSI_MERGE.items():
            if all(p in means for p in parts):
                means[g] = sum(means.pop(p) for p in parts) / len(parts)
    if rule in ("types", "vsi8"):
        return sum(means.values()) / len(means)
    if rule == "vsti5":
        groups = [[t for t in types if t in sums] for types in VSTI_GROUPS.values()]
        vals = [sum(sums[t] for t in g) / sum(ns[t] for t in g) for g in groups if g]
        return sum(vals) / len(vals)
    raise ValueError(rule)


def pick_rule(bench: str, by_type: dict[str, list[float]], overall: float) -> tuple[str, bool]:
    sums, ns = {t: sum(v) for t, v in by_type.items()}, {t: len(v) for t, v in by_type.items()}
    rules = HEADLINE_RULES.get(bench, ("items", "types"))
    for rule in rules:
        if abs(100 * headline(rule, sums, ns) - overall) <= 0.05:
            return rule, True
    return rules[0], False


def compare(bench: str, ra: dict, rb: dict, sa: dict, sb: dict) -> dict:
    ids = sorted(set(sa) & set(sb))
    types = sorted({sa[i][0] for i in ids})
    a = {t: np.array([sa[i][1] for i in ids if sa[i][0] == t]) for t in types}
    b = {t: np.array([sb[i][1] for i in ids if sa[i][0] == t]) for t in types}
    rule_a, ok_a = pick_rule(bench, {t: list(v) for t, v in a.items()}, ra["overall"])
    rule_b, ok_b = pick_rule(bench, {t: list(v) for t, v in b.items()}, rb["overall"])
    rule = rule_a if ok_a else rule_b
    ns = {t: len(a[t]) for t in types}
    full_a = 100 * headline(rule, {t: float(a[t].sum()) for t in types}, ns)
    full_b = 100 * headline(rule, {t: float(b[t].sum()) for t in types}, ns)
    rng = np.random.default_rng(SEED)
    boot_a = {t: np.empty(N_BOOT) for t in types}
    boot_b = {t: np.empty(N_BOOT) for t in types}
    for start in range(0, N_BOOT, CHUNK):
        k = min(CHUNK, N_BOOT - start)
        for t in types:
            idx = rng.integers(0, ns[t], size=(k, ns[t]))
            boot_a[t][start:start + k] = a[t][idx].sum(axis=1)
            boot_b[t][start:start + k] = b[t][idx].sum(axis=1)
    diff = 100 * (headline(rule, boot_b, ns) - headline(rule, boot_a, ns))
    out = {"bench": bench, "model_a": ra["model"], "model_b": rb["model"], "n": len(ids), "rule": rule,
           "rule_reproduces_overall": {"a": ok_a, "b": ok_b},
           "a": round(full_a, 2), "b": round(full_b, 2), "diff": round(full_b - full_a, 2),
           "ci95": [round(float(np.percentile(diff, 2.5)), 2), round(float(np.percentile(diff, 97.5)), 2)]}
    binary = all(set(np.unique(v)) <= {0.0, 1.0} for v in list(a.values()) + list(b.values()))
    if binary:
        only_a = int(sum(((a[t] == 1) & (b[t] == 0)).sum() for t in types))
        only_b = int(sum(((a[t] == 0) & (b[t] == 1)).sum() for t in types))
        out.update({"only_a": only_a, "only_b": only_b, "mcnemar_p": mcnemar_exact(only_a, only_b)})
    units = ({g: [t for t in ts if t in a] for g, ts in VSTI_GROUPS.items()} if rule == "vsti5"
             else {t: [t] for t in types})
    per = {}
    for u, ts in units.items():
        if not ts:
            continue
        n_u = sum(ns[t] for t in ts)
        pa, pb = sum(float(a[t].sum()) for t in ts) / n_u, sum(float(b[t].sum()) for t in ts) / n_u
        d = 100 * (sum(boot_b[t] for t in ts) - sum(boot_a[t] for t in ts)) / n_u
        per[u] = {"n": n_u, "a": round(100 * pa, 2), "b": round(100 * pb, 2), "diff": round(100 * (pb - pa), 2),
                  "ci95": [round(float(np.percentile(d, 2.5)), 2), round(float(np.percentile(d, 97.5)), 2)]}
    out["per_type"] = per
    return out


def fmt_p(p: float | None) -> str:
    return "—" if p is None else ("<0.0001" if p < 1e-4 else f"{p:.4f}" if p < 0.01 else f"{p:.2f}")


def markdown(rows: list[dict]) -> str:
    lines = ["**逐题配对比较**（差 = B − A，单位是点；95% 区间来自题内配对 bootstrap，"
             f"{N_BOOT} 次、种子 {SEED}；只有 A 对 / 只有 B 对和 McNemar p 只对每题非对即错的 benchmark 给出）", "",
             "| A | B | benchmark | 题数 | A 的分 | B 的分 | 差 [95% 区间] | 只有 A 对 / 只有 B 对 | p |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        flag = "" if all(r["rule_reproduces_overall"].values()) else "（总分口径未复现）"
        pair = f"{r['only_a']} / {r['only_b']}" if "only_a" in r else "—"
        lines.append(f"| {r['model_a']} | {r['model_b']} | {r['bench']}{flag} | {r['n']} | {r['a']:.1f} | {r['b']:.1f} | "
                     f"{r['diff']:+.1f} [{r['ci95'][0]:+.1f}, {r['ci95'][1]:+.1f}] | {pair} | "
                     f"{fmt_p(r.get('mcnemar_p'))} |")
    for bench in ("VSI-Bench", "VSTI-Bench"):
        sub = [r for r in rows if r["bench"] == bench]
        if not sub:
            continue
        lines += ["", f"**{bench} 分题型**（VSTI 按 VLM-3R 的五类；差 = B − A，点）", "",
                  "| A | B | 题型 | 题数 | A | B | 差 [95% 区间] |", "|---|---|---|---|---|---|---|"]
        for r in sub:
            for u, s in r["per_type"].items():
                lines.append(f"| {r['model_a']} | {r['model_b']} | {u} | {s['n']} | {s['a']:.1f} | {s['b']:.1f} | "
                             f"{s['diff']:+.1f} [{s['ci95'][0]:+.1f}, {s['ci95'][1]:+.1f}] |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--pairs", nargs="+", required=True, help="A:B, B minus A is reported")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    res = load_results(args.stage)
    cache: dict = {}
    rows = []
    for pair in args.pairs:
        ma, mb = pair.split(":")
        for bench in BENCH_ORDER:
            if (ma, bench) not in res or (mb, bench) not in res:
                continue
            for m in (ma, mb):
                if (m, bench) not in cache:
                    cache[(m, bench)] = item_scores(res[(m, bench)])
            rows.append(compare(bench, res[(ma, bench)], res[(mb, bench)], cache[(ma, bench)], cache[(mb, bench)]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps(rows, indent=1, ensure_ascii=False) + "\n")
    md = markdown(rows)
    args.out.with_suffix(".md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
