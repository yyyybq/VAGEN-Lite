"""Reward line, phase 3: the two preregistered verdicts of A's extension (docs/reward/PREREG_P3_A2.md section 2),
written before its results.

2.1 X0, V4 against S0 on three seed pairs (validation-best checkpoints): rw_g_250 - rw_s0_150, rw_s2g_150 -
    rw_s2s0_300, rw_s3g_300 - rw_s3s0_300; passes when all three differences are > 0 and their mean is >= +3.0
    (difference = 100 x (correct_a - correct_b) / 510, unrounded).
2.2 EmbSpatial on the two new pairs (rw_s2g_150 vs rw_s2s0_300, rw_s3g_300 vs rw_s3s0_300), paired per question as
    PREREG_P3_A 2.4 (reward.compare_qa.score_pair): holds when both pairs have the V4 model higher (item-mean
    difference > 0) and p < 0.05.

Answers are read with reward.compare_qa's copies of the downstream line's rules, from the diagnostics folders of
rw_p3a_diag (A: rw_g_250, rw_s0_150, rw_s2s0_300) and rw_p3a2_diag (the extension), whichever holds a model.

    python -m reward.verdict_p3a2 --x0 X0_QUESTIONS --diag A_DIAG --diag A2_DIAG --easi-root EASI_ROOT --out FILE.json
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from reward import compare_qa as cq

X0_PAIRS = (("seed 20261019", "rw_g_250", "rw_s0_150"), ("seed 20261021", "rw_s2g_150", "rw_s2s0_300"),
            ("seed 20261023", "rw_s3g_300", "rw_s3s0_300"))
EMB_PAIRS = X0_PAIRS[1:]
MARGIN, ALPHA = 3.0, 0.05


def find_answers(diags: List[Path], name: str) -> Path:
    hits = [d / "downstream_qa" / f"x0_{name}.jsonl" for d in diags if (d / "downstream_qa" / f"x0_{name}.jsonl").exists()]
    if len(hits) != 1:
        raise SystemExit(f"x0 answers of {name}: found {len(hits)} in {diags}")
    return hits[0]


def run(x0q: Path, diags: List[Path], easi_root: Path) -> Dict[str, Any]:
    qs = cq._jsonl(x0q)
    x0 = {}
    for lab, a, b in X0_PAIRS:
        pa = cq.downstream_correct(qs, find_answers(diags, a), "x0")
        pb = cq.downstream_correct(qs, find_answers(diags, b), "x0")
        x0[lab] = {"a": a, "b": b, **cq.pair(pa, pb)}
    d = [x0[lab]["diff"] for lab, _, _ in X0_PAIRS]
    mean = sum(d) / len(d)
    emb = {}
    for lab, a, b in EMB_PAIRS:
        emb[lab] = {"a": a, "b": b, **cq.score_pair(cq.bench_items(easi_root, "embspatial", a),
                                                     cq.bench_items(easi_root, "embspatial", b))}
    return {"x0": {"pairs": x0, "mean": mean, "all_positive": all(v > 0 for v in d),
                   "passes": all(v > 0 for v in d) and mean >= MARGIN},
            "embspatial": {"pairs": emb, "holds": all(v["diff_item_mean"] > 0 and v["p"] < ALPHA for v in emb.values())}}


def write_md(res: Dict[str, Any], path: Path) -> None:
    x, e = res["x0"], res["embspatial"]
    L = ["## 2.1 X0：V4 对 S0，三对种子", "", "| 种子 | 比较 | 差（个点） | 胜 / 负 | p |", "| --- | --- | --- | --- | --- |"]
    for lab, r in x["pairs"].items():
        L.append(f"| {lab} | {r['a']} 对 {r['b']} | {r['diff']:+.2f} | {r['wins']} / {r['losses']} | {r['p']:.3g} |")
    L += ["", f"- 平均 {x['mean']:+.2f}；三对都 > 0：{'是' if x['all_positive'] else '否'}；判定：{'过' if x['passes'] else '没过'}",
          "", "## 2.2 EmbSpatial：新的两对", "", "| 种子 | 比较 | 逐题平均之差（个点） | 胜 / 负 / 平 | p |", "| --- | --- | --- | --- | --- |"]
    for lab, r in e["pairs"].items():
        L.append(f"| {lab} | {r['a']} 对 {r['b']} | {r['diff_item_mean']:+.2f} | {r['wins']} / {r['losses']} / {r['ties']} | {r['p']:.3g} |")
    L += ["", f"- 判定：{'成立' if e['holds'] else '不成立'}（两对都要 V4 更高且 p < 0.05）"]
    path.write_text("\n".join(L) + "\n")


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--x0", required=True)
    ap.add_argument("--diag", action="append", required=True)
    ap.add_argument("--easi-root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = run(Path(a.x0), [Path(d) for d in a.diag], Path(a.easi_root))
    out = Path(a.out)
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n")
    write_md(res, out.with_suffix(".md"))
    print(out.with_suffix(".md").read_text())


if __name__ == "__main__":
    main()
