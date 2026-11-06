"""Analysis 3 (exploratory): standard against goal-swapped tasks from the same start, per checkpoint and group, on
the train and validation scenes.  Reads a3/eval_pairs/<model>/results.jsonl; writes a3/a3.json and a3/a3.md.
A pair is a standard task and its swap (A and B exchanged); the test is McNemar's exact test on the discordant
pairs (two-sided)."""
import json
import math
import sys
from pathlib import Path

A = Path(sys.argv[1])  # .../analysis_20261106/a3
EV = A / "eval_pairs"
GROUPS = {"C0": ("c0q3_s1s0_150", "c0q3_s1d_300"), "C": ("m3cs1s0_050", "sft_m3c_e4"),
          "Cr": ("m3crs1s0_250", "m3crs1d_250"), "D": ("m3ds1s0_300", "m3ds1d_300"),
          "D_noP0": ("m3dns1s0_300", "m3dns1d_300")}
SPLITS = ("train", "val")


def mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def pairs(name: str, split: str) -> list:
    rows = [json.loads(l) for l in (EV / name / "results.jsonl").read_text().splitlines() if l.strip()]
    sel = json.loads((EV / "selection.json").read_text())["parts"]
    ok = {r["task_id"]: bool(r["success"]) for r in rows if "error" not in r}
    out = []
    for tid in sel[f"{split}_standard"]["task_ids"]:
        base = tid[: -len("_ps")]
        swap = base + "_ps_swap"
        if tid in ok and swap in ok:
            out.append((ok[tid], ok[swap]))
    return out


def stats(ps: list, test: bool = True) -> dict:
    """``test`` False (the group and all-checkpoint rows: the same pairs counted once per checkpoint, so not
    independent): no p value."""
    n = len(ps)
    s, w = sum(a for a, _ in ps), sum(b for _, b in ps)
    b = sum(1 for a, x in ps if a and not x)  # standard right, swap wrong
    c = sum(1 for a, x in ps if x and not a)
    return {"pairs": n, "standard": round(s / n, 4) if n else None, "swap": round(w / n, 4) if n else None,
            "swap_minus_standard_pp": round(100 * (w - s) / n, 1) if n else None,
            "only_standard": b, "only_swap": c, "both": sum(1 for a, x in ps if a and x),
            "p_mcnemar": round(mcnemar(b, c), 4) if test else None}


def main() -> None:
    out = {"models": {}, "groups": {}, "all": {}}
    for g, names in GROUPS.items():
        for n in names:
            out["models"][n] = {s: stats(pairs(n, s)) for s in SPLITS}
        out["groups"][g] = {s: stats([p for n in names for p in pairs(n, s)], test=False) for s in SPLITS}
    out["all"] = {s: stats([p for ns in GROUPS.values() for n in ns for p in pairs(n, s)], test=False) for s in SPLITS}
    (A / "a3.json").write_text(json.dumps(out, indent=1))
    pct = lambda v: "—" if v is None else f"{100 * v:.1f}%"  # noqa: E731
    head = ("| 组 / 检查点 | 场景 | 配对数 | 标准题成功率 | 对调题成功率 | 对调 − 标准（百分点） | 只标准成功 | 只对调成功 | McNemar p |\n"
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    lines = ["# 分析 3：目标对调（探索性）", "",
             "每对题是同一个起点上的标准题和它的对调题（A、B 角色互换，成功区域重算）。只有投影关系一族（遮挡族对调后都不合格，见 "
             "pairs_run/out/summary.json）。检验：配对的 McNemar 精确检验（双侧），只看两题结果不同的配对；组和全部的合计行把"
             "同一批题按检查点重复计入，不独立，不给 p。", "", head]
    for g, names in GROUPS.items():
        for s in SPLITS:
            r = out["groups"][g][s]
            lines.append(f"| **{g}**（两个检查点合计） | {s} | {r['pairs']} | {pct(r['standard'])} | {pct(r['swap'])} | "
                         f"{r['swap_minus_standard_pp']} | {r['only_standard']} | {r['only_swap']} | {'—' if r['p_mcnemar'] is None else r['p_mcnemar']} |")
        for n in names:
            for s in SPLITS:
                r = out["models"][n][s]
                lines.append(f"| {n} | {s} | {r['pairs']} | {pct(r['standard'])} | {pct(r['swap'])} | "
                             f"{r['swap_minus_standard_pp']} | {r['only_standard']} | {r['only_swap']} | {'—' if r['p_mcnemar'] is None else r['p_mcnemar']} |")
    for s in SPLITS:
        r = out["all"][s]
        lines.append(f"| **全部 10 个检查点** | {s} | {r['pairs']} | {pct(r['standard'])} | {pct(r['swap'])} | "
                     f"{r['swap_minus_standard_pp']} | {r['only_standard']} | {r['only_swap']} | {'—' if r['p_mcnemar'] is None else r['p_mcnemar']} |")
    # the pairs are the independent units: per pair the mean over the 10 checkpoints of (swap - standard), then the
    # sign test over the pairs (two-sided, ties left out)
    names = [n for ns in GROUPS.values() for n in ns]
    lines += ["", "按题配对的符号检验（每对题先对 10 个检查点取平均的“对调 − 标准”，再看 25 对里正负各几对；双侧，平的不算）：", "",
              "| 场景 | 配对数 | 对调更好的对 | 标准更好的对 | 打平 | 每对平均差（百分点） | 符号检验 p |", "| --- | --- | --- | --- | --- | --- | --- |"]
    out["pair_sign_test"] = {}
    for s in SPLITS:
        per = [pairs(n, s) for n in names]
        diffs = [sum(int(p[i][1]) - int(p[i][0]) for p in per) / len(per) for i in range(len(per[0]))]
        pos, neg = sum(d > 0 for d in diffs), sum(d < 0 for d in diffs)
        p = mcnemar(neg, pos)  # the same exact binomial test on the non-tied pairs
        out["pair_sign_test"][s] = {"pairs": len(diffs), "swap_better": pos, "standard_better": neg,
                                    "ties": len(diffs) - pos - neg, "mean_pp": round(100 * sum(diffs) / len(diffs), 1),
                                    "p_sign": round(p, 4)}
        r = out["pair_sign_test"][s]
        lines.append(f"| {s} | {r['pairs']} | {pos} | {neg} | {r['ties']} | {r['mean_pp']} | {r['p_sign']} |")
    (A / "a3.json").write_text(json.dumps(out, indent=1))
    (A / "a3.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
