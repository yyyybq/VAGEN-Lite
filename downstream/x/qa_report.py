#!/usr/bin/env python3
"""Tables of the question-answering batch (run_qa_batch.sh's results/ and pd/) and the Qwen3 gate, computed with the
rules the user fixed on 2026-10-30 (REFERENCE section 0):
    1. RS (or RD) is at least 10.0 points above the Qwen3 base on X0's 510 questions, same prompt, with the paired
       exact McNemar p < 0.05;
    2. no answer bias: in each of the three kinds the share of replies that are A lies within 40%-60% (the right
       answer is A for half of each kind), and at most 5% of the replies cannot be read;
    3. no loss of general ability on G0: accuracy over all of G0 not below the base's, or below it with McNemar
       p >= 0.05.
The gate passes when one of RS, RD meets all three.

    python downstream/x/qa_report.py --dir QA_DIR [--out QA_DIR/results/report.md]   (any Python; json only)
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

KINDS = ("left_right", "near_far", "front_back")
KIND_ZH = {"all": "全部", "left_right": "左右", "near_far": "远近", "front_back": "前后", "object": "含物体",
           "fixture_only": "只有设施", "existence": "存在", "identification": "辨认", "open_closed": "开合",
           "counting": "数量"}
MODELS = ("B", "S", "RS", "RD", "Q3B", "Q3S", "Q3RS", "Q3RD")


def load(path: Path):
    return json.loads(Path(path).read_text()) if Path(path).is_file() else None


def pct(x) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:.1f}%"


def gate(q0: dict, g0: dict) -> dict:
    """The three rules for Q3RS and Q3RD (q0: x0b_qa analyze against Q3B; g0: g0_qa analyze)."""
    out = {}
    for m in ("Q3RS", "Q3RD"):
        if not q0 or m not in q0 or "Q3B" not in q0:
            out[m] = {"available": False}
            continue
        a = q0[m]["all"]
        diff = a["accuracy"] - q0["Q3B"]["all"]["accuracy"]
        p = (a.get("vs_reference") or {}).get("p")
        r1 = diff >= 0.10 - 1e-12 and p is not None and p < 0.05
        shares = q0[m].get("a_share") or {}
        unparsed = a["unparsed_rate"]
        r2 = all(0.40 <= shares.get(k, float("nan")) <= 0.60 for k in KINDS) and unparsed <= 0.05
        r3 = None
        if g0 and m in g0 and "Q3B" in g0:
            ga, gb = g0[m]["all"], g0["Q3B"]["all"]
            gp = (ga.get("vs_reference") or {}).get("p")
            r3 = ga["accuracy"] >= gb["accuracy"] or (gp is not None and gp >= 0.05)
        out[m] = {"available": True, "diff_points": 100 * diff, "p": p, "rule1": bool(r1), "a_share": shares,
                  "unparsed_rate": unparsed, "rule2": bool(r2), "rule3": r3,
                  "passes": bool(r1 and r2 and r3) if r3 is not None else None}
    passes = [v.get("passes") for v in out.values() if v.get("available")]
    if not passes:
        out["gate"] = None  # Qwen3's answers are not in yet
    else:
        out["gate"] = True if any(p is True for p in passes) else (None if any(p is None for p in passes) else False)
    return out


def table(res: dict, groups, title: str, extra=None) -> list:
    if not res:
        return [f"（{title}：还没有结果）", ""]
    lines = [f"| 模型 | " + " | ".join(KIND_ZH.get(g, g) for g in groups) + " | 解析不出 | 和参照比（只对 / 参照只对，p） |",
             "| --- " * (len(groups) + 3) + "|"]
    for m in MODELS:
        if m not in res:
            continue
        r = res[m]
        cells = [f"{pct(r[g]['accuracy'])}（{pct(r[g]['ci'][0])}–{pct(r[g]['ci'][1])}）" if g in r else "—" for g in groups]
        vs = r["all"].get("vs_reference")
        ref = f"{vs['only_model']} / {vs['only_reference']}，p = {vs['p']:.3g}" if vs else "—"
        un = r["all"].get("unparsed_rate")
        if un is None and r["all"].get("n"):  # G0's rows count the unparsed replies
            un = r["all"].get("unparsed", 0) / r["all"]["n"]
        lines.append(f"| {m} | " + " | ".join(cells) + f" | {pct(un)} | {ref} |")
    return lines + [""]


def probe_table(res: dict) -> list:
    if not res:
        return ["（线性探针：还没有结果）", ""]
    lines = ["| 模型 | 层（1/4、1/2、3/4、末） | 关系：全部 | 左右 | 远近 | 前后 | 字母：全部 |", "| --- " * 7 + "|"]
    for m in MODELS:
        if m not in res:
            continue
        for layer, gs in res[m].items():
            lines.append(f"| {m} | {layer} | {pct(gs['all']['relation']['accuracy'])} | "
                         + " | ".join(pct(gs[k]['relation']['accuracy']) for k in KINDS)
                         + f" | {pct(gs['all']['letter']['accuracy'])} |")
    return lines + [""]


def pd_table(pd: dict) -> list:
    if not pd:
        return []
    lines = [f"组 {pd['group']}：相对距离 |A − B| / |B|（视觉塔 / 合并层 / 语言模型 / 词嵌入 / lm_head / 全部）", "",
             "| 差 | 视觉塔 | 合并层 | 语言模型 | 词嵌入 | lm_head | 全部 |", "| --- " * 7 + "|"]
    for lab, _, _ in pd["deltas"]:
        c = pd["coarse"]
        lines.append(f"| {lab} | " + " | ".join(f"{c[k]['relative'][lab]:.4f}" if k in c else "—"
                                                 for k in ("vision", "merger", "llm", "embed", "lm_head"))
                     + f" | {pd['all']['relative'][lab]:.4f} |")
    lines += ["", "| 余弦（全部参数） | 值 |", "| --- | --- |"]
    lines += [f"| {k} | {v:.3f} |" for k, v in pd["all"]["cosine"].items()]
    return lines + [""]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args(argv)
    r = args.dir / "results"
    q0b, q0q = load(r / "q0_vs_B.json"), load(r / "q0_vs_Q3B.json")
    g0 = load(r / "g0.json")
    lines = ["# 问答批次的表（自动生成）", "", "## X0（Qwen2.5 四个是 10-26 的答案；Qwen3 四个是 Q0）", ""]
    lines += table(q0b, ("all",) + KINDS, "X0 对 B")
    lines += ["和 Qwen3 底座比：", ""] + table(q0q, ("all",) + KINDS, "X0 对 Q3B")
    shares = {m: (q0b or {}).get(m, {}).get("a_share") for m in MODELS if q0b and m in q0b}
    lines += ["回答是 A 的比例（左右 / 远近 / 前后）：" + "；".join(
        f"{m} " + " / ".join(pct(s.get(k)) for k in KINDS) for m, s in shares.items() if s), ""]
    lines += ["## G0", ""] + table(g0, ("all", "existence", "identification", "open_closed", "counting"), "G0")
    if g0:
        lines += ["格式遵从率 / 回答 A 的比例 / 存在题回答 yes 的比例：" + "；".join(
            f"{m} {pct(v['all']['compliance'])} / {pct(v['all']['a_share'])} / {pct(v['all']['yes_share'])}"
            for m, v in g0.items()), ""]
    for v, name in (("x0b1", "X0b-1（InteriorGS 画面）"), ("x0b2", "X0b-2（主线带标签的问法，只认字母）"),
                    ("x0b2_text", "X0b-2（宽松读法：标签里写选项的文字也算）")):
        lines += [f"## {name}", ""] + table(load(r / f"{v}_vs_B.json"), ("all",) + KINDS, name)
        lines += ["和 Qwen3 底座比：", ""] + table(load(r / f"{v}_vs_Q3B.json"), ("all",) + KINDS, name)
    lines += ["## 线性探针（X0 的题，按场景分组 5 折）", ""] + probe_table(load(r / "probe.json"))
    lines += ["## 参数距离", ""]
    for g in ("qwen25", "qwen3"):
        lines += pd_table(load(args.dir / "pd" / f"{g}.json"))
    gt = gate(q0q, g0)
    lines += ["## Qwen3 闸门（REFERENCE 第 0 节的三条细则）", "", "```", json.dumps(gt, indent=1, ensure_ascii=False), "```"]
    text = "\n".join(lines) + "\n"
    out = args.out or (r / "report.md")
    out.write_text(text, encoding="utf-8")
    (r / "gate.json").write_text(json.dumps(gt, indent=1) + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
