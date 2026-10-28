"""P1_STATS.md from a built bank (OUT/p1_*.jsonl, OUT/stats.json).

    python -m data_gen.transition_qa.report --out OUT --md docs/transition_qa/P1_STATS.md
"""

from __future__ import annotations

import argparse
import collections
import json
import random
from pathlib import Path
from typing import Any, Dict, List

from . import generate as GEN

FILES = ("p1_train.jsonl", "p1_test.jsonl", "p1_probe.jsonl")
CLASS_ZH = {"state": "状态题 (a)", "forward": "前向题 (b)", "inverse": "逆向题 (c)"}


def _table(header: List[str], rows: List[List[Any]]) -> str:
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _pct(a: float, b: float) -> str:
    return f"{100.0 * a / b:.1f}%" if b else "–"


def text_only_section(out: Path, data: Dict[str, List[Dict[str, Any]]]) -> List[str]:
    """Section 7: how much the text alone answers, per question type, on the test and probe files: chance, the best
    rule learned on the training file (shortcuts.py) and the original 7B given only the stem and the options
    (OUT/blind/<test|probe>/<name>.json from ``qa_probe run --no-images``; user 2026-10-20, a fixed column)."""
    from . import shortcuts as SC

    best: Dict[Any, Dict[str, Any]] = {}
    for r in SC.report(out):
        key = (r["file"], r["qtype"])
        if key not in best or r["accuracy"] > best[key]["accuracy"]:
            best[key] = r
    lines = ["## 7. 只看文字能答对多少", "",
             "同一道题不给图、只给题干和选项时的准确率，和随机猜比；明显高于随机猜，说明题目文字泄露了答案。"
             "\"学出来的规则\"：只用训练集文字学到的最强一条（`shortcuts.py`，例如同一组选项在训练集里最常对的那个）；"
             "\"原版 7B 不看图\"：原版 Qwen2.5-VL-7B 只拿到题干和选项（`qa_probe run --no-images`，题前一句"
             "\"看不到图也请选最可能的答案\"，不采样），读不出的算错。", ""]
    for f, split in (("p1_test.jsonl", "test"), ("p1_probe.jsonl", "probe")):
        rs = data.get(f) or []
        if not rs:
            continue
        blind = {p.stem: json.loads(p.read_text()) for p in sorted((out / "blind" / split).glob("*.json"))}
        header = ["题类", "题数", "随机猜", "学出来的规则（哪条）"] + [f"{n} 不看图" for n in blind] + \
                 [f"{n} 读不出" for n in blind]
        rows = []
        for q in sorted({r["qtype"] for r in rs}):
            sub = [r for r in rs if r["qtype"] == q]
            chance = 0.5 if sub[0]["answer"] in ("Yes", "No") else 1.0 / len(sub[0]["options"])
            b = best.get((f, q))
            rows.append([q, len(sub), f"{100 * chance:.1f}%", f"{100 * b['accuracy']:.1f}%（{b['rule']}）" if b else "–"]
                        + [_fmt_acc(((res.get("qtype") or {}).get(q) or {}).get("accuracy")) for res in blind.values()]
                        + [_fmt_acc(((res.get("qtype") or {}).get(q) or {}).get("unreadable")) for res in blind.values()])
        chance_all = sum(0.5 if r["answer"] in ("Yes", "No") else 1.0 / len(r["options"]) for r in rs) / len(rs)
        rows.append(["全部", len(rs), f"{100 * chance_all:.1f}%", ""]
                    + [_fmt_acc(res["all"]["accuracy"]) for res in blind.values()]
                    + [_fmt_acc(res["all"]["unreadable"]) for res in blind.values()])
        lines += [f"**{f}**", "", _table(header, rows), ""]
    return lines


def _fmt_acc(x: Any) -> str:
    return "–" if x is None else f"{100 * float(x):.1f}%"


def write(out: Path, md: Path) -> Dict[str, Any]:
    stats = json.loads((out / "stats.json").read_text())
    data = {f: GEN.read_jsonl(out / f) for f in FILES}
    lines: List[str] = ["# P1 转移题：统计", "",
                        f"由 `python -m data_gen.transition_qa.report` 从 `{out}` 生成。"
                        "题库怎么来的、各题类的答案用哪个函数算，见 `PROGRESS.md` 第 1.3 节和第 2 节。", ""]
    # ------------------------------------------------------------------ sizes and classes
    lines += ["## 1. 题数和三类题的比例", ""]
    rows = []
    for f, rs in data.items():
        c = collections.Counter(r["qclass"] for r in rs)
        rows.append([f, len(rs)] + [f"{c[k]}（{_pct(c[k], len(rs))}）" for k in ("state", "forward", "inverse")])
    cfg = stats.get("config") or {}
    lines += [_table(["文件", "题数", "状态题 (a)", "前向题 (b)", "逆向题 (c)"], rows), "",
              f"三类题的规则（用户 2026-10-19）：状态题全部保留（答案平衡之后）；前向题最多是状态题的 "
              f"{cfg.get('forward_max_vs_state')} 倍；逆向题最多占题库的 {100 * float(cfg.get('inverse_max_share', 0)):.0f}%；"
              f"\"动作后任务要求会满足吗\"（goal_after）最多占前向题的一半；单个场景最多 {100 * float(cfg.get('scene_cap', 0)):.0f}%。"
              "超出的按（题类、答案、是否变化）分层抽。", ""]
    for split in ("train", "val"):
        cls = (stats.get(split) or {}).get("classes") or {}
        if cls:
            lines.append(f"- {split} 场景：" + "；".join(f"{CLASS_ZH[k]}可用 {v['available']}、留下 {v['kept']}"
                                                        for k, v in sorted(cls.items()) if k in CLASS_ZH))
            caps = cls.get("caps") or {}
            if caps:
                lines.append(f"  - 上限：前向题 {caps.get('forward')} 道，逆向题 {caps.get('inverse')} 道")
            capped = cls.get("goal_cap_dropped") or {}
            if capped:
                lines.append("  - goal 上限去掉：" + "，".join(f"{CLASS_ZH[k]} {n} 道" for k, n in sorted(capped.items())))
    if stats.get("probe_excluded"):
        lines.append(f"- 测试集去掉了探针集的 {stats['probe_excluded']} 道题（两份不重叠）"
                     + (f"，以及人工核对从探针去掉的 {stats['probe_dropped_excluded']} 道" if stats.get("probe_dropped_excluded") else "")
                     + "。")
    for name, h in (stats.get("held") or {}).items():
        parts = h.get("by_family_qtype") or h["by_qtype"]
        lines.append(f"- {name} 暂扣 {h['held']} 道（" + "、".join(f"{q} {n}" for q, n in sorted(parts.items()))
                     + f"），在 {name.replace('.jsonl', '_held.jsonl')}：答案来自遮挡的框判据（遮挡题和遮挡任务的 goal 题），"
                     "用户核对探针时发现这个判据的标签大多不对（2026-10-23、10-27）。")
    probe_stats = out / "probe_stats.json"
    if probe_stats.exists():
        ps = json.loads(probe_stats.read_text())
        over = ps["config"].get("per_qtype_overrides") or {}
        by_q = ps["by_qtype"]
        short = {q: v["short"] for q, v in by_q.items() if v["short"] and over.get(q) != "all"}
        capped = {q: v["short"] for q, v in by_q.items() if v["short"] and over.get(q) == "all"}  # the unchanged cap
        in_file = collections.Counter(r["qtype"] for r in data.get("p1_probe.jsonl", []))

        def checked(q: str, n: int) -> str:  # the user's checks drop questions after the picking (2026-10-23, 10-28)
            return f"（人工核对后留 {in_file[q]} 道）" if in_file and in_file[q] != n else ""

        def whole(q: str) -> str:
            n = by_q[q]["picked"]
            return f"全部可用的 {n} 道" + checked(q, n)

        sizes = (f"每个题类 {ps['config']['per_qtype']} 道，其中 "
                 + "、".join(f"{q} " + (whole(q) if v == "all" else f"{v} 道" + checked(q, v))
                            for q, v in sorted(over.items())) + "（预注册要的题类）"
                 if over else f"每个题类 {ps['config']['per_qtype']} 道")
        lines.append(f"- 探针集：{ps['questions']} 道，{sizes}，来自 {ps['scenes']} 个 val 场景"
                     + (f"；不够的题类：{json.dumps(short)}" if short else "；其余题类都够")
                     + ("".join(f"；{q} 另有 {n} 道\"没变化\"的超过 10% 上限，没取" for q, n in sorted(capped.items())))
                     + "。")
        lo = ps.get("label_overrides") or {}
        if lo.get("relabelled") or lo.get("dropped"):
            moves = "、".join(f"{k.replace('->', ' → ')} {n}" for k, n in sorted((lo.get("moves") or {}).items()))
            # one clause per check (2026-10-28: a second check, of the occlusion tasks' goal questions)
            checks = "".join(f"；{c['date']} 核对{c.get('what_zh') or c['what']}：对 {c['right']}、错 {c['wrong']}"
                             + (f"、有歧义 {c['ambiguous']}" if c.get("ambiguous") else "")
                             + f"，重标 {c['relabel']}、去掉 {c['drop']}" for c in lo.get("checks") or [])
            lines.append(f"- 探针集的人工核对（{Path(lo['file']).name}）：{lo['relabelled']} 道按核对的答案重标（{moves}），"
                         f"{lo['dropped']} 道不该出的题去掉{checks}。")
    lines.append("")
    # ------------------------------------------------------------------ family x qtype x split
    lines += ["## 2. 按任务族 × 题类 × 划分的题数", ""]
    qtypes = sorted({r["qtype"] for rs in data.values() for r in rs})
    fams = sorted({r["family"] for rs in data.values() for r in rs})
    for f, rs in data.items():
        c = collections.Counter((r["family"], r["qtype"]) for r in rs)
        rows = [[fam] + [c[(fam, q)] for q in qtypes] + [sum(c[(fam, q)] for q in qtypes)] for fam in fams]
        rows.append(["合计"] + [sum(c[(fam, q)] for fam in fams) for q in qtypes] + [len(rs)])
        lines += [f"**{f}**", "", _table(["任务族"] + qtypes + ["合计"], rows), ""]
    # ------------------------------------------------------------------ answers
    lines += ["## 3. 答案分布", "",
              "\"字母\"是正确选项的字母；\"偏差\"是各字母占比和均匀分布之差的最大值（验收线 5 个百分点）。"
              "\"答案值\"是选项的含义（例如 left / right、full / partial / none）。", ""]
    worst = 0.0
    for f, rs in data.items():
        rows = []
        for q in qtypes:
            sub = [r for r in rs if r["qtype"] == q]
            if not sub:
                continue
            letters = collections.Counter(r["answer"] for r in sub)
            n_opts = 2 if sub[0]["answer"] in ("Yes", "No") else len(sub[0]["options"])
            keys = ["Yes", "No"] if sub[0]["answer"] in ("Yes", "No") else list("ABCD"[:n_opts])
            dev = max(abs(letters[k] / len(sub) - 1.0 / len(keys)) for k in keys)
            worst = max(worst, dev) if f != "p1_probe.jsonl" else worst
            values = collections.Counter(r["answer_value"] for r in sub)
            rows.append([q, len(sub), " / ".join(f"{k} {_pct(letters[k], len(sub))}" for k in keys),
                         f"{100 * dev:.1f}", "，".join(f"{k} {v}" for k, v in values.most_common(6))])
        lines += [f"**{f}**", "", _table(["题类", "题数", "字母", "偏差（百分点）", "答案值（前 6）"], rows), ""]
    lines += [f"训练集和测试集里字母偏差的最大值：{100 * worst:.1f} 个百分点"
              f"（探针集每类只有约 40 道，不计入；它的偏差见上表）。", ""]
    # ------------------------------------------------------------------ forward caps
    lines += ["## 4. 前向题：没变化的题、答案和现在不同的题", "",
              "\"没变化\"：被问的量在动作前后的变化小于门槛（例如转身后到物体的距离不变）；上限 10%。"
              "\"答案变了\"：动作后的答案和现在的答案不同；控制在 35%–65%，防止模型学成\"把现在的答案反过来\"。", ""]
    rows = []
    for f, rs in data.items():
        for q in qtypes:
            sub = [r for r in rs if r["qtype"] == q and r["qclass"] == "forward"]
            if not sub:
                continue
            known = [r for r in sub if r["flip"] is not None]
            rows.append([f, q, len(sub), _pct(sum(not r["changed"] for r in sub), len(sub)),
                         _pct(sum(bool(r["flip"]) for r in known), len(known)) if known else "–"])
    lines += [_table(["文件", "题类", "题数", "没变化", "答案变了"], rows), ""]
    # ------------------------------------------------------------------ funnel
    lines += ["## 5. 过滤：没出的题和原因", "",
              f"每条轨迹的每个不同的帧都当锚点状态（2026-10-18 的题库每条轨迹最多 3 个）。"
              f"原始候选题 {stats['raw_records']} 道（训练和 val 场景合计），"
              "下表是在候选阶段就没出的题（按题类和原因）；之后的平衡、比例、场景上限去掉的数量见第 6 节。", ""]
    funnel = stats["funnel"]
    by_q: Dict[str, Dict[str, int]] = collections.defaultdict(dict)
    for k, v in funnel.items():
        if ":" in k:
            q, why = k.split(":", 1)
            by_q[q][why] = v
    rows = [[q, by_q[q].get("emitted", 0) + by_q[q].get("emitted_interior", 0),
             "，".join(f"{w} {n}" for w, n in sorted(by_q[q].items(), key=lambda x: -x[1]) if not w.startswith("emitted"))]
            for q in sorted(by_q)]
    lines += [_table(["题类", "出了", "没出（原因 数量）"], rows), "",
              f"- 锚点状态 {funnel.get('anchor_states', 0)} 个；一致性检查：{json.dumps(stats['checks'], ensure_ascii=False)}"
              "（`terminal_not_success`、`success_before_terminal` 都没有出现就是 0）。", ""]
    lines += ["原因的含义：`not_visible` 被问物体在当前画面里不满足可见规则；`near_boundary` 离判定阈值太近；"
              "`beyond_45deg` 左右题里两物体的连线和相机左右轴夹角超过 45°；`blocked_move` 动作会被碰撞规则挡住；"
              "`behind_camera` / `out_of_frame` / `cut_off_vertically` 物体在相机后方、出画、上下不完整；"
              "`blocked_turn` 逆向题的那个回合被挡住了（两张图一样）。", ""]
    # ------------------------------------------------------------------ build drops
    lines += ["## 6. 平衡和上限去掉的题", ""]
    depth = stats.get("occlusion_depth") or {}
    if depth.get("by_label"):
        lines += [f"**渲染深度核对遮挡题**（2026-10-21，`occ_depth.py`）：答\"完全挡住\"的要求目标在渲染图里被挡 ≥ "
                  f"{depth['full_min']:.2f}，答\"部分挡住\"的 ≥ {depth['partial_min']:.2f}，否则在平衡之前去掉；"
                  "答\"没有\"的不核对。", "",
                  _table(["题类 / 答案", "核对", "去掉", "没有深度结果（也去掉）"],
                         [[k, v.get("checked", 0), v.get("dropped", 0), v.get("no_depth", 0)]
                          for k, v in depth["by_label"].items()]), ""]
    for split in ("train", "val"):
        rep = stats.get(split) or {}
        rows = [[q, v.get("before"), json.dumps(v.get("dropped", {}), ensure_ascii=False), v.get("after_balance")]
                for q, v in sorted((rep.get("balance") or {}).items())]
        if rows:
            lines += [f"**{split} 场景**", "", _table(["题类", "平衡前", "去掉（原因）", "平衡后"], rows), "",
                      f"场景上限（单个场景不超过 8%）去掉：{json.dumps(rep.get('scene_cap_dropped', {}))}", ""]
    # ------------------------------------------------------------------ text only
    lines += text_only_section(out, data)
    # ------------------------------------------------------------------ examples
    lines += ["## 8. 每类两道原题", ""]
    rng = random.Random(7)
    for q in qtypes:
        pool = [r for r in data["p1_train.jsonl"] if r["qtype"] == q] or [r for r in data["p1_test.jsonl"] if r["qtype"] == q]
        for r in rng.sample(pool, min(2, len(pool))):
            lines += [f"**{q}**（{r['family']}，{r['state_role']}，`{r['sample_id']}`）", "", "```",
                      r["messages"][0]["content"], f"答案：{r['messages'][1]['content']}", "```",
                      "图：" + "；".join(f"`{p}`" for p in r["images"]), ""]
    md.parent.mkdir(parents=True, exist_ok=True)
    md.write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return {"worst_letter_deviation": worst, "files": {f: len(rs) for f, rs in data.items()}}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True)
    ap.add_argument("--md", required=True)
    args = ap.parse_args()
    print(json.dumps(write(Path(args.out), Path(args.md)), indent=1))


if __name__ == "__main__":
    main()
