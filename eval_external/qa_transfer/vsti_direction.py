"""VSTI-Bench "camera movement direction" questions, offline from the saved answers: which direction each model
answers, and whether it tells the directions apart at all.

    python -m eval_external.qa_transfer.vsti_direction --stage step2 --models base-7B ours-SFT ... \
        --sft R10=PATH/trajectory_train.jsonl --out eval_external/results/vsti_direction

Each item asks for the camera's main direction of movement between two frames, "relative to its orientation at the
start", with two to four of Forward / Backward / Left / Right as options.  An answer's option comes from
parse_check's rule (the same for every model); the official per-item score is kept as it is.  Per model:
- accuracy, accuracy per true direction and their mean (each true direction weighted equally);
- how often each direction is answered, overall and when it is offered;
- discrimination: on items offering both Forward and Backward, the share answering Backward when the truth is
  Backward minus the share answering Backward when the truth is Forward (Left / Right likewise, answering Right),
  with a Newcombe 95% interval.  0 means the answer does not depend on the true direction.
Without any model: the accuracy of guessing at random and of the best fixed preference order (answer the
highest-ranked offered direction, never looking at the video).  --sft counts the translation actions in SFT targets.
Writes OUT.md and OUT.json.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import re
from collections import Counter
from pathlib import Path

from eval_external.common.stats import wilson
from eval_external.qa_transfer.parse_check import mcq_letter
from eval_external.qa_transfer.summary import load_results

QTYPE = "camera_movement_direction"
DIRS = ("Forward", "Backward", "Left", "Right")
ACTIONS = {"move_forward": "Forward", "move_backward": "Backward", "move_left": "Left", "move_right": "Right"}
_RANGE = re.compile(r"between frame (\d+) and frame (\d+) of (\d+)")
_ACT = re.compile(r"move_(?:forward|backward|left|right)")


def rows(result: dict) -> list[dict]:
    out = []
    for line in open(result["files"][1]):
        x = json.loads(line)
        doc = x["vsibench_score"]
        if isinstance(doc, str):
            doc = json.loads(doc)
        if doc["question_type"] != QTYPE:
            continue
        opts = [o.split(". ", 1)[1].strip() for o in doc["options"]]
        letter = mcq_letter(str(doc["prediction"]), len(opts), doc["options"])
        m = _RANGE.search(doc["question"])
        out.append({"id": x["doc_id"], "truth": doc["ground_truth"].strip(), "options": opts,
                    "pred": opts["ABCD".index(letter)] if letter else None,
                    "score": float(doc["accuracy"]), "span": int(m.group(2)) - int(m.group(1)) if m else None})
    return out


def pct(k: float, n: int) -> float | None:
    return round(100 * k / n, 1) if n else None


def newcombe(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float, float]:
    """p1 - p2 in points with Newcombe's hybrid score interval (from the two Wilson intervals)."""
    p1, l1, u1 = wilson(k1, n1)
    p2, l2, u2 = wilson(k2, n2)
    d = p1 - p2
    return (round(100 * d, 1), round(100 * (d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)), 1),
            round(100 * (d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)), 1))


def discrimination(rs: list[dict], pair: tuple[str, str]) -> dict:
    """Share answering pair[1] when the truth is pair[1], minus the same share when the truth is pair[0]."""
    both = [r for r in rs if all(p in r["options"] for p in pair)]
    hit = [r for r in both if r["truth"] == pair[1]]
    fa = [r for r in both if r["truth"] == pair[0]]
    k1, k2 = sum(r["pred"] == pair[1] for r in hit), sum(r["pred"] == pair[1] for r in fa)
    d, lo, hi = newcombe(k1, len(hit), k2, len(fa))
    return {"answer": pair[1], f"when_truth_{pair[1]}": pct(k1, len(hit)), "n_truth_" + pair[1]: len(hit),
            f"when_truth_{pair[0]}": pct(k2, len(fa)), "n_truth_" + pair[0]: len(fa), "diff": d, "ci95": [lo, hi]}


def summarize(rs: list[dict]) -> dict:
    n = len(rs)
    k = sum(r["score"] for r in rs)
    _, lo, hi = wilson(k, n)
    pred = Counter(r["pred"] for r in rs)
    by_truth = {d: pct(sum(r["score"] for r in rs if r["truth"] == d), sum(r["truth"] == d for r in rs))
                for d in DIRS}
    out = {"n": n, "accuracy": pct(k, n), "ci95": [round(100 * lo, 1), round(100 * hi, 1)],
           "unparsed": pred.get(None, 0), "accuracy_by_truth": by_truth,
           "balanced_accuracy": round(sum(by_truth.values()) / len(DIRS), 1),
           "answer_share": {d: pct(pred.get(d, 0), n) for d in DIRS},
           "answered_when_offered": {d: pct(sum(r["pred"] == d for r in rs if d in r["options"]),
                                            sum(d in r["options"] for r in rs)) for d in DIRS},
           "discrimination": {"forward_backward": discrimination(rs, ("Forward", "Backward")),
                              "left_right": discrimination(rs, ("Left", "Right"))}}
    return out


def baselines(rs: list[dict]) -> dict:
    """Accuracy without looking at the video: random guessing, and every fixed preference order."""
    def fixed(order):
        return pct(sum(next(d for d in order if d in r["options"]) == r["truth"] for r in rs), len(rs))
    orders = sorted(((fixed(o), o) for o in itertools.permutations(DIRS)), reverse=True)
    return {"random": pct(sum(1 / len(r["options"]) for r in rs), len(rs)),
            "best_fixed_order": {"order": list(orders[0][1]), "accuracy": orders[0][0]},
            "worst_fixed_order": {"order": list(orders[-1][1]), "accuracy": orders[-1][0]},
            "truth": dict(Counter(r["truth"] for r in rs).most_common()),
            "offered": {d: pct(sum(d in r["options"] for r in rs), len(rs)) for d in DIRS},
            "true_when_offered": {d: pct(sum(r["truth"] == d for r in rs if d in r["options"]),
                                         sum(d in r["options"] for r in rs)) for d in DIRS},
            "n_options": dict(sorted(Counter(len(r["options"]) for r in rs).items()))}


def sft_translations(path: str) -> dict:
    """Shares of the four translation actions among those in the assistant turns of an SFT jsonl."""
    c = Counter()
    for line in open(path):
        for m in json.loads(line)["messages"]:
            if m["role"] == "assistant":
                c.update(ACTIONS[a] for a in _ACT.findall(str(m["content"])))
    tot = sum(c.values())
    return {"n_translations": tot, "share": {d: pct(c[d], tot) for d in DIRS}}


def markdown(table: dict, base: dict, sft: dict) -> str:
    t = base["truth"]
    lines = ["**VSTI-Bench 相机运动方向**（修正后标签，"
             f"{sum(t.values())} 题；每题 2–4 个选项，取自 Forward / Backward / Left / Right）", "",
             "真实方向：" + "，".join(f"{d} {v} 题" for d, v in t.items()) + "。选项数：" +
             "，".join(f"{k} 个选项 {v} 题" for k, v in base["n_options"].items()) + "。",
             f"不看画面：随机猜 {base['random']}%；按固定偏好作答，最好的顺序 "
             f"{' > '.join(base['best_fixed_order']['order'])} 得 {base['best_fixed_order']['accuracy']}%，"
             f"最差的 {' > '.join(base['worst_fixed_order']['order'])} 得 {base['worst_fixed_order']['accuracy']}%。", "",
             "| 模型 | 准确率 [95% 区间] | 按真实方向的准确率 F / B / L / R | 四个方向的平均 | "
             "回答的占比 F / B / L / R | 选项里有 B 时选 B 的比例 |",
             "|---|---|---|---|---|---|"]
    for m, s in table.items():
        lines.append(f"| {m} | {s['accuracy']} [{s['ci95'][0]}, {s['ci95'][1]}] | "
                     + " / ".join(str(s["accuracy_by_truth"][d]) for d in DIRS) + f" | {s['balanced_accuracy']} | "
                     + " / ".join(str(s["answer_share"][d]) for d in DIRS)
                     + f" | {s['answered_when_offered']['Backward']} |")
    lines += ["", "**辨别力**：同时给出两个相反方向的题里，真实方向是 B（R）时答 B（R）的比例，减去真实方向是 F（L）时答 B（R）"
              "的比例；单位是点，[Newcombe 95% 区间]。0 表示回答和真实方向无关。", "",
              "| 模型 | 前后：真 B 答 B / 真 F 答 B | 前后辨别力 | 左右：真 R 答 R / 真 L 答 R | 左右辨别力 |",
              "|---|---|---|---|---|"]
    for m, s in table.items():
        fb, lr = s["discrimination"]["forward_backward"], s["discrimination"]["left_right"]
        lines.append(f"| {m} | {fb['when_truth_Backward']} / {fb['when_truth_Forward']} | {fb['diff']:+.1f} "
                     f"[{fb['ci95'][0]:+.1f}, {fb['ci95'][1]:+.1f}] | {lr['when_truth_Right']} / {lr['when_truth_Left']} | "
                     f"{lr['diff']:+.1f} [{lr['ci95'][0]:+.1f}, {lr['ci95'][1]:+.1f}] |")
    lines.append(f"\n（前后：真 B {fb['n_truth_Backward']} 题、真 F {fb['n_truth_Forward']} 题；"
                 f"左右：真 R {lr['n_truth_Right']} 题、真 L {lr['n_truth_Left']} 题）")
    if sft:
        lines += ["", "**SFT 训练目标里的平移动作**（四种平移动作各占的比例）", "",
                  "| 数据 | 平移动作数 | F / B / L / R |", "|---|---|---|"]
        for label, s in sft.items():
            lines.append(f"| {label} | {s['n_translations']} | " + " / ".join(str(s["share"][d]) for d in DIRS) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--bench", default="VSTI-Bench")
    ap.add_argument("--sft", action="append", default=[], help="LABEL=PATH of an SFT trajectory jsonl")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    res = load_results(args.stage)
    table, base = {}, None
    for m in args.models:
        if (m, args.bench) not in res:
            continue
        rs = rows(res[(m, args.bench)])
        table[m] = summarize(rs)
        base = base or baselines(rs)
    sft = {label: sft_translations(path) for label, path in (s.split("=", 1) for s in args.sft)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".json").write_text(json.dumps({"baselines": base, "sft": sft, "models": table}, indent=1,
                                                        ensure_ascii=False) + "\n")
    md = markdown(table, base, sft)
    args.out.with_suffix(".md").write_text(md)
    print(md)


if __name__ == "__main__":
    main()
