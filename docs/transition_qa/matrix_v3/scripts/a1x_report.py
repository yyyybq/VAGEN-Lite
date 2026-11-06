"""Analysis 1, extension (exploratory): the answers to the one-action questions (a1x_build.py) of D seed 20261019's
S0 and V4 checkpoints, from qa_probe's saved answers.  Writes OUT/a1x.json and OUT/a1x.md.

    python a1x_report.py OUT        # OUT/questions/<model>.jsonl and OUT/answers/<model>.jsonl
"""
import collections
import json
import math
import sys
from pathlib import Path

OUT = Path(sys.argv[1])
MODELS = {"m3ds1s0_300": "S0", "m3ds1d_300": "V4"}


def rate(k, n):
    return None if not n else round(k / n, 4)


def binom_two_sided(k, n):
    if n == 0:
        return 1.0
    lo = min(k, n - k)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(lo + 1)) / 2 ** n)


def main():
    res = {}
    for name, side in MODELS.items():
        qs = {json.loads(l)["sample_id"]: json.loads(l) for l in (OUT / "questions" / f"{name}.jsonl").read_text().splitlines() if l.strip()}
        ans = [json.loads(l) for l in (OUT / "answers" / f"{name}.jsonl").read_text().splitlines() if l.strip()]
        by_turn = collections.defaultdict(list)
        c = collections.Counter()
        for a in ans:
            q = qs[a["sample_id"]]
            f = q["a1x"]
            yes_gold = q["answer"] == "Yes"
            said_yes = a.get("parsed") == "Yes"
            c["questions"] += 1
            c["readable"] += bool(a.get("readable"))
            c["gold_yes"] += yes_gold
            c["gold_yes_said_yes"] += yes_gold and said_yes
            c["gold_no"] += not yes_gold
            c["gold_no_said_yes"] += (not yes_gold) and said_yes
            by_turn[(f["index"], f["turn"])].append((f, yes_gold, said_yes))
        # turns one action away from success: does the model say Yes to a right action, and to the wrong ones?
        near = [v for v in by_turn.values() if any(y for _, y, _ in v)]
        t = collections.Counter()
        for v in near:
            right = [s for _, y, s in v if y]
            wrong = [s for _, y, s in v if not y]
            t["turns"] += 1
            t["said_yes_to_a_right_action"] += any(right)
            t["said_yes_to_a_wrong_action"] += any(wrong)
            t["right_yes_only"] += any(right) and not any(wrong)
            t["right_questions"] += len(right)
            t["right_said_yes"] += sum(right)
            t["wrong_questions"] += len(wrong)
            t["wrong_said_yes"] += sum(wrong)
            # the turn's own choice (failed episodes: it never met the task)
            chosen_first = [f for f, _, _ in v if f["first_chosen"]]
            t["chose_a_right_action_first"] += any(y for f, y, _ in v if f["first_chosen"])
            t["chose_nothing_readable"] += not chosen_first
        res[name] = {
            "side": side, **dict(c),
            "recall_yes": rate(c["gold_yes_said_yes"], c["gold_yes"]),           # said Yes when the action meets the task
            "false_yes": rate(c["gold_no_said_yes"], c["gold_no"]),               # said Yes when it does not
            "near_turns": dict(t),
            "near_knows_rate": rate(t["said_yes_to_a_right_action"], t["turns"]),
            "near_right_yes_rate": rate(t["right_said_yes"], t["right_questions"]),
            "near_wrong_yes_rate": rate(t["wrong_said_yes"], t["wrong_questions"]),
            # sign test over near turns: Yes to a right action more often than to a wrong one, per turn
        }
        pairs = []
        for v in near:
            r = [s for _, y, s in v if y]
            w = [s for _, y, s in v if not y]
            pairs.append((sum(r) / len(r)) - (sum(w) / len(w) if w else 0.0))
        pos, neg = sum(p > 0 for p in pairs), sum(p < 0 for p in pairs)
        res[name]["near_sign_test"] = {"right_more": pos, "wrong_more": neg, "ties": len(pairs) - pos - neg,
                                       "p": round(binom_two_sided(min(pos, neg), pos + neg), 4)}
    (OUT / "a1x.json").write_text(json.dumps(res, indent=1))
    pct = lambda v: "—" if v is None else f"{100 * v:.1f}%"  # noqa: E731
    lines = ["# 分析 1 扩展：答案可以是“是”的题（探索性）", "",
             "D 种子 20261019 的 S0、V4 预登记检查点，测试集失败局的每个回合画面；每个基本动作各问一题"
             "“相机这样动一下，任务会不会达成”（P1 的说法）。答案用评测环境不渲染地重放算出。", "",
             "| 检查点 | 题数 | 答案为是的题 | 答是的比例（答案为是） | 答是的比例（答案为否） | 一步可达的回合 | 其中对某个对的动作答是 | 对的动作答是 / 错的动作答是 | 符号检验 p |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for name, r in res.items():
        nt = r["near_turns"]
        lines.append(f"| {r['side']}（{name}） | {r['questions']} | {r['gold_yes']} | {pct(r['recall_yes'])} | {pct(r['false_yes'])} | "
                     f"{nt.get('turns', 0)} | {pct(r['near_knows_rate'])} | {pct(r['near_right_yes_rate'])} / {pct(r['near_wrong_yes_rate'])} | "
                     f"{r['near_sign_test']['p']} |")
    lines += ["", "一步可达的回合：从这一回合的画面出发，6 个基本动作里至少有一个做一下就达成。失败局里模型在这些回合都没选它（选了的话这局就成功了）。",
              "符号检验：每个一步可达的回合比较“对的动作答是的比例”和“错的动作答是的比例”，看前者更高的回合多不多（双侧）。"]
    (OUT / "a1x.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
