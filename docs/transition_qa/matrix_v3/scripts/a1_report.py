"""Analysis 1 (exploratory): "knows but does not act", from a1_build.py's questions and the two probe jobs' answers.

Per checkpoint and family: the accuracy of goal_now and goal_after (in failed episodes every right answer is "No",
so the accuracy is the share of turns without a false "Yes"); how often the turn's actions moved away from the goal
(d after > d before), stayed (=) or moved closer (<); and the share of turns where the model answered both questions
right and still chose actions that raised d ("knows but does not act"), with the false "Yes" rates by the remaining
distance.  Writes OUT/a1.{json,md}.

    python a1_report.py A1_DIR
"""
import collections
import json
import sys
from pathlib import Path

MODELS = {"m3ds1s0_300": "D 种子 1 S0（第 300 次）", "m3ds1d_300": "D 种子 1 V4（第 300 次）"}
FAM = ["projective_relations", "absolute_positioning", "screen_occupancy", "equidistance", "occlusion_alignment", "fov_inclusion"]
FAM_ZH = {"projective_relations": "投影", "absolute_positioning": "绝对距离", "screen_occupancy": "画面占比",
          "equidistance": "等距", "occlusion_alignment": "遮挡", "fov_inclusion": "视野"}


def load(p: Path) -> list:
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def stats(ts: list) -> dict:
    n = len(ts)
    known = [t for t in ts if t["d_before"] is not None and t["d_after"] is not None]
    up = [t for t in known if t["d_after"] > t["d_before"]]
    both = [t for t in ts if t["now_ok"] and t["after_ok"]]
    kd = [t for t in both if t in up]
    return {"turns": n, "goal_now_acc": sum(t["now_ok"] for t in ts) / n,
            "goal_after_acc": sum(t["after_ok"] for t in ts if t["after_ok"] is not None) / max(1, sum(t["after_ok"] is not None for t in ts)),
            "d_known": len(known), "d_up": len(up), "d_same": sum(t["d_after"] == t["d_before"] for t in known),
            "d_down": sum(t["d_after"] < t["d_before"] for t in known), "both_right": len(both), "both_right_d_up": len(kd),
            "share_knows_not_acts": len(kd) / n, "of_d_up_both_right": len(kd) / max(1, len(up)),
            "of_both_right_d_up": len(kd) / max(1, len(both))}


def main() -> int:
    A = Path(sys.argv[1])
    rep: dict = {"models": {}}
    md = ["# 分析 1：知而不行（探索性）", "", __doc__.split("\n\n")[1].replace("\n", " "), ""]
    allturns = []
    for m in MODELS:
        q = {r["sample_id"]: r for r in load(A / "questions" / f"{m}.jsonl")}
        a = {r["sample_id"]: r for r in load(A / "answers" / f"{m}.jsonl")}
        assert set(q) == set(a), m
        turns: dict = {}
        for sid, r in q.items():
            f = r["a1"]
            t = turns.setdefault((f["index"], f["turn"]), {"model": m, "family": r["family"], "d_before": f["d_before"],
                                                         "d_after": f["d_after"], "now_ok": None, "after_ok": None})
            t["now_ok" if r["qtype"] == "goal_now" else "after_ok"] = bool(a[sid]["correct"])
            t[r["qtype"] + "_said"] = a[sid]["parsed"]
        ts = list(turns.values())
        assert all(t["now_ok"] is not None for t in ts)
        allturns += ts
        rep["models"][m] = {"all": stats(ts), "by_family": {f: stats([t for t in ts if t["family"] == f]) for f in FAM}}
        # false "Yes" by the remaining distance
        fy = collections.defaultdict(lambda: [0, 0])
        for t in ts:
            for qt, d in (("goal_now", t["d_before"]), ("goal_after", t["d_after"])):
                if t.get(qt + "_said") is None and qt == "goal_after":
                    continue
                band = "未知" if d is None else ("1" if d <= 1 else "2" if d == 2 else "3–4" if d <= 4 else "5 以上")
                fy[(qt, band)][1] += 1
                fy[(qt, band)][0] += t.get(qt + "_said") == "Yes"
        rep["models"][m]["false_yes_by_d"] = {f"{k[0]}/{k[1]}": v for k, v in fy.items()}
    rep["pooled"] = stats(allturns)
    pct = lambda x: f"{100 * x:.1f}%"  # noqa: E731
    md += ["## 合起来（两个检查点）", "",
           "| 检查点 | 回合 | `goal_now` 准确率 | `goal_after` 准确率 | 动作后 d 变大 / 不变 / 变小 | 两题都答对 | 两题都答对却让 d 变大（占全部回合） | 让 d 变大的回合里两题都答对 |",
           "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for m, name in list(MODELS.items()) + [("pooled", "两个合起来")]:
        s = rep["pooled"] if m == "pooled" else rep["models"][m]["all"]
        md.append(f"| {name} | {s['turns']} | {pct(s['goal_now_acc'])} | {pct(s['goal_after_acc'])} | "
                  f"{pct(s['d_up'] / s['d_known'])} / {pct(s['d_same'] / s['d_known'])} / {pct(s['d_down'] / s['d_known'])} | "
                  f"{pct(s['both_right'] / s['turns'])} | {pct(s['share_knows_not_acts'])}（{s['both_right_d_up']} / {s['turns']}） | "
                  f"{pct(s['of_d_up_both_right'])}（{s['both_right_d_up']} / {s['d_up']}） |")
    md += ["", "## 按任务族（两个检查点合起来）", "",
           "| 任务族 | 回合 | `goal_now` 准确率 | `goal_after` 准确率 | d 变大 | 两题都答对却让 d 变大 |", "| --- | --- | --- | --- | --- | --- |"]
    for f in FAM:
        s = stats([t for t in allturns if t["family"] == f])
        md.append(f"| {FAM_ZH[f]} | {s['turns']} | {pct(s['goal_now_acc'])} | {pct(s['goal_after_acc'])} | "
                  f"{pct(s['d_up'] / max(1, s['d_known']))} | {pct(s['share_knows_not_acts'])} |")
    md += ["", "## 误答\"是\"的比例，按离成功还差几步（d）", "",
           "`goal_now` 按回合开始时的 d，`goal_after` 按执行这回合的动作之后的 d。", "",
           "| 检查点 | 题 | d = 1 | d = 2 | d = 3–4 | d ≥ 5 |", "| --- | --- | --- | --- | --- | --- |"]
    for m, name in MODELS.items():
        fy = rep["models"][m]["false_yes_by_d"]
        for qt in ("goal_now", "goal_after"):
            cells = []
            for b in ("1", "2", "3–4", "5 以上"):
                y, n = fy.get(f"{qt}/{b}", [0, 0])
                cells.append(f"{pct(y / n) if n else '—'}（{y} / {n}）")
            md.append(f"| {name} | `{qt}` | " + " | ".join(cells) + " |")
    md.append("")
    (A / "a1.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (A / "a1.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
