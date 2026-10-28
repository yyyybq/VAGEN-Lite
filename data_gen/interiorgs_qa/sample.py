"""Draw the training set and the evaluation sets from the question bank (pipeline bank), in one pass.

    python -m data_gen.interiorgs_qa.sample --bank BANK [--bank BANK2 ...] [--exclude RULES] --plan configs/sampling_plan_20261015.json --out SEL

Generation and sampling are separate: the bank holds every question the generator made, unbalanced; this script
alone decides what goes into training and evaluation, over the whole bank at once. (Balancing inside the
generator does not survive sharding: S2 balanced inside each of 4 small runs still came out about 70% "no".)

Rules (user decisions of 2026-10-15, values in the plan file):

* split by scene, from the question's scene (split manifest): train scenes -> training set, val / test scenes ->
  the two evaluation sets; no scene contributes to two splits;
* training budget in image inputs (Act-SFT: 24,564 per epoch); a single-image question counts 1, a multi-image
  question its number of images, a video conversation 32 (frames counted once, every question of that video asked
  in the same conversation);
* video scenes: a fraction of the train scenes (plan 1 of the report: one third) gets one 32-frame video
  conversation (the scene's video with the most questions) and nothing else, unless ``rare_groups_everywhere``;
* the rest of the budget goes to single- and multi-image questions, the same number per subtask (water-filling:
  subtasks that cannot supply their share take what they have, the rest is shared by the others);
* answer cap, training and evaluation alike: within one question type (as a reader sees it: the subtask, with
  EmbSpatial's four relation subtasks as one type and BLINK relation questions split by relation word), no answer
  takes more than ``answer_cap`` of the questions (``answer_cap_binary`` for two-option types); numeric answers
  are not capped;
* object cap: one object appears in at most ``object_cap`` questions of one group in one scene (training;
  ``eval_object_cap`` for evaluation);
* benchmark tasks (tasks.py): every bank question must belong to exactly one benchmark task and match it (benchmark,
  group, prompt layout, answer kind); one mismatch stops the run. Each output line carries its task, i.e. the
  benchmark and the benchmark's own category, so training and test questions map 1:1 to rows of the benchmarks.

Output: SEL/{train,val,test}.jsonl (one line per question: run_dir, the question record and its task; export_lf.py,
export_test.py and evaluate.py read them), SEL/summary.json and SEL/summary.md (counts per group and per benchmark
task, image inputs, tokens, answer shares).
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import tasks

RARE_DEFAULT = ("M3", "M6", "M8", "M9", "M10", "M11", "V5", "V6")
TEMPLATE_FAMILY = {"ES-01": "ES-relation", "ES-02": "ES-relation", "ES-03": "ES-relation", "ES-04": "ES-relation"}


def qtype(q: Dict[str, Any]) -> str:
    st = q["subtasks"][0]
    if st == "BL-03":
        v = (q.get("gt") or {}).get("values") or {}
        return "BL-03:" + str(v.get("relation") or v.get("asked") or "")
    return TEMPLATE_FAMILY.get(st, st)


def answer_key(q: Dict[str, Any]) -> Optional[str]:
    if q["answer_type"] == "numeric":
        return None
    return str(q["answer"]).lower()


def n_options(q: Dict[str, Any]) -> int:
    return len(q["variants"]["orig"].get("options") or [])


def image_inputs(q: Dict[str, Any]) -> int:
    return len(q["extra"]["images"])


def video_key(q: Dict[str, Any]) -> Tuple[str, Tuple[str, ...]]:
    return q["scene_id"], tuple(q["views"])


def load_bank(dirs: Sequence[str], exclude: str = "") -> List[Tuple[str, Dict[str, Any]]]:
    """Questions of the banks, without those matched by ``exclude`` (qa.parse_exclude rules, BANK_SUFFIX:GROUP[:SUBTASK]:
    e.g. a bank's video groups when a later bank redid them)."""
    from .qa import excluded, parse_exclude

    rules = parse_exclude(exclude)
    out = []
    for d in dirs:
        p = Path(d).resolve()
        for l in (p / "questions.jsonl").read_text().splitlines():
            if l.strip():
                q = json.loads(l)
                if not excluded(p, q, rules):
                    out.append((str(p), q))
    return out


def obj_ids(q: Dict[str, Any]) -> List[str]:
    return [o["id"] for o in q["objects"] if o.get("role") != "shared_landmark"]


class Picker:
    """Selection under the object cap; answer caps are enforced per question type by ``cap_answers``."""

    def __init__(self, object_cap: int) -> None:
        self.object_cap = object_cap
        self.uses: Counter = Counter()

    def ok(self, q: Dict[str, Any]) -> bool:
        return all(self.uses[(q["scene_id"], q["group"], o)] < self.object_cap for o in obj_ids(q))

    def take(self, q: Dict[str, Any]) -> None:
        for o in obj_ids(q):
            self.uses[(q["scene_id"], q["group"], o)] += 1

    def give_back(self, q: Dict[str, Any]) -> None:
        for o in obj_ids(q):
            self.uses[(q["scene_id"], q["group"], o)] -= 1


def round_robin(cands: List[Tuple[str, Dict[str, Any]]], target: int, picker: Picker,
                rng: random.Random) -> List[Tuple[str, Dict[str, Any]]]:
    """Up to ``target`` questions, cycling over answers (least-used answer first) so answers stay as even as the
    supply allows; numeric questions are taken in random order."""
    by_ans: Dict[Optional[str], List[Tuple[str, Dict[str, Any]]]] = defaultdict(list)
    for c in cands:
        by_ans[answer_key(c[1])].append(c)
    for v in by_ans.values():
        rng.shuffle(v)
    got: List[Tuple[str, Dict[str, Any]]] = []
    count: Counter = Counter()
    while len(got) < target:
        live = [a for a, v in by_ans.items() if v]
        if not live:
            break
        live.sort(key=lambda a: (count[a], rng.random()))
        progressed = False
        for a in live:
            while by_ans[a]:
                c = by_ans[a].pop()
                if picker.ok(c[1]):
                    picker.take(c[1])
                    got.append(c)
                    count[a] += 1
                    progressed = True
                    break
            if progressed:
                break
        if not progressed:
            break
    return got


def cap_answers(sel: List[Tuple[str, Dict[str, Any]]], cap: float, cap_binary: float,
                picker: Optional[Picker] = None) -> Tuple[List[Tuple[str, Dict[str, Any]]], int]:
    """Drop questions of the most common answer of a question type until no answer exceeds its cap."""
    by_t: Dict[str, List[Tuple[str, Dict[str, Any]]]] = defaultdict(list)
    for c in sel:
        by_t[qtype(c[1])].append(c)
    keep, dropped = [], 0
    for t, cs in by_t.items():
        if answer_key(cs[0][1]) is None:
            keep += cs
            continue
        c_max = cap_binary if n_options(cs[0][1]) == 2 else cap
        groups: Dict[str, List] = defaultdict(list)
        for c in cs:
            groups[answer_key(c[1])].append(c)
        while True:
            n = sum(len(v) for v in groups.values())
            a, v = max(groups.items(), key=lambda kv: len(kv[1]))
            if n == 0 or len(v) <= c_max * n + 1e-9 or len(groups) == 1 and n <= 1:
                break
            if len(groups) == 1:          # a type with one answer only: nothing can balance it
                dropped += len(v)
                for c in v:
                    if picker:
                        picker.give_back(c[1])
                groups.pop(a)
                break
            c = v.pop()
            dropped += 1
            if picker:
                picker.give_back(c[1])
        for v in groups.values():
            keep += v
    return keep, dropped


def water_fill(supply: Dict[str, int], weight: Dict[str, float], budget: float) -> Dict[str, int]:
    """Equal count per subtask, limited by supply, so that sum(count * image inputs) fits the budget."""
    lo, hi = 0.0, float(max(supply.values()) if supply else 0)
    for _ in range(60):
        mid = (lo + hi) / 2
        cost = sum(min(mid, s) * weight[k] for k, s in supply.items())
        lo, hi = (mid, hi) if cost <= budget else (lo, mid)
    return {k: int(min(lo, s)) for k, s in supply.items()}


def summarise(name: str, sel: List[Tuple[str, Dict[str, Any]]], conversations: int = 0) -> Dict[str, Any]:
    qs = [q for _, q in sel]
    vids = {video_key(q) for q in qs if q["media"] == "video"}
    inputs = sum(image_inputs(q) for q in qs if q["media"] != "video") + 32 * len(vids)
    by_group: Dict[str, Dict[str, Any]] = {}
    for g in sorted({q["group"] for q in qs}, key=lambda x: (x[0] != "S", x[0] != "M", int(x[1:]))):
        gq = [q for q in qs if q["group"] == g]
        shares = {}
        for t in sorted({qtype(q) for q in gq}):
            tq = [q for q in gq if qtype(q) == t and answer_key(q) is not None]
            if tq:
                c = Counter(answer_key(q) for q in tq)
                a, k = c.most_common(1)[0]
                shares[t] = [a, round(k / len(tq), 3), len(tq)]
        uses = Counter((q["scene_id"], o) for q in gq for o in obj_ids(q))
        by_group[g] = {"questions": len(gq), "scenes": len({q["scene_id"] for q in gq}),
                       "subtasks": dict(Counter(q["subtasks"][0] for q in gq)),
                       "image_inputs": (sum(image_inputs(q) for q in gq) if g[0] != "V" else None),
                       "top_answer_share": shares, "max_object_uses_per_scene": max(uses.values()) if uses else 0,
                       "tokens256_mean": round(statistics.mean(q["tokens"]["256"]["total"] for q in gq), 1),
                       "tokens512_mean": round(statistics.mean(q["tokens"]["512"]["total"] for q in gq), 1)}
    pitch = [p for q in qs for p in (q["extra"].get("pitch_deg") or []) if p is not None]
    return {"split": name, "questions": len(qs), "scenes": len({q["scene_id"] for q in qs}),
            "video_conversations": len(vids), "image_inputs": inputs,
            "by_media": dict(Counter(q["media"] for q in qs)), "by_group": by_group,
            "pitch_deg": ({"n": len(pitch), "median": round(statistics.median(pitch), 1),
                           "p10": round(sorted(pitch)[len(pitch) // 10], 1),
                           "p90": round(sorted(pitch)[9 * len(pitch) // 10], 1)} if pitch else None)}


def line(run_dir: str, q: Dict[str, Any]) -> str:
    return json.dumps({"run_dir": run_dir, "question": q, "task": tasks.record(q["subtasks"][0])}) + "\n"


def task_table(bank: List[Tuple[str, Dict[str, Any]]], sel: Dict[str, List[Tuple[str, Dict[str, Any]]]]) -> Dict[str, Any]:
    """Every benchmark task (tasks.TASKS, benchmark by benchmark) with its questions in the bank and in each split."""
    supply: Dict[str, Counter] = defaultdict(Counter)
    for _, q in bank:
        supply[q["subtasks"][0]][q["split"]] += 1
    n = {s: Counter(q["subtasks"][0] for _, q in v) for s, v in sel.items()}
    keys = list(tasks.TASKS)
    order = sorted(keys, key=lambda k: (tasks.BENCHMARKS.index(tasks.TASKS[k].benchmark), keys.index(k)))
    return {k: dict(tasks.record(k), bank=dict(supply[k]), **{s: n[s][k] for s in sel}) for k in order}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bank", action="append", required=True)
    ap.add_argument("--exclude", default="", help="BANK_SUFFIX:GROUP[:SUBTASK],... questions of a bank that a later bank replaced")
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    plan = json.loads(Path(a.plan).read_text())
    rng = random.Random(plan.get("seed", 20261015))
    rare = set(plan.get("rare_groups", RARE_DEFAULT))
    items = load_bank(a.bank, a.exclude)
    mismatches = []
    for _, q in items:
        try:
            tasks.check(q)
        except tasks.TaskMismatch as e:
            mismatches.append(str(e))
    if mismatches:
        raise SystemExit(f"{len(mismatches)} bank questions do not match their benchmark task, e.g.\n" + "\n".join(mismatches[:20]))
    by_split: Dict[str, List[Tuple[str, Dict[str, Any]]]] = defaultdict(list)
    for it in items:
        by_split[it[1]["split"]].append(it)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summary: Dict[str, Any] = {"plan": plan, "bank_questions": len(items),
                               "bank_by_split": {k: len(v) for k, v in by_split.items()}}

    # ---------------- training
    tr = plan["train"]
    train = by_split.get("train", [])
    scenes = sorted({q["scene_id"] for _, q in train})
    n_scenes = tr.get("n_scenes") or len(scenes)
    videos: Dict[str, Dict[Tuple, List]] = defaultdict(lambda: defaultdict(list))
    for it in train:
        if it[1]["media"] == "video":
            videos[it[1]["scene_id"]][video_key(it[1])].append(it)
    with_video = [s for s in scenes if videos.get(s)]
    rng.shuffle(with_video)
    n_video = min(len(with_video), round(tr.get("video_scene_fraction", 1 / 3) * len(scenes)))
    video_scenes = set(with_video[:n_video])
    picker = Picker(int(plan.get("object_cap", 2)))
    video_sel: List[Tuple[str, Dict[str, Any]]] = []
    for s in sorted(video_scenes):
        best = max(videos[s].values(), key=len)
        video_sel += best
    video_sel, vdrop = cap_answers(video_sel, plan["answer_cap"], plan["answer_cap_binary"])
    n_conv = len({video_key(q) for _, q in video_sel})
    budget = tr["image_input_budget"] * (len(scenes) / n_scenes if tr.get("scale_budget_to_scenes") else 1.0)
    rest_budget = budget - 32 * n_conv
    pool = [it for it in train if it[1]["media"] != "video"
            and (it[1]["scene_id"] not in video_scenes
                 or (tr.get("rare_groups_everywhere") and it[1]["group"] in rare))]
    by_sub: Dict[str, List] = defaultdict(list)
    for it in pool:
        by_sub[it[1]["subtasks"][0]].append(it)
    supply = {k: len(v) for k, v in by_sub.items()}
    weight = {k: statistics.mean(image_inputs(q) for _, q in v) for k, v in by_sub.items()}
    targets = water_fill(supply, weight, rest_budget)
    sel: List[Tuple[str, Dict[str, Any]]] = []
    for k in sorted(by_sub, key=lambda k: supply[k]):      # scarce subtasks pick their objects first
        sel += round_robin(by_sub[k], targets[k], picker, rng)
    sel, sdrop = cap_answers(sel, plan["answer_cap"], plan["answer_cap_binary"], picker)
    train_sel = sel + video_sel
    summary["train_targets"] = {"video_scenes": len(video_scenes), "video_conversations": n_conv,
                                "budget": round(budget), "budget_single_multi": round(rest_budget),
                                "per_subtask": targets, "supply": supply,
                                "dropped_by_answer_cap": {"single_multi": sdrop, "video": vdrop}}
    summary["train"] = summarise("train", train_sel)

    # ---------------- evaluation: every question of val / test scenes under the same caps
    ev = plan.get("eval", {})
    sels: Dict[str, List[Tuple[str, Dict[str, Any]]]] = {"train": train_sel}
    for split in ("val", "test"):
        pk = Picker(int(plan.get("eval_object_cap", 3)))
        cand = by_split.get(split, [])
        by_sub = defaultdict(list)
        for it in cand:
            by_sub[it[1]["subtasks"][0]].append(it)
        es: List[Tuple[str, Dict[str, Any]]] = []
        for k in sorted(by_sub, key=lambda k: len(by_sub[k])):
            es += round_robin(by_sub[k], ev.get("max_per_subtask") or 10 ** 9, pk, rng)
        es, d = cap_answers(es, plan["answer_cap"], plan["answer_cap_binary"], pk)
        summary[split] = summarise(split, es)
        summary[split]["dropped_by_answer_cap"] = d
        sels[split] = es
    for split, s in sels.items():
        (out / f"{split}.jsonl").write_text("".join(line(r, q) for r, q in s))
    summary["by_task"] = task_table(items, sels)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    lines = ["# 抽样结果", ""]
    for split in ("train", "val", "test"):
        s = summary.get(split)
        if not s:
            continue
        lines += [f"## {split}：{s['questions']} 题，{s['scenes']} 个场景，图片输入 {s['image_inputs']} 次，"
                  f"视频对话 {s['video_conversations']} 段", "", "| 组 | 题数 | 场景 | 最常见答案占比（题型：答案 占比 题数） |", "| --- | --- | --- | --- |"]
        for g, b in s["by_group"].items():
            sh = "；".join(f"{t}: {x[0]} {x[1]:.0%} ({x[2]})" for t, x in b["top_answer_share"].items())
            lines.append(f"| {g} | {b['questions']} | {b['scenes']} | {sh} |")
        lines.append("")
    lines += ["## 按 benchmark 任务（tasks.py；类别名与 benchmark 自己的数据一致）", "",
              "| benchmark | 类别 | 任务 | 定义 | 题库（train / val / test 场景） | 训练 | val | test |",
              "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for k, r in summary["by_task"].items():
        bank = " / ".join(str(r["bank"].get(s, 0)) for s in ("train", "val", "test"))
        flag = "" if r["test"] else " **（test 无题）**"
        lines.append(f"| {r['benchmark']} | {r['category']} | {k} | {'一致' if r['definition'] == tasks.SAME else '有差异'} | "
                     f"{bank} | {r['train']} | {r['val']} | {r['test']}{flag} |")
    lines.append("")
    (out / "summary.md").write_text("\n".join(lines))
    print(json.dumps({k: {kk: summary[k][kk] for kk in ("questions", "scenes", "image_inputs", "video_conversations")}
                      for k in ("train", "val", "test") if k in summary}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
