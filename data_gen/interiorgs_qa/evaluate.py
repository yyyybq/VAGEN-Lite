"""In-domain evaluation: run a model on a question set, score with the benchmarks' own rules, summarise.

    # questions from render/bank runs (replaced questions dropped as on the review site), or a sampler selection
    python -m data_gen.interiorgs_qa.evaluate --model HF_DIR --run RUN1 --run RUN2 [--exclude RULES] \
        --res 256,512 --families orig,h1 --out OUT
    python -m data_gen.interiorgs_qa.evaluate --model HF_DIR --selection SEL/test.jsonl --res 256 --out OUT

Inference: vLLM (greedy, at most 512 new tokens), the model's own chat template with its default system
prompt (the same one LLaMA-Factory's qwen2_vl template uses in training), images first and then the question,
exactly as export_lf.py writes training records. A video question gets its 32 frames as one video. The
prompt length vLLM reports is stored per request, so the visual-token counts can be checked against training.

Scoring (scoring.py): every question in the original benchmark format ("orig", evaluation group 1) and in the
held-out wording ("h1", evaluation group 2). MMSI-Bench and 3DSRBench questions are also asked once per rotation
of their options (circular evaluation: correct only if every rotation is correct); MMSI reports both the plain
and the circular score. Numeric VSI questions are scored with MRA.

Summary: by group, by benchmark and by wording group (and resolution), each number with its question count and
a 95% interval (Wilson for accuracies, mean +- 1.96 standard errors for MRA), next to the chance level (the
mean of 1 / number of options; M6 counts as 50% because its answers are only "front-left" or "front-right",
user decision of 2026-10-15). Each benchmark is also broken down by its own categories (tasks.py: VSI-Bench
question types, MMSI-Bench question types, MindCube rotation / around / among, 3DSRBench categories, EmbSpatial
relations, ViewSpatial question types, BLINK subsets), so every number lines up with a row of that benchmark's report.
The results are for the report only: not shown on the review site, not used to filter questions.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .export_lf import image_path, load_runs, load_selection
from .qa import render
from .scoring import circular_variants, score
from .tasks import BENCHMARKS, TASKS

CIRCULAR = ("MMSI-Bench", "3DSRBench")
CHANCE_OVERRIDE = {"M6": 0.5}


def wilson(k: float, n: int, z: float = 1.96) -> Tuple[Optional[float], Optional[float]]:
    if n == 0:
        return None, None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return round(max(0.0, c - h), 4), round(min(1.0, c + h), 4)


def mean_ci(xs: Sequence[float]) -> Dict[str, Any]:
    n = len(xs)
    if not n:
        return {"n": 0}
    m = statistics.mean(xs)
    se = statistics.stdev(xs) / math.sqrt(n) if n > 1 else 0.0
    return {"n": n, "mean": round(m, 4), "ci95": [round(m - 1.96 * se, 4), round(m + 1.96 * se, 4)]}


def acc_ci(xs: Sequence[float]) -> Dict[str, Any]:
    n = len(xs)
    if not n:
        return {"n": 0}
    k = sum(xs)
    lo, hi = wilson(k, n)
    return {"n": n, "mean": round(k / n, 4), "ci95": [lo, hi]}


def chance(q: Dict[str, Any]) -> Optional[float]:
    if q["answer_type"] == "numeric":
        return None
    if q["group"] in CHANCE_OVERRIDE:
        return CHANCE_OVERRIDE[q["group"]]
    opts = q["variants"]["orig"].get("options") or []
    return 1.0 / max(len(opts), 2)


def build_requests(items, families: Sequence[str], res: int) -> List[Dict[str, Any]]:
    reqs = []
    for run, q in items:
        media = [image_path(run, p, res) for p in q["extra"]["images"]]
        for fam in families:
            if fam not in q["variants"]:
                continue
            rots = circular_variants(q, fam) if q["benchmark"] in CIRCULAR else [(render(q, fam),
                                                                                 q["variants"][fam]["answer"])]
            for r, (text, ans) in enumerate(rots):
                reqs.append({"qid": q["qid"], "family": fam, "rotation": r, "text": text, "answer": ans,
                             "media": q["media"], "files": media, "res": res})
    return reqs


def run_vllm(model: str, reqs: List[Dict[str, Any]], max_model_len: int, max_tokens: int) -> None:
    import os

    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")   # greedy decoding; avoids a kernel compile
    import numpy as np
    from PIL import Image
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    proc = AutoProcessor.from_pretrained(model)
    llm = LLM(model=model, max_model_len=max_model_len, gpu_memory_utilization=0.85, seed=0,
              limit_mm_per_prompt={"image": 4, "video": 1})
    params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    batch = []
    for r in reqs:
        if r["media"] == "video":
            content = [{"type": "video"}, {"type": "text", "text": r["text"]}]
            mm = {"video": np.stack([np.array(Image.open(f).convert("RGB")) for f in r["files"]])}
        else:
            content = [{"type": "image"} for _ in r["files"]] + [{"type": "text", "text": r["text"]}]
            mm = {"image": [Image.open(f).convert("RGB") for f in r["files"]]}
        prompt = proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                          add_generation_prompt=True)
        batch.append({"prompt": prompt, "multi_modal_data": mm})
    outs = llm.generate(batch, params)
    for r, o in zip(reqs, outs):
        r["reply"] = o.outputs[0].text
        r["prompt_tokens"] = len(o.prompt_token_ids)


def summarise(questions: Dict[str, Dict[str, Any]], reqs: List[Dict[str, Any]]) -> Dict[str, Any]:
    # per (qid, family, res): plain score (rotation 0) and circular hit (all rotations right)
    per: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    for r in reqs:
        q = questions[r["qid"]]
        key = (r["qid"], r["family"], r["res"])
        if r["rotation"] == 0:
            v = dict(q["variants"][r["family"]])
            v["text"], v["answer"] = r["text"], r["answer"]
            qq = dict(q, variants={**q["variants"], r["family"]: v})
            per.setdefault(key, {})["plain"] = score(qq, r["reply"], r["family"])
        if q["benchmark"] in CIRCULAR and q["variants"][r["family"]].get("options"):
            v = dict(q["variants"][r["family"]], answer=r["answer"])
            qq = dict(q, variants={**q["variants"], r["family"]: v})
            per.setdefault(key, {}).setdefault("rot", []).append(score(qq, r["reply"], r["family"])["accuracy"])
    rows = []
    for (qid, fam, res), s in per.items():
        q = questions[qid]
        row = {"qid": qid, "family": fam, "res": res, "group": q["group"], "benchmark": q["benchmark"],
               "subtask": q["subtasks"][0], "chance": chance(q)}
        t = TASKS.get(row["subtask"])
        row["category"] = t.category if t else row["subtask"]
        if "MRA" in s.get("plain", {}):
            row["mra"] = s["plain"]["MRA"]
        elif "plain" in s:
            row["acc"] = s["plain"]["accuracy"]
        if "rot" in s:
            row["circular"] = 1.0 if all(x >= 1.0 for x in s["rot"]) else 0.0
        rows.append(row)

    def block(rs: List[Dict[str, Any]]) -> Dict[str, Any]:
        out = {"questions": len(rs)}
        acc = [r["acc"] for r in rs if "acc" in r]
        if acc:
            out["accuracy"] = acc_ci(acc)
            ch = [r["chance"] for r in rs if "acc" in r and r["chance"] is not None]
            out["chance"] = round(statistics.mean(ch), 4) if ch else None
        circ = [r["circular"] for r in rs if "circular" in r]
        if circ:
            out["circular_accuracy"] = acc_ci(circ)
            out["circular_chance"] = round(statistics.mean(r["chance"] ** len_rot(r) for r in rs if "circular" in r), 4)
        mra = [r["mra"] for r in rs if "mra" in r]
        if mra:
            out["mra"] = mean_ci(mra)
        return out

    def len_rot(r):
        opts = questions[r["qid"]]["variants"][r["family"]].get("options") or []
        return max(len(opts), 1)

    summary: Dict[str, Any] = {}
    for res in sorted({r["res"] for r in rows}):
        for fam in sorted({r["family"] for r in rows}):
            rs = [r for r in rows if r["res"] == res and r["family"] == fam]
            k = f"res{res}/{fam}"
            summary[k] = {"all": block(rs),
                          "official": official_merge(rs),
                          "by_group": {g: block([r for r in rs if r["group"] == g]) for g in sorted({r["group"] for r in rs},
                                       key=lambda x: (x[0] != "S", x[0] != "M", int(x[1:])))},
                          "by_benchmark": {b: block([r for r in rs if r["benchmark"] == b]) for b in sorted({r["benchmark"] for r in rs})},
                          "by_category": {f"{b} | {c}": block([r for r in rs if r["benchmark"] == b and r["category"] == c])
                                          for b, c in sorted({(r["benchmark"], r["category"]) for r in rs},
                                                             key=lambda bc: (BENCHMARKS.index(bc[0]) if bc[0] in BENCHMARKS else 99, bc[1]))}}
    return {"summary": summary, "rows": rows}


# ----------------------------------------------------------------------------- official benchmark scores
# How each benchmark merges its questions into one score (user decision 5 of 2026-10-15: the benchmark's own way):
# * VSI-Bench: the three relative-direction levels are averaged into one task first, then the mean over its 8 tasks
#   (MRA for the 4 numeric tasks, accuracy for the others) -- lmms-eval vsibench _compute_all_subscores;
# * MindCube: accuracy over all questions except translation, which is listed on its own (MindCube-Tiny in EASI-8
#   has rotation, around and among only);
# * MMSI-Bench, BLINK, ViewSpatial-Bench, EmbSpatial-Bench: accuracy over all questions (lmms-eval: MMSI "average"
#   is over all questions, BLINK's group score is size-weighted, ViewSpatial / EmbSpatial overall over all questions);
#   MMSI also gets the circular score;
# * 3DSRBench: CircularEval over all questions (the benchmark's own metric); the plain accuracy (EASI's headline
#   number for 3DSRBench) is given next to it.
# Intervals: Wilson for one accuracy over all questions, a bootstrap over questions (within each task) for means of
# task scores.
VSI_TASKS = {k: t.main for k, t in TASKS.items() if t.benchmark == "VSI-Bench"}   # direction levels share one task
MINDCUBE_TRANSLATION = ("MC-translation-transitive",)
APPENDIX_GROUPS = ("M6", "M10", "M11", "V6")   # too few questions: appendix only, no conclusions (decision 5)


def _score_of(r: Dict[str, Any]) -> Optional[float]:
    return r.get("mra", r.get("acc"))


def _bootstrap(parts: Dict[str, List[float]], combine, n_boot: int = 2000, seed: int = 0) -> Tuple[float, float]:
    import random as _r

    rng = _r.Random(seed)
    vals = []
    for _ in range(n_boot):
        sample = {k: [v[rng.randrange(len(v))] for _ in v] for k, v in parts.items() if v}
        vals.append(combine(sample))
    vals.sort()
    return round(vals[int(0.025 * n_boot)], 4), round(vals[int(0.975 * n_boot) - 1], 4)


def official_merge(rs: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    by_b: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rs:
        by_b[r["benchmark"]].append(r)
    # VSI-Bench
    v = by_b.get("VSI-Bench", [])
    if v:
        tasks: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
        for r in v:
            s = _score_of(r)
            if s is not None:
                tasks[VSI_TASKS.get(r["subtask"], r["subtask"])][r["subtask"]].append(s)

        def vsi_overall(parts: Dict[str, List[float]]) -> float:
            per_task = defaultdict(list)
            for key, xs in parts.items():
                task, sub = key.split("|")
                per_task[task].append(statistics.mean(xs))
            return statistics.mean(statistics.mean(ys) for ys in per_task.values())

        flat = {f"{t}|{s}": xs for t, d in tasks.items() for s, xs in d.items()}
        overall = vsi_overall(flat)
        lo, hi = _bootstrap(flat, vsi_overall)
        task_scores = {t: round(statistics.mean(statistics.mean(xs) for xs in d.values()), 4) for t, d in tasks.items()}
        out["VSI-Bench"] = {"metric": "8 类任务得分的平均（方向题三档先平均）", "score": round(overall, 4), "ci95": [lo, hi],
                            "questions": len(v), "tasks": task_scores, "tasks_present": len(task_scores)}
    # MindCube
    m = by_b.get("MindCube", [])
    if m:
        main = [r["acc"] for r in m if r["subtask"] not in MINDCUBE_TRANSLATION and "acc" in r]
        tr = [r["acc"] for r in m if r["subtask"] in MINDCUBE_TRANSLATION and "acc" in r]
        if main:
            out["MindCube"] = dict(acc_ci(main), metric="全部题的准确率（不含平移题）", score=round(statistics.mean(main), 4))
        if tr:
            out["MindCube（平移，单列）"] = dict(acc_ci(tr), metric="平移题的准确率", score=round(statistics.mean(tr), 4))
    # pooled benchmarks
    for b, name in (("MMSI-Bench", "MMSI-Bench"), ("BLINK", "BLINK"), ("ViewSpatial-Bench", "ViewSpatial-Bench"),
                    ("EmbSpatial-Bench", "EmbSpatial-Bench")):
        xs = [r["acc"] for r in by_b.get(b, []) if "acc" in r]
        if xs:
            out[name] = dict(acc_ci(xs), metric="全部题的准确率", score=round(statistics.mean(xs), 4))
            if b == "MMSI-Bench":
                c = [r["circular"] for r in by_b[b] if "circular" in r]
                if c:
                    out[name]["circular"] = acc_ci(c)
    d3 = by_b.get("3DSRBench", [])
    c = [r["circular"] for r in d3 if "circular" in r]
    if c:
        out["3DSRBench"] = dict(acc_ci(c), metric="轮换选项评分（全部题）", score=round(statistics.mean(c), 4),
                                plain=acc_ci([r["acc"] for r in d3 if "acc" in r]))
    return out


def fmt(b: Dict[str, Any]) -> str:
    parts = []
    if "accuracy" in b:
        a = b["accuracy"]
        parts.append(f"准确率 {a['mean']:.0%}（{a['ci95'][0]:.0%}–{a['ci95'][1]:.0%}，{a['n']} 题；随机 {b['chance']:.0%}）")
    if "circular_accuracy" in b:
        a = b["circular_accuracy"]
        parts.append(f"轮换选项 {a['mean']:.0%}（{a['ci95'][0]:.0%}–{a['ci95'][1]:.0%}，{a['n']} 题）")
    if "mra" in b:
        m = b["mra"]
        parts.append(f"MRA {m['mean']:.2f}（{m['ci95'][0]:.2f}–{m['ci95'][1]:.2f}，{m['n']} 题）")
    return "；".join(parts)


def fmt_official(name: str, o: Dict[str, Any]) -> str:
    ci = o.get("ci95") or [None, None]
    n = o.get("questions", o.get("n"))
    pct = name != "VSI-Bench"
    f = (lambda x: f"{x:.0%}") if pct else (lambda x: f"{x:.3f}")
    s = f"{f(o['score'])}（{f(ci[0])}–{f(ci[1])}，{n} 题）"
    if "circular" in o:
        c = o["circular"]
        s += f"；轮换选项 {c['mean']:.0%}（{c['ci95'][0]:.0%}–{c['ci95'][1]:.0%}）"
    if "plain" in o:
        c = o["plain"]
        s += f"；普通评分 {c['mean']:.0%}（{c['ci95'][0]:.0%}–{c['ci95'][1]:.0%}）"
    if "tasks" in o:
        s += f"；{o['tasks_present']} 类任务在内"
    return s


def markdown(res: Dict[str, Any], title: str) -> str:
    lines = [f"# {title}", ""]
    for k, s in res["summary"].items():
        lines += [f"## {k}", "", "### 主结果：按 benchmark，官方合并方式", "", "| benchmark | 合并方式 | 结果 |", "| --- | --- | --- |"]
        lines += [f"| {b} | {o['metric']} | {fmt_official(b, o)} |" for b, o in s["official"].items()]
        lines += ["", f"全部选择题合计：{fmt(s['all'])}", "", "### 各组", "", "| 组 | 结果 |", "| --- | --- |"]
        lines += [f"| {g} | {fmt(b)} |" for g, b in s["by_group"].items() if g not in APPENDIX_GROUPS]
        app = [(g, b) for g, b in s["by_group"].items() if g in APPENDIX_GROUPS]
        if app:
            lines += ["", "### 附录：题太少的组（只列题数和区间，不单独下结论）", "", "| 组 | 结果 |", "| --- | --- |"]
            lines += [f"| {g} | {fmt(b)} |" for g, b in app]
        if s.get("by_category"):
            lines += ["", "### 按 benchmark 自己的类别", "", "| benchmark | 类别 | 结果 |", "| --- | --- | --- |"]
            lines += [f"| {k.split(' | ')[0]} | {k.split(' | ')[1]} | {fmt(b)} |" for k, b in s["by_category"].items()]
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--run", action="append", default=[])
    ap.add_argument("--exclude", default="")
    ap.add_argument("--selection", default="")
    ap.add_argument("--res", default="256", help="comma separated: 256,512")
    ap.add_argument("--families", default="orig,h1")
    ap.add_argument("--max-model-len", type=int, default=12288)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--title", default="同域评测")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    items = load_selection(a.selection) if a.selection else load_runs(a.run, a.exclude)
    if a.limit:
        items = items[:a.limit]
    questions = {q["qid"]: q for _, q in items}
    fams = a.families.split(",")
    reqs = [r for res in (int(x) for x in a.res.split(",")) for r in build_requests(items, fams, res)]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    run_vllm(a.model, reqs, a.max_model_len, a.max_tokens)
    (out / "predictions.jsonl").write_text("".join(json.dumps({k: v for k, v in r.items() if k != "files"},
                                                              ensure_ascii=False) + "\n" for r in reqs))
    res = summarise(questions, reqs)
    res["meta"] = {"model": a.model, "questions": len(items), "requests": len(reqs), "families": fams,
                   "resolutions": a.res, "seconds": round(time.time() - t0, 1)}
    (out / "summary.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    (out / "summary.md").write_text(markdown(res, a.title))
    print(json.dumps(res["meta"]))
    print((out / "summary.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
