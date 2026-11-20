"""In-domain evaluation: run a model on a question set, score with the benchmarks' own rules, summarise.

    # questions from render/bank runs (replaced questions dropped as on the review site), or a sampler selection
    python -m data_gen.interiorgs_qa.evaluate --model HF_DIR --run RUN1 --run RUN2 [--exclude RULES] \
        --res 256,512 --families orig,h1 --out OUT
    python -m data_gen.interiorgs_qa.evaluate --model HF_DIR --selection SEL/test.jsonl --res 256 --out OUT

Inference: vLLM (greedy, at most 512 new tokens), the model's own chat template with its default system
prompt (the same one LLaMA-Factory's qwen2_vl template uses in training), images first and then the question,
exactly as export_lf.py writes training records. A video question gets its 32 frames as one video; for Qwen3-VL,
whose vLLM processor needs a video's metadata, with the metadata LLaMA-Factory gives a list of frames in training
(model_matrix/sft/lf_qwen3vl.py): the frames as given, indices 0..n-1, fps --video-fps (LLaMA-Factory's video_fps,
default 2.0). The prompt length vLLM reports is stored per request, so the visual-token counts can be checked
against training.

q38 (the Qwen3.8-27B rerun line, base zero-shot, 2026-11-20): a model whose config.json model_type is "qwen3_5"
(Qwen3.5 / Qwen3.8; vLLM 0.28 builds its multimodal processor from Qwen3-VL's) gets its videos with the same metadata
as Qwen3-VL, its chat template called with enable_thinking False (the template thinks unless told not to, and 512
tokens of thinking would leave no answer) and vLLM's own sampling defaults (generation_config "vllm"), as
data_gen/interiorgs_v10/evaluation.py does for it (MODEL_TYPE_CHAT_KWARGS / MODEL_TYPE_LLM_KWARGS there); its
summary.json meta records them.  --gpu-memory-utilization (default: 0.85 as before) gives vLLM a larger share of
the card for the 27B model.  Every other model type runs exactly as before (vagen/q38/tests/test_qa_q38.py compares
the calls with a264ae46's).

Scoring (scoring.py): every question in the original benchmark format ("orig", evaluation group 1) and in the
held-out wording ("h1", evaluation group 2). MMSI-Bench and 3DSRBench questions are also asked once per rotation
of their options (circular evaluation: correct only if every rotation is correct); MMSI reports both the plain
and the circular score. Numeric VSI questions are scored with MRA.

Summary: by group, by benchmark and by wording group (and resolution), each number with its question count and
a 95% interval (Wilson for accuracies, mean +- 1.96 standard errors for MRA), next to the chance level (the
mean of 1 / number of options; M6 counts as 50% because its answers are only "front-left" or "front-right",
user decision of 2026-10-15). The results are for the report only: not shown on the review site, not used to
filter questions.
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

CIRCULAR = ("MMSI-Bench", "3DSRBench")
CHANCE_OVERRIDE = {"M6": 0.5}
# q38: the model types whose vLLM video processor needs the metadata (qwen3_5: vLLM 0.28's Qwen3.5 model reuses
# Qwen3-VL's multimodal processor, vllm/model_executor/models/qwen3_5.py), and the q38 type's chat-template and vLLM
# arguments (as data_gen/interiorgs_v10/evaluation.py MODEL_TYPE_CHAT_KWARGS / MODEL_TYPE_LLM_KWARGS)
VIDEO_METADATA_TYPES = ("qwen3_vl", "qwen3_5")
MODEL_TYPE_CHAT_KWARGS: Dict[str, Dict[str, Any]] = {"qwen3_5": {"enable_thinking": False}}
MODEL_TYPE_LLM_KWARGS: Dict[str, Dict[str, Any]] = {"qwen3_5": {"generation_config": "vllm"}}
DEFAULT_GPU_MEMORY_UTILIZATION = 0.85


def video_input(frames, model_type: str, fps: float):
    """A video's vLLM input: its frames, and for Qwen3-VL (vLLM 0.11 fails on a Qwen3-VL video without metadata) the
    metadata LLaMA-Factory gives a list of frames in training: frames as given, indices 0..n-1, its video_fps.
    q38: Qwen3.5 / Qwen3.8 ("qwen3_5") the same way."""
    if model_type not in VIDEO_METADATA_TYPES:
        return frames
    n = len(frames)
    return frames, {"fps": fps, "frames_indices": list(range(n)), "total_num_frames": n, "duration": n / fps,
                    "do_sample_frames": False}


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


def q38_record(model_type: str, gpu_memory_utilization: Optional[float]) -> Dict[str, Any]:
    """What a q38 run adds to summary.json's meta: nothing for the other model types without the option (their
    summary.json stays as before)."""
    rec: Dict[str, Any] = {}
    if model_type in MODEL_TYPE_CHAT_KWARGS or model_type in MODEL_TYPE_LLM_KWARGS:
        rec.update(model_type=model_type, chat_template_kwargs=dict(MODEL_TYPE_CHAT_KWARGS.get(model_type, {})),
                   llm_kwargs=dict(MODEL_TYPE_LLM_KWARGS.get(model_type, {})))
    if gpu_memory_utilization is not None:
        rec["gpu_memory_utilization"] = gpu_memory_utilization
    return rec


def run_vllm(model: str, reqs: List[Dict[str, Any]], max_model_len: int, max_tokens: int,
             video_fps: float = 2.0, gpu_memory_utilization: Optional[float] = None) -> None:
    import os

    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")   # greedy decoding; avoids a kernel compile
    import numpy as np
    from PIL import Image
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    proc = AutoProcessor.from_pretrained(model)
    model_type = json.loads((Path(model) / "config.json").read_text()).get("model_type", "")
    chat_kwargs = MODEL_TYPE_CHAT_KWARGS.get(model_type, {})  # q38: {} for every type but qwen3_5
    llm = LLM(model=model, max_model_len=max_model_len,
              gpu_memory_utilization=(DEFAULT_GPU_MEMORY_UTILIZATION if gpu_memory_utilization is None
                                      else gpu_memory_utilization), seed=0,
              limit_mm_per_prompt={"image": 4, "video": 1}, **MODEL_TYPE_LLM_KWARGS.get(model_type, {}))
    params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    batch = []
    for r in reqs:
        if r["media"] == "video":
            content = [{"type": "video"}, {"type": "text", "text": r["text"]}]
            mm = {"video": video_input(np.stack([np.array(Image.open(f).convert("RGB")) for f in r["files"]]),
                                       model_type, video_fps)}
        else:
            content = [{"type": "image"} for _ in r["files"]] + [{"type": "text", "text": r["text"]}]
            mm = {"image": [Image.open(f).convert("RGB") for f in r["files"]]}
        prompt = proc.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                          add_generation_prompt=True, **chat_kwargs)
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
                          "by_benchmark": {b: block([r for r in rs if r["benchmark"] == b]) for b in sorted({r["benchmark"] for r in rs})}}
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
VSI_TASKS = {"VSI-count": "object_counting", "VSI-distance-absolute": "object_abs_distance",
             "VSI-size-object": "object_size_estimation", "VSI-size-room": "room_size_estimation",
             "VSI-distance-relative": "object_rel_distance", "VSI-route": "route_planning",
             "VSI-order": "obj_appearance_order", "VSI-direction-easy": "object_rel_direction",
             "VSI-direction-medium": "object_rel_direction", "VSI-direction-hard": "object_rel_direction"}
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
    ap.add_argument("--video-fps", type=float, default=2.0,
                    help="a video's fps for Qwen3-VL (LLaMA-Factory's video_fps in training, its default 2.0)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--title", default="同域评测")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu-memory-utilization", type=float, default=None,
                    help="q38: vLLM's share of the card (default 0.85, as before); 0.9 for the 27B model")
    a = ap.parse_args()
    if a.gpu_memory_utilization is not None and not 0 < a.gpu_memory_utilization <= 0.95:
        ap.error("--gpu-memory-utilization must be in (0, 0.95]")
    items = load_selection(a.selection) if a.selection else load_runs(a.run, a.exclude)
    if a.limit:
        items = items[:a.limit]
    questions = {q["qid"]: q for _, q in items}
    fams = a.families.split(",")
    reqs = [r for res in (int(x) for x in a.res.split(",")) for r in build_requests(items, fams, res)]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    run_vllm(a.model, reqs, a.max_model_len, a.max_tokens, a.video_fps, a.gpu_memory_utilization)
    (out / "predictions.jsonl").write_text("".join(json.dumps({k: v for k, v in r.items() if k != "files"},
                                                              ensure_ascii=False) + "\n" for r in reqs))
    res = summarise(questions, reqs)
    model_type = json.loads((Path(a.model) / "config.json").read_text()).get("model_type", "")
    res["meta"] = {"model": a.model, "questions": len(items), "requests": len(reqs), "families": fams,
                   "resolutions": a.res, "video_fps": a.video_fps, "seconds": round(time.time() - t0, 1),
                   **q38_record(model_type, a.gpu_memory_utilization)}
    (out / "summary.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
    (out / "summary.md").write_text(markdown(res, a.title))
    print(json.dumps(res["meta"]))
    print((out / "summary.md").read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
