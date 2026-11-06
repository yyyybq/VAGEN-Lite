"""Diagnostic 3: accuracy on the P1 probe questions, per question type, over a run's checkpoints.

    # one card: every model in turn (a separate process per model, so vLLM frees the card in between)
    python -m eval.diagnostics.qa_probe run --probe p1_probe.jsonl --out OUT \
        --model rs0_050=/path/to/global_step_50/actor/huggingface --model ...
    python -m eval.diagnostics.qa_probe run --probe ... --out OUT --from-eval EVAL_DIR --names rs0_050,rs0_100
    # table and figure: checkpoint step -> accuracy per question type
    python -m eval.diagnostics.qa_probe curve --out OUT --chain "R11-SFT=sft_formal_e{1..10}@154" ...
    # the pre-registered comparison of two groups (PREREG_D_vs_C.md section 3.3): group A - group B, two seeds each
    python -m eval.diagnostics.qa_probe compare --probe p1_probe.jsonl --out OUT --label D_vs_C \
        --a "D=20261019:d_s0_300,d_d_300;20261021:d2_s0_300,d2_d_300" \
        --b "C=20261019:c_s0_300,c_d_300;20261021:c2_s0_300,c2_d_300"

``run`` asks each probe question exactly as the training row has it (no system prompt, the images where the
``<image>`` markers stand), greedy, at most 256 new tokens, and reads the answer leniently (``parse_answer``: a
yes / no, an option letter in several spellings, the option's text, number or key word, after a <think> block).
An answer it cannot read counts as wrong and is reported as unreadable, because the action-only models answer in
their action format.  Separately, an answer not in the asked format (the letter alone, or Yes / No alone) counts as
non-compliant even when it can be read (user, 2026-10-19: lenient reading, the format non-compliance rate on its own).
OUT/<name>.jsonl has every answer, OUT/<name>.json the accuracies.

``run --no-images``: the text-only baseline (user, 2026-10-20: the original 7B given only the stem and the options,
a fixed column from now on).  The images are left out, every ``<image>`` marker becomes ``[image not shown]`` and
BLIND_NOTE goes before the question, so the model still picks an option; everything else as above.

``curve``: a chain is NAME=MODELS@STEPS_PER_MODEL or NAME=m1:s1,m2:s2,... (model name : training step); it writes
OUT/curve.md (one table per chain: step, overall, chance, strict, unreadable, non-compliant, per class, per question
type) and OUT/curve.png.  ``--reference NAME=LABEL`` adds a fixed row to every table (e.g. the text-only baseline)
and a dotted line to the figure.

``compare``: a group is [GROUP=]SEED[;SEED], a seed being [LABEL:]S0_RESULT,D_RESULT (that seed's two RLs, S0
first), or a single result (one pair, for checks; it serves every pair of the other group).  Pairs are the same seed
and the same side (S0 against S0, D against D); two groups of two seeds need the same seed labels (unlabelled seeds
are seed1, seed2 in order).  ``goal_now`` and ``goal_after`` each: the accuracy difference and the exact McNemar test
per pair, per seed (the mean of its two pairs; McNemar on its 2 x n question-RL pairs) and pooled (the mean of all
pair differences; McNemar on all question-RL pairs).  The two occlusion types pooled: the balanced accuracy (the mean
of the full / partial / none accuracies) per pair, per seed and pooled, with a 95% CI and two-sided p from a bootstrap
that resamples questions within each answer value (the same draw for both groups and every pair).  Verdicts as the
pre-registration has them (``verdict``): "same sign" over the two seeds' differences (pre-registration v3, the user's
decision of 2026-10-30); with one seed, over its S0 and D differences (v2).  Writes OUT/compare_<label>.json and .md.
"""

from __future__ import annotations

import argparse
import base64
import collections
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

MAX_TOKENS = 256  # the base model reasons before it answers; 64 cut 18% of its answers before the letter
LETTERS = "ABCD"


# =============================================================================
# reading an answer
# =============================================================================
def _clean(text: str) -> str:
    """Special tokens removed; the part after a closed <think> block; the inside of an <answer> block."""
    t = re.sub(r"<\|[^>]*\|>", " ", str(text or ""))
    t = t.replace("（", "(").replace("）", ")").replace("*", "")
    if "</think>" in t:
        t = t.rsplit("</think>", 1)[1]
    m = re.search(r"<answer>(.*?)(?:</answer>|$)", t, re.DOTALL | re.IGNORECASE)
    if m:
        t = m.group(1)
    return t.strip()


def compliant(text: str, options: Sequence[str], yes_no: bool) -> bool:
    """The answer in exactly the asked format: "Yes" / "No", or the option letter alone (a final period allowed),
    as the training rows have it."""
    t = re.sub(r"<\|[^>]*\|>", " ", str(text or "")).strip()
    if yes_no:
        return re.fullmatch(r"(?i)(yes|no)\.?", t) is not None
    return re.fullmatch(rf"[{LETTERS[: len(options)]}]\.?", t) is not None


FORMATS = ("exact", "letter_variant", "reasoning", "action_tag", "other")


def format_class(text: str, options: Sequence[str], yes_no: bool) -> str:
    """How an answer departs from the asked format: exact (the letter or Yes / No alone); letter_variant (the
    letter in brackets and / or with the option's text, one short line); reasoning (an explanation, with or without
    a <think> block); action_tag (the action format, <action>...</action>); other."""
    t = re.sub(r"<\|[^>]*\|>", " ", str(text or "")).strip()
    if compliant(t, options, yes_no):
        return "exact"
    if re.search(r"<\s*/?\s*action\s*>", t, re.IGNORECASE):
        return "action_tag"
    one_line = "\n" not in t and len(t) <= 120
    if yes_no and one_line and re.match(r"(?i)^\W*(yes|no)\b", t):
        return "letter_variant"
    if not yes_no and one_line and re.match(rf"^\W*\(?[{LETTERS[: len(options)]}]\)?(?:[\s.:)]|$)", t):
        return "letter_variant"
    if "<think>" in t.lower() or len(t.split()) >= 8:
        return "reasoning"
    return "other"


def _numbers(s: str) -> List[float]:
    return [float(x) for x in re.findall(r"(?<![\d.])\d+(?:\.\d+)?", s)]


KEYWORDS = {"to the left": ("left",), "to the right": ("right",)}  # option text -> words that name it alone
DEGREES = {"fully": ("fully", "completely", "entirely"), "partly": ("partly", "partially"), "not": ("not",)}


def parse_answer(text: str, options: Sequence[str], yes_no: bool) -> Optional[str]:
    """The answer a model gave, read leniently: "Yes" / "No", or an option letter; None when it cannot be read.
    Accepted: the letter at the start ("B", "(B)", "B."), "answer / option (is) B", "(B)" anywhere, a lone lower-case
    letter, the option's text, the option's number (distance, screen height), left / right, fully / partly / not
    hidden; after a closed <think> block, inside an <answer> block."""
    t = _clean(text)
    low = t.lower()
    if yes_no:
        m = re.search(r"\b(yes|no)\b", t, re.IGNORECASE)
        return m.group(1).capitalize() if m else None
    letters = LETTERS[: len(options)]
    m = re.match(rf"^\(?([{letters}])\)?(?:[\s.):,]|$)", t)
    if m:
        return m.group(1)
    m = re.fullmatch(rf"\(?([{letters.lower()}])\)?\.?", t)
    if m:
        return m.group(1).upper()
    m = re.search(rf"(?:answer|option|choice)\s*(?:is|:)?\s*\(?([{letters}{letters.lower()}])\)?\b", t, re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m = re.search(rf"\(([{letters}])\)", t)
    if m:
        return m.group(1)
    hits = [LETTERS[i] for i, o in enumerate(options) if o and o.lower() in low]
    if len(hits) == 1:
        return hits[0]
    words = {i: KEYWORDS.get(o.lower(), ()) for i, o in enumerate(options)}
    if all(words.values()):
        hits = [LETTERS[i] for i, ws in words.items() if any(re.search(rf"\b{w}\b", low) for w in ws)]
        if len(hits) == 1:
            return hits[0]
    names = [o.split(" ", 1)[0].lower() for o in options]
    if set(names) == set(DEGREES) and all(" hidden " in f" {o} " for o in options):
        hits = [LETTERS[i] for i, n in enumerate(names) if any(re.search(rf"\b{w}\b", low) for w in DEGREES[n])]
        if len(hits) == 1:
            return hits[0]
    opt_nums = [_numbers(o) for o in options]
    if all(len(n) == 1 for n in opt_nums):
        said = set(_numbers(t))
        hits = [LETTERS[i] for i, n in enumerate(opt_nums) if n[0] in said]
        if len(hits) == 1:
            return hits[0]
    return None


_NUM = r"-?\d+(?:\.\d+)?"


def judge_pointing(row: Dict[str, Any], text: str) -> Dict[str, Any]:
    """A pointing question's answer (group G of matrix v3, data_gen/transition_qa/pointing_qa.py; row "judge" point or
    box, "box" the true box in pixels of "render_size", "coords" pixel or norm1000): the first 2 / 4 numbers of the
    answer as the point / the box in the row's coordinates; right when the point lies in the true box / the boxes'
    IoU is at least 0.5.  {"parsed", "readable", "correct", "compliant": exactly "(x,y)" / "(x1,y1,x2,y2)"}."""
    t = _clean(text)
    n = 2 if row["judge"] == "point" else 4
    nums = [float(v) for v in re.findall(_NUM, t)][:n]
    shape = r"\(\s*" + r"\s*,\s*".join([_NUM] * n) + r"\s*\)\.?"
    out = {"parsed": None, "readable": False, "correct": False, "compliant": re.fullmatch(shape, t) is not None}
    if len(nums) < n:
        return out
    w, h = row.get("render_size") or (256, 256)
    sx, sy = (w / 1000.0, h / 1000.0) if row.get("coords") == "norm1000" else (1.0, 1.0)
    x1, y1, x2, y2 = (float(v) for v in row["box"])
    if n == 2:
        x, y = nums[0] * sx, nums[1] * sy
        correct = x1 <= x <= x2 and y1 <= y <= y2
    else:
        a = (min(nums[0], nums[2]) * sx, min(nums[1], nums[3]) * sy, max(nums[0], nums[2]) * sx,
             max(nums[1], nums[3]) * sy)
        inter = max(0.0, min(a[2], x2) - max(a[0], x1)) * max(0.0, min(a[3], y2) - max(a[1], y1))
        union = (a[2] - a[0]) * (a[3] - a[1]) + (x2 - x1) * (y2 - y1) - inter
        correct = union > 0 and inter / union >= 0.5
    return dict(out, parsed=",".join(f"{v:g}" for v in nums), readable=True, correct=bool(correct))


def free_form(row: Dict[str, Any]) -> bool:
    """A question answered in free form (pointing: coordinates), neither by an option letter nor by Yes / No."""
    return not (row.get("options") or []) and row["messages"][1]["content"] not in ("Yes", "No")


def score(rows: Sequence[Dict[str, Any]], outputs: Sequence[str]) -> List[Dict[str, Any]]:
    """Each answer read and judged.  A free-form question (``free_form``) is judged by ``judge_pointing`` when it is a
    pointing question (row "judge" point / box), else kept as text, unjudged (readable and correct False)."""
    out = []
    for r, text in zip(rows, outputs):
        gold = r["messages"][1]["content"]
        if free_form(r):
            j = judge_pointing(r, text) if r.get("judge") in ("point", "box") else None
            out.append({"sample_id": r["sample_id"], "qtype": r["qtype"], "qclass": r["qclass"], "family": r["family"],
                        "gold": gold, "output": text, "parsed": j["parsed"] if j else None,
                        "correct": bool(j and j["correct"]), "readable": bool(j and j["readable"]),
                        "compliant": bool(j and j["compliant"]), "format": "exact" if j and j["compliant"] else "other",
                        "n_options": None})
            continue
        yes_no = gold in ("Yes", "No")
        got = parse_answer(text, r.get("options") or [], yes_no)
        out.append({"sample_id": r["sample_id"], "qtype": r["qtype"], "qclass": r["qclass"], "family": r["family"],
                    "gold": gold, "output": text, "parsed": got, "correct": got == gold, "readable": got is not None,
                    "compliant": compliant(text, r.get("options") or [], yes_no),
                    "format": format_class(text, r.get("options") or [], yes_no),
                    "n_options": 2 if yes_no else len(r.get("options") or [])})
    return out


def summary(answers: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    def acc(sel: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        n = len(sel)
        if not n:
            return {"n": 0, "accuracy": None, "unreadable": None, "noncompliant": None, "chance": None}
        return {"n": n, "accuracy": sum(a["correct"] for a in sel) / n,
                "unreadable": sum(not a["readable"] for a in sel) / n,
                "noncompliant": sum(not a.get("compliant", True) for a in sel) / n,
                "strict_accuracy": sum(a["correct"] and a.get("compliant", True) for a in sel) / n,
                "chance": sum(1.0 / a["n_options"] for a in sel) / n if all(a.get("n_options") for a in sel) else None,
                "formats": {f: sum(a.get("format") == f for a in sel) / n for f in FORMATS} if all("format" in a for a in sel) else None}
    out = {"all": acc(answers)}
    for key in ("qclass", "qtype"):
        groups: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
        for a in answers:
            groups[a[key]].append(a)
        out[key] = {k: acc(v) for k, v in sorted(groups.items())}
    for q, sel in out["qtype"].items():  # yes / no types: accuracy when the answer is Yes and when it is No
        rows = [a for a in answers if a["qtype"] == q]
        if rows and all(a["gold"] in ("Yes", "No") for a in rows):
            sel["by_gold"] = {g: (sum(a["correct"] for a in rows if a["gold"] == g) / max(1, sum(a["gold"] == g for a in rows)))
                              for g in ("Yes", "No")}
            sel["said_yes"] = sum(a["parsed"] == "Yes" for a in rows) / len(rows)
    return out


OPTION_TYPES = ("correct", "count", "opposite", "other_kind", "action_and_count", "unreadable")


def inverse_confusion(answers: Sequence[Dict[str, Any]], probe: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    """action_between: true action -> what the chosen option was: the answer, the same action another number of times
    (count), the opposite action or the other kind on the same side the same number of times (opposite / other_kind),
    the other action another number of times (action_and_count: the fourth cell of the 2 x 2 options, 2026-10-19), or
    unreadable."""
    from data_gen.interiorgs_v10.contract import INVERSE_ACTION
    from data_gen.transition_qa.shortcuts import parse_action_option

    table: Dict[str, Dict[str, int]] = collections.defaultdict(lambda: {t: 0 for t in OPTION_TYPES})
    for a in answers:
        if a["qtype"] != "action_between":
            continue
        row = probe[a["sample_id"]]
        opts = [parse_action_option(o) for o in row["options"]]
        right = opts[LETTERS.index(row["answer"])]
        if a["parsed"] not in tuple(LETTERS[: len(opts)]):
            kind = "unreadable"
        else:
            chosen = opts[LETTERS.index(a["parsed"])]
            kind = ("correct" if chosen == right else "count" if chosen[0] == right[0]
                    else "action_and_count" if chosen[1] != right[1]
                    else "opposite" if chosen[0] == INVERSE_ACTION[right[0]] else "other_kind")
        table[right[0]][kind] += 1
    return dict(table)


def cmd_confusion(args: argparse.Namespace) -> None:
    out = Path(args.out)
    probe = {r["sample_id"]: r for r in (json.loads(l) for l in Path(args.probe).read_text().splitlines() if l.strip())}
    res = {}
    for name in [n for n in args.names.split(",") if n]:
        answers = [json.loads(l) for l in (out / f"{name}.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        res[name] = inverse_confusion(answers, probe)
    (out / "inverse_confusion.json").write_text(json.dumps(res, indent=1))
    for name, table in res.items():
        tot = collections.Counter()
        for row in table.values():
            tot.update(row)
        print(name, json.dumps(dict(tot)))


# =============================================================================
# running a model (one card)
# =============================================================================
def _image_part(path: str) -> Dict[str, Any]:
    data = base64.b64encode(Path(path).read_bytes()).decode()
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + data}}


BLIND_NOTE = "The images for this question are not shown to you. Still choose the most likely answer.\n"
BLIND_MARK = "[image not shown]"


def chat(row: Dict[str, Any], blind: bool = False) -> List[Dict[str, Any]]:
    """The probe row's user turn with its images where the <image> markers stand; ``blind``: text only."""
    text = row["messages"][0]["content"]
    parts = text.split("<image>")
    if len(parts) - 1 != len(row["images"]):
        raise ValueError(f"{row['sample_id']}: {len(parts) - 1} <image> markers, {len(row['images'])} images")
    if blind:
        return [{"role": "user", "content": [{"type": "text", "text": BLIND_NOTE + BLIND_MARK.join(parts)}]}]
    content: List[Dict[str, Any]] = []
    for i, piece in enumerate(parts):
        if piece:
            content.append({"type": "text", "text": piece})
        if i < len(row["images"]):
            content.append(_image_part(row["images"][i]))
    return [{"role": "user", "content": content}]


def run_one(name: str, model: str, probe: Path, out: Path, blind: bool = False) -> None:  # pragma: no cover - GPU
    from vllm import LLM, SamplingParams

    rows = [json.loads(l) for l in probe.read_text().splitlines() if l.strip()]
    t0 = time.time()
    llm = LLM(model=model, gpu_memory_utilization=0.85, max_model_len=4096, enforce_eager=True,
              limit_mm_per_prompt={"image": 2}, trust_remote_code=False)
    load = time.time() - t0
    params = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)
    res = llm.chat([chat(r, blind) for r in rows], sampling_params=params, use_tqdm=False)
    answers = score(rows, [r.outputs[0].text for r in res])
    with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
        for a in answers:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")
    (out / f"{name}.json").write_text(json.dumps({"name": name, "model": model, "probe": str(probe),
                                                  "images": not blind, "load_seconds": round(load, 1),
                                                  "answer_seconds": round(time.time() - t0 - load, 1),
                                                  **summary(answers)}, indent=1))


def cmd_rescore(args: argparse.Namespace) -> None:
    """Read the saved answers again with the current parser and format rules (no GPU): OUT/<name>.jsonl and the
    accuracies in OUT/<name>.json are rewritten; the model, timings and probe path are kept.  Answers to questions no
    longer in the probe (dropped by a label check) are left out; a relabelled question is scored on its new answer."""
    out = Path(args.out)
    rows = {r["sample_id"]: r for r in (json.loads(l) for l in Path(args.probe).read_text().splitlines() if l.strip())}
    for path in sorted(out.glob("*.jsonl")):
        old = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
        gone = sum(a["sample_id"] not in rows for a in old)
        old = [a for a in old if a["sample_id"] in rows]
        answers = score([rows[a["sample_id"]] for a in old], [a["output"] for a in old])
        with open(path, "w", encoding="utf-8") as f:
            for a in answers:
                f.write(json.dumps(a, ensure_ascii=False) + "\n")
        meta_path = path.with_suffix(".json")
        meta = json.loads(meta_path.read_text()) if meta_path.exists() else {"name": path.stem}
        meta = {k: v for k, v in meta.items() if k in ("name", "model", "probe", "images", "load_seconds", "answer_seconds")}
        meta_path.write_text(json.dumps({**meta, **summary(answers)}, indent=1))
        print(f"{path.stem}: " + json.dumps({k: summary(answers)["all"][k] for k in ("accuracy", "noncompliant")})
              + (f" ({gone} answers to questions no longer in the probe left out)" if gone else ""))


def models_from_eval(ev: Path, names: Sequence[str]) -> List[Tuple[str, str]]:
    return [(n, json.loads((ev / n / "model.json").read_text())["model"]) for n in names]


def cmd_run(args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    models = [tuple(m.split("=", 1)) for m in args.model]
    if args.from_eval:
        models += models_from_eval(Path(args.from_eval), [n for n in args.names.split(",") if n])
    for name, model in models:
        if (out / f"{name}.json").exists():
            print(f"{name}: done before", flush=True)
            continue
        if args.in_process:
            run_one(name, model, Path(args.probe), out, args.no_images)
        else:  # a fresh process per model: vLLM releases the card only when its process ends
            cmd = [sys.executable, "-m", "eval.diagnostics.qa_probe", "run", "--probe", args.probe, "--out", str(out),
                   "--model", f"{name}={model}", "--in-process"] + (["--no-images"] if args.no_images else [])
            rc = subprocess.run(cmd, env=os.environ.copy()).returncode
            print(f"{name}: exit {rc}", flush=True)
            continue
        print(f"{name}: " + json.dumps(json.loads((out / f"{name}.json").read_text())["all"]), flush=True)


# =============================================================================
# curves
# =============================================================================
def parse_chain(spec: str) -> Tuple[str, List[Tuple[str, int]]]:
    """NAME=prefix{a..b}@STEP (model k gets step k * STEP) or NAME=model:step,model:step."""
    name, rest = spec.split("=", 1)
    m = re.fullmatch(r"(.*)\{(\d+)\.\.(\d+)\}(.*)@(\d+)", rest)
    if m:
        pre, a, b, post, step = m.group(1), int(m.group(2)), int(m.group(3)), m.group(4), int(m.group(5))
        return name, [(f"{pre}{k}{post}", k * step) for k in range(a, b + 1)]
    pts = []
    for item in rest.split(","):
        model, step = item.rsplit(":", 1)
        pts.append((model, int(step)))
    return name, pts


def cmd_curve(args: argparse.Namespace) -> None:
    out = Path(args.out)
    chains = [parse_chain(c) for c in args.chain]
    refs = []  # fixed rows: (label, name, result)
    for spec in args.reference:
        name, label = spec.split("=", 1) if "=" in spec else (spec, spec)
        if (out / f"{name}.json").exists():
            refs.append((label, name, json.loads((out / f"{name}.json").read_text())))
    md = ["# P1 探针曲线", "",
          f"每个检查点在 `{args.probe_name}` 上的准确率（%）。不采样，最多生成 {MAX_TOKENS} 个词元；答案宽松地读（见 `parse_answer`），"
          "读不出的算错，另列\"读不出\"。\"不合规\"：回答不是题目要求的格式（只写选项字母，或只写 Yes / No），"
          "即使宽松地读得出也算；\"严格\"：合规并且答对。随机猜：是非题、左右、哪个更近 50%，遮挡 33%，其余四选一 25%，"
          "\"随机\"列是这份题按题数加权的随机猜准确率。", ""]
    curves: Dict[str, List[Tuple[int, Dict[str, Any]]]] = {}
    for cname, pts in chains:
        rows = []
        for model, step in pts:
            p = out / f"{model}.json"
            if not p.exists():
                continue
            rows.append((step, model, json.loads(p.read_text())))
        if not rows:
            continue
        curves[cname] = [(s, r) for s, _, r in rows]
        qtypes = sorted({q for _, _, r in rows for q in r["qtype"]})
        rows = [(label, name, r) for label, name, r in refs] + rows  # the fixed rows first, their label as the step
        md += [f"## {cname}", "",
               "| 步数 | 检查点 | 全部 | 随机 | 严格 | 读不出 | 不合规 | 其中：括号 / 带选项文字 | 推理 | 动作格式 | 状态题 | 前向题 | 逆向题 | "
               + " | ".join(qtypes) + " |",
               "| " + " | ".join("---" for _ in range(13 + len(qtypes))) + " |"]

        def pct(x):
            return "–" if x is None else f"{100 * x:.1f}"

        for step, model, r in rows:
            cls, a = r["qclass"], r["all"]
            fm = a.get("formats") or {}
            md.append(f"| {step} | {model} | {pct(a['accuracy'])} | {pct(a.get('chance'))} | {pct(a.get('strict_accuracy'))} | "
                      f"{pct(a['unreadable'])} | {pct(a.get('noncompliant'))} | {pct(fm.get('letter_variant'))} | "
                      f"{pct(fm.get('reasoning'))} | {pct(fm.get('action_tag'))} | "
                      + " | ".join(pct((cls.get(c) or {}).get("accuracy")) for c in ("state", "forward", "inverse"))
                      + " | " + " | ".join(pct((r["qtype"].get(q) or {}).get("accuracy")) for q in qtypes) + " |")
        md.append("")
    (out / "curve.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, len(curves), figsize=(4.2 * len(curves), 3.4), squeeze=False, sharey=True)
        for ax, (cname, pts) in zip(axes[0], curves.items()):
            steps = [s for s, _ in pts]
            for key, label, style in (("all", "all", "-"), ("state", "state", "--"), ("forward", "forward", "-."),
                                      ("inverse", "inverse", ":")):
                ys = [(r["all"] if key == "all" else r["qclass"].get(key, {})).get("accuracy") for _, r in pts]
                ax.plot(steps, [None if y is None else 100 * y for y in ys], style, marker="o", ms=3, label=label)
            for _, name, r in refs:  # the figure has no CJK font: the reference's name, not its label
                ax.axhline(100 * r["all"]["accuracy"], color="grey", ls=":", lw=1)
                ax.text(steps[0], 100 * r["all"]["accuracy"], f" {name}", fontsize=7, color="grey", va="bottom")
            chance = pts[0][1]["all"].get("chance")
            if chance is not None:
                ax.axhline(100 * chance, color="black", ls="--", lw=0.8)
                ax.text(steps[-1], 100 * chance, "chance ", fontsize=7, ha="right", va="bottom")
            ax.set_title(cname)
            ax.set_xlabel("training step")
            ax.grid(alpha=0.3)
        axes[0][0].set_ylabel("probe accuracy (%)")
        axes[0][0].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out / "curve.png", dpi=130)
    except Exception as exc:  # the table is the result; the figure is a convenience
        print(f"figure skipped: {exc}")
    print((out / "curve.md").read_text(encoding="utf-8")[:3000])


# =============================================================================
# comparing two groups (docs/transition_qa/PREREG_D_vs_C.md section 3.3)
# =============================================================================
GOAL_TYPES = ("goal_now", "goal_after")
OCC_TYPES = ("occluded_now", "occluded_after")
ANSWER_VALUES = {"goal": ("yes", "no"), "occlusion": ("full", "partial", "none")}  # the probe rows' answer_value
ALPHA = 0.05
MIN_DIFF = {"goal": 0.10, "occlusion": 0.20}  # the pooled difference a "difference" needs (user, 2026-10-21)
BOOT_REPS, BOOT_SEED = 2000, 20261021
TOL = 1e-9  # a difference this close to a threshold (or to 0) counts as on it


def mcnemar(only_a: int, only_b: int) -> float:
    """Exact two-sided McNemar test: the binomial test (p = 1/2) on the only_a + only_b discordant questions."""
    m = only_a + only_b
    if not m:
        return 1.0
    return min(1.0, 2 * sum(math.comb(m, i) for i in range(min(only_a, only_b) + 1)) / 2 ** m)


def same_sign(diffs: Sequence[float]) -> bool:
    """The differences all above 0 or all below; a difference of 0 has no sign."""
    return all(d > TOL for d in diffs) or all(d < -TOL for d in diffs)


def verdict(diff: float, diffs: Sequence[float], min_diff: float, p: Optional[float] = None) -> Dict[str, Any]:
    """"difference" iff |pooled difference| >= min_diff, ``diffs`` have the same sign (the two seeds' differences,
    pre-registration v3; one seed's S0 and D differences, v2), and p < ALPHA when a p value decides (goal; for
    occlusion the CI and p are only reported).  ``failed``: the conditions not met."""
    failed = []
    if p is not None and not p < ALPHA:
        failed.append("p")
    if abs(diff) < min_diff - TOL:
        failed.append("size")
    if not same_sign(diffs):
        failed.append("sign")
    return {"verdict": "no difference" if failed else "difference", "failed": failed, "same_sign": same_sign(diffs)}


def stratified_draws(sizes: Sequence[int], reps: int, seed: int) -> List[Any]:
    """Per answer value, a (reps, n_v) array of question indices drawn with replacement within that value, so every
    replicate keeps the n_v questions of each answer.  One draw serves both groups and every pair."""
    import numpy as np

    rng = np.random.default_rng(seed)
    return [rng.integers(0, n, size=(reps, n)) for n in sizes]


def _boot(d: Any) -> Dict[str, Any]:
    """The 95% percentile CI and the two-sided p = min(1, 2 min(P(d* <= 0), P(d* >= 0))) of bootstrap differences."""
    import numpy as np

    r = np.round(d, 12)  # a replicate within rounding of 0 counts as 0 on both sides
    return {"ci": [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
            "p": float(min(1.0, 2 * min((r <= 0).mean(), (r >= 0).mean())))}


def _acc(correct: Dict[str, bool], ids: Sequence[str]) -> Optional[float]:
    return sum(correct[s] for s in ids) / len(ids) if ids else None


def load_correct(out: Path, name: str, probe: Dict[str, Dict[str, Any]], qtypes: Sequence[str]) -> Dict[str, bool]:
    """sample_id -> correct for every probe question of ``qtypes``.  The result must have them all, with the probe's
    question type and answer, so a result from another probe version fails here."""
    lines = (out / f"{name}.jsonl").read_text(encoding="utf-8").splitlines()
    rows = {a["sample_id"]: a for a in (json.loads(l) for l in lines if l.strip())}
    need = [s for s, r in probe.items() if r["qtype"] in qtypes]
    missing = [s for s in need if s not in rows]
    if missing:
        raise ValueError(f"{name}: {len(missing)} of the {len(need)} probe questions missing (e.g. {missing[0]})")
    other = [s for s in need if rows[s].get("qtype") != probe[s]["qtype"]
             or rows[s].get("gold") != probe[s].get("answer", rows[s].get("gold"))]
    if other:
        raise ValueError(f"{name}: {len(other)} questions with another type or answer than the probe's (e.g. {other[0]})")
    return {s: bool(rows[s]["correct"]) for s in need}


Pair = Tuple[str, str, str, str, str]  # (label, a, b, seed, side)


def seed_groups(pairs: Sequence[Pair]) -> Dict[str, List[int]]:
    """{seed: the indices of its pairs} when the pairs come from two seeds, else {} (the sign is then over the pairs:
    one seed's S0 and D, or a single pair)."""
    groups: Dict[str, List[int]] = {}
    for i, p in enumerate(pairs):
        groups.setdefault(p[3], []).append(i)
    return groups if len(groups) > 1 else {}


def compare_mcnemar(res: Dict[str, Dict[str, bool]], pairs: Sequence[Pair], ids: Sequence[str],
                    value: Dict[str, str], values: Sequence[str]) -> Dict[str, Any]:
    """goal: per pair (label, a, b, seed, side) the accuracy difference a - b and the exact McNemar test; per seed
    (two seeds) the mean of its pairs' differences and McNemar on its pairs; pooled: the mean of all pair differences,
    and McNemar on the len(pairs) x n question-RL pairs.  The sign is over the seeds when there are two."""
    n = len(ids)
    if not n:
        raise ValueError("no probe questions of this type")
    by = {v: [s for s in ids if value[s] == v] for v in values}
    per = []
    for label, a, b, seed, side in pairs:
        only_a = sum(res[a][s] and not res[b][s] for s in ids)
        only_b = sum(res[b][s] and not res[a][s] for s in ids)
        per.append({"pair": label, "seed": seed, "side": side, "a": a, "b": b, "n": n,
                    "acc_a": _acc(res[a], ids), "acc_b": _acc(res[b], ids),
                    "by_answer_a": {v: _acc(res[a], by[v]) for v in values},
                    "by_answer_b": {v: _acc(res[b], by[v]) for v in values},
                    "diff": (only_a - only_b) / n, "only_a": only_a, "only_b": only_b,
                    "discordance": (only_a + only_b) / n, "p": mcnemar(only_a, only_b)})
    per_seed = []
    for seed, idx in seed_groups(pairs).items():
        oa, ob = sum(per[i]["only_a"] for i in idx), sum(per[i]["only_b"] for i in idx)
        per_seed.append({"seed": seed, "n": n * len(idx), "diff": sum(per[i]["diff"] for i in idx) / len(idx),
                         "only_a": oa, "only_b": ob, "discordance": (oa + ob) / (n * len(idx)), "p": mcnemar(oa, ob)})
    only_a, only_b = sum(r["only_a"] for r in per), sum(r["only_b"] for r in per)
    diffs = [r["diff"] for r in per]
    diff, p = sum(diffs) / len(diffs), mcnemar(only_a, only_b)
    signs = [s["diff"] for s in per_seed] or diffs
    pooled = {"n": n * len(per), "diff": diff, "only_a": only_a, "only_b": only_b,
              "discordance": (only_a + only_b) / (n * len(per)), "p": p, **verdict(diff, signs, MIN_DIFF["goal"], p)}
    return {"n": n, "n_by_answer": {v: len(by[v]) for v in values}, "per_pair": per, "per_seed": per_seed,
            "pooled": pooled}


def compare_balanced(res: Dict[str, Dict[str, bool]], pairs: Sequence[Pair], ids: Sequence[str],
                     value: Dict[str, str], values: Sequence[str], reps: int = BOOT_REPS,
                     seed: int = BOOT_SEED) -> Dict[str, Any]:
    """Occlusion: per pair the balanced-accuracy difference a - b (the mean of the per-answer accuracies), per seed
    (two seeds) and pooled the mean of the pair differences; each with a CI and p from the stratified paired
    bootstrap (``stratified_draws``, the same draw for every model).  The verdict does not use the CI or p; its sign
    is over the seeds when there are two."""
    import numpy as np

    strata = [[s for s in ids if value[s] == v] for v in values]
    if not all(strata):
        raise ValueError(f"an answer value without questions: {dict(zip(values, map(len, strata)))}")
    draws = stratified_draws([len(st) for st in strata], reps, seed)
    per, boots = [], []
    for label, a, b, seed_label, side in pairs:
        d = np.zeros(reps)
        for st, idx in zip(strata, draws):
            xa = np.array([res[a][s] for s in st], float)
            xb = np.array([res[b][s] for s in st], float)
            d += (xa[idx].mean(1) - xb[idx].mean(1)) / len(strata)
        boots.append(d)
        acc_a = {v: _acc(res[a], st) for v, st in zip(values, strata)}
        acc_b = {v: _acc(res[b], st) for v, st in zip(values, strata)}
        bal_a, bal_b = sum(acc_a.values()) / len(values), sum(acc_b.values()) / len(values)
        only_a = sum(res[a][s] and not res[b][s] for s in ids)
        only_b = sum(res[b][s] and not res[a][s] for s in ids)
        per.append({"pair": label, "seed": seed_label, "side": side, "a": a, "b": b, "n": len(ids),
                    "bal_a": bal_a, "bal_b": bal_b, "by_answer_a": acc_a, "by_answer_b": acc_b,
                    "diff": bal_a - bal_b, "only_a": only_a, "only_b": only_b,
                    "discordance": (only_a + only_b) / len(ids), **_boot(d)})
    per_seed = []
    for seed_label, idx in seed_groups(pairs).items():
        oa, ob = sum(per[i]["only_a"] for i in idx), sum(per[i]["only_b"] for i in idx)
        per_seed.append({"seed": seed_label, "n": len(ids) * len(idx),
                         "diff": sum(per[i]["diff"] for i in idx) / len(idx), "only_a": oa, "only_b": ob,
                         "discordance": (oa + ob) / (len(ids) * len(idx)),
                         **_boot(sum(boots[i] for i in idx) / len(idx))})
    diffs = [r["diff"] for r in per]
    diff = sum(diffs) / len(diffs)
    only_a, only_b = sum(r["only_a"] for r in per), sum(r["only_b"] for r in per)
    signs = [s["diff"] for s in per_seed] or diffs
    pooled = {"n": len(ids) * len(per), "diff": diff, "only_a": only_a, "only_b": only_b,
              "discordance": (only_a + only_b) / (len(ids) * len(per)), **_boot(sum(boots) / len(boots)),
              **verdict(diff, signs, MIN_DIFF["occlusion"])}
    return {"n": len(ids), "n_by_answer": {v: len(st) for v, st in zip(values, strata)}, "per_pair": per,
            "per_seed": per_seed, "pooled": pooled}


Seeds = List[Tuple[str, List[str]]]  # [(seed label, [S0 result, D result] or [one result])]
SEED_LABEL = re.compile(r"^([\w.-]+):(.+)$")


def parse_group(spec: str) -> Tuple[str, Seeds]:
    """[GROUP=]SEED[;SEED] -> (group name, [(seed label, result names)]).  A seed is [LABEL:]S0_RESULT,D_RESULT;
    two seeds (pre-registration v3) both need two results, unlabelled ones are seed1, seed2.  One seed may also be a
    single result (for checks).  No GROUP: the result names.  A result name may carry a folder (relative to OUT, or
    absolute) when the groups' results lie in different folders."""
    name, rest = spec.split("=", 1) if "=" in spec else ("", spec)
    seeds: Seeds = []
    for part in (p for p in rest.split(";") if p):
        m = SEED_LABEL.match(part)
        label, models = (m.group(1), m.group(2)) if m else ("", part)
        seeds.append((label, [x for x in models.split(",") if x]))
    if len(seeds) == 1 and len(seeds[0][1]) not in (1, 2):
        raise ValueError(f"a seed is one result or two (S0 RL, D RL): {spec}")
    if len(seeds) == 2:
        if any(len(models) != 2 for _, models in seeds):
            raise ValueError(f"with two seeds every seed is two results (S0 RL, D RL): {spec}")
        seeds = [(label or f"seed{i + 1}", models) for i, (label, models) in enumerate(seeds)]
        if seeds[0][0] == seeds[1][0]:
            raise ValueError(f"two seeds with one label: {spec}")
    if not 1 <= len(seeds) <= 2:
        raise ValueError(f"a group is one seed or two: {spec}")
    results = [x for _, models in seeds for x in models]
    return name or "-".join(Path(m).name for m in results), seeds


def make_pairs(a: Sequence[str], b: Sequence[str]) -> List[Tuple[str, str, str]]:
    """One seed each, (label, a, b): S0 against S0 and D against D; a single result serves both pairs; one result
    each: one pair."""
    if max(len(a), len(b)) == 1:
        return [("single", a[0], b[0])]
    return [(label, a[min(i, len(a) - 1)], b[min(i, len(b) - 1)]) for i, label in enumerate(("S0", "D"))]


def make_seed_pairs(a: Seeds, b: Seeds) -> List[Pair]:
    """(label, a, b, seed, side): the same seed and the same side.  One seed each: ``make_pairs``.  Two seeds each:
    the seed labels must agree, the labels are SEED/S0 and SEED/D.  A single result against two seeds serves every
    pair; one seed against two is refused (the pairs go by seed)."""
    def single(s: Seeds) -> bool:
        return len(s) == 1 and len(s[0][1]) == 1

    if len(a) == 1 and len(b) == 1:
        if a[0][0] and b[0][0] and a[0][0] != b[0][0]:
            raise ValueError(f"the seeds do not match: {a[0][0]} against {b[0][0]}")
        seed = a[0][0] or b[0][0]
        return [(label, x, y, seed, label) for label, x, y in make_pairs(a[0][1], b[0][1])]
    if single(a) or single(b):
        one, many = (a, b) if single(a) else (b, a)
        out = []
        for seed, models in many:
            for side, m in zip(("S0", "D"), models):
                x, y = (one[0][1][0], m) if single(a) else (m, one[0][1][0])
                out.append((f"{seed}/{side}", x, y, seed, side))
        return out
    if len(a) != len(b):
        raise ValueError(f"the pairs go by seed: {len(a)} seed(s) against {len(b)}")
    out = []
    for (sa, ma), (sb, mb) in zip(a, b):
        if sa != sb:
            raise ValueError(f"the seeds do not match: {sa} against {sb}")
        out += [(f"{sa}/{side}", x, y, sa, side) for side, x, y in zip(("S0", "D"), ma, mb)]
    return out


def _as_seeds(group: Tuple[str, Any]) -> Tuple[str, Seeds]:
    """A group as (name, result names) - one seed - or as (name, seeds)."""
    name, x = group
    return (name, [("", list(x))]) if x and isinstance(x[0], str) else (name, list(x))


def compare(probe_path: Path, out: Path, a: Tuple[str, Any], b: Tuple[str, Any], label: str,
            reps: int = BOOT_REPS, seed: int = BOOT_SEED) -> Dict[str, Any]:
    """Group a - group b on the probe, as section 3.3 of the pre-registration has it (the module docstring).  A group
    is (name, result names) for one seed or (name, [(seed label, result names)])."""
    a, b = _as_seeds(a), _as_seeds(b)
    lines = Path(probe_path).read_text(encoding="utf-8").splitlines()
    probe = {r["sample_id"]: r for r in (json.loads(l) for l in lines if l.strip())}
    value = {s: r.get("answer_value") for s, r in probe.items()}
    pairs = make_seed_pairs(a[1], b[1])
    models = {k: [m for _, ms in g[1] for m in ms] for k, g in (("a", a), ("b", b))}
    res = {m: load_correct(Path(out), m, probe, GOAL_TYPES + OCC_TYPES) for m in dict.fromkeys(models["a"] + models["b"])}
    over = "seeds" if seed_groups(pairs) else "sides" if len(pairs) > 1 else "single"

    def ids(types: Sequence[str]) -> List[str]:
        return [s for s, r in probe.items() if r["qtype"] in types]

    result: Dict[str, Any] = {
        "label": label, "probe": str(probe_path), "out": str(out),
        "groups": {k: {"name": g[0], "results": models[k], "seeds": [{"seed": s, "results": ms} for s, ms in g[1]]}
                   for k, g in (("a", a), ("b", b))},
        "pairs": [{"pair": p, "a": x, "b": y, "seed": s, "side": side} for p, x, y, s, side in pairs],
        "rules": {"alpha": ALPHA, "min_diff": MIN_DIFF, "reps": reps, "seed": seed, "same_sign_over": over}}
    for q in GOAL_TYPES:
        result[q] = compare_mcnemar(res, pairs, ids((q,)), value, ANSWER_VALUES["goal"])
    result["occlusion"] = {"qtypes": list(OCC_TYPES), **compare_balanced(
        res, pairs, ids(OCC_TYPES), value, ANSWER_VALUES["occlusion"], reps, seed)}
    return result


VERDICT_ZH = {"difference": "**有差异**", "no difference": "没有差异"}
PAIR_ZH = {"S0": "S0", "D": "D", "single": "单对"}
# what "same sign" is over (rules.same_sign_over): the two seeds (pre-registration v3), one seed's two sides (v2),
# or a single pair (checks)
SIGN_RULE_ZH = {"seeds": "两个种子的差同号（每个种子的差是它 S0、D 两对的平均；预登记第三版，用户 2026-10-30）",
                "sides": "S0、D 两对的差同号（只有一个种子：预登记第二版的规则）",
                "single": "差不为 0（只有一对，检查用）"}
SIGN_FAIL_ZH = {"seeds": "两个种子的差不同号或为 0", "sides": "S0 和 D 的差不同号或为 0", "single": "差为 0"}


def _pp(x: Optional[float], signed: bool = False) -> str:
    return "–" if x is None else f"{100 * x:+.1f}" if signed else f"{100 * x:.1f}"


def _p(p: float) -> str:
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def _pair_name(x: Dict[str, Any]) -> str:
    return f"种子 {x['seed']} {PAIR_ZH[x['side']]}" if "/" in x["pair"] else PAIR_ZH[x["pair"]]


def _why(pooled: Dict[str, Any], kind: str, over: str) -> str:
    text = {"p": "p ≥ 0.05", "size": f"|差| < {100 * MIN_DIFF[kind]:.0f} 个百分点", "sign": SIGN_FAIL_ZH[over]}
    return "；没达到：" + "、".join(text[f] for f in pooled["failed"]) if pooled["failed"] else ""


def compare_markdown(r: Dict[str, Any]) -> str:
    A, B = r["groups"]["a"]["name"], r["groups"]["b"]["name"]
    over = r["rules"].get("same_sign_over") or ("single" if len(r["pairs"]) == 1 else "sides")
    occ = r["occlusion"]
    nv, rules = occ["n_by_answer"], r["rules"]

    def detail(m: Dict[str, Any]) -> str:
        """The conclusion line's parts: per seed (with its two pairs), or per pair."""
        if over == "single":
            return ""
        if over == "sides":
            return "；" + "、".join(f"{PAIR_ZH[x['pair']]} {_pp(x['diff'], True)}" for x in m["per_pair"])
        return "；" + "、".join(
            f"种子 {s['seed']} {_pp(s['diff'], True)}（" + "、".join(
                f"{PAIR_ZH[x['side']]} {_pp(x['diff'], True)}" for x in m["per_pair"] if x["seed"] == s["seed"]) + "）"
            for s in m["per_seed"])

    sign_rule = SIGN_RULE_ZH[over]
    md = [f"# 探针比较：{A} − {B}（`{r['label']}`）", "",
          f"`python -m eval.diagnostics.qa_probe compare`，按 `docs/transition_qa/PREREG_D_vs_C.md` 第 3.3 节。"
          f"题集 `{r['probe']}`，回答在 `{r['out']}`。", "",
          "- 配对（同一种子、同一侧）：" + "；".join(f"{_pair_name(p)}：`{p['a']}` 对 `{p['b']}`" for p in r["pairs"]) + "。",
          "- goal 两类各自比：每对的准确率差，McNemar 精确检验（只看一对一错的题，双侧二项检验）；合起来的差是各对的差的平均，"
          f"McNemar 用 {len(r['pairs'])} × n 个\"题 × RL\"" + ("；每个种子另报它两对的平均和 McNemar" if over == "seeds" else "")
          + f"。判定\"有差异\"：合起来 p < {ALPHA}，且 |差| ≥ {100 * MIN_DIFF['goal']:.0f} 个百分点，且{sign_rule}。",
          f"- 遮挡两类合起来（{occ['n']} 道：完全 / 部分 / 没有 {nv['full']} / {nv['partial']} / {nv['none']} 道）比平衡准确率"
          f"（三种答案各自准确率的平均，随机 33.3%）。95% 置信区间和双侧 p 来自自助法（{rules['reps']:,} 次，种子 {rules['seed']}，"
          "在每种答案里按题重抽，两组、各对用同一组抽样）。判定\"有差异\"：|合起来的差| ≥ "
          f"{100 * MIN_DIFF['occlusion']:.0f} 个百分点，且{sign_rule}；置信区间和 p 照报，不参与判定。",
          "- 不一致率：一边对、一边错的题占的比例。"]
    md += ["", "## 结论", ""]
    for q in GOAL_TYPES:
        pl = r[q]["pooled"]
        md.append(f"- `{q}`：{VERDICT_ZH[pl['verdict']]}（{A} − {B} {_pp(pl['diff'], True)} 个百分点，McNemar p = "
                  f"{_p(pl['p'])}{detail(r[q])}{_why(pl, 'goal', over)}）")
    pl = occ["pooled"]
    md += [f"- 遮挡两类合起来的平衡准确率：{VERDICT_ZH[pl['verdict']]}（{A} − {B} {_pp(pl['diff'], True)} 个百分点，"
           f"95% 置信区间 {_pp(pl['ci'][0], True)} 到 {_pp(pl['ci'][1], True)}，p = {_p(pl['p'])}"
           f"{detail(occ)}{_why(pl, 'occlusion', over)}）", "",
           "## goal 两类：准确率（%）", "",
           "括号里是答案为 Yes / No 时的准确率。", "",
           f"| 题类 | 配对 | 题数 | {A}（Yes / No） | {B}（Yes / No） | {A} − {B} | 只有 {A} 对 / 只有 {B} 对 | 不一致率 | "
           "McNemar p | 判定 |",
           "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]

    def cell(acc: Optional[float], by: Dict[str, Optional[float]]) -> str:
        return f"{_pp(acc)}（{' / '.join(_pp(x) for x in by.values())}）"

    for q in GOAL_TYPES:
        for x in r[q]["per_pair"]:
            md.append(f"| `{q}` | {_pair_name(x)} | {x['n']} | {cell(x['acc_a'], x['by_answer_a'])} | "
                      f"{cell(x['acc_b'], x['by_answer_b'])} | {_pp(x['diff'], True)} | {x['only_a']} / {x['only_b']} | "
                      f"{_pp(x['discordance'])} | {_p(x['p'])} | |")
        for s in r[q].get("per_seed") or []:
            md.append(f"| `{q}` | 种子 {s['seed']}（两对） | {s['n']} | – | – | {_pp(s['diff'], True)} | "
                      f"{s['only_a']} / {s['only_b']} | {_pp(s['discordance'])} | {_p(s['p'])} | |")
        pl = r[q]["pooled"]
        md.append(f"| `{q}` | 合起来 | {pl['n']} | – | – | {_pp(pl['diff'], True)} | {pl['only_a']} / {pl['only_b']} | "
                  f"{_pp(pl['discordance'])} | {_p(pl['p'])} | {VERDICT_ZH[pl['verdict']]} |")
    md += ["", "## 遮挡两类合起来：平衡准确率（%）", "",
           "括号里是答案为完全 / 部分 / 没有时的准确率；单个配对的置信区间和 p 只作参考。", "",
           f"| 配对 | 题数 | {A}（完全 / 部分 / 没有） | {B}（完全 / 部分 / 没有） | {A} − {B} | 95% 置信区间 | p | "
           f"只有 {A} 对 / 只有 {B} 对 | 不一致率 | 判定 |",
           "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for x in occ["per_pair"]:
        md.append(f"| {_pair_name(x)} | {x['n']} | {cell(x['bal_a'], x['by_answer_a'])} | "
                  f"{cell(x['bal_b'], x['by_answer_b'])} | {_pp(x['diff'], True)} | "
                  f"{_pp(x['ci'][0], True)} 到 {_pp(x['ci'][1], True)} | {_p(x['p'])} | {x['only_a']} / {x['only_b']} | "
                  f"{_pp(x['discordance'])} | |")
    for s in occ.get("per_seed") or []:
        md.append(f"| 种子 {s['seed']}（两对） | {s['n']} | – | – | {_pp(s['diff'], True)} | {_pp(s['ci'][0], True)} 到 "
                  f"{_pp(s['ci'][1], True)} | {_p(s['p'])} | {s['only_a']} / {s['only_b']} | {_pp(s['discordance'])} | |")
    po = occ["pooled"]
    md.append(f"| 合起来 | {po['n']} | – | – | {_pp(po['diff'], True)} | {_pp(po['ci'][0], True)} 到 "
              f"{_pp(po['ci'][1], True)} | {_p(po['p'])} | {po['only_a']} / {po['only_b']} | {_pp(po['discordance'])} | "
              f"{VERDICT_ZH[po['verdict']]} |")
    return "\n".join(md) + "\n"


def cmd_compare(args: argparse.Namespace) -> None:
    out = Path(args.out)
    a, b = parse_group(args.a), parse_group(args.b)
    label = args.label or f"{a[0]}_vs_{b[0]}"
    result = compare(Path(args.probe), out, a, b, label, args.reps, args.seed)
    (out / f"compare_{label}.json").write_text(json.dumps(result, indent=1, ensure_ascii=False), encoding="utf-8")
    md = compare_markdown(result)
    (out / f"compare_{label}.md").write_text(md, encoding="utf-8")
    print(md.split("## goal")[0])


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--probe", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--model", action="append", default=[], help="NAME=PATH (repeatable)")
    r.add_argument("--from-eval", default="", help="an evaluation folder: the models of its result folders")
    r.add_argument("--names", default="", help="with --from-eval: comma-separated result folder names")
    r.add_argument("--no-images", action="store_true", help="the text-only baseline: stem and options, no images")
    r.add_argument("--in-process", action="store_true", help=argparse.SUPPRESS)
    r.set_defaults(func=cmd_run)
    x = sub.add_parser("confusion", help="action_between: the true action against the kind of option chosen")
    x.add_argument("--probe", required=True)
    x.add_argument("--out", required=True)
    x.add_argument("--names", required=True, help="comma-separated result names in OUT")
    x.set_defaults(func=cmd_confusion)
    s = sub.add_parser("rescore")
    s.add_argument("--probe", required=True)
    s.add_argument("--out", required=True)
    s.set_defaults(func=cmd_rescore)
    c = sub.add_parser("curve")
    c.add_argument("--out", required=True)
    c.add_argument("--chain", action="append", required=True)
    c.add_argument("--probe-name", default="p1_probe.jsonl", help="how the table names the probe set")
    c.add_argument("--reference", action="append", default=[], help="NAME=LABEL: a fixed row (e.g. the text-only baseline)")
    c.set_defaults(func=cmd_curve)
    k = sub.add_parser("compare", help="group A - group B, as PREREG_D_vs_C.md section 3.3 has it")
    k.add_argument("--probe", required=True)
    k.add_argument("--out", required=True, help="the folder with the result files; compare_<label>.json / .md go here")
    k.add_argument("--a", required=True, help="[GROUP=]SEED[;SEED], a seed being [LABEL:]S0_RESULT,D_RESULT "
                                              "(two seeds: pre-registration v3), or [GROUP=]RESULT (result names in "
                                              "OUT; a name may carry a folder, relative to OUT or absolute)")
    k.add_argument("--b", required=True, help="as --a; the differences are A - B")
    k.add_argument("--label", default="", help="the output name (default: A's name _vs_ B's name)")
    k.add_argument("--reps", type=int, default=BOOT_REPS, help="bootstrap replicates")
    k.add_argument("--seed", type=int, default=BOOT_SEED, help="bootstrap seed")
    k.set_defaults(func=cmd_compare)
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
