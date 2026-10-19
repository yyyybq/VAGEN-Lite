"""Diagnostic 3: accuracy on the P1 probe questions, per question type, over a run's checkpoints.

    # one card: every model in turn (a separate process per model, so vLLM frees the card in between)
    python -m eval.diagnostics.qa_probe run --probe p1_probe.jsonl --out OUT \
        --model rs0_050=/path/to/global_step_50/actor/huggingface --model ...
    python -m eval.diagnostics.qa_probe run --probe ... --out OUT --from-eval EVAL_DIR --names rs0_050,rs0_100
    # table and figure: checkpoint step -> accuracy per question type
    python -m eval.diagnostics.qa_probe curve --out OUT --chain "R11-SFT=sft_formal_e{1..10}@154" ...

``run`` asks each probe question exactly as the training row has it (no system prompt, the images where the
``<image>`` markers stand), greedy, at most 256 new tokens, and reads the answer leniently (``parse_answer``: a
yes / no, an option letter in several spellings, the option's text, number or key word, after a <think> block).
An answer it cannot read counts as wrong and is reported as unreadable, because the action-only models answer in
their action format.  Separately, an answer not in the asked format (the letter alone, or Yes / No alone) counts as
non-compliant even when it can be read (user, 2026-10-19: lenient reading, the format non-compliance rate on its own).
OUT/<name>.jsonl has every answer, OUT/<name>.json the accuracies.

``curve``: a chain is NAME=MODELS@STEPS_PER_MODEL or NAME=m1:s1,m2:s2,... (model name : training step); it writes
OUT/curve.md (one table per chain: step, overall, chance, strict, unreadable, non-compliant, per class, per question
type) and OUT/curve.png.
"""

from __future__ import annotations

import argparse
import base64
import collections
import json
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


def score(rows: Sequence[Dict[str, Any]], outputs: Sequence[str]) -> List[Dict[str, Any]]:
    out = []
    for r, text in zip(rows, outputs):
        gold = r["messages"][1]["content"]
        yes_no = gold in ("Yes", "No")
        got = parse_answer(text, r.get("options") or [], yes_no)
        out.append({"sample_id": r["sample_id"], "qtype": r["qtype"], "qclass": r["qclass"], "family": r["family"],
                    "gold": gold, "output": text, "parsed": got, "correct": got == gold, "readable": got is not None,
                    "compliant": compliant(text, r.get("options") or [], yes_no),
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
                "chance": sum(1.0 / a["n_options"] for a in sel) / n if all(a.get("n_options") for a in sel) else None}
    out = {"all": acc(answers)}
    for key in ("qclass", "qtype"):
        groups: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
        for a in answers:
            groups[a[key]].append(a)
        out[key] = {k: acc(v) for k, v in sorted(groups.items())}
    return out


# =============================================================================
# running a model (one card)
# =============================================================================
def _image_part(path: str) -> Dict[str, Any]:
    data = base64.b64encode(Path(path).read_bytes()).decode()
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + data}}


def chat(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    """The probe row's user turn with its images where the <image> markers stand."""
    text = row["messages"][0]["content"]
    parts = text.split("<image>")
    if len(parts) - 1 != len(row["images"]):
        raise ValueError(f"{row['sample_id']}: {len(parts) - 1} <image> markers, {len(row['images'])} images")
    content: List[Dict[str, Any]] = []
    for i, piece in enumerate(parts):
        if piece:
            content.append({"type": "text", "text": piece})
        if i < len(row["images"]):
            content.append(_image_part(row["images"][i]))
    return [{"role": "user", "content": content}]


def run_one(name: str, model: str, probe: Path, out: Path) -> None:  # pragma: no cover - needs a GPU
    from vllm import LLM, SamplingParams

    rows = [json.loads(l) for l in probe.read_text().splitlines() if l.strip()]
    t0 = time.time()
    llm = LLM(model=model, gpu_memory_utilization=0.85, max_model_len=4096, enforce_eager=True,
              limit_mm_per_prompt={"image": 2}, trust_remote_code=False)
    load = time.time() - t0
    params = SamplingParams(temperature=0.0, max_tokens=MAX_TOKENS)
    res = llm.chat([chat(r) for r in rows], sampling_params=params, use_tqdm=False)
    answers = score(rows, [r.outputs[0].text for r in res])
    with open(out / f"{name}.jsonl", "w", encoding="utf-8") as f:
        for a in answers:
            f.write(json.dumps(a, ensure_ascii=False) + "\n")
    (out / f"{name}.json").write_text(json.dumps({"name": name, "model": model, "probe": str(probe),
                                                  "load_seconds": round(load, 1),
                                                  "answer_seconds": round(time.time() - t0 - load, 1),
                                                  **summary(answers)}, indent=1))


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
            run_one(name, model, Path(args.probe), out)
        else:  # a fresh process per model: vLLM releases the card only when its process ends
            cmd = [sys.executable, "-m", "eval.diagnostics.qa_probe", "run", "--probe", args.probe, "--out", str(out),
                   "--model", f"{name}={model}", "--in-process"]
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
        md += [f"## {cname}", "",
               "| 步数 | 检查点 | 全部 | 随机 | 严格 | 读不出 | 不合规 | 状态题 | 前向题 | 逆向题 | " + " | ".join(qtypes) + " |",
               "| " + " | ".join("---" for _ in range(10 + len(qtypes))) + " |"]

        def pct(x):
            return "–" if x is None else f"{100 * x:.1f}"

        for step, model, r in rows:
            cls, a = r["qclass"], r["all"]
            md.append(f"| {step} | {model} | {pct(a['accuracy'])} | {pct(a.get('chance'))} | {pct(a.get('strict_accuracy'))} | "
                      f"{pct(a['unreadable'])} | {pct(a.get('noncompliant'))} | "
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


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--probe", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--model", action="append", default=[], help="NAME=PATH (repeatable)")
    r.add_argument("--from-eval", default="", help="an evaluation folder: the models of its result folders")
    r.add_argument("--names", default="", help="with --from-eval: comma-separated result folder names")
    r.add_argument("--in-process", action="store_true", help=argparse.SUPPRESS)
    r.set_defaults(func=cmd_run)
    c = sub.add_parser("curve")
    c.add_argument("--out", required=True)
    c.add_argument("--chain", action="append", required=True)
    c.add_argument("--probe-name", default="p1_probe.jsonl", help="how the table names the probe set")
    c.set_defaults(func=cmd_curve)
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
