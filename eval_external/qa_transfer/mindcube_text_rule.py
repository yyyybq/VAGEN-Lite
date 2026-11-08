"""MindCube diagnostic (decided 2026-10-20): one rule for every model, applied offline to the saved answers; the
official score stays VLMEvalKit's rule-based one.

    python -m eval_external.qa_transfer.mindcube_text_rule --stages step1 step2 --out eval_external/results/mindcube_text_rule

The rule: an answer that is not an option letter, but whose text is exactly the text of one option, counts as
that option.  "Answer" is the content of <answer>...</answer> when the reply has that block, else the whole reply;
"is an option letter" means it is a letter on its own or starts with one followed by ". ", ")" or ":" ("A. Yes").
"Exactly the text" compares after stripping surrounding spaces and one trailing period, case-sensitively.  Every
other answer keeps the official result.

Why it matters: VLMEvalKit's extractor (matching_func.can_match_option, step 2) reads an <answer> block that starts
with a capital A-J as that letter, so "<answer>Directly right</answer>" counts as D and "<answer>Blue bin</answer>"
as B, whatever the options are.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

from eval_external.qa_transfer.collect import QA_RUNS

BENCHES = ("MindCube", "MindCube-Tiny")
_BLOCK = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.S | re.I)
_LETTER = re.compile(r"\(?[A-J]\)?(?:$|[.):]\s*.*)", re.S)
_OPTION = re.compile(r"\s*([A-J])[.)]\s*(.*)", re.S)


def answer_text(prediction: str) -> str:
    m = _BLOCK.search(prediction or "")
    return (m.group(1) if m else (prediction or "")).strip()


def rule_letter(prediction: str, candidates: list[str]) -> str | None:
    """The option the rule picks, or None when it does not apply (a letter answer, or no unique exact match)."""
    text = answer_text(prediction)
    if _LETTER.fullmatch(text):
        return None
    text = text[:-1].strip() if text.endswith(".") else text
    options = {}
    for c in candidates:
        m = _OPTION.fullmatch(str(c))
        if m:
            options[m.group(1)] = m.group(2).strip()
    hits = [k for k, v in options.items() if v == text]
    return hits[0] if len(hits) == 1 else None


def rescore(xlsx: Path) -> dict:
    import pandas as pd

    df = pd.read_excel(xlsx)
    n = len(df)
    official = applied = 0
    diag = 0.0
    for _, r in df.iterrows():
        hit = float(r["hit"])
        cands = ast.literal_eval(r["candidates"]) if isinstance(r["candidates"], str) else []
        letter = rule_letter(str(r["prediction"]), cands)
        official += hit
        if letter is not None:
            applied += 1
            diag += float(letter == str(r["answer"]).strip())
        else:
            diag += hit
    return {"n": n, "official_acc": round(100 * official / n, 2), "rule_applied": applied,
            "rule_applied_pct": round(100 * applied / n, 2), "diagnostic_acc": round(100 * diag / n, 2)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stages", nargs="+", default=["step1", "step2"])
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    rows, seen = [], set()
    for stage in args.stages:
        for rd in sorted((QA_RUNS / stage).iterdir()):
            res = rd / "results.json"
            if rd.name.startswith("_") or not res.exists() or rd.resolve() in seen:
                continue
            r = json.loads(res.read_text())
            if r["bench"] not in BENCHES:
                continue
            seen.add(rd.resolve())
            xlsx = Path(next(p for p in r["files"] if p.endswith(".xlsx")))
            rows.append({"model": r["model"], "bench": r["bench"], "reported_overall": r["overall"], **rescore(xlsx)})
    lines = ["| 模型 | 题集 | 题数 | 正式判分（逐题平均） | 规则改判的回答 | 加规则后（诊断） | 差 |", "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['model']} | {r['bench']} | {r['n']} | {r['official_acc']} | {r['rule_applied']}"
                     f"（{r['rule_applied_pct']}%） | {r['diagnostic_acc']} | {r['diagnostic_acc'] - r['official_acc']:+.1f} |")
    print("\n".join(lines))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.with_suffix(".json").write_text(json.dumps(rows, indent=1) + "\n")
        args.out.with_suffix(".md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
