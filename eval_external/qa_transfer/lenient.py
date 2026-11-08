"""Lenient re-scoring of a finished QA run, item by item, so that "could not be parsed" and "answered wrong" are
counted apart (task of 2026-11-05: models trained on actions tend to answer "(B)" or with an option's text, which a
backend's own letter extraction scores as wrong).

    python -m eval_external.qa_transfer.lenient RUN_DIR [RUN_DIR ...]     # writes RUN_DIR/lenient.json, items_lenient.jsonl

One rule for every model; per item:
  1. If the backend's own extraction found a valid option letter (multiple choice) or a number (numeric), the
     backend's score stands (status "official").  So the lenient score is never below the official one.
  2. Otherwise, multiple choice: an option letter written another way ("(B)", "[B]", "B) ...", "Answer: B", "the
     answer is (B)", a single "(B)" anywhere), status "letter"; else the text of exactly one option (equal after
     normalising case, punctuation and spaces, or the only option text contained in the answer, a longer option
     that contains a shorter one winning), status "text".  Numeric: the first number in the answer (digits or a
     number word up to twenty), scored by the backend's own MRA formula, status "number".
  3. Nothing found: status "unparseable", score 0.
The answer part is the text inside the last <answer>...</answer> if there is one, else the reply without <think>.
Headlines are recomputed the benchmark's way (paired.pick_rule, checked on the official item scores: the rule must
reproduce the run's official overall within 0.05 points).

lenient.json: overall and per-type scores (lenient and official), counts per status, multiple-choice items wrong
after a parse, and two checks: the official scores re-read per item reproduce the official overall, and for
numeric items the official extraction rescored by our MRA copy equals the official item score.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from eval_external.qa_transfer.benchmarks import BENCHMARKS
from eval_external.qa_transfer.parse_check import WORDS, _core

LETTERS = "ABCDEFGHIJ"
VSTI_MCQ = {"obj_obj_relative_pos_nf", "obj_obj_relative_pos_ud", "obj_obj_relative_pos_lr", "camera_obj_rel_dist_v1",
            "camera_obj_rel_dist_v2", "camera_obj_rel_dist_v3", "camera_movement_direction"}


# ---------- the parsing rule ----------

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", str(s).lower())).strip()


def strip_label(option: str) -> str:
    """'A. back-left' -> 'back-left'."""
    return re.sub(r"^\s*\(?[A-J][\.\):：]\s*", "", str(option)).strip()


def letter_forms(core: str, letters: str) -> str | None:
    text = core.strip()
    m = re.fullmatch(r"[\s\(\[\{*`'\"]*([A-J])[\s\)\]\}\.:：*`'\"]*", text)
    if m and m.group(1) in letters:
        return m.group(1)
    m = re.match(r"\s*(?:\(([A-J])\)|\[([A-J])\]|([A-J])\s*[\.\):：])(?=\s|$)", text)
    if m:
        g = next(x for x in m.groups() if x)
        if g in letters:
            return g
    found = re.findall(r"(?:answer|option|choice)\s*(?:is|:|：)?\s*[\(\[]?([A-J])[\)\]]?(?=$|[\s\.\),:;])", text,
                       flags=re.I)
    found = [f.upper() for f in found if f.upper() in letters]
    if found:
        return found[-1]
    paren = {g for g in re.findall(r"\(([A-J])\)", text) if g in letters}
    if len(paren) == 1:
        return paren.pop()
    return None


def text_match(core: str, options: list[str]) -> int | None:
    ans = _norm(core)
    opts = [_norm(strip_label(o)) for o in options]
    if not ans:
        return None
    equal = [i for i, o in enumerate(opts) if o and o == ans]
    if len(equal) == 1:
        return equal[0]
    inside = [i for i, o in enumerate(opts) if o and re.search(rf"(?:^| ){re.escape(o)}(?: |$)", ans)]
    if len(inside) == 1:
        return inside[0]
    if inside:
        longest = max(inside, key=lambda i: len(opts[i]))
        if all(opts[i] in opts[longest] for i in inside):
            return longest
    return None


def lenient_letter(pred: str, options: list[str]) -> tuple[str | None, str]:
    letters = LETTERS[:len(options)]
    core = _core(str(pred))
    got = letter_forms(core, letters)
    if got:
        return got, "letter"
    i = text_match(core, options)
    if i is not None:
        return letters[i], "text"
    return None, "unparseable"


def first_number(pred: str) -> float | None:
    core = _core(str(pred))
    m = re.search(r"-?\d+(?:\.\d+)?", core.replace(",", ""))
    if m:
        return float(m.group(0))
    for w in re.findall(r"[a-z]+", core.lower()):
        if w in WORDS:
            return float(WORDS[w])
    return None


# ---------- the backends' MRA formulas (copied; checked against the official item scores every run) ----------

def mra_vlmeval(pred: float, target: float) -> float:
    """vlmeval/dataset/utils/spatial_bench/cal_scores.py mean_relative_accuracy(start .5, end .95, interval .05)."""
    num_pts = int((0.95 - 0.5) / 0.05 + 2)
    conf = np.linspace(0.5, 0.95, num_pts)
    err = abs(pred - target) if target == 0.0 else abs((pred - target) / target)
    return float((err <= (1 - conf)).astype(float).mean())


def mra_vsti(pred: float, target: float) -> float:
    """VLM-3R vstibench utils.py mean_relative_accuracy(start .5, end .95, interval .05)."""
    num_pts = (0.95 - 0.5) / 0.05 + 2
    conf = np.linspace(0.5, 0.95, int(num_pts))
    return float((abs(pred - target) / target <= 1 - conf).mean())


def _num(x) -> float | None:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


# ---------- reading a run's items ----------

def _options_vlmeval(rec: dict) -> list[str]:
    cands = rec.get("candidates")
    if isinstance(cands, str) and cands.strip():
        s = cands.strip()
        if s.startswith("["):
            return [str(c) for c in ast.literal_eval(s)]
        return [line for line in s.split("\n") if line.strip()]
    return [str(rec[c]) for c in LETTERS if c in rec and isinstance(rec[c], str)]


def items_vlmeval(xlsx: str) -> list[dict]:
    import pandas as pd

    out = []
    for _, x in pd.read_excel(xlsx).iterrows():
        rec = x.to_dict()
        qtype = str(rec.get("question_type", rec.get("category", "all")))
        mcq = rec.get("task_type", "MCQ") == "MCQ"
        pred = "" if rec.get("prediction") is None or (isinstance(rec.get("prediction"), float)
                                                        and math.isnan(rec["prediction"])) else str(rec["prediction"])
        if mcq:
            options = _options_vlmeval(rec)
            letters = LETTERS[:len(options)]
            gt = str(rec["answer"]).strip()[:1].upper()
            ext = rec.get("pred_extracted")
            parsed = isinstance(ext, str) and ext.strip().upper() in letters and len(ext.strip()) == 1
            official = float(rec["hit"])
            out.append({"id": str(rec["index"]), "type": qtype, "mcq": True, "options": options, "answer": gt,
                        "pred": pred, "official": official, "official_parsed": parsed,
                        "official_letter": ext.strip().upper() if parsed else None})
        else:
            gt = _num(rec["answer"])
            ext = _num(rec.get("pred_extracted"))
            official = float(rec["MRA:.5:.95:.05"]) if _num(rec.get("MRA:.5:.95:.05")) is not None else 0.0
            out.append({"id": str(rec["index"]), "type": qtype, "mcq": False, "answer": gt, "pred": pred,
                        "official": official, "official_parsed": ext is not None, "official_number": ext,
                        "mra": "vlmeval"})
    return out


def _resp(sample: dict) -> str:
    r = sample.get("resps", sample.get("filtered_resps"))
    while isinstance(r, list) and r:
        r = r[0]
    return "" if r is None else str(r)


def items_lmms(task: str, samples: str) -> list[dict]:
    out = []
    for line in open(samples):
        s = json.loads(line)
        if task.startswith("sat"):
            options = re.findall(r"(?m)^([A-J])\. (.*)$", s["input"])
            opts = [f"{a}. {t}" for a, t in options]
            letters = LETTERS[:len(opts)]
            filt = s["filtered_resps"]
            while isinstance(filt, list) and filt:
                filt = filt[0]
            filt = str(filt).strip()
            parsed = len(filt) == 1 and filt.upper() in letters
            out.append({"id": str(s["doc_id"]), "type": "all", "mcq": True, "options": opts,
                        "answer": str(s["target"]).strip().upper(), "pred": _resp(s),
                        "official": float(s["exact_match"]), "official_parsed": parsed,
                        "official_letter": filt.upper() if parsed else None})
            continue
        doc = s.get("vsibench_score")
        if isinstance(doc, str):
            doc = ast.literal_eval(doc)
        pred = str(doc.get("prediction", ""))
        fuzzy = pred.split(" ")[0].rstrip(".").strip()
        if doc["question_type"] in VSTI_MCQ:
            opts = list(doc["options"])
            letters = LETTERS[:len(opts)]
            parsed = len(fuzzy) == 1 and fuzzy.upper() in letters
            out.append({"id": str(s["doc_id"]), "type": doc["question_type"], "mcq": True, "options": opts,
                        "answer": str(doc["mc_answer"]).strip().upper(), "pred": pred,
                        "official": float(doc["accuracy"]), "official_parsed": parsed,
                        "official_letter": fuzzy.upper() if parsed else None})
        else:
            ext = _num(fuzzy)
            out.append({"id": str(s["doc_id"]), "type": doc["question_type"], "mcq": False,
                        "answer": _num(doc["ground_truth"]), "pred": pred,
                        "official": float(doc["MRA:.5:.95:.05"]), "official_parsed": ext is not None,
                        "official_number": ext, "mra": "vsti"})
    return out


def rescore_item(it: dict) -> dict:
    if it["official_parsed"]:
        return {"lenient": it["official"], "how": "official"}
    if it["mcq"]:
        letter, how = lenient_letter(it["pred"], it["options"])
        return {"lenient": float(letter == it["answer"]) if letter else 0.0, "how": how, "letter": letter}
    n = first_number(it["pred"])
    if n is None or it["answer"] is None:
        return {"lenient": 0.0, "how": "unparseable"}
    mra = mra_vlmeval if it["mra"] == "vlmeval" else mra_vsti
    try:
        return {"lenient": mra(n, it["answer"]), "how": "number", "number": n}
    except ZeroDivisionError:
        return {"lenient": 0.0, "how": "number", "number": n}


def run_items(result: dict) -> list[dict]:
    """Per-item records of a run from the files its results.json lists."""
    if result["backend"] == "vlmevalkit":
        return items_vlmeval([p for p in result["files"] if p.endswith(".xlsx")][0])
    return items_lmms(BENCHMARKS[result["bench"]]["task"], result["files"][1])


def summarize(bench: str, items: list[dict], official_overall: float) -> dict:
    from eval_external.qa_transfer.paired import headline, pick_rule

    by_type = defaultdict(list)
    for it in items:
        by_type[it["type"]].append(it["official"])
    rule, ok = pick_rule(bench, by_type, official_overall)
    ns = {t: len(v) for t, v in by_type.items()}

    def head(key: str) -> float:
        sums = defaultdict(float)
        for it in items:
            sums[it["type"]] += it[key]
        return 100 * headline(rule, dict(sums), ns)

    def per_type(key: str) -> dict:
        acc = defaultdict(list)
        for it in items:
            acc[it["type"]].append(it[key])
        return {t: round(100 * sum(v) / len(v), 2) for t, v in sorted(acc.items())}

    hows = Counter(it["how"] for it in items)
    mcq = [it for it in items if it["mcq"]]
    numeric_check = [abs((mra_vlmeval if it["mra"] == "vlmeval" else mra_vsti)(it["official_number"], it["answer"])
                         - it["official"]) < 1e-9
                     for it in items if not it["mcq"] and it["official_parsed"] and it["answer"]]
    return {
        "bench": bench, "rule": rule, "rule_reproduces_official_overall": ok, "n_items": len(items),
        "official": {"overall": round(head("official"), 2), "per_type": per_type("official"),
                     "backend_overall": official_overall},
        "lenient": {"overall": round(head("lenient"), 2), "per_type": per_type("lenient")},
        "status": dict(hows),
        "n_unparseable": hows.get("unparseable", 0),
        "unparseable_pct": round(100 * hows.get("unparseable", 0) / max(len(items), 1), 2),
        "n_mcq": len(mcq),
        "n_mcq_wrong_parsed": sum(1 for it in mcq if it["how"] != "unparseable" and it["lenient"] == 0),
        "n_official_right_lenient_wrong": sum(1 for it in items if it["official"] > it["lenient"] + 1e-12),
        "numeric_mra_copy_matches": f"{sum(numeric_check)}/{len(numeric_check)}",
    }


def rescore_run(run_dir: Path) -> dict:
    result = json.loads((run_dir / "results.json").read_text())
    items = run_items(result)
    for it in items:
        it.update(rescore_item(it))
    summary = summarize(result["bench"], items, result["overall"])
    summary.update({"model": result["model"], "run_dir": str(run_dir)})
    with open(run_dir / "items_lenient.jsonl", "w", newline="\n") as f:
        for it in items:
            rec = {k: it.get(k) for k in ("id", "type", "mcq", "answer", "official", "official_parsed", "lenient",
                                          "how", "letter", "number")}
            rec["pred"] = it["pred"][:400]
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    (run_dir / "lenient.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    args = ap.parse_args()
    for run in args.runs:
        s = rescore_run(run.resolve())
        print(f"{s['model']:16s} {s['bench']:18s} official {s['official']['overall']:6.2f} lenient "
              f"{s['lenient']['overall']:6.2f}  unparseable {s['n_unparseable']} ({s['unparseable_pct']}%)  "
              f"status {s['status']}  rule {s['rule']} ok={s['rule_reproduces_official_overall']}  "
              f"official>lenient {s['n_official_right_lenient_wrong']}  mra copy {s['numeric_mra_copy_matches']}")


if __name__ == "__main__":
    main()
