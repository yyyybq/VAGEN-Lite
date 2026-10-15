"""Scoring as the benchmarks' official code does it (sources pinned in docs/interiorgs_qa/COVERAGE.md).

* multiple choice: accuracy of the extracted option letter;
* VSI-Bench numeric answers: mean relative accuracy (MRA) over thresholds 0.50-0.95, step 0.05
  (thinking-in-space@51e089c, lmms_eval/tasks/vsibench/utils.py), letters by its fuzzy_matching + exact_match;
* MMSI-Bench: plain scoring (MMSIBenchDataset.evaluate with extract_single_choice_with_word_boundary) and
  circular scoring (MMSIBenchCircular: four rotations ABCD/BCDA/CDAB/DABC of the options, a question counts
  only when all four are right) -- InternRobotics/MMSI-Bench@13e58a2;
* 3DSRBench: circular scoring (every rotation of the options right; compute_3drbench_results_circular.py);
* MindCube: extract_answer of mll-lab-nu/MindCube@6d8fd9a src/evaluation/core/extractors.py, exact compare;
* ViewSpatial-Bench: last line of the reply, first standalone A-D (evaluate.py extract_option);
* BLINK: the rule-based part of its extraction (the paper adds a GPT fallback, not reproduced here);
* SITE: chance-adjusted accuracy over the questions (sum of 1/k over k-option questions).

The small extraction functions are reproduced from the cited files so the rules match exactly.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

LETTERS = "ABCDEFGH"


# ----------------------------------------------------------------------------- VSI-Bench
def vsi_fuzzy(pred: str) -> str:
    return pred.split(" ")[0].rstrip(".").strip()


def vsi_mca(pred: str, gt_letter: str) -> float:
    return 1.0 if vsi_fuzzy(pred).lower() == gt_letter.lower() else 0.0


def vsi_mra(pred: str, gt: float, start: float = 0.5, end: float = 0.95, interval: float = 0.05) -> float:
    try:
        p = float(vsi_fuzzy(pred))
    except (TypeError, ValueError):
        return 0.0
    thr = np.linspace(start, end, int((end - start) / interval + 2))
    return float((abs(p - gt) / gt <= 1 - thr).mean())


# ----------------------------------------------------------------------------- MMSI-Bench
def mmsi_extract(pred: Optional[str]) -> Optional[str]:
    if pred is None:
        return None
    pred = str(pred)
    m = re.search(r"``([^`]*)``", pred)
    if m:
        pred = m.group(1)
    m = re.search(r"`([^`]*)`", pred)
    if m:
        pred = m.group(1)
    m = re.search(r"\b[A-D]\b(?!\s[a-zA-Z])", pred)
    return m.group() if m else None


def mmsi_vanilla(pred: str, answer: str) -> float:
    ex = mmsi_extract(pred)
    if ex is None:
        return 0.0
    a, p = answer.lower().strip(), ex.lower().strip()
    if a == p[0] or (p[0] == "(" and a == p[1:2]) or (p[:7] == "option " and a == p[7:8]) \
            or (p[:14] == "the answer is " and a == p[14:15]):
        return 1.0
    return 0.0


def rotations(options: Sequence[str], answer_letter: str) -> List[Tuple[List[str], str]]:
    """MMSI cp4-style rotations (for k options: k rotations): new option lists and the answer letter."""
    k = len(options)
    ai = LETTERS.index(answer_letter)
    out = []
    for s in range(k):
        order = [(i + s) % k for i in range(k)]
        out.append(([options[j] for j in order], LETTERS[order.index(ai)]))
    return out


def circular_hit(scores: Sequence[float]) -> float:
    return 1.0 if all(s >= 1.0 for s in scores) else 0.0


# ----------------------------------------------------------------------------- MindCube
def mindcube_extract(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    m = list(re.finditer(r"([A-E])\.", text))
    if m:
        return m[-1].group(1)
    sec = re.search(r"<Answer>(.*?)(?:<|$)", text, re.DOTALL)
    if sec:
        for pat in (r"[Mm]y answer is ([A-E])", r"[Mm]y answer is ([A-E])\.", r"[Tt]he answer is ([A-E])",
                    r"(?:Answer: )?([A-E])\.", r"\b([A-E])\b"):
            mm = list(re.finditer(pat, sec.group(1)))
            if mm:
                return mm[-1].group(1)
    for pat in (r"(?:Answer: )?([A-E])\. [A-Za-z0-9 \-\(\)\'\",]+(?=(?:\n|$|\.|\"))", r"(?:Answer: )?([A-E])\. [A-Za-z0-9 \-\(\)\'\"]+",
                r"(?:^|\n)(?:Answer: )?([A-E])(?:\.|$|\s)", r"[\*\"]([A-E])[\*\"]", r"\bAnswer:?\s*([A-E])\b",
                r"[Mm]y answer is ([A-E])", r"[Mm]y answer is ([A-E])\.", r"answer is ([A-E])"):
        mm = list(re.finditer(pat, text))
        if mm:
            return mm[-1].group(1)
    lines = text.split("\n")
    hits = [(i, re.search(r"([A-E])\. [A-Za-z0-9 \-\(\)\'\",]+", l)) for i, l in enumerate(lines)]
    hits = [(i, h.group(1)) for i, h in hits if h]
    if hits:
        return hits[-1][1]
    for l in reversed(lines):
        mm = re.search(r"\b([A-E])\b", l)
        if mm:
            return mm.group(1)
    return None


# ----------------------------------------------------------------------------- ViewSpatial, BLINK, generic
def viewspatial_extract(text: str) -> Optional[str]:
    last = str(text).split("\n")[-1]
    m = re.search(r"\b([A-D])\b", last, re.IGNORECASE)
    return m.group(1).upper() if m else None


def blink_extract(text: str) -> Optional[str]:
    m = re.search(r"\(([A-H])\)", str(text))
    if m:
        return m.group(1)
    m = re.search(r"\b([A-H])\b", str(text))
    return m.group(1) if m else None


def site_caa(correct: Sequence[float], n_options: Sequence[int]) -> float:
    chance = sum(1.0 / k for k in n_options)
    return (sum(correct) - chance) / max(len(correct) - chance, 1e-9)


# ----------------------------------------------------------------------------- dispatch
def score(question: Dict[str, Any], reply: str, family: str = "orig") -> Dict[str, float]:
    """Score one reply to one phrasing of a question record (qa.Question.to_json())."""
    v = question["variants"][family]
    bench = question["benchmark"]
    ans = v["answer"]
    if question["answer_type"] == "numeric":
        return {"MRA": vsi_mra(reply, float(ans))}
    if bench == "VSI-Bench":
        return {"accuracy": vsi_mca(reply, ans)}
    if bench == "MMSI-Bench":
        return {"accuracy": mmsi_vanilla(reply, ans)}
    if bench == "MindCube":
        return {"accuracy": 1.0 if mindcube_extract(reply) == ans else 0.0}
    if bench == "ViewSpatial-Bench":
        return {"accuracy": 1.0 if viewspatial_extract(reply) == ans else 0.0}
    if bench == "BLINK":
        return {"accuracy": 1.0 if blink_extract(reply) == ans else 0.0}
    # EmbSpatial, 3DSRBench (per rotation), anything else: first standalone letter
    m = re.search(r"\b([A-H])\b", str(reply))
    return {"accuracy": 1.0 if m and m.group(1) == ans else 0.0}


def circular_variants(question: Dict[str, Any], family: str = "orig") -> List[Tuple[str, str]]:
    """(prompt, answer letter) for every rotation of the options, rendered in the family's current layout
    (qa.render: the original-benchmark wordings as scripts/easi_eval.py sends them)."""
    from .qa import render

    v = question["variants"][family]
    if not v.get("options"):
        return [(render(question, family), v["answer"])]
    return [(render(question, family, opts), a) for opts, a in rotations(v["options"], v["answer"])]


def self_check(questions: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Every training answer (qa.answer_target, the form each wording's instruction asks for), given as the
    reply, must score 1 under its benchmark's rules."""
    from .qa import answer_target

    bad = []
    for q in questions:
        for fam, v in q["variants"].items():
            reply = answer_target(q, fam)
            s = score(q, reply, fam)
            if min(s.values()) < 1.0:
                bad.append((q["qid"], fam, reply, s))
    return {"checked": len(questions), "failures": bad[:20], "n_failures": len(bad)}
