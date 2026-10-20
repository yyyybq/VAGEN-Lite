"""Text-only shortcut checks of the P1 bank: can a question be answered from its text and options alone, without the
images?  Each rule sees only what a model sees in the text; its accuracy is compared with chance.

    python -m data_gen.transition_qa.shortcuts --out BANK [--md report.md]

Rules (docs/transition_qa/DIAG3_PROBE.md, "捷径检查"):

- ``inverse_frequency`` (action_between): the option whose action appears twice and whose count is the most common.
  With the distractors of 2026-10-18/19, (a, k2), (inv a, k), (b, k), that was always the answer; with the 2 x 2 of
  2026-10-19 ({a, b} x {k, k2}) every option qualifies and the rule guesses.
- ``learned_options`` (every type with options, on the test and probe files): the option text the training file
  most often had right for the same set of options; it catches priors (an action pair or a count pair whose answer
  is usually the same one, a distance window whose answer is usually the same step).  A set never seen in
  training is a guess.
- ``learned_action`` (goal, left_right, closer, both_in_view, occluded, on the test and probe files): the answer
  (its meaning: yes / no, left / right, full / partial / none, ...) the training file most often had for the same
  family and action with its count (``stem_cell``: what the stem says besides the objects).  It found the goal
  types' priors of 2026-10-20 (goal_after 72%, goal_now 65% against 50%).
- ``option_order`` (both_in_view, occluded): the answer is moved to a position in turn and the other options keep
  their order; when that order is the fixed one of the question template, the moved option stands out.  (build draws
  a random order first; the probe set of 2026-10-19 did not.)
- ``first_option`` (distance_bin, height_bin): the four options are neighbouring steps and the answer's position is
  drawn among the windows that fit; the most frequent answer letter for each smallest option, learned on the
  training file, is applied to the other files.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import generate as GEN

LETTERS = "ABCD"


def parse_action_option(text: str) -> Tuple[str, int]:
    """'the camera turned left by 40 degrees' -> ('turn_left', 2); 0.3 m and 20 degrees per step."""
    from . import questions as Q

    return Q.parse_action_text(text)


def inverse_frequency(row: Dict[str, Any]) -> List[int]:
    opts = [parse_action_option(o) for o in row["options"]]
    acts = collections.Counter(a for a, _ in opts)
    counts = collections.Counter(k for _, k in opts)
    top = max(counts.values())
    return [i for i, (a, k) in enumerate(opts) if acts[a] == 2 and counts[k] == top]


def _first_object(row: Dict[str, Any]) -> str:
    q = row["messages"][0]["content"]
    m = re.search(r"will (.*?) and (.*?) both be inside|are (.*?) and (.*?) both inside", q)
    return (m.group(1) or m.group(3)) if m else ""


def option_order(row: Dict[str, Any]) -> List[int]:
    opts = row["options"]
    if row["qtype"].startswith("both_in_view"):
        a = _first_object(row)
        idx = [0 if o == "both are" else 3 if o == "neither" else (1 if o == f"only {a}" else 2) for o in opts]
    else:
        idx = [["fully", "partly", "not"].index(o.split(" ")[0]) for o in opts]
    return [i for i in range(len(idx)) if [x for j, x in enumerate(idx) if j != i] == sorted(x for j, x in enumerate(idx) if j != i)]


def _expected(rows: Sequence[Dict[str, Any]], rule) -> float:
    """Expected accuracy of a rule that guesses among its candidates (among all options when it has none)."""
    total = 0.0
    for r in rows:
        cands = rule(r) or list(range(len(r["options"])))
        total += (LETTERS.index(r["answer"]) in cands) / len(cands)
    return total / len(rows) if rows else float("nan")


def answer_text(r: Dict[str, Any]) -> str:
    """The right option's text (Yes / No for the yes / no types)."""
    return r["options"][LETTERS.index(r["answer"])] if r["options"] else r["answer"]


def learned_table(train: Sequence[Dict[str, Any]], qtype: str, signature) -> Dict[Any, str]:
    """The answer most often right in the training file for each signature (an option set; for the yes / no types
    the family and the action)."""
    tab: Dict[Any, collections.Counter] = collections.defaultdict(collections.Counter)
    for r in train:
        if r["qtype"] == qtype:
            tab[signature(r)][answer_text(r)] += 1
    return {k: c.most_common(1)[0][0] for k, c in tab.items()}


def learned_accuracy(rows: Sequence[Dict[str, Any]], table: Dict[Any, str], signature) -> float:
    """A rule that picks the option text the training file most often had right for the same option set; guesses
    when the set never occurred in training."""
    total = 0.0
    for r in rows:
        best = table.get(signature(r))
        total += (best == answer_text(r)) if best is not None else 1.0 / (len(r["options"]) or 2)
    return total / len(rows) if rows else float("nan")


VALUE_KINDS = ("goal", "left_right", "closer", "both_in_view", "occluded")  # the answer's meaning is among every option set


stem_cell = GEN.stem_cell  # the family and the action with its count (build balances the answers within it)


def learned_value_table(train: Sequence[Dict[str, Any]], qtype: str) -> Dict[Any, str]:
    tab: Dict[Any, collections.Counter] = collections.defaultdict(collections.Counter)
    for r in train:
        if r["qtype"] == qtype:
            tab[stem_cell(r)][r["answer_value"]] += 1
    return {k: c.most_common(1)[0][0] for k, c in tab.items()}


def learned_value_accuracy(rows: Sequence[Dict[str, Any]], table: Dict[Any, str]) -> float:
    """A rule that answers what the training file most often had for the same family and action (the answer's
    meaning, e.g. "no" or "full", which every option set of the type has); guesses for a cell never seen."""
    total = 0.0
    for r in rows:
        best = table.get(stem_cell(r))
        total += (best == r["answer_value"]) if best is not None else 1.0 / (len(r["options"]) or 2)
    return total / len(rows) if rows else float("nan")


def first_option_table(train: Sequence[Dict[str, Any]], qtype: str) -> Dict[str, str]:
    tab: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for r in train:
        if r["qtype"] == qtype:
            tab[r["options"][0]][r["answer"]] += 1
    return {k: c.most_common(1)[0][0] for k, c in tab.items()}


def report(bank: Path) -> List[Dict[str, Any]]:
    files = {f: GEN.read_jsonl(bank / f) for f in ("p1_train.jsonl", "p1_test.jsonl", "p1_probe.jsonl") if (bank / f).exists()}
    out = []
    for f, rows in files.items():
        by_q: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
        for r in rows:
            by_q[r["qtype"]].append(r)
        for q, rs in sorted(by_q.items()):
            chance = 1.0 / (len(rs[0]["options"]) or 2)
            rules = []
            if q == "action_between":
                rules.append(("inverse_frequency", _expected(rs, inverse_frequency)))
            elif q.startswith(("both_in_view", "occluded")):
                rules.append(("option_order", _expected(rs, option_order)))
            elif q.startswith(("distance_bin", "height_bin")) and f != "p1_train.jsonl":
                table = first_option_table(files["p1_train.jsonl"], q)
                rules.append(("first_option", sum(table.get(r["options"][0]) == r["answer"] for r in rs) / len(rs)))
            if rs[0]["options"] and f != "p1_train.jsonl":  # any type: the option set -> the commonest right option
                sig = lambda r: frozenset(r["options"])  # noqa: E731
                rules.append(("learned_options", learned_accuracy(rs, learned_table(files["p1_train.jsonl"], q, sig), sig)))
            if q.rsplit("_", 1)[0] in VALUE_KINDS and f != "p1_train.jsonl":  # the stem's family and action -> the commonest answer
                rules.append(("learned_action", learned_value_accuracy(rs, learned_value_table(files["p1_train.jsonl"], q))))
            for rule, acc in rules:
                out.append({"file": f, "qtype": q, "n": len(rs), "rule": rule, "accuracy": acc, "chance": chance})
    return out


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="the bank folder (p1_train / p1_test / p1_probe.jsonl)")
    ap.add_argument("--md", default="", help="also write a markdown table here")
    a = ap.parse_args(argv)
    rows = report(Path(a.out))
    lines = ["| 文件 | 题类 | 题数 | 规则 | 只看文字的准确率 | 随机猜 |", "| --- | --- | --- | --- | --- | --- |"]
    lines += [f"| {r['file']} | {r['qtype']} | {r['n']} | {r['rule']} | {100 * r['accuracy']:.1f}% | {100 * r['chance']:.1f}% |"
              for r in rows]
    text = "\n".join(lines)
    if a.md:
        Path(a.md).write_text(text + "\n", encoding="utf-8")
    print(text)
    print(json.dumps({"rules": len(rows)}))


if __name__ == "__main__":
    main()
