"""Reward line, phase 3, experiment A (docs/reward/PREREG_P3_A.md section 2): question-by-question paired comparison
of two models on X0, G0 and the P1 probe, and the preregistered verdicts on X0.  Written before any result of A.

Inputs (all read only):
* X0 / G0: the downstream line's questions (questions.jsonl, rows with "id", "kind", "options", "answer") and the
  answers the reward-mix chain's diagnostics step wrote (DIAG/downstream_qa/<set>_<model>.jsonl, rows {id, reply}).
  Replies are read with the downstream line's own rules, copied here verbatim (x0_qa.parse_letter and
  g0_qa.parse_reply at downstream 8731eb9e); a reply that parses to nothing is wrong.  The per-model correct counts
  are checked against the chain's DIAG/downstream_qa/summary.json when it has the model (a mismatch stops the tool).
* P1: DIAG/probe_P1/<model>.jsonl (rows with "sample_id", "correct", "compliant"): "correct" is the lenient reading;
  strict = correct and compliant, reported too.

For a pair (a, b) on one set: n, a's and b's correct counts, the difference in points (100 x (correct_a - correct_b) /
n, unrounded), wins (only a right), losses (only b right) and the two-sided exact sign test on wins vs losses
(McNemar's exact test).  Verdicts (X0 only, section 2.1): V4 and O pass at a difference >= +3.0 against S0 seed 1;
O-keep passes when the mean of its two seed differences is >= +3.0 and both are > 0.

    python -m reward.compare_qa --x0 X0_QUESTIONS --g0 G0_QUESTIONS --diag TR/diag --out FILE.json

Writes FILE.json and FILE.md.  EASI-8's per-benchmark pairing (section 2.4) is added before EASI-8's results.
"""

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

S1, S2 = "rw_s0_150", "rw_s2s0_300"
MAIN = (("V4", "rw_g_250", S1), ("O", "rw_o_200", S1), ("O-keep s1", "rw_ok_250", S1), ("O-keep s2", "rw_s2ok_250", S2))
LAST = (("V4 last", "rw_g_300", "rw_s0_200"), ("O last", "rw_o_300", "rw_s0_200"),
        ("O-keep s1 last", "rw_ok_300", "rw_s0_200"), ("O-keep s2 last", "rw_s2ok_300", S2))
C0 = "sft_q3formal_e8"
MODELS = ("rw_s0_150", "rw_s0_200", "rw_s2s0_300", "rw_g_250", "rw_g_300", "rw_o_200", "rw_o_300", "rw_ok_250",
          "rw_ok_300", "rw_s2ok_250", "rw_s2ok_300", C0)
MARGIN = 3.0
NUMBER_WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
                "nine": 9, "ten": 10}


def parse_letter(text: str):  # downstream x0_qa.parse_letter, verbatim
    """The option letter of a model's reply ('A' / 'B'), None if it names neither or both."""
    t = re.sub(r"<[^>]*>", " ", str(text or "")).strip()
    found = set(re.findall(r"\b([AB])\b", t))
    if len(found) > 1:  # names both options
        return None
    m = re.match(r"^\W*(?:answer\s*[:：]?\s*|option\s*)?\(?([AB])\)?(?:[\s.,:)\]]|$)", t, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return found.pop() if found else None


def parse_reply(q: dict, reply: str):  # downstream g0_qa.parse_reply, verbatim
    """A two-choice reply's letter (x0_qa.parse_letter, or a reply that is exactly one option's text), a counting
    reply's number (the first integer or number word); None when unreadable."""
    text = str(reply or "").strip()
    if q["kind"] == "counting":
        m = re.search(r"\d+", text)
        if m:
            return m.group(0).lstrip("0") or "0"
        w = re.search(r"\b(" + "|".join(NUMBER_WORDS) + r")\b", text.lower())
        return str(NUMBER_WORDS[w.group(1)]) if w else None
    letter = parse_letter(text)
    if letter:
        return letter
    plain = text.lower().rstrip(".").strip()
    for i, o in enumerate(q["options"]):
        if plain == str(o).lower():
            return "AB"[i]
    return None


def sign_p(w: int, l: int) -> float:
    """Two-sided exact sign test (binomial, p = 1/2) on w wins and l losses."""
    n = w + l
    if n == 0:
        return 1.0
    k = min(w, l)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def _jsonl(p: Path) -> List[Dict[str, Any]]:
    with open(p) as f:
        return [json.loads(l) for l in f if l.strip()]


def downstream_correct(questions: List[dict], answers: Path, kind: str) -> Dict[str, Dict[str, Any]]:
    """{question id: {"right": bool, "parsed": ...}} for one model's answers on X0 ("x0") or G0 ("g0")."""
    replies = {r["id"]: r.get("reply") for r in _jsonl(answers)}
    out = {}
    for q in questions:
        rep = replies.get(q["id"])
        parsed = parse_letter(rep) if kind == "x0" else parse_reply(q, rep)
        out[q["id"]] = {"right": parsed == q["answer"], "parsed": parsed}
    return out


def p1_correct(path: Path) -> Dict[str, Dict[str, Any]]:
    return {r["sample_id"]: {"right": bool(r.get("correct")), "strict": bool(r.get("correct") and r.get("compliant")),
                             "parsed": r.get("parsed")} for r in _jsonl(path)}


def pair(a: Dict[str, Dict[str, Any]], b: Dict[str, Dict[str, Any]], key: str = "right") -> Dict[str, Any]:
    ids = sorted(set(a) & set(b))
    if set(a) != set(b):
        raise SystemExit(f"the two answer sets cover different questions ({len(a)} vs {len(b)})")
    ra = sum(a[i][key] for i in ids)
    rb = sum(b[i][key] for i in ids)
    w = sum(1 for i in ids if a[i][key] and not b[i][key])
    l = sum(1 for i in ids if b[i][key] and not a[i][key])
    n = len(ids)
    return {"n": n, "a": ra, "b": rb, "diff": 100.0 * (ra - rb) / n if n else None, "wins": w, "losses": l,
            "p": sign_p(w, l)}


def verdicts(x0: Dict[Tuple[str, str], Dict[str, Any]]) -> Dict[str, Any]:
    d = {lab: x0[(a, b)]["diff"] for lab, a, b in MAIN}
    ok_mean = (d["O-keep s1"] + d["O-keep s2"]) / 2
    return {
        "V4": {"diff": d["V4"], "pass": d["V4"] >= MARGIN, "seeds": 1},
        "O": {"diff": d["O"], "pass": d["O"] >= MARGIN, "seeds": 1},
        "O-keep": {"diffs": [d["O-keep s1"], d["O-keep s2"]], "mean": ok_mean,
                   "pass": ok_mean >= MARGIN and d["O-keep s1"] > 0 and d["O-keep s2"] > 0, "seeds": 2},
    }


def run(x0q: Path, g0q: Path, diag: Path) -> Dict[str, Any]:
    qs = {"x0": _jsonl(x0q), "g0": _jsonl(g0q)}
    per: Dict[str, Dict[str, Dict[str, Dict[str, Any]]]] = {"x0": {}, "g0": {}, "p1": {}}
    for m in MODELS:
        for s in ("x0", "g0"):
            per[s][m] = downstream_correct(qs[s], diag / "downstream_qa" / f"{s}_{m}.jsonl", s)
        per["p1"][m] = p1_correct(diag / "probe_P1" / f"{m}.jsonl")
    summ = diag / "downstream_qa" / "summary.json"
    if summ.exists():  # the chain's own counts must agree with this reading
        with open(summ) as f:
            chain = json.load(f).get("models") or {}
        for m in MODELS:
            for s in ("x0", "g0"):
                c = ((chain.get(m) or {}).get(s) or {}).get("correct")
                mine = sum(v["right"] for v in per[s][m].values())
                if c is not None and c != mine:
                    raise SystemExit(f"{s} {m}: the chain counts {c} right, this reading {mine}")
    out: Dict[str, Any] = {"models": {m: {s: {"correct": sum(v["right"] for v in per[s][m].values()),
                                              "n": len(per[s][m]),
                                              "unparsed": sum(1 for v in per[s][m].values() if v["parsed"] is None)}
                                          for s in per} for m in MODELS},
                           "pairs": {}}
    for m in MODELS:
        out["models"][m]["p1"]["strict"] = sum(v["strict"] for v in per["p1"][m].values())
    x0_main = {}
    for lab, a, b in MAIN + LAST + tuple((f"{m} vs C0", m, C0) for m in MODELS if m != C0):
        row = {s: pair(per[s][a], per[s][b]) for s in per}
        row["p1_strict"] = pair(per["p1"][a], per["p1"][b], "strict")
        out["pairs"][lab] = {"a": a, "b": b, **row}
        x0_main[(a, b)] = row["x0"]
    out["verdicts_x0"] = verdicts(x0_main)
    return out


def write_md(res: Dict[str, Any], path: Path) -> None:
    f = lambda c: f"{c['diff']:+.1f}（{c['wins']} / {c['losses']}，p {c['p']:.3g}）"  # noqa: E731
    L = ["## 各模型答对题数", "", "| 模型 | X0（510） | G0（300） | P1（838） | P1 严格 | 解析不出 X0 / G0 / P1 |",
         "| --- | --- | --- | --- | --- | --- |"]
    for m, r in res["models"].items():
        L.append(f"| {m} | {r['x0']['correct']} | {r['g0']['correct']} | {r['p1']['correct']} | {r['p1']['strict']} | "
                 f"{r['x0']['unparsed']} / {r['g0']['unparsed']} / {r['p1']['unparsed']} |")
    L += ["", "## 配对（差：个点；胜 / 负；双侧精确符号检验）", "", "| 比较 | X0 | G0 | P1 | P1 严格 |",
          "| --- | --- | --- | --- | --- |"]
    for lab, r in res["pairs"].items():
        L.append(f"| {lab}（{r['a']} 对 {r['b']}） | {f(r['x0'])} | {f(r['g0'])} | {f(r['p1'])} | {f(r['p1_strict'])} |")
    v = res["verdicts_x0"]
    L += ["", "## X0 主判据（预登记 2.1）", "",
          f"- V4：{v['V4']['diff']:+.2f}，{'过' if v['V4']['pass'] else '没过'}（一个种子）",
          f"- O：{v['O']['diff']:+.2f}，{'过' if v['O']['pass'] else '没过'}（一个种子）",
          f"- O-keep：{v['O-keep']['diffs'][0]:+.2f} / {v['O-keep']['diffs'][1]:+.2f}，平均 {v['O-keep']['mean']:+.2f}，"
          f"{'过' if v['O-keep']['pass'] else '没过'}"]
    path.write_text("\n".join(L) + "\n")


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--x0", required=True)
    ap.add_argument("--g0", required=True)
    ap.add_argument("--diag", required=True, help="the diagnostics chain's TR/diag")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    res = run(Path(a.x0), Path(a.g0), Path(a.diag))
    out = Path(a.out)
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n")
    write_md(res, out.with_suffix(".md"))
    print(out.with_suffix(".md").read_text())


if __name__ == "__main__":
    main()
