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
    python -m reward.compare_qa easi --root EASI_ROOT --scores EASI8_SCORE.json --out FILE.json

Writes FILE.json and FILE.md.

EASI-8 (sections 2.2 and 2.4; written 11-07 before any phase-3 EASI result was opened):
* the verdict reads reward/easi8_score.py's output (its "rejudged" mean per model): O-keep's extrapolation holds when
  the mean of its two seed differences is >= +1.0 and both are > 0; V4 and O have one seed, so it cannot hold for
  them (their differences are reported);
* the pairing is per benchmark, question by question, from lmms-eval's per-sample files
  (EASI_ROOT/<job>/<model>/<the one folder with *_results.json>/*_samples_*.jsonl): each question's own score under
  the benchmark's overall metric (MMSI "average", ViewSpatial "overall_accuracy", BLINK "blink_acc", 3DSR
  "vanilla_accuracy", EmbSpatial "embspatial_acc"; VSI-Bench "vsibench_overall": the exact match of a choice question,
  the mean relative accuracy of a numeric one; SITE: its image and video jobs as one benchmark, the raw per-question
  correctness "accuracy"."overall"; MindCube: re-judged by the option an answer names, as easi8_score.mindcube_lenient,
  with the official per-question score reported too).  Wins = questions where the first model scores higher, losses
  = where it scores lower; ties (zero differences) are left out of the two-sided exact sign test.  The eight-benchmark
  mean gets a difference only.
"""

import argparse
import ast
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


# ---------------------------------------------------------------- EASI-8 (sections 2.2, 2.4)
EASI_MAIN = (("V4", "rw_g_250", S1), ("O", "rw_o_200", S1), ("O-keep s1", "rw_ok_250", S1),
             ("O-keep s2", "rw_s2ok_250", S2))
EASI_BENCH = ("vsi_bench", "mmsi_bench", "mindcube_tiny", "viewspatial", "site", "blink", "3dsrbench", "embspatial")
EASI_MARGIN = 1.0
_METRIC = {"mmsi_bench": "average", "viewspatial": "overall_accuracy", "blink": "blink_acc",
           "3dsrbench": "vanilla_accuracy", "embspatial": "embspatial_acc", "vsi_bench": "vsibench_overall",
           "mindcube_tiny": "overall_accuracy", "site_image": "accuracy", "site_video": "accuracy"}


def _d(x: Any) -> Dict[str, Any]:
    return x if isinstance(x, dict) else ast.literal_eval(str(x))


def _item_key(job: str, row: Dict[str, Any], m: Dict[str, Any]) -> str:
    for k in ("qid", "question_id", "id"):
        if m.get(k) not in (None, ""):
            return f"{job}:{m[k]}"
    return f"{job}:{row['doc_id']}"


def _item_score(job: str, m: Dict[str, Any]) -> float:
    if job == "vsi_bench":  # a choice question carries "accuracy" (exact match), a numeric one "MRA:.5:.95:.05"
        if "accuracy" in m:
            return float(m["accuracy"])
        mra = [v for k, v in m.items() if k.startswith("MRA")]
        if len(mra) != 1:
            raise SystemExit(f"VSI question {m.get('id')}: no single score field")
        return float(mra[0])
    if job in ("site_image", "site_video"):
        return float(m["overall"])
    if job in ("blink", "embspatial"):
        return 1.0 if m["is_correct"] in (True, "True", 1, 1.0) else 0.0
    return float(m["score"])


def mindcube_right(row: Dict[str, Any]) -> bool:
    """easi8_score.mindcube_lenient's judgement of one MindCube-Tiny answer (the option it names, by letter, else the
    one option whose text it gives), per question."""
    norm = lambda s: re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()  # noqa: E731
    x = row.get("filtered_resps") or row.get("resps")
    while isinstance(x, list) and x:
        x = x[0]
    s = re.sub(r"</?answer>", "", str(x)).strip()
    q = row["input"].split("[Question]")[-1].strip()
    opts = dict(re.findall(r"(?:^|\s)([A-F])\.\s(.*?)(?=\s[A-F]\.\s|$)", q, flags=re.S))
    target = re.search(r"<answer>\s*([A-F])", str(row["target"])).group(1)
    mm = re.match(r"^\(?([A-F])[\.\):]?(\s|$)", s)
    if mm:
        pred = mm.group(1)
    else:
        hits = [k for k, v in opts.items() if norm(v) and norm(v) == norm(s)] or \
               [k for k, v in opts.items() if norm(v) and (norm(v) in norm(s) or norm(s) in norm(v))]
        pred = hits[0] if len(hits) == 1 else None
    return pred == target


def easi_items(root: Path, job: str, name: str, rejudge: bool = True) -> Dict[str, float]:
    """{question key: score} of one model on one EASI job (SITE's two jobs are separate here)."""
    d = root / job / name
    sub = [s for s in d.iterdir() if s.is_dir() and any(s.glob("*_results.json"))]
    if len(sub) != 1:
        raise SystemExit(f"{d}: expected one result folder, found {sub}")
    out: Dict[str, float] = {}
    for f in sorted(sub[0].glob("*_samples_*.jsonl")):
        for row in _jsonl(f):
            m = _d(row[_METRIC[job]])
            key = _item_key(job, row, m)
            if key in out:
                raise SystemExit(f"{f}: question {key} twice")
            if job == "mindcube_tiny" and rejudge:
                out[key] = 1.0 if mindcube_right(row) else 0.0
            else:
                out[key] = _item_score(job, m)
    return out


def bench_items(root: Path, bench: str, name: str, rejudge: bool = True) -> Dict[str, float]:
    if bench == "site":
        return {**easi_items(root, "site_image", name), **easi_items(root, "site_video", name)}
    return easi_items(root, bench, name, rejudge)


def score_pair(a: Dict[str, float], b: Dict[str, float]) -> Dict[str, Any]:
    if set(a) != set(b):
        raise SystemExit(f"the two models cover different questions ({len(a)} vs {len(b)})")
    w = sum(1 for k in a if a[k] > b[k])
    l = sum(1 for k in a if a[k] < b[k])
    n = len(a)
    return {"n": n, "mean_a": 100 * sum(a.values()) / n, "mean_b": 100 * sum(b.values()) / n,
            "diff_item_mean": 100 * (sum(a.values()) - sum(b.values())) / n, "wins": w, "losses": l,
            "ties": n - w - l, "p": sign_p(w, l)}


def easi_verdict(scores: Dict[str, Any]) -> Dict[str, Any]:
    d = {lab: scores[a]["rejudged"] - scores[b]["rejudged"] for lab, a, b in EASI_MAIN}
    ok = (d["O-keep s1"] + d["O-keep s2"]) / 2
    return {"V4": {"diff": d["V4"], "holds": False, "seeds": 1, "note": "one seed: cannot hold under the brief"},
            "O": {"diff": d["O"], "holds": False, "seeds": 1, "note": "one seed: cannot hold under the brief"},
            "O-keep": {"diffs": [d["O-keep s1"], d["O-keep s2"]], "mean": ok, "seeds": 2,
                       "each_ge_margin": [d["O-keep s1"] >= EASI_MARGIN, d["O-keep s2"] >= EASI_MARGIN],
                       "holds": ok >= EASI_MARGIN and d["O-keep s1"] > 0 and d["O-keep s2"] > 0},
            "official_diffs": {lab: scores[a]["official"] - scores[b]["official"] for lab, a, b in EASI_MAIN}}


def run_easi(root: Path, scores_path: Path) -> Dict[str, Any]:
    with open(scores_path) as f:
        scores = json.load(f)
    out: Dict[str, Any] = {"verdict": easi_verdict(scores), "pairs": {}}
    for lab, a, b in EASI_MAIN:
        row = {bench: score_pair(bench_items(root, bench, a), bench_items(root, bench, b)) for bench in EASI_BENCH}
        row["mindcube_official"] = score_pair(easi_items(root, "mindcube_tiny", a, rejudge=False),
                                              easi_items(root, "mindcube_tiny", b, rejudge=False))
        out["pairs"][lab] = {"a": a, "b": b, "rejudged_a": scores[a]["rejudged"], "rejudged_b": scores[b]["rejudged"],
                             "benchmarks": row}
    return out


def write_easi_md(res: Dict[str, Any], path: Path) -> None:
    v = res["verdict"]
    L = ["## EASI-8 外推（预登记 2.2，重判分）", "",
         f"- O-keep：{v['O-keep']['diffs'][0]:+.2f} / {v['O-keep']['diffs'][1]:+.2f}，平均 {v['O-keep']['mean']:+.2f}，"
         f"{'外推成立' if v['O-keep']['holds'] else '外推不成立'}（各自 ≥ 1：{v['O-keep']['each_ge_margin']}）",
         f"- V4：{v['V4']['diff']:+.2f}（一个种子，按任务书的条件外推不能成立）",
         f"- O：{v['O']['diff']:+.2f}（一个种子，按任务书的条件外推不能成立）", "",
         "## 逐基准配对（胜 / 负 / 平；双侧精确符号检验，平的不进检验；差是逐题得分平均之差，个点）", "",
         "| 比较 | " + " | ".join(EASI_BENCH) + " | MindCube 官方 |", "|" + " --- |" * (len(EASI_BENCH) + 2)]
    for lab, r in res["pairs"].items():
        cells = []
        for k in list(EASI_BENCH) + ["mindcube_official"]:
            c = r["benchmarks"][k]
            cells.append(f"{c['diff_item_mean']:+.1f}（{c['wins']}/{c['losses']}/{c['ties']}，p {c['p']:.3g}）")
        L.append(f"| {lab}（{r['a']} 对 {r['b']}，重判 {r['rejudged_a']:.2f} 对 {r['rejudged_b']:.2f}） | " + " | ".join(cells) + " |")
    path.write_text("\n".join(L) + "\n")


def main(argv: Optional[Sequence[str]] = None) -> None:
    argv = list(argv) if argv is not None else None
    import sys
    args_in = argv if argv is not None else sys.argv[1:]
    if args_in and args_in[0] == "easi":
        ap = argparse.ArgumentParser(description="EASI-8 part of PREREG_P3_A (sections 2.2, 2.4)")
        ap.add_argument("--root", required=True, help="the EASI result root: ROOT/<job>/<model>/")
        ap.add_argument("--scores", required=True, help="reward/easi8_score.py's FILE.json for the six models")
        ap.add_argument("--out", required=True)
        a = ap.parse_args(args_in[1:])
        res = run_easi(Path(a.root), Path(a.scores))
        out = Path(a.out)
        out.write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n")
        write_easi_md(res, out.with_suffix(".md"))
        print(out.with_suffix(".md").read_text())
        return
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--x0", required=True)
    ap.add_argument("--g0", required=True)
    ap.add_argument("--diag", required=True, help="the diagnostics chain's TR/diag")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(args_in)
    res = run(Path(a.x0), Path(a.g0), Path(a.diag))
    out = Path(a.out)
    out.write_text(json.dumps(res, indent=1, ensure_ascii=False) + "\n")
    write_md(res, out.with_suffix(".md"))
    print(out.with_suffix(".md").read_text())


if __name__ == "__main__":
    main()
