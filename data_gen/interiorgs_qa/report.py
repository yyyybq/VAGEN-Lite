"""Numbers for the report: combos per group, pass rates, full-scale estimates, blind baselines, budgets.

    python -m data_gen.interiorgs_qa.report --census CENSUS/census.json --runs R1,R2,... --stats-runs S1,S2 \
        --out OUT.json

* combos: the CPU census (``pipeline census``): object combos per scene that satisfy a group's 3D conditions
  and have at least one legal camera passing the v10 prefilter (no caps);
* pass rate: share of rendered combos that pass the rendered-depth checks (from the render runs);
* estimate: median combos per scene x pass rate = valid combos per scene; x 800 train / 1,000 scenes;
* blind baseline: for each question, guess the most frequent answer among the OTHER questions of the same
  question type and the same object categories (else the same question type); compare with a uniform random guess
  (numeric answers: MRA of the median of the other answers). The question type is what a reader can see in the
  text: the subtask, except that EmbSpatial's left/right/above/below subtasks share one template (their subtask
  ID is the answer itself, so conditioning on it would leak the answer);
* tokens and image inputs per question from the records.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from .qa import excluded, parse_exclude
from .scoring import vsi_mra

TRAIN_SCENES = 800
ALL_SCENES = 1000


def _jsonl(p: Path) -> List[Dict[str, Any]]:
    out = []
    if p.exists():
        for l in p.read_text().splitlines():
            try:
                out.append(json.loads(l))
            except ValueError:
                continue
    return out


# census caps (GenCtx.cap(..., census=N)) sample some enumerations; scale those combo kinds back up
SCALED_KINDS = {"S3": ("triples", {"set_depth"}), "S4": ("triples", {"triple_closer"}),
                "M1": ("seeds", {"translate", "rotate", "pitch"}), "M2": ("seeds", None), "M4": ("seeds", None),
                "M6": ("grid_points", None), "M11": ("grid_points", None), "M7": ("grid_points", None),
                "M10": ("pairs", None)}


def scaled_combos(group: str, c: Dict[str, Any]) -> float:
    total = float(c.get("combos_cpu", 0))
    if group not in SCALED_KINDS:
        return total
    unit, kinds = SCALED_KINDS[group]
    tot, ex = c.get(unit + "_total", 0), c.get(unit + "_examined", 0)
    if not ex or tot <= ex:
        return total
    f = tot / ex
    if kinds is None:
        return total * f
    part = sum(v for k, v in c.items() if k.startswith("combos_cpu:") and k.split(":", 1)[1] in kinds)
    return total - part + part * f


BLIND_MIN = 30


def scan(spec: str) -> Dict[str, Dict[str, Any]]:
    """Verified combos per scanned scene for rare groups; grid-point sampling (M6/M11) is scaled back up.

    The render runs cap verified combos per scene (--verify-cap), so this is a lower bound when a scene has more."""
    out: Dict[str, Dict[str, Any]] = {}
    for part in [x for x in spec.split(";") if x]:
        g, rr = part.split("=")
        agg: Counter = Counter()
        for r in rr.split("+"):
            s = json.loads((Path(r) / "stats.json").read_text())
            for sid, v in s["scenes"].items():
                c = (v.get("stats") or {}).get(g)
                if c is None:
                    continue
                agg["scenes"] += 1
                for k in ("combos_cpu", "rendered_combos", "verified_combos", "questions",
                          "grid_points_total", "grid_points_examined"):
                    agg[k] += c.get(k, 0)
        f = agg["grid_points_total"] / agg["grid_points_examined"] if agg["grid_points_examined"] else 1.0
        per = agg["verified_combos"] * f / agg["scenes"] if agg["scenes"] else 0.0
        out[g] = {"scenes": agg["scenes"], "combos_cpu": agg["combos_cpu"], "rendered": agg["rendered_combos"],
                  "verified": agg["verified_combos"], "sampling_scale": round(f, 2),
                  "verified_per_scene": round(per, 3), "full_train_est": int(round(per * TRAIN_SCENES)),
                  "full_all_est": int(round(per * ALL_SCENES))}
    return out


def canon(q: Dict[str, Any]) -> str:
    return str(q["answer"]).lower()


TEMPLATE_FAMILY = {"ES-01": "ES-relation", "ES-02": "ES-relation", "ES-03": "ES-relation", "ES-04": "ES-relation"}


def qtype(q: Dict[str, Any]) -> str:
    """The question type as a reader sees it in the text (the same template can serve several groups)."""
    st = q["subtasks"][0]
    if st == "BL-03":       # the relation word is part of the question ("left of", "on top of", "touching", ...)
        v = (q.get("gt") or {}).get("values") or {}
        return "BL-03:" + str(v.get("relation") or v.get("asked") or "")
    return TEMPLATE_FAMILY.get(st, st)


def _cats(q: Dict[str, Any]) -> Tuple[str, ...]:
    return tuple(sorted(o["display"] for o in q["objects"] if o["role"] not in ("shared_landmark",)))


def blind(questions: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Guess without the image: pool all questions of the same type (across groups), prefer those with the same
    object categories, and answer with their most frequent answer (numeric: their median)."""
    by_type: Dict[str, List[int]] = defaultdict(list)
    for i, q in enumerate(questions):
        by_type[qtype(q)].append(i)
    res: Dict[str, Dict[str, list]] = defaultdict(lambda: {"hits": [], "rand": [], "mra": [], "n": [], "num": []})
    for i, q in enumerate(questions):
        g = q["group"]
        res[g]["n"].append(1)
        others = [questions[j] for j in by_type[qtype(q)] if j != i]
        same = [x for x in others if _cats(x) == _cats(q)]
        pool = same or others
        if q["answer_type"] == "numeric":
            res[g]["num"].append(1)
            if pool:
                guess = statistics.median(float(x["answer"]) for x in pool)
                res[g]["mra"].append(vsi_mra(str(guess), float(q["answer"])))
            continue
        opts = q["variants"]["orig"].get("options") or []
        res[g]["rand"].append(1.0 / max(len(opts), 2))
        if not pool:
            res[g]["hits"].append(0.0)
            continue
        guess = Counter(canon(x) for x in pool).most_common(1)[0][0]
        res[g]["hits"].append(1.0 if guess == canon(q) else 0.0)
    by_group: Dict[str, Dict[str, Any]] = {}
    for g, r in res.items():
        dist: Dict[str, Counter] = defaultdict(Counter)
        for q in questions:
            if q["group"] == g and q["answer_type"] != "numeric":
                dist[qtype(q)][canon(q)] += 1
        by_group[g] = {"questions": len(r["n"]), "mc_questions": len(r["hits"]),
                       "blind_hit_rate": round(float(np.mean(r["hits"])), 3) if r["hits"] else None,
                       "random_hit_rate": round(float(np.mean(r["rand"])), 3) if r["rand"] else None,
                       "numeric_questions": len(r["num"]),
                       "blind_mra_median_guess": round(float(np.mean(r["mra"])), 3) if r["mra"] else None,
                       "answer_share_top": {t: [c.most_common(1)[0][0], round(c.most_common(1)[0][1] / sum(c.values()), 2),
                                                sum(c.values())] for t, c in dist.items()}}
    return by_group


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--census", required=True)
    ap.add_argument("--runs", required=True, help="sample runs (comma separated)")
    ap.add_argument("--stats-runs", default="", help="larger runs for pass rates and blind baselines")
    ap.add_argument("--rate-override", default="", help="GROUP=run1+run2;GROUP2=run3 (pass rates from these runs)")
    ap.add_argument("--scan", default="", help="GROUP=run1+run2;... rare groups: verified combos per scanned scene "
                    "from runs that rendered every CPU combo (a larger scene sample than the census)")
    ap.add_argument("--sample-extra", default="", help="more sample runs: their questions count, their stats do not "
                    "enter the pass rates (runs that regenerated only some subtasks)")
    ap.add_argument("--stats-extra", default="", help="same, for the stats runs")
    ap.add_argument("--exclude", default="", help="RUN_SUFFIX:GROUP[:SUBTASK],qid:QID,... questions replaced by later runs")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    census = json.loads(Path(a.census).read_text())
    per_scene: Dict[str, List[float]] = defaultdict(list)
    raw_per_scene: Dict[str, List[int]] = defaultdict(list)
    for sc in census:
        for g, c in (sc.get("stats") or {}).items():
            raw_per_scene[g].append(int(c.get("combos_cpu", 0)))
            per_scene[g].append(scaled_combos(g, c))
    runs = [Path(r) for r in a.runs.split(",") if r]
    sruns = [Path(r) for r in a.stats_runs.split(",") if r] or runs
    rules = parse_exclude(a.exclude)
    extra_s = [Path(r) for r in a.sample_extra.split(",") if r]
    extra_t = [Path(r) for r in a.stats_extra.split(",") if r]
    sample = [q for r in runs + extra_s for q in _jsonl(r / "questions.jsonl") if not excluded(r, q, rules)]
    statq = [q for r in sruns + extra_t for q in _jsonl(r / "questions.jsonl") if not excluded(r, q, rules)]
    rend, ver = Counter(), Counter()
    fail: Dict[str, Counter] = defaultdict(Counter)
    override: Dict[str, List[Path]] = {}
    for part in [x for x in a.rate_override.split(";") if x]:
        g_, runs_ = part.split("=")
        override[g_] = [Path(x) for x in runs_.split("+")]
    # groups with an override take their pass rates and blind-baseline questions only from those runs
    statq = [q for q in statq if q["group"] not in override] + \
        [q for g_, rr in override.items() for r in rr for q in _jsonl(r / "questions.jsonl")
         if q["group"] == g_ and not excluded(r, q, rules)]
    # groups with too few questions in the stats runs use the sample runs (sample and top-up runs) instead
    n_stat = Counter(q["group"] for q in statq)
    statq = [q for q in statq if n_stat[q["group"]] >= BLIND_MIN] + \
        [q for q in sample if n_stat[q["group"]] < BLIND_MIN]
    rate_runs = sorted(set(sruns) | set(runs) | {r for rr in override.values() for r in rr})
    for r in rate_runs:
        s = json.loads((r / "stats.json").read_text())
        for sid, v in s["scenes"].items():
            for g, c in (v.get("stats") or {}).items():
                if g in override and r not in override[g]:
                    continue
                rend[g] += c.get("rendered_combos", 0)
                ver[g] += c.get("verified_combos", 0)
            for g, c in (v.get("fail") or {}).items():
                if g in override and r not in override[g]:
                    continue
                fail[g].update(c)
    out: Dict[str, Any] = {"groups": {}}
    for g in sorted(set(per_scene) | set(rend), key=lambda x: (x[0] != "S", x[0] != "M", int(x[1:]))):
        combos = per_scene.get(g, [])
        pr = ver[g] / rend[g] if rend[g] else None
        med = float(np.median(combos)) if combos else None
        qs = [q for q in sample if q["group"] == g]
        n_img = [len(q["views"]) for q in qs]
        tok512 = [q["tokens"]["512"]["total"] for q in qs]
        tok256 = [q["tokens"]["256"]["total"] for q in qs]
        out["groups"][g] = {
            "combos_per_scene_median": med, "combos_per_scene_range": [round(min(combos), 1), round(max(combos), 1)] if combos else None,
            "combos_raw_total_20": int(sum(raw_per_scene.get(g, []))),
            "combos_total_20": round(float(sum(combos)), 1) if combos else None, "scenes_with_combos": sum(1 for c in combos if c > 0),
            "rendered": rend[g], "verified": ver[g], "pass_rate": round(pr, 3) if pr is not None else None,
            "valid_per_scene_est": round(med * pr, 1) if med is not None and pr is not None else None,
            "valid_mean_per_scene_est": round(float(np.mean(combos)) * pr, 1) if combos and pr is not None else None,
            "full_train_est": int(round(float(np.mean(combos)) * pr * TRAIN_SCENES)) if combos and pr is not None else None,
            "full_all_est": int(round(float(np.mean(combos)) * pr * ALL_SCENES)) if combos and pr is not None else None,
            "sample_questions": len(qs), "sample_subtasks": dict(Counter(q["subtasks"][0] for q in qs)),
            "images_per_question": round(float(np.mean(n_img)), 2) if n_img else None,
            "tokens_512_mean": round(float(np.mean(tok512)), 1) if tok512 else None,
            "tokens_256_mean": round(float(np.mean(tok256)), 1) if tok256 else None,
            "top_fail": fail[g].most_common(4),
        }
    out["scan"] = scan(a.scan)
    out["blind"] = blind(statq)
    out["sample_total"] = len(sample)
    out["stats_total"] = len(statq)
    out["subtask_counts_sample"] = dict(Counter(s for q in sample for s in q["subtasks"]))
    Path(a.out).write_text(json.dumps(out, indent=1, ensure_ascii=False))
    for g, v in out["groups"].items():
        b = out["blind"].get(g, {})
        print(g, {k: v[k] for k in ("combos_per_scene_median", "pass_rate", "valid_mean_per_scene_est", "full_train_est",
                                    "sample_questions", "images_per_question", "tokens_256_mean")},
              {k: b.get(k) for k in ("blind_hit_rate", "random_hit_rate", "blind_mra_median_guess", "questions")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
