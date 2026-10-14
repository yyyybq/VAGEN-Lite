"""Numbers for the report: combos per group, pass rates, full-scale estimates, blind baselines, budgets.

    python -m data_gen.interiorgs_qa.report --census CENSUS/census.json --runs R1,R2,... --stats-runs S1,S2 \
        --out OUT.json

* combos: the CPU census (``pipeline census``): object combos per scene that satisfy a group's 3D conditions
  and have at least one legal camera passing the v10 prefilter (no caps);
* pass rate: share of rendered combos that pass the rendered-depth checks (from the render runs);
* estimate: median combos per scene x pass rate = valid combos per scene; x 800 train / 1,000 scenes;
* blind baseline: for each question, guess the most frequent answer among the OTHER questions with the same
  subtask and the same object categories (else the same subtask); compare with a uniform random guess
  (numeric answers: MRA of the median of the other answers);
* tokens and image inputs per question from the records.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .scoring import vsi_mra

TRAIN_SCENES = 800
ALL_SCENES = 1000


def _jsonl(p: Path) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def canon(q: Dict[str, Any]) -> str:
    return str(q["answer"]).lower()


def blind(questions: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    by_group: Dict[str, Dict[str, Any]] = {}
    groups = defaultdict(list)
    for q in questions:
        groups[q["group"]].append(q)
    for g, qs in groups.items():
        hits, rand, mra_blind, n_num = [], [], [], 0
        for i, q in enumerate(qs):
            st = q["subtasks"][0]
            cats = tuple(sorted(o["display"] for o in q["objects"] if o["role"] not in ("shared_landmark",)))
            others = [x for j, x in enumerate(qs) if j != i and x["subtasks"][0] == st]
            same = [x for x in others if tuple(sorted(o["display"] for o in x["objects"]
                                                      if o["role"] not in ("shared_landmark",))) == cats]
            pool = same or others
            if q["answer_type"] == "numeric":
                n_num += 1
                if pool:
                    guess = statistics.median(float(x["answer"]) for x in pool)
                    mra_blind.append(vsi_mra(str(guess), float(q["answer"])))
                continue
            opts = q["variants"]["orig"].get("options") or []
            rand.append(1.0 / max(len(opts), 2))
            if not pool:
                hits.append(0.0)
                continue
            guess = Counter(canon(x) for x in pool).most_common(1)[0][0]
            hits.append(1.0 if guess == canon(q) else 0.0)
        by_group[g] = {"questions": len(qs), "mc_questions": len(hits),
                       "blind_hit_rate": round(float(np.mean(hits)), 3) if hits else None,
                       "random_hit_rate": round(float(np.mean(rand)), 3) if rand else None,
                       "numeric_questions": n_num,
                       "blind_mra_median_guess": round(float(np.mean(mra_blind)), 3) if mra_blind else None}
    return by_group


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--census", required=True)
    ap.add_argument("--runs", required=True, help="sample runs (comma separated)")
    ap.add_argument("--stats-runs", default="", help="larger runs for pass rates and blind baselines")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    census = json.loads(Path(a.census).read_text())
    per_scene: Dict[str, List[int]] = defaultdict(list)
    for sc in census:
        for g, c in (sc.get("stats") or {}).items():
            per_scene[g].append(int(c.get("combos_cpu", 0)))
    runs = [Path(r) for r in a.runs.split(",") if r]
    sruns = [Path(r) for r in a.stats_runs.split(",") if r] or runs
    sample = [q for r in runs for q in _jsonl(r / "questions.jsonl")]
    statq = [q for r in sruns for q in _jsonl(r / "questions.jsonl")]
    rend, ver = Counter(), Counter()
    fail: Dict[str, Counter] = defaultdict(Counter)
    for r in sruns:
        s = json.loads((r / "stats.json").read_text())
        for sid, v in s["scenes"].items():
            for g, c in (v.get("stats") or {}).items():
                rend[g] += c.get("rendered_combos", 0)
                ver[g] += c.get("verified_combos", 0)
            for g, c in (v.get("fail") or {}).items():
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
            "combos_per_scene_median": med, "combos_per_scene_range": [min(combos), max(combos)] if combos else None,
            "combos_total_20": int(sum(combos)) if combos else None, "scenes_with_combos": sum(1 for c in combos if c > 0),
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
