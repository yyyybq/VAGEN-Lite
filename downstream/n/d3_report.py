#!/usr/bin/env python3
"""D3 (a) axis results (D3_PLAN_20261106 section 4; the user's 2026-11-07 item 3): each N1 model on the variants
(same start, target or relation swapped; d3_gen.py) with history and N1's robot body, against its own (g-d) episodes
on the originals of the same instances.  No simulator; reads episodes, writes OUT/results.json and OUT/report.md.

    python downstream/n/d3_report.py --variants GEN/variants.jsonl --eval EVAL/gd --orig N3/gd \
        --originals N1.jsonl N1B.jsonl --models B S RS RD Q3B Q3S Q3RS Q3RD [--exclude FILE ...] --out OUT

Per model and variant (keyed by the original's instance id): strict success (the variant's formal judge), relaxed
success (max shaping score >= 0.5, EVAL_PROTOCOL section 6), wrong target (the episode ends where the original's
box judge passes and the variant's formal judge does not; final_judge_original), wrong target at any turn (some
pose after a turn passes the original's box judge, recomputed from turn_log; supplementary), and the original's
success in the (g-d) episodes (both successful = both).  Paired: variant against original per model (exact McNemar
and the paired interval, report.stats.paired; the interval from resampling whole scenes, g_report.scene_boot), and
between models on the variants (per generation SFT, sparse and dense RL against the base, sparse and dense against
SFT; Holm over the 5 in each generation, set before the run in the plan).
Versions (instance sets): all; without left / right; each --exclude file (instance ids of the originals; the
exclusion version is per table, all-bad the sensitivity check, EVAL_PROTOCOL section 7); without variants near the
threshold; without the variants whose own success poses already pass the original's box judge (both_solvable);
without originals in the suspect scenes; without partial-scene variants; variants whose shortest path is not
longer than the original's, and those whose shortest path is longer.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

GENERATIONS = (("B", "S", "RS", "RD"), ("Q3B", "Q3S", "Q3RS", "Q3RD"))
SUSPECT_SCENES = ("PreSoakPan_s1004", "SteamInMicrowave_s1000", "WashLettuce_s1000", "WashLettuce_s3003")
RELAXED = 0.5


def read_jsonl(path) -> list:
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def episodes(folder: Path, model: str) -> dict:
    p = Path(folder) / model / "episodes.jsonl"
    return {r["instance_id"]: r for r in read_jsonl(p) if not r.get("error")} if p.is_file() else {}


def original_of(variant_id: str) -> str:
    return variant_id[:-len("_d3a")] if variant_id.endswith("_d3a") else variant_id


def any_turn_original(ep: dict, item: dict, judge) -> bool:
    """Some pose after a turn (or the start) passes the original's box judge."""
    poses = [t.get("pose_after") for t in ep.get("turn_log") or [] if t.get("pose_after")]
    if ep.get("turn_log"):
        poses.insert(0, ep["turn_log"][0].get("pose_before"))
    return any(p is not None and judge(item, p)["success"] for p in poses)


def per_model(var: dict, orig: dict, items: dict, judge=None) -> dict:
    """{original id: row} for the variants this model has an episode on (and the original episode)."""
    rows = {}
    for vid, ep in var.items():
        oid = original_of(vid)
        o = orig.get(oid)
        if o is None:
            continue
        fj = ep.get("final_judge_original") or {}
        strict = bool(ep["success"])
        rows[oid] = {"strict": strict, "relaxed": float(ep.get("max_shaping_score") or 0.0) >= RELAXED,
                     "wrong_target": bool(fj.get("success")) and not strict, "original": bool(o["success"]),
                     "any_turn_original": (any_turn_original(ep, items[oid], judge) and not strict) if judge else None}
    return rows


def both_solvable(variants: list, judge) -> set:
    """Originals whose variant's own shortest or best success pose already passes the original's box judge."""
    out = set()
    for v in variants:
        item = v["d3"]["original_item"]
        if any(judge(item, v[t]["pose"])["success"] for t in ("shortest", "best")):
            out.add(v["d3"]["original"])
    return out


def summarise(rows: dict, ids: list, scenes: dict) -> dict:
    from report import stats
    import g_report as G

    ids = [i for i in ids if i in rows]
    n = len(ids)
    if not n:
        return {"n": 0}
    k = lambda f: sum(bool(rows[i][f]) for i in ids)  # noqa: E731
    failed = [i for i in ids if not rows[i]["strict"]]
    p = stats.paired({i: rows[i]["strict"] for i in ids}, {i: rows[i]["original"] for i in ids})
    p["scene"] = G.scene_boot({i: int(rows[i]["strict"]) - int(rows[i]["original"]) for i in ids}, scenes)
    out = {"n": n, "strict": k("strict"), "relaxed": k("relaxed"), "original": k("original"),
           "both": sum(rows[i]["strict"] and rows[i]["original"] for i in ids),
           "wrong_target": k("wrong_target"), "failed": len(failed),
           "wrong_target_of_failed": sum(rows[i]["wrong_target"] for i in failed) / len(failed) if failed else math.nan,
           "variant_minus_original": p}
    if all(rows[i]["any_turn_original"] is not None for i in ids):
        out["any_turn_original"] = k("any_turn_original")
    return out


def compare(rows_a: dict, rows_b: dict, ids: list, scenes: dict) -> dict:
    from report import stats
    import g_report as G

    ids = [i for i in ids if i in rows_a and i in rows_b]
    p = stats.paired({i: rows_a[i]["strict"] for i in ids}, {i: rows_b[i]["strict"] for i in ids})
    p["scene"] = G.scene_boot({i: int(rows_a[i]["strict"]) - int(rows_b[i]["strict"]) for i in ids}, scenes)
    return p


def comparisons(models: list) -> list:
    out = []
    for gen in GENERATIONS:
        base, sft, rs, rd = gen
        if all(m in models for m in gen):
            out.append([(sft, base), (rs, base), (rd, base), (rs, sft), (rd, sft)])
    return out


def versions(variants: list, originals: dict, excludes: dict, both: set) -> dict:
    ids = [v["d3"]["original"] for v in variants]
    by = {v["d3"]["original"]: v for v in variants}
    partial = {v["d3"]["original"] for v in variants if v.get("d3", {}).get("partial_scene")}
    out = {"all": ids, "no_left_right": [i for i in ids if by[i]["task_type"] != "projective_relations"]}
    for name, ex in excludes.items():
        out[f"minus_{name}"] = [i for i in ids if i not in ex]
    out["no_near_threshold"] = [i for i in ids if not by[i].get("near_threshold")]
    out["no_both_solvable"] = [i for i in ids if i not in both]
    out["no_suspect_scenes"] = [i for i in ids if by[i]["scene_id"] not in SUSPECT_SCENES]
    if partial:
        out["no_partial_scenes"] = [i for i in ids if i not in partial]
    out["not_longer"] = [i for i in ids if i in originals
                         and len(by[i]["shortest"]["path"]) <= len(originals[i]["shortest"]["path"])]
    out["longer"] = [i for i in ids if i in originals
                     and len(by[i]["shortest"]["path"]) > len(originals[i]["shortest"]["path"])]
    return out


def run(args) -> dict:
    from report import stats

    variants = read_jsonl(args.variants)
    partial = set(json.loads(Path(args.partial).read_text())) if args.partial else set()
    for v in variants:
        if v["instance_id"] in partial or v["d3"]["original"] in partial:
            v["d3"]["partial_scene"] = True
    originals = {r["instance_id"]: r for f in args.originals for r in read_jsonl(f)}
    items = {v["d3"]["original"]: v["d3"]["original_item"] for v in variants}
    scenes = {v["d3"]["original"]: v["scene_id"] for v in variants}
    judge = None
    if not args.no_judge:
        import n_env as N
        judge = N.box_judge
    both = both_solvable(variants, judge) if judge else set()
    excludes = {Path(f).stem: {x.strip() for x in open(f) if x.strip()} for f in args.exclude or []}
    vers = versions(variants, originals, excludes, both)
    rows = {m: per_model(episodes(args.eval, m), episodes(args.orig, m), items, judge) for m in args.models}
    res = {"models": args.models, "both_solvable": sorted(both), "versions": {}}
    for vname, ids in vers.items():
        r = res["versions"][vname] = {"n_ids": len(ids), "per_model": {}, "between": []}
        for m in args.models:
            r["per_model"][m] = summarise(rows[m], ids, scenes)
        for fam in comparisons(args.models):
            ps = [compare(rows[a], rows[b], ids, scenes) for a, b in fam]
            adj = stats.holm([p.get("p", math.nan) for p in ps])
            for (a, b), p, h in zip(fam, ps, adj):
                r["between"].append(dict(p, a=a, b=b, holm=h))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=1, default=float))
    (out / "report.md").write_text(markdown(res))
    return res


def pct(x) -> str:
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100 * x:+.1f}"


def pfmt(p) -> str:
    if p is None or math.isnan(p):
        return "—"
    return f"{p:.2f}" if p >= 0.1 else f"{p:.3f}" if p >= 0.001 else f"{p:.1e}"


def markdown(res: dict) -> str:
    lines = []
    for vname, r in res["versions"].items():
        lines += [f"## {vname}（{r['n_ids']} 道原题）", "",
                  "| 模型 | n | 变体严格成功 | 原题（(g-d)） | 变体 − 原题（配对 95%；整场景；p） | 宽松 | 错选目标（全部 / 失败中） | 任一轮满足原题（失败中） | 两版都成 |",
                  "| --- | ---: | ---: | ---: | --- | ---: | --- | ---: | ---: |"]
        for m, s in r["per_model"].items():
            if not s.get("n"):
                lines.append(f"| {m} | 0 | | | | | | | |")
                continue
            p = s["variant_minus_original"]
            n = s["n"]
            lines.append(f"| {m} | {n} | {s['strict']}（{100 * s['strict'] / n:.1f}%） | {s['original']}（{100 * s['original'] / n:.1f}%） | "
                         f"{pct(p['diff'])}（{pct(p['low'])} 到 {pct(p['high'])}；{pct(p['scene']['low'])} 到 {pct(p['scene']['high'])}；{pfmt(p['p'])}） | "
                         f"{s['relaxed']} | {s['wrong_target']} / {100 * s['wrong_target_of_failed']:.1f}% | "
                         f"{s.get('any_turn_original', '—')} | {s['both']} |")
        lines += ["", "| 对比（变体上） | 差（配对 95%；整场景） | p | Holm（每代 5 组） |", "| --- | --- | ---: | ---: |"]
        for p in r["between"]:
            lines.append(f"| {p['a']} − {p['b']} | {pct(p['diff'])}（{pct(p['low'])} 到 {pct(p['high'])}；"
                         f"{pct(p['scene']['low'])} 到 {pct(p['scene']['high'])}） | {pfmt(p['p'])} | {pfmt(p['holm'])} |")
        lines.append("")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variants", required=True)
    ap.add_argument("--eval", required=True, help="the variants' episodes: EVAL/gd with <MODEL>/episodes.jsonl")
    ap.add_argument("--orig", required=True, help="the originals' (g-d) episodes: N3/gd")
    ap.add_argument("--originals", nargs="+", required=True, help="N1's instance files (the originals' shortest paths)")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--exclude", nargs="*", default=None, help="files of original instance ids, one version each")
    ap.add_argument("--partial", default=None, help="JSON list of variant or original ids from partial scenes")
    ap.add_argument("--no-judge", action="store_true", help="skip the box-judge measures (tests)")
    ap.add_argument("--out", required=True)
    run(ap.parse_args(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())
