"""Read-only diagnostics of a finished ``run_all`` output directory.

Answers the pre-GPU questions from the stored scene files only (nothing is
regenerated): d* histograms, how many feasible candidates the lazy search
certified before the per-scene quota stopped it, occlusion_alignment yield,
occlusion blocking, and a simulated scene-group split.

    python -m data_gen.interiorgs_v10.diagnose_run --out RUN_OUT \
        --scenes-root /mnt/umm/users/yinbaiqiao/InteriorGS_meta \
        --r1-frozen exps/vagen_active_spatial/R1-clean-Projective-v0/frozen_v1 \
        --json diag.json
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

FAMILIES = ("projective_relations", "absolute_positioning", "screen_occupancy", "equidistance",
            "occlusion_alignment")
BANDS = ((1, 1), (2, 3), (4, 6), (7, 8))


def _load(out: Path) -> List[Dict[str, Any]]:
    res = []
    for p in sorted((out / "scenes").glob("*.json")):
        r = json.loads(p.read_text())
        r["_path"] = str(p)
        res.append(r)
    return res


def _pct(vals: Sequence[float], q: float) -> float:
    return float(np.percentile(np.asarray(vals, float), q)) if len(vals) else float("nan")


def d_star_tables(scenes: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    overall: Counter = Counter()
    per_family: Dict[str, Counter] = defaultdict(Counter)
    for r in scenes:
        for row in r.get("rows", []):
            d = int((row.get("difficulty") or {}).get("d_star"))
            overall[d] += 1
            per_family[row["task_type"]][d] += 1

    def bands(c: Counter) -> Dict[str, int]:
        return {f"{lo}-{hi}": sum(c[d] for d in range(lo, hi + 1)) for lo, hi in BANDS}
    return {
        "overall": {str(d): overall[d] for d in range(1, 9)},
        "overall_bands": bands(overall),
        "per_family": {f: {str(d): per_family[f][d] for d in range(1, 9)} for f in FAMILIES},
        "per_family_bands": {f: bands(per_family[f]) for f in FAMILIES},
        "per_family_total": {f: sum(per_family[f].values()) for f in FAMILIES},
    }


def candidate_lower_bounds(scenes: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Certified-feasible episodes per scene = accepted + band_quota_full (surplus).

    The search is lazy: it stops as soon as the band targets are full, so this
    is a LOWER bound on the feasible candidates of a scene.
    """
    lb, items_ep = [], []
    for r in scenes:
        s = r["summary"]
        if not (s.get("n_items_with_endpoints") or 0):
            continue
        c = r["funnel"].get("init_candidates", {})
        lb.append(int(c.get("accepted", 0)) + int(c.get("band_quota_full", 0)))
        items_ep.append(int(s.get("n_items_with_endpoints") or 0))
    lb_a = np.asarray(lb)
    return {
        "scenes": len(lb),
        "lower_bound": {"min": int(lb_a.min()), "p10": _pct(lb, 10), "median": _pct(lb, 50),
                        "mean": float(lb_a.mean()), "p90": _pct(lb, 90), "max": int(lb_a.max())},
        "share_ge": {str(k): float((lb_a >= k).mean()) for k in (14, 17, 20, 24, 30)},
        "items_with_endpoints": {"min": int(min(items_ep)), "p10": _pct(items_ep, 10),
                                 "median": _pct(items_ep, 50), "max": int(max(items_ep))},
    }


def occlusion_alignment(scenes: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    tot: Counter = Counter()
    tni_with, tni_without = [], []
    scenes_with = 0
    for r in scenes:
        f = r["funnel"]
        tot["items"] += f.get("items", {}).get("occlusion_alignment_items", 0)
        has = f.get("endpoints", {}).get("occlusion_alignment:has_endpoint", 0)
        tot["items_with_endpoint"] += has
        tot["rows"] += sum(1 for row in r.get("rows", []) if row["task_type"] == "occlusion_alignment")
        for k, v in f.get("family_occlusion_alignment_sampler", {}).items():
            tot[f"sampler:{k}"] += v
        for k, v in f.get("family_occlusion_alignment_params", {}).items():
            tot[f"params:{k}"] += v
        if not (r["summary"].get("n_items_with_endpoints") or 0):
            continue
        tni = f.get("init_candidates", {}).get("terminal_not_identifiable", 0)
        if has:
            scenes_with += 1
            tni_with.append(tni)
        else:
            tni_without.append(tni)
    return {
        "totals": dict(tot),
        "scenes_with_oa_endpoint_item": scenes_with,
        "init_terminal_not_identifiable_per_scene": {
            "scenes_with_oa_endpoint_items": {"n": len(tni_with), "mean": float(np.mean(tni_with)) if tni_with else 0.0,
                                              "sum": int(np.sum(tni_with)) if tni_with else 0},
            "scenes_without": {"n": len(tni_without), "mean": float(np.mean(tni_without)) if tni_without else 0.0,
                               "sum": int(np.sum(tni_without)) if tni_without else 0},
        },
    }


def occlusion_blocking(scenes: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    proj: Counter = Counter()
    fam: Dict[str, Counter] = defaultdict(Counter)
    for r in scenes:
        f = r["funnel"]
        for k, v in f.get("endpoint_item", {}).items():
            proj[k] += v
        for fn in FAMILIES[1:]:
            for k, v in f.get(f"family_{fn}_sampler", {}).items():
                fam[fn][k] += v
            for k, v in f.get(f"family_{fn}_params", {}).items():
                fam[fn][f"params:{k}"] += v
    n_proj = sum(proj.values())
    return {
        "projective_items": n_proj,
        "projective_item_outcomes": dict(proj.most_common()),
        "projective_occlusion_blocked": proj.get("no_endpoint_blocker_occlusion", 0),
        "projective_occlusion_share": proj.get("no_endpoint_blocker_occlusion", 0) / max(1, n_proj),
        "families": {k: dict(v.most_common()) for k, v in fam.items()},
    }


def stuck_scenes(scenes: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for r in scenes:
        s = r["summary"]
        if s.get("status") != "no_tasks" or not (s.get("n_objects") or 0):
            continue
        f = r["funnel"]
        items = {k.replace("_items", ""): v for k, v in f.get("items", {}).items() if k.endswith("_items")}
        eps = f.get("endpoints", {})
        fam_out = {}
        for fn in FAMILIES[1:]:
            smp = f.get(f"family_{fn}_sampler", {})
            prm = f.get(f"family_{fn}_params", {})
            if smp or prm:
                fam_out[fn] = {"params": prm, "sampler": smp}
        init = f.get("init_candidates", {})
        out.append({
            "scene_id": s["scene_id"], "n_objects": s.get("n_objects"), "n_pairs": s.get("n_pairs"),
            "n_items": s.get("n_items"), "n_items_with_endpoints": s.get("n_items_with_endpoints"),
            "stage": ("initial_states" if s.get("n_items_with_endpoints") else
                      ("endpoints" if s.get("n_items") else "items")),
            "items": items,
            "endpoints": {k: v for k, v in eps.items() if v},
            "projective_blockers": f.get("endpoint_item", {}),
            "projective_position": {k: v for k, v in f.get("endpoint_position", {}).items()
                                    if k in ("legal", "illegal_too_close_to_wall", "illegal_outside_rooms",
                                             "illegal_collision_object", "prefilter_relation_side",
                                             "prefilter_distance_band", "prefilter_yaw_interval")},
            "families": fam_out,
            "init_candidates": init,
            "shortfall": s.get("shortfall"),
        })
    return out


_FPS: Dict[str, Any] = {}


def _similarity_job(pair):
    from . import scene_identity

    a, b = pair
    return a, b, scene_identity.similarity(_FPS[a], _FPS[b])


def group_same_source_parallel(fps: Dict[str, Any], workers: int, threshold: float = 0.6,
                               prefilter: float = 0.3):
    """``scene_identity.group_same_source`` (same prefilter, threshold and components)
    with the pairwise similarities spread over a process pool."""
    from multiprocessing import get_context

    from . import scene_identity

    ids = sorted(fps)
    cos = scene_identity._token_cosine([fps[i] for i in ids]) if ids else np.zeros((0, 0))
    todo = []
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            ni, nj = fps[ids[i]]["n"], fps[ids[j]]["n"]
            if cos[i, j] < prefilter or min(ni, nj) < 0.5 * max(ni, nj):
                continue
            todo.append((ids[i], ids[j]))
    _FPS.clear()
    _FPS.update(fps)
    with get_context("fork").Pool(max(1, workers)) as pool:
        sims = pool.map(_similarity_job, todo, chunksize=64)
    parent = {s: s for s in ids}

    def find(s: str) -> str:
        while parent[s] != s:
            parent[s] = parent[parent[s]]
            s = parent[s]
        return s

    pairs = []
    for a, b, s in sims:
        if s >= threshold:
            pairs.append((a, b, s))
            parent[find(a)] = find(b)
    comp: Dict[str, List[str]] = defaultdict(list)
    for s in ids:
        comp[find(s)].append(s)
    groups = sorted(sorted(g) for g in comp.values() if len(g) >= 2)
    return groups, pairs, len(todo)


def split_simulation(scenes: Sequence[Dict[str, Any]], scenes_root: Path,
                     r1_frozen: Optional[Path], workers: int = 1) -> Dict[str, Any]:
    from . import release, scene_identity

    ids = sorted(r["summary"]["scene_id"] for r in scenes)
    tasks = {r["summary"]["scene_id"]: len(r.get("rows", [])) for r in scenes}
    fps = {}
    for sid in ids:
        labels = json.loads((scenes_root / sid / "labels.json").read_text())
        fps[sid] = scene_identity.fingerprint(labels)
    groups, pairs, n_compared = group_same_source_parallel(fps, workers)
    reserved: Dict[str, str] = {}
    if r1_frozen is not None:
        for name, split in (("train.jsonl", "train"), ("eval_policy_rows.jsonl", "test")):
            p = r1_frozen / name
            if p.is_file():
                for line in p.open():
                    sid = json.loads(line).get("scene_id")
                    if sid and sid in fps:
                        reserved[sid] = split
    manifest = release.assign_splits(ids, groups=groups, reserved=reserved)
    lookup = release.split_lookup(manifest)
    aliases = release.groups_to_aliases(groups)
    leaks = []
    by_group = defaultdict(set)
    for sid in ids:
        by_group[aliases.get(sid, sid)].add(lookup[sid])
    for g, sps in by_group.items():
        if len(sps) > 1:
            leaks.append(g)
    table = {}
    for sp in release.SPLITS:
        ss = [s for s in ids if lookup[s] == sp]
        table[sp] = {"scenes": len(ss), "scenes_with_tasks": sum(1 for s in ss if tasks[s] > 0),
                     "tasks": sum(tasks[s] for s in ss)}
    return {
        "pairs_compared": n_compared,
        "same_source_groups": groups,
        "same_source_pairs": [[a, b, round(float(s), 3)] for a, b, s in pairs],
        "scenes_in_groups": sum(len(g) for g in groups),
        "reserved_pins": Counter(reserved.values()),
        "physical_groups": manifest["physical_groups"],
        "splits": table,
        "groups_crossing_splits": leaks,
    }


def run_metrics(scenes: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Headline numbers of one run (main set, reserve pool, items, time)."""
    fam_rows: Counter = Counter()
    fam_res: Counter = Counter()
    d_rows: Counter = Counter()
    d_res: Counter = Counter()
    items: Counter = Counter()
    items_ep: Counter = Counter()
    status: Counter = Counter()
    per_scene_total, secs = [], []
    for r in scenes:
        s = r["summary"]
        status[s.get("status")] += 1
        for row in r.get("rows", []):
            fam_rows[row["task_type"]] += 1
            d_rows[int(row["difficulty"]["d_star"])] += 1
        for row in r.get("reserve_rows", []) or []:
            fam_res[row["task_type"]] += 1
            d_res[int(row["difficulty"]["d_star"])] += 1
        for k, v in r["funnel"].get("endpoints", {}).items():
            if k.endswith(":has_endpoint"):
                items_ep[k.split(":")[0]] += v
            if k.endswith(":has_endpoint") or k.endswith(":no_endpoint"):
                items[k.split(":")[0]] += v
        if s.get("status") == "ok":
            secs.append(float(s.get("seconds") or 0.0))
        per_scene_total.append(len(r.get("rows", [])) + len(r.get("reserve_rows", []) or []))
    tot = np.asarray(per_scene_total or [0])
    return {
        "scenes": len(scenes), "status": dict(status),
        "scenes_with_tasks": sum(1 for r in scenes if r.get("rows")),
        "tasks": sum(fam_rows.values()), "tasks_by_family": {f: fam_rows[f] for f in FAMILIES},
        "tasks_by_d_star": {str(d): d_rows[d] for d in range(1, 9)},
        "reserve": sum(fam_res.values()), "reserve_by_family": {f: fam_res[f] for f in FAMILIES},
        "reserve_by_d_star": {str(d): d_res[d] for d in range(1, 9)},
        "items": sum(items.values()), "items_with_endpoint": sum(items_ep.values()),
        "items_by_family": {f: items[f] for f in FAMILIES},
        "items_with_endpoint_by_family": {f: items_ep[f] for f in FAMILIES},
        "main_plus_reserve_per_scene": {"min": int(tot.min()), "p10": _pct(tot, 10), "median": _pct(tot, 50),
                                        "share_ge_20": float((tot >= 20).mean())},
        "seconds_ok_scene": {"mean": float(np.mean(secs)) if secs else 0.0, "median": _pct(secs, 50),
                             "max": float(max(secs)) if secs else 0.0},
    }


def compare_runs(runs: Sequence[str], reference: Optional[Path]) -> Dict[str, Any]:
    """``LABEL=DIR`` runs on the same scenes; identity of rows vs ``reference`` per run."""
    from .contract import stable_hash

    out: Dict[str, Any] = {}
    ref = {}
    if reference is not None:
        for p in sorted((reference / "scenes").glob("*.json")):
            ref[p.stem] = stable_hash(json.loads(p.read_text()).get("rows", []))
    for spec in runs:
        label, path = spec.split("=", 1)
        scenes = _load(Path(path))
        m = run_metrics(scenes)
        if ref:
            same = [r["summary"]["scene_id"] for r in scenes
                    if r["summary"]["scene_id"] in ref and stable_hash(r.get("rows", [])) == ref[r["summary"]["scene_id"]]]
            m["rows_identical_to_reference"] = len(same)
            m["rows_differ_from_reference"] = sorted(r["summary"]["scene_id"] for r in scenes
                                                     if r["summary"]["scene_id"] in ref
                                                     and r["summary"]["scene_id"] not in set(same))
        out[label] = m
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--compare", nargs="+", metavar="LABEL=DIR", help="tabulate several runs instead")
    ap.add_argument("--reference", type=Path, help="with --compare: run whose rows define identity")
    ap.add_argument("--out", type=Path, help="run_all output directory")
    ap.add_argument("--scenes-root", type=Path, help="needed for the split simulation (fingerprints)")
    ap.add_argument("--r1-frozen", type=Path, help="R1 frozen_v1 dir: pin its train / eval scenes")
    ap.add_argument("--json", type=Path)
    ap.add_argument("--workers", type=int, default=1, help="processes for the pairwise scene similarity")
    args = ap.parse_args(argv)
    if args.compare:
        rep = compare_runs(args.compare, args.reference)
        text = json.dumps(rep, indent=1)
        if args.json:
            args.json.write_text(text)
        print(text)
        return 0
    if args.out is None:
        ap.error("--out is required unless --compare")
    scenes = _load(args.out)
    rep: Dict[str, Any] = {
        "run": str(args.out),
        "scenes": len(scenes),
        "d_star": d_star_tables(scenes),
        "candidates_before_quota": candidate_lower_bounds(scenes),
        "occlusion_alignment": occlusion_alignment(scenes),
        "occlusion_blocking": occlusion_blocking(scenes),
        "stuck_scenes": stuck_scenes(scenes),
    }
    if args.scenes_root:
        rep["splits"] = split_simulation(scenes, args.scenes_root, args.r1_frozen, args.workers)
    text = json.dumps(rep, indent=1, default=lambda o: dict(o) if isinstance(o, Counter) else str(o))
    if args.json:
        args.json.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
