"""Driver: CPU census of combos, and question generation with rendering (GPU, or the box proxy on CPU).

    # how many object combos each group finds per scene (CPU, many processes)
    python -m data_gen.interiorgs_qa.pipeline census --scenes-file F --out OUT [--workers 32]
    # sample questions: render, verify on depth, emit (GPU job; --proxy for a CPU dry run)
    python -m data_gen.interiorgs_qa.pipeline render --scenes-file F --out OUT --quota quota.json --gs-root PLY

Output of ``render`` (OUT/): questions.jsonl, views.jsonl, views/<scene>/*.png|*.depth.npz, stats.json.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

from .common import GenCtx
from .scene_ctx import SceneCtx, load_split
from .views import ViewBank

SPLIT_MANIFEST = Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/"
                      "interiorgs_v10_gen_20261014/split_manifest.json")


def group_table() -> Dict[str, Any]:
    from . import multi, single, video

    t = {}
    t.update({k: ("single", v) for k, v in single.GROUPS.items()})
    t.update({k: ("multi", v) for k, v in multi.GROUPS.items()})
    t.update({k: ("video", v) for k, v in video.GROUPS.items()})
    return t


def subtasks_of(group: str) -> List[str]:
    from . import multi, single, video

    for mod in (single, multi, video):
        if group in mod.SUBTASKS:
            return mod.SUBTASKS[group]
    raise KeyError(group)


class _NoRender:
    kind = "none"

    def render(self, *a, **k):
        raise RuntimeError("census does not render")

    def set_scene(self, s):
        pass


def census_scene(args) -> Dict[str, Any]:
    sid, groups, split = args
    from .single import Runner

    t0 = time.time()
    out: Dict[str, Any] = {"scene_id": sid}
    try:
        ctx = SceneCtx(sid)
        out["naming"] = ctx.naming_census()
        g = GenCtx(ctx, ViewBank(ctx, _NoRender(), Path("/tmp")), split)
        g.census = True
        table = group_table()
        for grp in groups:
            kind, (combos, realize) = table[grp]
            t = time.time()
            r = Runner(g, grp, {}, 0, census=True)
            r.run(combos(g), realize(r))
            g.stats[grp]["seconds"] = round(time.time() - t, 2)
        out["stats"] = {k: dict(v) for k, v in g.stats.items()}
    except Exception as exc:  # report, keep going
        out["error"] = f"{type(exc).__name__}: {exc}"
        out["traceback"] = traceback.format_exc()
    out["seconds"] = round(time.time() - t0, 1)
    return out


def cmd_census(a: argparse.Namespace) -> None:
    scenes = [l.strip() for l in Path(a.scenes_file).read_text().splitlines() if l.strip()]
    split = load_split(SPLIT_MANIFEST)
    groups = a.groups.split(",") if a.groups else list(group_table())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(s, groups, split.get(s, "?")) for s in scenes]
    with mp.get_context("fork").Pool(min(a.workers, len(jobs))) as pool:
        res = pool.map(census_scene, jobs, chunksize=1)
    (out / "census.json").write_text(json.dumps(res, indent=1))
    tot: Dict[str, Counter] = defaultdict(Counter)
    for r in res:
        for grp, c in (r.get("stats") or {}).items():
            tot[grp].update({k: v for k, v in c.items() if isinstance(v, (int, float))})
    print(json.dumps({k: dict(v) for k, v in tot.items()}, indent=1))
    errs = [r["scene_id"] for r in res if "error" in r]
    if errs:
        print("errors:", errs)


def cmd_render(a: argparse.Namespace) -> None:
    from .render import CachedRenderer, GSRenderer, ProxyRenderer
    from .single import Runner

    scenes = [l.strip() for l in Path(a.scenes_file).read_text().splitlines() if l.strip()]
    split = load_split(SPLIT_MANIFEST)
    quota_total: Dict[str, Dict[str, int]] = json.loads(Path(a.quota).read_text())
    per_scene_cap = a.per_scene
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    table = group_table()
    remaining = {g: Counter(q) for g, q in quota_total.items()}
    stats_all: Dict[str, Any] = {"scenes": {}, "renderer": "proxy" if a.proxy else "gs"}
    base = None if a.proxy else GSRenderer(a.gs_root)
    qf = open(out / "questions.jsonl", "a")
    for sid in scenes:
        t0 = time.time()
        try:
            ctx = SceneCtx(sid)
            if a.proxy:
                R = CachedRenderer(ProxyRenderer(ctx))
            else:
                base.set_scene(sid)
                R = CachedRenderer(base)
            bank = ViewBank(ctx, R, out, size=a.size)
            g = GenCtx(ctx, bank, split.get(sid, "?"))
            for grp in quota_total:
                if a.groups and grp not in a.groups.split(","):
                    continue
                kind, (combos, realize) = table[grp]
                quota = {s: min(per_scene_cap, n) for s, n in remaining[grp].items() if n > 0}
                t = time.time()
                r = Runner(g, grp, quota, a.verify_cap)
                try:
                    r.run(combos(g), realize(r))
                except Exception as exc:
                    g.reject(grp, "exception:" + type(exc).__name__)
                    stats_all.setdefault("exceptions", []).append({"scene": sid, "group": grp,
                                                                   "traceback": traceback.format_exc()})
                for s, n in quota.items():
                    remaining[grp][s] -= n - max(r.need[s], 0)
                g.stats[grp]["seconds"] = round(time.time() - t, 2)
            for q in g.questions:
                qf.write(json.dumps(q.to_json(), default=_np) + "\n")
            qf.flush()
            bank.dump(out / "views.jsonl")
            stats_all["scenes"][sid] = {"stats": {k: dict(v) for k, v in g.stats.items()},
                                        "fail": {k: dict(v) for k, v in g.fail.items()},
                                        "renders": R.renders, "questions": len(g.questions),
                                        "naming": ctx.naming_census(), "seconds": round(time.time() - t0, 1)}
        except Exception as exc:
            stats_all["scenes"][sid] = {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
        (out / "stats.json").write_text(json.dumps(stats_all, indent=1, default=_np))
        print(sid, stats_all["scenes"][sid].get("questions"), "questions", round(time.time() - t0, 1), "s", flush=True)
        if all(v <= 0 for c in remaining.values() for v in c.values()) and not a.all_scenes:
            break
    stats_all["remaining_quota"] = {g: {s: n for s, n in c.items() if n > 0} for g, c in remaining.items()}
    (out / "stats.json").write_text(json.dumps(stats_all, indent=1, default=_np))
    qf.close()


def _np(x):
    import numpy as np

    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.bool_):
        return bool(x)
    raise TypeError(type(x))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("census")
    c.add_argument("--scenes-file", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--groups", default="")
    c.add_argument("--workers", type=int, default=20)
    r = sub.add_parser("render")
    r.add_argument("--scenes-file", required=True)
    r.add_argument("--out", required=True)
    r.add_argument("--quota", required=True)
    r.add_argument("--groups", default="")
    r.add_argument("--per-scene", type=int, default=1)
    r.add_argument("--verify-cap", type=int, default=4)
    r.add_argument("--size", type=int, default=512)
    r.add_argument("--proxy", action="store_true")
    r.add_argument("--all-scenes", action="store_true", help="visit every scene even when quotas are met")
    r.add_argument("--gs-root", default="/mnt/umm/users/yinbaiqiao/InteriorGS_ply")
    a = ap.parse_args()
    {"census": cmd_census, "render": cmd_render}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
