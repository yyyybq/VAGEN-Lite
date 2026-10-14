"""Who can be named in a question (user prompt, section 5.3), on the CPU census scenes.

1. Reproduce the first census: of all labels.json boxes, how many get a scene-unique v10_r5 description
   (COVERAGE.md 5.1: 441 of 6,693 with the no-task list of 2026-10-13), and split the rest by the rule that
   drops them (objects.candidate_objects funnel: no-task list, semantic exclusions, size rules, then the
   referring rules).
2. Single-image questions only need the name to be unique in the image: count the objects that pass the
   single-object filters but have no scene-unique description, and of those how many can be shown alone --
   some legal camera sees the object clearly (v10 face_visible_v3) while no other object a reader would
   call by the same name is seen at all.

    python -m data_gen.interiorgs_qa.naming_census --scenes-file F --out OUT.json
"""

from __future__ import annotations

import argparse
import json
import math
import multiprocessing as mp
from collections import Counter
from pathlib import Path
from typing import Any, Dict

import numpy as np

from .geom import CAMERA_HEIGHT_M, camera_K, look_at
from .scene_ctx import NO_TASK_LIST, NO_TASK_LIST_R5, SceneCtx

K = camera_K(512)


def image_unique_possible(ctx: SceneCtx, oid: str) -> bool:
    box = ctx.obb[oid]
    eyes = []
    for r in (1.2, 1.7, 2.3, 3.0, 3.8):
        for k in range(24):
            a = 2 * math.pi * k / 24
            eyes.append(box.center[:2] + r * np.array([math.cos(a), math.sin(a)]))
    eyes = np.asarray(eyes)
    eyes = eyes[ctx.legal(eyes)]
    if not len(eyes):
        return False
    poses = np.stack([look_at((e[0], e[1], CAMERA_HEIGHT_M), box.center) for e in eyes])
    ok = ctx.face_visible(oid, poses, K, 512)
    if not ok.any():
        return False
    poses = poses[ok]
    alone = np.ones(len(poses), bool)
    for other in ctx.same_name(oid):
        if other in ctx.obb and ctx.duplicates.get(other) != oid:
            alone &= ~ctx.face_seen_any(other, poses, K, 512)
            if not alone.any():
                return False
    return bool(alone.any())


def one(args) -> Dict[str, Any]:
    sid, = args
    out: Dict[str, Any] = {"scene_id": sid}
    for tag, lst in (("no_task_20261013", NO_TASK_LIST_R5), ("no_task_20261014", NO_TASK_LIST)):
        ctx = SceneCtx(sid, no_task_list=lst)
        rec = {"boxes": len(ctx.objects), "funnel": ctx.funnel, "scene_named": len(ctx.named),
               "eligible": len(ctx.eligible)}
        if tag == "no_task_20261014":
            extra = sorted(ctx.eligible - set(ctx.named))
            rec["eligible_not_scene_named"] = len(extra)
            rec["image_unique_possible"] = sum(image_unique_possible(ctx, o) for o in extra)
            named_ok = [o for o in ctx.named if o in ctx.obb]
            rec["scene_named_with_box"] = len(named_ok)
            # how the scene-named objects are described
            rec["ref_methods"] = dict(Counter(ctx.named[o].ref_method for o in ctx.named))
        out[tag] = rec
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes-file", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=20)
    a = ap.parse_args()
    scenes = [l.strip() for l in Path(a.scenes_file).read_text().splitlines() if l.strip()]
    with mp.get_context("fork").Pool(min(a.workers, len(scenes))) as pool:
        res = pool.map(one, [(s,) for s in scenes], chunksize=1)
    tot: Dict[str, Counter] = {}
    for tag in ("no_task_20261013", "no_task_20261014"):
        c = Counter()
        f = Counter()
        for r in res:
            rec = r[tag]
            c.update({k: v for k, v in rec.items() if isinstance(v, int)})
            for stage, reasons in rec["funnel"].items():
                for k, v in reasons.items():
                    f[f"{stage}:{k}"] += v
            if "ref_methods" in rec:
                for k, v in rec["ref_methods"].items():
                    c["ref_method:" + k] += v
        tot[tag] = {"totals": dict(c), "funnel": dict(sorted(f.items()))}
    Path(a.out).write_text(json.dumps({"scenes": res, "totals": tot}, indent=1))
    print(json.dumps({k: v["totals"] for k, v in tot.items()}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
