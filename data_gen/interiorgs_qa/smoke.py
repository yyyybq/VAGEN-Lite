"""GPU smoke test: render a few constructed views with RGB + depth and compare with the box proxy.

    python -m data_gen.interiorgs_qa.smoke --out OUT --scenes S1,S2 --gs-root PLY
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from .common import pairs_near, single_pool
from .render import CachedRenderer, GSRenderer, ProxyRenderer
from .scene_ctx import SceneCtx
from .views import ViewBank, frame_candidates


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--scenes", required=True)
    ap.add_argument("--gs-root", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    gs = GSRenderer(args.gs_root)
    rows = []
    for sid in args.scenes.split(","):
        t0 = time.time()
        ctx = SceneCtx(sid)
        gs.set_scene(sid)
        R = CachedRenderer(gs)
        bank = ViewBank(ctx, R, out)
        proxy = ProxyRenderer(ctx)
        pool = single_pool(ctx)
        n = 0
        for a, b in pairs_near(ctx, pool):
            cams = frame_candidates(ctx, [a, b])
            if not cams:
                continue
            t = time.time()
            vid, va = bank.view(cams[0], "smoke")
            dt = time.time() - t
            bank.save(vid)
            _, pdepth, palpha = proxy.render(cams[0], 512)
            valid = (va.alpha > 0.5) & (palpha > 0.5)
            diff = np.abs(va.depth - pdepth)[valid]
            row = {"scene": sid, "view": vid, "render_s": round(dt, 3), "objects": {}}
            for o in (a, b):
                row["objects"][ctx.disp[o]] = va.obj(o).to_json()
            row["depth_vs_proxy"] = {"median_abs_m": float(np.median(diff)), "p90_abs_m": float(np.percentile(diff, 90)),
                                     "gs_depth_range": [float(va.depth[va.alpha > 0.5].min()), float(va.depth.max())],
                                     "alpha_mean": float(va.alpha.mean())}
            rows.append(row)
            print(json.dumps(row), flush=True)
            n += 1
            if n >= 4:
                break
        print("scene", sid, "seconds", round(time.time() - t0, 1), flush=True)
    (out / "smoke.json").write_text(json.dumps(rows, indent=1))
    bank.dump(out / "views.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
