"""How many Projective / FOV items would have a success pose at a fixed pitch?

Same method as the synthetic estimate in REPORT_SYNTHETIC.md
(``endpoints._pitch_hypothetical``): every legal position of the target rooms
on the 0.25 m grid x yaws every 10 deg, success = frozen canonical gates AND
the generator occlusion gate.  Items are the ones the pipeline builds
(``run_scene._projective_items`` with its 40-item budget and seed) plus one
FOV item per selected pair (first 20).  Reachability under a motion model is
NOT checked here; the horizon-locked run_all experiment does that.

    python -m data_gen.interiorgs_v10.pitch_audit --scenes-root ROOT --only S1 S2 ... \
        --workers 10 --json pitch_audit.json
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

PITCHES = (0.0, -10.0, -15.0, -20.0, -30.0)
MAX_FOV_ITEMS = 20


def audit_scene(scene_dir: str, pitches: Sequence[float] = PITCHES) -> Dict[str, Any]:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    from .contract import Funnel
    from .endpoints import DEFAULT_BUDGET, _pitch_hypothetical
    from .objects import candidate_objects, select_pairs
    from .run_scene import RunConfig, _projective_items
    from .sampling import fov_item
    from .scene import load_scene

    sid = Path(scene_dir).name
    out: Dict[str, Any] = {"scene_id": sid, "items": {}, "hits": {}}
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            scene = load_scene(Path(scene_dir), sid)
        conv = getattr(scene, "convention_v2", None) or scene.convention
        if not conv.get("usable", conv.get("status") == "frozen"):
            out["status"] = "convention_unusable"
            return out
        cfg = RunConfig()
        objs, _ = candidate_objects(scene, cfg.object_preset)
        pairs, _ = select_pairs(scene, objs, cfg.object_preset, seed=cfg.seed)
        groups = {"projective_relations": _projective_items(scene, pairs, cfg, Funnel()),
                  "fov_inclusion": [fov_item(scene.scene_id, a, b) for a, b in pairs[:MAX_FOV_ITEMS]]}
        for task, items in groups.items():
            out["items"][task] = len(items)
            hits = {str(p): 0 for p in pitches}
            hits["any_nonzero_pitch"] = 0
            hits["only_nonzero_pitch"] = 0
            for item in items:
                h = _pitch_hypothetical(scene, item, pitches, DEFAULT_BUDGET)
                for p, ok in h.items():
                    hits[str(p)] += int(ok)
                nz = any(ok for p, ok in h.items() if p != 0.0)
                hits["any_nonzero_pitch"] += int(nz)
                hits["only_nonzero_pitch"] += int(nz and not h.get(0.0, False))
            out["hits"][task] = hits
        out["status"] = "ok"
    except Exception as exc:  # keep going; the scene is reported
        out["status"] = f"error:{type(exc).__name__}:{exc}"
    return out


def summarize(rows: Sequence[Dict[str, Any]], pitches: Sequence[float] = PITCHES) -> Dict[str, Any]:
    tot: Dict[str, Dict[str, int]] = {}
    for r in rows:
        for task, n in r.get("items", {}).items():
            t = tot.setdefault(task, {"items": 0})
            t["items"] += n
            for k, v in r.get("hits", {}).get(task, {}).items():
                t[k] = t.get(k, 0) + v
    for task, t in tot.items():
        base = max(1, t.get("0.0", 0))
        t["gain_vs_pitch0"] = {str(p): round(t.get(str(p), 0) / base - 1.0, 4) for p in pitches if p != 0.0}
    return {"scenes": len(rows), "by_task": tot,
            "status": {s: sum(1 for r in rows if r.get("status") == s) for s in {r.get("status") for r in rows}}}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes-root", required=True, type=Path)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--json", type=Path)
    args = ap.parse_args(argv)
    dirs: List[str] = sorted(str(p.parent) for p in args.scenes_root.glob("*/labels.json"))
    if args.only:
        keep = set(args.only)
        dirs = [d for d in dirs if Path(d).name in keep]
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        rows = list(pool.map(audit_scene, dirs))
    rep = {"method": "endpoints._pitch_hypothetical on pipeline items (legal 0.25 m grid x 10 deg yaw)",
           "pitches": list(PITCHES), "summary": summarize(rows), "per_scene": rows}
    text = json.dumps(rep, indent=1)
    if args.json:
        args.json.write_text(text)
    print(json.dumps(rep["summary"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
