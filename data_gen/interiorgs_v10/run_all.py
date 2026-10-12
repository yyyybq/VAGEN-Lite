"""Run the v10 generator over many InteriorGS scenes with a process pool.

Resumable: each scene writes ``<out>/scenes/<scene_id>.json`` atomically and is
skipped next time.  Sharding (``--shard i --num-shards n``) lets several jobs
split the scene list deterministically.  ``--aggregate`` rebuilds the
dataset-level report from whatever scene files exist.

    python -m data_gen.interiorgs_v10.run_all \
        --scenes-root /mnt/umm/users/yinbaiqiao/InteriorGS_meta --out OUT --workers 32
    python -m data_gen.interiorgs_v10.run_all --out OUT --aggregate

Outputs (in OUT):
    scenes/<id>.json          rows + funnel + summary per scene
    tasks.jsonl               all task rows (after --aggregate)
    summary.json              coverage, totals, per-type / per-d* counts, merged funnel
    per_scene.csv             one line per scene
    env_structure_overrides.json  {scene_id: sign} the RL env must pass to CollisionDetector
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .contract import Funnel


def find_scene_dirs(root: Path) -> List[Path]:
    root = Path(root)
    if (root / "labels.json").is_file():
        return [root]
    out = sorted(p.parent for p in root.rglob("labels.json"))
    return out


def _scene_worker(scene_dir: str, out_dir: str, cfg_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    # one scene per call; keep BLAS single-threaded inside pool workers
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    import contextlib
    import io

    from .run_scene import RunConfig, _jsonable, generate_scene

    cfg = RunConfig(**cfg_kwargs)
    sid = Path(scene_dir).name
    target = Path(out_dir) / "scenes" / f"{sid}.json"
    with contextlib.redirect_stdout(io.StringIO()):
        res = generate_scene(scene_dir, sid, cfg)
    tmp = target.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(_jsonable(res)))
    tmp.replace(target)
    return res["summary"]


def run(scenes_root: Path, out: Path, *, workers: int, shard: int = 0, num_shards: int = 1,
        limit: Optional[int] = None, only: Optional[Sequence[str]] = None,
        cfg_kwargs: Optional[Dict[str, Any]] = None) -> None:
    out = Path(out)
    (out / "scenes").mkdir(parents=True, exist_ok=True)
    dirs = find_scene_dirs(scenes_root)
    if only:
        keep = set(only)
        dirs = [d for d in dirs if d.name in keep]
    dirs = [d for i, d in enumerate(dirs) if i % num_shards == shard]
    if limit:
        dirs = dirs[:limit]
    todo = [d for d in dirs if not (out / "scenes" / f"{d.name}.json").exists()]
    print(f"[run_all] {len(dirs)} scenes in shard, {len(todo)} to do, workers={workers}", flush=True)
    t0 = time.time()
    done = 0
    cfg_kwargs = cfg_kwargs or {}
    if workers <= 1:
        for d in todo:
            s = _scene_worker(str(d), str(out), cfg_kwargs)
            done += 1
            print(f"[run_all] {done}/{len(todo)} {s['scene_id']} {s['status']} tasks={s['n_tasks']} "
                  f"{s['seconds']}s", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futs = {pool.submit(_scene_worker, str(d), str(out), cfg_kwargs): d for d in todo}
            for fut in as_completed(futs):
                d = futs[fut]
                done += 1
                try:
                    s = fut.result()
                    print(f"[run_all] {done}/{len(todo)} {s['scene_id']} {s['status']} tasks={s['n_tasks']} "
                          f"{s['seconds']}s elapsed={time.time() - t0:.0f}s", flush=True)
                except Exception as exc:  # worker crash: record and continue
                    (out / "scenes" / f"{d.name}.crash.txt").write_text(repr(exc))
                    print(f"[run_all] {done}/{len(todo)} {d.name} CRASH {exc!r}", flush=True)


def aggregate(out: Path, *, target_per_scene: int = 11, total_target: int = 10000,
              n_scenes_expected: Optional[int] = None) -> Dict[str, Any]:
    out = Path(out)
    funnel = Funnel()
    per_scene: List[Dict[str, Any]] = []
    overrides: Dict[str, float] = {}
    n_rows = 0
    by_type: Dict[str, int] = {}
    by_d: Dict[str, int] = {}
    status_counts: Dict[str, int] = {}
    reserve: List[Dict[str, Any]] = []
    with (out / "tasks.jsonl").open("w") as fh:
        for p in sorted((out / "scenes").glob("*.json")):
            res = json.loads(p.read_text())
            s = res["summary"]
            for stage, reasons in res.get("funnel", {}).items():
                for reason, n in reasons.items():
                    funnel.add(stage, reason, n)
            for r in res.get("rows", []):
                fh.write(json.dumps(r) + "\n")
                n_rows += 1
                by_type[r["task_type"]] = by_type.get(r["task_type"], 0) + 1
                d = str((r.get("difficulty") or {}).get("d_star"))
                by_d[d] = by_d.get(d, 0) + 1
            reserve.extend(res.get("reserve_rows") or [])
            conv = s.get("convention") or {}
            if conv.get("env_override_required"):
                overrides[s["scene_id"]] = float(conv.get("sign"))
            status_counts[s.get("status", "?")] = status_counts.get(s.get("status", "?"), 0) + 1
            per_scene.append({
                "scene_id": s.get("scene_id"), "status": s.get("status"), "n_tasks": s.get("n_tasks", 0),
                "n_objects": s.get("n_objects"), "n_pairs": s.get("n_pairs"), "n_items": s.get("n_items"),
                "n_items_with_endpoints": s.get("n_items_with_endpoints"), "seconds": s.get("seconds"),
                "convention_status": conv.get("status"), "error": s.get("error"),
            })
    n_scenes = len(per_scene)
    expected = n_scenes_expected or n_scenes
    with_tasks = sum(1 for r in per_scene if r["n_tasks"] > 0)
    at_target = sum(1 for r in per_scene if r["n_tasks"] >= target_per_scene)
    summary = {
        "scenes_processed": n_scenes,
        "scenes_expected": expected,
        "scenes_with_tasks": with_tasks,
        "scene_coverage": round(with_tasks / max(expected, 1), 4),
        "scenes_at_target": at_target,
        "target_per_scene": target_per_scene,
        "tasks_total": n_rows,
        "tasks_target": total_target,
        "tasks_by_type": dict(sorted(by_type.items())),
        "tasks_by_d_star": dict(sorted(by_d.items())),
        "scene_status": status_counts,
        "env_structure_overrides": len(overrides),
        "funnel": funnel.to_dict(),
    }
    if reserve:  # only runs with RunConfig.reserve_k > 0 produce reserve rows
        with (out / "tasks_reserve.jsonl").open("w") as fh:
            for r in reserve:
                fh.write(json.dumps(r) + "\n")
        r_type: Dict[str, int] = {}
        r_d: Dict[str, int] = {}
        for r in reserve:
            r_type[r["task_type"]] = r_type.get(r["task_type"], 0) + 1
            d = str((r.get("difficulty") or {}).get("d_star"))
            r_d[d] = r_d.get(d, 0) + 1
        summary["reserve_total"] = len(reserve)
        summary["reserve_by_type"] = dict(sorted(r_type.items()))
        summary["reserve_by_d_star"] = dict(sorted(r_d.items()))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    (out / "env_structure_overrides.json").write_text(json.dumps(overrides, indent=1, sort_keys=True))
    with (out / "per_scene.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_scene[0].keys()) if per_scene else ["scene_id"])
        w.writeheader()
        for r in per_scene:
            w.writerow(r)
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scenes-root")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--target-per-scene", type=int, default=14)
    ap.add_argument("--scene-time-budget", type=float, default=600.0)
    ap.add_argument("--families", default="projective_relations,absolute_positioning,screen_occupancy,"
                                          "equidistance,occlusion_alignment")
    ap.add_argument("--include-fov", action="store_true")
    ap.add_argument("--aggregate", action="store_true", help="only rebuild summary from scene files")
    ap.add_argument("--expected-scenes", type=int)
    # optional features (defaults reproduce the default generation exactly)
    ap.add_argument("--reserve-k", type=int, default=0, help="reserve episodes per scene -> tasks_reserve.jsonl")
    ap.add_argument("--family-quotas", default=None, help='per-scene "family:min:max,..." (max * = unlimited)')
    ap.add_argument("--family-quota-soft-max", action="store_true",
                    help="a family may exceed its max when no other family can fill the scene")
    ap.add_argument("--terminal-identifiable", choices=("all", "family"), default="all")
    ap.add_argument("--motion-model", choices=("r1", "horizon_locked"), default="r1",
                    help="horizon_locked is generator-only; the RL env implements r1 only")
    ap.add_argument("--pitch-deg", type=float, default=0.0)
    ap.add_argument("--diagnostics", action="store_true", help="keep per-item search reports in scene summaries")
    args = ap.parse_args(argv)
    if not args.aggregate:
        if not args.scenes_root:
            ap.error("--scenes-root is required unless --aggregate")
        cfg_kwargs = {
            "target_tasks_per_scene": args.target_per_scene,
            "scene_time_budget_s": args.scene_time_budget,
            "families": tuple(f for f in args.families.split(",") if f),
            "include_fov": bool(args.include_fov),
            "reserve_k": args.reserve_k,
            "family_quotas": args.family_quotas,
            "family_quota_soft_max": bool(args.family_quota_soft_max),
            "terminal_identifiable": args.terminal_identifiable,
            "motion_model": args.motion_model,
            "pitch_deg": args.pitch_deg,
            "diagnostics": bool(args.diagnostics),
        }
        run(Path(args.scenes_root), Path(args.out), workers=args.workers, shard=args.shard,
            num_shards=args.num_shards, limit=args.limit, only=args.only, cfg_kwargs=cfg_kwargs)
    summary = aggregate(Path(args.out), n_scenes_expected=args.expected_scenes)
    print(json.dumps({k: v for k, v in summary.items() if k != "funnel"}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
