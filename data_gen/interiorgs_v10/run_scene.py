"""Per-scene v10 generation: scene -> objects/refs -> items -> endpoints -> certified episodes.

Every stage writes reasons into one Funnel so that a full run produces a
scene-by-scene, reason-by-reason account of where candidates are lost.

    from data_gen.interiorgs_v10.run_scene import RunConfig, generate_scene
    result = generate_scene(scene_dir, cfg=RunConfig())
    result["rows"]     # task rows (contract.TASK_ROW_REQUIRED_KEYS + lineage)
    result["funnel"]   # {stage: {reason: count}}
    result["summary"]  # per-scene counts, timings, convention record

CLI (one scene, mainly for debugging):
    python -m data_gen.interiorgs_v10.run_scene --scene-dir DIR [--out out.json]
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from . import contract as C
from .contract import Funnel, stable_hash


@dataclasses.dataclass
class RunConfig:
    object_preset: str = "v10_relaxed"
    # task families; projective/fov use the frozen v1 judge, the others canonical v2
    families: Tuple[str, ...] = (
        "projective_relations",
        "absolute_positioning",
        "screen_occupancy",
        "equidistance",
        "occlusion_alignment",
    )
    include_fov: bool = False  # FOV stays provisional until its human-negative calibration lands
    target_tasks_per_scene: int = 14
    max_projective_items: int = 40
    max_items_per_new_family: int = 8
    max_endpoints: int = 8
    endpoint_mode: str = "v10"
    scene_time_budget_s: float = 600.0
    seed: int = 0
    require_usable_convention: bool = True


def _jsonable(x: Any) -> Any:
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    return x


def _projective_items(scene: Any, pairs: Sequence[Tuple[Any, Any]], cfg: RunConfig,
                      funnel: Funnel) -> List[Dict[str, Any]]:
    from .sampling import fov_item, projective_item

    rng = np.random.default_rng(cfg.seed)
    out: List[Dict[str, Any]] = []
    for a, b in pairs:
        for relation in ("left", "right"):
            out.append(projective_item(scene.scene_id, a, b, relation))
        if cfg.include_fov:
            out.append(fov_item(scene.scene_id, a, b))
    if len(out) > cfg.max_projective_items:
        keep = sorted(rng.choice(len(out), size=cfg.max_projective_items, replace=False))
        funnel.add("items", "projective_over_budget_dropped", len(out) - len(keep))
        out = [out[i] for i in keep]
    funnel.add("items", "projective_items", len(out))
    return out


def _family_items(scene: Any, family_name: str, objs: Sequence[Any], cfg: RunConfig,
                  funnel: Funnel) -> List[Dict[str, Any]]:
    from .families.canonical_v2 import FAMILIES
    from .families.common import enumerate_tuples

    fam = FAMILIES[family_name]
    symmetric = family_name in ("equidistance", "size_distance_invariance", "centering")
    tuples = enumerate_tuples(fam, objs, seed=cfg.seed, max_tuples=4 * cfg.max_items_per_new_family,
                              funnel=funnel, ordered=not symmetric)
    rng = np.random.default_rng(cfg.seed + 7)
    items: List[Dict[str, Any]] = []
    for tup in tuples:
        menu = fam.param_menu(tup, funnel)
        if not menu:
            continue
        params = menu[int(rng.integers(len(menu)))]
        items.append(fam.build_item(scene.scene_id, tup, params, None))
        if len(items) >= cfg.max_items_per_new_family:
            break
    funnel.add("items", f"{family_name}_items", len(items))
    return items


def _endpoints_for(scene: Any, item: Dict[str, Any], cfg: RunConfig, funnel: Funnel
                   ) -> Tuple[List[np.ndarray], Any]:
    """Success poses and a success predicate for one item."""
    task_type = item["task_type"]
    if task_type in ("projective_relations", "fov_inclusion"):
        from .endpoints import sample_endpoints, success_fn

        eps, f = sample_endpoints(scene, item, mode=cfg.endpoint_mode, max_endpoints=cfg.max_endpoints,
                                  seed=cfg.seed)
        funnel.merge(f)
        return [np.asarray(e["c2w"], float) for e in eps], success_fn(item, scene)
    from .families.canonical_v2 import FAMILIES

    fam = FAMILIES[task_type]
    cands = fam.success_pose_candidates(scene, item, cfg.seed, max_candidates=cfg.max_endpoints, funnel=funnel)

    def _sfn(c2w: Any, _item: Dict[str, Any] = item) -> bool:
        return bool(fam.judge(_item, np.asarray(c2w, float))["success"])

    return [np.asarray(c["c2w"], float) for c in cands], _sfn


def _task_row(scene: Any, item: Dict[str, Any], ep: Dict[str, Any]) -> Dict[str, Any]:
    row = json.loads(json.dumps(_jsonable(item)))
    row["init_camera"] = {"intrinsics": C.NATIVE_K, "extrinsics": _jsonable(ep["init_c2w"])}
    row["difficulty"] = _jsonable(ep["difficulty"])
    row["initial_observability"] = _jsonable(ep.get("initial_observability"))
    row["turns_audit_only"] = _jsonable(ep.get("turns"))
    row["camera_model_version"] = C.CAMERA_MODEL_VERSION
    row["action_protocol_version"] = C.ACTION_PROTOCOL_VERSION
    row["generator_version"] = C.GENERATOR_VERSION
    row["structure_convention"] = {
        k: _jsonable(v) for k, v in (getattr(scene, "convention_v2", None) or scene.convention).items()
        if k in ("sign", "status", "version", "reason", "usable", "env_override_required", "conflict_with_v1")
    }
    # Read by vagen env.reset: it checks version + sign and, with patches/env_structure_override.patch,
    # applies the calibrated sign for scenes the v1 rule leaves ambiguous.
    from vagen.envs.active_spatial.collision_detector import COLLISION_CONVENTION_VERSION

    conv = getattr(scene, "convention_v2", None) or {}
    row["collision_convention"] = {
        "version": COLLISION_CONVENTION_VERSION,
        "structure_y_sign": float(scene.detector.structure_y_sign),
        "requires_env_override": bool(conv.get("env_override_required", False)),
        "calibration_version": conv.get("version"),
    }
    row["task_id"] = "v10_" + stable_hash({k: row[k] for k in (
        "scene_id", "task_type", "target_object", "target_region", "init_camera")})[:24]
    return row


def generate_scene(scene_dir: Any, scene_id: Optional[str] = None, cfg: Optional[RunConfig] = None
                   ) -> Dict[str, Any]:
    from .initial_states import episodes_for_scene
    from .objects import candidate_objects, select_pairs, select_singles
    from .scene import load_scene

    cfg = cfg or RunConfig()
    t0 = time.perf_counter()
    funnel = Funnel()
    timings: Dict[str, float] = {}
    summary: Dict[str, Any] = {"scene_id": scene_id or Path(scene_dir).name, "status": "started"}
    rows: List[Dict[str, Any]] = []

    def _tick(name: str, start: float) -> float:
        now = time.perf_counter()
        timings[name] = round(timings.get(name, 0.0) + now - start, 3)
        return now

    try:
        t = time.perf_counter()
        scene = load_scene(Path(scene_dir), scene_id)
        summary["scene_id"] = scene.scene_id
        conv = getattr(scene, "convention_v2", None) or scene.convention
        summary["convention"] = {k: _jsonable(v) for k, v in conv.items()
                                 if k in ("sign", "status", "usable", "conflict_with_v1", "env_override_required")}
        t = _tick("load", t)
        if cfg.require_usable_convention and not conv.get("usable", conv.get("status") == "frozen"):
            funnel.add("scene", "convention_unusable")
            summary.update(status="convention_unusable", n_tasks=0, tasks_by_type={}, tasks_by_d_star={},
                           seconds=round(time.perf_counter() - t0, 2))
            return {"rows": rows, "funnel": funnel.to_dict(), "summary": summary, "timings": timings}

        objs, f = candidate_objects(scene, cfg.object_preset)
        funnel.merge(f)
        pairs, f = select_pairs(scene, objs, cfg.object_preset, seed=cfg.seed)
        funnel.merge(f)
        singles, f = select_singles(scene, objs, cfg.object_preset)
        funnel.merge(f)
        summary.update(n_objects=len(objs), n_pairs=len(pairs), n_singles=len(singles))
        t = _tick("objects", t)

        items: List[Dict[str, Any]] = []
        if "projective_relations" in cfg.families:
            items += _projective_items(scene, pairs, cfg, funnel)
        for fam in cfg.families:
            if fam == "projective_relations":
                continue
            try:
                items += _family_items(scene, fam, objs, cfg, funnel)
            except Exception as exc:  # a broken family must not kill the scene
                funnel.add("items", f"{fam}_error:{type(exc).__name__}")
        summary["n_items"] = len(items)
        t = _tick("items", t)

        entries = []
        for k, item in enumerate(items):
            if time.perf_counter() - t0 > cfg.scene_time_budget_s * 0.5:
                funnel.add("endpoints", "scene_time_budget_skip", len(items) - k)
                break
            try:
                eps, sfn = _endpoints_for(scene, item, cfg, funnel)
            except Exception as exc:
                funnel.add("endpoints", f"error:{type(exc).__name__}")
                continue
            funnel.add("endpoints", f"{item['task_type']}:{'has_endpoint' if eps else 'no_endpoint'}")
            if eps:
                entries.append({"item": item, "success_fn": sfn, "endpoints": eps, "key": k})
        summary["n_items_with_endpoints"] = len(entries)
        # interleave task types so the round-robin episode allocator mixes families
        by_type: Dict[str, List[Dict[str, Any]]] = {}
        for e in entries:
            by_type.setdefault(e["item"]["task_type"], []).append(e)
        entries = []
        queues = list(by_type.values())
        while any(queues):
            for q in queues:
                if q:
                    entries.append(q.pop(0))
        t = _tick("endpoints", t)

        remaining = max(10.0, cfg.scene_time_budget_s - (time.perf_counter() - t0))
        episodes, f, report = episodes_for_scene(scene, entries, target_per_scene=cfg.target_tasks_per_scene,
                                                 seed=cfg.seed, time_budget_s=remaining)
        funnel.merge(f)
        t = _tick("episodes", t)
        by_key = {e["key"]: e["item"] for e in entries}
        for ep in episodes:
            item = by_key.get(ep.get("item_key"))
            if item is None:
                funnel.add("rows", "episode_without_item")
                continue
            rows.append(_task_row(scene, item, ep))
        seen = set()
        uniq = []
        for r in rows:
            if r["task_id"] in seen:
                funnel.add("rows", "duplicate_task_id")
                continue
            seen.add(r["task_id"])
            uniq.append(r)
        rows = uniq
        summary["shortfall"] = _jsonable(report.get("shortfall"))
        summary["status"] = "ok" if rows else "no_tasks"
    except Exception as exc:
        funnel.add("scene", f"error:{type(exc).__name__}")
        summary["status"] = "error"
        summary["error"] = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        summary["traceback"] = traceback.format_exc()[-2000:]
    summary["n_tasks"] = len(rows)
    by_type: Dict[str, int] = {}
    by_band: Dict[str, int] = {}
    for r in rows:
        by_type[r["task_type"]] = by_type.get(r["task_type"], 0) + 1
        d = (r.get("difficulty") or {}).get("d_star")
        by_band[str(d)] = by_band.get(str(d), 0) + 1
    summary["tasks_by_type"] = by_type
    summary["tasks_by_d_star"] = by_band
    summary["seconds"] = round(time.perf_counter() - t0, 2)
    return {"rows": rows, "funnel": funnel.to_dict(), "summary": summary, "timings": timings}


def main(argv: Optional[Sequence[str]] = None) -> int:  # pragma: no cover - CLI
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene-dir", required=True)
    ap.add_argument("--scene-id")
    ap.add_argument("--out")
    ap.add_argument("--target", type=int, default=RunConfig.target_tasks_per_scene)
    args = ap.parse_args(argv)
    cfg = RunConfig(target_tasks_per_scene=args.target)
    res = generate_scene(args.scene_dir, args.scene_id, cfg)
    text = json.dumps(_jsonable(res), indent=1)
    if args.out:
        Path(args.out).write_text(text)
    print(json.dumps(res["summary"], indent=1))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
