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
    # Optional features; the defaults reproduce the 20261011c generation bit for bit.
    reserve_k: int = 0                    # extra reserve episodes per scene (result["reserve_rows"])
    family_quotas: Any = None             # {task_type: (min, max)} or "fam:min:max,..." per scene
    family_quota_soft_max: bool = False   # families may exceed their max when no other family can fill
    terminal_identifiable: str = "all"    # "family": only the family's terminal roles must be identifiable
    motion_model: str = "r1"              # "horizon_locked" is generator-only (the RL env lacks it)
    pitch_deg: float = 0.0                # fixed pitch for horizon_locked
    diagnostics: bool = False             # keep per-item search reports in the summary
    # Budget: "wall_clock" = scene_time_budget_s seconds (20261011c; output depends on machine load);
    # "work" = budget_work_units of counted work (contract.WORK x work_weights), reproducible.
    budget_mode: str = "wall_clock"
    budget_work_units: float = 600.0
    work_weights: Any = None              # {operation: weight}; None -> DEFAULT_WORK_WEIGHTS
    ignored_objects: Optional[str] = None  # configs/ignored_objects_*.json: drop these boxes (20261012 only)
    no_task_objects: Optional[str] = None  # configs/no_task_objects_*.json: keep the boxes, never target/reference
    runtime_success_check: bool = False   # Projective/FOV: reject starts where the env succeeds before d*
    # "box_v1": half of each target's box unhidden (20261011c / 20261012); "env": the env's own rules
    # (vagen/envs/active_spatial/target_visibility.py: one face of each required target and of its reference
    # object clearly visible, the 45-degree left/right rule) for success, start-frame and final-frame checks
    # alike.  "face_v1" is the 2026-10-13 name of "env" (that config no longer reproduces exactly).
    visibility_rule: str = "box_v1"
    # 2026-10-14, round 6 (off by default; generation_r6_20261014.json turns them on):
    # scenes whose structure.json has no walls (outdoor areas, open halls) get no tasks; occlusion alignment
    # starts with its hidden target clearly visible (start_context.py).
    require_structure_walls: bool = False
    oa_start_target_visible: bool = False


_IGNORED_CACHE: Dict[str, Dict[str, List[str]]] = {}


def ignored_object_ids(path: Optional[str], scene_id: str) -> List[str]:
    """``ins_id`` values listed for ``scene_id`` under "scenes" of an object-list file
    (configs/ignored_objects_*.json or configs/no_task_objects_*.json; never the "uncertain" key)."""
    if not path:
        return []
    if path not in _IGNORED_CACHE:
        p = Path(path)
        if not p.is_absolute() and not p.exists():  # config files name it relative to the repository root
            p = Path(__file__).resolve().parents[2] / p
        data = json.loads(p.read_text())
        _IGNORED_CACHE[path] = {str(s): sorted(str(o["ins_id"]) for o in objs)
                                for s, objs in (data.get("scenes") or {}).items()}
    return list(_IGNORED_CACHE[path].get(str(scene_id), []))


# Weight of one counted operation (contract.WORK kinds) in work units.  Fitted (non-negative least
# squares on relative error) to the per-stage seconds of an unbudgeted run of the 100 sample scenes
# with 10 workers on the 14-CPU fujian login pod, so that one unit is about one second of
# single-core time there (cluster_runs/REPORT_FULLRUN_20261012.md).  Kinds fitted to zero are
# still counted (summary["work"]) but do not spend budget.
DEFAULT_WORK_WEIGHTS: Dict[str, float] = {
    "poses_scored": 3.15e-5,
    "visible_fraction": 1.35e-3,
    "family_judge": 2.04e-4,
    "pose_verify": 3.81e-2,
    "occlusion_ray_tests": 7.76e-9,
    "exact_judge": 0.0,
    "reverse_candidates": 0.0,
    "examined": 0.0,
    "certify_states": 0.0,
    "replay": 0.0,
}


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
    symmetric = family_name in ("equidistance", "size_distance_invariance", "centering", "fov_inclusion")
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
    if task_type == "projective_relations" or (task_type == "fov_inclusion"
                                               and item.get("canonical_task_metric_version") == C.CANONICAL_TASK_METRIC_V1):
        from .endpoints import sample_endpoints, success_fn

        eps, f = sample_endpoints(scene, item, mode=cfg.endpoint_mode, max_endpoints=cfg.max_endpoints,
                                  seed=cfg.seed)
        funnel.merge(f)
        return [np.asarray(e["c2w"], float) for e in eps], success_fn(item, scene)
    from .families.canonical_v2 import FAMILIES

    fam = FAMILIES[task_type]
    cands = fam.success_pose_candidates(scene, item, cfg.seed, max_candidates=cfg.max_endpoints, funnel=funnel)

    from . import occlusion

    face_rule = occlusion.env_rule()

    def _sfn(c2w: Any, _item: Dict[str, Any] = item) -> bool:
        C.WORK.add("family_judge")
        c2w = np.asarray(c2w, float)
        if not fam.judge(_item, c2w)["success"]:
            return False
        if not face_rule:
            return True
        # the env's rules: judge AND every required target and reference clearly visible (target_visibility.py)
        from vagen.envs.active_spatial.target_visibility import rule_gates

        C.WORK.add("visible_fraction")
        return bool(rule_gates(_item, c2w, occlusion.face_occluders_for(scene))["ok"])

    return [np.asarray(c["c2w"], float) for c in cands], _sfn


def _task_row(scene: Any, item: Dict[str, Any], ep: Dict[str, Any]) -> Dict[str, Any]:
    from .start_context import apply_refs

    row = json.loads(json.dumps(_jsonable(item)))
    apply_refs(row, ep.get("target_refs"))  # referring rules v3: the start decides "the X" or its room form
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
    if getattr(scene, "ignored_object_ids", None):
        # 20261012 rows only (the env patch that read this was removed on 2026-10-13)
        row["collision_convention"]["ignored_object_ids"] = list(scene.ignored_object_ids)
    model, pitch = C.motion_model()
    if model != "r1":
        # not trainable as-is: the RL env only implements the r1 motion model
        row["action_protocol_version"] = f"{C.ACTION_PROTOCOL_VERSION}+{model}_pitch{pitch:g}"
        row["motion_model"] = {"model": model, "pitch_deg": pitch, "env_supported": False}
    row["task_id"] = "v10_" + stable_hash({k: row[k] for k in (
        "scene_id", "task_type", "target_object", "target_region", "init_camera")})[:24]
    return row


def generate_scene(scene_dir: Any, scene_id: Optional[str] = None, cfg: Optional[RunConfig] = None
                   ) -> Dict[str, Any]:
    from . import occlusion

    cfg = cfg or RunConfig()
    previous = C.motion_model()
    previous_rule = occlusion.visibility_rule()
    C.set_motion_model(cfg.motion_model, cfg.pitch_deg)
    occlusion.set_visibility_rule(cfg.visibility_rule)
    try:
        return _generate_scene(scene_dir, scene_id, cfg)
    finally:
        C.set_motion_model(*previous)
        occlusion.set_visibility_rule(previous_rule)


def _counts(rows: Sequence[Dict[str, Any]]) -> Tuple[Dict[str, int], Dict[str, int]]:
    by_type: Dict[str, int] = {}
    by_band: Dict[str, int] = {}
    for r in rows:
        by_type[r["task_type"]] = by_type.get(r["task_type"], 0) + 1
        d = (r.get("difficulty") or {}).get("d_star")
        by_band[str(d)] = by_band.get(str(d), 0) + 1
    return by_type, by_band


def _generate_scene(scene_dir: Any, scene_id: Optional[str], cfg: RunConfig) -> Dict[str, Any]:
    from .initial_states import episodes_for_scene
    from .objects import candidate_objects, select_pairs, select_singles
    from .scene import load_scene

    t0 = time.perf_counter()
    funnel = Funnel()
    timings: Dict[str, float] = {}
    summary: Dict[str, Any] = {"scene_id": scene_id or Path(scene_dir).name, "status": "started"}
    rows: List[Dict[str, Any]] = []
    reserve_rows: List[Dict[str, Any]] = []
    if cfg.budget_mode not in ("wall_clock", "work"):
        raise ValueError(f"budget_mode must be 'wall_clock' or 'work', not {cfg.budget_mode!r}")
    work_mode = cfg.budget_mode == "work"
    weights = dict(cfg.work_weights or DEFAULT_WORK_WEIGHTS)
    budget = float(cfg.budget_work_units if work_mode else cfg.scene_time_budget_s)
    C.WORK.reset()

    def spent() -> float:
        return C.WORK.units(weights) if work_mode else time.perf_counter() - t0

    def _tick(name: str, start: float) -> float:
        now = time.perf_counter()
        timings[name] = round(timings.get(name, 0.0) + now - start, 3)
        return now

    try:
        t = time.perf_counter()
        sid = scene_id or Path(scene_dir).name
        scene = load_scene(Path(scene_dir), scene_id, ignored_ids=ignored_object_ids(cfg.ignored_objects, sid),
                           no_task_ids=ignored_object_ids(cfg.no_task_objects, sid))
        if scene.ignored_object_ids:
            summary["ignored_object_ids"] = list(scene.ignored_object_ids)
        if scene.no_task_object_ids:
            summary["no_task_object_ids"] = list(scene.no_task_object_ids)
        summary["scene_id"] = scene.scene_id
        conv = getattr(scene, "convention_v2", None) or scene.convention
        summary["convention"] = {k: _jsonable(v) for k, v in conv.items()
                                 if k in ("sign", "status", "usable", "conflict_with_v1", "env_override_required")}
        t = _tick("load", t)
        if cfg.require_structure_walls and not getattr(scene.detector, "occlusion_walls", None):
            funnel.add("scene", "no_structure_walls")
            summary.update(status="no_structure_walls", n_tasks=0, tasks_by_type={}, tasks_by_d_star={},
                           seconds=round(time.perf_counter() - t0, 2))
            return {"rows": rows, "funnel": funnel.to_dict(), "summary": summary, "timings": timings}
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
            if spent() > budget * 0.5:
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

        units_after_endpoints = spent()
        counts_after_endpoints = C.WORK.snapshot()
        remaining = max(10.0, budget - units_after_endpoints)
        episodes, f, report = episodes_for_scene(scene, entries, target_per_scene=cfg.target_tasks_per_scene,
                                                 seed=cfg.seed, time_budget_s=None if work_mode else remaining,
                                                 work_budget=remaining if work_mode else None,
                                                 work_weights=weights if work_mode else None,
                                                 family_quotas=cfg.family_quotas, reserve_k=cfg.reserve_k,
                                                 terminal_identifiable=cfg.terminal_identifiable,
                                                 soft_max=cfg.family_quota_soft_max,
                                                 runtime_success_check=cfg.runtime_success_check,
                                                 **({"oa_start_target_visible": True}
                                                    if cfg.oa_start_target_visible else {}))
        funnel.merge(f)
        t = _tick("episodes", t)
        if work_mode:
            summary["work"] = {
                "budget_units": budget, "units_after_endpoints": round(units_after_endpoints, 3),
                "units_total": round(spent(), 3),
                "endpoint_stage_hit": bool(funnel.counts.get("endpoints", {}).get("scene_time_budget_skip")),
                "initial_state_stage_hit": bool(report.get("timed_out")),
                "main_set_units": round(units_after_endpoints + float(report.get("main_work_units") or 0.0), 3),
                "reserve_hit": bool(report.get("reserve_timed_out")),
                "counts_after_endpoints": counts_after_endpoints, "counts_total": C.WORK.snapshot(),
            }
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
        for rank, ep in enumerate(report.get("reserve") or []):
            item = by_key.get(ep.get("item_key"))
            if item is None:
                continue
            row = _task_row(scene, item, ep)
            if row["task_id"] in seen:
                funnel.add("rows", "duplicate_reserve_task_id")
                continue
            seen.add(row["task_id"])
            row["reserve"] = True
            row["reserve_rank"] = rank
            reserve_rows.append(row)
        summary["shortfall"] = _jsonable(report.get("shortfall"))
        if cfg.family_quotas:
            summary["family_counts"] = report.get("family_counts")
        if cfg.reserve_k > 0:
            summary["reserve_band_counts"] = report.get("reserve_band_counts")
        if cfg.diagnostics:
            summary["band_counts"] = report.get("band_counts")
            summary["per_item"] = _jsonable(report.get("per_item"))
        summary["status"] = "ok" if rows else "no_tasks"
    except Exception as exc:
        funnel.add("scene", f"error:{type(exc).__name__}")
        summary["status"] = "error"
        summary["error"] = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        summary["traceback"] = traceback.format_exc()[-2000:]
    summary["n_tasks"] = len(rows)
    summary["tasks_by_type"], summary["tasks_by_d_star"] = _counts(rows)
    if cfg.reserve_k > 0:
        summary["n_reserve"] = len(reserve_rows)
        summary["reserve_by_type"], summary["reserve_by_d_star"] = _counts(reserve_rows)
    summary["seconds"] = round(time.perf_counter() - t0, 2)
    result = {"rows": rows, "funnel": funnel.to_dict(), "summary": summary, "timings": timings}
    if cfg.reserve_k > 0:
        result["reserve_rows"] = reserve_rows
    return result


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
