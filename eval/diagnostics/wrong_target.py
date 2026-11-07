"""Wrong-target rate and soft success (the user's instruction of 2026-11-07, item 6b; part of the failure breakdown).

For each failed evaluation episode of a family whose two roles are asymmetric (projective_relations: "A appears to the
<relation> of B"; occlusion_alignment: "A is hidden behind B"), the episode's end pose and its best pose (the pose with
the fewest failed success conditions, as the evaluation recorded them; ties to the earlier, the start pose included)
are judged again by the same task with A and B exchanged (the success region of the other reading of the sentence).
Met at either pose: the episode counts as a wrong-target episode.  The other four families have one target
(absolute_positioning, screen_occupancy) or symmetric roles (equidistance, fov_inclusion), so they have no
wrong-target reading.

The swap exchanges A and B in target_object and in the region's parameters (projective: object_a/b_center;
occlusion: object_a/b_id and _center, occluded / occluder), as exps/matrix_v3/watch/goal_swap.py does, but keeps every
pair: the generator's pair rules for a new occlusion task (new target not too flat, new occluder high enough and not
see-through) decide which tasks may be generated, not how a pose is judged; an episode whose swapped pair breaks one
is counted and flagged (``swap_rule``).

Poses are replayed from the task's init_camera with the episode's recorded actions by the generator's kinematics
(data_gen.interiorgs_v10.difficulty.apply_action_batch) and the runtime's collision rule (a colliding move leaves the
pose unchanged and still uses one of the 12 primitive steps; FastGeometry.transitions_legal), and judged by the env's
own function (target_visibility.score_with_visibility with the scene's face occluders).  Checks per episode: the
replayed collisions per turn equal the recorded ones, and the original task's failed conditions at every replayed turn
end equal the recorded ``failed_gates``; an episode failing a check is reported and left out of the rates.

Soft success (every family): not a success, and at its best pose exactly one condition failed, the family's tolerance
condition (TOLERANCE_GATE: the distance, equal-distance, occupancy band, the projective relation's pixel margin, the
occlusion's hidden share); fov_inclusion has no tolerance condition (its soft success is its strict success).  Strict
success is the evaluation's own ``success``.
"""
from __future__ import annotations

import collections
import contextlib
import copy
import io
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

SWAP_FAMILIES = ("projective_relations", "occlusion_alignment")
TOLERANCE_GATE = {"absolute_positioning": "distance", "equidistance": "equidistant", "screen_occupancy": "occupancy",
                  "projective_relations": "margin", "occlusion_alignment": "target_hidden"}
GS_ROOT = Path(os.environ.get("VAGEN_GS_ROOT", "/mnt/umm/users/yinbaiqiao/InteriorGS_ply"))
MAX_PRIMITIVE_STEPS = 12


def _limit_threads() -> None:
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(k, "1")


def swap_item(row: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    """The task with A and B exchanged, and the generator's pair rules the swapped pair breaks (occlusion only)."""
    objs = row["target_object"]["objects"]
    if row["task_type"] not in SWAP_FAMILIES or len(objs) != 2:
        raise ValueError(f"{row.get('task_id')}: no swap for {row['task_type']} with {len(objs)} objects")
    s = copy.deepcopy(row)
    s["target_object"]["objects"] = [s["target_object"]["objects"][1], s["target_object"]["objects"][0]]
    p = s["target_region"]["params"]
    p["object_a_center"], p["object_b_center"] = p["object_b_center"], p["object_a_center"]
    broken: List[str] = []
    if row["task_type"] == "occlusion_alignment":
        from data_gen.interiorgs_v10.families import occlusion_alignment as OA

        p["object_a_id"], p["object_b_id"] = p["object_b_id"], p["object_a_id"]
        p["occluded"], p["occluder"] = p["occluder"], p["occluded"]
        a, b = s["target_object"]["objects"]
        if a["bbox_max"][2] - a["bbox_min"][2] < OA.MIN_TARGET_HEIGHT_M:
            broken.append("new_target_too_flat")
        if b["bbox_max"][2] - b["bbox_min"][2] < OA.MIN_OCCLUDER_HEIGHT_M:
            broken.append("new_occluder_too_low")
        if OA._non_occluding(b["label"]):
            broken.append("new_occluder_see_through")
    s["task_id"] = row["task_id"] + "_swap"
    return s, broken


def failed_gates(metric: Optional[Dict[str, Any]]) -> Optional[List[str]]:
    if not metric:
        return None
    return sorted(k for k, v in (metric.get("gates") or {}).items() if not v)


class Judge:
    """Scenes, their geometry and face occluders, loaded once each."""

    def __init__(self) -> None:
        self.scenes: Dict[str, Any] = {}

    def scene(self, sid: str) -> Tuple[Any, Any, Any]:
        if sid not in self.scenes:
            from data_gen.interiorgs_v10.difficulty import FastGeometry
            from data_gen.interiorgs_v10.occlusion import face_occluders_for
            from data_gen.interiorgs_v10.scene import load_scene

            with contextlib.redirect_stdout(io.StringIO()):
                sc = load_scene(GS_ROOT / sid, sid)
            self.scenes[sid] = (sc, FastGeometry.for_scene(sc), face_occluders_for(sc))
        return self.scenes[sid]

    def metric(self, item: Dict[str, Any], c2w: Any, occluders: Any) -> Dict[str, Any]:
        import numpy as np

        from vagen.envs.active_spatial.target_visibility import score_with_visibility

        return score_with_visibility(item, np.asarray(c2w, float).reshape(4, 4), occluders)

    def replay(self, row: Dict[str, Any], trajectory: List[Dict[str, Any]], geom: Any) -> Tuple[List[Any], List[int]]:
        """The pose after each turn and each turn's collisions, from init_camera and the recorded actions."""
        import numpy as np

        from data_gen.interiorgs_v10.difficulty import _split, apply_action_batch, to_c2w

        P, R = _split(np.asarray(row["init_camera"]["extrinsics"], float))
        left, poses, cols = MAX_PRIMITIVE_STEPS, [], []
        for t in trajectory:
            n = 0
            for a in t.get("actions") or []:
                if left <= 0:
                    break
                Pn, Rn = apply_action_batch(P, R, a)
                left -= 1
                if a.startswith("move_") and not bool(geom.transitions_legal(P, Pn, R)[0]):
                    n += 1
                    continue
                P, R = Pn, Rn
            poses.append(to_c2w(P, R)[0])
            cols.append(n)
        return poses, cols


def soft_success(episode: Dict[str, Any]) -> bool:
    """Not a success, and at the best pose only the family's tolerance condition failed."""
    if episode.get("success"):
        return False
    gate = TOLERANCE_GATE.get(str(episode.get("task_type")))
    states = [(episode.get("start") or {}).get("failed_gates")] + [t.get("failed_gates") for t in episode.get("trajectory") or []]
    states = [s for s in states if s is not None]
    if not states or gate is None:
        return False
    best = min(states, key=len)
    return best == [gate]


def best_index(episode: Dict[str, Any]) -> int:
    """-1 for the start pose, else the turn whose end pose has the fewest failed conditions (ties to the earlier)."""
    best, k = (episode.get("start") or {}).get("failed_gates"), -1
    for i, t in enumerate(episode.get("trajectory") or []):
        g = t.get("failed_gates")
        if g is not None and (best is None or len(g) < len(best)):
            best, k = g, i
    return k


def judge_episode(judge: Judge, row: Dict[str, Any], episode: Dict[str, Any]) -> Dict[str, Any]:
    """One failed episode of a swap family: the checks, and the swapped task at the end and best poses."""
    sc, geom, occ = judge.scene(row["scene_id"])
    traj = episode.get("trajectory") or []
    poses, cols = judge.replay(row, traj, geom)
    rec_cols = [int(t.get("collisions") or 0) for t in traj]
    gates = [failed_gates(judge.metric(row, c, occ)) for c in poses]
    rec_gates = [t.get("failed_gates") for t in traj]
    out: Dict[str, Any] = {"task_id": row["task_id"], "family": row["task_type"],
                           "collisions_match": cols == rec_cols, "gates_match": gates == rec_gates}
    if not (out["collisions_match"] and out["gates_match"]):
        out["mismatch"] = {"turns": [i for i in range(len(traj)) if cols[i] != rec_cols[i] or gates[i] != rec_gates[i]][:5]}
    swapped, broken = swap_item(row)
    k = best_index(episode)
    init = row["init_camera"]["extrinsics"]
    end_pose = poses[-1] if poses else init
    best_pose = init if k < 0 else poses[k]
    m_end, m_best = judge.metric(swapped, end_pose, occ), judge.metric(swapped, best_pose, occ)
    out.update(best_turn=k, swap_rule=broken,
               swap_end=bool(m_end.get("success")), swap_best=bool(m_best.get("success")),
               swap_end_failed=failed_gates(m_end), swap_best_failed=failed_gates(m_best))
    out["wrong_target"] = out["swap_end"] or out["swap_best"]
    out["swap_soft"] = (not out["wrong_target"]) and (failed_gates(m_best) == [TOLERANCE_GATE[row["task_type"]]]
                                                      or failed_gates(m_end) == [TOLERANCE_GATE[row["task_type"]]])
    return out


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def analyse(tasks: Dict[str, Dict[str, Any]], episodes: Iterable[Dict[str, Any]], judge: Optional[Judge] = None
            ) -> Dict[str, Any]:
    """Per family: episodes, strict and soft successes, and for the swap families the failed episodes judged with the
    roles exchanged (rates over the failed episodes that passed both checks)."""
    _limit_threads()
    judge = judge or Judge()
    fam: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    per: List[Dict[str, Any]] = []
    for e in episodes:
        if "error" in e:
            continue
        f = str(e.get("task_type"))
        for key in (f, "all"):
            fam[key]["episodes"] += 1
            fam[key]["strict"] += int(bool(e.get("success")))
            fam[key]["soft_only"] += int(soft_success(e))
        if f not in SWAP_FAMILIES:
            continue
        rec: Dict[str, Any] = {"task_id": e["task_id"], "family": f, "success": bool(e.get("success"))}
        if not e.get("success"):
            try:
                rec.update(judge_episode(judge, tasks[e["task_id"]], e))
            except Exception as exc:  # recorded; the rest goes on
                rec["error"] = f"{type(exc).__name__}: {exc}"
        per.append(rec)
        for key in (f, "swap_families"):
            c = fam[key]
            c["episodes"] += int(key == "swap_families")
            c["strict"] += int(key == "swap_families" and rec["success"])
            if rec["success"]:
                continue
            c["failed"] += 1
            if "error" in rec or not (rec.get("collisions_match") and rec.get("gates_match")):
                c["unchecked"] += 1
                continue
            c["judged"] += 1
            c["wrong_target"] += int(rec["wrong_target"])
            c["wrong_end"] += int(rec["swap_end"])
            c["wrong_best"] += int(rec["swap_best"])
            c["swap_soft"] += int(rec["swap_soft"])
            c["wrong_with_rule_broken"] += int(rec["wrong_target"] and bool(rec.get("swap_rule")))
    summary = {}
    for k, c in fam.items():
        d = dict(c)
        n = d.get("episodes", 0)
        d["strict_rate"] = d.get("strict", 0) / n if n else None
        d["soft_rate"] = (d.get("strict", 0) + d.get("soft_only", 0)) / n if n and k != "swap_families" else None
        if d.get("judged"):
            d["wrong_target_rate"] = d["wrong_target"] / d["judged"]
        summary[k] = d
    return {"families": summary, "episodes": per}


FAMILY_ORDER = ("projective_relations", "absolute_positioning", "screen_occupancy", "equidistance",
                "occlusion_alignment", "fov_inclusion")


def _pct(x: Optional[float]) -> str:
    return "–" if x is None else f"{100.0 * x:.1f}%"


def table(families: Dict[str, Any]) -> List[str]:
    """The markdown lines of one model's soft success and wrong-target rate."""
    lines = ["严格成功 / 软成功（软：最好状态只差该族的容差那一道门）；错选目标：失败局的终点或最好状态满足 A、B 互换后的任务。", "",
             "| 任务族 | 题数 | 严格成功 | 软成功 | 失败 | 判过 | 错选目标 | 错选率 |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for f in [f for f in FAMILY_ORDER if f in families] + ["all", "swap_families"]:
        d = families.get(f)
        if not d:
            continue
        name = {"all": "合计", "swap_families": "两个可互换的族"}.get(f, f)
        swap = f in SWAP_FAMILIES or f == "swap_families"
        lines.append(f"| {name} | {d.get('episodes', 0)} | {_pct(d.get('strict_rate'))} | {_pct(d.get('soft_rate'))} | "
                     + (f"{d.get('failed', 0)} | {d.get('judged', 0)} | {d.get('wrong_target', 0)} | "
                        f"{_pct(d.get('wrong_target_rate'))} |" if swap else "– | – | – | – |"))
    return lines + [""]


def mcnemar(a: Dict[str, bool], b: Dict[str, bool]) -> Dict[str, Any]:
    """Exact McNemar (two-sided binomial on the discordant tasks) of two runs' wrong-target indicators over the tasks
    both have."""
    keys = sorted(set(a) & set(b))
    only_a = sum(1 for k in keys if a[k] and not b[k])
    only_b = sum(1 for k in keys if b[k] and not a[k])
    n = only_a + only_b
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(0, min(only_a, only_b) + 1)) / 2 ** n)
    return {"tasks": len(keys), "only_first": only_a, "only_second": only_b, "p": p}


def indicators(result: Dict[str, Any]) -> Dict[str, bool]:
    """task_id -> wrong-target episode (a success counts False; an unchecked failure is left out)."""
    out = {}
    for r in result["episodes"]:
        if r["success"]:
            out[r["task_id"]] = False
        elif "error" not in r and r.get("collisions_match") and r.get("gates_match"):
            out[r["task_id"]] = bool(r["wrong_target"])
    return out
