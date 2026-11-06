"""Goal-swap evaluation sets (the user's decisions of 2026-11-06: analysis 3, and the "goal-swap" third evaluation
reading for the main line).

Only the two families whose two roles are asymmetric can swap: projective_relations ("Move so that A appears to the
<relation> of B in your view.") and occlusion_alignment ("Move so that A is hidden behind B from your viewpoint, while
B stays in view.").  absolute_positioning and screen_occupancy have one target; equidistance and fov_inclusion are
symmetric, so exchanging their roles leaves the same task.

A swap keeps the scene, the start pose (init_camera) and the family, exchanges A and B in target_object, in the
target region's parameters (projective: object_a/b_center; occlusion: object_a/b_id and _center, occluded /
occluder) and in the text (the generator's own templates: sampling.projective_description,
families.occlusion_alignment.describe; the refs are each object's own), and recomputes the success region's
first-success depth d* with the generator's exact BFS (difficulty.certify_d_star, d_max 8, as the v10 generator;
the certificate gives turns_audit_only).  Dropped: no success within 8 actions (the region is empty within reach),
the search cap hit, success already at the start (d* = 0), and for occlusion a new pair the family's rule rejects
(new target lower than MIN_TARGET_HEIGHT_M, new occluder lower than MIN_OCCLUDER_HEIGHT_M or see-through; the
other pair rules are symmetric).  Checks: each original's d* recomputed equals its file's, and the original's
certificate end pose fails the swapped task (the two regions are exclusive).

Splits: val = the two families' tasks of eval_val_v2_screened, test = of eval_test_v2_screened; train = tasks the
matrix v3 RL trained on (rl train.jsonl; train scenes), in a seeded order until the val split's family counts are
reached.  OUT/<split>_swap/{tasks.jsonl, selection.json}; OUT/train_standard/ the train pairs' originals (val and
test standard results are the screened sets' own); OUT/summary.json.

    python goal_swap.py OUT [--splits val,test,train] [--limit N] [--workers 4]
"""
import argparse
import collections
import contextlib
import copy
import io
import json
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

ROOT = Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix")
sys.path.insert(0, str(ROOT))
M3 = ROOT / "exps" / "matrix_v3"
GS_ROOT = Path("/mnt/umm/users/yinbaiqiao/InteriorGS_ply")
SOURCES = {"val": M3 / "eval_val_v2_screened" / "tasks.jsonl", "test": M3 / "eval_test_v2_screened" / "tasks.jsonl",
           "train": M3 / "m3_d_s20261019" / "rl_m3ds1s0" / "train.jsonl"}
FAMILIES = ("projective_relations", "occlusion_alignment")
SEED, D_MAX = 20261106, 8
VISIBILITY_RULE = "env"  # the generator's occlusion.set_visibility_rule for these tasks (checked: originals' d* recomputed)


def swap_row(row: dict) -> tuple:
    """(swapped row, None) or (None, why not)."""
    from data_gen.interiorgs_v10.families import occlusion_alignment as OA
    from data_gen.interiorgs_v10.sampling import projective_description

    objs = row["target_object"]["objects"]
    if len(objs) != 2:
        return None, "not_two_objects"
    s = copy.deepcopy(row)
    s["target_object"]["objects"] = [s["target_object"]["objects"][1], s["target_object"]["objects"][0]]
    p = s["target_region"]["params"]
    a, b = s["target_object"]["objects"]
    if row["task_type"] == "projective_relations":
        if projective_description(objs[0]["ref"], objs[1]["ref"], p["relation"]) != row["task_description"]:
            return None, "text_template_mismatch"
        p["object_a_center"], p["object_b_center"] = p["object_b_center"], p["object_a_center"]
        for k in ("object_a_id", "object_b_id"):
            assert k not in p, k
        s["task_description"] = projective_description(a["ref"], b["ref"], p["relation"])
    else:
        if OA.describe(row) != row["task_description"]:
            return None, "text_template_mismatch"
        assert p["occluded"] == p["object_a_id"] == objs[0]["id"] and p["occluder"] == p["object_b_id"] == objs[1]["id"], p
        p["object_a_id"], p["object_b_id"] = p["object_b_id"], p["object_a_id"]
        p["object_a_center"], p["object_b_center"] = p["object_b_center"], p["object_a_center"]
        p["occluded"], p["occluder"] = p["occluder"], p["occluded"]
        if a["bbox_max"][2] - a["bbox_min"][2] < OA.MIN_TARGET_HEIGHT_M:
            return None, "new_target_too_flat"
        if b["bbox_max"][2] - b["bbox_min"][2] < OA.MIN_OCCLUDER_HEIGHT_M:
            return None, "new_occluder_too_low"
        if OA._non_occluding(b["label"]):
            return None, "new_occluder_see_through"
        s["task_description"] = OA.describe(s)
    s["task_id"] = row["task_id"] + "_swap"
    s["goal_swap"] = {"of": row["task_id"], "swap": "A<->B", "original_description": row["task_description"]}
    return s, None


def success_fn_for(item: dict, scene) -> object:
    """The generator's success predicate for an item: canonical v1 rows by difficulty.make_success_fn (Projective v1
    rows with the env's own rules, endpoints.runtime_success_fn); v2 rows by families.canonical_v2 (the env's own
    function for the v2 families, as initial_states._runtime_success_fn notes)."""
    from data_gen.interiorgs_v10.difficulty import make_success_fn
    from data_gen.interiorgs_v10.families.canonical_v2 import score_canonical_task_v2

    if item.get("canonical_task_metric_version") == "canonical_spatial_task_h1_v1":
        if item["task_type"] == "projective_relations":
            from data_gen.interiorgs_v10.endpoints import runtime_success_fn

            return runtime_success_fn(item, scene)
        return make_success_fn(item)
    return lambda c2w: score_canonical_task_v2(item, c2w)


def work(rows: list) -> list:
    import numpy as np

    from data_gen.interiorgs_v10 import occlusion
    from data_gen.interiorgs_v10.difficulty import _replay_pose, _split, as_oracle, certify_d_star
    from data_gen.interiorgs_v10.initial_states import chunk_turns
    from data_gen.interiorgs_v10.scene import load_scene

    occlusion.set_visibility_rule(VISIBILITY_RULE)
    scenes, out = {}, []
    for row in rows:
        t0 = time.perf_counter()
        rec = {"of": row["task_id"], "family": row["task_type"], "scene_id": row["scene_id"],
               "d_star_original": int(row["difficulty"]["d_star"])}
        try:
            s, why = swap_row(row)
            if s is None:
                rec["drop"] = why
                out.append(rec)
                continue
            sid = row["scene_id"]
            if sid not in scenes:
                with contextlib.redirect_stdout(io.StringIO()):
                    scenes[sid] = load_scene(GS_ROOT / sid, sid)
            init = np.asarray(row["init_camera"]["extrinsics"], float)
            cert0 = row["difficulty"].get("certificate") or []
            c0 = certify_d_star(scenes[sid], success_fn_for(row, scenes[sid]), init, d_max=D_MAX, max_terminal_checks=0,
                                upper_bound=len(cert0) or None, hint_actions=cert0 or None)
            rec["d_star_original_recomputed"] = c0["d_star"]
            P0, R0 = _split(init)
            end = _replay_pose(P0, R0, list(c0["certificate"] or cert0))
            swap_fn = as_oracle(success_fn_for(s, scenes[sid]))
            rec["original_end_fails_swap"] = not bool(swap_fn(np.asarray(end, float).reshape(4, 4)))
            c = certify_d_star(scenes[sid], swap_fn, init, d_max=D_MAX, max_terminal_checks=0)
            rec.update(d_star_swap=c["d_star"], lower_bound_complete=bool(c["lower_bound_complete"]),
                       states=c.get("states"), seconds=round(time.perf_counter() - t0, 2))
            if c["d_star"] is None:
                rec["drop"] = "unreachable_within_8" if c["lower_bound_complete"] else "search_cap"
            elif int(c["d_star"]) == 0:
                rec["drop"] = "met_at_start"
            else:
                s["difficulty"] = {"d_star": int(c["d_star"]), "lower_bound_complete": bool(c["lower_bound_complete"]),
                                   "certificate": list(c["certificate"]), "version": row["difficulty"].get("version"),
                                   "key_mode": "exact", "certificate_source": "bfs", "goal_swap_of": row["difficulty"]}
                s["turns_audit_only"] = chunk_turns(list(c["certificate"]))
                rec["row"] = s
        except Exception as exc:  # recorded, the split goes on
            rec["drop"] = f"error: {type(exc).__name__}: {exc}"
        out.append(rec)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--splits", default="val,test,train")
    ap.add_argument("--limit", type=int, default=0, help="at most N candidates a split (a timing test)")
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {"rules": __doc__.split("\n\n")[2], "seed": SEED, "d_max": D_MAX, "splits": {}}
    val_counts = collections.Counter(json.loads(l)["task_type"] for l in SOURCES["val"].read_text().splitlines()
                                     if l.strip() and json.loads(l)["task_type"] in FAMILIES)
    for split in a.splits.split(","):
        rows = [json.loads(l) for l in SOURCES[split].read_text().splitlines() if l.strip()]
        rows = [r for r in rows if r["task_type"] in FAMILIES]
        if split == "train":
            random.Random(SEED).shuffle(rows)
        if a.limit:
            rows = rows[:a.limit]
        recs, want = [], (dict(val_counts) if split == "train" else None)
        with Pool(a.workers) as pool:
            if want is None:
                recs = [x for part in pool.map(work, [rows[i::a.workers] for i in range(a.workers)]) for x in part]
            else:  # train: candidates in the seeded order, batch by batch, until each family has its val count
                todo = {f: [r for r in rows if r["task_type"] == f] for f in FAMILIES}
                for f in FAMILIES:
                    got, i = 0, 0
                    while got < want[f] and i < len(todo[f]):
                        batch = todo[f][i:i + 4 * a.workers]
                        i += len(batch)
                        res = [x for part in pool.map(work, [batch[j::a.workers] for j in range(a.workers)]) for x in part]
                        order = {r["task_id"]: k for k, r in enumerate(batch)}
                        for x in sorted(res, key=lambda x: order[x["of"]]):
                            if "row" in x and got < want[f]:
                                got += 1
                                recs.append(x)
                            elif "row" not in x:
                                recs.append(x)
        kept = [x for x in recs if "row" in x]
        d = out / f"{split}_swap"
        d.mkdir(parents=True, exist_ok=True)
        (d / "tasks.jsonl").write_text("".join(json.dumps(x["row"], ensure_ascii=False) + "\n" for x in kept), encoding="utf-8")
        sel = {"split": split, "source": str(SOURCES[split]), "families": dict(collections.Counter(x["family"] for x in kept)),
               "candidates": len(recs), "kept": len(kept), "dropped": dict(collections.Counter(x["drop"] for x in recs if "drop" in x)),
               "checks": {"original_d_star_same": sum(x.get("d_star_original_recomputed") == x["d_star_original"] for x in kept),
                          "original_end_fails_swap": sum(bool(x.get("original_end_fails_swap")) for x in kept)},
               "target_counts": want, "seed": SEED if split == "train" else None,
               "tasks": [{k: v for k, v in x.items() if k != "row"} | ({"task_id": x["row"]["task_id"]} if "row" in x else {})
                         for x in recs]}
        (d / "selection.json").write_text(json.dumps(sel, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
        if split == "train":
            ids = {x["of"] for x in kept}
            std = [r for r in rows if r["task_id"] in ids]
            sd = out / "train_standard"
            sd.mkdir(parents=True, exist_ok=True)
            (sd / "tasks.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in std), encoding="utf-8")
            (sd / "selection.json").write_text(json.dumps({"split": "train", "source": str(SOURCES[split]), "kept": len(std),
                                                           "of_swap": str(d / "tasks.jsonl"), "task_ids": sorted(ids)}, indent=1) + "\n")
        summary["splits"][split] = {k: sel[k] for k in ("families", "candidates", "kept", "dropped", "checks", "target_counts")}
        secs = [x["seconds"] for x in recs if "seconds" in x]
        summary["splits"][split]["seconds"] = {"total": round(sum(secs), 1), "max": max(secs or [0])}
        print(split, json.dumps(summary["splits"][split], ensure_ascii=False))
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
