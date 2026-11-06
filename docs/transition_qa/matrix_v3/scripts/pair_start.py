"""Goal-swap pairs with a shared start (the user's 甲 of 2026-11-06, for analysis 3 and the goal-swap reading).

goal_swap.py found that from the original start only 6 of the 188 swapped tasks of the screened validation and test
sets are reachable within 8 actions.  So each pair (the original task, its A<->B swap from goal_swap.swap_row) gets
a new start pose shared by both versions: the same scene, the same two objects and family, only the roles differ.

The start follows the generator's start rules (initial_states.ItemSearch._examine): a legal pose (scene.is_pose_legal),
neither task met there, at least one of the two objects identifiable (the env's face rule, occlusion.identifiable
under visibility rule "env"), for occlusion each version's hidden object visible (start_context.oa_target_visible's
condition); and both tasks must need 1-8 actions (difficulty.certify_d_star with goal_swap.success_fn_for, d_max 8).
Candidates: the floor grid (0.3 m, scene's legal points) of the two objects' rooms within BAND_M of the 2D line
through their centres and up to BEYOND_M past either end (which of the two appears on the left, and which hides the
other, depends on the side of that line; looking along it the two overlap, so neither version holds yet), facing the
pair's midpoint +-40 degrees in 20-degree steps, in a seeded order; the first candidate that passes is kept, at most
MAX_SEARCHED candidates with both searches a pair (occlusion fewer, its searches are slower).

Splits: val and test = the two families' tasks of the screened sets; train = tasks the matrix v3 RL trained on
(train scenes), in a seeded order until the val split's kept counts per family are reached.  OUT/<split>_standard
and OUT/<split>_swap, each {tasks.jsonl, selection.json}: the same pairs in the same order, task ids <original>_ps and
<original>_ps_swap, the new start in init_camera, difficulty / turns_audit_only / initial_observability recomputed,
"goal_swap_pair" naming the original and its start; OUT/summary.json.

    python pair_start.py OUT [--splits val,test,train] [--limit N] [--workers 13]
"""
import argparse
import collections
import contextlib
import copy
import hashlib
import io
import json
import math
import random
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import goal_swap as GS  # noqa: E402  (puts the matrix worktree on sys.path)

BAND_M, BEYOND_M = 1.0, 3.0
MAX_SEARCHED = {"projective_relations": 100, "occlusion_alignment": 12}
YAW_OFFSETS_DEG = (-40.0, -20.0, 0.0, 20.0, 40.0)  # around the direction to the pair's midpoint


def candidates(scene, row: dict, seed: int) -> list:
    import numpy as np

    from data_gen.interiorgs_v10.contract import pose_from_xy_yaw
    from data_gen.interiorgs_v10.sampling import floor_grid

    objs = row["target_object"]["objects"]
    a = np.asarray(objs[0]["center"][:2], float)
    b = np.asarray(objs[1]["center"][:2], float)
    rooms = {scene.room_of(float(c[0]), float(c[1])) for c in (a, b)} - {None}
    pts = floor_grid(scene, 0.3, room_indices=rooms or None)
    ab = b - a
    L = float(np.linalg.norm(ab)) or 1e-6
    u = ab / L
    rel = pts - a
    along = rel @ u
    across = np.abs(rel[:, 0] * u[1] - rel[:, 1] * u[0])
    keep = (across <= BAND_M) & (((along >= -BEYOND_M) & (along <= 0.0)) | ((along >= L) & (along <= L + BEYOND_M)))
    pts = pts[keep]
    mid = (a + b) / 2
    out = []
    for x, y in pts:  # looking along the line from beyond one end: the two overlap, so neither version holds yet
        base = math.atan2(float(mid[1]) - y, float(mid[0]) - x)
        out += [(float(x), float(y), base + math.radians(d)) for d in YAW_OFFSETS_DEG]
    random.Random(seed).shuffle(out)
    return [pose_from_xy_yaw(x, y, yaw) for x, y, yaw in out]


def work(rows: list) -> list:
    import numpy as np

    from data_gen.interiorgs_v10 import occlusion
    from data_gen.interiorgs_v10.difficulty import as_oracle, certify_d_star
    from data_gen.interiorgs_v10.initial_states import chunk_turns
    from data_gen.interiorgs_v10.scene import load_scene

    occlusion.set_visibility_rule(GS.VISIBILITY_RULE)
    scenes, out = {}, []
    for k, row in enumerate(rows):
        t0 = time.perf_counter()
        rec = {"of": row["task_id"], "family": row["task_type"], "scene_id": row["scene_id"],
               "d_star_original_start": int(row["difficulty"]["d_star"])}
        try:
            swap, why = GS.swap_row(row)
            if swap is None:
                rec["drop"] = why
                out.append(rec)
                continue
            sid = row["scene_id"]
            if sid not in scenes:
                with contextlib.redirect_stdout(io.StringIO()):
                    scenes[sid] = load_scene(GS.GS_ROOT / sid, sid)
            scene = scenes[sid]
            by_id = {str(o.id): o for o in scene.objects}
            ids = [str(o["id"]) for o in row["target_object"]["objects"]]
            f_orig, f_swap = as_oracle(GS.success_fn_for(row, scene)), as_oracle(GS.success_fn_for(swap, scene))
            tried = collections.Counter()
            found = None
            seed = int(hashlib.sha256(f"{GS.SEED}:{row['task_id']}".encode()).hexdigest()[:8], 16)
            for c2w in candidates(scene, row, seed):
                if tried["searched"] >= MAX_SEARCHED[row["task_type"]]:
                    break
                if not bool(scene.is_pose_legal(c2w)[0]):
                    tried["not_legal"] += 1
                    continue
                if f_orig(c2w) or f_swap(c2w):
                    tried["met_at_start"] += 1
                    continue
                obs = {i: occlusion.identifiable(scene, c2w, by_id[i]) for i in ids}
                if sum(bool(o["identifiable"]) for o in obs.values()) < 1:
                    tried["no_target_identifiable"] += 1
                    continue
                if row["task_type"] == "occlusion_alignment" and not all(bool(o["identifiable"]) for o in obs.values()):
                    tried["hidden_target_not_visible"] += 1   # each version hides the other object: both visible
                    continue
                tried["searched"] += 1
                c1 = certify_d_star(scene, f_orig, c2w, d_max=GS.D_MAX, max_terminal_checks=0)
                if c1["d_star"] is None:
                    tried["original_unreachable"] += 1
                    continue
                c2 = certify_d_star(scene, f_swap, c2w, d_max=GS.D_MAX, max_terminal_checks=0)
                if c2["d_star"] is None:
                    tried["swap_unreachable"] += 1
                    continue
                found = (c2w, c1, c2, obs)
                break
            rec["tried"] = dict(tried)
            rec["seconds"] = round(time.perf_counter() - t0, 1)
            if found is None:
                rec["drop"] = "no_shared_start"
                out.append(rec)
                continue
            c2w, c1, c2, obs = found
            rows_out = []
            for version, base, c in (("standard", row, c1), ("swap", swap, c2)):
                r = copy.deepcopy(base)
                r["task_id"] = row["task_id"] + ("_ps" if version == "standard" else "_ps_swap")
                r["init_camera"]["extrinsics"] = np.asarray(c2w, float).tolist()
                r["difficulty"] = {"d_star": int(c["d_star"]), "lower_bound_complete": bool(c["lower_bound_complete"]),
                                   "certificate": list(c["certificate"]), "version": row["difficulty"].get("version"),
                                   "key_mode": "exact", "certificate_source": "bfs"}
                r["turns_audit_only"] = chunk_turns(list(c["certificate"]))
                r["initial_observability"] = {"source": "occlusion.identifiable", "targets": obs}
                r["goal_swap_pair"] = {"of": row["task_id"], "version": version, "start": "shared",
                                       "original_init_extrinsics": row["init_camera"]["extrinsics"],
                                       "original_d_star": int(row["difficulty"]["d_star"])}
                rows_out.append(r)
            rec.update(d_star_standard=int(c1["d_star"]), d_star_swap=int(c2["d_star"]), rows=rows_out)
        except Exception as exc:  # recorded; the split goes on
            rec["drop"] = f"error: {type(exc).__name__}: {exc}"
        out.append(rec)
    return out


def run_split(pool, split: str, want, workers: int, limit: int, families=GS.FAMILIES) -> list:
    rows = [json.loads(l) for l in GS.SOURCES[split].read_text().splitlines() if l.strip()]
    rows = [r for r in rows if r["task_type"] in families]
    if split == "train":
        random.Random(GS.SEED).shuffle(rows)
    if limit:
        rows = rows[:limit]
    if want is None:
        return [x for part in pool.map(work, [rows[i::workers] for i in range(workers)]) for x in part]
    recs = []
    for f in GS.FAMILIES:
        todo, got, i = [r for r in rows if r["task_type"] == f], 0, 0
        while got < want.get(f, 0) and i < len(todo):
            batch = todo[i:i + 2 * workers]
            i += len(batch)
            order = {r["task_id"]: k for k, r in enumerate(batch)}
            res = sorted((x for part in pool.map(work, [batch[j::workers] for j in range(workers)]) for x in part),
                         key=lambda x: order[x["of"]])
            for x in res:
                if "rows" in x and got >= want[f]:
                    continue
                got += "rows" in x
                recs.append(x)
    return recs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--splits", default="val,test,train")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=13)
    ap.add_argument("--families", default=",".join(GS.FAMILIES), help="a timing test of one family")
    a = ap.parse_args()
    fams = tuple(a.families.split(","))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summary = {"rules": __doc__.split("\n\n")[2], "band_m": BAND_M, "max_searched": MAX_SEARCHED, "seed": GS.SEED, "splits": {}}
    kept_val = None
    with Pool(a.workers) as pool:
        for split in a.splits.split(","):
            want = None if split != "train" else (kept_val or {f: 1 for f in GS.FAMILIES})
            recs = run_split(pool, split, want, a.workers, a.limit, fams)
            kept = [x for x in recs if "rows" in x]
            for version, i in (("standard", 0), ("swap", 1)):
                d = out / f"{split}_{version}"
                d.mkdir(parents=True, exist_ok=True)
                (d / "tasks.jsonl").write_text("".join(json.dumps(x["rows"][i], ensure_ascii=False) + "\n" for x in kept),
                                               encoding="utf-8")
                (d / "selection.json").write_text(json.dumps({
                    "split": split, "version": version, "source": str(GS.SOURCES[split]), "pairs": len(kept),
                    "families": dict(collections.Counter(x["family"] for x in kept)), "target_counts": want,
                    "candidates": len(recs), "dropped": dict(collections.Counter(x["drop"] for x in recs if "drop" in x)),
                    "tasks": [{k: v for k, v in x.items() if k != "rows"} for x in recs]}, indent=1, ensure_ascii=False) + "\n",
                    encoding="utf-8")
            fam = collections.Counter(x["family"] for x in kept)
            if split == "val":
                kept_val = dict(fam)
            secs = [x.get("seconds", 0) for x in recs]
            summary["splits"][split] = {"candidates": len(recs), "pairs": len(kept), "families": dict(fam),
                                        "dropped": dict(collections.Counter(x["drop"] for x in recs if "drop" in x)),
                                        "d_star_standard": dict(collections.Counter(x["d_star_standard"] for x in kept)),
                                        "d_star_swap": dict(collections.Counter(x["d_star_swap"] for x in kept)),
                                        "seconds_total": round(sum(secs), 1), "seconds_max": max(secs or [0]),
                                        "target_counts": want}
            print(split, json.dumps(summary["splits"][split], ensure_ascii=False), flush=True)
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
