#!/usr/bin/env python3
"""D3 (a) axis (the user's task book of 2026-11-06, docs/downstream/tasks/TASKS_20261106_D1_D3.md): for every N1
instance of the three swappable task types, a variant from the same start in the same scene with the relation
reversed or the target swapped, kept only if it is solvable from that start by N1's own instance search
(n_gen.find: the box judge over the reachable poses, then rendered checks).  Only makes questions; no evaluation.

    python downstream/n/d3_gen.py scenes --instances A.jsonl [B.jsonl ...]        (TASK:SEED per line)
    python downstream/n/d3_gen.py scene --instances ... --task T --seed S --out OUT [--soft-limit 600]
    python downstream/n/d3_gen.py collect --instances ... --out OUT

Rules (RULES; the task book's three):
  left / right (projective_relations): the same two objects in the same roles, the relation reversed;
  absolute distance (absolute_positioning): another object of the scene, the same distance;
  screen occupancy (screen_occupancy): another object of the scene, the same share.
  Not made: equal distance and both-in-view (a swap asks the same question), occlusion (near the floor).
A candidate is dropped if the upstream tuple rules refuse it, if the original's shortest or most comfortable success
pose already passes its box judge (it could not tell the two targets apart), or (inside n_gen.find) if its start
already succeeds; the first candidate n_gen.find solves is the variant.  Another object's order is seeded per instance
(RNG_SEED); at most n_gen's candidate limit per instance.
Scene check before any variant: the rebuilt scene must have the instance's start pose and its objects' boxes (else
"scene_mismatch", no variant), and the original must still succeed at its shortest pose (full judge, rendered; else
"original_fails").
scene writes OUT/scenes/<task>_s<seed>.json after every original (per original: the variant or the reason there is
none, and the search tally), then leaves with os._exit as n_gen does.  collect writes OUT/variants.jsonl (N1's
instance format; instance_id = the original's + SUFFIX; "d3": the original's id, item, params and objects, the rule)
and OUT/summary.json (per type: originals, variants, reasons).
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
import n_env as N  # noqa: E402
import n_gen as G  # noqa: E402
import n_smoke as S  # noqa: E402

RULES = {"projective_relations": "左右关系反向", "absolute_positioning": "绝对距离换目标",
         "screen_occupancy": "画面占比换目标"}
NOT_MADE = {"equidistance": "对调是同一道题", "fov_inclusion": "对调是同一道题", "occlusion_alignment": "近迁移接近地板"}
RNG_SEED = 20261106
SUFFIX = "_d3a"
FLIP = {"left": "right", "right": "left"}
POSE_TOL, BOX_TOL = 1e-6, 1e-4


def read_instances(paths) -> list:
    out = []
    for p in paths:
        out += [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]
    return [r for r in out if r["task_type"] in RULES]


def scene_list(rows: list) -> list:
    return sorted({(r["task"], int(r["seed"])) for r in rows})


def candidates(inst: dict, ents: list, rng: random.Random) -> list:
    """[(entities in role order, params)] for one original, in the order they are tried."""
    t = inst["task_type"]
    by_key = {e["key"]: e for e in ents}
    if t == "projective_relations":
        roles = [by_key.get(o["key"], o) for o in inst["objects"]]
        return [(roles, {**inst["params"], "relation": FLIP[inst["params"]["relation"]]})]
    keep = {o["key"] for o in inst["objects"]}
    others = [e for e in ents if e["key"] not in keep]
    rng.shuffle(others)
    return [([e], dict(inst["params"])) for e in others]


def scene_check(scene, inst: dict) -> str | None:
    """None if the rebuilt scene matches the instance, else the reason."""
    if any(abs(a - b) > POSE_TOL for a, b in zip(scene.start, inst["start_pose"])):
        return "scene_mismatch:start"
    by_key = {e["key"]: e for e in scene.entities}
    for o in inst["objects"]:
        e = by_key.get(o["key"])
        if e is None:
            return "scene_mismatch:object"
        for f in ("bbox_min", "bbox_max"):
            if any(abs(a - b) > BOX_TOL for a, b in zip(e[f], o[f])):
                return "scene_mismatch:box"
    return None


def tells_apart(item: dict, inst: dict) -> bool:
    """False if the original's success poses already pass the candidate's box judge."""
    return not any(N.box_judge(item, inst[k]["pose"])["success"] for k in ("shortest", "best") if inst.get(k))


def variant_for(scene, states: dict, order: list, inst: dict, deadline: float) -> tuple:
    """(variant instance or None, record)."""
    rec = {"original": inst["instance_id"], "task_type": inst["task_type"], "rule": RULES[inst["task_type"]]}
    bad = scene_check(scene, inst)
    if bad:
        return None, {**rec, "reason": bad}
    oj = scene.judge(inst["item"], inst["shortest"]["pose"])
    if not oj["success"]:
        return None, {**rec, "reason": "original_fails", "box": bool(N.box_judge(inst["item"], inst["shortest"]["pose"])
                                                                     ["success"]), "visible": oj.get("visible")}
    sid = f"{scene.task}_s{scene.seed}"
    rng = random.Random(f"{RNG_SEED}:{inst['instance_id']}")
    pre = collections.Counter()
    cands = []
    for ents, params in candidates(inst, scene.entities, rng):
        try:
            item = N.make_item(inst["task_type"], sid, list(ents), params, scene.start)
        except ValueError as exc:  # the upstream tuple rules
            pre["refused:" + str(exc).split(": ", 1)[-1][:40]] += 1
            continue
        if not tells_apart(item, inst):
            pre["original_pose_passes"] += 1
            continue
        cands.append((ents, params))
    rec["candidates"] = len(cands)
    var, tally = G.find(scene, states, order, inst["task_type"], rng, deadline, cands=cands)
    rec["tally"] = {**dict(pre), **dict(tally)}
    if var is None:
        return None, {**rec, "reason": "no_solvable_candidate" if cands else "no_candidate"}
    var["instance_id"] = inst["instance_id"] + SUFFIX
    var["d3"] = {"original": inst["instance_id"], "rule": RULES[inst["task_type"]], "original_item": inst["item"],
                 "original_params": inst["params"], "original_objects": inst["objects"]}
    return var, {**rec, "variant": var["instance_id"], "objects": [o["name"] for o in var["objects"]],
                 "params": var["params"], "shortest_actions": var["shortest"]["actions"]}


def scene_cmd(args) -> int:
    from downstream.eval import closed_loop

    closed_loop.stable_counter_regions()
    out = Path(args.out)
    (out / "scenes").mkdir(parents=True, exist_ok=True)
    mine = [r for r in read_instances(args.instances) if (r["task"], int(r["seed"])) == (args.task, args.seed)]
    sid = f"{args.task}_s{args.seed}"
    path = out / "scenes" / f"{sid}.json"
    t0 = time.time()
    rec = {"task": args.task, "seed": args.seed, "scene_id": sid, "complete": False, "originals": len(mine),
           "variants": {}, "records": []}
    scene = None
    try:
        scene = N.Scene(args.task, args.seed, hide_robot=True)
        reach = S.reachable(scene)
        states = reach["states"]
        order = sorted(states, key=lambda k: states[k][1])
        rec.update(reachable=len(states), capped=reach["capped"], seconds_setup=round(time.time() - t0, 1))
        G.write_json(path, rec)
        for inst in sorted(mine, key=lambda r: r["task_type"]):
            t1 = time.time()
            var, r = variant_for(scene, states, order, inst, t0 + args.soft_limit)
            r["seconds"] = round(time.time() - t1, 1)
            rec["records"].append(r)
            if var is not None:
                rec["variants"][inst["instance_id"]] = var
            rec["seconds"] = round(time.time() - t0, 1)
            G.write_json(path, rec)
            print(f"{inst['instance_id']}: {r.get('variant') or r['reason']} ({r['seconds']} s)", flush=True)
        rec["complete"] = True
    except Exception as exc:  # reported in the scene file, not fatal for the job
        import traceback

        rec["error"] = f"{type(exc).__name__}: {exc}"[:400]
        rec["traceback"] = traceback.format_exc()[-1500:]
    rec["seconds"] = round(time.time() - t0, 1)
    G.write_json(path, rec)
    sys.stdout.flush()
    sys.stderr.flush()
    if scene is not None:
        os._exit(0)  # RoboCasa's env.close() can hang for good (n_gen.scene_cmd)
    return 0


def collect(instances, out: Path) -> dict:
    rows = read_instances(instances)
    scenes = {}
    for p in sorted((Path(out) / "scenes").glob("*.json")):
        if not p.name.endswith((".timeout.json", ".tmp")):
            d = json.loads(p.read_text())
            scenes[d["scene_id"]] = d
    variants, summary = [], {t: collections.Counter() for t in RULES}
    for r in rows:
        s = scenes.get(r["scene_id"])
        rec = next((x for x in (s or {}).get("records", []) if x["original"] == r["instance_id"]), None)
        c = summary[r["task_type"]]
        c["originals"] += 1
        if s is None:
            c["scene_not_run"] += 1
        elif rec is None:
            c["scene_error" if s.get("error") else "not_reached"] += 1
        elif rec.get("variant"):
            c["variants"] += 1
            variants.append(s["variants"][r["instance_id"]])
        else:
            c[rec["reason"]] += 1
    with open(Path(out) / "variants.jsonl", "w") as f:
        for v in sorted(variants, key=lambda v: v["instance_id"]):
            f.write(json.dumps(v, default=str) + "\n")
    rep = {"rules": RULES, "not_made": NOT_MADE, "by_type": {t: dict(c) for t, c in summary.items()},
           "originals": len(rows), "variants": len(variants), "scenes": len(scene_list(rows)),
           "scenes_done": sum(1 for s in scenes.values() if s.get("complete")),
           "scene_seconds": sorted(s.get("seconds", 0) for s in scenes.values())}
    G.write_json(Path(out) / "summary.json", rep)
    return rep


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("scenes", "scene", "collect"):
        s = sub.add_parser(name)
        s.add_argument("--instances", nargs="+", required=True, type=Path)
        if name != "scenes":
            s.add_argument("--out", required=True, type=Path)
        if name == "scene":
            s.add_argument("--task", required=True)
            s.add_argument("--seed", required=True, type=int)
            s.add_argument("--soft-limit", type=float, default=600.0)
    args = ap.parse_args(argv)
    if args.cmd == "scenes":
        for t, s in scene_list(read_instances(args.instances)):
            print(f"{t}:{s}")
        return 0
    if args.cmd == "scene":
        return scene_cmd(args)
    print(json.dumps(collect(args.instances, args.out)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
