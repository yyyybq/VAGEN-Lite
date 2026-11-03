#!/usr/bin/env python3
"""N3 (the transfer ladder; the user's decision 2 of 2026-11-03): tier (e)'s instances and tier (c)'s solvability check.

    python downstream/n/n3_gen.py scene --task T --seed S --out GEN --camera robot [n_gen.py scene's options]
    python downstream/n/n3_gen.py recheck-scene --task T --seed S --instances FILE --collision sphere|robot --out DIR
    python downstream/n/n3_gen.py recheck-collect --instances FILE --out DIR

scene: n_gen.py scene (one scene's instances, one per task type) with n3_env's camera ("robot": the robot camera's
  height and field of view, level; tier e); every instance records the camera it was made with (instance "camera").
  collect and page are n_gen.py's own.
recheck-scene: for each instance of FILE in this scene (tier c: the merged 600; "robot" re-checks them under N1's own
  rule in a fresh process), the poses reachable from the start under the rule (n3_env.reachable, at most 12 actions,
  15,000 poses), scanned breadth first with the box judge; each pose that passes it is judged in full (rendering,
  at most MAX_CHECKS per instance): the first full success at >= 1 action makes the instance solvable under the rule,
  with that shortest path.  Its camera-A frame there is checked for colour noise (n_frames: a corrupted render can
  pass or fail the judge falsely), and so is the start frame.  Writes DIR/scenes/<scene>.json, then leaves with
  os._exit (RoboCasa's close can hang, as in n_gen).
recheck-collect: DIR/instances_checked.jsonl (FILE's instances, each with its "recheck"), DIR/solvable.jsonl (those
  solvable under the rule with clean frames: the evaluation's instances) and DIR/summary.json (per type: solvable,
  unsolvable, corrupted, missing scenes).
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
import n3_env as N3  # noqa: E402
import n_env as N  # noqa: E402
import n_frames as F  # noqa: E402
import n_gen as G  # noqa: E402
import n_smoke as S  # noqa: E402

MAX_CHECKS = 24


def read_jsonl(path) -> list:
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


# ---------------------------------------------------------------------------------------- tier (e)
def scene_cmd(args) -> int:
    """n_gen.scene_cmd with n3_env's Scene (the camera) and the camera recorded on every instance."""
    camera = args.camera
    N.Scene = lambda task, seed, hide_robot=True: N3.Scene(task, seed, hide_robot=hide_robot, camera=camera)
    find = G.find

    def find_with_camera(scene, *a, **k):
        inst, tally = find(scene, *a, **k)
        if inst is not None:
            inst["camera"] = scene.camera_record()
        return inst, tally

    G.find = find_with_camera
    return G.scene_cmd(args)


# ---------------------------------------------------------------------------------------- tier (c)
def recheck_instance(scene, states: dict, order: list, inst: dict) -> dict:
    """Solvable under the scene's rule?  See the module text."""
    item = inst["item"]
    checks, out = 0, {"solvable": False, "checks": 0}
    for k in order:
        pose, depth = states[k][0], states[k][1]
        if depth < 1 or not N.box_judge(item, pose)["success"]:
            continue
        if checks >= MAX_CHECKS:
            out["stopped"] = "checks"
            break
        j = scene.judge(item, pose)
        checks += 1
        if j["success"]:
            img, _ = scene.render(pose)
            out.update(solvable=True, actions=depth, path=S.path_to(states, k), pose=list(pose),
                       visible=j["visible"], chroma=round(F.chroma(img), 2))
            break
    out["checks"] = checks
    return out


def recheck_scene(args) -> int:
    from downstream.eval import closed_loop

    closed_loop.stable_counter_regions()
    out = Path(args.out)
    (out / "scenes").mkdir(parents=True, exist_ok=True)
    sid = f"{args.task}_s{args.seed}"
    insts = [x for x in read_jsonl(args.instances) if x["task"] == args.task and int(x["seed"]) == args.seed]
    t0 = time.time()
    rec = {"scene_id": sid, "collision": args.collision, "instances": {}, "complete": False}
    scene = None
    try:
        hide = bool(insts[0].get("hide_robot", True)) if insts else True
        scene = N3.Scene(args.task, args.seed, hide_robot=hide, collision=args.collision)
        start_img, _ = scene.render(scene.start)
        rec["start_chroma"] = round(F.chroma(start_img), 2)
        if args.collision == "sphere":
            rec["start_inside"] = scene.sphere_hit(scene.start, scene.start)
        reach = N3.reachable(scene)
        states = reach["states"]
        rec.update(reachable=len(states), blocked=reach["blocked"], capped=reach["capped"],
                   seconds_setup=round(time.time() - t0, 1))
        order = sorted(states, key=lambda k: states[k][1])
        for inst in insts:
            rec["instances"][inst["instance_id"]] = recheck_instance(scene, states, order, inst)
        rec["complete"] = True
    except Exception as exc:  # recorded, not fatal for the job
        import traceback

        rec["error"] = f"{type(exc).__name__}: {exc}"[:400]
        rec["traceback"] = traceback.format_exc()[-1500:]
    rec["seconds"] = round(time.time() - t0, 1)
    G.write_json(out / "scenes" / f"{sid}.json", rec)
    sys.stdout.flush()
    sys.stderr.flush()
    if scene is not None:
        os._exit(0)
    return 0


def clean(r: dict, start_chroma) -> bool:
    return (start_chroma is not None and start_chroma <= F.CORRUPT
            and (not r.get("solvable") or r.get("chroma", 0.0) <= F.CORRUPT))


def recheck_collect(instances: Path, out: Path) -> dict:
    out = Path(out)
    scenes = {}
    for p in sorted((out / "scenes").glob("*.json")):
        if not p.name.endswith(".timeout.json"):
            r = json.loads(p.read_text())
            scenes[r["scene_id"]] = r
    rows, solvable = [], []
    summary = {"instances": 0, "per_type": {}, "missing_scenes": [], "scene_errors": {}}
    per = collections.defaultdict(collections.Counter)
    for inst in read_jsonl(instances):
        s = scenes.get(inst["scene_id"])
        if s is None or not s.get("complete"):
            r = {"status": "missing" if s is None else "error"}
            if s is None and inst["scene_id"] not in summary["missing_scenes"]:
                summary["missing_scenes"].append(inst["scene_id"])
            if s is not None:
                summary["scene_errors"][inst["scene_id"]] = s.get("error")
        else:
            r = dict(s["instances"].get(inst["instance_id"]) or {"status": "missing"})
            r["clean"] = clean(r, s.get("start_chroma"))
            r["start_inside"] = s.get("start_inside")
            r["status"] = "corrupted" if not r["clean"] else ("solvable" if r.get("solvable") else "unsolvable")
        per[inst["task_type"]][r["status"]] += 1
        rows.append(dict(inst, recheck=r))
        if r["status"] == "solvable":
            solvable.append(inst)
        summary["instances"] += 1
    summary["per_type"] = {t: dict(per[t]) for t in N.FAMILIES if t in per}
    summary["status"] = dict(sum(per.values(), collections.Counter()))
    summary["collision"] = sorted({s.get("collision") for s in scenes.values()})
    for name, data in (("instances_checked.jsonl", rows), ("solvable.jsonl", solvable)):
        with open(out / name, "w") as f:
            for x in data:
                f.write(json.dumps(x, default=str) + "\n")
    (out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("scene")
    s.add_argument("--task", required=True)
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--soft-limit", type=float, default=1000.0)
    s.add_argument("--families", nargs="+", default=list(N.FAMILIES))
    s.add_argument("--rng-seed", type=int, default=G.SEED)
    s.add_argument("--path-frames", action="store_true")
    s.add_argument("--camera", choices=N3.CAMERAS, default="robot")
    r = sub.add_parser("recheck-scene")
    r.add_argument("--task", required=True)
    r.add_argument("--seed", type=int, required=True)
    r.add_argument("--instances", required=True)
    r.add_argument("--collision", choices=N3.COLLISIONS, required=True)
    r.add_argument("--out", required=True)
    c = sub.add_parser("recheck-collect")
    c.add_argument("--instances", required=True)
    c.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "scene":
        return scene_cmd(args)
    if args.cmd == "recheck-scene":
        return recheck_scene(args)
    print(json.dumps(recheck_collect(Path(args.instances), Path(args.out))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
