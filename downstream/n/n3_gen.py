#!/usr/bin/env python3
"""N3 (the transfer ladder; the user's decision 2 of 2026-11-03): tier (e)'s instances and tier (c)'s solvability check.

    python downstream/n/n3_gen.py scene --task T --seed S --out GEN --camera robot [n_gen.py scene's options]
    python downstream/n/n3_gen.py recheck-scene --task T --seed S --instances FILE --collision sphere|robot --out DIR
    python downstream/n/n3_gen.py recheck-collect --instances FILE --out DIR

scene: n_gen.py scene (one scene's instances, one per task type) with n3_env's camera ("robot": the robot camera's
  height and field of view, level; tier e); every instance records the camera it was made with (instance "camera").
  collect and page are n_gen.py's own.
recheck-scene: for each instance of FILE in this scene (tier c: the merged 600; "robot" re-checks them under N1's own
  rule in a fresh process), first its own shortest path (found under N1's rule) is replayed under the rule (replay):
  if no move is blocked and the full rule holds somewhere along it, the instance is solvable ("via": "path", the first
  such pose).  Otherwise the poses reachable from the start under the rule (n3_env.reachable, at most 12 actions,
  MAX_STATES poses, one search per scene) are scanned breadth first with the box judge; each pose that passes it is
  judged in full (rendering, at most MAX_CHECKS per instance): the first full success at >= 1 action makes it
  solvable ("via": "search").  A search that was capped or ran out of checks without a success proves nothing
  (collected as "unknown", not "unsolvable"; 11-03's first run capped every scene at 15,000 poses under the sphere).
  The camera-A frame at the success pose is checked for colour noise (n_frames: a corrupted render can pass or fail
  the judge falsely), and so is the start frame.  Writes DIR/scenes/<scene>.json, then leaves with
  os._exit (RoboCasa's close can hang, as in n_gen).
recheck-collect: DIR/instances_checked.jsonl (FILE's instances, each with its "recheck"), DIR/solvable.jsonl (those
  solvable under the rule with clean frames: the evaluation's instances) and DIR/summary.json (per type: solvable,
  unsolvable, unknown, corrupted, missing scenes).
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
# poses searched per scene when an instance's own path fails: the sphere lets the camera pass over counters, so its
# reachable area is far larger than the robot's (11-03's first try: 15,000, every scene capped)
MAX_STATES = {"sphere": 120000, "robot": 15000}


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
def frame_record(scene, item, pose, j) -> dict:
    img, _ = scene.render(pose)
    return {"pose": list(pose), "visible": j["visible"], "chroma": round(F.chroma(img), 2)}


def replay(scene, inst: dict) -> dict:
    """The instance's own shortest path (found under N1's rule) replayed under the scene's rule: blocked at which
    action, or the first pose along it where the full rule holds (solvable, via "path"), or neither."""
    item, pose = inst["item"], tuple(inst["start_pose"])
    path = list((inst.get("shortest") or {}).get("path") or [])
    for k, action in enumerate(path):
        new = N.apply_action(pose, action)
        if scene.blocked(pose, new):
            return {"solvable": False, "blocked_at": k + 1, "blocked_action": action}
        pose = new
        if N.box_judge(item, pose)["success"]:
            j = scene.judge(item, pose)
            if j["success"]:
                return dict(frame_record(scene, item, pose, j), solvable=True, via="path", actions=k + 1,
                            path=path[:k + 1])
    return {"solvable": False, "path_without_success": len(path)}


def recheck_instance(scene, states: dict, order: list, inst: dict) -> dict:
    """Solvable under the scene's rule, searched over the reachable poses (see the module text)."""
    item = inst["item"]
    checks, out = 0, {"solvable": False, "checks": 0, "via": "search"}
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
            out.update(frame_record(scene, item, pose, j), solvable=True, actions=depth, path=S.path_to(states, k))
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
        rec["seconds_setup"] = round(time.time() - t0, 1)
        reach = None
        for inst in insts:
            r = replay(scene, inst)
            if not r["solvable"]:  # the own path is blocked or never succeeds: search (one search per scene)
                if reach is None:
                    reach = N3.reachable(scene, max_states=MAX_STATES[args.collision])
                    states = reach["states"]
                    order = sorted(states, key=lambda k: states[k][1])
                    rec.update(reachable=len(states), blocked=reach["blocked"], capped=reach["capped"])
                r = dict(recheck_instance(scene, states, order, inst), replay=r, capped=reach["capped"])
            rec["instances"][inst["instance_id"]] = r
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
            if not r["clean"]:
                r["status"] = "corrupted"
            elif r.get("solvable"):
                r["status"] = "solvable"
            elif r.get("capped") or r.get("stopped"):  # a search that did not see every pose proves nothing
                r["status"] = "unknown"
            else:
                r["status"] = "unsolvable"
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
