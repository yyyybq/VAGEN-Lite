#!/usr/bin/env python3
"""Re-judge the N1 and N3 (e) instance records in a process with the render fix (the render-cause report
RENDER_CAUSE_20261107 item 1; the user's 2026-11-07 item 5).  The records were made by processes whose renderer could
be broken by the cyclic garbage collector (n_env.fresh_render_context, 56270ef2): a visible-area check done on a
garbage frame can make an instance look solvable when it is not, or record the wrong shares.  Here each scene is
rebuilt with the fix on and, for every instance of it, the recorded start, shortest and best poses are judged again
(the box judge and the rendered visibility, n_env.Scene.judge), and the three question images are rendered again and
compared with the recorded ones pixel by pixel.  No search, no model.

    python downstream/n/rejudge.py scenes --set NAME=FILE [NAME=FILE ...]          (NAME:TASK:SEED per line)
    python downstream/n/rejudge.py scene --set NAME=FILE --task T --seed S --out OUT
    python downstream/n/rejudge.py collect --set NAME=FILE [...] --out OUT

A set is an instance file (N1's instances_eval.jsonl / instances_final.jsonl, N3 (e)'s instances_final.jsonl); its
images lie next to it (the instance's "images" paths are relative to the file's folder).  An instance with a
"camera" record (tier e) is rebuilt with n3_env's Scene and that camera; the camera is set for the whole process
(n3_env.use_camera), so one process takes one set's instances of one scene.
Per instance (scene_cmd writes OUT/scenes/<set>/<scene>.json, then leaves with os._exit as n_gen does):
  scene: d3_gen.scene_check (the start pose and the objects' boxes as recorded), else the instance is not judged;
  poses: per tag (start, shortest, best) the success now and recorded, the visible shares now and recorded, and
    whether they agree (same success; every recorded share within VIS_TOL or VIS_REL of the new one);
  holds: the start still fails and both success poses still succeed (the instance is solvable as recorded);
  agrees: holds and every visible share agrees;
  images: per tag, whether the new render equals the recorded image pixel for pixel, the mean and the largest
    absolute difference (0-255); the new images go to OUT/images/<set>/.
collect writes OUT/rejudge.jsonl (one row per instance; a scene that never finished gives rows with "missing") and
OUT/summary.json (per set and task type: judged, holds, agrees, images equal, and the ids that do not hold or whose
images differ).
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

TAGS = ("start", "shortest", "best")
VIS_TOL, VIS_REL = 1e-4, 0.02  # as d3_gen's render check


def read_jsonl(path) -> list:
    return [json.loads(x) for x in Path(path).read_text().splitlines() if x.strip()]


def parse_sets(specs) -> dict:
    out = {}
    for s in specs:
        name, _, path = s.partition("=")
        if not name or not path:
            raise ValueError(f"--set wants NAME=FILE, got {s!r}")
        out[name] = Path(path)
    return out


def scene_keys(sets: dict) -> list:
    """NAME:TASK:SEED for every scene of every set, in file order, each once."""
    seen, out = set(), []
    for name, path in sets.items():
        for r in read_jsonl(path):
            k = f"{name}:{r['task']}:{int(r['seed'])}"
            if k not in seen:
                seen.add(k)
                out.append(k)
    return out


def recorded(inst: dict, tag: str) -> dict:
    """The recorded judge at a tag's pose: {"pose", "success", "visible"}."""
    if tag == "start":
        j = inst.get("start_judge") or {}
        return {"pose": inst["start_pose"], "success": bool(j.get("success")), "visible": j.get("visible")}
    j = (inst.get(tag) or {}).get("judge") or {}
    return {"pose": inst[tag]["pose"], "success": bool(j.get("success")), "visible": j.get("visible")}


def shares_agree(got, rec) -> bool:
    if not rec:
        return True
    got = got or {}
    return all(abs(float(got.get(k, 0.0)) - float(v)) <= max(VIS_TOL, VIS_REL * float(v)) for k, v in rec.items())


def judge_poses(scene, inst: dict) -> dict:
    out = {}
    for tag in TAGS:
        rec = recorded(inst, tag)
        j = scene.judge(inst["item"], rec["pose"])
        out[tag] = {"success": bool(j["success"]), "recorded_success": rec["success"],
                    "visible": j.get("visible"), "recorded_visible": rec["visible"],
                    "agrees": bool(j["success"]) == rec["success"] and shares_agree(j.get("visible"), rec["visible"])}
    return out


def holds(poses: dict) -> bool:
    return (not poses["start"]["success"]) and poses["shortest"]["success"] and poses["best"]["success"]


def compare_image(new, old_path: Path) -> dict:
    import numpy as np
    from PIL import Image

    if not old_path.is_file():
        return {"equal": None, "missing": True}
    old = np.asarray(Image.open(old_path).convert("RGB"), dtype=np.int16)
    new = np.asarray(new, dtype=np.int16)
    if old.shape != new.shape:
        return {"equal": False, "shape": [list(old.shape), list(new.shape)]}
    d = np.abs(old - new)
    return {"equal": bool(not d.any()), "mean_abs": round(float(d.mean()), 4), "max_abs": int(d.max())}


def make_scene(inst: dict):
    import n_env as N

    cam = inst.get("camera")
    if cam:
        import n3_env as N3

        values = {"height_m": cam["height_m"], "fovy_deg": cam["fovy_deg"]}
        return N3.Scene(inst["task"], int(inst["seed"]), hide_robot=bool(inst.get("hide_robot", True)),
                        camera=cam.get("mode", "robot"), camera_values=values)
    return N.Scene(inst["task"], int(inst["seed"]), hide_robot=bool(inst.get("hide_robot", True)))


def judge_instance(scene, inst: dict, img_dir: Path, out_img: Path) -> dict:
    import d3_gen as D
    from PIL import Image

    row = {"instance_id": inst["instance_id"], "task_type": inst["task_type"], "scene_id": inst["scene_id"]}
    mismatch = D.scene_check(scene, inst)
    if mismatch:
        row["scene"] = mismatch
        return row
    row["scene"] = "ok"
    row["poses"] = judge_poses(scene, inst)
    row["holds"] = holds(row["poses"])
    row["agrees"] = row["holds"] and all(p["agrees"] for p in row["poses"].values())
    row["images"] = {}
    for tag in TAGS:
        img, _ = scene.render(tuple(recorded(inst, tag)["pose"]))
        Image.fromarray(img).save(out_img / f"{inst['instance_id']}_{tag}.png")
        rel = (inst.get("images") or {}).get(tag)
        row["images"][tag] = compare_image(img, img_dir / rel) if rel else {"equal": None, "missing": True}
    return row


def scene_cmd(args) -> int:
    import n_env as N
    from downstream.eval import closed_loop

    closed_loop.stable_counter_regions()
    name, path = next(iter(parse_sets(args.set).items()))
    mine = [r for r in read_jsonl(path) if (r["task"], int(r["seed"])) == (args.task, args.seed)]
    sid = f"{args.task}_s{args.seed}"
    out = Path(args.out)
    (out / "scenes" / name).mkdir(parents=True, exist_ok=True)
    out_img = out / "images" / name
    out_img.mkdir(parents=True, exist_ok=True)
    rec_path = out / "scenes" / name / f"{sid}.json"
    t0 = time.time()
    rec = {"set": name, "scene_id": sid, "complete": False, "instances": len(mine), "render_fix": N.render_fix_on(),
           "rows": []}
    scene = None
    try:
        scene = make_scene(mine[0])
        rec["seconds_setup"] = round(time.time() - t0, 1)
        for inst in mine:
            rec["rows"].append(judge_instance(scene, inst, path.parent, out_img))
            rec["seconds"] = round(time.time() - t0, 1)
            rec_path.write_text(json.dumps(rec, indent=1))
        rec["complete"] = True
    except Exception as exc:  # recorded, not fatal for the job
        import traceback

        rec["error"] = f"{type(exc).__name__}: {exc}"[:400]
        rec["traceback"] = traceback.format_exc()[-1500:]
    rec["seconds"] = round(time.time() - t0, 1)
    rec_path.write_text(json.dumps(rec, indent=1))
    print(f"{name}/{sid}: {len(rec['rows'])} of {len(mine)} ({rec['seconds']} s){' ' + rec['error'] if 'error' in rec else ''}",
          flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    if scene is not None:
        os._exit(0)  # RoboCasa's env.close() can hang for good (n_gen.scene_cmd)
    return 0


def collect(sets: dict, out: Path) -> dict:
    rows = []
    for name, path in sets.items():
        done = {}
        for p in (out / "scenes" / name).glob("*.json"):
            r = json.loads(p.read_text())
            for row in r.get("rows", []):
                done[row["instance_id"]] = row
        for inst in read_jsonl(path):
            row = done.get(inst["instance_id"]) or {"instance_id": inst["instance_id"], "task_type": inst["task_type"],
                                                     "scene_id": inst["scene_id"], "missing": True}
            rows.append(dict(row, set=name))
    summary = {}
    for name in sets:
        mine = [r for r in rows if r["set"] == name]
        per = collections.defaultdict(collections.Counter)
        for r in mine:
            for key in ("all", r["task_type"]):
                c = per[key]
                c["instances"] += 1
                if r.get("missing"):
                    c["missing"] += 1
                    continue
                if r.get("scene") != "ok":
                    c["scene_mismatch"] += 1
                    continue
                c["judged"] += 1
                c["holds"] += bool(r["holds"])
                c["agrees"] += bool(r["agrees"])
                imgs = r.get("images", {})
                c["images_all_equal"] += all(v.get("equal") for v in imgs.values())
                for tag in TAGS:
                    c[f"{tag}_image_equal"] += bool((imgs.get(tag) or {}).get("equal"))
        summary[name] = {
            "counts": {k: dict(v) for k, v in per.items()},
            "not_holding": sorted(r["instance_id"] for r in mine if r.get("scene") == "ok" and not r["holds"]),
            "not_agreeing": sorted(r["instance_id"] for r in mine if r.get("scene") == "ok" and not r["agrees"]),
            "images_differ": sorted(r["instance_id"] for r in mine if r.get("scene") == "ok"
                                    and not all(v.get("equal") for v in r.get("images", {}).values())),
            "scene_mismatch": sorted(r["instance_id"] for r in mine if r.get("scene") not in (None, "ok")),
            "missing": sorted(r["instance_id"] for r in mine if r.get("missing")),
        }
    out.mkdir(parents=True, exist_ok=True)
    (out / "rejudge.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("scenes")
    a.add_argument("--set", nargs="+", required=True)
    b = sub.add_parser("scene")
    b.add_argument("--set", nargs=1, required=True)
    b.add_argument("--task", required=True)
    b.add_argument("--seed", type=int, required=True)
    b.add_argument("--out", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--set", nargs="+", required=True)
    c.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "scenes":
        print("\n".join(scene_keys(parse_sets(args.set))))
        return 0
    if args.cmd == "scene":
        return scene_cmd(args)
    s = collect(parse_sets(args.set), Path(args.out))
    print(json.dumps({k: v["counts"].get("all") for k, v in s.items()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
