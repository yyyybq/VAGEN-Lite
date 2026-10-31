#!/usr/bin/env python3
"""N1 instances (the user's decision 3 of 2026-10-31): the design's six task types in RoboCasa scenes, 50 of each,
each proven solvable from its start; camera A without the robot, plain object names, no margin added to any threshold
(near-threshold instances are marked for a separate count).

    python downstream/n/n_gen.py scenes --x0 X0_QUESTIONS.jsonl [--n 120]           # the scene order, one per line
    python downstream/n/n_gen.py scene --task T --seed S --out GEN [--soft-limit 1000] # simulator; one scene
    python downstream/n/n_gen.py collect --out GEN [--per-family 50]
    python downstream/n/n_gen.py page --out GEN [--per-family 4] --page PAGE.html

scenes: X0's scenes (the closed-loop evaluation's resets, seeds from 1000) with the tasks taken in turn, so the
  first n spread over the tasks.
scene (one process per scene; run_n_gen.sh kills it after the scene's time limit and records the scene as skipped):
  1. the scene reset as the closed-loop evaluation does, camera A placed and checked (n_env.Scene, robot hidden);
  2. the base poses reachable from the start (n_smoke.reachable: breadth first, at most 12 actions, 15,000 poses);
  3. for each task type in turn (n_env.FAMILIES), candidates in a seeded order (left / right: ordered pairs and both
     relations; equal distance and both-in-view: pairs; occlusion: ordered (hidden target, occluder) pairs; absolute
     distance: an object and a distance of the upstream menu; screen occupancy: an object and a share of the menu);
     the upstream tuple rules refuse some (heights, gaps: build_item raises), and a candidate whose start already
     succeeds is skipped.  The reachable poses are scanned in breadth-first order with the box judge; at most
     MAX_RENDER_CHECKS of the passing ones are rendered for the segmentation rule; the first full success at
     >= MIN_ACTIONS actions is the shortest success (as in the smoke).  Then all reachable poses that pass the box
     judge are ranked by their tightest box margin (n_env.margins) and the best BEST_CHECKS rendered: the best full
     success is the instance's most comfortable success pose, and the gates even it passes only narrowly are the
     instance's near_threshold (empty = it can be done with room to spare).  One instance per task type and scene;
     past --soft-limit seconds no new candidate is started.
  Writes GEN/scenes/<task>_s<seed>.json after every task type ("complete" once all six are done; a killed process
  leaves the types it finished), and GEN/images/<task>_s<seed>_<type>_{start,shortest,best}.png (camera A).  The
  process then leaves with os._exit without closing the environment (RoboCasa's env.close() can hang for good: 6 of
  the trial's 12 scenes did).
collect: --per-family instances per type from the scene files, the tasks taken in turn in a seeded order (each type
  spreads over the tasks); writes GEN/instances_eval.jsonl (one instance per line: the episode runner's input) and
  GEN/summary.json (per type: instances found / taken, tasks, near-threshold count, parameters, shortest actions;
  scenes skipped for time, failed, without instances).
page: the review page (start, shortest and most comfortable success images with the objects marked, instruction,
  path, the numbers behind every gate).
"""
from __future__ import annotations

import argparse
import base64
import collections
import html
import io
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
import n_smoke as S  # noqa: E402

MIN_ACTIONS = 2
MAX_RENDER_CHECKS = 12
BEST_CHECKS = 3
MAX_CANDIDATES = {"projective_relations": 40}  # tuple-rule-passing candidates tried per type and scene
DEFAULT_CANDIDATES = 24
SEED = 20261031
TYPE_ZH = {"projective_relations": "左右关系", "equidistance": "等距", "absolute_positioning": "绝对距离",
           "occlusion_alignment": "遮挡", "fov_inclusion": "同框", "screen_occupancy": "画面占比"}


# --------------------------------------------------------------------------------------------- scene order
def scene_order(x0_path: Path, n: int = 0) -> list:
    """X0's (task, seed) scenes, the tasks in turn (each task's seeds ascending), the first n."""
    by_task = collections.defaultdict(set)
    for line in Path(x0_path).read_text().splitlines():
        if line.strip():
            q = json.loads(line)
            by_task[q["task"]].add(int(q["seed"]))
    lists = {t: sorted(s) for t, s in by_task.items()}
    out, i = [], 0
    while any(i < len(v) for v in lists.values()):
        out += [(t, lists[t][i]) for t in sorted(lists) if i < len(lists[t])]
        i += 1
    return out[:n] if n else out


# ------------------------------------------------------------------------------------------- candidates
def candidates(task_type: str, ents: list, rng: random.Random) -> list:
    """[(entities in role order, params)] in a seeded order (see the module text)."""
    if task_type == "projective_relations":
        c = [((a, b), {"relation": r}) for a in ents for b in ents if a["key"] != b["key"] for r in ("left", "right")]
    elif task_type in ("equidistance", "fov_inclusion"):
        c = [((a, b) if rng.random() < 0.5 else (b, a), {}) for i, a in enumerate(ents) for b in ents[i + 1:]]
    elif task_type == "occlusion_alignment":
        c = [((a, b), {}) for a in ents for b in ents if a["key"] != b["key"]]
    elif task_type == "absolute_positioning":
        c = [((e,), {"distance_m": d}) for e in ents for d in N.family_module(task_type).DISTANCES_M]
    elif task_type == "screen_occupancy":
        c = [((e,), {"occupancy_ratio": k}) for e in ents for k in N.family_module(task_type).OCCUPANCY_RATIOS]
    else:
        raise ValueError(task_type)
    rng.shuffle(c)
    return c


def score(margins: dict) -> float:
    """The tightest gate's slack (1 = every gate passed comfortably)."""
    return min(margins.values()) if margins else 1.0


def numbers(item: dict, j: dict) -> dict:
    """What a judge result is reported with: success, gates, the numbers behind them, slacks, visibility."""
    meas = {k: (list(v) if isinstance(v, tuple) else v) for k, v in N.measurements(item, j).items()}
    m = j["metric"]
    return {"success": bool(j["success"]), "gates": m.get("gates"), "measurements": meas,
            "margins": N.margins(item, j), "near": N.near_threshold(item, j), "visible": j.get("visible"),
            "shaping_score": m.get("shaping_score", m.get("score"))}


def find(scene, states: dict, order: list, task_type: str, rng: random.Random, deadline: float, log=print):
    """The scene's instance of one task type, or None; and the tally of candidates by outcome."""
    sid = f"{scene.task}_s{scene.seed}"
    tally = collections.Counter()
    tried = 0
    limit = MAX_CANDIDATES.get(task_type, DEFAULT_CANDIDATES)
    for ents, params in candidates(task_type, scene.entities, rng):
        if tried >= limit or time.time() > deadline:
            tally["stopped_" + ("limit" if tried >= limit else "time")] += 1
            break
        try:
            item = N.make_item(task_type, sid, list(ents), params, scene.start)
        except ValueError as exc:  # the upstream tuple rules
            tally["refused:" + str(exc).split(": ", 1)[-1][:40]] += 1
            continue
        tried += 1
        start_box = N.box_judge(item, scene.start)
        start_full = scene.judge(item, scene.start) if start_box["success"] else None
        if start_full is not None and start_full["success"]:
            tally["start_succeeds"] += 1
            continue
        box_ok, shortest, checks = [], None, 0
        for k in order:
            pose, depth = states[k][0], states[k][1]
            if depth < MIN_ACTIONS:
                continue
            bj = N.box_judge(item, pose)
            if not bj["success"]:
                continue
            box_ok.append((k, bj))
            if shortest is None:
                if checks >= MAX_RENDER_CHECKS:
                    break
                full = scene.judge(item, pose)
                checks += 1
                if full["success"]:
                    shortest = (k, full)
        if shortest is None:
            tally["render_checks_failed" if box_ok else "no_box_success"] += 1
            continue
        ranked = sorted(box_ok, key=lambda kb: (-score(N.margins(item, kb[1])), states[kb[0]][1]))
        best = shortest
        best_score = score(N.margins(item, shortest[1]))
        for k, _ in ranked[:BEST_CHECKS]:
            if k == shortest[0]:
                continue
            full = scene.judge(item, states[k][0])
            checks += 1
            if full["success"] and score(N.margins(item, full)) > best_score:
                best, best_score = (k, full), score(N.margins(item, full))
        if start_full is None:
            start_full = scene.judge(item, scene.start)
        tally["instance"] += 1

        def pose_rec(kj):
            k, j = kj
            return {"pose": list(states[k][0]), "actions": states[k][1], "path": S.path_to(states, k),
                    "judge": numbers(item, j)}

        inst = {"instance_id": f"{sid}_{task_type}", "scene_id": sid, "task": scene.task, "seed": scene.seed,
                "task_type": task_type, "hide_robot": scene.hide_robot, "objects": [S.strip(e) for e in ents],
                "params": params, "description": item["task_description"], "item": item,
                "start_pose": list(scene.start), "start_judge": numbers(item, start_full),
                "shortest": pose_rec(shortest), "best": pose_rec(best), "near_threshold": numbers(item, best[1])["near"],
                "box_success_poses": len(box_ok), "render_checks": checks, "candidates_tried": tried}
        return inst, tally
    return None, tally


def save_images(scene, inst: dict, out: Path) -> None:
    """Start, shortest and best success images; the objects' visible-pixel centres at the two success poses."""
    import numpy as np
    from PIL import Image

    keys = [o["key"] for o in inst["objects"]]
    geoms = {e["key"]: e["geoms"] for e in scene.entities if e["key"] in keys}
    inst["images"], inst["marks"] = {}, {}
    for tag, pose in (("start", inst["start_pose"]), ("shortest", inst["shortest"]["pose"]),
                      ("best", inst["best"]["pose"])):
        img, seg = scene.render(tuple(pose))
        name = f"{inst['instance_id']}_{tag}.png"
        Image.fromarray(img).save(out / "images" / name)
        inst["images"][tag] = f"images/{name}"
        marks = {}
        for i, k in enumerate(keys):
            ys, xs = np.nonzero(np.isin(seg, list(geoms.get(k, ()))))
            if len(xs):
                marks["abc"[i]] = [float(xs.mean() * N.OUT_SIZE / N.RENDER_W), float(ys.mean() * N.OUT_SIZE / N.RENDER_H)]
        inst["marks"][tag] = marks


def write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, default=str) + "\n")
    os.replace(tmp, path)


def scene_cmd(args) -> int:
    from downstream.eval import closed_loop

    closed_loop.stable_counter_regions()
    out = Path(args.out)
    (out / "scenes").mkdir(parents=True, exist_ok=True)
    (out / "images").mkdir(parents=True, exist_ok=True)
    sid = f"{args.task}_s{args.seed}"
    path = out / "scenes" / f"{sid}.json"
    t0 = time.time()
    deadline = t0 + args.soft_limit
    rec = {"task": args.task, "seed": args.seed, "scene_id": sid, "hide_robot": True, "complete": False,
           "instances": {}, "tallies": {}, "seconds_by_type": {}}
    scene = None
    try:
        scene = N.Scene(args.task, args.seed, hide_robot=True)
        rec["camera_check"] = scene.camera_matches(scene.start)
        rec["entities"] = [e["name"] for e in scene.entities]
        reach = S.reachable(scene)
        states = reach["states"]
        rec.update(reachable=len(states), blocked=reach["blocked"], capped=reach["capped"],
                   seconds_setup=round(time.time() - t0, 1))
        order = sorted(states, key=lambda k: states[k][1])
        write_json(path, rec)
        for t in args.families:
            t1 = time.time()
            inst, tally = find(scene, states, order, t, random.Random(f"{SEED}:{sid}:{t}"), deadline)
            if inst is not None:
                save_images(scene, inst, out)
            rec["instances"][t] = inst
            rec["tallies"][t] = dict(tally)
            rec["seconds_by_type"][t] = round(time.time() - t1, 1)
            rec["seconds"] = round(time.time() - t0, 1)
            write_json(path, rec)
            print(f"{sid} {t}: {'instance' if inst else 'none'} {dict(tally)} ({rec['seconds_by_type'][t]} s)",
                  flush=True)
        rec["complete"] = True
    except Exception as exc:  # reported in the scene file, not fatal for the job
        import traceback

        rec["error"] = f"{type(exc).__name__}: {exc}"[:400]
        rec["traceback"] = traceback.format_exc()[-1500:]
    rec["seconds"] = round(time.time() - t0, 1)
    write_json(path, rec)
    # 2026-10-31: RoboCasa's env.close() (and the finalizers that free the same OSMesa context) can hang for good
    # after a finished scene, as in the closed-loop evaluation; the trial's processes did it in 6 of 12 scenes.  The
    # record is written, so the process leaves without closing or running finalizers; the OS frees everything.
    sys.stdout.flush()
    sys.stderr.flush()
    if scene is not None:
        os._exit(0)
    return 0


# ---------------------------------------------------------------------------------------------- collect
def load_scenes(out: Path) -> tuple:
    scenes, skipped = [], []
    for p in sorted((Path(out) / "scenes").glob("*.json")):
        if p.name.endswith(".timeout.json"):
            skipped.append(json.loads(p.read_text()))
        elif not p.name.endswith(".tmp"):
            scenes.append(json.loads(p.read_text()))
    return scenes, skipped


def collect(out: Path, per_family: int = 50, seed: int = SEED) -> dict:
    out = Path(out)
    scenes, skipped = load_scenes(out)
    rng = random.Random(seed)
    tasks = sorted({s["task"] for s in scenes})
    rng.shuffle(tasks)
    chosen, summary = [], {"per_family": per_family, "scenes": len(scenes), "families": {}}
    for t in N.FAMILIES:
        pool = collections.defaultdict(list)
        for s in sorted(scenes, key=lambda s: s["seed"]):
            inst = (s.get("instances") or {}).get(t)
            if inst:
                pool[s["task"]].append(inst)
        found = sum(len(v) for v in pool.values())
        taken, i = [], 0
        while len(taken) < per_family and any(i < len(pool[k]) for k in tasks):
            for k in tasks:
                if i < len(pool[k]) and len(taken) < per_family:
                    taken.append(pool[k][i])
            i += 1
        chosen += taken
        params = collections.Counter(json.dumps(x["params"], sort_keys=True) for x in taken)
        summary["families"][t] = {
            "found": found, "taken": len(taken), "tasks": len({x["task"] for x in taken}),
            "near_threshold": sum(1 for x in taken if x["near_threshold"]),
            "near_gates": dict(collections.Counter(g for x in taken for g in x["near_threshold"])),
            "params": dict(params), "shortest_actions": sorted(x["shortest"]["actions"] for x in taken),
            "tallies": dict(sum((collections.Counter(s.get("tallies", {}).get(t, {})) for s in scenes),
                                collections.Counter()))}
    summary.update(
        instances=len(chosen), complete_scenes=sum(1 for s in scenes if s.get("complete")),
        scenes_with_error=[{"scene_id": s["scene_id"], "error": s.get("error")} for s in scenes if s.get("error")],
        scenes_skipped_for_time=[{"scene_id": f"{k['task']}_s{k['seed']}", "rc": k.get("rc"),
                                  "seconds": k.get("seconds")} for k in skipped],
        scenes_without_instance=[s["scene_id"] for s in scenes if not any((s.get("instances") or {}).values())],
        setup_seconds=sorted(s.get("seconds_setup") for s in scenes if s.get("seconds_setup") is not None),
        scene_seconds=sorted(s.get("seconds") for s in scenes if s.get("seconds") is not None))
    with open(out / "instances_eval.jsonl", "w") as f:
        for x in chosen:
            f.write(json.dumps(x, default=str) + "\n")
    write_json(out / "summary.json", summary)
    return summary


# ------------------------------------------------------------------------------------------------- page
def b64(path: Path, marks=None) -> str:
    from PIL import Image, ImageDraw

    img = Image.open(path).convert("RGB").resize((512, 512))
    if marks:
        d = ImageDraw.Draw(img)
        for tag, color in (("a", (220, 40, 40)), ("b", (40, 90, 230)), ("c", (30, 150, 60))):
            if tag in marks:
                x, y = marks[tag][0] * 2, marks[tag][1] * 2
                d.ellipse([x - 10, y - 10, x + 10, y + 10], outline=color, width=4)
                d.text((x + 12, y - 8), tag.upper(), fill=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def fmt_meas(meas: dict) -> str:
    parts = []
    for k, v in (meas or {}).items():
        if isinstance(v, list):
            parts.append(f"{k} 误差 {S.fmt(v[0])} / 容差 {S.fmt(v[1])}")
        else:
            parts.append(f"{k} {S.fmt(v)}")
    return "；".join(parts)


def page(out: Path, dest: Path, per_family: int = 4) -> int:
    out = Path(out)
    insts = [json.loads(x) for x in (out / "instances_eval.jsonl").read_text().splitlines() if x.strip()]
    summary = json.loads((out / "summary.json").read_text())
    rows = []
    for t in N.FAMILIES:
        fam = [x for x in insts if x["task_type"] == t][:per_family]
        st = summary["families"][t]
        rows.append(f"<h2>{TYPE_ZH[t]}（{t}）：取 {st['taken']} 个，门槛附近 {st['near_threshold']} 个</h2>")
        for x in fam:
            names = "，".join(f"{'ABC'[i]} = {html.escape(o['name'])}" for i, o in enumerate(x["objects"]))
            figs = "".join(
                f'<figure><img src="{b64(out / x["images"][tag], x["marks"].get(tag))}" alt="{cap}">'
                f"<figcaption>{cap}</figcaption></figure>"
                for tag, cap in (("start", "起点"), ("shortest", f"最近的成功位姿（{x['shortest']['actions']} 个动作）"),
                                 ("best", f"余量最大的成功位姿（{x['best']['actions']} 个动作）")))
            near = "、".join(x["near_threshold"]) or "无"
            rows.append(
                f'<section><h3>{html.escape(x["instance_id"])}</h3><p class="desc">{html.escape(x["description"])}</p>'
                f'<p class="meta">{names}（红 A、蓝 B：分割图里可见像素的中心）</p><div class="imgs">{figs}</div>'
                f'<p class="meta">最近的成功：{html.escape(" ".join(x["shortest"]["path"]))}；'
                f'{fmt_meas(x["shortest"]["judge"]["measurements"])}</p>'
                f'<p class="meta">余量最大的成功：{fmt_meas(x["best"]["judge"]["measurements"])}；仍在门槛附近的：{near}</p>'
                f'<p class="meta">起点：{fmt_meas(x["start_judge"]["measurements"])}</p></section>')
    style = ("<style>:root{--bg:#fbfaf7;--fg:#1d1d1b;--muted:#6b6a65;--line:#dedbd2;--accent:#a3402a}"
             "@media (prefers-color-scheme: dark){:root:not([data-theme=light]){--bg:#171716;--fg:#ecebe6;--muted:#a19f97;"
             "--line:#3a3935;--accent:#e08a6f;color-scheme:dark}}:root[data-theme=dark]{--bg:#171716;--fg:#ecebe6;"
             "--muted:#a19f97;--line:#3a3935;--accent:#e08a6f;color-scheme:dark}body{background:var(--bg);color:var(--fg);"
             "font:15px/1.6 system-ui,'PingFang SC','Noto Sans CJK SC',sans-serif;max-width:1180px;margin:0 auto;"
             "padding-block:24px;padding-left:max(16px,env(safe-area-inset-left,0px));"
             "padding-right:max(16px,env(safe-area-inset-right,0px))}h1{font-size:22px}h2{font-size:18px;margin-top:28px}"
             "h3{font-size:15px;margin:0 0 6px}section{border-top:1px solid var(--line);padding:14px 0}"
             ".imgs{display:flex;flex-wrap:wrap;gap:10px}figure{margin:0;flex:1 1 260px;min-width:0}"
             "img{width:100%;max-width:400px;height:auto;border:1px solid var(--line)}"
             "figcaption,.meta{color:var(--muted);font-size:13px}.desc{border-left:3px solid var(--accent);padding-left:10px}</style>")
    intro = (f"<h1>N1 实例样例（每类前 {per_family} 个）</h1><p class=\"meta\">{summary['scenes']} 个场景，"
             f"共取 {summary['instances']} 个实例。相机 A 离地 1.5 米、水平、水平视场 90°，不画机器人；判定照主线，"
             "可见性用分割图（要求可见的目标各占画面至少 0.3%）。门槛附近：连余量最大的成功位姿也只是勉强过某道门槛。</p>")
    Path(dest).write_text(f"<title>N1 实例样例</title><meta name=\"claude:viewport-fit-sides\" content=\"cover\">"
                          f"{style}{intro}{''.join(rows)}", encoding="utf-8")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("scenes")
    a.add_argument("--x0", required=True)
    a.add_argument("--n", type=int, default=0)
    s = sub.add_parser("scene")
    s.add_argument("--task", required=True)
    s.add_argument("--seed", type=int, required=True)
    s.add_argument("--out", required=True)
    s.add_argument("--soft-limit", type=float, default=1000.0)
    s.add_argument("--families", nargs="+", default=list(N.FAMILIES))
    c = sub.add_parser("collect")
    c.add_argument("--out", required=True)
    c.add_argument("--per-family", type=int, default=50)
    p = sub.add_parser("page")
    p.add_argument("--out", required=True)
    p.add_argument("--page", required=True)
    p.add_argument("--per-family", type=int, default=4)
    args = ap.parse_args(argv)
    if args.cmd == "scenes":
        for t, sd in scene_order(Path(args.x0), args.n):
            print(f"{t}:{sd}")
        return 0
    if args.cmd == "scene":
        return scene_cmd(args)
    if args.cmd == "collect":
        print(json.dumps(collect(Path(args.out), args.per_family), default=str))
        return 0
    return page(Path(args.out), Path(args.page), args.per_family)


if __name__ == "__main__":
    sys.exit(main())
