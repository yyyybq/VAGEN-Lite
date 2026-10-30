#!/usr/bin/env python3
"""N series smoke test, part 1 (the user's decision of 2026-10-30: left / right first, 20 instances; send the start
images, the success-pose images and the judge's numbers): instances of the upstream projective task ("Move so that
the A appears to the left / right of the B in your view.") in RoboCasa scenes, each proven solvable.

    python downstream/n/n_smoke.py generate --x0 X0_QUESTIONS.jsonl --out GEN --part I/K   # simulator Python
    python downstream/n/n_smoke.py page --gen GEN1 [GEN2 ...] --n 20 --out PAGE.html

generate, per scene (X0's scenes in order, every K-th from the I-th; reset as the closed-loop evaluation does):
  1. camera A placed and checked against the upstream camera pose (n_env.Scene.camera_matches);
  2. every base pose reachable from the start in the upstream actions (breadth first, at most MAX_DEPTH = 12 actions,
     the main line's validation budget of primitive actions per episode, max_episode_steps in its val_env.yaml, and
     MAX_STATES poses; poses equal to 5 cm and one heading step count once; a move whose pose collides is blocked);
  3. ordered pairs of entities with names unique in the scene, both relations, in a seeded random order: the
     reachable poses in breadth-first order are judged with the upstream box judge and 45-degree rule; the first
     that passes is rendered and its segmentation checked (each of A, B >= 0.3% of the image); the first pair whose
     shortest success needs >= MIN_ACTIONS actions and whose start does not already succeed is the scene's instance.
Writes GEN/instances.jsonl (task, seed, A, B, relation, description, start and success poses, the action path, the
judge's numbers at both poses, the search's size) and GEN/images/<scene>_{start,success}.png (camera A, 256 x 256).
page: the review page of the first N instances (the start image, the success image with A and B marked, the
instruction, the shortest path and the numbers behind each gate).
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import random
import sys
import time
from collections import deque
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1]))
import n_env as N  # noqa: E402

MAX_DEPTH = 12  # the main line's validation env: max_episode_steps 12 primitive actions per episode (val_env.yaml)
MAX_STATES = 15000
MIN_ACTIONS = 2
MAX_PAIRS = 40
MAX_RENDER_CHECKS = 12


def reachable(scene, max_depth: int = MAX_DEPTH, max_states: int = MAX_STATES) -> dict:
    """{key: (pose, depth, parent key, action)} of the poses reachable from the start, breadth first."""
    start = scene.start
    k0 = N.pose_key(start)
    seen = {k0: (start, 0, None, None)}
    blocked = set()
    queue = deque([k0])
    capped = False
    while queue:
        k = queue.popleft()
        pose, depth, _, _ = seen[k]
        if depth >= max_depth:
            continue
        for action in N.ACTIONS:
            new = N.apply_action(pose, action)
            nk = N.pose_key(new)
            if nk in seen or nk in blocked:
                continue
            scene.set_pose(new)
            if scene.collision() is not None:
                blocked.add(nk)
                continue
            seen[nk] = (new, depth + 1, k, action)
            queue.append(nk)
            if len(seen) >= max_states:
                capped = True
                queue.clear()
                break
    scene.set_pose(start)
    return {"states": seen, "blocked": len(blocked), "capped": capped}


def path_to(states: dict, key) -> list:
    out = []
    while states[key][2] is not None:
        out.append(states[key][3])
        key = states[key][2]
    return out[::-1]


def gate_numbers(j: dict) -> dict:
    m = j["metric"]
    return {"success": j["success"], "gates": m.get("gates"), "relation_margin_px": m.get("relation_margin_px"),
            "inside_frame_fraction_min": m.get("inside_frame_fraction_min"),
            "lateral_angle_deg": j["lateral"].get("angle_deg"), "lateral_ok": j["lateral"].get("ok"),
            "visible_fraction": j.get("visible")}


def find_instance(scene, states: dict, rng: random.Random, log=print):
    ents = scene.entities
    pairs = [(a, b, rel) for a in ents for b in ents if a["key"] != b["key"] for rel in ("left", "right")]
    rng.shuffle(pairs)
    order = sorted(states, key=lambda k: states[k][1])  # breadth first
    sid = f"{scene.task}_s{scene.seed}"
    for a, b, rel in pairs[:MAX_PAIRS]:
        item = N.projective_item(sid, a, b, rel, scene.start)
        start_box = N.box_judge(item, scene.start)
        if start_box["success"]:
            start_full = scene.judge(item, scene.start, a["key"], b["key"])
            if start_full["success"]:
                continue
        checks = 0
        for k in order:
            pose, depth = states[k][0], states[k][1]
            if depth < MIN_ACTIONS or not N.box_judge(item, pose)["success"]:
                continue
            full = scene.judge(item, pose, a["key"], b["key"])
            checks += 1
            if full["success"]:
                start_full = scene.judge(item, scene.start, a["key"], b["key"])
                return {"item": item, "a": a, "b": b, "relation": rel, "success_key": k, "success_pose": pose,
                        "depth": depth, "path": path_to(states, k), "start_judge": gate_numbers(start_full),
                        "success_judge": gate_numbers(full), "render_checks": checks}
            if checks >= MAX_RENDER_CHECKS:
                break
    return None


def strip(e: dict) -> dict:
    return {k: e[k] for k in ("key", "name", "kind", "bbox_min", "bbox_max", "center")}


def generate(args) -> int:
    import numpy as np
    from PIL import Image

    sys.path.insert(0, str(HERE.parents[1]))
    from downstream.eval import closed_loop

    closed_loop.stable_counter_regions()
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    scenes = []
    for line in Path(args.x0).read_text().splitlines():
        if line.strip():
            q = json.loads(line)
            if (q["task"], int(q["seed"])) not in scenes:
                scenes.append((q["task"], int(q["seed"])))
    if args.part:
        i, kk = (int(x) for x in args.part.split("/"))
        scenes = scenes[i - 1::kk]
    done, found = set(), 0
    if (out / "instances.jsonl").is_file():
        for line in (out / "instances.jsonl").read_text().splitlines():
            r = json.loads(line)
            done.add((r["task"], r["seed"]))
            found += bool(r.get("instance"))
    with open(out / "instances.jsonl", "a") as f:
        for task, seed in scenes:
            if found >= args.per_part or (task, seed) in done:
                continue
            t0 = time.time()
            rec = {"task": task, "seed": seed}
            scene = None
            try:
                scene = N.Scene(task, seed)
                rec["camera_check"] = scene.camera_matches(scene.start)
                rec["entities"] = len(scene.entities)
                reach = reachable(scene)
                rec.update(reachable=len(reach["states"]), blocked=reach["blocked"], capped=reach["capped"])
                inst = find_instance(scene, reach["states"], random.Random(f"{task}:{seed}"))
                if inst is None:
                    rec["instance"] = None
                else:
                    sid = f"{task}_s{seed}"
                    img0, _ = scene.render(scene.start)
                    img1, seg1 = scene.render(inst["success_pose"])
                    Image.fromarray(img0).save(out / "images" / f"{sid}_start.png")
                    Image.fromarray(img1).save(out / "images" / f"{sid}_success.png")
                    marks = {}
                    for tag, e in (("a", inst["a"]), ("b", inst["b"])):
                        ys, xs = np.nonzero(np.isin(seg1, list(e["geoms"])))
                        if len(xs):
                            marks[tag] = [float(xs.mean() * N.OUT_SIZE / N.RENDER_W), float(ys.mean() * N.OUT_SIZE / N.RENDER_H)]
                    rec["instance"] = {
                        "scene_id": sid, "relation": inst["relation"], "a": strip(inst["a"]), "b": strip(inst["b"]),
                        "description": inst["item"]["task_description"], "item": inst["item"],
                        "start_pose": list(scene.start), "success_pose": list(inst["success_pose"]),
                        "shortest_actions": inst["depth"], "path": inst["path"], "start_judge": inst["start_judge"],
                        "success_judge": inst["success_judge"], "render_checks": inst["render_checks"],
                        "success_marks": marks, "images": {"start": f"images/{sid}_start.png",
                                                           "success": f"images/{sid}_success.png"}}
                    found += 1
            except Exception as exc:  # one bad scene is reported, not fatal
                import traceback
                rec["error"] = f"{type(exc).__name__}: {exc}"[:400]
                rec["traceback"] = traceback.format_exc()[-1500:]
            finally:
                if scene is not None:
                    scene.close()
            rec["seconds"] = round(time.time() - t0, 1)
            f.write(json.dumps(rec, default=str) + "\n")
            f.flush()
            print(f"{task} {seed}: {'instance' if rec.get('instance') else rec.get('error', 'none')} "
                  f"({rec['seconds']} s, {rec.get('reachable')} poses)", flush=True)
    return 0


def b64(path: Path, marks=None) -> str:
    from PIL import Image, ImageDraw

    img = Image.open(path).convert("RGB").resize((512, 512))
    if marks:
        d = ImageDraw.Draw(img)
        for tag, color in (("a", (220, 40, 40)), ("b", (40, 90, 230))):
            if tag in marks:
                x, y = marks[tag][0] * 2, marks[tag][1] * 2
                d.ellipse([x - 10, y - 10, x + 10, y + 10], outline=color, width=4)
                d.text((x + 12, y - 8), tag.upper(), fill=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def fmt(v) -> str:
    return "—" if v is None else (f"{v:.3f}" if isinstance(v, float) else html.escape(str(v)))


def page(args) -> int:
    recs = []
    for g in args.gen:
        for line in (Path(g) / "instances.jsonl").read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                r["_dir"] = g
                recs.append(r)
    inst = [r for r in recs if r.get("instance")][:args.n]
    episodes = {}
    if getattr(args, "episodes", None) and Path(args.episodes).is_file():
        for line in Path(args.episodes).read_text().splitlines():
            if line.strip():
                e = json.loads(line)
                episodes[e.get("scene_id")] = e
    rows = []
    for i, r in enumerate(inst, 1):
        x = r["instance"]
        ep = episodes.get(x["scene_id"])
        ep_txt = ""
        if ep:
            ep_txt = (f'<p class="meta">模型的回合：{"成功" if ep.get("success") else "未成功"}，{ep.get("turns")} 轮、'
                      f'{ep.get("steps")} 个动作、被挡 {ep.get("blocked")} 次、读不出 {ep.get("invalid_turns")} 轮；'
                      f'第一轮回答：{html.escape(str((ep.get("turn_log") or [{}])[0].get("reply", ""))[:200])}'
                      f'{html.escape(ep.get("error", ""))}</p>')
        sj, uj = x["start_judge"], x["success_judge"]
        g = uj["gates"] or {}
        gate_txt = "；".join(f"{k} {'✓' if v else '✗'}" for k, v in g.items())
        vis = uj["visible_fraction"] or {}
        rows.append(
            f'<section><h2>{i}. {html.escape(r["task"])} · 种子 {r["seed"]}</h2>'
            f'<p class="desc">{html.escape(x["description"])}</p><div class="imgs">'
            f'<figure><img src="{b64(Path(r["_dir"]) / x["images"]["start"])}" alt="起点"><figcaption>起点（相机 A）</figcaption></figure>'
            f'<figure><img src="{b64(Path(r["_dir"]) / x["images"]["success"], x.get("success_marks"))}" alt="成功位姿">'
            f'<figcaption>找到的成功位姿（红 = A：{html.escape(x["a"]["name"])}，蓝 = B：{html.escape(x["b"]["name"])}；'
            f'分割图里可见像素的中心）</figcaption></figure></div>'
            f'<p class="meta">最短 {x["shortest_actions"]} 个动作：{html.escape(" ".join(x["path"]))}</p>'
            f'<p class="meta">成功位姿的判定：{gate_txt}；左右间隔 {fmt(uj["relation_margin_px"])} 像素（要 ≥ 12）；框在画面内的最小比例 '
            f'{fmt(uj["inside_frame_fraction_min"])}（要 ≥ 0.5）；B→A 与相机左右轴的夹角 {fmt(uj["lateral_angle_deg"])}°（要 ≤ 45）；'
            f'分割图可见比例 A {fmt(vis.get("a"))}、B {fmt(vis.get("b"))}（要 ≥ 0.003）</p>'
            f'<p class="meta">起点的判定：成功 {sj["success"]}；左右间隔 {fmt(sj["relation_margin_px"])}；夹角 {fmt(sj["lateral_angle_deg"])}°；'
            f'可见比例 {fmt((sj["visible_fraction"] or {}).get("a"))} / {fmt((sj["visible_fraction"] or {}).get("b"))}</p>'
            f'<p class="meta">搜索：可达位姿 {r.get("reachable")}（被挡 {r.get("blocked")}，到上限 {r.get("capped")}），'
            f'相机 A 自检误差 {fmt(r.get("camera_check"))}</p>{ep_txt}</section>')
    style = ("<style>:root{--bg:#fbfaf7;--fg:#1d1d1b;--muted:#6b6a65;--line:#dedbd2;--accent:#a3402a}"
             "@media (prefers-color-scheme: dark){:root:not([data-theme=light]){--bg:#171716;--fg:#ecebe6;--muted:#a19f97;"
             "--line:#3a3935;--accent:#e08a6f;color-scheme:dark}}:root[data-theme=dark]{--bg:#171716;--fg:#ecebe6;"
             "--muted:#a19f97;--line:#3a3935;--accent:#e08a6f;color-scheme:dark}body{background:var(--bg);color:var(--fg);"
             "font:15px/1.6 system-ui,'PingFang SC','Noto Sans CJK SC',sans-serif;max-width:1040px;margin:0 auto;"
             "padding-block:24px;padding-left:max(16px,env(safe-area-inset-left,0px));"
             "padding-right:max(16px,env(safe-area-inset-right,0px))}h1{font-size:22px}h2{font-size:16px;margin:0 0 6px}"
             "section{border-top:1px solid var(--line);padding:16px 0}.imgs{display:flex;flex-wrap:wrap;gap:12px}"
             "figure{margin:0;flex:1 1 300px;min-width:0}img{width:100%;max-width:512px;height:auto;border:1px solid var(--line)}"
             "figcaption,.meta{color:var(--muted);font-size:13px}.desc{border-left:3px solid var(--accent);padding-left:10px}</style>")
    n_scenes = len(recs)
    intro = (f"<h1>N 冒烟：左右关系 {len(inst)} 个实例</h1><p class=\"meta\">在 {n_scenes} 个 RoboCasa 场景里找实例（X0 的场景，"
             "评测种子），每个都用上游的动作（前后左右 0.3 米、左右转 20°）从起点做广度优先搜索，确认能到达成功位姿；"
             "判定是主线的投影判定和 45° 规则，可见性按你定的改成分割图（A、B 各占画面至少 0.3%）。相机 A：离地 1.5 米、水平、"
             "水平视场 90°。</p>")
    Path(args.out).write_text(f"<title>N 冒烟样例</title><meta name=\"claude:viewport-fit-sides\" content=\"cover\">"
                              f"{style}{intro}{''.join(rows)}", encoding="utf-8")
    print(json.dumps({"scenes": n_scenes, "instances": len(inst),
                      "errors": sum(1 for r in recs if r.get("error"))}))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--x0", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--part", default=None)
    g.add_argument("--per-part", type=int, default=3, help="stop after this many instances")
    p = sub.add_parser("page")
    p.add_argument("--gen", nargs="+", required=True)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--episodes", default=None, help="n_episode.py's episodes.jsonl: add each instance's episode")
    p.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    return {"generate": generate, "page": page}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
