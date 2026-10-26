#!/usr/bin/env python3
"""X0 (the user's decision of 2026-10-26): in-domain spatial questions from RoboCasa365 resets, asked zero-shot to
the four start VLMs.  This file makes the questions; docs/downstream/reports/X_DESIGNS_20261026.md has the design.

    python downstream/x/x0_qa.py generate --tasks OpenDrawer TurnOffStove --seeds 1000-1001 --out OUT
    python downstream/x/x0_qa.py samples --qa OUT/questions.jsonl --n 20 --out OUT/samples.html

Run `generate` with the RoboCasa simulator Python (downstream/eval/sim_env.sh: `sim THREADS $SIM_PY ...`).  Each
(task, seed) is reset exactly as the closed-loop evaluation does (gym.make('robocasa/<task>', split=target,
seed=s), env.reset(seed=s), stable counter regions); the first frame of the three cameras is rendered by the
simulator's own renderer (RGB, depth, geom segmentation; flipped upright, which equals the observation image) and
saved.  Entities are the scene's objects (named by their category) and compact fixtures (sink, stove, microwave,
...); walls, floors, counters, cabinets and drawers are left out (repeated names, no clear position).  An entity is
visible in an image when it has at least MIN_PIXELS segmentation pixels; only names unique among the visible
entities of that image are used.  Three kinds of two-choice question per image:
    left_right   is A left or right of B in this image            |x centroid difference| >= LR_PIXELS
    near_far     which of A, B is closer to this camera           depth medians differ >= NF_METERS and >= NF_RATIO
    front_back   with the robot's facing direction as forward,    forward coordinates differ >= FB_METERS
                 is A in front of B or behind it                  (forward: the left agent-view camera's heading)
The option order is drawn at random, so A and B are right about equally often.  Writes OUT/images/*.png,
OUT/candidates.jsonl (every question that passes the thresholds, with the numbers behind its answer) and
OUT/scenes.jsonl (per reset: instruction, visible entities per camera, whether the render equals the observation).
`samples` picks N candidates spread over kinds, cameras and scenes and writes a review page (the image the model
sees, the same image with the two entities marked, the prompt and the answer) plus samples.jsonl.
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import random
import re
import sys
import time
from pathlib import Path

WORKTREE = Path(__file__).resolve().parents[2]
CAMERAS = ("robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand")
CAMERA_ZH = {"robot0_agentview_left": "左侧固定相机", "robot0_agentview_right": "右侧固定相机",
             "robot0_eye_in_hand": "手腕相机"}
CAMERA_EN = {"robot0_agentview_left": "the robot's left camera", "robot0_agentview_right": "the robot's right camera",
             "robot0_eye_in_hand": "the robot's wrist camera"}
SIZE = 256
MIN_PIXELS = 150
LR_PIXELS = 30
NF_METERS, NF_RATIO = 0.2, 1.2
FB_METERS = 0.2
SKIP_FIXTURES = {"Wall", "Floor", "Counter", "Box", "HingeCabinet", "SingleCabinet", "Drawer", "OpenCabinet",
                 "HousingCabinet", "WallAccessory", "Window", "FramedWindow", "Wall Stack"}
KINDS = ("left_right", "near_far", "front_back")


def parse_seeds(text: str) -> list:
    out = []
    for part in text.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def display_name(text: str) -> str:
    """'wooden_spoon' -> 'wooden spoon'; fixture names lose their group suffix ('knife_block_main_group')."""
    text = re.sub(r"_(main|right|left|room)(_group)?$", "", str(text))
    text = re.sub(r"_(main|right|left)_group$", "", text)
    return re.sub(r"_\d+$", "", text).replace("_", " ").strip()


# ------------------------------------------------------------------------------------------------- geometry (pure)
def visible_stats(mask_xy, depth_values):
    """(pixels, x centroid, y centroid, depth median) of an entity's visible pixels."""
    xs, ys = mask_xy
    n = len(xs)
    if n == 0:
        return 0, None, None, None
    d = sorted(depth_values)
    return n, sum(xs) / n, sum(ys) / n, d[n // 2]


def answer_left_right(a: dict, b: dict):
    """'left' / 'right' (A relative to B in the image) or None when too close to call."""
    dx = a["cx"] - b["cx"]
    return None if abs(dx) < LR_PIXELS else ("left" if dx < 0 else "right")


def answer_near_far(a: dict, b: dict):
    """The closer of A, B to the camera ('A' / 'B') or None."""
    da, db = a["depth"], b["depth"]
    near, far = min(da, db), max(da, db)
    if far - near < NF_METERS or far < NF_RATIO * near:
        return None
    return "A" if da < db else "B"


def answer_front_back(a: dict, b: dict):
    """'front' / 'behind' (A relative to B along the robot's forward axis) or None."""
    d = a["forward"] - b["forward"]
    return None if abs(d) < FB_METERS else ("front" if d > 0 else "behind")


def question(kind: str, a: dict, b: dict, camera: str, rng: random.Random):
    """One two-choice question about A and B in one camera image, or None if the pair is too close to call."""
    an, bn = a["name"], b["name"]
    if kind == "left_right":
        truth = answer_left_right(a, b)
        text = f"In this image, is the {an} to the left or to the right of the {bn}?"
        zh = f"在这张图里，{an} 在 {bn} 的左边还是右边？"
        choices = ["left", "right"]
    elif kind == "near_far":
        truth = answer_near_far(a, b)
        truth = None if truth is None else (an if truth == "A" else bn)
        text = f"Which is closer to the camera that took this image: the {an} or the {bn}?"
        zh = f"{an} 和 {bn}，哪个离拍这张图的相机更近？"
        choices = [an, bn]
    else:
        truth = answer_front_back(a, b)
        text = (f"Taking the direction the robot is facing as forward, is the {an} in front of the {bn} "
                f"or behind it?")
        zh = f"以机器人面对的方向为前，{an} 在 {bn} 的前面还是后面？"
        choices = ["in front of it", "behind it"]
        truth = None if truth is None else ("in front of it" if truth == "front" else "behind it")
    if truth is None:
        return None
    if rng.random() < 0.5:
        choices = choices[::-1]
    letter = "AB"[choices.index(truth)]
    prompt = (f"This image is from {CAMERA_EN[camera]}. {text}\nA. {choices[0]}\nB. {choices[1]}\n"
              "Answer with the letter of the correct option only.")
    return {"kind": kind, "camera": camera, "question": text, "question_zh": zh, "options": choices,
            "answer": letter, "answer_text": truth, "prompt": prompt,
            "a": {k: a[k] for k in ("key", "name", "pixels", "cx", "cy", "depth", "forward")},
            "b": {k: b[k] for k in ("key", "name", "pixels", "cx", "cy", "depth", "forward")}}


# ---------------------------------------------------------------------------------------------- simulator side
def render_view(sim, camera: str):
    """Upright RGB (equal to the observation image), depth in metres and geom ids per pixel (-1: none)."""
    import numpy as np
    from robosuite.utils.camera_utils import get_real_depth_map

    rgb, depth = sim.render(camera_name=camera, width=SIZE, height=SIZE, depth=True)
    seg = sim.render(camera_name=camera, width=SIZE, height=SIZE, segmentation=True)
    seg = np.asarray(seg)
    if seg.ndim == 3:  # robosuite: (objtype, objid) per pixel
        geom = np.where(seg[..., 0] == 5, seg[..., 1], -1)  # mjOBJ_GEOM
    else:
        geom = seg
    depth = get_real_depth_map(sim, depth)
    return np.ascontiguousarray(rgb[::-1]), np.ascontiguousarray(depth[::-1]), np.ascontiguousarray(geom[::-1])


def scene_entities(k) -> list:
    """[{key, name, kind, geoms, pos}] of the scene's objects (by category) and compact fixtures."""
    import mujoco
    import numpy as np

    m = k.sim.model._model
    body_names = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i) or "" for i in range(m.nbody)]
    geom_body = [body_names[m.geom_bodyid[g]] for g in range(m.ngeom)]
    cats = {c.get("name"): str((c.get("info") or {}).get("cat") or c.get("name")) for c in getattr(k, "object_cfgs", [])}
    out = []
    for key, obj in k.objects.items():
        prefix = getattr(obj, "naming_prefix", f"{key}_")
        geoms = {g for g, b in enumerate(geom_body) if b.startswith(prefix)}
        bid = k.sim.model.body_name2id(obj.root_body)
        out.append({"key": key, "name": display_name(cats.get(key, key)), "kind": "object", "geoms": geoms,
                    "pos": np.asarray(k.sim.data.body_xpos[bid], float)})
    for key, fx in k.fixtures.items():
        if type(fx).__name__ in SKIP_FIXTURES:
            continue
        prefix = getattr(fx, "naming_prefix", f"{key}_")
        geoms = {g for g, b in enumerate(geom_body) if b.startswith(prefix)}
        name = getattr(fx, "nat_lang", None)
        name = display_name(key) if not name or name == key else name
        out.append({"key": key, "name": display_name(name), "kind": "fixture", "geoms": geoms,
                    "pos": np.asarray(getattr(fx, "pos", [0, 0, 0]), float)})
    return [e for e in out if e["geoms"]]


def forward_axis(k):
    """The robot's facing direction on the floor plane: the left agent-view camera's heading (unit xy vector) and
    the camera position (the origin of the forward coordinate)."""
    import numpy as np

    sim = k.sim
    cid = sim.model.camera_name2id("robot0_agentview_left")
    look = -np.asarray(sim.data.cam_xmat[cid]).reshape(3, 3)[:, 2]
    f = np.array([look[0], look[1], 0.0])
    return f / np.linalg.norm(f), np.asarray(sim.data.cam_xpos[cid], float)


def scene_questions(task: str, seed: int, out: Path, rng: random.Random) -> tuple:
    """Reset (task, seed) as the evaluation does, render the three cameras, return (scene record, questions)."""
    import gymnasium as gym
    import numpy as np
    import robocasa.wrappers.gym_wrapper  # noqa: F401
    from PIL import Image

    env = gym.make(f"robocasa/{task}", enable_render=True, split="target", seed=seed)
    try:
        obs, _ = env.reset(seed=seed)
        k = env.unwrapped.env
        ents = scene_entities(k)
        fwd, origin = forward_axis(k)
        scene = {"task": task, "seed": seed, "lang": k.get_ep_meta().get("lang"), "views": {}}
        qs = []
        for cam in CAMERAS:
            rgb, depth, geom = render_view(k.sim, cam)
            same = bool(np.array_equal(rgb, np.asarray(obs[f"video.{cam}"])))
            path = out / "images" / f"{task}_s{seed}_{cam}.png"
            Image.fromarray(rgb).save(path)
            vis = []
            for e in ents:
                mask = np.isin(geom, list(e["geoms"]))
                ys, xs = np.nonzero(mask)
                n, cx, cy, dmed = visible_stats((xs.tolist(), ys.tolist()), depth[mask].tolist())
                if n >= MIN_PIXELS:
                    vis.append(dict(e, pixels=int(n), cx=round(cx, 1), cy=round(cy, 1), depth=round(float(dmed), 3),
                                    forward=round(float(np.dot(e["pos"] - origin, fwd)), 3)))
            names = [v["name"] for v in vis]
            vis = [v for v in vis if names.count(v["name"]) == 1]
            scene["views"][cam] = {"image": str(path.relative_to(out)), "matches_observation": same,
                                   "visible": [{k2: v[k2] for k2 in ("key", "name", "kind", "pixels", "cx", "cy", "depth",
                                                                     "forward")} for v in vis]}
            for i, a in enumerate(vis):
                for b in vis[i + 1:]:
                    first, second = (a, b) if rng.random() < 0.5 else (b, a)
                    for kind in KINDS:
                        q = question(kind, first, second, cam, rng)
                        if q:
                            qs.append(dict(q, task=task, seed=seed, image=str(path.relative_to(out))))
        return scene, qs
    finally:
        env.close()


def generate(args) -> int:
    sys.path.insert(0, str(WORKTREE))
    from downstream.eval import closed_loop

    closed_loop.stable_counter_regions()
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.rng_seed)
    with open(out / "candidates.jsonl", "a") as qf, open(out / "scenes.jsonl", "a") as sf:
        for task in args.tasks:
            for seed in parse_seeds(args.seeds):
                t = time.time()
                try:
                    scene, qs = scene_questions(task, seed, out, rng)
                except Exception as exc:  # one bad reset is reported, not fatal
                    sf.write(json.dumps({"task": task, "seed": seed, "error": f"{type(exc).__name__}: {exc}"[:300]}) + "\n")
                    continue
                scene["seconds"] = round(time.time() - t, 1)
                scene["candidates"] = len(qs)
                sf.write(json.dumps(scene) + "\n")
                for q in qs:
                    qf.write(json.dumps(q) + "\n")
                print(f"{task} {seed}: {len(qs)} candidate questions, {scene['seconds']} s", flush=True)
    return 0


# ------------------------------------------------------------------------------------------- picking and review
def pick(cands: list, n: int, rng: random.Random) -> list:
    """n questions spread over kinds, cameras and scenes (round robin), each kind's answers balanced A / B."""
    by = {}
    for q in cands:
        by.setdefault((q["kind"], q["camera"]), []).append(q)
    for v in by.values():
        rng.shuffle(v)
    out, used = [], set()
    keys = sorted(by)
    while len(out) < n and any(by.values()):
        for key in keys:
            if len(out) >= n or not by[key]:
                continue
            want = "AB"[sum(1 for q in out if q["kind"] == key[0]) % 2]
            pool = by[key]
            choice = next((q for q in pool if q["answer"] == want and (q["task"], q["seed"]) not in used), None) \
                or next((q for q in pool if (q["task"], q["seed"]) not in used), None) or pool[0]
            pool.remove(choice)
            out.append(choice)
            used.add((choice["task"], choice["seed"]))
            if len(used) >= len({(q["task"], q["seed"]) for q in cands}):
                used.clear()
    return out


def annotate(path: Path, q: dict) -> str:
    """The image with the two entities' visible-pixel centres marked (A red, B blue) as a data URI."""
    from PIL import Image, ImageDraw

    img = Image.open(path).convert("RGB").resize((SIZE * 2, SIZE * 2))
    d = ImageDraw.Draw(img)
    for ent, color in ((q["a"], (220, 40, 40)), (q["b"], (40, 90, 230))):
        x, y = ent["cx"] * 2, ent["cy"] * 2
        d.ellipse([x - 9, y - 9, x + 9, y + 9], outline=color, width=4)
        d.text((x + 11, y - 8), ent["name"], fill=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def clean(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


KIND_ZH = {"left_right": "左右", "near_far": "远近", "front_back": "前后"}


def samples(args) -> int:
    qa = Path(args.qa)
    root = qa.parent
    cands = [json.loads(line) for line in qa.read_text().splitlines() if line.strip()]
    picked = pick(cands, args.n, random.Random(args.rng_seed))
    (root / "samples.jsonl").write_text("".join(json.dumps(q) + "\n" for q in picked))
    rows = []
    for i, q in enumerate(picked, 1):
        img = root / q["image"]
        nums = {"left_right": f"横向像素中心：{q['a']['name']} {q['a']['cx']:.0f}，{q['b']['name']} {q['b']['cx']:.0f}（图宽 256）",
                "near_far": f"到相机的深度中位数：{q['a']['name']} {q['a']['depth']:.2f} 米，{q['b']['name']} {q['b']['depth']:.2f} 米",
                "front_back": f"沿机器人前向的坐标：{q['a']['name']} {q['a']['forward']:.2f} 米，{q['b']['name']} {q['b']['forward']:.2f} 米"}[q["kind"]]
        rows.append(
            f'<section><h2>{i}. {KIND_ZH[q["kind"]]} · {html.escape(q["task"])} 种子 {q["seed"]} · {CAMERA_ZH[q["camera"]]}</h2>'
            f'<div class="imgs"><figure><img src="{clean(img)}" alt="模型看到的图"><figcaption>模型看到的图（256×256）</figcaption></figure>'
            f'<figure><img src="{annotate(img, q)}" alt="标注"><figcaption>标注：红 = {html.escape(q["a"]["name"])}，蓝 = '
            f'{html.escape(q["b"]["name"])}（可见像素的中心）</figcaption></figure></div>'
            f'<pre>{html.escape(q["prompt"])}</pre><p>中文：{html.escape(q["question_zh"])}</p>'
            f'<p class="ans">标准答案：<b>{q["answer"]}</b>（{html.escape(q["answer_text"])}）</p><p class="meta">{html.escape(nums)}；'
            f'可见像素 {q["a"]["pixels"]} / {q["b"]["pixels"]}</p></section>')
    style = ("<style>:root{--bg:#fbfaf7;--fg:#1d1d1b;--muted:#6b6a65;--line:#dedbd2;--accent:#a3402a}"
             "@media (prefers-color-scheme: dark){:root:not([data-theme=light]){--bg:#171716;--fg:#ecebe6;--muted:#a19f97;"
             "--line:#3a3935;--accent:#e08a6f;color-scheme:dark}}"
             ":root[data-theme=dark]{--bg:#171716;--fg:#ecebe6;--muted:#a19f97;--line:#3a3935;--accent:#e08a6f;color-scheme:dark}"
             "body{background:var(--bg);color:var(--fg);font:15px/1.6 system-ui,'PingFang SC','Noto Sans CJK SC',sans-serif;"
             "max-width:980px;margin:0 auto;padding-block:24px;padding-left:max(16px,env(safe-area-inset-left,0px));"
             "padding-right:max(16px,env(safe-area-inset-right,0px))}h1{font-size:22px}h2{font-size:16px;margin:0 0 8px}"
             "section{border-top:1px solid var(--line);padding:18px 0}.imgs{display:flex;flex-wrap:wrap;gap:12px}"
             "figure{margin:0;flex:1 1 300px;min-width:0}img{width:100%;max-width:512px;height:auto;image-rendering:pixelated;"
             "border:1px solid var(--line)}figcaption,.meta{color:var(--muted);font-size:13px}"
             "pre{white-space:pre-wrap;background:transparent;border-left:3px solid var(--accent);padding:4px 10px;margin:10px 0}"
             ".ans b{color:var(--accent)}</style>")
    intro = (f"<h1>X0 问答样例（{len(picked)} 道）</h1><p class=\"meta\">从 {len({(q['task'], q['seed']) for q in cands})} 个重置的"
             f"第一帧里自动出了 {len(cands)} 道候选题，按题型、相机、场景轮流抽出这些。左图是模型看到的图，右图标出两个东西"
             "可见像素的中心，只给你核对用。判定门槛：左右至少差 30 像素；远近至少差 0.2 米且远者是近者的 1.2 倍以上；"
             "前后（沿机器人前向）至少差 0.2 米。</p>")
    Path(args.out).write_text(f"<title>X0 问答样例</title><meta name=\"claude:viewport-fit-sides\" content=\"cover\">{style}"
                              f"{intro}{''.join(rows)}", encoding="utf-8")
    print(json.dumps({"candidates": len(cands), "picked": len(picked),
                      "by_kind": {k: sum(1 for q in picked if q["kind"] == k) for k in KINDS}}))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--tasks", nargs="+", required=True)
    g.add_argument("--seeds", required=True)
    g.add_argument("--out", required=True)
    g.add_argument("--rng-seed", type=int, default=0)
    s = sub.add_parser("samples")
    s.add_argument("--qa", required=True)
    s.add_argument("--n", type=int, default=20)
    s.add_argument("--out", required=True)
    s.add_argument("--rng-seed", type=int, default=0)
    args = ap.parse_args(argv)
    return generate(args) if args.cmd == "generate" else samples(args)


if __name__ == "__main__":
    sys.exit(main())
