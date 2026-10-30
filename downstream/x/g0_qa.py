#!/usr/bin/env python3
"""G0 (the user's decision of 2026-10-30; design reports/Q0_G0_DESIGN_20261028.md): general, non-spatial questions
on X0's pictures, to measure how much general ability the upstream training (S, RS, RD; Qwen3 SFT, RS, RD) lost.
No colour or material questions (RoboCasa's object registry has neither; the user's decision).

    python downstream/x/g0_qa.py generate --x0 X0_QUESTIONS.jsonl --out GEN --part I/K   # simulator Python
    python downstream/x/g0_qa.py select   --gen GEN [GEN2 ...] --out G0.jsonl [--total 300]
    python downstream/x/g0_qa.py samples  --questions G0.jsonl --n 20 --out PAGE.html
    python downstream/x/g0_qa.py analyze  --questions G0.jsonl --answers M=A.jsonl ... --out R.json

generate: the 253 scenes and 384 images X0's 510 questions use, reset exactly as X0 did (x0_qa: gym.make with the
target split and the seed, stable counter regions), only reading the simulator state (RoboCasa unchanged).  Per
image: every entity (X0's: objects by category and compact fixtures), duplicates and hidden ones included, with its
visible pixels; per scene: the names of all its objects and fixtures (any kind, for the "not there" answers) and
the door openness (0..1, the fixture's own get_joint_state over its door joints) of the compact fixtures with doors.
Each re-rendered image is compared with X0's file pixel for pixel (select refuses to run if any differs).
select: four kinds, answers balanced (the design's numbers; open/closed and counting take what there is and
existence / identification fill up to TOTAL):
    existence       "Is there a <name> in this image?" A/B = yes/no in either order.  Yes: an entity with >= 400
                    visible pixels.  No: a name that no object or fixture of the scene has, drawn from the most
                    common names of X0's pool (POPE's "popular" negatives), not sharing a word or a synonym group
                    with anything in the scene.  Half yes, half no
    identification  "Which of these is in this image?" one visible (>= 400 px) entity, one name absent as above
    open_closed     "Is the door of the <fixture> open or closed?" fixtures whose name is unique in the image with
                    >= 400 px; open >= 0.5, closed <= 0.05, nothing in between.  Half each
    counting        "How many <names> can you see in this image? Answer with a number only."  Every entity of that
                    name has >= 400 px, and none has 50-400 px (a sliver makes the count ambiguous)
The right letter alternates A / B within each kind.  analyze: accuracy with 95% Wilson intervals per kind and
overall, format compliance (the reply, trimmed of spaces and a final period, is exactly a letter, or a number for
counting), bias (share of A among the two-choice replies; share of yes in existence), unparsed counted wrong, and
McNemar against the family's base (B for Qwen2.5 models, Q3B for Qwen3 ones).
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
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import x0_qa  # noqa: E402

VISIBLE = 400            # pixels: "visible" for existence, identification, open/closed and counting
SLIVER = 50              # counting: an entity with SLIVER..VISIBLE pixels makes the count ambiguous
UNIQUE_MIN = 50          # open/closed: the name is unique among the image's entities with at least this many pixels
OPEN_MIN, CLOSED_MAX = 0.5, 0.05
KINDS = ("existence", "identification", "open_closed", "counting")
TARGETS = {"existence": 100, "identification": 100, "open_closed": 60, "counting": 40}
SYNONYMS = [{"cup", "mug", "glass", "tumbler"}, {"pan", "pot", "saucepan", "skillet", "wok"},
            {"bottle", "jar", "can", "jug", "pitcher"}, {"bread", "baguette", "sandwich bread", "hotdog bun", "bagel"},
            {"knife", "knife block"}, {"kettle", "electric kettle", "kettle non electric", "teapot"},
            {"fridge", "refrigerator"}, {"oven", "toaster oven", "stove"}, {"bowl", "plate"},
            {"spoon", "wooden spoon"}, {"lemon", "lemon wedge", "lime"}, {"tangerine", "orange"}]
NUMBER_WORDS = {"zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
                "nine": 9, "ten": 10}


# -------------------------------------------------------------------------------------------- simulator side
def all_names(k) -> list:
    """Display names of every object and every fixture of the scene (any kind), for the absent names."""
    cats = {c.get("name"): str((c.get("info") or {}).get("cat") or c.get("name")) for c in getattr(k, "object_cfgs", [])}
    names = {x0_qa.display_name(cats.get(key, key)) for key in k.objects}
    for key, fx in k.fixtures.items():
        name = getattr(fx, "nat_lang", None)
        names.add(x0_qa.display_name(key if not name or name == key else name))
        names.add(x0_qa.display_name(type(fx).__name__.lower()))
    return sorted(n for n in names if n)


def door_openness(fx, k):
    """The most open door joint of a fixture (0 closed .. 1 open), None if it has no door joints."""
    try:
        joints = list(fx.door_joint_names)
        if not joints:
            return None
        return round(float(max(fx.get_joint_state(k, joints).values())), 4)
    except Exception:  # a fixture without the helpers: no door question
        return None


def scene_record(task: str, seed: int, cameras, x0_images: dict) -> dict:
    """Reset (task, seed) as X0 did and read what G0 needs (see the module text)."""
    import gymnasium as gym
    import numpy as np
    import robocasa.wrappers.gym_wrapper  # noqa: F401
    from PIL import Image

    env = gym.make(f"robocasa/{task}", enable_render=True, split="target", seed=seed)
    try:
        env.reset(seed=seed)
        k = env.unwrapped.env
        ents = x0_qa.scene_entities(k)
        doors = {e["key"]: door_openness(k.fixtures[e["key"]], k) for e in ents if e["kind"] == "fixture"}
        rec = {"task": task, "seed": seed, "names": all_names(k), "views": {}}
        for cam in cameras:
            rgb, _depth, geom = x0_qa.render_view(k.sim, cam)
            x0_rgb = np.asarray(Image.open(x0_images[cam]).convert("RGB"))
            view = []
            for e in ents:
                ys, xs = np.nonzero(np.isin(geom, list(e["geoms"])))
                n = len(xs)
                row = {"key": e["key"], "name": e["name"], "kind": e["kind"], "pixels": int(n)}
                if n:
                    row.update(cx=round(float(xs.mean()), 1), cy=round(float(ys.mean()), 1))
                if e["kind"] == "fixture" and doors.get(e["key"]) is not None:
                    row["openness"] = doors[e["key"]]
                view.append(row)
            rec["views"][cam] = {"image": str(x0_images[cam]), "same_as_x0": bool(np.array_equal(rgb, x0_rgb)),
                                 "entities": view}
        return rec
    finally:
        env.close()


def x0_scenes(x0_questions: Path) -> dict:
    """{(task, seed): {camera: X0 image path}} of X0's questions."""
    out = {}
    for line in Path(x0_questions).read_text().splitlines():
        if line.strip():
            q = json.loads(line)
            out.setdefault((q["task"], int(q["seed"])), {})[q["camera"]] = q["image_path"]
    return out


def generate(args) -> int:
    root = HERE.parents[1]
    sys.path.insert(0, str(root))
    from downstream.eval import closed_loop

    closed_loop.stable_counter_regions()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    scenes = sorted(x0_scenes(Path(args.x0)).items())
    if args.part:
        i, kk = (int(x) for x in args.part.split("/"))
        scenes = scenes[i - 1::kk]
    done = set()
    if (out / "scenes.jsonl").is_file():
        for line in (out / "scenes.jsonl").read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["task"], int(r["seed"])))
    with open(out / "scenes.jsonl", "a") as f:
        for (task, seed), cams in scenes:
            if (task, seed) in done:
                continue
            t = time.time()
            try:
                rec = scene_record(task, seed, sorted(cams), cams)
            except Exception as exc:  # one bad reset is reported, not fatal
                rec = {"task": task, "seed": seed, "error": f"{type(exc).__name__}: {exc}"[:300]}
            rec["seconds"] = round(time.time() - t, 1)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(f"{task} {seed}: {rec['seconds']} s", flush=True)
    return 0


# ---------------------------------------------------------------------------------------------- questions (pure)
def plural(name: str) -> str:
    last = name.split()[-1]
    if re.search(r"(s|x|z|ch|sh)$", last):
        return name + "es"
    if re.search(r"[^aeiou]y$", last):
        return name[:-1] + "ies"
    return name + "s"


def article(name: str) -> str:
    return "an" if name[:1].lower() in "aeiou" else "a"


def confusable(name: str, present) -> bool:
    """True when an absent name shares a word or a synonym group with something in the scene."""
    words = set(name.split())
    for p in present:
        if words & set(p.split()):
            return True
        for group in SYNONYMS:
            if name in group and p in group:
                return True
    return False


def prompt_two(camera: str, text: str, options) -> str:
    return (f"This image is from {x0_qa.CAMERA_EN[camera]}. {text}\nA. {options[0]}\nB. {options[1]}\n"
            "Answer with the letter of the correct option only.")


def two_choice(kind: str, base: dict, text: str, truth: str, other: str, rng: random.Random, **extra) -> dict:
    options = [truth, other] if rng.random() < 0.5 else [other, truth]
    return dict(base, kind=kind, question=text, options=options, answer="AB"[options.index(truth)],
                answer_text=truth, prompt=prompt_two(base["camera"], text, options), **extra)


def candidates(scenes: list, pool: list, rng: random.Random) -> list:
    """Every candidate question of the four kinds from generate's scene records; pool: names in order of how many
    scenes have them (the "popular" negatives come from its front)."""
    out = []
    for s in scenes:
        if "error" in s:
            continue
        present = set(s["names"])
        absent = [n for n in pool if n not in present and not confusable(n, present)]
        for cam, v in sorted(s["views"].items()):
            ents = v["entities"]
            base = {"task": s["task"], "seed": s["seed"], "camera": cam, "image_path": v["image"]}
            visible = [e for e in ents if e["pixels"] >= VISIBLE]
            names_here = {e["name"] for e in ents if e["pixels"] > 0}
            seen = set()
            for e in visible:
                if e["name"] in seen:
                    continue
                seen.add(e["name"])
                out.append(dict(base, kind="existence", question=f"Is there {article(e['name'])} {e['name']} in this image?",
                                options=["yes", "no"], truth="yes", target=e, id_part=e["key"]))
                if absent:
                    neg = rng.choice(absent[:20])
                    out.append(two_choice("identification", base, "Which of these is in this image?", e["name"], neg,
                                          rng, target=e, negative=neg, id_part=f"{e['key']}_{neg}"))
            for neg in absent[:20]:
                if neg not in names_here:
                    out.append(dict(base, kind="existence", question=f"Is there {article(neg)} {neg} in this image?",
                                    options=["yes", "no"], truth="no", negative=neg, id_part=f"no_{neg}"))
            counts = Counter(e["name"] for e in ents if e["pixels"] >= UNIQUE_MIN)
            for e in visible:
                if e.get("openness") is None or counts[e["name"]] != 1:
                    continue
                state = "open" if e["openness"] >= OPEN_MIN else "closed" if e["openness"] <= CLOSED_MAX else None
                if state:
                    out.append(two_choice("open_closed", base, f"Is the door of the {e['name']} open or closed?",
                                          state, "closed" if state == "open" else "open", rng, target=e,
                                          openness=e["openness"], id_part=e["key"]))
            for name in sorted({e["name"] for e in visible}):
                same = [e for e in ents if e["name"] == name]
                if any(SLIVER <= e["pixels"] < VISIBLE for e in same):
                    continue
                n = sum(1 for e in same if e["pixels"] >= VISIBLE)
                out.append(dict(base, kind="counting", question=f"How many {plural(name)} can you see in this image? "
                                                                 "Answer with a number only.",
                                options=None, answer=str(n), answer_text=str(n), count=n,
                                targets=[e for e in same if e["pixels"] >= VISIBLE], id_part=name.replace(" ", "_"),
                                prompt=f"This image is from {x0_qa.CAMERA_EN[cam]}. How many {plural(name)} can you "
                                       f"see in this image? Answer with a number only."))
    for q in out:
        q["id"] = f"g0_{q['task']}_s{q['seed']}_{q['camera']}_{q['kind']}_{q.pop('id_part')}"
    return out


def finish_existence(q: dict, letter: str) -> dict:
    """An existence question with its yes/no options ordered so the right answer has `letter`."""
    other = "no" if q["truth"] == "yes" else "yes"
    options = [q["truth"], other] if letter == "A" else [other, q["truth"]]
    return dict(q, options=options, answer=letter, answer_text=q["truth"],
                prompt=prompt_two(q["camera"], q["question"], options))


def relabel(q: dict, letter: str) -> dict:
    if q["answer"] == letter:
        return q
    options = q["options"][::-1]
    return dict(q, options=options, answer=letter, prompt=prompt_two(q["camera"], q["question"], options))


def select(cands: list, total: int, rng: random.Random, targets=None) -> list:
    """The set (see the module text): per kind balanced, spread over scenes and cameras, letters alternating."""
    targets = dict(targets or TARGETS)
    out = {}
    oc = [q for q in cands if q["kind"] == "open_closed"]
    half = min(targets["open_closed"] // 2, sum(q["answer_text"] == "open" for q in oc),
               sum(q["answer_text"] == "closed" for q in oc))
    out["open_closed"] = (x0_qa.spread([q for q in oc if q["answer_text"] == "open"], half, rng)
                          + x0_qa.spread([q for q in oc if q["answer_text"] == "closed"], half, rng))
    cnt = [q for q in cands if q["kind"] == "counting"]
    many = x0_qa.spread([q for q in cnt if q["count"] >= 2], targets["counting"] // 2, rng)
    out["counting"] = many + x0_qa.spread([q for q in cnt if q["count"] == 1], targets["counting"] - len(many), rng)
    spare = total - len(out["open_closed"]) - len(out["counting"])
    n_ex = spare // 2 + spare % 2
    n_id = spare // 2
    ex = [q for q in cands if q["kind"] == "existence"]
    yes = x0_qa.spread([q for q in ex if q["truth"] == "yes"], n_ex // 2 + n_ex % 2, rng)
    no = x0_qa.spread([q for q in ex if q["truth"] == "no"], n_ex // 2, rng)
    out["existence"] = [q for pair in zip(yes, no) for q in pair] + yes[len(no):] + no[len(yes):]
    out["identification"] = x0_qa.spread([q for q in cands if q["kind"] == "identification"], n_id, rng)
    chosen = []
    for kind in KINDS:
        for i, q in enumerate(out[kind]):
            if kind == "existence":
                q = finish_existence(q, "AB"[(i // 2) % 2] if q["truth"] == "yes" else "AB"[(i // 2 + 1) % 2])
            elif kind != "counting":
                q = relabel(q, "AB"[i % 2])
            chosen.append(q)
    return chosen


def read_scenes(gen_dirs) -> tuple:
    """(scene records, unreadable lines) of generate folders; a line that is not JSON (e.g. zero bytes the shared
    file system shows while another host is still writing) is counted and left out."""
    scenes, bad = [], 0
    for g in gen_dirs:
        for line in (Path(g) / "scenes.jsonl").read_text(errors="replace").splitlines():
            if not line.strip("\x00 \t"):
                continue
            try:
                scenes.append(json.loads(line))
            except ValueError:
                bad += 1
    return scenes, bad


def select_cmd(args) -> int:
    scenes, bad = read_scenes(args.gen)
    if bad:
        print(f"[g0] {bad} unreadable lines left out")
    bad = [(s["task"], s["seed"], cam) for s in scenes if "error" not in s
           for cam, v in s["views"].items() if not v["same_as_x0"]]
    errors = [(s["task"], s["seed"], s["error"]) for s in scenes if "error" in s]
    if bad or errors:
        print(f"[g0] stop: {len(bad)} images differ from X0's, {len(errors)} resets failed; first: {(bad + errors)[:5]}")
        return 5
    x0_pool = sorted({n for s in scenes for v in s["views"].values() for e in v["entities"] for n in [e["name"]]})
    freq = Counter(n for s in scenes for n in set(s["names"]))
    pool = sorted(x0_pool, key=lambda n: (-freq[n], n))
    rng = random.Random(args.rng_seed)
    cands = candidates(scenes, pool, rng)
    chosen = select(cands, args.total, random.Random(args.rng_seed))
    Path(args.out).write_text("".join(json.dumps(q) + "\n" for q in chosen))
    stats = {"scenes": len(scenes), "images": sum(len(s["views"]) for s in scenes),
             "candidates": dict(Counter(q["kind"] for q in cands)), "chosen": dict(Counter(q["kind"] for q in chosen)),
             "existence_yes": sum(1 for q in chosen if q["kind"] == "existence" and q["answer_text"] == "yes"),
             "open": sum(1 for q in chosen if q["kind"] == "open_closed" and q["answer_text"] == "open"),
             "counts": dict(Counter(q["answer"] for q in chosen if q["kind"] == "counting")),
             "letter_A": dict(Counter(q["kind"] for q in chosen if q["answer"] == "A"))}
    Path(args.out).with_suffix(".stats.json").write_text(json.dumps(stats, indent=1) + "\n")
    print(json.dumps(stats))
    return 0


# ------------------------------------------------------------------------------------------------ analysis
def parse_reply(q: dict, reply: str):
    """A two-choice reply's letter (x0_qa.parse_letter, or a reply that is exactly one option's text), a counting
    reply's number (the first integer or number word); None when unreadable."""
    text = str(reply or "").strip()
    if q["kind"] == "counting":
        m = re.search(r"\d+", text)
        if m:
            return m.group(0).lstrip("0") or "0"
        w = re.search(r"\b(" + "|".join(NUMBER_WORDS) + r")\b", text.lower())
        return str(NUMBER_WORDS[w.group(1)]) if w else None
    letter = x0_qa.parse_letter(text)
    if letter:
        return letter
    plain = text.lower().rstrip(".").strip()
    for i, o in enumerate(q["options"]):
        if plain == str(o).lower():
            return "AB"[i]
    return None


def compliant(q: dict, reply: str) -> bool:
    t = str(reply or "").strip()
    t = t[:-1] if t.endswith(".") else t
    return t.isdigit() if q["kind"] == "counting" else t in ("A", "B")


def analyze(questions: list, replies: dict, references: dict) -> dict:
    """replies: {model: {id: reply}}; references: {model: the model it is compared with (or None)}."""
    parsed = {m: {q["id"]: parse_reply(q, r.get(q["id"])) for q in questions} for m, r in replies.items()}
    groups = {"all": questions, **{k: [q for q in questions if q["kind"] == k] for k in KINDS}}
    out = {}
    for m in replies:
        out[m] = {}
        for g, qs in groups.items():
            right = [parsed[m][q["id"]] == q["answer"] for q in qs]
            k, n = sum(right), len(qs)
            two = [q for q in qs if q["kind"] != "counting"]
            got = [parsed[m][q["id"]] for q in two if parsed[m][q["id"]] in ("A", "B")]
            ex = [q for q in qs if q["kind"] == "existence" and parsed[m][q["id"]] in ("A", "B")]
            row = {"n": n, "correct": k, "accuracy": k / n if n else float("nan"), "ci": x0_qa.wilson(k, n),
                   "unparsed": sum(1 for q in qs if parsed[m][q["id"]] is None),
                   "compliance": sum(compliant(q, replies[m].get(q["id"])) for q in qs) / n if n else float("nan"),
                   "a_share": sum(1 for x in got if x == "A") / len(got) if got else float("nan"),
                   "yes_share": (sum(1 for q in ex if q["options"]["AB".index(parsed[m][q["id"]])] == "yes") / len(ex))
                   if ex else float("nan")}
            ref = references.get(m)
            if ref and ref in replies:
                only_m = sum(1 for q in qs if parsed[m][q["id"]] == q["answer"] and parsed[ref][q["id"]] != q["answer"])
                only_r = sum(1 for q in qs if parsed[ref][q["id"]] == q["answer"] and parsed[m][q["id"]] != q["answer"])
                row["vs_reference"] = {"reference": ref, "only_model": only_m, "only_reference": only_r,
                                       "p": x0_qa.mcnemar_exact(only_m, only_r)}
            out[m][g] = row
    return out


def reference_of(model: str):
    return None if model in ("B", "Q3B") else ("Q3B" if model.startswith("Q3") else "B")


def analyze_cmd(args) -> int:
    questions = [json.loads(line) for line in Path(args.questions).read_text().splitlines() if line.strip()]
    replies = {}
    for spec in args.answers:
        model, path = spec.split("=", 1)
        replies[model] = {}
        for line in Path(path).read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                replies[model][r["id"]] = r.get("reply")
    res = analyze(questions, replies, {m: reference_of(m) for m in replies})
    Path(args.out).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({m: round(r["all"]["accuracy"], 4) for m, r in res.items()}))
    return 0


# ------------------------------------------------------------------------------------------------- review page
def annotate(q: dict) -> str:
    from PIL import Image, ImageDraw

    img = Image.open(q["image_path"]).convert("RGB").resize((x0_qa.SIZE * 2, x0_qa.SIZE * 2))
    d = ImageDraw.Draw(img)
    for e in q.get("targets") or ([q["target"]] if q.get("target") else []):
        if e.get("cx") is not None:
            x, y = e["cx"] * 2, e["cy"] * 2
            d.ellipse([x - 9, y - 9, x + 9, y + 9], outline=(220, 40, 40), width=4)
            d.text((x + 11, y - 8), e["name"], fill=(220, 40, 40))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


KIND_ZH = {"existence": "存在", "identification": "辨认", "open_closed": "开合", "counting": "数量"}


def sample_note(q: dict) -> str:
    """The numbers behind a sample's answer, for the review page."""
    t = q.get("target") or {}
    if q["kind"] == "existence":
        return f"目标：{t['name']}（{t['pixels']} 像素）" if t else f"目标：{q.get('negative')}（场景里没有）"
    if q["kind"] == "identification":
        return f"在图里：{t.get('name')}（{t.get('pixels')} 像素）；场景里没有：{q.get('negative')}"
    if q["kind"] == "open_closed":
        return f"开合程度 {q.get('openness')}（≥ 0.5 开，≤ 0.05 关）"
    return "每个 " + "，".join(f"{e['pixels']} 像素" for e in q.get("targets", []))


def samples_cmd(args) -> int:
    qs = [json.loads(line) for line in Path(args.questions).read_text().splitlines() if line.strip()]
    rng = random.Random(args.rng_seed)
    picked = []
    for kind in KINDS:
        pool = [q for q in qs if q["kind"] == kind]
        rng.shuffle(pool)
        picked += pool[:max(1, args.n * len(pool) // max(1, len(qs)))]
    picked = picked[:args.n]
    rows = []
    for i, q in enumerate(picked, 1):
        note = sample_note(q)
        clean = "data:image/png;base64," + base64.b64encode(Path(q["image_path"]).read_bytes()).decode()
        rows.append(f'<section><h2>{i}. {KIND_ZH[q["kind"]]} · {html.escape(q["task"])} 种子 {q["seed"]} · '
                    f'{x0_qa.CAMERA_ZH[q["camera"]]}</h2><div class="imgs"><figure><img src="{clean}" alt="原图">'
                    f'<figcaption>模型看到的图（256×256）</figcaption></figure><figure><img src="{annotate(q)}" alt="标注">'
                    f'<figcaption>标出的实体（可见像素的中心）</figcaption></figure></div><pre>{html.escape(q["prompt"])}</pre>'
                    f'<p class="ans">标准答案：<b>{q["answer"]}</b>（{html.escape(str(q["answer_text"]))}）</p>'
                    f'<p class="meta">{html.escape(note)}</p></section>')
    style = x0_qa_style()
    Path(args.out).write_text(f"<title>G0 题目样例</title><meta name=\"claude:viewport-fit-sides\" content=\"cover\">"
                              f"{style}<h1>G0 题目样例（{len(picked)} 道）</h1>{''.join(rows)}", encoding="utf-8")
    print(f"[g0] {len(picked)} samples -> {args.out}")
    return 0


def x0_qa_style() -> str:
    return ("<style>:root{--bg:#fbfaf7;--fg:#1d1d1b;--muted:#6b6a65;--line:#dedbd2;--accent:#a3402a}"
            "@media (prefers-color-scheme: dark){:root:not([data-theme=light]){--bg:#171716;--fg:#ecebe6;--muted:#a19f97;"
            "--line:#3a3935;--accent:#e08a6f;color-scheme:dark}}"
            ":root[data-theme=dark]{--bg:#171716;--fg:#ecebe6;--muted:#a19f97;--line:#3a3935;--accent:#e08a6f;color-scheme:dark}"
            "body{background:var(--bg);color:var(--fg);font:15px/1.6 system-ui,'PingFang SC','Noto Sans CJK SC',sans-serif;"
            "max-width:980px;margin:0 auto;padding-block:24px;padding-left:max(16px,env(safe-area-inset-left,0px));"
            "padding-right:max(16px,env(safe-area-inset-right,0px))}h1{font-size:22px}h2{font-size:16px;margin:0 0 8px}"
            "section{border-top:1px solid var(--line);padding:18px 0}.imgs{display:flex;flex-wrap:wrap;gap:12px}"
            "figure{margin:0;flex:1 1 300px;min-width:0}img{width:100%;max-width:512px;height:auto;image-rendering:pixelated;"
            "border:1px solid var(--line)}figcaption,.meta{color:var(--muted);font-size:13px}"
            "pre{white-space:pre-wrap;border-left:3px solid var(--accent);padding:4px 10px;margin:10px 0}"
            ".ans b{color:var(--accent)}</style>")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--x0", required=True, help="X0's questions.jsonl (its scenes, cameras and image files)")
    g.add_argument("--out", required=True)
    g.add_argument("--part", default=None, help="I/K: every K-th scene from the I-th")
    s = sub.add_parser("select")
    s.add_argument("--gen", nargs="+", required=True, help="generate folders")
    s.add_argument("--total", type=int, default=300)
    s.add_argument("--out", required=True)
    s.add_argument("--rng-seed", type=int, default=0)
    p = sub.add_parser("samples")
    p.add_argument("--questions", required=True)
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--out", required=True)
    p.add_argument("--rng-seed", type=int, default=0)
    a = sub.add_parser("analyze")
    a.add_argument("--questions", required=True)
    a.add_argument("--answers", nargs="+", required=True, help="MODEL=answers.jsonl")
    a.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    return {"generate": generate, "select": select_cmd, "samples": samples_cmd, "analyze": analyze_cmd}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
