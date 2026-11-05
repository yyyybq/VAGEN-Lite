#!/usr/bin/env python3
"""N series, "can it answer, can it do" (the user's task book of 2026-11-05, item 7): two-choice questions on the
start image of N1's merged 600 instances, asked zero-shot to the eight models, then paired with their N1 episodes.

    python downstream/n/n_qa.py truth   --instances I.jsonl --scene SID --out DIR        # simulator Python, per scene
    python downstream/n/n_qa.py collect --truth DIR --out TRUTH.jsonl
    python downstream/n/n_qa.py make    --instances I.jsonl --gen GEN [GEN ...] --truth TRUTH.jsonl --out DIR
    python downstream/n/n_qa.py samples --questions Q.jsonl --n 20 --out PAGE.html
    python downstream/n/n_qa.py analyze --questions Q.jsonl --answers M=A.jsonl [...] --episodes EVAL [EVAL ...]
                                        --instances I.jsonl --out DIR

truth: every instance of the scene, from its start pose, each of the six actions for one step in N1's environment
  (n_env.Scene, N1's robot body; the robot hidden as in the episodes): blocked or not (n_env.Scene.try_action) and the
  shaping score after it (n_env.box_judge, the score the episodes record as max_shaping_score; a blocked action leaves
  the pose and the score where they were).  DIR/<SID>.jsonl.
make: three kinds of question, in X0's two-choice layout (a prompt naming the camera, two lettered options, "Answer
  with the letter of the correct option only."), the right letter drawn at random (seeded):
    left_right   "is the <A> to the left or to the right of the <B>?": both objects of the instance visible at the
                 start (segmentation share >= n_env.MIN_VISIBLE_FRACTION and a visible-pixel centre in the stored
                 marks), their centres at least LR_PIXELS apart across the 256-px image (X0's rule);
    near_far     "which is closer to the camera?": both visible as above; the horizontal distance from the camera to
                 each object's box footprint (the v2 judges' distance) differs by at least NF_METERS and by a factor
                 of NF_RATIO (X0's thresholds; X0 itself used the median depth of the visible pixels, which N1's
                 stored renders do not keep);
    best_action  "which single action brings the robot closer to finishing the task?" (the instance's own task
                 text): the action whose one step raises the shaping score the most against one drawn at random from
                 those at least DELTA below it; only where the best one raises the start score by at least DELTA.
  The names are the instance's object names (n_env.natural_name); an instance whose two names are the same is left
  out of the two object kinds.  DIR/questions.jsonl, DIR/summary.json.
samples: N questions spread over the kinds and task types, as a review page: the image (with the two objects marked
  for the object kinds), the prompt, the right answer and the numbers behind it.
analyze: two readings of a reply: strict (x0_qa.parse_letter: the option letter alone) and lenient (also a reply that
  is one option's text, or names exactly one option's text).  Per model, kind and reading: accuracy; the N1 episodes'
  success rate and mean max_shaping_score on the instances answered right against those answered wrong, also within
  the shortest path's length (2-3 / 4-5 / 6+ actions) and near-threshold or not; across models, on the instances where
  A answers right and B wrong, how often A succeeds and B fails against how often B succeeds and A fails (exact
  binomial test, as McNemar's).  DIR/report.json, DIR/report.md.
"""
from __future__ import annotations

import argparse
import collections
import html
import json
import math
import random
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "x"))
import n_env as N  # noqa: E402

LR_PIXELS = 30.0  # X0's left/right margin on a 256-px image
NF_METERS, NF_RATIO = 0.2, 1.2  # X0's near/far margins
DELTA = 0.02  # best_action: the least rise over the start, and the least gap to the other option
KINDS = ("left_right", "near_far", "best_action")
ACTION_TEXT = {"move_forward": "move forward", "move_backward": "move backward", "move_left": "move left",
               "move_right": "move right", "turn_left": "turn left", "turn_right": "turn right"}
CAMERA_TEXT = "This image is from the robot's camera."
ANSWER_TEXT = "Answer with the letter of the correct option only."
LENGTH_BANDS = (("2-3", 2, 3), ("4-5", 4, 5), ("6+", 6, 99))


def read_jsonl(path) -> list:
    p = Path(path)
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.is_file() else []


def write_jsonl(path, rows) -> None:
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in rows))


# ------------------------------------------------------------------------------------------------- truth
def shaping(judge: dict) -> float:
    """The episodes' shaping score of a box judge (n_episode.shaping)."""
    m = judge["metric"]
    return float(m.get("shaping_score", m.get("score")) or 0.0)


def one_step(scene, inst: dict) -> dict:
    """From the instance's start pose, each action for one step: blocked or not and the shaping score after it."""
    start = tuple(inst["start_pose"])
    base = shaping(N.box_judge(inst["item"], start))
    out = {"instance_id": inst["instance_id"], "start_shaping": base, "actions": {}}
    for a in N.ACTIONS:
        scene.set_pose(start)
        new, hit = scene.try_action(start, a)
        out["actions"][a] = {"blocked": hit is not None, "by": scene.owner(hit) if hit is not None else None,
                             "shaping": base if hit is not None else shaping(N.box_judge(inst["item"], new))}
    scene.set_pose(start)
    return out


def truth_cmd(args) -> int:
    insts = [r for r in read_jsonl(args.instances) if r["scene_id"] == args.scene]
    if not insts:
        print(f"[n_qa] no instance of {args.scene}")
        return 2
    args.out.mkdir(parents=True, exist_ok=True)
    rows = {}
    for hide in sorted({bool(i.get("hide_robot", False)) for i in insts}):  # n_episode's default and grouping
        scene = N.Scene(insts[0]["task"], int(insts[0]["seed"]), hide_robot=hide)
        try:
            for inst in insts:
                if bool(inst.get("hide_robot", False)) == hide:
                    rows[inst["instance_id"]] = one_step(scene, inst)
        finally:
            scene.close()
    rows = [rows[i["instance_id"]] for i in insts]
    write_jsonl(args.out / f"{args.scene}.jsonl", rows)
    print(f"[n_qa] {args.scene}: {len(rows)} instances")
    return 0


def collect_cmd(args) -> int:
    rows = [r for p in sorted(Path(args.truth).glob("*.jsonl")) for r in read_jsonl(p)]
    write_jsonl(args.out, rows)
    print(f"[n_qa] {len(rows)} instances from {len(list(Path(args.truth).glob('*.jsonl')))} scenes")
    return 0


# -------------------------------------------------------------------------------------------------- make
def footprint_distance(pose, obj: dict) -> float:
    """Horizontal distance from the camera (over the pose's x, y) to the object's box footprint; 0 inside it."""
    x, y = pose[0], pose[1]
    lo, hi = obj["bbox_min"], obj["bbox_max"]
    dx = max(lo[0] - x, 0.0, x - hi[0])
    dy = max(lo[1] - y, 0.0, y - hi[1])
    return math.hypot(dx, dy)


def visible_pair(inst: dict):
    """The two objects if both are visible at the start (share and stored centre) and named apart, else None."""
    objs = inst.get("objects") or []
    if len(objs) != 2:
        return None
    marks = (inst.get("marks") or {}).get("start") or {}
    vis = (inst.get("start_judge") or {}).get("visible") or {}
    if "a" not in marks or "b" not in marks:
        return None
    if any(vis.get(o["key"], 0.0) < N.MIN_VISIBLE_FRACTION for o in objs):
        return None
    names = [N.natural_name(o["name"]) for o in objs]
    if names[0] == names[1]:
        return None
    return objs, names, (marks["a"], marks["b"])


def two_choice(stem: str, options: list, right: int, rng: random.Random) -> tuple:
    """(prompt, options in their drawn order, the right letter)."""
    order = [0, 1] if rng.random() < 0.5 else [1, 0]
    shown = [options[i] for i in order]
    letter = "AB"[order.index(right)]
    prompt = f"{stem}\nA. {shown[0]}\nB. {shown[1]}\n{ANSWER_TEXT}"
    return prompt, shown, letter


def make_questions(insts: list, images: dict, truth: dict, seed: int = 20261105, delta: float = DELTA) -> tuple:
    rng = random.Random(seed)
    qs, skipped = [], collections.Counter()
    for inst in insts:
        iid = inst["instance_id"]
        common = {"instance_id": iid, "task_type": inst["task_type"], "image_path": images.get(iid)}
        if not common["image_path"]:
            skipped["no image"] += 1
            continue
        pair = visible_pair(inst)
        if pair is None:
            skipped["object kinds: not two visible, differently named objects"] += 1
        else:
            objs, names, (ma, mb) = pair
            dx = ma[0] - mb[0]
            if abs(dx) >= LR_PIXELS:
                subj = rng.randrange(2)
                other = 1 - subj
                left = (dx < 0) if subj == 0 else (dx > 0)  # the subject's centre left of the other's
                stem = f"{CAMERA_TEXT} In this image, is the {names[subj]} to the left or to the right of the {names[other]}?"
                prompt, shown, letter = two_choice(stem, ["left", "right"], 0 if left else 1, rng)
                qs.append(dict(common, id=f"{iid}::left_right", kind="left_right", prompt=prompt, options=shown,
                               answer=letter, numbers={"dx_px": round(dx, 1), "subject": names[subj],
                                                       "other": names[other]}))
            else:
                skipped["left_right: centres closer than 30 px"] += 1
            da, db = (footprint_distance(inst["start_pose"], o) for o in objs)
            near, far = (0, 1) if da < db else (1, 0)
            dn, dfar = min(da, db), max(da, db)
            if dfar - dn >= NF_METERS and dfar >= NF_RATIO * max(dn, 1e-9):
                stem = f"{CAMERA_TEXT} Which of these objects is closer to the camera?"
                prompt, shown, letter = two_choice(stem, [f"the {names[0]}", f"the {names[1]}"], near, rng)
                qs.append(dict(common, id=f"{iid}::near_far", kind="near_far", prompt=prompt, options=shown,
                               answer=letter, numbers={"distance_m": {names[0]: round(da, 3), names[1]: round(db, 3)}}))
            else:
                skipped["near_far: distances too close"] += 1
        t = truth.get(iid)
        if t is None:
            skipped["best_action: no truth"] += 1
            continue
        scores = {a: v["shaping"] for a, v in t["actions"].items()}
        best = max(scores, key=lambda a: (scores[a], -N.ACTIONS.index(a)))
        lower = [a for a in N.ACTIONS if scores[a] <= scores[best] - delta]
        if scores[best] < t["start_shaping"] + delta or not lower:
            skipped["best_action: no action rises enough or none clearly worse"] += 1
            continue
        other = rng.choice(lower)
        stem = (f"{CAMERA_TEXT} The robot's task: {inst['description']}\n"
                "Which single action brings the robot closer to finishing the task?")
        prompt, shown, letter = two_choice(stem, [ACTION_TEXT[best], ACTION_TEXT[other]], 0, rng)
        qs.append(dict(common, id=f"{iid}::best_action", kind="best_action", prompt=prompt, options=shown,
                       answer=letter, numbers={"start": round(t["start_shaping"], 4),
                                               "after": {a: round(s, 4) for a, s in scores.items()},
                                               "blocked": [a for a, v in t["actions"].items() if v["blocked"]],
                                               "best": best, "other": other}))
    return qs, dict(skipped)


def image_paths(insts: list, gens) -> dict:
    """instance id -> absolute path of its start image, from whichever generation folder holds it."""
    out = {}
    for inst in insts:
        rel = (inst.get("images") or {}).get("start") or f"images/{inst['instance_id']}_start.png"
        for g in gens:
            p = Path(g) / rel
            if p.is_file():
                out[inst["instance_id"]] = str(p)
                break
    return out


def make_cmd(args) -> int:
    insts = read_jsonl(args.instances)
    truth = {r["instance_id"]: r for r in read_jsonl(args.truth)}
    qs, skipped = make_questions(insts, image_paths(insts, args.gen), truth, args.seed, args.delta)
    args.out.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out / "questions.jsonl", qs)
    summary = {"instances": len(insts), "questions": len(qs),
               "per_kind": dict(collections.Counter(q["kind"] for q in qs)),
               "right_letter_A": {k: sum(q["answer"] == "A" for q in qs if q["kind"] == k) for k in KINDS},
               "skipped": skipped, "seed": args.seed, "delta": args.delta}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps(summary))
    return 0


# ----------------------------------------------------------------------------------------------- samples
def pick_samples(qs: list, n: int, seed: int = 0) -> list:
    """n questions taken in turn from the kinds, each kind's own taken in turn from its task types (shuffled)."""
    rng = random.Random(seed)
    pools = {k: collections.defaultdict(list) for k in KINDS}
    for q in qs:
        pools[q["kind"]][q["task_type"]].append(q)
    queues = {}
    for k in KINDS:
        for v in pools[k].values():
            rng.shuffle(v)
        types, queue = sorted(pools[k]), []
        while any(pools[k][t] for t in types):
            queue += [pools[k][t].pop() for t in types if pools[k][t]]
        queues[k] = queue
    out = []
    while len(out) < n and any(queues.values()):
        for k in KINDS:
            if queues[k] and len(out) < n:
                out.append(queues[k].pop(0))
    return out


def marked_image(q: dict, inst: dict) -> str:
    import base64
    import io

    from PIL import Image, ImageDraw

    img = Image.open(q["image_path"]).convert("RGB").resize((512, 512))
    d = ImageDraw.Draw(img)
    if q["kind"] != "best_action":
        marks = inst["marks"]["start"]
        for letter, obj in zip("ab", inst["objects"]):
            if letter in marks:
                x, y = marks[letter][0] * 2, marks[letter][1] * 2
                d.ellipse([x - 9, y - 9, x + 9, y + 9], outline=(220, 40, 40), width=4)
                d.text((x + 11, y - 8), N.natural_name(obj["name"]), fill=(220, 40, 40))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def samples_cmd(args) -> int:
    qs = read_jsonl(args.questions)
    insts = {r["instance_id"]: r for r in read_jsonl(args.instances)}
    picked = pick_samples(qs, args.n, args.seed)
    cards = []
    for k, q in enumerate(picked, 1):
        cards.append(
            f"<section><h2>{k}. {html.escape(q['kind'])} · {html.escape(q['task_type'])} · "
            f"{html.escape(q['instance_id'])}</h2><img src='{marked_image(q, insts[q['instance_id']])}' width='512'>"
            f"<pre>{html.escape(q['prompt'])}</pre><p><b>正确答案：{q['answer']}</b>（{html.escape(q['options']['AB'.index(q['answer'])])}）</p>"
            f"<pre>{html.escape(json.dumps(q['numbers'], ensure_ascii=False, indent=1))}</pre></section>")
    page = ("<!doctype html><meta charset='utf-8'><title>N 系列问答样例</title>"
            "<style>body{font-family:sans-serif;max-width:900px;margin:auto;padding:16px}section{border-bottom:1px solid #ccc;"
            "padding:12px 0}pre{white-space:pre-wrap;background:#f6f6f6;padding:8px}</style>"
            f"<h1>N 系列问答样例（{len(picked)} 道）</h1><p>红圈：两个物体在起始画面上的可见像素中心（模型看到的图里没有圈）。</p>"
            + "".join(cards))
    Path(args.out).write_text(page, encoding="utf-8")
    write_jsonl(Path(args.out).with_suffix(".jsonl"), picked)
    print(f"[n_qa] {len(picked)} samples -> {args.out}")
    return 0


# ----------------------------------------------------------------------------------------------- analyze
def parse_strict(reply: str):
    import x0_qa

    return x0_qa.parse_letter(reply)


def parse_lenient(reply: str, options: list):
    letter = parse_strict(reply)
    if letter:
        return letter
    t = re.sub(r"<[^>]*>", " ", str(reply or "")).strip().lower().rstrip(".").strip()
    for i, o in enumerate(options):
        if t == str(o).lower():
            return "AB"[i]
    hits = [i for i, o in enumerate(options) if str(o).lower() in t]
    return "AB"[hits[0]] if len(hits) == 1 else None


def binom_two_sided(k: int, n: int) -> float:
    if n == 0:
        return float("nan")
    lo = min(k, n - k)
    p = sum(math.comb(n, i) for i in range(lo + 1)) / 2 ** n
    return min(1.0, 2 * p)


def band_of(actions: int) -> str:
    for name, a, b in LENGTH_BANDS:
        if a <= actions <= b:
            return name
    return "other"


def mean(xs):
    xs = list(xs)
    return sum(xs) / len(xs) if xs else float("nan")


def episodes_of(dirs, model: str) -> dict:
    out = {}
    for d in dirs:
        for r in read_jsonl(Path(d) / model / "episodes.jsonl"):
            if not r.get("error"):
                out[r["instance_id"]] = r
    return out


def analyze(qs: list, answers: dict, episodes: dict, insts: dict) -> dict:
    """answers: {model: {question id: reply}}; episodes: {model: {instance id: row}}."""
    rep = {"models": {}, "pairs": {}}
    right = {}
    for m, ans in answers.items():
        rep["models"][m] = {}
        for reading in ("strict", "lenient"):
            for kind in KINDS:
                kq = [q for q in qs if q["kind"] == kind and q["id"] in ans]
                got = {q["id"]: (parse_strict(ans[q["id"]]) if reading == "strict"
                                 else parse_lenient(ans[q["id"]], q["options"])) for q in kq}
                ok = {q["instance_id"]: got[q["id"]] == q["answer"] for q in kq}
                right[(m, reading, kind)] = ok
                ep = episodes.get(m, {})
                row = {"n": len(kq), "correct": sum(ok.values()), "unparsed": sum(v is None for v in got.values()),
                       "accuracy": sum(ok.values()) / len(kq) if kq else float("nan"), "groups": {}}
                for name, sel in (("all", lambda i: True),
                                  *[(f"length {b}", (lambda b: lambda i: band_of(insts[i]["shortest"]["actions"]) == b)(b))
                                    for b, _, _ in LENGTH_BANDS],
                                  ("near threshold", lambda i: bool(insts[i].get("near_threshold"))),
                                  ("clear", lambda i: not insts[i].get("near_threshold"))):
                    cell = {}
                    for side, want in (("right", True), ("wrong", False)):
                        ids = [i for i, v in ok.items() if v is want and sel(i) and i in ep]
                        cell[side] = {"n": len(ids), "success": sum(bool(ep[i].get("success")) for i in ids),
                                      "max_shaping": mean(float(ep[i].get("max_shaping_score") or 0.0) for i in ids)}
                    row["groups"][name] = cell
                rep["models"][m][f"{reading}/{kind}"] = row
    models = list(answers)
    for reading in ("strict", "lenient"):
        for kind in KINDS:
            for a in models:
                for b in models:
                    if a == b:
                        continue
                    ra, rb = right.get((a, reading, kind), {}), right.get((b, reading, kind), {})
                    ea, eb = episodes.get(a, {}), episodes.get(b, {})
                    ids = [i for i in ra if ra.get(i) and rb.get(i) is False and i in ea and i in eb]
                    a_only = sum(bool(ea[i].get("success")) and not eb[i].get("success") for i in ids)
                    b_only = sum(bool(eb[i].get("success")) and not ea[i].get("success") for i in ids)
                    rep["pairs"][f"{reading}/{kind}/{a}>{b}"] = {"n": len(ids), "a_succeeds_b_fails": a_only,
                                                                 "b_succeeds_a_fails": b_only,
                                                                 "p": binom_two_sided(a_only, a_only + b_only)}
    return rep


def markdown(rep: dict) -> str:
    lines = ["# N 系列\"会答是否会做\"（自动生成）", "",
             "每格：答对实例上 N1 回合的成功数 / 实例数（平均最高塑形分） 对 答错实例上的同样数字。", ""]
    for key in sorted({k for r in rep["models"].values() for k in r}):
        lines += [f"## {key}", "", "| 模型 | 正确率 | 读不出 | 全部：答对 | 全部：答错 | 门槛附近：答对 | 门槛附近：答错 |",
                  "| --- | ---: | ---: | --- | --- | --- | --- |"]
        for m, r in rep["models"].items():
            row = r.get(key)
            if not row:
                continue

            def cell(c):
                return f"{c['success']}/{c['n']}（{c['max_shaping']:.3f}）" if c["n"] else "—"
            g = row["groups"]
            lines.append(f"| {m} | {row['correct']}/{row['n']}（{100 * row['accuracy']:.1f}%） | {row['unparsed']} | "
                         f"{cell(g['all']['right'])} | {cell(g['all']['wrong'])} | {cell(g['near threshold']['right'])} | "
                         f"{cell(g['near threshold']['wrong'])} |")
        lines.append("")
    return "\n".join(lines) + "\n"


def analyze_cmd(args) -> int:
    qs = read_jsonl(args.questions)
    insts = {r["instance_id"]: r for r in read_jsonl(args.instances)}
    answers = {}
    for spec in args.answers:
        m, path = spec.split("=", 1)
        answers[m] = {r["id"]: r.get("reply") for r in read_jsonl(path)}
    episodes = {m: episodes_of(args.episodes, m) for m in answers}
    rep = analyze(qs, answers, episodes, insts)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "report.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False) + "\n")
    (args.out / "report.md").write_text(markdown(rep), encoding="utf-8")
    print(json.dumps({m: {k: round(v["accuracy"], 3) for k, v in r.items()} for m, r in rep["models"].items()}))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("truth")
    t.add_argument("--instances", type=Path, required=True)
    t.add_argument("--scene", required=True)
    t.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("collect")
    c.add_argument("--truth", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)
    m = sub.add_parser("make")
    m.add_argument("--instances", type=Path, required=True)
    m.add_argument("--gen", type=Path, nargs="+", required=True)
    m.add_argument("--truth", type=Path, required=True)
    m.add_argument("--out", type=Path, required=True)
    m.add_argument("--seed", type=int, default=20261105)
    m.add_argument("--delta", type=float, default=DELTA)
    s = sub.add_parser("samples")
    s.add_argument("--questions", type=Path, required=True)
    s.add_argument("--instances", type=Path, required=True)
    s.add_argument("--n", type=int, default=20)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", type=Path, required=True)
    a = sub.add_parser("analyze")
    a.add_argument("--questions", type=Path, required=True)
    a.add_argument("--answers", nargs="+", required=True, help="MODEL=answers.jsonl (q0_ask.py's)")
    a.add_argument("--episodes", type=Path, nargs="+", required=True, help="N1's evaluation folders")
    a.add_argument("--instances", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    return {"truth": truth_cmd, "collect": collect_cmd, "make": make_cmd, "samples": samples_cmd,
            "analyze": analyze_cmd}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
