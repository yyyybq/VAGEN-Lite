#!/usr/bin/env python3
"""X0b (the user's decision of 2026-10-30, "按你拟的做"; design reports/X0B_PD_LP_DESIGN_20261028.md): split X0's gap
into the picture and the wording, with two variants of X0's three question kinds.

    python downstream/x/x0b_qa.py bank   --bank BANK --out X0B1.jsonl [--per-kind 170]   # X0b-1, upstream pictures
    python downstream/x/x0b_qa.py format --x0 X0_QUESTIONS.jsonl --out X0B2.jsonl         # X0b-2, upstream wording
    python downstream/x/x0b_qa.py analyze --questions Q.jsonl --answers M=A.jsonl ... --reference B --out R.json

X0b-1: the main line's QA bank (exps/interiorgs_qa_bank_20261027, read by path) has InteriorGS views (the 256 x 256
copies) and, per question, the two objects (names, oriented boxes) and the numbers behind the answer.  From its
validation and test scenes (none of them in the bank's training split) this makes X0's three kinds with X0's prompt
and X0's thresholds:
    left_right  the bank's left/right questions (EmbSpatial ES-01/02: visible-part 2D boxes at 512 px); kept when the
                box centres differ by >= 2 x LR_PIXELS (X0's 30 px at 256)
    near_far    the bank's two-object closer-to-camera questions (3DSRBench 3D-03: mean depth of the visible parts);
                kept when the depths differ by >= NF_METERS and the farther is >= NF_RATIO x the nearer (X0's)
    front_back  the same object pairs, forward taken as the camera's heading on the floor (the robot's facing
                direction in X0): the box centres' forward coordinates differ by >= FB_METERS (X0's)
Only objects named by their category alone (the bank's "unique_in_image") are used, as in X0.  The set takes up to
per_kind questions per kind spread over scenes and views, the right option alternating A / B (x0_qa.select).
X0b-2: X0's 510 questions and images, asked in the main line's QA layout that writes the answer in tags (the
MindCube layout of data_gen/interiorgs_qa/qa.py version 2, loaded from the main checkout by path: its fixed header,
the question, the options inline; the expected reply "<answer>A. left</answer>").  The camera sentence and the
question are X0's.  Replies are read by parse_tagged (the letter inside <answer>, else x0_qa.parse_letter).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import x0_qa  # noqa: E402

MAIN = Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite")
BANK = MAIN / "exps/interiorgs_qa_bank_20261027"
SPLITS = ("val", "test")
CAMERA_TEXT = "the robot's camera"
BANK_SIZE = 512  # the bank's 2D boxes are on the 512 x 512 render


def main_qa_module(main: Path = MAIN):
    """data_gen/interiorgs_qa/qa.py of the main checkout (standard library only), loaded by path, not copied."""
    spec = importlib.util.spec_from_file_location("main_interiorgs_qa", Path(main) / "data_gen/interiorgs_qa/qa.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod  # its dataclasses look their module up there
    spec.loader.exec_module(mod)
    return mod


# ----------------------------------------------------------------------------------------------- X0b-1 (bank)
def x0_prompt(text: str, options) -> str:
    return (f"This image is from {CAMERA_TEXT}. {text}\nA. {options[0]}\nB. {options[1]}\n"
            "Answer with the letter of the correct option only.")


def heading(c2w):
    """The camera's position and its viewing direction on the floor (unit xy), from the bank's camera-to-world
    matrix (OpenCV axes: the third column is the viewing direction)."""
    fx, fy = c2w[0][2], c2w[1][2]
    n = math.hypot(fx, fy)
    return (c2w[0][3], c2w[1][3]), (fx / n, fy / n)


def forward_of(center, cam, head) -> float:
    return (center[0] - cam[0]) * head[0] + (center[1] - cam[1]) * head[1]


def simple(obj: dict) -> bool:
    return obj.get("name_method") == "unique_in_image" and bool(obj.get("display"))


def entity(obj: dict, **extra) -> dict:
    return dict({"key": str(obj["id"]), "name": obj["display"], "kind": "object"}, **extra)


def bank_candidates(questions, views: dict, image_of, rng: random.Random) -> list:
    """X0-style candidates from bank question records (see the module text).  views: view_id -> views.jsonl row;
    image_of(scene_id, view_id) -> the 256 image path."""
    out = []
    for q in questions:
        if q.get("split") not in SPLITS or q.get("media") != "image" or len(q.get("views") or []) != 1:
            continue
        objs = q.get("objects") or []
        if len(objs) != 2 or not all(simple(o) for o in objs):
            continue
        view = views.get(q["views"][0])
        if view is None:
            continue
        sub = set(q.get("subtasks") or [])
        vals = (q.get("gt") or {}).get("values") or {}
        cands = []
        roles = {o.get("role"): o for o in objs}
        if sub & {"ES-01", "ES-02"} and "bbox_left" in vals and {"left_object", "right_object"} <= set(roles):
            left, right = roles["left_object"], roles["right_object"]
            cl = (vals["bbox_left"][0] + vals["bbox_left"][2]) / 2
            cr = (vals["bbox_right"][0] + vals["bbox_right"][2]) / 2
            if cr - cl >= x0_qa.LR_PIXELS * BANK_SIZE / x0_qa.SIZE:
                s = x0_qa.SIZE / BANK_SIZE
                ents = {left["id"]: entity(left, cx=round(cl * s, 1)), right["id"]: entity(right, cx=round(cr * s, 1))}
                cands.append(("left_right", ents))
        if "3D-03" in sub and len(vals.get("mean_depths") or []) == 2:
            d = vals["mean_depths"]
            ents = {o["id"]: entity(o, depth=float(d[i])) for i, o in enumerate(objs)}
            cands.append(("near_far", ents))
        cam, head = heading(view["c2w"])
        fwd = {o["id"]: entity(o, forward=round(forward_of(o["obb"]["center"], cam, head), 3)) for o in objs}
        cands.append(("front_back", fwd))
        for kind, ents in cands:
            a, b = list(ents.values())
            if rng.random() < 0.5:
                a, b = b, a
            made = x0_qa.question(kind, a, b, "robot0_agentview_left", rng)  # thresholds and wording are X0's
            if made is None:
                continue
            made = dict(made, camera="bank", prompt=x0_prompt(made["question"], made["options"]),
                        task=q["scene_id"], seed=view["view_id"], image_path=str(image_of(q["scene_id"], view["view_id"])),
                        bank_qid=q["qid"], bank_split=q["split"])
            out.append(made)
    return dedupe(out)


def dedupe(cands: list) -> list:
    """One question per (view, kind, unordered pair)."""
    seen, out = set(), []
    for c in cands:
        k = (c["seed"], c["kind"], frozenset((c["a"]["key"], c["b"]["key"])))
        if k not in seen:
            seen.add(k)
            out.append(c)
    return out


def with_letter_bank(q: dict, letter: str) -> dict:
    if q["answer"] == letter:
        return q
    options = q["options"][::-1]
    return dict(q, options=options, answer=letter, prompt=x0_prompt(q["question"], options))


def select_bank(cands: list, per_kind: int, rng: random.Random) -> list:
    out = []
    for kind in x0_qa.KINDS:
        chosen = x0_qa.spread([q for q in cands if q["kind"] == kind], per_kind, rng)
        for i, q in enumerate(chosen):
            q = with_letter_bank(q, "AB"[i % 2])
            out.append(dict(q, id=f"bank_{q['seed']}_{kind}_{q['a']['key']}_{q['b']['key']}", object_question=True))
    return out


def bank_cmd(args) -> int:
    bank = Path(args.bank)
    views = {}
    for line in (bank / "views.jsonl").read_text().splitlines():
        if line.strip():
            v = json.loads(line)
            views[v["view_id"]] = v
    rng = random.Random(args.rng_seed)

    def image_of(scene, view):
        return bank / "views" / scene / f"{view}_256.png"

    def records():
        with open(bank / "questions.jsonl") as f:
            for line in f:
                if line.strip():
                    yield json.loads(line)

    cands = bank_candidates(records(), views, image_of, rng)
    chosen = select_bank(cands, args.per_kind, random.Random(args.rng_seed))
    missing = [q["image_path"] for q in chosen if not Path(q["image_path"]).is_file()]
    if missing:
        print(f"[x0b] {len(missing)} images missing, e.g. {missing[:3]}")
        return 4
    Path(args.out).write_text("".join(json.dumps(q) + "\n" for q in chosen))
    stats = {k: {"questions": sum(1 for q in chosen if q["kind"] == k),
                 "candidates": sum(1 for q in cands if q["kind"] == k),
                 "scenes": len({q["task"] for q in chosen if q["kind"] == k})} for k in x0_qa.KINDS}
    Path(args.out).with_suffix(".stats.json").write_text(json.dumps(stats, indent=1) + "\n")
    print(json.dumps(stats))
    return 0


# --------------------------------------------------------------------------------------------- X0b-2 (wording)
def tagged_prompt(q: dict, qa) -> str:
    """X0's camera sentence and question in the main line's tagged-answer (MindCube) layout."""
    stem = q["prompt"].split("\n", 1)[0]  # "This image is from ... . <question>"
    return qa.STYLES["mindcube"](stem, list(q["options"]))


def format_cmd(args) -> int:
    qa = main_qa_module(Path(args.main))
    out = []
    for line in Path(args.x0).read_text().splitlines():
        if line.strip():
            q = json.loads(line)
            out.append(dict(q, prompt=tagged_prompt(q, qa), max_new_tokens=32, layout="mindcube_v2"))
    Path(args.out).write_text("".join(json.dumps(q) + "\n" for q in out))
    print(f"[x0b] {len(out)} questions in the tagged layout -> {args.out}")
    return 0


def parse_tagged(text: str):
    """The letter inside <answer>...</answer> ('A' / 'B'), else x0_qa.parse_letter on the whole reply."""
    import re

    m = re.search(r"<answer>\s*\(?([AB])\b", str(text or ""), flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    return x0_qa.parse_letter(text)


def parse_tagged_text(text: str, options):
    """The lenient reading of X0b-2 (2026-10-30: S, RS and RD put the option's words in the tags, not its letter):
    the letter as parse_tagged reads it, else the option whose text equals what is inside <answer>...</answer>
    (case, spaces and a final period ignored)."""
    import re

    letter = parse_tagged(text)
    if letter:
        return letter
    m = re.search(r"<answer>(.*?)</answer>", str(text or ""), flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return None
    inner = " ".join(m.group(1).strip().rstrip(".").lower().split())
    hits = [i for i, o in enumerate(options) if " ".join(str(o).lower().split()) == inner]
    return "AB"[hits[0]] if len(hits) == 1 else None


def tagged_ok(text: str) -> bool:
    """The reply is exactly the requested form: <answer>X. option</answer> (surrounding spaces allowed)."""
    import re

    return bool(re.fullmatch(r"\s*<answer>\s*[AB]\.\s*[^<]+</answer>\s*", str(text or "")))


# ------------------------------------------------------------------------------------------------- analysis
def a_share(questions, ans) -> dict:
    """Per kind: the share of parsed replies that are 'A' (the right answer is A for half of each kind)."""
    out = {}
    for k in x0_qa.KINDS:
        got = [ans.get(q["id"]) for q in questions if q["kind"] == k]
        parsed = [g for g in got if g in ("A", "B")]
        out[k] = sum(1 for g in parsed if g == "A") / len(parsed) if parsed else float("nan")
    return out


def analyze_cmd(args) -> int:
    questions = [json.loads(line) for line in Path(args.questions).read_text().splitlines() if line.strip()]
    answers, form = {}, {}
    options = {q["id"]: q["options"] for q in questions}
    for spec in args.answers:
        model, path = spec.split("=", 1)
        rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        if args.option_text:
            answers[model] = {r["id"]: parse_tagged_text(r.get("reply"), options.get(r["id"], [])) for r in rows}
        else:
            parse = parse_tagged if args.tagged else x0_qa.parse_letter
            answers[model] = {r["id"]: parse(r.get("reply")) for r in rows}
        if args.tagged:
            form[model] = sum(1 for r in rows if tagged_ok(r.get("reply"))) / len(rows) if rows else float("nan")
    res = x0_qa.analyze(questions, answers, args.reference)
    for model in res:
        res[model]["a_share"] = a_share(questions, answers[model])
        if args.tagged:
            res[model]["tagged_form_rate"] = form[model]
    Path(args.out).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({m: {"all": round(r["all"]["accuracy"], 4), "a_share": r["a_share"]} for m, r in res.items()}))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bank")
    b.add_argument("--bank", default=str(BANK))
    b.add_argument("--per-kind", type=int, default=170)
    b.add_argument("--out", required=True)
    b.add_argument("--rng-seed", type=int, default=0)
    f = sub.add_parser("format")
    f.add_argument("--x0", required=True, help="X0's questions.jsonl")
    f.add_argument("--main", default=str(MAIN))
    f.add_argument("--out", required=True)
    a = sub.add_parser("analyze")
    a.add_argument("--questions", required=True)
    a.add_argument("--answers", nargs="+", required=True, help="MODEL=answers.jsonl")
    a.add_argument("--reference", default="B")
    a.add_argument("--tagged", action="store_true", help="replies in the <answer> layout (X0b-2)")
    a.add_argument("--option-text", action="store_true",
                   help="X0b-2 read leniently: the option's words inside <answer> count as that option")
    a.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    return {"bank": bank_cmd, "format": format_cmd, "analyze": analyze_cmd}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
