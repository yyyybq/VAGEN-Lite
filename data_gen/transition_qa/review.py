"""Spot check of the built bank (step 2 acceptance: the user checks 30 random questions).

    python -m data_gen.transition_qa.review sample --sft-dir SFT --out OUT [--n 30]
    python -m data_gen.transition_qa.review refill --sft-dir SFT --old OLD_OUT --out OUT   # the same check, rebuilt bank
    python data_gen/interiorgs_v10/pilot_sco.py prepare --pilot OUT/review --gs-root PLY   # then submit
    python -m data_gen.transition_qa.review page --out OUT [--html OUT/review/p1_review.html]
    # occlusion questions with a suspect occluder (wardrobe, person, thin or irregular), 10 per group
    python -m data_gen.transition_qa.review occluders --sft-dir SFT --out OUT --extra EXTRA_TASKS
    python -m data_gen.transition_qa.review page --out OUT --dir review_occluders --no-funnel --title ... --html ...

``sample`` draws the questions (2 per question type from the training and test files, then random ones),
draws the task's objects on copies of their images (A red, B blue: ``admission.overlay_boxes``,
``pilot.draw_overlay``) and writes OUT/review/frames.jsonl: the view after the action of every forward question,
boxed, so the checker can see what the answer is about.  ``page`` collects everything into review_data.json
(images inline, and every question's filter checks: the measured value against the rule) and funnel_data.json
(per question type, why candidates were not asked) for the check page.
"""

from __future__ import annotations

import argparse
import base64
import collections
import json
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from . import generate as GEN
from . import geometry as G


def _image_pose(path: str, frames: Dict[str, Dict[str, Any]], interior: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    fid = GEN.frame_id_of(path)
    if fid in frames:
        return frames[fid]
    return interior[fid]


def cmd_sample(args: argparse.Namespace) -> None:
    sft, out = Path(args.sft_dir), Path(args.out)
    d = out / "review"
    (d / "img").mkdir(parents=True, exist_ok=True)
    rows = {r["task_id"]: r for r in GEN.read_jsonl(sft / "rows.jsonl")}
    frames = {f["frame_id"]: f for f in GEN.read_jsonl(sft / "frames.jsonl")}
    interior_path = out / "interior" / "frames.jsonl"
    interior = {f["frame_id"]: f for f in GEN.read_jsonl(interior_path)} if interior_path.exists() else {}
    pool = GEN.read_jsonl(out / "p1_train.jsonl") + GEN.read_jsonl(out / "p1_test.jsonl")
    rng = random.Random(args.seed)
    by_q: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for r in sorted(pool, key=lambda r: r["sample_id"]):
        by_q[r["qtype"]].append(r)
    picks: List[Dict[str, Any]] = []
    for q in sorted(by_q):
        picks += rng.sample(by_q[q], min(2, len(by_q[q])))
    rest = [r for r in pool if r["sample_id"] not in {p["sample_id"] for p in picks}]
    picks += rng.sample(rest, max(0, args.n - len(picks)))
    picks = picks[: args.n]
    rng.shuffle(picks)
    render, sample = [], []
    for i, r in enumerate(picks, 1):
        entry, post = _review_entry(i, r, rows[r["task_id"]], frames, interior, d)
        sample.append(entry)
        if post:
            render.append(post)
    GEN.write_jsonl(d / "sample.jsonl", sample)
    GEN.write_jsonl(d / "frames.jsonl", render)
    print(json.dumps({"questions": len(sample), "post_frames": len(render),
                      "by_qtype": collections.Counter(s["qtype"] for s in sample)}, indent=1))


def _review_entry(i: int, r: Dict[str, Any], row: Dict[str, Any], frames: Dict[str, Dict[str, Any]],
                  interior: Dict[str, Dict[str, Any]], d: Path) -> Any:
    """(sample entry, frame to render or None) of question ``i``: the task's objects boxed on copies of the
    question's images (A red, B blue) and, for a forward question, the view after the action."""
    from PIL import Image

    from data_gen.interiorgs_v10.admission import overlay_boxes
    from data_gen.interiorgs_v10.pilot import draw_overlay

    boxed = []
    for j, path in enumerate(r["images"]):
        f = _image_pose(path, frames, interior)
        img = draw_overlay(Image.open(path), overlay_boxes(row, np.asarray(f["c2w"], float)))
        name = f"q{i:02d}_{j}.png"
        img.save(d / "img" / name)
        boxed.append(name)
    post, render = None, None
    if r["qclass"] == "forward":
        f = _image_pose(r["images"][0], frames, interior)
        pose = G.repeated_action_pose(np.asarray(f["c2w"], float), r["action"], int(r["k"]))
        post = f"q{i:02d}_post"
        render = {"frame_id": post, "kind": "transition_qa_review", "task_id": r["task_id"],
                  "scene_id": r["scene"], "state_id": "post", "c2w": pose.tolist(),
                  "camera_item": f["camera_item"], "overlay": overlay_boxes(row, pose)}
    entry = {"n": i, "sample_id": r["sample_id"], "qtype": r["qtype"], "qclass": r["qclass"],
             "family": r["family"], "split": r["split"], "task_id": r["task_id"],
             "task_description": row["task_description"], "question": r["messages"][0]["content"],
             "answer": r["messages"][1]["content"], "options": r["options"], "boxed": boxed,
             "post_frame": post, "originals": r["images"],
             "objects": {s: e.get("ref") for s, e in zip("AB", row["target_object"]["objects"])}}
    return entry, render


def cmd_refill(args: argparse.Namespace) -> None:
    """The questions of an earlier check (``--old`` OUT) in a rebuilt bank (``--out``): a question the new bank still
    has is taken as the new bank writes it (its letters may have moved); one it dropped is replaced by a random
    question of the same type and split that the check does not hold yet (another task where possible).  The views
    after the action of kept forward questions are copied from the old check; the others are listed for rendering."""
    import shutil

    from data_gen.interiorgs_v10.pilot import safe_name

    sft, out, old = Path(args.sft_dir), Path(args.out), Path(args.old)
    d = out / "review"
    (d / "img").mkdir(parents=True, exist_ok=True)
    (d / "frames" / "boxes").mkdir(parents=True, exist_ok=True)
    rows = {r["task_id"]: r for r in GEN.read_jsonl(sft / "rows.jsonl")}
    frames = {f["frame_id"]: f for f in GEN.read_jsonl(sft / "frames.jsonl")}
    for extra in sorted(out.glob("extra_*")):  # tasks without SFT conversations (generate raw --tag)
        rows.update({r["task_id"]: r for r in GEN.read_jsonl(extra / "rows.jsonl")})
        frames.update({f["frame_id"]: f for f in GEN.read_jsonl(extra / "frames.jsonl")})
    interior = {}
    for base in (out, old):
        p = base / "interior" / "frames.jsonl"
        if p.exists():
            interior.update({f["frame_id"]: f for f in GEN.read_jsonl(p)})
    bank: Dict[str, Dict[str, Any]] = {}
    for name in ("p1_train.jsonl", "p1_test.jsonl"):
        bank.update({r["sample_id"]: r for r in GEN.read_jsonl(out / name)})
    by_type: Dict[Any, List[Dict[str, Any]]] = collections.defaultdict(list)
    for r in sorted(bank.values(), key=lambda r: r["sample_id"]):
        by_type[(r["qtype"], r["split"])].append(r)
    previous = GEN.read_jsonl(old / "review" / "sample.jsonl")
    rng = random.Random(args.seed)
    held = {s["sample_id"] for s in previous if s["sample_id"] in bank}
    tasks = {bank[s]["task_id"] for s in held}
    sample, render, report = [], [], []
    for s in sorted(previous, key=lambda s: s["n"]):
        if s["sample_id"] in bank:
            r, origin = bank[s["sample_id"]], "kept"
        else:
            pool = [r for r in by_type[(s["qtype"], s["split"])] if r["sample_id"] not in held]
            fresh = [r for r in pool if r["task_id"] not in tasks] or pool
            r, origin = rng.choice(fresh), "replacement"
            held.add(r["sample_id"])
            tasks.add(r["task_id"])
        entry, post = _review_entry(s["n"], r, rows[r["task_id"]], frames, interior, d)
        entry.update(origin=origin, replaced=None if origin == "kept" else s["sample_id"])
        if post:
            old_post = old / "review" / "frames" / "boxes" / safe_name(post["frame_id"])
            if origin == "kept" and old_post.is_file():
                shutil.copyfile(old_post, d / "frames" / "boxes" / old_post.name)
            else:
                render.append(post)
        sample.append(entry)
        report.append({"n": s["n"], "qtype": s["qtype"], "origin": origin, "sample_id": r["sample_id"],
                       "old_sample_id": s["sample_id"]})
    GEN.write_jsonl(d / "sample.jsonl", sample)
    GEN.write_jsonl(d / "frames.jsonl", render)
    print(json.dumps({"questions": len(sample), "kept": sum(x["origin"] == "kept" for x in report),
                      "replaced": [x for x in report if x["origin"] != "kept"], "to_render": len(render)}, indent=1))


# =============================================================================
# occluder check (user, 2026-10-20): occluders are solid boxes in the occlusion judge; check the suspect ones
# =============================================================================
# Occluders whose bounding box is a poor stand-in for their shape: open inside (wardrobe, question 18 of the first
# check), a person, thin or irregular objects.  Only "fully / partly hidden" answers can be wrong this way (the box
# covers more than the object), so the check draws from those.
OCCLUDER_GROUPS = {
    "wardrobe": ("wardrobe",),
    "person": ("person",),
    "thin": ("live streaming equipment", "computer", "monitor", "easel", "telescope", "musical instrument",
             "shower head", "dry branches", "tv", "microphone", "faucet combination", "clock", "sculpture"),
}
GROUP_ZH = {"wardrobe": "衣柜", "person": "人", "thin": "细或不规则"}


def occluder_group(label: str) -> str:
    return next((g for g, labels in OCCLUDER_GROUPS.items() if label in labels), "")


def spread_pick(cands: List[Dict[str, Any]], n: int, rng: random.Random, label_of) -> List[Dict[str, Any]]:
    """``n`` of ``cands``: (qtype, answer) groups in turn, then tasks and occluder labels not used yet."""
    cands = sorted(cands, key=lambda r: r["sample_id"])
    rng.shuffle(cands)
    buckets: Dict[Any, List[Dict[str, Any]]] = collections.defaultdict(list)
    for r in cands:
        buckets[(r["qtype"], r["answer_value"])].append(r)
    picked: List[Dict[str, Any]] = []
    tasks: set = set()
    labels: collections.Counter = collections.Counter()
    while len(picked) < n and any(buckets.values()):
        for key in sorted(buckets):
            b = buckets[key]
            if len(picked) >= n or not b:
                continue
            b.sort(key=lambda r: (r["task_id"] in tasks, labels[label_of(r)]))  # stable: keeps the drawn order
            r = b.pop(0)
            picked.append(r)
            tasks.add(r["task_id"])
            labels[label_of(r)] += 1
    return picked


def cmd_occluders(args: argparse.Namespace) -> None:
    """``--per-group`` occlusion questions of the bank (train, test, probe) with a "fully / partly hidden" answer per
    occluder group, boxed like ``sample``, into OUT/<dir>; plus population.json: every occlusion question of the bank
    by group and answer (to turn the error rate of the check into a share of the bank)."""
    sft, out = Path(args.sft_dir), Path(args.out)
    d = out / args.dir
    (d / "img").mkdir(parents=True, exist_ok=True)
    rows = {r["task_id"]: r for r in GEN.read_jsonl(sft / "rows.jsonl")}
    frames = {f["frame_id"]: f for f in GEN.read_jsonl(sft / "frames.jsonl")}
    for extra in [Path(p) for p in args.extra] + sorted(out.glob("extra_*")):
        rows.update({r["task_id"]: r for r in GEN.read_jsonl(extra / "rows.jsonl")})
        frames.update({f["frame_id"]: f for f in GEN.read_jsonl(extra / "frames.jsonl")})
    label_of = lambda r: rows[r["task_id"]]["target_object"]["objects"][1]["label"]  # noqa: E731
    occ = [dict(r, bank_file=f) for f in ("p1_train.jsonl", "p1_test.jsonl", "p1_probe.jsonl")
           for r in GEN.read_jsonl(out / f) if r["qtype"].startswith("occluded")]
    population: Dict[str, Dict[str, int]] = collections.defaultdict(collections.Counter)
    for r in occ:
        population[occluder_group(label_of(r)) or "other"][r["answer_value"]] += 1
    rng = random.Random(args.seed)
    picks = []
    for g in OCCLUDER_GROUPS:
        cands = [r for r in occ if occluder_group(label_of(r)) == g and r["answer_value"] in ("full", "partial")]
        picks += [(g, r) for r in spread_pick(cands, args.per_group, rng, label_of)]
    rng.shuffle(picks)
    sample, render = [], []
    for i, (g, r) in enumerate(picks, 1):
        entry, post = _review_entry(i, r, rows[r["task_id"]], frames, {}, d)
        entry.update(tag=f"遮挡物：{GROUP_ZH[g]}（{label_of(r)}）", occluder_group=g, occluder_label=label_of(r),
                     bank_file=r["bank_file"])
        sample.append(entry)
        if post:
            render.append(post)
    GEN.write_jsonl(d / "sample.jsonl", sample)
    GEN.write_jsonl(d / "frames.jsonl", render)
    (d / "population.json").write_text(json.dumps({g: dict(c) for g, c in population.items()}, indent=1))
    print(json.dumps({"questions": len(sample), "post_frames": len(render),
                      "by_group": collections.Counter(s["occluder_group"] for s in sample),
                      "by_qtype_answer": collections.Counter((s["qtype"], s["answer"]) for s in sample).most_common(),
                      "population": {g: dict(c) for g, c in population.items()}}, indent=1, default=str))


# =============================================================================
# filter checks: what each question passed, with the measured value against the rule (for the check page)
# =============================================================================
VISIBLE = {"left_right": "A、B 在当前画面都清楚可见（face_visible_v3）", "closer": "A、B 在当前画面都清楚可见（face_visible_v3）",
           "distance_bin": "A 在当前画面清楚可见（face_visible_v3）", "height_bin": "A 在当前画面清楚可见（face_visible_v3）",
           "occluded": "B（遮挡物）在当前画面清楚可见（face_visible_v3）"}


def _f(x: Any, nd: int = 2) -> str:
    return "–" if x is None else f"{float(x):.{nd}f}"


def filter_checks(rec: Dict[str, Any], cfg: Dict[str, Any]) -> List[Dict[str, str]]:
    """[{check, value, rule}] for one bank row: every filter of generate.py / questions.py it went through."""
    audit = json.loads(rec.get("audit") or "{}")
    qtype, kind = rec["qtype"], rec["qtype"].rsplit("_", 1)[0]
    after = rec["qclass"] == "forward"
    out: List[Dict[str, str]] = []

    def add(check: str, value: str, rule: str) -> None:
        out.append({"check": check, "value": value, "rule": rule})

    if qtype == "action_between":
        add("这一回合没有被挡住，前后两帧不同", "通过", "被挡住的回合两张图看不出动作，不出题")
        add("回合里只有一种动作", f"{rec['action']} × {rec['k']}", "混合动作的回合不出题")
        add("前一帧位姿做这个动作 k 次 = 后一帧位姿", "误差 ≤ 0.001", "用 contract.apply_action 复核")
        return out
    if after:
        add("动作的每一步都合法", f"{rec['action']} × {rec['k']}", "有一步被碰撞规则（厚墙）挡住就不出题")
    q_now = audit.get("quantity_before") if after else audit.get("quantity")
    q = audit.get("quantity_after") if after else audit.get("quantity")
    if kind == "goal":
        if "interior" in audit:
            s = audit["interior"]
            add("目标区域内部的成功状态", f"相对终点 右 {_f(s.get('offset_right_m'))} m、前 {_f(s.get('offset_forward_m'))} m、"
                f"转 {_f(s.get('offset_yaw_deg'), 0)}°", "挪 ±0.10 m、转 ±6° 仍然成功（终点帧大多贴着成功边界）")
        else:
            pot = audit.get("potential_after" if after else "potential")
            gates = audit.get("failed_gates_after" if after else "failed_gates") or []
            add("离成功边界够远", "通过", f"位姿挪 ±{cfg['goal_perturb_m']} m、转 ±{cfg['goal_perturb_deg']}° 后结论不变")
            add("势能（环境的同一次计算）", _f(pot), "≥ 0.5 当且仅当成功")
            add("没通过的门", "、".join(map(str, gates)) if gates else "无", "答“是”时为空")
        return out
    if kind in VISIBLE:  # forward questions too: the objects are asked about in the image the model sees
        add(VISIBLE[kind], "通过", "看不清的物体不问")

    def addq(check: str, value: str, rule: str) -> None:  # the measured quantity: of the pose after the action
        add(("动作后：" if after else "") + check, value, rule)
    if kind == "left_right":
        addq("两个物体中心都在相机前方", "通过", "")
        addq("B→A 连线和相机左右轴的夹角", "≤ 45°", "超过 45° 按任务定义是前后关系，不出题")
        if after:
            addq("动作后两个物体在画面内的比例", f"≥ {cfg['lr_post_min_inside']}", "")
        addq("两个中心的横坐标差 |uA − uB|", f"{abs(float(q)):.1f} px", f"≥ {cfg['lr_min_sep_px']:.0f} px（成功门 12 px 加 8 px 缓冲）")
    elif kind == "closer":
        addq("距离差 dA − dB（到包围盒地面投影最近边的水平距离）", f"{float(q):+.2f} m",
            f"|dA − dB| ≥ 较远者的 {100 * cfg['closer_min_rel']:.0f}%")
    elif kind == "distance_bin":
        g = round(float(q) / cfg["dist_step_m"]) * cfg["dist_step_m"]
        addq("到 A 的水平距离", f"{float(q):.2f} m（正确选项 {g:.1f} m，差 {abs(float(q) - g):.2f} m）",
            f"离正确选项 ≤ {cfg['dist_max_offset_m']} m，否则不出题")
    elif kind == "height_bin":
        g = round(float(q) / cfg["height_step"]) * cfg["height_step"]
        addq("A 上下完整在画面里", f"≥ {cfg['height_vertical_min']}", "上下被截的不问")
        addq("A 的高度占画面高度", f"{float(q):.3f}（正确选项 {g:.1f}，差 {abs(float(q) - g):.3f}）",
            f"离正确选项 ≤ {cfg['height_max_offset']}，否则不出题")
    elif kind == "occluded":
        lo, hi = cfg["hidden_partial"]
        rule = {"full": f"完全：≥ {cfg['hidden_full']}", "none": f"没有：≤ {cfg['hidden_none']}",
                "partial": f"部分：{lo}–{hi}"}[rec["answer_value"]]
        addq("A 的 8 个角在相机前方，轮廓在画面内的比例", f"≥ {cfg['oa_in_frame_min']}", "出画不算被挡（任务的门）")
        if rec["answer_value"] != "full":
            addq("A 也清楚可见", "通过", "答“没有 / 部分”时防止 A 其实被墙挡住")
        addq("A 被 B 挡住的比例（OA 判据本身）", _f(q, 3), rule + "；介于两档之间的不出题")
    elif kind == "both_in_view":
        a, b = (q or [None, None])
        addq("A、B 在画面内的比例", f"A {_f(a)}，B {_f(b)}",
            f"“在”：8 个角在前方、≥ {cfg['fov_in_min']} 且清楚可见；“不在”：≤ {cfg['fov_out_max']} 或在相机后方；介于之间不出题")
    if after:
        add("动作前 → 动作后被问的量", f"{_f(q_now) if not isinstance(q_now, list) else q_now} → "
            f"{_f(q) if not isinstance(q, list) else q}", "“–”：现在这一侧本身不出题（例如离阈值太近）")
        add("被问的量变了（changed）", "是" if rec.get("changed") else "否", "没变的最多占这一题类的 10%")
        if rec.get("flip") is not None:
            add("答案和现在不同（flip）", "是" if rec["flip"] else "否", "两种结果都常见时比例控制在 35%–65%")
    return out


def funnel_table(stats: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Per qtype: candidates asked, the reasons the others were not asked, and what balancing dropped (train)."""
    funnel = stats.get("funnel", {})
    rows: Dict[str, Dict[str, Any]] = collections.defaultdict(lambda: {"emitted": 0, "reasons": {}, "dropped": {}})
    for key, n in funnel.items():
        if ":" not in key:
            continue
        q, reason = key.split(":", 1)
        if reason.startswith("emitted"):
            rows[q]["emitted"] += n
        else:
            rows[q]["reasons"][reason] = n
    for q, rep in stats.get("train", {}).get("balance", {}).items():
        rows[q]["dropped"] = rep.get("dropped", {})
    return [dict(qtype=q, **v) for q, v in sorted(rows.items())]


def cmd_page(args: argparse.Namespace) -> None:
    from data_gen.interiorgs_v10.pilot import safe_name

    out = Path(args.out)
    d = out / args.dir
    cfg = json.loads((out / "config.json").read_text())
    bank = {r["sample_id"]: r for f in ("p1_train.jsonl", "p1_test.jsonl", "p1_probe.jsonl") if (out / f).exists()
            for r in GEN.read_jsonl(out / f)}

    def b64(p: Path) -> str:
        return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()

    items = []
    for s in GEN.read_jsonl(d / "sample.jsonl"):
        s = dict(s)
        s["images"] = [b64(d / "img" / n) for n in s.pop("boxed")]
        post = s.pop("post_frame")
        s["post_image"] = b64(d / "frames" / "boxes" / safe_name(post)) if post else None
        rec = bank[s["sample_id"]]
        s["state_role"] = rec["state_role"]
        s["checks"] = filter_checks(rec, cfg)
        items.append(s)
    (d / "review_data.json").write_text(json.dumps(items, ensure_ascii=False))
    funnel = [] if args.no_funnel else funnel_table(json.loads((out / "stats.json").read_text()))
    (d / "funnel_data.json").write_text(json.dumps(funnel, ensure_ascii=False))
    info = {"items": len(items), "bytes": (d / "review_data.json").stat().st_size}
    if args.html:  # one self-contained file: images inline, verdicts kept in the browser and exported as JSON
        def blob(x: Any) -> str:
            return json.dumps(x, ensure_ascii=False).replace("</", "<\\/")

        meta = {"bank": str(out.resolve()), "page": Path(args.html).name, "questions": len(items),
                "title": args.title or None, "note": args.note or None}
        page = TEMPLATE.read_text(encoding="utf-8")
        page = page.replace("__REVIEW_DATA__", blob(items)).replace("__FUNNEL_DATA__", blob(funnel))
        page = page.replace("__META__", blob(meta))
        Path(args.html).write_text(page, encoding="utf-8", newline="\n")
        info["html"] = str(Path(args.html).resolve())
    print(json.dumps(info))


TEMPLATE = Path(__file__).with_name("review_page.html")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("sample")
    a.add_argument("--sft-dir", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--n", type=int, default=30)
    a.add_argument("--seed", type=int, default=20261018)
    a.set_defaults(func=cmd_sample)
    f = sub.add_parser("refill")
    f.add_argument("--sft-dir", required=True)
    f.add_argument("--old", required=True, help="the bank folder of the earlier check (its review/sample.jsonl)")
    f.add_argument("--out", required=True, help="the rebuilt bank")
    f.add_argument("--seed", type=int, default=20261020)
    f.set_defaults(func=cmd_refill)
    o = sub.add_parser("occluders", help="occlusion questions with suspect occluders (wardrobe, person, thin)")
    o.add_argument("--sft-dir", required=True)
    o.add_argument("--out", required=True, help="the bank")
    o.add_argument("--dir", default="review_occluders")
    o.add_argument("--extra", action="append", default=[], help="a folder of extra tasks (rows.jsonl, frames.jsonl)")
    o.add_argument("--per-group", type=int, default=10)
    o.add_argument("--seed", type=int, default=20261020)
    o.set_defaults(func=cmd_occluders)
    b = sub.add_parser("page")
    b.add_argument("--out", required=True)
    b.add_argument("--dir", default="review", help="the check's folder under OUT")
    b.add_argument("--html", default="", help="also write the self-contained check page here")
    b.add_argument("--title", default="", help="the page's heading (default: the 30-question check)")
    b.add_argument("--note", default="", help="one more paragraph under the heading")
    b.add_argument("--no-funnel", action="store_true", help="leave out the table of candidates not asked")
    b.set_defaults(func=cmd_page)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
