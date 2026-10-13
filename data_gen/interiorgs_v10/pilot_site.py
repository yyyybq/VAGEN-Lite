"""Build the static labelling site of the pilot (one index.html + img/, opens by double-click, no server).

    python -m data_gen.interiorgs_v10.pilot_site --pilot PILOT --no-task-objects configs/no_task_objects_20261013.json \
        --uncertain-figures exps/interiorgs_v10_cabinet_audit_20261012/figures_uncertain --out SITE

PILOT holds pilot_tasks.jsonl, threshold_examples.jsonl, frames/{plain,boxes}/*.png and topdown/*.png
(``pilot.py``).  The page is blind: it shows no machine judgement (no d*, no reserve/formal pool, no visibility
measurement, no success region); threshold examples are shuffled so their bins cannot be guessed from the order.
Pure standard library, so it also runs on a laptop.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

FAMILY_ZH = {
    "projective_relations": "左右关系（从你的视角看 A 在 B 的左边或右边）",
    "absolute_positioning": "到物体的距离（离 X 约 d 米，X 在画面里）",
    "screen_occupancy": "物体在画面中的高度（X 占画面竖直方向约 k，从上到下完整可见）",
    "equidistance": "等距（离 X 和 Y 一样远，两者都在画面里）",
    "occlusion_alignment": "让目标被挡住（让 X 躲到 Y 后面看不见）",
    "fov_inclusion": "两个物体同框（X 和 Y 都完整地出现在画面里）",
}
SLOT_COLOR_ZH = {"A": "红框", "B": "蓝框", "C": "绿框", "D": "黄框"}
SITE_VERSION = "interiorgs_v10_pilot_site_v1"


def safe_name(frame_id: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in frame_id) + ".png"


def _jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def frame_role_zh(state_id: str, roles: Sequence[str], d: int) -> str:
    k = int(state_id[1:])
    if k == 0:
        return "起点画面"
    if k == d:
        return "终点画面"
    return f"中间关键帧（走了 {k} 步之后）"


METHOD_ZH = {"unique_label": "只写名字", "room": "用房间区分", "anchor": "用参照物区分", "size": "用大小比较"}
CHECK_GROUP_ZH = {
    "lateral_near_45": "左右题，终点时 B 指向 A 的线和左右方向的夹角接近 45°",
    "name_only": "所有目标都只写名字",
    "room": "有目标用房间区分",
    "reference": "有目标用参照物区分（参照物画绿框）",
}


def check_questions(t: Dict[str, Any], row: Dict[str, Any], d: int) -> List[Dict[str, Any]]:
    """The check page's questions for one task (texts are HTML-escaped here)."""
    from html import escape

    refs = "、".join(f"「{escape(x['ref'])}」" for x in t["targets"])
    qs = [{"key": "concise", "opts": "yes",
           "text": f"描述 {refs} 是否刚好够用：没有多余的修饰，需要区分的地方也说清楚了？"},
          {"key": "unique", "opts": "yes", "text": "按这些描述，能在整个场景里各自唯一找到目标吗？"}]
    group = row.get("check_group")
    end = f"s{d}"
    if group == "lateral_near_45":
        qs.append({"key": "lateral", "opts": "yes", "frame": end,
                   "text": "【终点画面】A（红框）和 B（蓝框）是左右关系，而不是一前一后？"})
    if group == "room":
        qs.append({"key": "room_name", "opts": "yes",
                   "text": "描述里的房间名（bedroom、kitchen 等）和俯视图、画面里这个房间的用途相符吗？"})
    if group == "reference":
        qs.append({"key": "ref_visible", "opts": "yes", "frame": end,
                   "text": "【终点画面】参照物（绿框）容易看清吗？"})
        qs.append({"key": "ref_apart", "opts": "yes",
                   "text": "参照物不是目标的一部分，也没有压在目标上或被目标压着？"})
    for x in t["targets"]:
        qs.append({"key": f"{end}:ident_{x['slot']}", "opts": "can", "frame": end,
                   "text": f"【终点画面】能认出{x['color']}（{x['slot']}）里的物体吗？"})
    return qs


def build_data(pilot: Path, no_task: Optional[Dict[str, Any]], uncertain_dir: Optional[Path],
               seed: int = 13, mode: str = "pilot") -> Dict[str, Any]:
    tasks = _jsonl(pilot / "pilot_tasks.jsonl")
    frames = {f["frame_id"]: f for f in _jsonl(pilot / "frames.jsonl")}
    rendered = set()
    manifest = pilot / "frames" / "render_manifest.json"
    if manifest.exists():
        rendered = {r["frame_id"] for r in json.loads(manifest.read_text())["rendered"]}
    out_tasks = []
    for i, r in enumerate(tasks):
        d = len((r.get("difficulty") or {}).get("certificate") or [])
        task_frames = sorted((f for f in frames.values() if f.get("task_id") == r["task_id"]),
                             key=lambda f: int(f["state_id"][1:]))
        slots = [s for s, _ in zip("ABCD", r["target_object"]["objects"])]
        targets = [{"slot": s, "color": SLOT_COLOR_ZH[s], "ref": e.get("ref") or f"the {e.get('label')}",
                    "label": e.get("label")} for s, e in zip(slots, r["target_object"]["objects"])]
        out_tasks.append({
            "id": r["task_id"], "n": i + 1, "scene": r["scene_id"], "family": r["task_type"],
            "family_zh": FAMILY_ZH.get(r["task_type"], r["task_type"]), "text": r.get("task_description", ""),
            "targets": targets, "is_oa": r["task_type"] == "occlusion_alignment",
            "frames": [{"sid": f["state_id"], "role": frame_role_zh(f["state_id"], f.get("roles", []), d),
                        "terminal": int(f["state_id"][1:]) == d,
                        "plain": f"img/plain/{safe_name(f['frame_id'])}", "boxes": f"img/boxes/{safe_name(f['frame_id'])}",
                        "rendered": f["frame_id"] in rendered or not rendered}
                       for f in task_frames],
            "topdown": f"img/topdown/{r['task_id']}.png",
        })
        if mode == "check":
            t = out_tasks[-1]
            methods = [METHOD_ZH.get(e.get("ref_method"), "旧规则") for e in r["target_object"]["objects"]]
            info_targets = list(t["targets"])
            if r["task_type"] == "occlusion_alignment":
                t["targets"] = t["targets"][1:]  # the hidden target cannot be recognised at the end
            t["questions"] = check_questions(t, r, d)
            t["info"] = [f"检查重点：{CHECK_GROUP_ZH.get(r.get('check_group'), r.get('check_group'))}",
                         "描述方式：" + "；".join(f"{x['slot']} {m}" for x, m in zip(info_targets, methods))]
            angle = r.get("check_lateral_angle_deg")
            if angle is not None:
                t["info"].append(f"终点时 B→A 与左右方向的夹角：{angle:.1f}°（规则要求不超过 45°）")
    examples = _jsonl(pilot / "threshold_examples.jsonl")
    rng = random.Random(seed)
    general = [e for e in examples if e["group"] == "general"]
    embedded = [e for e in examples if e["group"] == "embedded"]
    rng.shuffle(general)
    rng.shuffle(embedded)
    vis = []
    for group, items in (("一般物体", general), ("嵌入式物体（冰箱、烤箱、洗碗机等）", embedded)):
        for e in items:
            t = e["target"]
            vis.append({"id": e["example_id"], "group": group, "ref": t.get("ref") or f"the {t.get('label')}",
                        "label": t.get("label"), "plain": f"img/plain/{safe_name(e['example_id'])}",
                        "boxes": f"img/boxes/{safe_name(e['example_id'])}"})
    unc = []
    if no_task is not None:
        for sid, entries in sorted(no_task.get("scenes", {}).items()):
            for e in entries:
                if e.get("kind") != "uncertain":
                    continue
                fig = f"{sid}_{e['ins_id']}.png"
                if uncertain_dir is not None and not (uncertain_dir / fig).exists():
                    continue
                dims = [round(b - a, 2) for a, b in zip(e.get("bbox_min") or [0, 0, 0], e.get("bbox_max") or [0, 0, 0])]
                unc.append({"id": f"{sid}:{e['ins_id']}", "scene": sid, "ins_id": e["ins_id"], "label": e.get("label"),
                            "size": dims, "reason": e.get("reason") or "", "figure": f"img/uncertain/{fig}"})
    return {"version": SITE_VERSION, "mode": mode, "tasks": out_tasks, "visibility": vis, "uncertain": unc}


def build_site(pilot: Path, out: Path, no_task_path: Optional[Path] = None,
               uncertain_dir: Optional[Path] = None, mode: str = "pilot") -> Dict[str, Any]:
    no_task = json.loads(no_task_path.read_text(encoding="utf-8")) if no_task_path else None
    data = build_data(pilot, no_task, uncertain_dir, mode=mode)
    out.mkdir(parents=True, exist_ok=True)
    for sub in ("plain", "boxes"):
        src = pilot / "frames" / sub
        if src.exists():
            shutil.copytree(src, out / "img" / sub, dirs_exist_ok=True)
    if (pilot / "topdown").exists():
        shutil.copytree(pilot / "topdown", out / "img" / "topdown", dirs_exist_ok=True)
    if uncertain_dir is not None and data["uncertain"]:
        (out / "img" / "uncertain").mkdir(parents=True, exist_ok=True)
        for u in data["uncertain"]:
            shutil.copy2(uncertain_dir / Path(u["figure"]).name, out / "img" / "uncertain" / Path(u["figure"]).name)
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    data_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    template = (Path(__file__).with_name("pilot_site_template.html")).read_text(encoding="utf-8")
    html = (template.replace("/*__DATA__*/null", payload.replace("</", "<\\/"))
            .replace("__DATA_HASH__", data_hash))
    if mode == "check":
        html = html.replace("InteriorGS 试点标注", "InteriorGS 检查页")
    (out / "index.html").write_text(html, encoding="utf-8")
    summary = {"tasks": len(data["tasks"]), "task_frames": sum(len(t["frames"]) for t in data["tasks"]),
               "visibility_examples": len(data["visibility"]), "uncertain_boxes": len(data["uncertain"]),
               "data_hash": data_hash}
    (out / "site_summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pilot", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--no-task-objects", type=Path, default=None)
    ap.add_argument("--uncertain-figures", type=Path, default=None)
    ap.add_argument("--mode", choices=("pilot", "check"), default="pilot",
                    help="check: the 2026-10-14 check page (pilot.py check), questions about the three fixes")
    args = ap.parse_args(argv)
    print(json.dumps(build_site(args.pilot, args.out, args.no_task_objects, args.uncertain_figures, args.mode),
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
