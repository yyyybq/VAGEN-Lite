"""Analysis 5, the pointing pilot (exploratory): 30 "point at X" questions for the original Qwen3-VL-8B and one D
checkpoint (m3ds1s0_300), to see whether the answer format parses.

Questions (build): start frames of 30 screened validation tasks (validation scenes; frames/<index>_00.jpg of a
validation result, the frame the env renders at the task's start pose, 256 x 256), 5 a family in a seeded order, X a
target named in the task text that is identifiable at the start (initial_observability).  The true box is the
target's AABB projected with the env's own camera (canonical_camera.build_canonical_camera + visual_bbox_metrics.
_project_object, as occlusion.identifiable), clipped to the image.  Score: the first "(x, y)" (or "x, y") in the
answer, read as pixels of the 256 x 256 image and, apart, as Qwen-VL's 0-1000 normalised coordinates; right when the
point falls in the true box.

    python a5_point.py build OUT
    python a5_point.py score OUT ANSWERS_DIR
"""
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix")
sys.path.insert(0, str(ROOT))
M3 = ROOT / "exps" / "matrix_v3"
VAL = M3 / "eval_val_v2_screened"
FRAMES = VAL / "m3ds1s0_300" / "frames"
SEED, PER_FAMILY = 20261106, 5
FAMILIES = ("projective_relations", "absolute_positioning", "screen_occupancy", "equidistance", "occlusion_alignment",
            "fov_inclusion")
QUESTION = ("Point to {ref}. Answer with the pixel coordinates of one point on it in this 256 x 256 image, as (x, y), "
            "x from the left edge and y from the top edge.")


def build(out: Path) -> None:
    import numpy as np

    from data_gen.interiorgs_v10.contract import NATIVE_K, NATIVE_SIZE, RENDER_SIZE
    from vagen.envs.active_spatial.canonical_camera import CANONICAL_CAMERA_H1_RESIZE_V1, build_canonical_camera
    from vagen.envs.active_spatial.visual_bbox_metrics import _project_object

    rows = [json.loads(l) for l in (VAL / "tasks.jsonl").read_text().splitlines() if l.strip()]
    rng = random.Random(SEED)
    order = list(range(len(rows)))
    rng.shuffle(order)
    picked, per = [], {f: 0 for f in FAMILIES}
    for i in order:
        r = rows[i]
        f = r["task_type"]
        if per[f] >= PER_FAMILY:
            continue
        obs = r["initial_observability"]["targets"]
        ents = [e for e in r["target_object"]["objects"] if obs.get(str(e["id"]), {}).get("identifiable")]
        frame = FRAMES / f"{i:03d}_00.jpg"
        if not ents or not frame.exists():
            continue
        e = ents[0]
        cam = build_canonical_camera(K_native=NATIVE_K, native_size=NATIVE_SIZE, render_size=RENDER_SIZE, transform="resize",
                                     c2w=np.asarray(r["init_camera"]["extrinsics"], float),
                                     camera_model_version=CANONICAL_CAMERA_H1_RESIZE_V1)
        proj = _project_object(e, np.asarray(cam.c2w), np.asarray(cam.K_effective), *cam.render_resolution)
        box = proj.get("bbox")
        if not proj.get("visible") or not box or (box[2] - box[0]) * (box[3] - box[1]) < 16:
            continue
        per[f] += 1
        box = [round(float(v), 1) for v in box]
        picked.append({"messages": [{"role": "user", "content": "<image>\n" + QUESTION.format(ref=e["ref"])},
                                    {"role": "assistant", "content": f"({(box[0] + box[2]) / 2:.0f}, {(box[1] + box[3]) / 2:.0f})"}],
                       "images": [str(frame)], "sample_id": f"pt_{r['task_id']}", "pool": "PT", "qclass": "pointing",
                       "qtype": "point_now", "family": f, "task_id": r["task_id"], "scene": r["scene_id"], "split": "val",
                       "answer": "", "answer_value": "", "options": [], "box": box, "target": e["ref"],
                       "render_size": list(cam.render_resolution)})
    out.mkdir(parents=True, exist_ok=True)
    (out / "point_probe.jsonl").write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in picked), encoding="utf-8")
    print(len(picked), "questions;", per)


def parse_point(text: str):
    m = re.search(r"\(?\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\)?", text or "")
    return (float(m.group(1)), float(m.group(2))) if m else None


def score(out: Path, answers: Path) -> None:
    probe = {r["sample_id"]: r for r in (json.loads(l) for l in (out / "point_probe.jsonl").read_text().splitlines() if l.strip())}
    rep, md = {}, ["# 分析 5：指认题试版（30 道，探索性）", "", __doc__.split("\n\n")[1].replace("\n", " "), "",
                   "| 模型 | 题数 | 读得出坐标 | 按 256 像素算落在真值框内 | 按 0–1000 归一化算落在框内 | 回答样例 |", "| --- | --- | --- | --- | --- | --- |"]
    for f in sorted(answers.glob("*.jsonl")):
        ans = [json.loads(l) for l in f.read_text(encoding="utf-8").splitlines() if l.strip()]
        n = parsed = in_px = in_norm = 0
        examples = []
        for a in ans:
            p = probe[a["sample_id"]]
            n += 1
            pt = parse_point(a.get("output"))
            if len(examples) < 2:
                examples.append((a.get("output") or "").strip().replace("\n", " ")[:60])
            if pt is None:
                continue
            parsed += 1
            x0, y0, x1, y1 = p["box"]
            W, H = p["render_size"]
            in_px += x0 <= pt[0] <= x1 and y0 <= pt[1] <= y1
            in_norm += x0 <= pt[0] * W / 1000 <= x1 and y0 <= pt[1] * H / 1000 <= y1
        rep[f.stem] = {"n": n, "parsed": parsed, "in_box_pixels": in_px, "in_box_norm1000": in_norm, "examples": examples}
        md.append(f"| `{f.stem}` | {n} | {parsed} | {in_px} | {in_norm} | " + " / ".join(f"`{e}`" for e in examples) + " |")
    (out / "point_score.json").write_text(json.dumps(rep, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    (out / "point_score.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\n".join(md))


if __name__ == "__main__":
    if sys.argv[1] == "build":
        build(Path(sys.argv[2]))
    else:
        score(Path(sys.argv[2]), Path(sys.argv[3]))
