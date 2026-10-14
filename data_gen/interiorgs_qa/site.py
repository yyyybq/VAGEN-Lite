"""Build the offline review site: one index.html with the data embedded, images next to it (relative paths).

    python -m data_gen.interiorgs_qa.site --run OUT1 [--run OUT2 ...] --trials TRIALS_OUT --out SITE

Labels are saved the way the v10 check pages save them (data_gen/interiorgs_v10/pilot_site_template.html):
automatically in the browser's local storage, plus "export" to a JSON file and "import" of such a file.
Images are copied as JPEG (quality 92) to keep the folder small; the model inputs stay PNG on the server.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from .geom import OBB, camera_K, project
from .qa import excluded, parse_exclude

SITE_VERSION = "interiorgs_qa_review_site_v1"
TEMPLATE = Path(__file__).with_name("site_template.html")
GROUP_ZH = {
    "S1": "S1 左右", "S2": "S2 上下与高低", "S3": "S3 离相机远近", "S4": "S4 物体之间的距离", "S5": "S5 计数",
    "S6": "S6 定位框", "S7": "S7 平行", "M1": "M1 两图之间的相机关系", "M2": "M2 相机–物体 / 物体–物体 / 测量",
    "M3": "M3 区域", "M4": "M4 跨图点对应", "M5": "M5 外观（拍到哪一面）", "M6": "M6 多步推理", "M7": "M7 原地转身",
    "M8": "M8 绕着两个物体看", "M9": "M9 围着中心物体看", "M10": "M10 平移", "M11": "M11 想象站位",
    "V1": "V1 计数", "V2": "V2 距离", "V3": "V3 尺寸", "V4": "V4 方向", "V5": "V5 路线", "V6": "V6 首次出现顺序",
}


def _jsonl(p: Path) -> List[Dict[str, Any]]:
    return [json.loads(l) for l in p.read_text().splitlines() if l.strip()] if p.exists() else []


def copy_img(src: Path, dst_dir: Path, rel: str, cache: Dict[str, str]) -> str:
    """PNG -> JPEG copy under the site, name keeps the relative path; returns the site-relative path."""
    from PIL import Image

    key = str(src)
    if key in cache:
        return cache[key]
    out_rel = "img/" + rel.replace("views/", "").replace("/", "__").replace(".png", ".jpg")
    out = dst_dir / out_rel
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        Image.open(src).convert("RGB").save(out, quality=92)
    cache[key] = out_rel
    return out_rel


def overlays(q: Dict[str, Any], views: Dict[str, Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """Per image: projected 3D boxes (12 edges, 512-px coordinates) of the objects the question names."""
    out = []
    K = camera_K(512)
    for vid in q["views"]:
        rec = views.get(vid)
        boxes = []
        if rec is not None:
            c2w = np.array(rec["c2w"], float)
            for o in q["objects"]:
                ob = o.get("obb")
                if not ob:
                    continue
                c = np.array(ob["center"])
                ax = np.array(ob["axes"])
                h = np.array(ob["half"])
                signs = np.array([[-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
                                  [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]], float)
                corners = c + (signs * h) @ ax.T
                uv, z = project(corners, c2w, K)
                if np.any(z <= 0.05):
                    continue
                boxes.append({"label": o.get("name") or o.get("display"), "role": o.get("role"),
                              "uv": np.round(uv, 1).tolist()})
        out.append(boxes)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--trials", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="InteriorGS 空间问答样题审核")
    ap.add_argument("--exclude", default="", help="RUN_SUFFIX:GROUP[:SUBTASK],qid:QID,... questions replaced by later runs")
    a = ap.parse_args()
    rules = parse_exclude(a.exclude)
    site = Path(a.out)
    site.mkdir(parents=True, exist_ok=True)
    cache: Dict[str, str] = {}
    items = []
    for run in a.run:
        run = Path(run)
        views = {r["view_id"]: r for r in _jsonl(run / "views.jsonl")}
        for q in _jsonl(run / "questions.jsonl"):
            if excluded(run, q, rules):
                continue
            imgs = q["extra"]["images"]
            img512 = [copy_img(run / p, site, p, cache) for p in imgs]
            img256 = [copy_img(run / p.replace(".png", "_256.png"), site, p.replace(".png", "_256.png"), cache) for p in imgs]
            v = q["variants"]
            items.append({
                "id": q["qid"], "group": q["group"], "group_zh": GROUP_ZH.get(q["group"], q["group"]),
                "subtasks": q["subtasks"], "benchmark": q["benchmark"], "scene": q["scene_id"], "media": q["media"],
                "img512": img512, "img256": img256, "overlays": overlays(q, views),
                "orig": v["orig"], "train": v.get("t1"), "heldout": v.get("h1"),
                "answer_type": q["answer_type"], "gt": q["gt"],
                "objects": [{k: o.get(k) for k in ("name", "display", "role", "name_method", "id")} for o in q["objects"]],
                "tokens": q.get("tokens", {}), "extra": {k: q["extra"].get(k) for k in ("video_room", "duration_s") if k in q["extra"]},
            })
    trials: Dict[str, Any] = {"f": [], "g": [], "c": []}
    if a.trials:
        tdir = Path(a.trials)
        t = json.loads((tdir / "trials.json").read_text())
        for it in t.get("trial_f", []):
            imgs = {k: copy_img(tdir / p, site, "trial/" + p, cache) for k, p in it["images"].items()}
            trials["f"].append(dict(it, images=imgs, id=f"F_{it['scene_id']}_{it['oid']}"))
        for it in t.get("trial_g", []):
            imgs = {k: copy_img(tdir / p, site, "trial/" + p, cache) for k, p in it["images"].items()}
            trials["g"].append(dict(it, images=imgs, id=f"G_{it['scene_id']}_{it['oid']}"))
        for it in t.get("trial_colour", []):
            img = copy_img(tdir / it["image"], site, "trial/" + it["image"], cache)
            trials["c"].append(dict(it, image=img, id=f"C_{it['scene_id']}_{it['oid']}"))
    order = {g: i for i, g in enumerate(GROUP_ZH)}
    items.sort(key=lambda x: (order.get(x["group"], 99), x["subtasks"][0], x["id"]))
    data = {"version": SITE_VERSION, "title": a.title, "items": items, "trials": trials}
    blob = json.dumps(data, ensure_ascii=False)
    h = hashlib.sha256(blob.encode()).hexdigest()[:16]
    html = TEMPLATE.read_text(encoding="utf-8").replace("/*__DATA__*/null", blob).replace("__DATA_HASH__", h)
    (site / "index.html").write_text(html, encoding="utf-8")
    print(json.dumps({"items": len(items), "trial_f": len(trials["f"]), "trial_g": len(trials["g"]),
                      "trial_c": len(trials["c"]), "images": len(cache), "hash": h}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
