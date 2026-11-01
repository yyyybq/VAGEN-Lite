"""Per-scene features of the 1,000 InteriorGS scenes for the marginal-scene screen (user 2026-10-28): object labels,
rooms and floor area from the metadata mirror, the split, and how many tasks / questions of each data set use the
scene.  Writes OUT/scene_features.json; reads only (login node, one process, one thread)."""
import collections
import json
import os
import sys
from pathlib import Path

OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
META = Path("/mnt/umm/users/yinbaiqiao/InteriorGS_meta")
MAIN = Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial")
SFT = MAIN / "interiorgs_v10_r11_20261016/sft"
R12 = MAIN / "interiorgs_v10_r12_20261019"
BANK = Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite-data/exps/transition_qa_20261019b")


def area(poly):
    s = 0.0
    for (x1, y1), (x2, y2) in zip(poly, poly[1:] + poly[:1]):
        s += x1 * y2 - x2 * y1
    return abs(s) / 2


def count_by_scene(path, key="scene_id"):
    c = collections.Counter()
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                c[r.get(key) or r.get("scene") or r.get("scene_id")] += 1
    return c


manifest = json.loads((SFT / "split_manifest.json").read_text())
split = {s: k for k in ("train", "val", "test") for s in manifest[f"{k}_scenes"]}
sources = {
    "sft_rows": count_by_scene(SFT / "rows.jsonl"),
    "tasks_registry": count_by_scene(SFT / "task_registry.jsonl"),
    "eval_val_v2": count_by_scene(R12 / "eval_val_v2/tasks.jsonl"),
    "eval_test_v2": count_by_scene(R12 / "eval_test_v2/tasks.jsonl"),
    "p1_train": count_by_scene(BANK / "p1_train.jsonl", "scene"),
    "p1_test": count_by_scene(BANK / "p1_test.jsonl", "scene"),
    "p1_probe": count_by_scene(BANK / "p1_probe.jsonl", "scene"),
}
rl = sorted(MAIN.glob("interiorgs_v10_train_r12a_20261019/rl_*/train.jsonl"))
if rl:
    sources["rl_r12a_train"] = count_by_scene(rl[0])
frame = {}
with open(SFT / "frames.jsonl", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        frame.setdefault(r["scene_id"], r["frame_id"])
scenes = []
for s in sorted(os.listdir(META)):
    d = META / s
    if not (d / "labels.json").exists():
        continue
    labels = collections.Counter(str(o.get("label", "")).strip().lower() for o in json.loads((d / "labels.json").read_text()))
    st = json.loads((d / "structure.json").read_text()) if (d / "structure.json").exists() else {}
    rooms = [area([tuple(p[:2]) for p in r.get("profile", [])]) for r in st.get("rooms", []) if len(r.get("profile", [])) >= 3]
    holes = collections.Counter(h.get("type") for h in st.get("holes", []))
    scenes.append({"scene": s, "split": split.get(s), "group": manifest["aliases"].get(s), "objects": sum(labels.values()),
                   "labels": dict(labels.most_common()), "rooms": len(rooms), "area": round(sum(rooms), 1),
                   "max_room": round(max(rooms), 1) if rooms else 0.0, "doors": holes.get("DOOR", 0),
                   "windows": holes.get("WINDOW", 0), "frame": frame.get(s),
                   "use": {k: (c.get(s, 0) if c is not None else None) for k, c in sources.items()}})
(OUT / "scene_features.json").write_text(json.dumps({"rl_source": str(rl[0]) if rl else None, "scenes": scenes}, ensure_ascii=False))
print(json.dumps({"scenes": len(scenes), "by_split": collections.Counter(x["split"] for x in scenes),
                  "with_frame": sum(bool(x["frame"]) for x in scenes), "rl_source": str(rl[0]) if rl else None,
                  "totals": {k: sum((x["use"][k] or 0) for x in scenes) for k in sources}}, default=str))
