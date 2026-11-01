"""The scene-screen review page (user 2026-10-28): OUT/scene_types.json (scene_classify.py) -> OUT/scene_screen.html.
Cards with a frame (the round-11 SFT render of a task start) or the floor plan (occupancy.png) for every scene of tiers
A and B and every non-home scene; the 720 ordinary homes as compact rows."""
import base64
import io
import json
import sys
from pathlib import Path

from PIL import Image

OUT, TEMPLATE = Path(sys.argv[1]), Path(sys.argv[2])
META = Path("/mnt/umm/users/yinbaiqiao/InteriorGS_meta")
FRAMES = Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite/exps/vagen_active_spatial/interiorgs_v10_r11_20261016/sft/frames/plain")
d = json.loads((OUT / "scene_types.json").read_text())

RULES = {
    "exhibition": "有展墙、展台、雕塑、电子屏、沙盘这类东西，别的很少（结构和灯以外不到 200 件），或展墙在 4 面以上",
    "garage": "2 辆以上的车，或地面标线、岗亭",
    "track": "卡丁车、轮胎墙或路障",
    "outdoor": "游乐设施、滑梯、篮球架、路灯，或成片的围栏加绿植",
    "empty": "结构和灯以外不到 40 件东西（墙、门窗、灯、装饰画这类不算）",
    "hall": "100 把以上的椅子、成排的沙发、讲台或舞台，或 200 套以上餐具（宴会厅）",
    "market": "1,000 件以上包装食品、饮料、日用品，或 300 件以上再加 3 个以上货架、收银台",
    "spa": "桑拿炉、浴池加毛巾、按摩床，或 50 条以上毛巾",
    "game": "台球桌、麻将桌，或 3 台以上游戏机",
    "gym": "10 件以上运动器材",
    "lab": "试管、烧杯、实验用品 10 件以上",
    "ktv": "话筒和音响加起来 6 件以上，配沙发",
    "home": "床、卫浴、厨房三样里至少有两样",
    "food": "成套餐具、酒和酒杯、成排的杯子、冰块、茶壶或吧台",
    "shop": "眼镜、雪茄香烟、化妆品、面包、衣架婚纱鞋子这类货品成批出现",
    "library": "100 本以上的书（300 本以上直接归这里）",
    "office": "会议桌、6 台以上电脑显示器键盘，或成摞的文件夹",
    "class": "画板、画架、颜料、积木 15 件以上，或手工认定的幼儿园、儿童游戏室",
    "music": "乐器、录音设备、谱架 8 件以上",
    "studio": "手工认定（直播设备、化妆品、书架）",
    "clinic": "医疗设备、药品 10 件以上",
    "pharmacy": "药盒、药瓶 100 件以上",
    "pet": "宠物、宠物笼、猫爬架 10 件以上",
    "salon": "理发设备、吹风机、指甲油",
    "other": "上面的规则都不符合，看标签是休息室、书房、小铺子这类普通室内",
}
INTRO = ("InteriorGS 的 1,000 个场景没有“场景类型”这个字段，我按每个场景标注的物体给它归了类"
         "（例如有展台、展墙、别的东西很少就算展厅；有床、卫浴、厨房里两样就算住宅），规则分错的手工改了 {overrides} 个，再分三档。"
         "科技馆 0204_840158 在“建议排除”里。页面上的选择会自动保存；你定完告诉我，我按你的选择列出要改的地方"
         "（P1 题库、探针、测试集，以及主线的任务和评测集要不要跟着改），再动手。"
         "表里的数：场景按划分数；SFT 任务、RL 训练任务是第十一轮 SFT 和第十二轮 r12a 用到这些场景的任务数；"
         "评测是第十二轮 v2 的验证 413 道 / 测试 292 道里在这些场景的道数；P1 题是题库训练、测试、探针三份合计。")
TIER_TEXT = {
    "A": ("建议排除（{n} 个）", "展厅、车库、卡丁车场、游乐场和球场、几乎是空的场景，以及任何类型里物体不到 40 件的。默认“排除”。"),
    "B": ("你定（{n} 个）", "成排座位的礼堂和宴会厅、超市、洗浴、台球麻将、健身房、实验室、KTV，以及面积超过 300 m²、"
                        "最大一间超过 150 m² 或物体上千件的住宅和店。默认“还没定”，请逐组选。"),
    "C": ("建议保留（{n} 个）", "住宅和普通的商业场所：餐饮、商店、图书馆、办公、教室、诊所、药店、美容美发等。默认“保留”。"),
}


def b64(img, fmt, **kw):
    buf = io.BytesIO()
    img.save(buf, fmt, **kw)
    return f"data:image/{fmt.lower()};base64," + base64.b64encode(buf.getvalue()).decode()


def use(u):
    return {"sft": u.get("tasks_registry") or 0, "rl": u.get("rl_r12a_train") or 0, "ev": u.get("eval_val_v2") or 0,
            "et": u.get("eval_test_v2") or 0, "p1": sum(u.get(k) or 0 for k in ("p1_train", "p1_test", "p1_probe"))}


scenes, homes = [], []
for sid, s in sorted(d["scenes"].items()):
    row = {"id": sid, "sp": s["split"], "ty": s["type"], "ti": s["tier"], "ar": round(s["area"]), "rm": s["rooms"],
           "ob": s["objects"], "co": s["core"], "u": use(s["use"])}
    if s["tier"] == "C" and s["type"] == "home":
        homes.append(row)
        continue
    row["why"] = s["why"]
    row["top"] = s["top"]
    row["img"] = ""
    if s["frame"] and (FRAMES / f"{s['frame']}.png").exists():
        row["img"] = b64(Image.open(FRAMES / f"{s['frame']}.png").convert("RGB"), "JPEG", quality=72)
    occ = Image.open(META / sid / "occupancy.png").convert("L")
    occ.thumbnail((320, 240))
    row["occ"] = b64(occ, "PNG", optimize=True)
    scenes.append(row)
tt = d["summary"]["tier"]


def cols(a):
    return {"scenes": a["scenes"], "train": a["train"], "val": a["val"], "test": a["test"], "sft": a["tasks_registry"],
            "rl": a["rl_r12a_train"], "ev": a["eval_val_v2"], "et": a["eval_test_v2"],
            "p1": a["p1_train"] + a["p1_test"] + a["p1_probe"]}


tier_totals = {k: cols(tt[k]) for k in "ABC"}
tier_totals["all"] = {c: sum(tier_totals[k][c] for k in "ABC") for c in tier_totals["A"]}
meta = {"made": "2026-10-28", "intro": INTRO.format(overrides=d["overrides"]), "names": d["names"], "rules": RULES,
        "tierText": {k: [t.format(n=tt[k]["scenes"]), x] for k, (t, x) in TIER_TEXT.items()},
        "tierTotals": tier_totals, "totals": {c: tier_totals["all"][c] for c in ("sft", "rl", "ev", "et", "p1")}}
html = TEMPLATE.read_text(encoding="utf-8").replace("/*DATA*/null", json.dumps({"meta": meta, "scenes": scenes, "homes": homes},
                                                                                  ensure_ascii=False, separators=(",", ":")))
(OUT / "scene_screen.html").write_text(html, encoding="utf-8")
print(json.dumps({"cards": len(scenes), "with_frame": sum(bool(s["img"]) for s in scenes), "homes": len(homes),
                  "bytes": len(html.encode()), "tiers": {k: v["scenes"] for k, v in tier_totals.items()}}))
