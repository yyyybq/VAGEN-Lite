"""Scene types of the 1,000 InteriorGS scenes from their object labels (the metadata has no scene type), for the
marginal-scene screen (user 2026-10-28).  Reads OUT/scene_features.json (scene_features.py), writes
OUT/scene_types.json and prints one line per non-home scene for checking."""
import collections
import json
import sys
from pathlib import Path

OUT = Path(sys.argv[1])
scenes = json.loads((OUT / "scene_features.json").read_text())["scenes"]

BED = {"bed", "mattress", "bedside table", "bed combination"}
BATH = {"toilet", "closestool", "squatting toilets", "shower head", "shower room", "bathtub", "basin cabinet"}
KITCHEN = {"refrigerator", "fridge", "gas stoves", "smoke lampblack machine", "integrated cooktop", "cooking bench",
           "range hood", "hood", "stove"}
# structure, lights and small decoration: not counted as "things in the scene"
GENERIC = {"room", "window", "door", "downlights", "chandelier", "ceiling lamp", "plant", "ornament", "decorative painting",
           "curtain", "carpet", "wall", "floor", "ceiling", "suspended ceiling", "other", "flat panel light", "strip light",
           "track light", "spotlight", "linear lamp", "wall lamp", "wall design", "wall panels", "plaster line", "floor mold",
           "bulkhead", "partition", "venetian blinds", "trappings", "vase", "flowers", "flowerpot", "green plants", "light",
           "lamp", "window screening", "tuyere", "switch panel", "socket", "air conditioning outlets", "central air-conditioning",
           "pipeline", "styling column", "background wall", "decorative pendant", "wall stickers", "door frame", "niche",
           "ceiling combination", "ceiling combinations", "top surface modeling", "the top is carved", "edge combination",
           "interior doors", "door curtain", "steps", "stairs", "handrail", "guardrail", "iron fence", "unmarked"}


# checked by hand where the rules disagree with the label lists (2026-10-28)
OVERRIDES = {
    "0043_839876": "food",       # tea house (teapots, tea), a few hairdressing items
    "0047_839892": "class",      # children's activity room (pens, compasses, small chairs)
    "0029_839939": "exhibition", # sales office with a sand table
    "0402_840115": "exhibition", # showroom (top surface modelling, models, columns)
    "0404_840155": "food",       # bar (wine, ice cubes)
    "0434_839880": "spa",        # bath house (hand sinks, shower heads, toiletries)
    "0396_840157": "spa",        # bath house (towels, toiletries)
    "0091_839959": "spa",        # bath house (towels, slippers, shower heads)
    "0048_839893": "hall",       # hall with 100 chairs
    "0075_839936": "hall",       # hall with 64 chairs facing screens
    "0052_839901": "office",     # meeting room (sofas, pens, microphones)
    "0165_840085": "studio",     # live-streaming room
    "0166_840088": "studio",     # live-streaming room
    "0757_840087": "studio",     # live-streaming room (cosmetics)
    "0039_839888": "studio",     # live-streaming room (cosmetics, acoustic equipment)
    "0071_839932": "class",      # kindergarten (toys, small chairs, beds)
    "0162_840082": "class",      # kindergarten (toy blocks, beds)
    "0212_840152": "class",      # children's play room
    "0086_839945": "class",      # children's play room (toys, tent, consoles)
    "0420_840007": "class",      # children's play hall (toys, baskets), 674 m2
    "0088_839961": "shop",       # clothes shop
    "0021_840074": "shop",       # traditional pharmacy / tea shop (jars, bamboo slips)
    "0107_839992": "shop",       # small shop, nearly empty
    "0190_840097": "shop",       # gift shop
    "0409_840098": "shop",       # grocery
    "0118_840009": "food",       # fruit cafe
    "0213_840169": "market",     # drinks warehouse shop
    "0008_840170": "market",     # drinks shop
    "0044_839926": "market",     # supermarket (19,218 goods; some medicines)
    "0045_839925": "market",     # supermarket (16,266 goods)
    "0429_839912": "pet",        # pet shop with pet medicines
    "0184_840116": "food",       # wine bar / cellar (917 bottles)
    "0416_840042": "other",      # tea room / study (books, bamboo slips)
    "0419_840025": "food",       # cafe (coffee beans, cups, sofas)
    "0205_840168": "library",    # reading hall (10,763 books, 209 chairs)
    "0393_840188": "food",       # restaurant (44 place settings)
}


def n(L, *keys):
    return sum(L.get(k, 0) for k in keys)


def scene_type(s):
    """(type, rule) for one scene; the first rule that fits wins (v2: the shop rules before the exhibition rule)."""
    L = s["labels"]
    if s["scene"] in OVERRIDES:
        return OVERRIDES[s["scene"]], "by hand"
    home = (n(L, *BED) > 0) + (n(L, *BATH) > 0) + (n(L, *KITCHEN) > 0)
    if n(L, "car") >= 2 or n(L, "ground marking") >= 5 or n(L, "sentry box"):
        return "garage", "car / ground marking"
    if n(L, "kart") or n(L, "tire") >= 5 or n(L, "roadblock") >= 5:
        return "track", "kart / tire / roadblock"
    if n(L, "drawing board", "easel", "paint", "toy blocks") >= 15:
        return "class", "easels / paint / toy blocks"
    if (n(L, "amusement facilities") >= 5 or n(L, "slide") or n(L, "balloon") >= 50 or n(L, "basketball stands")
            or n(L, "street lamp") or (n(L, "iron fence") >= 15 and n(L, "landscape plants") >= 5)):
        return "outdoor", "amusement / court / fence + plants"
    if home >= 2:
        return "home", "bed / bath / kitchen"
    if n(L, "chair") >= 100 or n(L, "sofa") >= 60 and n(L, "chair", "sofa", "armchair") >= 80 or n(L, "podium", "stage")             or n(L, "chair combination") >= 10 or n(L, "dining plate", "knife", "fork") >= 200:
        return "hall", "rows of seats / banquet / stage"
    if n(L, "test tube", "beaker", "laboratory supplies", "microscope", "experimental supplies combination") >= 10:
        return "lab", "test tube / beaker"
    if n(L, "medicine box", "medicine bottle", "medicine tank", "medicines") >= 100:
        return "pharmacy", "medicine boxes"
    if n(L, "medical_devices", "medicines", "medicine bottle", "medicine tank") >= 10:
        return "clinic", "medical devices / medicines"
    if n(L, "pet", "pet cage", "pet nest", "pet food bowl", "cat tower") >= 10:
        return "pet", "pets"
    goods = n(L, "boxed food", "bagged food", "canned food", "fresh food", "vegetable", "drinks", "dairy", "delicatessen",
              "toiletries", "chocolate", "meat product")
    if goods >= 1000 or (goods >= 300 and n(L, "goods shelves", "pos") >= 3):  # v3: shelves or a till alone make a shop
        return "market", "1,000+ packaged goods, or 300+ with shelves / tills"
    if n(L, "cigar", "cigarette") >= 50:
        return "shop", "cigars"
    if n(L, "glasses") >= 100:
        return "shop", "glasses"
    if n(L, "cosmetic bottles", "lipstick", "eye-shadow") >= 50 and n(L, "towel") < 100:
        return "shop", "cosmetics"
    if n(L, "bread", "cake", "doughnut", "biscuit") >= 100:
        return "shop", "bakery"
    if n(L, "coat hanger", "wedding dress", "jewelry", "mannequin", "shoe") >= 10:
        return "shop", "clothes / shoes / jewellery"
    if n(L, "billiards", "cue", "pool table", "mahjong", "the mahjong table") or n(L, "games console") >= 3:
        return "game", "billiards / mahjong / consoles"
    if n(L, "microphone") + n(L, "acoustic equipment") >= 6 and n(L, "sofa", "multi person sofa", "body pillow") >= 4:
        return "ktv", "microphones + sofas"
    if n(L, "sports equipment", "rome bar", "sabre", "mask") >= 10:
        return "gym", "sports equipment"
    if n(L, "musical instrument", "recorders", "score stand", "drum", "drum kit", "instrument combination") >= 8:
        return "music", "instruments"
    if n(L, "hairdressing equipment", "hair dryer combination", "nail polish", "clippers") >= 2:
        return "salon", "hairdressing / nails"
    if (n(L, "sauna stove", "swimming pool", "pool") and n(L, "functional bed", "towel", "basin", "landscape stone") >= 4)             or n(L, "functional bed") >= 2 or n(L, "towel") >= 50:
        return "spa", "sauna / pool / massage beds / towels"
    exhibit = n(L, "exhibition wall", "flat exhibition stand", "display stand", "sculpture", "luminous character",
                "billboard", "electronic screen", "model", "e-reader", "sand table", "pavilion")
    if n(L, "exhibition wall") >= 4 or (exhibit >= 5 and s["core"] < 200) or n(L, "pavilion", "sand table"):
        return "exhibition", "exhibition walls / stands / screens, little else"
    if n(L, "book") >= 300:
        return "library", "300+ books"
    if n(L, "conference table") or n(L, "keyboard", "monitor", "computer", "notebook computer") >= 6 or n(L, "folders") >= 8:
        return "office", "desks / computers / conference table"
    if (n(L, "dining plate", "knife", "fork", "spoon", "chopsticks", "napkin") >= 20 or n(L, "wine", "wine glass", "red_wine_glass") >= 20
            or n(L, "cup") >= 15 or n(L, "ice cubes") >= 30 or n(L, "teapot") >= 10 or n(L, "bar counter", "water bar")):
        return "food", "tableware / drinks"
    if n(L, "book") >= 100:
        return "library", "100+ books"
    if s["core"] < 40:
        return "empty", "fewer than 40 objects besides structure and lights"
    return "other", "none of the rules"


for s in scenes:
    s["core"] = sum(v for k, v in s["labels"].items() if k not in GENERIC)
for s in scenes:
    s["type"], s["rule"] = scene_type(s)
NAME = {
    "home": "住宅", "food": "餐饮（餐厅、咖啡馆、酒吧、茶室）", "shop": "商店（眼镜、烟草、化妆品、面包、服装等）",
    "market": "超市、仓储式商店（上千件货品）", "library": "图书馆、书店", "office": "办公室、会议室",
    "class": "教室、画室、幼儿园、儿童游戏室", "music": "琴房、录音室", "studio": "直播间", "clinic": "诊所、医务室",
    "pharmacy": "药店", "lab": "实验室", "pet": "宠物店", "salon": "美容美发", "spa": "洗浴、桑拿、按摩",
    "game": "台球、麻将、游戏室", "ktv": "KTV 包间", "gym": "健身房、运动馆", "hall": "礼堂、剧场、宴会厅（成排座位）",
    "exhibition": "展厅、展馆、售楼处（科技馆在这里）", "garage": "车库、停车场、车展", "track": "卡丁车场",
    "outdoor": "游乐场、球场、户外广场", "empty": "几乎是空的", "other": "其他普通室内（休息室、书房、小铺子）",
}
TIER_A = {"exhibition", "garage", "track", "outdoor", "empty"}       # 建议排除
TIER_B = {"hall", "gym", "lab", "spa", "ktv", "game", "market"}       # 你定
FEW, HUGE, DENSE = 40, 300.0, 3000


def tier(s):
    """(tier, reasons): A exclude (recommended), B the user decides, C keep (recommended)."""
    why = []
    if s["core"] < FEW:
        why.append(f"结构和灯以外只有 {s['core']} 件东西")
    if s["area"] > HUGE:
        why.append(f"面积 {s['area']:.0f} m²")
    if s["max_room"] > 150 and s["type"] == "home":
        why.append(f"最大的一间 {s['max_room']:.0f} m²")
    if s["core"] > DENSE:
        why.append(f"{s['core']:,} 件东西（成千上万件小物）")
    if s["type"] in TIER_A or (s["core"] < FEW and s["type"] != "home"):
        return "A", why
    if s["type"] in TIER_B or why:
        return "B", why
    return "C", why


for s in scenes:
    s["tier"], s["why"] = tier(s)
USE = ("tasks_registry", "rl_r12a_train", "eval_val_v2", "eval_test_v2", "p1_train", "p1_test", "p1_probe")
summary = {}
for key, group in (("tier", lambda s: s["tier"]), ("type", lambda s: s["type"])):
    acc = collections.defaultdict(lambda: {"scenes": 0, "train": 0, "val": 0, "test": 0, **{u: 0 for u in USE}})
    for s in scenes:
        a = acc[group(s)]
        a["scenes"] += 1
        a[s["split"]] += 1
        for u in USE:
            a[u] += s["use"].get(u) or 0
    summary[key] = dict(acc)
totals = {u: sum(s["use"].get(u) or 0 for s in scenes) for u in USE}
(OUT / "scene_types.json").write_text(json.dumps({
    "names": NAME, "tier_a": sorted(TIER_A), "tier_b": sorted(TIER_B), "few": FEW, "huge": HUGE, "dense": DENSE,
    "totals": totals, "summary": summary,
    "overrides": len(OVERRIDES),
    "scenes": {s["scene"]: {**{k: s[k] for k in ("type", "rule", "core", "tier", "why", "split", "area", "rooms", "max_room",
                                                 "objects", "frame", "use", "group")},
                            "top": [[k, v] for k, v in sorted(s["labels"].items(), key=lambda kv: -kv[1]) if k not in GENERIC][:8]}
               for s in scenes}}, ensure_ascii=False, indent=0))
print(json.dumps(collections.Counter(s["type"] for s in scenes).most_common()))
print("totals", json.dumps(totals))
for k in ("A", "B", "C"):
    print("tier", k, json.dumps(summary["tier"][k]))
for t_, a in sorted(summary["type"].items(), key=lambda kv: (min(s["tier"] for s in scenes if s["type"] == kv[0]), -kv[1]["scenes"])):
    print(f"type {t_:10s} {NAME[t_]} :: " + json.dumps(a))
print("A by reason:", collections.Counter(s["type"] for s in scenes if s["tier"] == "A"))
print("B by reason:", collections.Counter(s["type"] if s["type"] in TIER_B else "flag:" + s["type"] for s in scenes if s["tier"] == "B"))
for t in []:
    rows = [s for s in scenes if s["type"] == t]
    print(f"\n== {t} ({len(rows)})")
    for s in rows:
        top = [(k, v) for k, v in sorted(s["labels"].items(), key=lambda kv: -kv[1]) if k not in GENERIC][:8]
        print(f"{s['scene']} {s['split'][:2]} core {s['core']} area {s['area']:.0f} | " + ", ".join(f"{k}:{v}" for k, v in top))
