"""configs/scene_screen/scene_exclusions_20261029.json: the tracked list of the 128 excluded scenes (2026-10-29)."""
import collections
import json
from pathlib import Path

ROOT = Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite-data")
S = ROOT / "exps/scene_screen_20261028"
e = json.loads((S / "scene_exclusions.json").read_text())
t = json.loads((S / "scene_types.json").read_text())
sc = t["scenes"]
ex = sorted(e["excluded"])
out = {
    "made": "2026-10-29",
    "decided": "the user's choices on the scene-screen page https://claude.ai/artifact/A5UDzK4r3o9Ckb7NhkRqKz over the presets "
               "of its three tiers (A exclude, B the user decides, C keep); docs/transition_qa/SCENE_SCREEN.md",
    "method": "InteriorGS has no scene-type field: types from the object labels of the metadata (exps/scene_screen_20261028: "
              f"scene_features.py, scene_classify.py with {t['overrides']} scenes set by hand), tiers by type, size and object count",
    "counts": {"scenes": len(sc), "excluded": len(ex), "kept": len(sc) - len(ex),
               "excluded_by_split": dict(collections.Counter(sc[s]["split"] for s in ex))},
    "excluded": ex,
    "type_of": {s: sc[s]["type"] for s in ex},
    "type_names": t["names"],
}
p = ROOT / "configs/scene_screen/scene_exclusions_20261029.json"
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
print(json.dumps(out["counts"]), p)
