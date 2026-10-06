#!/usr/bin/env python3
import json
import sys
from pathlib import Path
from PIL import Image

probe = Path(sys.argv[1])
summary = json.loads((probe / "summary.json").read_text())
errs = []
if (probe / "errors.jsonl").exists():
    errs = [json.loads(x) for x in (probe / "errors.jsonl").read_text().splitlines() if x.strip()]
rows = []
if (probe / "manifest.jsonl").exists():
    rows = [json.loads(x) for x in (probe / "manifest.jsonl").read_text().splitlines() if x.strip()]
ok = False
detail = {"summary": summary, "errors": errs, "opened": False}
if rows:
    img_path = Path(rows[0]["render"]["image_path"])
    im = Image.open(img_path)
    im.load()
    detail.update({
        "opened": True,
        "size": list(im.size),
        "mode": im.mode,
        "sample_id": rows[0]["sample_id"],
        "scene_id": rows[0]["scene_id"],
        "rgb_stats": rows[0]["render"]["rgb_stats"],
    })
    ok = bool(rows[0]["render"]["rgb_stats"].get("nonblank")) and im.size == (512, 512)
(probe / "probe_check.json").write_text(json.dumps(detail, indent=2) + "\n")
print(json.dumps(detail, indent=2))
sys.exit(0 if ok else 11)
