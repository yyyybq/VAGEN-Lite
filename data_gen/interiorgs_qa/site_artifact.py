"""Turn the offline review site (site.py) into a claude.ai page: labels saved in the page, images packed.

    python -m data_gen.interiorgs_qa.site_artifact --site exps/interiorgs_qa_site_20261014 --out OUT

A published page may hold at most 511 files, and the offline site has one JPEG per image (about 2,200), so
the images go into a few JSON "packs" ({site path: data URI}), one or more per question group, each kept
under ``--pack-mb`` (the per-file limit is 16 MB). The page fetches a group's pack when it shows a
question of that group. Trials are dropped (the user decided not to run trials F, G and colour).

Labels: the page writes one document per question to its database (collection ``labels``), so the user
labels on claude.ai from any browser and the labels are read back with the artifact data tool; the
browser's local storage only keeps the current position and display toggles. Export to JSON stays as a
backup (through the platform's download prompt).
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List

TEMPLATE = Path(__file__).with_name("site_artifact_template.html")


def read_site_data(index_html: Path) -> Dict[str, Any]:
    html = index_html.read_text(encoding="utf-8")
    m = re.search(r"const DATA = (.*?);\nconst DATA_HASH = ", html, re.S)
    if not m:
        raise SystemExit(f"no embedded data in {index_html}")
    return json.loads(m.group(1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--site", required=True, help="offline site built by site.py")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pack-mb", type=float, default=11.0, help="max size of one pack file (MB, after base64)")
    ap.add_argument("--title", default="InteriorGS 样题审核")
    a = ap.parse_args()
    site, out = Path(a.site), Path(a.out)
    (out / "packs").mkdir(parents=True, exist_ok=True)
    data = read_site_data(site / "index.html")
    items: List[Dict[str, Any]] = data["items"]

    # images per group, in item order; an image shared by several items goes to the first group that uses it
    by_group: Dict[str, List[str]] = defaultdict(list)
    seen = set()
    for it in items:
        for p in it["img512"] + it["img256"]:
            if p not in seen:
                seen.add(p)
                by_group[it["group"]].append(p)
    limit = a.pack_mb * 1e6
    pack_of: Dict[str, str] = {}
    packs: List[Dict[str, Any]] = []
    for g, paths in by_group.items():
        part, size, k = {}, 0, 0
        for p in paths + [None]:
            uri = None
            if p is not None:
                uri = "data:image/jpeg;base64," + base64.b64encode((site / p).read_bytes()).decode()
            if part and (p is None or size + len(uri) > limit):
                name = f"{g}_{k}"
                blob = json.dumps(part, separators=(",", ":"))
                (out / "packs" / f"{name}.json").write_text(blob)
                packs.append({"name": name, "group": g, "images": len(part), "bytes": len(blob)})
                for q in part:
                    pack_of[q] = name
                part, size, k = {}, 0, k + 1
            if p is not None:
                part[p] = uri
                size += len(uri) + len(p) + 6

    page = {"version": data["version"] + "+artifact", "title": a.title, "items": items, "pack_of": pack_of}
    blob = json.dumps(page, ensure_ascii=False, separators=(",", ":"))
    h = hashlib.sha256(blob.encode()).hexdigest()[:16]
    html = TEMPLATE.read_text(encoding="utf-8").replace("/*__DATA__*/null", blob).replace("__DATA_HASH__", h)
    (out / "index.html").write_text(html, encoding="utf-8")
    (out / "packs.json").write_text(json.dumps(packs, indent=1))
    total = sum(p["bytes"] for p in packs)
    print(json.dumps({"items": len(items), "packs": len(packs), "pack_bytes": total, "page_bytes": len(html.encode()),
                      "largest_pack": max(p["bytes"] for p in packs), "hash": h}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
