"""Unpack WalkerBench's panoramas (login node) and check that Standard-100 can run offline.

    python -m eval_external.closed_loop.walkerbench.prepare

Writes ROOT/data_prep/walkerbench.json: for the 100 standard tasks, how many of the panoramas they name (start,
targets, reference path) have an image file and a metadata record (links between panoramas) in the
checkout's data/cache.db.  Panoramas reached off the reference path are only known at run time.
"""

from __future__ import annotations

import json
import sqlite3
import zipfile
from pathlib import Path

ROOT = Path("/mnt/umm/users/yinbaiqiao/eval_external_assets")
ZIP = ROOT / "data" / "walkerbench_pano" / "panoramas.zip"
OUT_DIR = ROOT / "data" / "walkerbench_pano"
REPO = ROOT / "repos" / "WalkerBench"


def unpack() -> int:
    marker = OUT_DIR / ".unpacked"
    if marker.exists():
        return int(marker.read_text())
    with zipfile.ZipFile(ZIP) as z:
        names = [n for n in z.namelist() if n.endswith(".jpg")]
        for n in names:
            if not n.startswith("panoramas/") or ".." in n:
                raise RuntimeError(f"unexpected member {n}")
        z.extractall(OUT_DIR, members=names)
    marker.write_text(str(len(names)))
    return len(names)


def task_panos(task: dict) -> set[str]:
    ids = {task.get("spawn_point"), task.get("spawn_pano_id")}  # navigation / perception tasks
    ids.update(task.get("target_pano_ids") or [])
    ids.update(p.get("pano_id") for p in task.get("visual_path") or [])
    return {i for i in ids if i}


def main() -> None:
    n = unpack()
    db = sqlite3.connect(f"file:{REPO / 'data' / 'cache.db'}?mode=ro", uri=True)
    tables = [r[0] for r in db.execute("select name from sqlite_master where type='table'")]
    meta_table = next((t for t in tables if "meta" in t.lower()), None)
    cols = [r[1] for r in db.execute(f"pragma table_info({meta_table})")] if meta_table else []
    key = next((c for c in cols if "pano" in c.lower()), None)
    known = {r[0] for r in db.execute(f"select {key} from {meta_table}")} if key else set()
    tasks = sorted((REPO / "tasks" / "standard").glob("*.json"))
    need, have_img, have_meta = set(), set(), set()
    for t in tasks:
        ids = task_panos(json.loads(t.read_text()))
        need |= ids
    for i in need:
        if (OUT_DIR / "panoramas" / f"{i}_z2.jpg").exists():
            have_img.add(i)
        if i in known:
            have_meta.add(i)
    info = {"panoramas_unpacked": n, "tasks": len(tasks), "panoramas_named_by_tasks": len(need),
            "with_image": len(have_img), "with_metadata": len(have_meta), "metadata_table": meta_table,
            "metadata_rows": len(known), "missing_image_examples": sorted(need - have_img)[:10],
            "missing_metadata_examples": sorted(need - have_meta)[:10]}
    (ROOT / "data_prep").mkdir(exist_ok=True)
    (ROOT / "data_prep" / "walkerbench.json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
