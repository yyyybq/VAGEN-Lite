"""Complete the extracted SITE-Bench media folder that lmms-eval's SITE tasks read (HF_HOME/sitebench/data/...).

    python -m data_gen.interiorgs_qa.sco prepare --job JOB -- site_extract [--workers 8]

lmms-eval (api/task.py, ``video: True`` tasks) unpacks the dataset's zip archives into HF_HOME/<cache_dir> only when
that folder does not exist yet. On this cluster the folder was created in August by a partial extraction (18 source
datasets; CLEVR, MLVU and others missing), so every SITE run since stops at a missing image or video (2026-10-29).
This job unpacks every archive of the downloaded snapshot into the folder the way lmms-eval does (a file that already
exists is kept), except that an existing file whose size differs from the archive member is written again (a file
cut off by an interrupted extraction). Runs as an SCO job: the folder is root-owned and the archives are about 61 GiB.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import zipfile
from concurrent.futures import ProcessPoolExecutor
from typing import Dict

HF_HOME = "/mnt/umm/users/yinbaiqiao/.cache/huggingface"
SNAPSHOT = f"{HF_HOME}/hub/datasets--franky-veteran--SITE-Bench/snapshots/2f405b86266cd571b5f0032016b00a51f387aa57"
TARGET = f"{HF_HOME}/sitebench"


def extract(zip_path: str) -> Dict[str, int]:
    n = {"members": 0, "written": 0, "kept": 0, "rewritten_size_mismatch": 0}
    with zipfile.ZipFile(zip_path) as z:
        for info in z.infolist():
            n["members"] += 1
            if info.is_dir():
                continue
            target = os.path.join(TARGET, info.filename)
            if os.path.exists(target):
                if os.path.getsize(target) == info.file_size:
                    n["kept"] += 1
                    continue
                n["rewritten_size_mismatch"] += 1
            else:
                n["written"] += 1
            # several archives share folders and are unpacked in parallel: make the folder first, tolerating a race
            # (zipfile's own makedirs raised FileExistsError on 2026-10-29)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            z.extract(info, TARGET)
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    zips = sorted(glob.glob(os.path.join(SNAPSHOT, "*.zip")))
    print(json.dumps({"archives": len(zips), "target": TARGET}), flush=True)
    total: Dict[str, int] = {}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for z, n in zip(zips, ex.map(extract, zips)):
            print(os.path.basename(z), json.dumps(n), flush=True)
            for k, v in n.items():
                total[k] = total.get(k, 0) + v
    print("TOTAL", json.dumps(total), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
