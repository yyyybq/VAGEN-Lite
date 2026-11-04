"""After phase 2: the record of the archive of B' and D_noP0's unpicked SFT epochs (the user's 甲, 2026-11-04).
Checks that every archived entry is gone locally and every kept path is still there; writes
exps/matrix_v3/sft/archive_20261104/archive_20261104.json.  Usage: arch_record.py MANIFEST [MANIFEST ...]"""
import datetime
import json
import sys
from pathlib import Path

R = Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix/exps/matrix_v3/sft/archive_20261104")
plan = json.loads((R / "arch_plan.json").read_text())
groups, gone_all, kept_all = {}, True, True
for cell, c in plan.items():
    gone = all(not Path(p).exists() for p, _, _ in c["archive"])
    kept = all(Path(k).exists() for k in c["keep"])
    gone_all &= gone
    kept_all &= kept
    groups[cell] = {"kept": c["picked"], "kept_other": len(c["keep"]) - 1, "archived_deleted": [[p, b] for p, b, _ in c["archive"]],
                    "files": c["files"], "bytes": c["bytes"], "all_gone": gone, "keep_present": kept}
rec = {"utc": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
       "why": "the user, 2026-11-04: 甲 (先传 AOSS，核对无误再删) for B' and D_noP0's unpicked SFT epochs",
       "aoss": "fj:s3://baiqiao/vagen_lite_archive/VAGEN-Lite-matrix/exps/matrix_v3/sft/<cell>/model/...",
       "tool": "/mnt/umm/users/yinbaiqiao/aoss_archive_manifests/archive_aoss_r12.py (size check per file; phase 1 upload, "
               "phase 2 --delete after a second size check)",
       "manifests": sys.argv[1:], "groups": groups,
       "freed_gb": round(sum(g["bytes"] for g in groups.values()) / 1e9, 1),
       "freed_gib": round(sum(g["bytes"] for g in groups.values()) / 2**30, 1)}
(R / "archive_20261104.json").write_text(json.dumps(rec, indent=1, ensure_ascii=False) + "\n")
print("all archived entries gone:", gone_all, "| all kept paths present:", kept_all, "| freed", rec["freed_gb"], "GB")
