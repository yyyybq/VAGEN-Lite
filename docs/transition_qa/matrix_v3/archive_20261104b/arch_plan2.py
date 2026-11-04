"""Read-only plan for archiving the unpicked SFT epochs of E2 and Cr (the user's 甲, 2026-11-04; as arch_plan.py did for
B' and D_noP0): per cell the nine unpicked checkpoint folders and the eight top-level weight files go to AOSS and are
then deleted; the picked checkpoint and every other top-level file stay.  Checks: the pick matches cell_state.json
(and Cr's RL plans' start model), exactly ten checkpoints, everything to archive is ours to delete, no symlinks, no
live sco job names a path to delete (a job naming only the kept checkpoint is fine).  Writes /tmp/yb_msgs/arch_plan2.json.

    python arch_plan2.py JOBS_JSON
"""
import json
import os
import re
import sys
from pathlib import Path

M3 = Path("/mnt/umm/users/yinbaiqiao/VAGEN-Lite-matrix/exps/matrix_v3")
CELLS = {"E2": ("e2", []), "Cr": ("cr", ["m3_cr_s20261019", "m3_cr_s20261021"])}
WEIGHTS = [f"model-0000{i}-of-00007.safetensors" for i in range(1, 8)] + ["model.safetensors.index.json"]
uid = os.getuid()
jobs = json.loads(Path(sys.argv[1]).read_text())
live = [j for j in jobs if str(j.get("state")) not in ("SUCCEEDED", "FAILED", "STOPPED", "DELETED", "COMPLETED")]
out, problems = {}, []
for cell, (short, rl_chains) in CELLS.items():
    model = M3 / "sft" / cell / "model"
    st = json.loads((M3 / f"m3_{short}_sft" / "cell_state.json").read_text())
    picked = Path(st["pick_model"])
    assert st["stage"] in ("picked", "rl_started") and picked.parent == model and picked.is_dir(), (cell, st["stage"])
    for c in rl_chains:
        assert json.loads((M3 / c / "plan.json").read_text())["start_model"] == str(picked), c
    ckpts = sorted((p for p in model.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")),
                   key=lambda p: int(p.name.split("-")[1]))
    assert len(ckpts) == 10 and picked in ckpts, (cell, [p.name for p in ckpts])
    archive = [p for p in ckpts if p != picked] + [model / w for w in WEIGHTS]
    assert all(p.exists() for p in archive), cell
    keep = sorted(str(p) for p in model.iterdir() if p not in archive)
    pat = re.compile(rf"matrix_v3/sft/{cell}/model(?!/{picked.name}\b)")   # anything there but the kept checkpoint
    for j in live:
        if pat.search(json.dumps(j)):
            problems.append(f"live job {j.get('name')} ({j.get('display_name')}) names {cell}'s SFT folder")
    rows, nbytes, nfiles = [], 0, 0
    for a in archive:
        fs = [a] if a.is_file() else [Path(d) / f for d, _, names in os.walk(a) for f in names]
        size = 0
        for f in fs:
            s = f.lstat()
            if f.is_symlink():
                problems.append(f"symlink {f}")
            if s.st_uid != uid:
                problems.append(f"not ours {f}")
            size += s.st_size
        for d in ([a.parent] if a.is_file() else [Path(d) for d, _, _ in os.walk(a)] + [a.parent]):
            if not os.access(d, os.W_OK):
                problems.append(f"no write bit {d}")
        rows.append([str(a), size, len(fs)])
        nbytes += size
        nfiles += len(fs)
    out[cell] = {"model_dir": str(model), "picked": str(picked), "keep": keep, "archive": rows,
                 "bytes": nbytes, "files": nfiles}
    print(f"{cell}: keep {picked.name} + {len(keep) - 1} small top-level files; archive {len(archive)} entries, "
          f"{nfiles} files, {nbytes / 1e9:.1f} GB ({nbytes / 2**30:.1f} GiB)")
    print("   archive:", " ".join(Path(r[0]).name for r in rows))
print("total", f"{sum(c['bytes'] for c in out.values()) / 1e9:.1f} GB", "live jobs checked", len(live),
      "problems", len(set(problems)), sorted(set(problems))[:5])
Path("/tmp/yb_msgs/arch_plan2.json").write_text(json.dumps(out, indent=1) + "\n")
