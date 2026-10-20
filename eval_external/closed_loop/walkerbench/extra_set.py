"""WalkerBench extra300 (decided 2026-10-20): 300 tasks of the full set (tasks/full, 1,000 files) beyond Standard-100,
in Standard's proportions: 60 each of angle, dis, height, nav, vis (the file-name prefix, as score() groups them).

    python -m eval_external.closed_loop.walkerbench.extra_set      # login node; writes ROOT/data_prep/walkerbench_extra300

Candidates: full-set files whose name, content and (category, task_id) are not among Standard-100's, whose spawn
panorama is among the
unpacked ones, and (nav / vis) whose every reference-path panorama is too.  Each category is sampled with
random.Random(SEED) from its candidates in file-name order.  selection.json lists the counts and the files.
"""

from __future__ import annotations

import hashlib
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

from eval_external.closed_loop.walkerbench.run import PANORAMAS, REPO, ROOT, TYPES

SEED = 20261020
PER_TYPE = 60
OUT = ROOT / "data_prep" / "walkerbench_extra300"


def _pano_ids() -> set[str]:
    return {p.stem.rsplit("_z2", 1)[0] for p in PANORAMAS.iterdir()}


def _panos_needed(task: dict) -> list[str]:
    import ast

    need = [task.get("spawn_pano_id") or task.get("spawn_point")]
    vp = task.get("visual_path")
    if vp:
        vp = ast.literal_eval(vp) if isinstance(vp, str) else vp
        need += [s["pano_id"] for s in vp]
    return need


def main() -> None:
    have = _pano_ids()
    std = sorted((REPO / "tasks" / "standard").glob("*.json"))
    std_names = {p.name for p in std}
    std_hash = {hashlib.sha256(p.read_bytes()).hexdigest() for p in std}
    std_ids = {(p.name.split("_", 1)[0], json.loads(p.read_text())["task_id"]) for p in std}
    cands, dropped = defaultdict(list), defaultdict(lambda: defaultdict(int))
    for p in sorted((REPO / "tasks" / "full").glob("*.json")):
        kind = p.name.split("_", 1)[0]
        if p.name in std_names or hashlib.sha256(p.read_bytes()).hexdigest() in std_hash:
            dropped[kind]["in_standard"] += 1
            continue
        if (kind, json.loads(p.read_text())["task_id"]) in std_ids:  # same task, another file
            dropped[kind]["same_task_id_as_standard"] += 1
            continue
        if not all(x in have for x in _panos_needed(json.loads(p.read_text()))):
            dropped[kind]["panorama_missing"] += 1
            continue
        cands[kind].append(p)
    rng = random.Random(SEED)
    picked = {t: sorted(rng.sample(cands[t], PER_TYPE), key=lambda p: p.name) for t in TYPES}
    if (OUT / "tasks").exists():
        shutil.rmtree(OUT / "tasks")
    (OUT / "tasks").mkdir(parents=True)
    for files in picked.values():
        for p in files:
            shutil.copy2(p, OUT / "tasks" / p.name)
    sel = {"seed": SEED, "per_type": PER_TYPE, "source": "tasks/full of WalkerBench@b197f51",
           "candidates": {t: len(cands[t]) for t in TYPES}, "dropped": {t: dict(dropped[t]) for t in TYPES},
           "files": {t: [p.name for p in picked[t]] for t in TYPES}}
    (OUT / "selection.json").write_text(json.dumps(sel, indent=1) + "\n")
    print(json.dumps({k: sel[k] for k in ("candidates", "dropped")}, indent=1))
    print(f"{sum(len(v) for v in picked.values())} tasks -> {OUT / 'tasks'}")


if __name__ == "__main__":
    main()
