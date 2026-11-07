"""Reward line, phase 3, experiment C (the user's 2026-11-07: decision 3): the source folder for the und-mix pool of C0
e8, the questions C0's SFT never trained on (C0 trained on trajectories only).

* P1_train: the transition-question training rows (78,786, the P1 line's p1_train.jsonl) in the matrix SFT's row form
  (data_gen.sft_mixture.build.load_p1 on matrix-v3: messages, images, videos [], pool, source_id = sample_id, scene,
  stratum = qtype, meta = {qtype, qclass, family, task_id, questions: 1} as JSON), so vagen.und_mix prepare finds their
  ids, tasks, scenes and images;
* P0_train: the matrix D group's P0_train rows as they are, minus the rows with a video (the 32-frame videos would go
  through reward-design's Qwen2-VL video layout, PLAN_P3_20261106 decision 3);
* dataset_info.json: the two datasets as matrix D's dataset_info.json declares them;
* train.yaml: C0's own SFT recipe (template, cutoff, no pixel limits), which the pool's LLaMA-Factory path reads.

    python -m reward.build_und_src --p1 P1_TRAIN.jsonl --p0 D_P0_TRAIN.jsonl --dataset-info D_DATASET_INFO.json \
        --recipe C0_TRAIN.yaml --out SRC_DIR

Refuses an OUT that exists.  Writes OUT/source.json (inputs, sha256, counts).  No default paths.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any, Dict, Optional, Sequence


def sha(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def p1_row(r: Dict[str, Any]) -> Dict[str, Any]:
    meta = {"qtype": r["qtype"], "qclass": r["qclass"], "family": r["family"], "task_id": r["task_id"], "questions": 1}
    return {"messages": r["messages"], "images": list(r.get("images") or []), "videos": list(r.get("videos") or []),
            "pool": "P1", "source_id": str(r["sample_id"]), "scene": str(r["scene"]), "stratum": str(r["qtype"]),
            "meta": json.dumps(meta, ensure_ascii=False, sort_keys=True)}


def build(p1: Path, p0: Path, dataset_info: Path, recipe: Path, out: Path) -> Dict[str, Any]:
    if out.exists():
        raise SystemExit(f"{out} exists: refusing to overwrite")
    info = json.loads(dataset_info.read_text())
    for name in ("P0_train", "P1_train"):
        if name not in info:
            raise SystemExit(f"{dataset_info} has no {name}")
    out.mkdir(parents=True)
    n1 = n0 = v0 = 0
    with open(p1) as f, open(out / "P1_train.jsonl", "w", encoding="utf-8", newline="\n") as g:
        for line in f:
            if line.strip():
                g.write(json.dumps(p1_row(json.loads(line)), ensure_ascii=False) + "\n")
                n1 += 1
    with open(p0) as f, open(out / "P0_train.jsonl", "w", encoding="utf-8", newline="\n") as g:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("videos"):
                v0 += 1
                continue
            g.write(json.dumps(r, ensure_ascii=False) + "\n")
            n0 += 1
    keep = {k: dict(info[k], file_name=f"{k}.jsonl") for k in ("P0_train", "P1_train")}
    (out / "dataset_info.json").write_text(json.dumps(keep, indent=1, ensure_ascii=False) + "\n")
    shutil.copyfile(recipe, out / "train.yaml")
    src = {"p1": {"file": str(p1), "sha256": sha(p1), "rows": n1},
           "p0": {"file": str(p0), "sha256": sha(p0), "rows_kept": n0, "video_rows_dropped": v0},
           "dataset_info": str(dataset_info), "recipe": {"file": str(recipe), "sha256": sha(recipe)},
           "out": {k: sha(out / k) for k in ("P0_train.jsonl", "P1_train.jsonl", "dataset_info.json", "train.yaml")}}
    (out / "source.json").write_text(json.dumps(src, indent=1) + "\n")
    return src


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    for k in ("p1", "p0", "dataset-info", "recipe", "out"):
        ap.add_argument(f"--{k}", required=True)
    a = ap.parse_args(argv)
    print(json.dumps(build(Path(a.p1), Path(a.p0), Path(a.dataset_info), Path(a.recipe), Path(a.out)), indent=1))


if __name__ == "__main__":
    main()
