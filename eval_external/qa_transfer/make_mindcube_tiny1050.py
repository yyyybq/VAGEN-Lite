"""Build MindCube-Tiny with MindCube's own 1,050 items: EASI's 1,040-row table plus the 10 rows it lacks.

    python -m eval_external.qa_transfer.make_mindcube_tiny1050

EASI's MindCubeBench_tiny_raw_qa.tsv (lmms-lab-si/EASI-Leaderboard-Data@52bb1dd, the table every public EASI
number uses) has 1,040 of the 1,050 items of MindCube's MindCube_tinybench.jsonl (MLL-Lab/MindCube@9c941b4); the 10
missing ones are among items of scene 94e4f0e8....  This script rebuilds every official item the way EASI built its
rows (image tokens, question, answer letter, image paths under data/, the raw-QA input prompt, category, candidates,
number of options), checks that the rebuild equals EASI's row exactly for all 1,040 items it has, and writes
LMUData/MindCubeBench_tiny1050_raw_qa.tsv: EASI's 1,040 rows unchanged and in order, then the 10 rebuilt rows.
The table's md5 goes to LMUData/MindCubeBench_tiny1050_raw_qa.json, where vlmeval_launch reads it.
"""

from __future__ import annotations

import ast
import glob
import hashlib
import json
import os
from pathlib import Path

import pandas as pd

from eval_external.qa_transfer.benchmarks import MINDCUBE_TINY1050

ROOT = Path("/mnt/umm/users/yinbaiqiao/eval_external_assets")
LMU = ROOT / "LMUData"
EASI_TINY = LMU / "MindCubeBench_tiny_raw_qa.tsv"
EASI_MD5 = "35f69fc30d7c2d2880417ce0769f5347"  # vlmeval/dataset/mindcubebench.py DATASET_MD5
SNAPSHOT = "datasets--MLL-Lab--MindCube/snapshots/9c941b46a6bd65b6914669ef7a579948fc9c8467"
COLUMNS = ["index", "question", "answer", "image_path", "input_prompt", "category", "candidates", "num_options"]


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def candidates(question: str) -> list[str]:
    """'... A. No B. Yes' -> ['A. No', 'B. Yes']: from the last ' A. ', each next letter in turn."""
    start = question.rfind(" A. ")
    if start < 0:
        raise ValueError(f"no options in {question!r}")
    cuts, pos = [start + 1], start + 1
    for letter in "BCDEFGH":
        nxt = question.find(f" {letter}. ", pos)
        if nxt < 0:
            break
        cuts.append(nxt + 1)
        pos = nxt + 1
    cuts.append(len(question))
    return [question[a:b].strip() for a, b in zip(cuts, cuts[1:])]


def rebuild(item: dict, template: str) -> dict:
    tokens = "<image>" * len(item["images"])
    cands = candidates(item["question"])
    return {"index": item["id"], "question": f"{tokens}\n{item['question']}", "answer": item["gt_answer"],
            "image_path": str(["data/" + p for p in item["images"]]),
            "input_prompt": f"{tokens}\n{template}{item['question']}", "category": item["id"].split("_")[0],
            "candidates": str(cands), "num_options": len(cands)}


def main() -> None:
    if md5(EASI_TINY) != EASI_MD5:
        raise SystemExit(f"{EASI_TINY}: md5 differs from EASI's {EASI_MD5}")
    easi = pd.read_csv(EASI_TINY, sep="\t", dtype=str, keep_default_na=False)
    hf = Path(os.environ.get("HF_HOME", ROOT / "hf_home")) / "hub"
    jsonl = glob.glob(str(hf / SNAPSHOT / "data" / "raw" / "MindCube_tinybench.jsonl"))
    if len(jsonl) != 1:
        raise SystemExit(f"MindCube_tinybench.jsonl not found under {hf / SNAPSHOT}")
    official = [json.loads(line) for line in open(jsonl[0])]
    first = easi.iloc[0]
    n_img = len(ast.literal_eval(first["image_path"]))
    q = first["question"].split("\n", 1)[1]
    template = first["input_prompt"][len("<image>" * n_img) + 1:-len(q)]
    by_id = {o["id"]: o for o in official}
    mismatch = []
    for _, row in easi.iterrows():
        mine = rebuild(by_id[row["index"]], template)
        diff = [c for c in COLUMNS if str(mine[c]) != row[c]]
        if diff:
            mismatch.append((row["index"], diff))
    if mismatch:
        raise SystemExit(f"rebuild differs from EASI's rows for {len(mismatch)} items, e.g. {mismatch[:3]}")
    extra = [rebuild(o, template) for o in official if o["id"] not in set(easi["index"])]
    images = [p for r in extra for p in ast.literal_eval(r["image_path"])]
    missing = [p for p in images if not (hf / SNAPSHOT / p).exists()]
    if missing:
        raise SystemExit(f"images missing: {missing}")
    out = pd.concat([easi, pd.DataFrame(extra, columns=COLUMNS).astype(str)], ignore_index=True)
    if len(out) != 1050 or out["index"].nunique() != 1050:
        raise SystemExit(f"expected 1,050 distinct items, got {len(out)} ({out['index'].nunique()} distinct)")
    path = LMU / f"{MINDCUBE_TINY1050}.tsv"
    out.to_csv(path, sep="\t", index=False)
    back = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    if not back.iloc[:len(easi)].equals(easi):
        raise SystemExit("the written table does not read back to EASI's rows")
    info = {"md5": md5(path), "items": len(out), "easi_rows": len(easi), "easi_md5": EASI_MD5,
            "added": [r["index"] for r in extra], "official": str(jsonl[0]),
            "rebuild_equals_easi_rows": len(easi)}
    (LMU / f"{MINDCUBE_TINY1050}.json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
