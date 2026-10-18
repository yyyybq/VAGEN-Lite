"""Build SAT-Syn in the same form as lmms-eval's SAT-Real (nv-njb/SAT), so both use one prompt and one scorer.

    python -m eval_external.qa_transfer.make_sat_syn

SAT-Real in lmms-eval is nv-njb/SAT: array/SAT's test split (150 real-image items) with images stored as
Sequence(Image()) and the two options shuffled once, row by row in order, by one random.Random(42).  Applying
that procedure to array/SAT's test split reproduces nv-njb's option order for 150 of 150 items (checked here
every run).  SAT-Syn is the same procedure applied to array/SAT's val split (4,001 synthetic items), written to
ROOT/data/sat_syn_ee/val.parquet and read by lmms_tasks/sat_syn_ee.yaml.
"""

from __future__ import annotations

import glob
import json
import random
from pathlib import Path

import datasets
import pyarrow.parquet as pq

ROOT = Path("/mnt/umm/users/yinbaiqiao/eval_external_assets")
ARRAY = ROOT / "data" / "sat_array"          # array/SAT@bda5dde (assets.yaml sat_syn)
REAL = ROOT / "data" / "sat_real" / "data"   # nv-njb/SAT@85f0ea8 (assets.yaml sat_real)
OUT = ROOT / "data" / "sat_syn_ee"
SEED = 42
FEATURES = datasets.Features({
    "image_bytes": datasets.Sequence(datasets.Image()),
    "question": datasets.Value("string"),
    "answers": datasets.Sequence(datasets.Value("string")),
    "correct_answer_idx": datasets.Value("int32"),
    "correct_answer": datasets.Value("string"),
    "question_type": datasets.Value("string"),
})


def convert(rows: list[dict]) -> list[dict]:
    rng = random.Random(SEED)
    out = []
    for r in rows:
        answers = list(r["answers"])
        rng.shuffle(answers)
        out.append({
            "image_bytes": [{"bytes": b, "path": None} for b in r["image_bytes"]],
            "question": r["question"],
            "answers": answers,
            "correct_answer_idx": answers.index(r["correct_answer"]),
            "correct_answer": r["correct_answer"],
            "question_type": r["question_type"],
        })
    return out


def check_against_real() -> int:
    mine = convert(pq.read_table(ARRAY / "SAT_test.parquet").to_pylist())
    real = pq.read_table(glob.glob(str(REAL / "*.parquet"))[0],
                         columns=["question", "answers", "correct_answer_idx"]).to_pylist()
    same = sum(m["question"] == r["question"] and m["answers"] == list(r["answers"])
               and m["correct_answer_idx"] == r["correct_answer_idx"] for m, r in zip(mine, real))
    if same != len(real):
        raise SystemExit(f"procedure reproduces nv-njb/SAT for only {same} of {len(real)} items")
    return same


def main() -> None:
    same = check_against_real()
    rows = convert(pq.read_table(ARRAY / "SAT_val.parquet").to_pylist())
    OUT.mkdir(parents=True, exist_ok=True)
    ds = datasets.Dataset.from_list(rows, features=FEATURES)
    ds.to_parquet(str(OUT / "val.parquet"))
    counts = {}
    for r in rows:
        counts[r["question_type"]] = counts.get(r["question_type"], 0) + 1
    info = {"source": "array/SAT@bda5dde942d7f7b41bff7935f086ed9a9e348ae3 SAT_val.parquet", "seed": SEED,
            "items": len(rows), "question_types": counts, "check_vs_nv-njb_SAT": f"{same}/{same} identical"}
    (OUT / "build_info.json").write_text(json.dumps(info, indent=1) + "\n")
    print(json.dumps(info))


if __name__ == "__main__":
    main()
