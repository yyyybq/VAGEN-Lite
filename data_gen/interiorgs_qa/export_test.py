"""Write the evaluation sets benchmark by benchmark, each question under the benchmark's own category.

    python -m data_gen.interiorgs_qa.export_test --selection SEL/val.jsonl --selection SEL/test.jsonl --out OUT

OUT/<split>/<benchmark>.jsonl, one line per question (the split is the selection file's name):

* ``id``, ``scene_id``, ``task`` (COVERAGE.md task ID), ``benchmark``, ``category_field`` and ``category`` (the
  benchmark's own field and value, e.g. VSI-Bench ``question_type`` = ``object_counting``), ``main_category`` (what
  the benchmark averages it into), ``definition`` (``same`` as the benchmark, or ``differs``: the difference is in
  ``ground_truth``);
* ``media`` and ``images``: the image files at 256 and at 512 (for a video, its 32 frames in order);
* ``question``: the benchmark-format prompt exactly as scripts/easi_eval.py sends it for that benchmark
  (``qa.render(q, "orig")``); ``question_heldout``: the held-out wording (never in training);
* ``options``, ``answer`` (the option letter, or the number for numeric VSI tasks), ``answer_text``;
* ``circular``: MMSI-Bench and 3DSRBench are scored with rotated options as well (evaluate.py does it);
* ``ground_truth``: how the answer was computed, and how the definition differs from the benchmark's.

OUT/tasks.json: every benchmark task (tasks.py) with its question count per split; a task with no test question is
listed with 0. evaluate.py scores the same selection files with the benchmarks' rules and reports by these categories.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List

from . import tasks
from .export_lf import image_path, load_selection
from .qa import render

CIRCULAR = ("MMSI-Bench", "3DSRBench")


def item(run: Path, q: Dict[str, Any]) -> Dict[str, Any]:
    task = tasks.check(q)
    t = tasks.record(task)
    orig = q["variants"]["orig"]
    rels = q["extra"]["images"]
    numeric = q["answer_type"] == "numeric"
    return {
        "id": q["qid"], "scene_id": q["scene_id"], "task": task, "benchmark": t["benchmark"],
        "category_field": t["field"], "category": t["category"], "main_category": t["main"],
        "definition": t["definition"], "group": q["group"], "media": q["media"],
        "images": {str(res): [image_path(run, p, res) for p in rels] for res in (256, 512)},
        "question": render(q, "orig"),
        "question_heldout": render(q, "h1") if "h1" in q["variants"] else None,
        "options": orig.get("options") or [],
        "answer": q["answer"] if numeric else orig["answer"],
        "answer_text": orig.get("answer_text", str(q["answer"])),
        "circular": t["benchmark"] in CIRCULAR and bool(orig.get("options")),
        "ground_truth": q.get("gt"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--selection", action="append", required=True, help="sampler output (val.jsonl, test.jsonl)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out)
    counts: Dict[str, Counter] = defaultdict(Counter)
    for sel in a.selection:
        split = Path(sel).stem
        by_b: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for run, q in load_selection(sel):
            it = item(run, q)
            by_b[it["benchmark"]].append(it)
            counts[it["task"]][split] += 1
        d = out / split
        d.mkdir(parents=True, exist_ok=True)
        for b, its in by_b.items():
            its.sort(key=lambda x: (x["category"], x["task"], x["id"]))
            (d / f"{b}.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in its))
        print(json.dumps({"split": split, "by_benchmark": {b: len(v) for b, v in sorted(by_b.items())}}))
    table = {k: dict(tasks.record(k), questions=dict(counts[k])) for k in tasks.TASKS}
    (out / "tasks.json").write_text(json.dumps(table, indent=1, ensure_ascii=False))
    empty = [k for k, v in table.items() if not v["questions"].get("test")]
    print(json.dumps({"tasks": len(table), "tasks_without_test_questions": empty}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
