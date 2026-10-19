"""Redraw the BLINK-style marks (points, boxes) of questions already generated, with the current drawing code.

    python -m data_gen.interiorgs_qa.redraw_marks --run RUN [--run RUN ...] --backup DIR [--dry-run]

2026-10-19: the user found the letters of marked images unreadable at 256. Two causes: the SCO workers have
no system DejaVu font, so the labels were drawn in PIL's 10 px default font even at 512 (``draw._font`` now
takes matplotlib's copy from the shared env), and the 256 copy was the 512 image with its marks shrunk
(``ViewBank.save_marked`` now resizes first and marks afterwards). This script redoes the marks of questions
generated before. Each marked image is rebuilt from the clean render saved next to
it (``views/<scene>/<view>.png``) and the marks recorded in the question (review fields ``points`` for BL-02,
``drawn_boxes`` for BL-05, ``ref`` and ``candidates`` for BL-06), at 512 and at 256. Old files are copied to
``--backup`` (same relative paths under the run's name) before they are replaced; a 512 copy that comes out
pixel-identical is left untouched. Prints one line per question.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List

import numpy as np

from .draw import draw_boxes, draw_points
from .single import BOX_COLORS

MARKED = ("BL-02", "BL-05", "BL-06")


def drawers(q: Dict[str, Any]) -> List[Callable[[Any, float], None]]:
    """One ``draw(img, k)`` per marked image of the question, as the generator drew them."""
    st, rv = q["subtasks"][0], q["review"]
    if st == "BL-02":
        pts = [tuple(p["xy"]) for p in rv["points"]]
        labs = [p["label"] for p in rv["points"]]
        return [lambda im, k: draw_points(im, pts, labs, k=k)]
    if st == "BL-05":
        labs = list(rv["drawn_boxes"])
        boxes = [rv["drawn_boxes"][lab] for lab in labs]
        return [lambda im, k: draw_boxes(im, boxes, labs, BOX_COLORS, k=k)]
    if st == "BL-06":
        ref = tuple(rv["ref"])
        labs = list(rv["candidates"])
        cands = [tuple(rv["candidates"][lab]) for lab in labs]
        return [lambda im, k: draw_points(im, [ref], ["REF"], k=k), lambda im, k: draw_points(im, cands, labs, k=k)]
    raise ValueError(st)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", required=True)
    ap.add_argument("--backup", required=True, help="old marked images are copied here first")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    from PIL import Image

    n_q = n_256 = n_512 = 0
    for run in map(Path, a.run):
        with open(run / "questions.jsonl") as f:
            qs = [json.loads(line) for line in f if line.strip()]
        for q in qs:
            if q["subtasks"][0] not in MARKED:
                continue
            rels = q["extra"]["images"]
            fns = drawers(q)
            if len(fns) != len(rels):
                raise SystemExit(f"{q['qid']}: {len(rels)} images, {len(fns)} mark sets")
            same512 = []
            for rel, fn in zip(rels, fns):
                src = run / rel
                src256 = src.with_name(src.stem + "_256.png")
                base = Image.open(src.with_name(src.name.split("__")[0] + ".png")).convert("RGB")
                big, small = base.copy(), base.resize((256, 256), Image.BILINEAR)
                fn(big, 1.0)
                fn(small, 256 / base.size[0])
                same = bool(np.array_equal(np.asarray(Image.open(src).convert("RGB")), np.asarray(big)))
                same512.append(same)
                if a.dry_run:
                    continue
                for p in ([src256] if same else [src, src256]):
                    dst = Path(a.backup) / run.name / p.relative_to(run)
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    if not dst.exists():
                        shutil.copy2(p, dst)
                small.save(src256)
                n_256 += 1
                if not same:
                    big.save(src)
                    n_512 += 1
            n_q += 1
            print(json.dumps({"run": run.name, "qid": q["qid"], "group": q["group"], "subtask": q["subtasks"][0],
                              "images": len(rels), "512_unchanged": same512}))
    print(json.dumps({"questions": n_q, "rewritten_256": n_256, "rewritten_512": n_512, "dry_run": a.dry_run}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
