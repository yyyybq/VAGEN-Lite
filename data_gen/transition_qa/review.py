"""Spot check of the built bank (step 2 acceptance: the user checks 30 random questions).

    python -m data_gen.transition_qa.review sample --sft-dir SFT --out OUT [--n 30]
    python data_gen/interiorgs_v10/pilot_sco.py prepare --pilot OUT/review --gs-root PLY   # then submit
    python -m data_gen.transition_qa.review page --out OUT

``sample`` draws the questions (2 per question type from the training and test files, then random ones),
draws the task's objects on copies of their images (A red, B blue: ``admission.overlay_boxes``,
``pilot.draw_overlay``) and writes OUT/review/frames.jsonl: the view after the action of every forward question,
boxed, so the checker can see what the answer is about.  ``page`` collects everything into review_data.json
(images inline) for the check page.
"""

from __future__ import annotations

import argparse
import base64
import collections
import json
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from . import generate as GEN
from . import geometry as G


def _image_pose(path: str, frames: Dict[str, Dict[str, Any]], interior: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    fid = GEN.frame_id_of(path)
    if fid in frames:
        return frames[fid]
    return interior[fid]


def cmd_sample(args: argparse.Namespace) -> None:
    from PIL import Image

    from data_gen.interiorgs_v10.admission import overlay_boxes
    from data_gen.interiorgs_v10.pilot import draw_overlay

    sft, out = Path(args.sft_dir), Path(args.out)
    d = out / "review"
    (d / "img").mkdir(parents=True, exist_ok=True)
    rows = {r["task_id"]: r for r in GEN.read_jsonl(sft / "rows.jsonl")}
    frames = {f["frame_id"]: f for f in GEN.read_jsonl(sft / "frames.jsonl")}
    interior_path = out / "interior" / "frames.jsonl"
    interior = {f["frame_id"]: f for f in GEN.read_jsonl(interior_path)} if interior_path.exists() else {}
    pool = GEN.read_jsonl(out / "p1_train.jsonl") + GEN.read_jsonl(out / "p1_test.jsonl")
    rng = random.Random(args.seed)
    by_q: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for r in sorted(pool, key=lambda r: r["sample_id"]):
        by_q[r["qtype"]].append(r)
    picks: List[Dict[str, Any]] = []
    for q in sorted(by_q):
        picks += rng.sample(by_q[q], min(2, len(by_q[q])))
    rest = [r for r in pool if r["sample_id"] not in {p["sample_id"] for p in picks}]
    picks += rng.sample(rest, max(0, args.n - len(picks)))
    picks = picks[: args.n]
    rng.shuffle(picks)
    render, sample = [], []
    for i, r in enumerate(picks, 1):
        row = rows[r["task_id"]]
        boxed = []
        for j, path in enumerate(r["images"]):
            f = _image_pose(path, frames, interior)
            img = draw_overlay(Image.open(path), overlay_boxes(row, np.asarray(f["c2w"], float)))
            name = f"q{i:02d}_{j}.png"
            img.save(d / "img" / name)
            boxed.append(name)
        post = None
        if r["qclass"] == "forward":
            f = _image_pose(r["images"][0], frames, interior)
            pose = G.repeated_action_pose(np.asarray(f["c2w"], float), r["action"], int(r["k"]))
            post = f"q{i:02d}_post"
            render.append({"frame_id": post, "kind": "transition_qa_review", "task_id": r["task_id"],
                           "scene_id": r["scene"], "state_id": "post", "c2w": pose.tolist(),
                           "camera_item": f["camera_item"], "overlay": overlay_boxes(row, pose)})
        sample.append({"n": i, "sample_id": r["sample_id"], "qtype": r["qtype"], "qclass": r["qclass"],
                       "family": r["family"], "split": r["split"], "task_id": r["task_id"],
                       "task_description": row["task_description"], "question": r["messages"][0]["content"],
                       "answer": r["messages"][1]["content"], "options": r["options"], "boxed": boxed,
                       "post_frame": post, "originals": r["images"],
                       "objects": {s: e.get("ref") for s, e in zip("AB", row["target_object"]["objects"])}})
    GEN.write_jsonl(d / "sample.jsonl", sample)
    GEN.write_jsonl(d / "frames.jsonl", render)
    print(json.dumps({"questions": len(sample), "post_frames": len(render),
                      "by_qtype": collections.Counter(s["qtype"] for s in sample)}, indent=1))


def cmd_page(args: argparse.Namespace) -> None:
    from data_gen.interiorgs_v10.pilot import safe_name

    d = Path(args.out) / "review"

    def b64(p: Path) -> str:
        return "data:image/png;base64," + base64.b64encode(p.read_bytes()).decode()

    items = []
    for s in GEN.read_jsonl(d / "sample.jsonl"):
        s = dict(s)
        s["images"] = [b64(d / "img" / n) for n in s.pop("boxed")]
        post = s.pop("post_frame")
        s["post_image"] = b64(d / "frames" / "boxes" / safe_name(post)) if post else None
        items.append(s)
    (d / "review_data.json").write_text(json.dumps(items, ensure_ascii=False))
    print(json.dumps({"items": len(items), "bytes": (d / "review_data.json").stat().st_size}))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("sample")
    a.add_argument("--sft-dir", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--n", type=int, default=30)
    a.add_argument("--seed", type=int, default=20261018)
    a.set_defaults(func=cmd_sample)
    b = sub.add_parser("page")
    b.add_argument("--out", required=True)
    b.set_defaults(func=cmd_page)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
