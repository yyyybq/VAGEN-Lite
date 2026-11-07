"""G-oracle of matrix v3 (the user's instruction of 2026-11-07, item 6c): D's action trajectories P2q with every
turn's frame replaced by the frame with the task's targets' truth boxes drawn in (pool P2m).  The boxes are the RL
env's oracle marker (vagen/envs/active_spatial/oracle_marker.py: pointing.truth on the frame's pose, only the targets
clearly visible, P0's draw_boxes in the fixed style), so SFT, RL (rl-config --oracle-marker) and the oracle-mode
evaluation show the same marks.  The marks are a training input drawn on the image; no text changes (the rows keep
D's action_names text byte for byte).

    python -m data_gen.sft_mixture.marker_convert --rows ROWS --frames FRAMES --in P2q_train.jsonl \\
        --out-dir OUT --split train [--gs-root GS] [--workers 4]

writes OUT/P2m_<split>.jsonl (the input's rows and order, ``images`` pointing at OUT/frames/<frame_id>.png) and
OUT/P2m_<split>_stats.json."""

from __future__ import annotations

import argparse
import collections
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from data_gen.sft_mixture.targets_convert import ConvertError, frame_id, parts, read_jsonl

_INPUTS: Dict[str, Any] = {}


def convert_row(row: Dict[str, Any], item: Dict[str, Any], occ: Any, out_frames: Path) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """(the row with its images replaced by the marked copies, statistics); a marked copy already there is kept."""
    from PIL import Image

    from vagen.envs.active_spatial import oracle_marker as OM

    key, msgs, images, _ = parts(row)
    users = [m for m in msgs if m["role"] == "user"]
    if len(users) != len(images):
        raise ConvertError(f"{row.get('source_id')}: {len(users)} user turns, {len(images)} images")
    new_images, st = [], collections.Counter()
    for img in images:
        fid = frame_id(img)
        if fid not in _INPUTS["poses"]:
            raise ConvertError(f"{row.get('source_id')}: no pose for frame {fid}")
        marks = OM.boxes(item, _INPUTS["poses"][fid], occ)
        dst = out_frames / f"{fid}.png"
        if not dst.exists():
            tmp = dst.with_name(dst.name + ".tmp.png")
            OM.draw(Image.open(img), marks).save(tmp)
            tmp.replace(dst)
        new_images.append(str(dst))
        st["frames"] += 1
        st["boxes"] += len(marks)
        st["frames_no_box"] += int(not marks)
        for letter, _ in marks:
            st[f"boxes_{letter}"] += 1
    images_key = "images" if "images" in row else "conversation_image_paths"
    return dict(row, **{images_key: new_images}), dict(st)


def _scene_job(args: Tuple[str, List[int]]):
    from vagen.envs.active_spatial import pointing as PT

    scene, idx = args
    t0 = time.time()
    items = [_INPUTS["rows"][parts(_INPUTS["in"][i])[3]] for i in idx]
    occ = PT.scene_occluders(_INPUTS["gs_root"], scene, _INPUTS["wall_model"], PT.scene_structure_sign(items))
    done, tot = [], collections.Counter()
    for i in idx:
        row = _INPUTS["in"][i]
        conv, st = convert_row(row, _INPUTS["rows"][parts(row)[3]], occ, _INPUTS["out_frames"])
        done.append((i, conv))
        tot.update(st)
    return scene, done, dict(tot), round(time.time() - t0, 1)


def main(argv: Sequence[str] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--rows", required=True, help="the task rows (rows.jsonl of the SFT folder)")
    ap.add_argument("--frames", required=True, help="frames.jsonl of the SFT folder (frame_id -> c2w)")
    ap.add_argument("--in", dest="inp", required=True, help="P2q rows (sft_mixture build output)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--gs-root", default="/mnt/umm/users/yinbaiqiao/InteriorGS_ply")
    ap.add_argument("--wall-model", default="thick")
    ap.add_argument("--workers", type=int, default=1)
    args = ap.parse_args(argv)
    out = Path(args.out_dir)
    (out / "frames").mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(Path(args.inp))
    _INPUTS.update(rows={r["task_id"]: r for r in read_jsonl(Path(args.rows))},
                   poses={f["frame_id"]: f["c2w"] for f in read_jsonl(Path(args.frames))},
                   gs_root=str(args.gs_root), wall_model=args.wall_model, in_=None, out_frames=out / "frames")
    _INPUTS["in"] = rows
    missing = sorted({parts(r)[3] for r in rows if parts(r)[3] not in _INPUTS["rows"]})
    if missing:
        raise SystemExit(f"{len(missing)} tasks not in --rows, e.g. {missing[0]}")
    by_scene: Dict[str, List[int]] = collections.defaultdict(list)
    for i, r in enumerate(rows):
        by_scene[_INPUTS["rows"][parts(r)[3]]["scene_id"]].append(i)
    jobs = sorted(by_scene.items())
    converted: Dict[int, Dict[str, Any]] = {}
    tot: collections.Counter = collections.Counter()
    t0 = time.time()
    if args.workers > 1:
        import multiprocessing as mp

        with mp.get_context("fork").Pool(args.workers) as pool:
            for k, (scene, done, st, sec) in enumerate(pool.imap_unordered(_scene_job, jobs, chunksize=1), 1):
                converted.update(done)
                tot.update(st)
                print(f"[{k}/{len(jobs)} {time.time() - t0:.0f}s] {scene}: {len(done)} rows, {sec} s", flush=True)
    else:
        for k, job in enumerate(jobs, 1):
            scene, done, st, sec = _scene_job(job)
            converted.update(done)
            tot.update(st)
    name = f"P2m_{args.split}.jsonl"
    with open(out / name, "w", encoding="utf-8", newline="\n") as fh:
        for i, r in enumerate(rows):  # the input's order; the text unchanged
            assert parts(converted[i])[1] == parts(r)[1]
            fh.write(json.dumps(converted[i], ensure_ascii=False) + "\n")
    stats = {"input": str(args.inp), "output": name, "rows": len(rows), "scenes": len(jobs), **dict(tot),
             "frames_no_box_share": round(tot["frames_no_box"] / max(1, tot["frames"]), 4),
             "seconds": round(time.time() - t0, 1)}
    (out / f"P2m_{args.split}_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
