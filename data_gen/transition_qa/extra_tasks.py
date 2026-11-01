"""Shortest-path trajectories of extra tasks that have no SFT conversation: the 15 occlusion tasks the round-12
v2 validation set added (reserve tasks; user, 2026-10-19: render their frames when the probe set lacks occlusion
questions, no need to ask).

    python -m data_gen.transition_qa.extra_tasks frames --tasks T.jsonl --ids-from selection.json --sft-dir SFT --out DIR
    python data_gen/interiorgs_v10/pilot_sco.py prepare --pilot DIR --gs-root PLY      # then submit
    python -m data_gen.transition_qa.extra_tasks trajectories --dir DIR
    python -m data_gen.transition_qa.generate raw --sft-dir DIR --scenes-root META --out OUT --tag oa15

``frames``: DIR/rows.jsonl (the task rows), DIR/frames.jsonl (every turn start and the end of the certificate, cut
into turns as the SFT export cuts it: ``training_prep.sft_frames``), DIR/split_manifest.json (the SFT one, copied).
``trajectories``: DIR/sft_trajectories.jsonl in the SFT export's shape as far as ``generate raw`` reads it: the
absolute image paths and one assistant turn ``<action>a|a|</action>`` per chunk of the certificate.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from . import generate as GEN

MAX_FRAMES = 2000  # the render line above which the user is asked (task statement, section 0)


def _chunks(row: Dict[str, Any], policy: Dict[str, Any]) -> List[List[str]]:
    from data_gen.interiorgs_v10.admission import certificate, chunk_certificate

    exp = policy["sft"]["export"]
    return [list(c) for c in chunk_certificate(certificate(row), exp["chunking"], exp["max_actions_per_turn"])]


def cmd_frames(args: argparse.Namespace) -> None:
    from data_gen.interiorgs_v10.admission import load_policy
    from data_gen.interiorgs_v10.training_prep import sft_frames

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.ids_from:
        ids = set(json.loads(Path(args.ids_from).read_text())["added_task_ids"])
    else:
        ids = {i for i in args.ids.split(",") if i}
    rows = [r for r in GEN.read_jsonl(Path(args.tasks)) if r["task_id"] in ids]
    missing = ids - {r["task_id"] for r in rows}
    if missing:
        raise SystemExit(f"{len(missing)} task ids not in {args.tasks}, e.g. {sorted(missing)[0]}")
    sft = Path(args.sft_dir)
    manifest = json.loads((sft / "split_manifest.json").read_text())
    split_of = {s: k for k in ("train", "val", "test") for s in manifest[f"{k}_scenes"]}
    for r in rows:
        if split_of.get(r["scene_id"]) != r["split"] or r["split"] != args.split:
            raise SystemExit(f"{r['task_id']}: split {r['split']}, scene {r['scene_id']} is "
                             f"{split_of.get(r['scene_id'])} in the manifest (wanted {args.split})")
    policy = load_policy()
    frames = sft_frames(rows, policy)
    if len(frames) > MAX_FRAMES:
        raise SystemExit(f"{len(frames)} frames, above the {MAX_FRAMES} line: ask the user")
    GEN.write_jsonl(out / "rows.jsonl", [dict(r, extra_source=args.source) for r in rows])
    GEN.write_jsonl(out / "frames.jsonl", frames)
    shutil.copyfile(sft / "split_manifest.json", out / "split_manifest.json")
    print(json.dumps({"tasks": len(rows), "frames": len(frames), "scenes": len({r["scene_id"] for r in rows}),
                      "families": sorted({r["task_type"] for r in rows})}, indent=1))


def cmd_trajectories(args: argparse.Namespace) -> None:
    from data_gen.interiorgs_v10.admission import load_policy
    from data_gen.interiorgs_v10.pilot import safe_name

    d = Path(args.dir).resolve()  # absolute image paths: training and probes run elsewhere
    policy = load_policy()
    rendered = json.loads((d / "frames" / "render_manifest.json").read_text())
    ok = {r["frame_id"] for r in rendered["rendered"] if float(r["std"]) >= 2.0}
    trajs, skipped = [], []
    for row in GEN.read_jsonl(d / "rows.jsonl"):
        chunks = _chunks(row, policy)
        bounds = [0]
        for c in chunks:
            bounds.append(bounds[-1] + len(c))
        fids = [f"{row['task_id']}__s{b}" for b in bounds]
        paths = [d / "frames" / "plain" / safe_name(f) for f in fids]
        if any(f not in ok for f in fids) or not all(p.is_file() for p in paths):
            skipped.append(row["task_id"])
            continue
        conv: List[Dict[str, Any]] = [{"role": "system", "content": "shortest path of an extra task (no SFT conversation)"}]
        for c in chunks:
            conv.append({"role": "user", "content": "<image>"})
            conv.append({"role": "assistant", "content": "<action>" + "|".join(c) + "|</action>"})
        trajs.append({"id": f"extra_{row['task_id']}", "source_task_id": row["task_id"], "scene_id": row["scene_id"],
                      "split": row["split"], "image_paths": [str(p) for p in paths], "conversations": conv,
                      "reserve": False, "extra_source": row.get("extra_source")})
    GEN.write_jsonl(d / "sft_trajectories.jsonl", trajs)
    print(json.dumps({"trajectories": len(trajs), "skipped_not_rendered": skipped,
                      "render_errors": len(rendered.get("errors", []))}, indent=1))


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("frames")
    f.add_argument("--tasks", required=True, help="a tasks.jsonl holding the rows (e.g. eval_val_v2/tasks.jsonl)")
    f.add_argument("--ids-from", default="", help="a selection.json: its added_task_ids")
    f.add_argument("--ids", default="", help="comma-separated task ids (instead of --ids-from)")
    f.add_argument("--sft-dir", required=True, help="the SFT folder whose split_manifest.json the bank uses")
    f.add_argument("--split", default="val")
    f.add_argument("--source", default="eval_val_v2_oa_added", help="recorded on every row (extra_source)")
    f.add_argument("--out", required=True)
    f.set_defaults(func=cmd_frames)
    t = sub.add_parser("trajectories")
    t.add_argument("--dir", required=True)
    t.set_defaults(func=cmd_trajectories)
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
