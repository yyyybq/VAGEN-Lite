"""Group G of matrix v3 (the user's instruction of 2026-11-06, part 4.1): the pointing pool Pg on the SFT trajectories'
frames.  Two kinds of question about X, a target the task text names (target_object.objects, the task text's own
words): "point to X" (answer: a point on it) and "draw a box around X" (answer: its top-left and bottom-right
corners), on the 256 x 256 camera; P1's scene split (train scenes -> pg_train, validation scenes -> pg_test and the
probe).  The truth is pointing.truth's (vagen/envs/active_spatial/pointing.py, the env's own): X clearly visible by
the success metrics' face rule, its box the projected 3D box clipped to the frame.  Judging: a point inside the true
box; a box with IoU >= 0.5 (eval/diagnostics/qa_probe.py judge_pointing).

    python -m data_gen.transition_qa.pointing_qa raw   --sft-dir SFT --gs-root GS --out OUT [--workers 4]
    python -m data_gen.transition_qa.pointing_qa build --out OUT --split-manifest M --probe-scenes-from TASKS
                                                       [--train 32498] [--test 1000] [--probe 60] [--seed 20261106]

raw: per scene, every distinct frame of every trajectory (P1's anchor states, reserve trajectories left out) and
every target clearly visible there: one candidate with its box in pixels (OUT/raw/<scene>.jsonl and .json; finished
scenes are skipped).  build: P1's scene exclusions; the probe (validation scenes of TASKS, half point and half box,
at most one question a task, families in turn), the test set (validation questions of the other tasks) and the
train set (train scenes); at most one question per frame and target, its kind fixed by a hash of frame and target;
both coordinate formats (prompt.TARGET_FORMATS): OUT/<format>/pg_{train,test,probe}.jsonl (LLaMA-Factory rows plus
the truth: judge, box, coords) and OUT/stats.json."""

from __future__ import annotations

import argparse
import collections
import json
import random
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from . import generate as GEN
from .questions import stable_int

KINDS = ("point", "box")
QUESTION = {
    ("point", "pixel"): "Point to {ref}. Answer with the pixel coordinates (x,y) of one point on it in this 256 x 256 "
                        "image, x from the left edge and y from the top edge.",
    ("point", "norm1000"): "Point to {ref}. Answer with the coordinates (x,y) of one point on it, x and y from 0 to "
                           "1000 across the image, x from the left edge and y from the top edge.",
    ("box", "pixel"): "Draw a box around {ref}. Answer with the pixel coordinates (x1,y1,x2,y2) of its top-left and "
                      "bottom-right corners in this 256 x 256 image.",
    ("box", "norm1000"): "Draw a box around {ref}. Answer with the coordinates (x1,y1,x2,y2) of its top-left and "
                         "bottom-right corners, x and y from 0 to 1000 across the image.",
}
_INPUTS: Dict[str, Any] = {}


# =============================================================================
# raw
# =============================================================================
def scene_candidates(scene_id: str, cfg: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    from vagen.envs.active_spatial import pointing as PT

    occ = PT.scene_occluders(cfg["gs_root"], scene_id, cfg["wall_model"])
    out: List[Dict[str, Any]] = []
    funnel: collections.Counter = collections.Counter()
    for traj in _INPUTS["by_scene"][scene_id]:
        item = _INPUTS["rows"][traj["source_task_id"]]
        images, _ = GEN.trajectory_steps(traj, _INPUTS["poses"])
        refs = PT.target_refs(item)
        funnel["trajectories"] += 1
        for pos, role in GEN.pick_anchor_positions(images, item["task_id"], 0):
            im = images[pos]
            truth = PT.truth(item, im["c2w"], occ)
            funnel["states"] += 1
            for t, ref in zip(truth, refs):
                funnel["targets"] += 1
                if not t["in_view"]:
                    funnel["not_clearly_visible"] += 1
                    continue
                out.append({"pool": "Pg", "family": item["task_type"], "task_id": item["task_id"], "scene": scene_id,
                            "split": traj["split"], "state_id": im["state_id"], "state_pos": pos, "state_role": role,
                            "image": im["image"], "slot": t["letter"], "ref": ref, "box": t["box"],
                            "n_slots": len(truth), "n_in_view": sum(x["in_view"] for x in truth)})
    funnel["candidates"] = len(out)
    return out, {"funnel": dict(funnel)}


def _scene_job(args: Tuple[str, Dict[str, Any]]) -> Tuple[str, str]:
    scene_id, cfg = args
    raw = Path(cfg["out"]) / "raw"
    done = raw / f"{scene_id}.json"
    if done.exists():
        return scene_id, "skipped"
    t0 = time.time()
    try:
        recs, info = scene_candidates(scene_id, cfg)
    except Exception:
        (raw / f"{scene_id}.error.txt").write_text(traceback.format_exc())
        return scene_id, "error"
    GEN.write_jsonl(raw / f"{scene_id}.jsonl", recs)
    info.update(scene=scene_id, records=len(recs), seconds=round(time.time() - t0, 1))
    done.write_text(json.dumps(info, indent=1))
    return scene_id, f"{len(recs)} candidates, {info['seconds']} s"


def cmd_raw(args: argparse.Namespace) -> None:
    sft, out = Path(args.sft_dir), Path(args.out)
    (out / "raw").mkdir(parents=True, exist_ok=True)
    cfg = {"gs_root": str(args.gs_root), "wall_model": args.wall_model, "out": str(out)}
    manifest = json.loads((sft / "split_manifest.json").read_text())
    split_of = {s: k for k in ("train", "val", "test") for s in manifest[f"{k}_scenes"]}
    _INPUTS["rows"] = {r["task_id"]: r for r in GEN.read_jsonl(sft / "rows.jsonl")}
    _INPUTS["poses"] = {f["frame_id"]: f["c2w"] for f in GEN.read_jsonl(sft / "frames.jsonl")}
    by_scene: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
    for t in GEN.read_jsonl(sft / "sft_trajectories.jsonl"):
        if t.get("reserve"):
            continue
        if split_of.get(t["scene_id"]) != t["split"] or t["split"] not in ("train", "val"):
            raise SystemExit(f"{t['id']}: split {t['split']} but scene {t['scene_id']} is "
                             f"{split_of.get(t['scene_id'])} in the manifest")
        by_scene[t["scene_id"]].append(t)
    _INPUTS["by_scene"] = by_scene
    scenes = sorted(by_scene)
    if args.scenes:
        scenes = [s for s in args.scenes.split(",") if s]
    if args.limit_scenes:
        scenes = scenes[: args.limit_scenes]
    (out / "config.json").write_text(json.dumps(dict(cfg, sft_dir=str(sft), scenes=len(scenes)), indent=1))
    jobs = [(s, cfg) for s in scenes]
    t0 = time.time()
    if args.workers > 1:
        import multiprocessing as mp

        with mp.get_context("fork").Pool(args.workers) as pool:
            for i, (sid, msg) in enumerate(pool.imap_unordered(_scene_job, jobs, chunksize=1), 1):
                print(f"[{i}/{len(jobs)} {time.time() - t0:.0f}s] {sid}: {msg}", flush=True)
    else:
        for i, job in enumerate(jobs, 1):
            sid, msg = _scene_job(job)
            print(f"[{i}/{len(jobs)} {time.time() - t0:.0f}s] {sid}: {msg}", flush=True)
    errors = sorted(p.name for p in (out / "raw").glob("*.error.txt"))
    if errors:
        raise SystemExit(f"{len(errors)} scenes failed, e.g. {errors[0]}")


# =============================================================================
# build
# =============================================================================
def read_raw(out: Path) -> Tuple[List[Dict[str, Any]], collections.Counter]:
    recs, funnel = [], collections.Counter()
    for info in sorted((out / "raw").glob("*.json")):
        funnel.update(json.loads(info.read_text())["funnel"])
        recs += GEN.read_jsonl(info.with_suffix(".jsonl"))
    return recs, funnel


def kind_of(r: Dict[str, Any]) -> str:
    return KINDS[stable_int(r["task_id"], r["state_id"], r["slot"], "pg_kind") % 2]


def sample_id(r: Dict[str, Any]) -> str:
    return "pg_" + format(stable_int(r["task_id"], r["state_id"], r["slot"], r["kind"]), "012x")


def by_kind(rs: Sequence[Dict[str, Any]], n: int, rng: random.Random) -> List[Dict[str, Any]]:
    """``n`` rows, half of each kind as far as there are (all when fewer), drawn with ``rng``."""
    groups = {k: [r for r in rs if r["kind"] == k] for k in KINDS}
    for g in groups.values():
        rng.shuffle(g)
    want = {k: n // 2 + (n % 2 if k == KINDS[0] else 0) for k in KINDS}
    short = {k: max(0, want[k] - len(g)) for k, g in groups.items()}
    for k in KINDS:  # what one kind lacks, the other gives
        other = KINDS[1 - KINDS.index(k)]
        want[other] += short[k]
    picked = [r for k in KINDS for r in groups[k][: want[k]]]
    return sorted(picked, key=lambda r: r["sample_id"])


def probe_select(rs: Sequence[Dict[str, Any]], n: int, rng: random.Random) -> List[Dict[str, Any]]:
    """``n`` questions, half of each kind, at most one a task, families in turn."""
    picks: List[Dict[str, Any]] = []
    used: set = set()
    for k in KINDS:
        fam: Dict[str, List[Dict[str, Any]]] = collections.defaultdict(list)
        for r in rs:
            if r["kind"] == k:
                fam[r["family"]].append(r)
        for g in fam.values():
            rng.shuffle(g)
        want, order = n // 2 + (n % 2 if k == KINDS[0] else 0), sorted(fam)
        got = 0
        while got < want and any(fam.values()):
            for f in order:
                while fam[f] and fam[f][-1]["task_id"] in used:
                    fam[f].pop()
                if fam[f] and got < want:
                    r = fam[f].pop()
                    used.add(r["task_id"])
                    picks.append(r)
                    got += 1
    return sorted(picks, key=lambda r: r["sample_id"])


def to_lf(rec: Dict[str, Any], fmt: str) -> Dict[str, Any]:
    """The LLaMA-Factory row of a candidate in the coordinates of ``fmt`` (a targets format), with the truth."""
    from vagen.envs.active_spatial import pointing as PT
    from vagen.envs.active_spatial.prompt import TARGET_FORMATS

    coords = TARGET_FORMATS[fmt]
    s = 1000.0 / 256.0 if coords == "norm1000" else 1.0
    x1, y1, x2, y2 = (float(v) * s for v in rec["box"])
    if rec["kind"] == "point":
        x, y = PT.box_point([x1, y1, x2, y2])
        answer = f"({x},{y})"
    else:
        answer = f"({int(round(x1))},{int(round(y1))},{int(round(x2))},{int(round(y2))})"
    question = QUESTION[(rec["kind"], coords)].format(ref=rec["ref"])
    return {"messages": [{"role": "user", "content": "<image>\n" + question}, {"role": "assistant", "content": answer}],
            "images": [rec["image"]], "sample_id": rec["sample_id"], "pool": "Pg", "qclass": "pointing",
            "qtype": f"{rec['kind']}_now", "family": rec["family"], "task_id": rec["task_id"], "scene": rec["scene"],
            "split": rec["split"], "state_id": rec["state_id"], "state_role": rec["state_role"],
            "state_index": int(rec["state_pos"]), "answer": answer, "answer_value": answer, "options": [],
            "judge": rec["kind"], "box": list(rec["box"]), "coords": coords, "render_size": [256, 256],
            "slot": rec["slot"], "target": rec["ref"], "n_in_view": rec["n_in_view"]}


def cmd_build(args: argparse.Namespace) -> None:
    from vagen.envs.active_spatial.prompt import TARGET_FORMATS

    out = Path(args.out)
    recs, funnel = read_raw(out)
    for r in recs:
        r["kind"] = kind_of(r)
        r["sample_id"] = sample_id(r)
    splits = GEN._check_scenes(recs, args.split_manifest)
    ex_path, ex_scenes = GEN.read_scene_exclusions({"exclude_scenes": GEN.SCENE_EXCLUSIONS})
    train, gone_train = GEN.drop_scenes(splits["train"], ex_scenes)
    val, gone_val = GEN.drop_scenes(splits["val"], ex_scenes)
    probe_scenes = {json.loads(l)["scene_id"] for l in Path(args.probe_scenes_from).read_text().splitlines() if l.strip()}
    probe = probe_select([r for r in val if r["scene"] in probe_scenes], args.probe, random.Random(f"{args.seed}:probe"))
    probe_tasks = {r["task_id"] for r in probe}
    test = by_kind([r for r in val if r["task_id"] not in probe_tasks], args.test, random.Random(f"{args.seed}:test"))
    train = by_kind(train, args.train, random.Random(f"{args.seed}:train"))
    files = {"pg_train.jsonl": train, "pg_test.jsonl": test, "pg_probe.jsonl": probe}
    ids = [r["sample_id"] for rs in files.values() for r in rs]
    if len(set(ids)) != len(ids):
        raise SystemExit("sample ids repeat")
    for fmt in TARGET_FORMATS:
        (out / fmt).mkdir(parents=True, exist_ok=True)
        for name, rs in files.items():
            GEN.write_jsonl(out / fmt / name, (to_lf(r, fmt) for r in rs))
    count = lambda rs, key: dict(sorted(collections.Counter(r[key] for r in rs).items()))  # noqa: E731
    stats = {"seed": args.seed, "funnel": dict(funnel), "candidates": {"train": len(splits["train"]), "val": len(splits["val"])},
             "scene_exclusion": {"file": ex_path, "scenes": len(ex_scenes), "dropped_train": len(gone_train),
                                 "dropped_val": len(gone_val)},
             "probe_scenes_from": args.probe_scenes_from, "formats": list(TARGET_FORMATS),
             "files": {name: {"questions": len(rs), "kind": count(rs, "kind"), "family": count(rs, "family"),
                              "slot": count(rs, "slot"), "scenes": len({r["scene"] for r in rs}),
                              "tasks": len({r["task_id"] for r in rs})} for name, rs in files.items()}}
    (out / "stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps({k: v["questions"] for k, v in stats["files"].items()}))


def main(argv: Sequence[str] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("raw")
    r.add_argument("--sft-dir", required=True)
    r.add_argument("--gs-root", default="/mnt/umm/users/yinbaiqiao/InteriorGS_ply")
    r.add_argument("--out", required=True)
    r.add_argument("--wall-model", default="thick")
    r.add_argument("--workers", type=int, default=1)
    r.add_argument("--scenes", default="")
    r.add_argument("--limit-scenes", type=int, default=0)
    b = sub.add_parser("build")
    b.add_argument("--out", required=True)
    b.add_argument("--split-manifest", required=True)
    b.add_argument("--probe-scenes-from", required=True, help="tasks.jsonl of the validation set the probe's scenes come from")
    b.add_argument("--train", type=int, default=32498, help="train questions (P1's train state questions: 32,498)")
    b.add_argument("--test", type=int, default=1000)
    b.add_argument("--probe", type=int, default=60)
    b.add_argument("--seed", type=int, default=20261106)
    args = ap.parse_args(argv)
    {"raw": cmd_raw, "build": cmd_build}[args.cmd](args)


if __name__ == "__main__":
    main()
