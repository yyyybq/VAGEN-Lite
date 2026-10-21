"""The depth-map occlusion judge (vagen/envs/active_spatial/occlusion_depth.py) on the bank's occlusion questions:
render, calibrate against the user's checks, compare with the box judge (the user's decision of 2026-10-21).

``run`` renders each frame of DIR/frames.jsonl (occ_depth prepare) at the scoring camera and keeps the depth and
opacity (DIR/maps/<sample>.npz) and the image (DIR/rgb/<sample>.jpg), so ``calibrate`` and ``confusion`` recompute
the judge with other parameters on the CPU.  DIR/results.jsonl has, per frame, the box judge (hidden fraction and
in-frame share, canonical_task_metrics_v2.occlusion_silhouette), the depth judge with its defaults, both objects'
boxes and the camera.

    python -m data_gen.transition_qa.occ_depth prepare --out BANK --sft-dir SFT --extra EXTRA --dir NAME [--ids a,b]
    python -m data_gen.transition_qa.occ_judge job --pilot BANK/NAME --gs-root PLY [--pool zoetrope] [--submit]
    python -m data_gen.transition_qa.occ_judge calibrate --pilot BANK/NAME --verdicts V.json ... [--set k=v ...]
    python -m data_gen.transition_qa.occ_judge confusion --pilot BANK/NAME [--set k=v ...]
"""

from __future__ import annotations

import argparse
import collections
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from . import generate as GEN
from .review import occluder_group
from vagen.envs.active_spatial import occlusion_depth as OD

VERSION = "transition_qa_occ_judge_v1"
LABELS = ("full", "partial", "none")
ZH = {"full": "完全", "partial": "部分", "none": "没有", None: "不出题"}
WHY_ZH = {"too_few_pixels": "像素太少", "hidden_by_other": "被其他东西挡", "none_not_visible": "没挡但看不清",
          "between_bands": "落在两档之间", "": ""}


# ---------------------------------------------------------------------------
# helpers (no GPU)
# ---------------------------------------------------------------------------
def box_of(obj: Dict[str, Any]) -> Dict[str, Any]:
    lo, hi = OD.bounds(obj)
    return {"bbox_min": [round(float(x), 5) for x in lo], "bbox_max": [round(float(x), 5) for x in hi],
            "label": obj.get("label"), "id": obj.get("id", obj.get("object_id"))}


def load_maps(pilot: Path, sample_id: str) -> Tuple[np.ndarray, np.ndarray]:
    z = np.load(pilot / "maps" / f"{sample_id}.npz")
    return z["depth"].astype(np.float32), z["alpha"].astype(np.float32) / 255.0


def rejudge(pilot: Path, row: Dict[str, Any], params: Optional[Dict[str, float]] = None,
            with_classes: bool = False) -> Dict[str, Any]:
    """The depth judge for one results row from its saved maps, with ``params`` over OD.DEFAULTS."""
    depth, alpha = load_maps(pilot, row["sample_id"])
    p = {**OD.DEFAULTS, **(params or {})}
    margin = OD.b_margin_for(row["b"], p["b_margin"])
    cls = OD.classify(depth, alpha, np.asarray(row["K"]), np.asarray(row["c2w"]), row["a"], row["b"], b_margin=margin,
                      near_tol=p["near_tol"], min_alpha=p["min_alpha"])
    m = OD.measure(cls)
    answer, why = OD.label(m, p)
    out = {**m, "label": answer, "no_question": why, "b_margin": margin}
    if with_classes:
        out["classes"] = cls
    return out


def parse_sets(items: Iterable[str]) -> Dict[str, float]:
    out = {}
    for item in items:
        key, value = item.split("=", 1)
        if key not in OD.DEFAULTS:
            raise SystemExit(f"unknown parameter {key!r}; known: {sorted(OD.DEFAULTS)}")
        out[key] = float(value)
    return out


def read_verdicts(paths: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """sample_id -> {verdict, note, tag} from the review pages' exported JSON."""
    out: Dict[str, Dict[str, Any]] = {}
    for p in paths:
        data = json.loads(Path(p).read_text(encoding="utf-8"))
        for v in data["verdicts"]:
            if v.get("verdict"):
                out[v["sample_id"]] = {"verdict": v["verdict"], "note": v.get("note") or "", "tag": v.get("tag") or ""}
    return out


def agrees(verdict: str, bank: str, new: Optional[str]) -> bool:
    """A verdict "right" agrees when the judge keeps the bank's answer; "wrong" when it does not give it (another
    answer or no question)."""
    return (new == bank) if verdict == "right" else (new != bank)


# ---------------------------------------------------------------------------
# job / run
# ---------------------------------------------------------------------------
def cmd_job(args: argparse.Namespace) -> None:
    from data_gen.interiorgs_v10 import job_sco
    from .occ_depth import write_job

    launch = write_job(Path(args.pilot), args.gs_root, args.pool, module="data_gen.transition_qa.occ_judge")
    print(json.dumps({"launch": str(launch)}))
    if args.submit:
        print(job_sco.submit(Path(args.pilot).resolve()))


def cmd_run(args: argparse.Namespace) -> None:  # pragma: no cover - needs a GPU and the 3DGS files
    from PIL import Image

    from data_gen.interiorgs_v10.contract import RENDER_SIZE
    from vagen.envs.active_spatial import canonical_task_metrics_v2 as M2
    from vagen.envs.active_spatial.env import runtime_render_camera_parameters
    from vagen.envs.active_spatial.render.unified_renderer import UnifiedRenderGS

    pilot = Path(args.pilot)
    (pilot / "maps").mkdir(exist_ok=True)
    (pilot / "rgb").mkdir(exist_ok=True)
    tasks = {t["task_id"]: t for t in GEN.read_jsonl(pilot / "tasks.jsonl")}
    frames = GEN.read_jsonl(pilot / "frames.jsonl")
    done = {r["sample_id"] for r in GEN.read_jsonl(pilot / "results.jsonl")} if (pilot / "results.jsonl").exists() else set()
    unified, scene_id = None, None
    t0 = time.time()
    with open(pilot / "results.jsonl", "a") as out:
        for f in frames:
            if f["sample_id"] in done:
                continue
            try:
                if f["scene_id"] != scene_id:
                    scene_id = f["scene_id"]
                    if unified is None:
                        unified = UnifiedRenderGS(render_backend="local", gs_root=args.gs_root, scene_id=scene_id,
                                                  gpu_device=0)
                    unified.set_scene(scene_id)
                renderer = unified._ensure_local()
                item = tasks[f["task_id"]]
                c2w = np.asarray(f["c2w"], float)
                K_native = np.asarray(f["camera_item"]["init_camera"]["intrinsics"], float)
                K, w2c = runtime_render_camera_parameters(f["camera_item"], c2w, K_native, RENDER_SIZE)
                K_s, c2w_s, size = OD.scoring_camera(item, c2w)  # the camera the task is scored with
                if size != tuple(RENDER_SIZE) or not (np.allclose(K, K_s) and np.allclose(np.linalg.inv(w2c), c2w_s, atol=1e-6)):
                    raise RuntimeError("render camera differs from the scoring camera")
                rgb, depth, alpha = renderer.render_depth_from_cam_param(K, w2c, *RENDER_SIZE)
                roles = OD.task_roles(item, c2w)
                if roles is None:
                    raise RuntimeError("occlusion roles not found")
                a, b = roles
                res = OD.judge(depth, alpha, K_s, c2w_s, a, b)
                res["judge_version"] = res.pop("version")
                sil = M2.occlusion_silhouette(M2._Context(item, c2w), a, b)
                np.savez_compressed(pilot / "maps" / f"{f['sample_id']}.npz", depth=depth.astype(np.float16),
                                    alpha=np.clip(np.round(alpha * 255), 0, 255).astype(np.uint8))
                Image.fromarray(rgb).save(pilot / "rgb" / f"{f['sample_id']}.jpg", quality=92)
                row = {"sample_id": f["sample_id"], "task_id": f["task_id"], "scene_id": f["scene_id"],
                       "qtype": f["qtype"], "answer_value": f["answer_value"],
                       "box_hidden": round(float(sil["hidden_fraction"]), 4),
                       "box_in_frame": round(float(sil["in_frame_fraction"]), 4),
                       **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()},
                       "a": box_of(a), "b": box_of(b), "K": np.asarray(K_s).tolist(), "c2w": np.asarray(c2w_s).tolist(),
                       "version": VERSION}
            except Exception as exc:  # keep going; the reports count errors
                row = {"sample_id": f["sample_id"], "scene_id": f["scene_id"], "error": f"{type(exc).__name__}: {exc}"}
            out.write(json.dumps(row) + "\n")
            out.flush()
    (pilot / "run_done.json").write_text(json.dumps({"version": VERSION, "seconds": round(time.time() - t0, 1),
                                                     "frames": len(frames)}))


# ---------------------------------------------------------------------------
# calibrate / confusion
# ---------------------------------------------------------------------------
def results(pilot: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rows = GEN.read_jsonl(pilot / "results.jsonl")
    return [r for r in rows if "error" not in r], [r for r in rows if "error" in r]


def cmd_calibrate(args: argparse.Namespace) -> None:
    """The judge against the user's verdicts: per question, and the agreement of the box judge, the first depth
    check (occ_depth: full needs depth_any >= 0.70, partial >= 0.10) and this judge with ``--set`` parameters."""
    pilot = Path(args.pilot)
    params = parse_sets(args.set)
    verdicts = read_verdicts(args.verdicts)
    ok, errors = results(pilot)
    first = {}
    if args.first_check and Path(args.first_check).exists():
        need = {"full": 0.70, "partial": 0.10}
        for r in GEN.read_jsonl(Path(args.first_check)):
            if "error" not in r:
                first[r["sample_id"]] = r["answer_value"] not in need or r["depth_any"] >= need[r["answer_value"]]
    lines, tally = [], collections.Counter()
    for r in sorted(ok, key=lambda r: (verdicts.get(r["sample_id"], {}).get("verdict", "~"), r["sample_id"])):
        v = verdicts.get(r["sample_id"])
        if v is None:
            continue
        new = rejudge(pilot, r, params)
        bank = r["answer_value"]
        tally["n"] += 1
        tally["box"] += agrees(v["verdict"], bank, bank)
        if r["sample_id"] in first:
            tally["first_n"] += 1
            tally["first"] += agrees(v["verdict"], bank, bank if first[r["sample_id"]] else None)
        a = agrees(v["verdict"], bank, new["label"])
        tally["new"] += a
        lines.append({"sample_id": r["sample_id"], "verdict": v["verdict"], "group": occluder_group(r["b"]["label"] or ""),
                      "b_label": r["b"]["label"], "qtype": r["qtype"], "bank": bank, "box_hidden": r["box_hidden"],
                      "hidden_by_b": round(new["hidden_by_b"], 3), "hidden_by_other": round(new["hidden_by_other"], 3),
                      "visible": round(new["visible"], 3), "pixels": new["pixels"], "new": new["label"],
                      "why": new["no_question"], "agree": a, "note": v["note"][:60]})
    summary = {"params": {**OD.DEFAULTS, **params}, "checked": tally["n"], "box judge agrees": tally["box"],
               "first depth check agrees": f"{tally['first']} / {tally['first_n']}", "depth judge agrees": tally["new"],
               "errors": len(errors)}
    print(json.dumps(summary, ensure_ascii=False))
    for x in lines:
        print(f"{'✓' if x['agree'] else '✗'} {x['verdict']:5s} {x['group'] or 'furniture':9s} {str(x['b_label'])[:22]:22s} "
              f"{x['qtype']:14s} bank {x['bank']:7s} box {x['box_hidden']:.2f} | B {x['hidden_by_b']:.2f} "
              f"other {x['hidden_by_other']:.2f} vis {x['visible']:.2f} px {x['pixels']:5d} -> {ZH[x['new']]:3s} "
              f"{WHY_ZH.get(x['why'], x['why'])} {x['note']}")
    if args.out:
        Path(args.out).write_text(json.dumps({"summary": summary, "questions": lines}, ensure_ascii=False, indent=1),
                                  encoding="utf-8")


def cmd_confusion(args: argparse.Namespace) -> None:
    """Old (the box judge: the bank's answer) against new (the depth judge) over every frame, per question type."""
    pilot = Path(args.pilot)
    params = parse_sets(args.set)
    ok, errors = results(pilot)
    cells: Dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    whys: collections.Counter = collections.Counter()
    labelled = []
    for r in ok:
        new = rejudge(pilot, r, params)
        for q in ("all", r["qtype"]):
            cells[q][(r["answer_value"], new["label"])] += 1
        if new["label"] is None:
            whys[(r["answer_value"], new["no_question"])] += 1
        labelled.append({"sample_id": r["sample_id"], "qtype": r["qtype"], "split": r.get("split"), "old": r["answer_value"],
                         "new": new["label"], "why": new["no_question"], "group": occluder_group(r["b"]["label"] or "") or "furniture",
                         **{k: round(new[k], 4) for k in ("hidden_by_b", "hidden_by_other", "visible")}, "pixels": new["pixels"]})
    cols = ("full", "partial", "none", None)
    report = {"params": {**OD.DEFAULTS, **params}, "frames": len(ok), "errors": len(errors), "tables": {}}
    for q in sorted(cells, key=lambda q: (q != "all", q)):
        table = {old: {ZH[new]: cells[q][(old, new)] for new in cols} for old in LABELS}
        report["tables"][q] = table
        print(f"== {q}")
        print("old \\ new  " + "  ".join(f"{ZH[c]:>5s}" for c in cols) + "   total")
        for old in LABELS:
            n = sum(cells[q][(old, c)] for c in cols)
            print(f"{ZH[old]:8s}  " + "  ".join(f"{cells[q][(old, c)]:5d}" for c in cols) + f"   {n:5d}"
                  + ("   (" + " / ".join(f"{100 * cells[q][(old, c)] / n:.0f}%" for c in cols) + ")" if n else ""))
    report["no_question"] = {f"{old}/{why}": n for (old, why), n in sorted(whys.items())}
    print("no question:", json.dumps(report["no_question"], ensure_ascii=False))
    if args.out:
        Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
        GEN.write_jsonl(Path(args.out).with_suffix(".labels.jsonl"), labelled)


def main(argv: Optional[Sequence[str]] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    j = sub.add_parser("job")
    j.add_argument("--pilot", required=True)
    j.add_argument("--gs-root", required=True)
    j.add_argument("--pool", default="zoetrope")
    j.add_argument("--submit", action="store_true")
    j.set_defaults(func=cmd_job)
    r = sub.add_parser("run")
    r.add_argument("--pilot", required=True)
    r.add_argument("--gs-root", required=True)
    r.set_defaults(func=cmd_run)
    c = sub.add_parser("calibrate")
    c.add_argument("--pilot", required=True)
    c.add_argument("--verdicts", action="append", default=[])
    c.add_argument("--first-check", default="", help="occ_depth results.jsonl of the same questions (the first check)")
    c.add_argument("--set", action="append", default=[], help="k=v over occlusion_depth.DEFAULTS (repeatable)")
    c.add_argument("--out", default="")
    c.set_defaults(func=cmd_calibrate)
    x = sub.add_parser("confusion")
    x.add_argument("--pilot", required=True)
    x.add_argument("--set", action="append", default=[])
    x.add_argument("--out", default="")
    x.set_defaults(func=cmd_confusion)
    args = ap.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
