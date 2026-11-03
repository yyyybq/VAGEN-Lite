"""Reward line, phase 2 direction 3 (PLAN_DIR3_20261103.md section 2): the checks of the statement truth
(vagen/statement_truth.py) before any SFT data is built.  CPU only.

eval: an evaluation result with recorded trajectories (results.jsonl: per turn the failed conditions after the turn,
      "start" the start pose's) is replayed without rendering in the evaluation's env (as reward/fail_end_steps.py),
      every episode; at the start pose and after every turn the truth is computed at the env's pose and must agree
      with the record: "missing" equal to the recorded failed conditions' groups, "face" equal to the recorded
      *_clearly_visible conditions, and the turn's task-met flag.

sft:  the C0 SFT data (sft_trajectories.jsonl, rows.jsonl, frames.jsonl: every frame's c2w by its image name):
      the truth at every user turn's frame (what the model saw before answering) and at the last frame of each
      trajectory, which the generator judged a success (release.build_sft_record), so "missing" must be none there;
      per frame the footprint distance to object a (the "dist" edges: the terciles over the families without a
      requested distance), the seconds per frame, and whether the task names object a before b (the statement's A, B
      order) by the objects' referring expressions.

    python -m reward.statement_check eval --result DIR --out FILE.json [--tasks FILE] [--env-template YAML]
        [--gs-root DIR] [--workers 4]
    python -m reward.statement_check sft --sft DIR --out FILE.json [--gs-root DIR] [--limit N] [--workers 4]

FILE.json (summary and every disagreement) and, for sft, FILE.frames.jsonl.gz (one line per frame).
"""

import argparse
import gzip
import json
import time
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

TEMPLATE = Path(__file__).resolve().parents[1] / "data_gen" / "interiorgs_v10" / "configs" / "rl_env_v10_S0.yaml"
GS_ROOT = "/mnt/umm/users/yinbaiqiao/InteriorGS_ply"
PLACEHOLDER_EDGES = (1.0, 2.0)  # the sft check records metres; its non-absolute dist labels are not used


def recorded(failed):
    """(missing groups, {letter: clearly visible}) of a recorded list of failed conditions."""
    from vagen.statement_truth import missing_groups

    hidden = {g[len("target_"):-len("_clearly_visible")].upper() for g in failed
              if g.startswith("target_") and g.endswith("_clearly_visible")}
    return missing_groups(failed), hidden


def eval_worker(args):
    tasks_path, template, gs_root, model_json, items = args
    import yaml

    from vagen.envs.active_spatial.env import ActiveSpatialEnv
    from vagen.envs.active_spatial.env_config import ActiveSpatialEnvConfig
    from vagen.statement_truth import truth

    model = json.loads(Path(model_json).read_text())
    cfg = dict(yaml.safe_load(Path(template).read_text())["envs"][0]["config"])
    cfg.update(jsonl_path=str(tasks_path), gs_root=gs_root, render_backend=None, r1_legacy_success=False)
    if model.get("prompt_format"):
        cfg["prompt_format"] = model["prompt_format"]
    cfg.update(model.get("env_switches") or {})
    env = ActiveSpatialEnv(ActiveSpatialEnvConfig(**{k: v for k, v in cfg.items()
                                                     if k in ActiveSpatialEnvConfig.__dataclass_fields__}))

    def fake_render(**_kwargs):
        env.last_image_std = 30.0
        env.last_low_info_frame = False
        env.consecutive_low_info_frame_count = 0
        return {"obs_str": "", "multi_modal_data": {}}

    env._render = fake_render
    position = {}
    for i, line in enumerate(l for l in Path(tasks_path).read_text().splitlines() if l.strip()):
        position.setdefault(json.loads(line)["task_id"], i)

    def compare(rec_failed, met):
        m = env._calculate_canonical_metric()
        t = truth(env.current_item, env.view_engine.get_pose(), env._visibility_occluders(), PLACEHOLDER_EDGES,
                  metric=m)
        want_missing, hidden = recorded(rec_failed)
        got_hidden = {k for k, v in t["face"].items() if not v}
        bad = []
        if t["missing"] != want_missing:
            bad.append({"field": "missing", "got": t["missing"], "want": want_missing})
        if got_hidden != hidden:
            bad.append({"field": "face", "got": sorted(got_hidden), "want": sorted(hidden)})
        if met is not None and t["success"] != bool(met):
            bad.append({"field": "met", "got": t["success"], "want": bool(met)})
        return t, bad

    out = []
    for r in items:
        try:
            env.reset(seed=position[r["task_id"]])
            assert env.current_item["task_id"] == r["task_id"]
            frames, bad = 0, []
            t, b = compare((r.get("start") or {}).get("failed_gates") or [], None)
            frames += 1
            bad += [dict(x, turn="start") for x in b]
            for i, step in enumerate(r["trajectory"]):
                _, _, done, _ = env.step(step["answer"])
                t, b = compare(step.get("failed_gates") or [], step.get("met"))
                frames += 1
                bad += [dict(x, turn=i) for x in b]
                if done:
                    break
            out.append({"task_id": r["task_id"], "task_type": r["task_type"], "frames": frames, "bad": bad})
        except Exception as exc:
            out.append({"task_id": r["task_id"], "task_type": r.get("task_type"), "error": f"{type(exc).__name__}: {exc}"})
    return out


def run_eval(args):
    result = Path(args.result)
    tasks = Path(args.tasks) if args.tasks else result.parent / "tasks.jsonl"
    rows = [json.loads(l) for l in (result / "results.jsonl").read_text().splitlines() if l.strip()]
    rows = [r for r in rows if "error" not in r and r.get("trajectory") is not None]
    chunks = [(str(tasks), args.env_template, args.gs_root, str(result / "model.json"), rows[k:k + 10])
              for k in range(0, len(rows), 10)]
    res = []
    with Pool(args.workers) as pool:
        for part in pool.imap_unordered(eval_worker, chunks):
            res += part
    ok = [x for x in res if "error" not in x]
    by_field = Counter(b["field"] for x in ok for b in x["bad"])
    summary = {"result": str(result), "episodes": len(res), "errors": len(res) - len(ok),
               "frames": sum(x["frames"] for x in ok), "frames_disagreeing": sum(len(x["bad"]) for x in ok),
               "episodes_disagreeing": sum(bool(x["bad"]) for x in ok), "by_field": dict(by_field),
               "by_family": {k: sum(x["frames"] for x in ok if x["task_type"] == k)
                             for k in sorted({x["task_type"] for x in ok})}}
    Path(args.out).write_text(json.dumps({"summary": summary, "errors": [x for x in res if "error" in x],
                                          "disagreements": [x for x in ok if x["bad"]][:200]}, indent=1))
    print(json.dumps(summary, indent=1))


def frame_id(path):
    return Path(path).stem


def sft_worker(args):
    gs_root, scene_id, jobs = args
    from vagen.envs.active_spatial.collision_detector import CollisionDetector
    from vagen.envs.active_spatial.target_visibility import OccluderSet
    from vagen.statement_truth import truth

    det = CollisionDetector()
    det.load_scene_from_gs_root(gs_root, scene_id)
    occ = OccluderSet.from_detector(det)
    out = []
    for j in jobs:
        try:
            frames = []
            for k, (fid, c2w) in enumerate(j["frames"]):
                t0 = time.perf_counter()
                t = truth(j["row"], c2w, occ, PLACEHOLDER_EDGES)
                frames.append({"frame_id": fid, "turn": k, "last": k == len(j["frames"]) - 1 and j["has_last"],
                               "where": t["where"], "face": t["face"], "dist_raw": t["dist"], "dist_m": t["dist_m"],
                               "missing": t["missing"], "success": t["success"],
                               "seconds": round(time.perf_counter() - t0, 4)})
            out.append({"id": j["id"], "task_id": j["row"]["task_id"], "task_type": j["row"]["task_type"],
                        "order": j["order"], "frames": frames})
        except Exception as exc:
            out.append({"id": j["id"], "task_id": j["row"]["task_id"], "task_type": j["row"]["task_type"],
                        "error": f"{type(exc).__name__}: {exc}"})
    return out


def mention_order(row):
    """'ab' when the task names object a before b (by the objects' referring expressions), 'ba', or 'unknown'."""
    objs = (row.get("target_object") or {}).get("objects") or []
    if len(objs) < 2:
        return "single"
    text = str(row.get("task_description") or "").lower()
    pos = []
    for o in objs[:2]:
        ref = str(o.get("ref") or "").lower().strip()
        pos.append(text.find(ref) if ref else -1)
    if min(pos) < 0:
        return "unknown"
    return "ab" if pos[0] < pos[1] else ("ba" if pos[1] < pos[0] else "unknown")


def run_sft(args):
    sft = Path(args.sft)
    rows = {}
    for line in (sft / "rows.jsonl").read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            rows[r["task_id"]] = r
    c2w = {}
    with open(sft / "frames.jsonl") as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                c2w[r["frame_id"]] = r["c2w"]
    by_scene = defaultdict(list)
    n = missing_frames = 0
    with open(sft / "sft_trajectories.jsonl") as f:
        for line in f:
            if not line.strip():
                continue
            if args.limit and n >= args.limit:
                break
            n += 1
            rec = json.loads(line)
            row = rows[rec["source_task_id"]]
            seen = [frame_id(m["image_path"]) for m in rec["conversations"] if m.get("role") == "user"
                    and m.get("image_path")]
            last = frame_id(rec["image_paths"][-1])
            ids = seen + ([last] if last not in seen else [])
            missing_frames += sum(i not in c2w for i in ids)
            by_scene[row["scene_id"]].append({"id": rec["id"], "row": row, "order": mention_order(row),
                                              "has_last": last not in seen,
                                              "frames": [(i, c2w[i]) for i in ids if i in c2w]})
    jobs = [(args.gs_root, s, js[k:k + 50]) for s, js in sorted(by_scene.items()) for k in range(0, len(js), 50)]
    res = []
    t0 = time.time()
    with Pool(args.workers) as pool:
        for part in pool.imap_unordered(sft_worker, jobs):
            res += part
    wall = time.time() - t0
    ok = [x for x in res if "error" not in x]
    seen = [fr for x in ok for fr in x["frames"] if not fr["last"]]
    lasts = [fr for x in ok for fr in x["frames"] if fr["last"]]
    other = [fr["dist_m"] for x in ok if x["task_type"] != "absolute_positioning" for fr in x["frames"] if not fr["last"]]
    edges = [round(float(q), 3) for q in np.quantile(other, [1 / 3, 2 / 3])] if other else None
    with gzip.open(Path(args.out).with_suffix(".frames.jsonl.gz"), "wt") as f:
        for x in ok:
            for fr in x["frames"]:
                f.write(json.dumps({"id": x["id"], "task_type": x["task_type"], **fr}) + "\n")
    where = Counter(v for fr in seen for v in fr["where"].values())
    face = Counter(v for fr in seen for v in fr["face"].values())
    missing = Counter(",".join(fr["missing"]) or "none" for fr in seen)
    summary = {"conversations": n, "errors": len(res) - len(ok), "frames_without_pose": missing_frames,
               "turn_frames": len(seen), "last_frames": len(lasts),
               "last_frames_success": sum(fr["success"] for fr in lasts),
               "turn_frames_success": sum(fr["success"] for fr in seen),
               "seconds_per_frame_mean": round(float(np.mean([fr["seconds"] for x in ok for fr in x["frames"]])), 4),
               "wall_seconds": round(wall, 1), "workers": args.workers,
               "dist_edges_m": edges, "where": dict(where), "face_yes": dict(face), "missing": dict(missing.most_common()),
               "mention_order": dict(Counter(x["order"] for x in ok)),
               "mention_order_by_family": {k: dict(Counter(x["order"] for x in ok if x["task_type"] == k))
                                           for k in sorted({x["task_type"] for x in ok})}}
    bad_last = [{"id": x["id"], "task_type": x["task_type"], "missing": fr["missing"]} for x in ok for fr in x["frames"]
                if fr["last"] and not fr["success"]]
    Path(args.out).write_text(json.dumps({"summary": summary, "errors": [x for x in res if "error" in x][:200],
                                          "last_not_success": bad_last[:500]}, indent=1))
    print(json.dumps(summary, indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval")
    e.add_argument("--result", required=True)
    e.add_argument("--tasks", default=None)
    e.add_argument("--env-template", default=str(TEMPLATE))
    e.add_argument("--gs-root", default=GS_ROOT)
    e.add_argument("--out", required=True)
    e.add_argument("--workers", type=int, default=4)  # the login node: at most 4 cores a task
    s = sub.add_parser("sft")
    s.add_argument("--sft", required=True)
    s.add_argument("--gs-root", default=GS_ROOT)
    s.add_argument("--out", required=True)
    s.add_argument("--limit", type=int, default=0)
    s.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    run_eval(args) if args.cmd == "eval" else run_sft(args)


if __name__ == "__main__":
    main()
