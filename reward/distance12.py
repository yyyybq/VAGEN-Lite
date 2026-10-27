"""Reward line, step 2 of task B: depth-12 distance tables under the user's limits of 2026-10-27.

The table of a task is the main line's (data_gen.interiorgs_v10.distance_table, same states, transitions, success
rule, backward search and file format, so the env's TableCache reads it unchanged); this module only adds:

* limits: a task is over the limit when its graph does not reach the depth or holds more than ``max_states`` poses,
  or when judging its prefilter-passing states takes more than ``judge_cpu_hours`` of the process's CPU time;
  over-limit tasks are recorded (no table) and, with ``--backup-ids`` and ``--target``, replaced by backups of the
  same (family, d*) stratum (then the same family's nearest d*, then any), so the set still holds ``target`` tables;
* the strict prefilter (``--prefilter strict``, the user's supplementary rule 3): for families whose success needs
  every task object clearly visible (target_visibility.REQUIRED_VISIBLE_SLOTS names every object of the row: e.g.
  projective relations, FOV inclusion, equidistance), every object's bounding sphere must meet the viewing cone, not
  just one (main line's may_succeed applied object by object and and-ed); other rows keep the main line's rule.  A
  clearly visible face has sample points in front of the camera and inside the image, so the object's sphere meets
  the cone and the rule only skips states that cannot succeed; the 2a check builds both and requires equal tables;
* per task: states per level, the process's peak memory, CPU seconds of each part, the prefilter used.

    python -m reward.distance12 build --rows ROWS --task-ids FILE --out DIR --gs-root DIR
        [--backup-ids FILE (--target N | --fill)] [--shard i/n] [--depth 12] [--max-states 6000000]
        [--judge-cpu-hours 8] [--prefilter loose|strict] [--workers N] [--check-all]

With --shard i/n (two machines) each job builds the tasks and backups hashing to i and fills its own share
(--fill: as many tables as its tasks); its files carry the suffix _iofn (summary, progress, final_ids, replacements,
DONE), and a resumed run reads every shard's summary.
    python -m reward.distance12 compare A_DIR B_DIR [--task-ids FILE]
    python -m reward.distance12 count-prefilter --rows ROWS --task-ids FILE --gs-root DIR [--depth 12]
    python -m reward.distance12 check-prefilter --rows ROWS --task-ids FILE --gs-root DIR [--workers N]

(--gs-root on the cluster: /mnt/umm/users/yinbaiqiao/InteriorGS_ply, the launch script's GS_ROOT.)

OUT/summary.jsonl gets one line per task built or over the limit, OUT/progress.json the running totals; with
``--target`` / ``--fill`` OUT/final_ids.txt (the tasks with tables) and OUT/replacements.json are written at the
end, and OUT/DONE.json marks the set complete.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import resource
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from data_gen.interiorgs_v10.difficulty import FastGeometry, to_c2w
from data_gen.interiorgs_v10.distance_table import (backward_distances, in_shard, load, may_succeed, reachable_graph,
                                                    save, table_path)

VERSION = "reward_distance12_v1"  # the arrays are the main line's interiorgs_v10_distance_table_v1 layout
DEFAULT_DEPTH = 12
DEFAULT_MAX_STATES = 6_000_000
DEFAULT_JUDGE_CPU_HOURS = 8.0
CHECK_EVERY = 256  # judged states between two looks at the CPU clock


def strict_objects(row: Mapping[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """The objects every one of which must meet the viewing cone, or None when the strict rule does not apply: the
    row's family requires every object of ``target_object.objects`` clearly visible (as many required slots as
    objects, all matched by the visibility rule's own lookup)."""
    from vagen.envs.active_spatial import target_visibility as TV

    objects = (row.get("target_object") or {}).get("objects") or []
    slots = TV.REQUIRED_VISIBLE_SLOTS.get(str(row.get("task_type") or ""), (0,))
    required = TV.required_targets(dict(row))
    if len(objects) < 2 or not (len(slots) == len(objects) == len(required)):
        return None
    return [obj for _, obj in required]


def prefilter_mask(row: Mapping[str, Any], c2w: np.ndarray, mode: str) -> Tuple[np.ndarray, str]:
    """(states that may succeed, the rule used): the main line's may_succeed, or with ``mode`` "strict" and a row
    the strict rule applies to, may_succeed of each required object alone, and-ed."""
    if mode == "strict":
        objs = strict_objects(row)
        if objs is not None:
            ok = np.ones(len(c2w), dtype=bool)
            for obj in objs:
                ok &= may_succeed({**row, "target_object": {**(row.get("target_object") or {}), "objects": [obj]}}, c2w)
            return ok, "strict"
    return may_succeed(row, c2w), "loose"


def build_task(scene: Any, row: Mapping[str, Any], depth: int = DEFAULT_DEPTH, *, max_states: int = DEFAULT_MAX_STATES,
               judge_cpu_hours: float = DEFAULT_JUDGE_CPU_HOURS, prefilter: str = "loose", check_all: bool = False,
               oracle: Any = None) -> Dict[str, Any]:
    """One task's table as distance_table.build makes it ({"keys", "dist", "depth_of", meta...}, status "ok"), or a
    record with status "over_states" / "over_time" and no arrays."""
    t0, c0 = time.perf_counter(), time.process_time()
    if oracle is None:
        from data_gen.interiorgs_v10.release import env_success_oracle

        oracle = env_success_oracle(row, scene)
    geom = FastGeometry.for_scene(scene)
    init = np.asarray(row["init_camera"]["extrinsics"], float)
    keys, P, R, depth_of, edges = reachable_graph(geom, init, depth, max_states)
    c1 = time.process_time()
    per_level = np.bincount(depth_of.astype(np.int64), minlength=depth + 1).tolist()
    d_star = (row.get("difficulty") or {}).get("d_star")
    rec: Dict[str, Any] = {"version": VERSION, "task_id": row.get("task_id"), "task_type": row.get("task_type"),
                           "d_star": None if d_star is None else int(d_star), "depth": int(depth),
                           "depth_reached": int(depth_of.max()), "states": int(len(keys)), "edges": int(len(edges)),
                           "per_level": per_level, "max_states": int(max_states),
                           "judge_cpu_hours_limit": float(judge_cpu_hours)}
    if int(depth_of.max()) < depth or len(keys) > max_states:
        rec.update(status="over_states", cpu_seconds={"graph": round(c1 - c0, 1)},
                   peak_rss_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2))
        return rec
    c2w = to_c2w(P, R)
    passed, rule = prefilter_mask(row, c2w, prefilter)
    idx = np.flatnonzero(passed)
    success = np.zeros(len(c2w), dtype=bool)
    limit = float(judge_cpu_hours) * 3600.0
    cj = time.process_time()
    for j, i in enumerate(idx):
        success[i] = bool(oracle(c2w[i]))
        if j % CHECK_EVERY == 0 and time.process_time() - cj > limit:
            rec.update(status="over_time", prefilter=rule, prefilter_pass=int(passed.sum()), judged=int(j + 1),
                       cpu_seconds={"graph": round(c1 - c0, 1), "judge": round(time.process_time() - cj, 1)},
                       peak_rss_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2))
            return rec
    c2 = time.process_time()
    missed = None
    if check_all:  # judge the states the prefilter skipped too (the 2a check; no time limit)
        rest = np.flatnonzero(~passed)
        missed = int(sum(bool(oracle(c2w[i])) for i in rest))
    c3 = time.process_time()
    dist = backward_distances(len(keys), edges, success)
    order = np.argsort(keys, kind="stable")
    rec.update(status="ok", keys=keys[order], dist=dist[order], depth_of=depth_of[order], d_init=int(dist[0]),
               agree=None if d_star is None else int(dist[0]) == int(d_star), success_states=int(success.sum()),
               prefilter=rule, prefilter_pass=int(passed.sum()), prefilter_missed_success=missed,
               unreached_share=round(float((dist < 0).mean()), 4),
               cpu_seconds={"graph": round(c1 - c0, 1), "judge": round(c2 - cj, 1), "check": round(c3 - c2, 1),
                            "total": round(time.process_time() - c0, 1)},
               seconds={"total": round(time.perf_counter() - t0, 1)},
               peak_rss_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2))
    return rec


def _meta(rec: Mapping[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in rec.items() if k not in ("keys", "dist", "depth_of")}


def _work(args: Tuple[List[Dict[str, Any]], str, str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows, gs_root, out, opts = args
    from data_gen.interiorgs_v10.scene import load_scene

    scenes: Dict[str, Any] = {}
    res = []
    for row in rows:
        path = table_path(Path(out), row["task_id"])
        if path.is_file():
            res.append({"task_id": row["task_id"], "status": "exists"})
            continue
        try:
            sid = str(row["scene_id"])
            if sid not in scenes:
                with contextlib.redirect_stdout(io.StringIO()):
                    scenes = {sid: load_scene(Path(gs_root) / sid, sid, wall_model="thick")}
            rec = build_task(scenes[sid], row, **opts)
            if rec["status"] == "ok":
                save(path, rec)
            res.append(_meta(rec))
        except Exception as exc:  # one task failing does not stop the rest
            res.append({"task_id": row.get("task_id"), "status": "error", "error": f"{type(exc).__name__}: {exc}"})
    return res


def _batches(rows: Sequence[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    out: List[List[Dict[str, Any]]] = []
    for r in sorted(rows, key=lambda r: (str(r["scene_id"]), r["task_id"])):  # one scene per batch
        if not out or out[-1][0]["scene_id"] != r["scene_id"] or len(out[-1]) >= 8:
            out.append([])
        out[-1].append(r)
    return out


def _read_ids(path: Optional[str]) -> List[str]:
    return [l.strip() for l in Path(path).read_text().splitlines() if l.strip()] if path else []


def _stratum(row: Mapping[str, Any]) -> Tuple[str, int]:
    return str(row["task_type"]), int((row.get("difficulty") or {}).get("d_star") or 0)


def pick_backup(failed: Mapping[str, Any], backups: List[Dict[str, Any]], used: set) -> Optional[Dict[str, Any]]:
    """The replacement of an over-limit task: the first unused backup of its stratum, else of its family with the
    nearest d*, else the first unused one."""
    fam, d = _stratum(failed)
    free = [b for b in backups if b["task_id"] not in used]
    for pool in ([b for b in free if _stratum(b) == (fam, d)],
                 sorted((b for b in free if b["task_type"] == fam), key=lambda b: abs(_stratum(b)[1] - d)), free):
        if pool:
            return pool[0]
    return None


def run_build(args: argparse.Namespace) -> None:
    from multiprocessing import Pool

    rows_all = [json.loads(l) for l in Path(args.rows).read_text().splitlines() if l.strip()]
    by_id = {r["task_id"]: r for r in rows_all}
    primary = [by_id[t] for t in _read_ids(args.task_ids) if in_shard(t, args.shard)]
    backups = [by_id[t] for t in _read_ids(args.backup_ids) if in_shard(t, args.shard)]
    target = len(primary) if args.fill else args.target
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = f"_{args.shard.replace('/', 'of')}" if args.shard else ""
    log_path, prog_path = out / f"summary{tag}.jsonl", out / f"progress{tag}.json"
    done: Dict[str, Dict[str, Any]] = {}
    for p in sorted(out.glob("summary*.jsonl")):  # resume: tasks already recorded (by any shard) keep their record
        for l in p.read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                if r.get("status") in ("ok", "over_states", "over_time"):
                    done[r["task_id"]] = r
    for r in primary + backups:
        if r["task_id"] not in done and table_path(out, r["task_id"]).is_file():
            meta = load(table_path(out, r["task_id"]))
            done[r["task_id"]] = {**{k: v for k, v in meta.items() if k not in ("keys", "dist", "depth_of")},
                                  "status": "ok"}
    opts = {"depth": args.depth, "max_states": args.max_states, "judge_cpu_hours": args.judge_cpu_hours,
            "prefilter": args.prefilter, "check_all": args.check_all}
    t0 = time.time()
    cpu_hours = sum(float((r.get("cpu_seconds") or {}).get("total") or 0) for r in done.values()) / 3600

    def run(rows: List[Dict[str, Any]], pool: Any, log: Any, stage: str) -> None:
        nonlocal cpu_hours
        todo = [r for r in rows if r["task_id"] not in done]
        n = 0
        for part in pool.imap_unordered(_work, [(b, args.gs_root, str(out), opts) for b in _batches(todo)]):
            for rec in part:
                rec["stage"] = stage
                log.write(json.dumps(rec) + "\n")
                n += 1
                if rec.get("status") in ("ok", "over_states", "over_time"):
                    done[rec["task_id"]] = rec
                    cpu_hours += float((rec.get("cpu_seconds") or {}).get("total")
                                       or sum((rec.get("cpu_seconds") or {}).values())) / 3600
            log.flush()
            prog_path.write_text(json.dumps({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "stage": stage,
                                             "shard": args.shard, "stage_done": n, "stage_todo": len(todo),
                                             "recorded": counts(), "cpu_hours": round(cpu_hours, 1),
                                             "wall_hours": round((time.time() - t0) / 3600, 2)}, indent=1))

    mine = {r["task_id"] for r in primary + backups}

    def counts() -> Dict[str, int]:
        st: Dict[str, int] = defaultdict(int)
        for t, r in done.items():
            if t in mine:
                st[r["status"]] += 1
        return dict(st)

    with Pool(args.workers, maxtasksperchild=1) as pool, open(log_path, "a") as log:
        run(primary, pool, log, "primary")
        final = [r for r in primary if done.get(r["task_id"], {}).get("status") == "ok"]
        replacements = []
        if target:
            used = {r["task_id"] for r in primary}
            failed = [r for r in primary if done.get(r["task_id"], {}).get("status") != "ok"]
            while len(final) < target and failed:
                wave = []
                for f in failed:
                    b = pick_backup(f, backups, used)
                    if b is None:
                        break
                    used.add(b["task_id"])
                    wave.append((f, b))
                if not wave:
                    break
                run([b for _, b in wave], pool, log, "backup")
                failed = []
                for f, b in wave:
                    status = done.get(b["task_id"], {}).get("status")
                    replacements.append({"replaced": f["task_id"], "reason": done.get(f["task_id"], {}).get("status"),
                                         "by": b["task_id"], "by_status": status,
                                         "stratum": list(_stratum(f)), "by_stratum": list(_stratum(b))})
                    if status == "ok":
                        final.append(b)
                    else:
                        failed.append(f)
            (out / f"final_ids{tag}.txt").write_text("".join(t + "\n" for t in sorted(r["task_id"] for r in final)))
            (out / f"replacements{tag}.json").write_text(json.dumps(replacements, indent=1))
    summary = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "shard": args.shard, "tasks": len(primary),
               "recorded": counts(), "final": len(final), "target": target,
               "replacements": len(replacements), "cpu_hours": round(cpu_hours, 1),
               "wall_hours": round((time.time() - t0) / 3600, 2), "options": opts}
    (out / f"DONE{tag}.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary))


def run_compare(args: argparse.Namespace) -> None:
    """Are two directories' tables identical (keys, dist, depth_of) for the given tasks (default: those in A)?"""
    a, b = Path(args.a), Path(args.b)
    ids = _read_ids(args.task_ids) or sorted(p.stem for p in a.glob("*.npz"))
    res = []
    for t in ids:
        pa, pb = table_path(a, t), table_path(b, t)
        if not (pa.is_file() and pb.is_file()):
            res.append({"task_id": t, "same": None, "missing": [str(p) for p in (pa, pb) if not p.is_file()]})
            continue
        ta, tb = load(pa), load(pb)
        same = all(np.array_equal(ta[k], tb[k]) for k in ("keys", "dist", "depth_of"))
        res.append({"task_id": t, "same": bool(same), "states": int(len(ta["keys"]))})
    print(json.dumps({"compared": len(res), "same": sum(1 for r in res if r["same"]),
                      "different": [r["task_id"] for r in res if r["same"] is False],
                      "missing": [r["task_id"] for r in res if r["same"] is None]}, indent=1))


def run_count_prefilter(args: argparse.Namespace) -> None:
    """Graph only: how many states each prefilter passes (no judging)."""
    from data_gen.interiorgs_v10.scene import load_scene

    rows = {json.loads(l)["task_id"]: json.loads(l) for l in Path(args.rows).read_text().splitlines() if l.strip()}
    for t in _read_ids(args.task_ids):
        row = rows[t]
        with contextlib.redirect_stdout(io.StringIO()):
            scene = load_scene(Path(args.gs_root) / str(row["scene_id"]), str(row["scene_id"]), wall_model="thick")
        keys, P, R, depth_of, edges = reachable_graph(FastGeometry.for_scene(scene),
                                                      np.asarray(row["init_camera"]["extrinsics"], float), args.depth,
                                                      DEFAULT_MAX_STATES)
        c2w = to_c2w(P, R)
        loose, _ = prefilter_mask(row, c2w, "loose")
        strict, rule = prefilter_mask(row, c2w, "strict")
        print(json.dumps({"task_id": t, "task_type": row["task_type"], "d_star": _stratum(row)[1], "states": len(keys),
                          "loose_pass": int(loose.sum()), "strict_pass": int(strict.sum()), "rule": rule,
                          "strict_over_loose": round(float(strict.sum()) / max(1, int(loose.sum())), 4),
                          "strict_not_subset": int((strict & ~loose).sum())}), flush=True)


def _judge_chunk(args: Tuple[Dict[str, Any], str, np.ndarray, np.ndarray]) -> List[int]:
    row, gs_root, idx, c2w = args
    from data_gen.interiorgs_v10.release import env_success_oracle
    from data_gen.interiorgs_v10.scene import load_scene

    with contextlib.redirect_stdout(io.StringIO()):
        scene = load_scene(Path(gs_root) / str(row["scene_id"]), str(row["scene_id"]), wall_model="thick")
    oracle = env_success_oracle(row, scene)
    return [int(i) for i, m in zip(idx, c2w) if bool(oracle(m))]


def run_check_prefilter(args: argparse.Namespace) -> None:
    """The 2a check of the prefilters: judge every state the loose prefilter skips (chunks over ``--workers``
    processes) and count the successful ones (must be 0); also how many states the strict rule skips on top."""
    from multiprocessing import Pool

    from data_gen.interiorgs_v10.scene import load_scene

    rows = {json.loads(l)["task_id"]: json.loads(l) for l in Path(args.rows).read_text().splitlines() if l.strip()}
    with Pool(args.workers) as pool:
        for t in _read_ids(args.task_ids):
            row = rows[t]
            c0 = time.perf_counter()
            with contextlib.redirect_stdout(io.StringIO()):
                scene = load_scene(Path(args.gs_root) / str(row["scene_id"]), str(row["scene_id"]), wall_model="thick")
            keys, P, R, depth_of, edges = reachable_graph(FastGeometry.for_scene(scene),
                                                          np.asarray(row["init_camera"]["extrinsics"], float),
                                                          args.depth, DEFAULT_MAX_STATES)
            c2w = to_c2w(P, R)
            loose, _ = prefilter_mask(row, c2w, "loose")
            strict, rule = prefilter_mask(row, c2w, "strict")
            rest = np.flatnonzero(~loose)
            parts = [(row, args.gs_root, rest[k:k + args.chunk], c2w[rest[k:k + args.chunk]])
                     for k in range(0, len(rest), args.chunk)]
            hits = [i for part in pool.imap_unordered(_judge_chunk, parts) for i in part]
            print(json.dumps({"task_id": t, "task_type": row["task_type"], "d_star": _stratum(row)[1],
                              "states": int(len(keys)), "loose_pass": int(loose.sum()), "strict_pass": int(strict.sum()),
                              "strict_rule": rule, "loose_skipped_judged": int(len(rest)),
                              "loose_skipped_successful": len(hits), "seconds": round(time.perf_counter() - c0, 1)}),
                  flush=True)


def main(argv: Optional[Sequence[str]] = None) -> None:
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):  # one core per worker
        os.environ.setdefault(var, "1")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--rows", required=True)
    b.add_argument("--task-ids", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--backup-ids", default=None)
    b.add_argument("--target", type=int, default=None, help="replace over-limit tasks until N tables")
    b.add_argument("--fill", action="store_true", help="--target = the number of (this shard's) tasks")
    b.add_argument("--shard", default=None, help="i/n: only the tasks and backups hashing to i (distance_table's rule)")
    b.add_argument("--gs-root", required=True)
    b.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    b.add_argument("--max-states", type=int, default=DEFAULT_MAX_STATES)
    b.add_argument("--judge-cpu-hours", type=float, default=DEFAULT_JUDGE_CPU_HOURS)
    b.add_argument("--prefilter", choices=("loose", "strict"), default="loose")
    b.add_argument("--workers", type=int, default=12)
    b.add_argument("--check-all", action="store_true", help="judge the prefilter-skipped states too (2a check)")
    c = sub.add_parser("compare")
    c.add_argument("a")
    c.add_argument("b")
    c.add_argument("--task-ids", default=None)
    p = sub.add_parser("count-prefilter")
    p.add_argument("--rows", required=True)
    p.add_argument("--task-ids", required=True)
    p.add_argument("--gs-root", required=True)
    p.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    k = sub.add_parser("check-prefilter")
    k.add_argument("--rows", required=True)
    k.add_argument("--task-ids", required=True)
    k.add_argument("--gs-root", required=True)
    k.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    k.add_argument("--workers", type=int, default=12)
    k.add_argument("--chunk", type=int, default=20000)
    args = ap.parse_args(argv)
    {"build": run_build, "compare": run_compare, "count-prefilter": run_count_prefilter,
     "check-prefilter": run_check_prefilter}[args.cmd](args)


if __name__ == "__main__":
    main()
