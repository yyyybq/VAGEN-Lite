"""Reward line (the user's task of 2026-11-05, section 6.2): perturbed-start evaluation tasks.

For every task of an evaluation set that has a depth-12 distance table (reward_runs/tables/eval705 for the 705 tasks of
eval_val_v2 + eval_test_v2), a pose exactly k actions from the start (k = 2, then 4) whose true remaining number of
actions is exactly d* + k (the start's own remaining number plus k: the agent was pushed k actions the wrong way)
becomes the task's new start.  Nothing else of the row changes (target, success rule, step limit); only new task
files are written, the env is not touched.

The poses come from the tables' own graph (data_gen.interiorgs_v10.distance_table.reachable_graph from the row's
init_camera, the env's transitions and collision checks, depth k), each looked up in the task's table for its
remaining number; among the candidates one is drawn from a seed of the task id and k (reproducible).  Every new start
is checked by replaying the env without rendering: after reset the env's pose is the new start, its contract key's
remaining number in the table is d* + k, and the task is not already met there.

    python -m reward.perturbed_starts build --set val=EVAL_VAL/tasks.jsonl --set test=EVAL_TEST/tasks.jsonl \
        --tables reward_runs/tables/eval705 --gs-root /mnt/umm/users/yinbaiqiao/InteriorGS_ply --k 2 --k 4 \
        --out OUT [--workers 12]

OUT/<set>_k<k>/tasks.jsonl (the rows with a new start; row["perturbation"] tells how it was made, row["difficulty"]
["d_star"] is d* + k), OUT/summary.json and OUT/summary.md (per set and k: tasks, tasks with a new start, without a
candidate, failed checks; by family and by d*).  Data making: run it as a one-card sco job (reward.sco_job).
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import io
import json
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

TEMPLATE = Path(__file__).resolve().parents[1] / "data_gen" / "interiorgs_v10" / "configs" / "rl_env_v10_S0.yaml"
VERSION = "reward_perturbed_starts_v1"


def pick(task_id: str, k: int, n: int) -> int:
    """The candidate drawn for a task and k (a seed from both, so the draw never depends on the run)."""
    seed = int(hashlib.sha1(f"{task_id}:{k}".encode()).hexdigest()[:12], 16)
    return int(np.random.default_rng(seed).integers(n))


def candidates(table: Dict[str, Any], keys: np.ndarray, depth_of: np.ndarray, k: int, d_star: int) -> np.ndarray:
    """Indices (into the graph's states) of the states k actions from the start whose remaining number in the table
    is d* + k."""
    order = np.searchsorted(table["keys"], keys)
    order = np.minimum(order, len(table["keys"]) - 1)
    found = table["keys"][order] == keys
    dist = np.where(found, table["dist"][order], -1)
    return np.flatnonzero((depth_of == k) & found & (dist == d_star + k))


def _env(tasks_path: Path, gs_root: str):
    import yaml

    from vagen.envs.active_spatial.env import ActiveSpatialEnv
    from vagen.envs.active_spatial.env_config import ActiveSpatialEnvConfig

    cfg = dict(yaml.safe_load(TEMPLATE.read_text())["envs"][0]["config"])
    cfg.update(jsonl_path=str(tasks_path), gs_root=gs_root, render_backend=None, r1_legacy_success=False)
    env = ActiveSpatialEnv(ActiveSpatialEnvConfig(**{k: v for k, v in cfg.items()
                                                     if k in ActiveSpatialEnvConfig.__dataclass_fields__}))

    def fake_render(**_kwargs):  # as reward.fail_end_steps: the replay needs poses, not pictures
        env.last_image_std = 30.0
        env.last_low_info_frame = False
        env.consecutive_low_info_frame_count = 0
        return {"obs_str": "", "multi_modal_data": {}}

    env._render = fake_render
    return env


def _work(args: Tuple[List[Dict[str, Any]], str, str, List[int], str]) -> List[Dict[str, Any]]:
    """One scene's tasks: per task and k a new row (or the reason there is none), each checked by an env replay."""
    rows, tables, gs_root, ks, scratch = args
    from data_gen.interiorgs_v10 import distance_table as DT
    from data_gen.interiorgs_v10.difficulty import FastGeometry, _split, contract_keys, to_c2w
    from data_gen.interiorgs_v10.scene import load_scene

    with contextlib.redirect_stdout(io.StringIO()):
        scene = load_scene(Path(gs_root) / str(rows[0]["scene_id"]), str(rows[0]["scene_id"]), wall_model="thick")
    geom = FastGeometry.for_scene(scene)
    out = []
    for row in rows:
        tid = row["task_id"]
        try:
            table = DT.load(DT.table_path(Path(tables), tid))
            d_star = int(table.get("d_init", (row.get("difficulty") or {}).get("d_star")))
            init = np.asarray(row["init_camera"]["extrinsics"], float)
            keys, P, R, depth_of, _ = DT.reachable_graph(geom, init, max(ks))
            for k in ks:
                idx = candidates(table, keys, depth_of, k, d_star)
                rec = {"task_id": tid, "task_type": row.get("task_type"), "d_star": d_star, "k": k,
                       "candidates": int(len(idx)), "states_at_k": int((depth_of == k).sum())}
                if not len(idx):
                    out.append(dict(rec, status="no_candidate"))
                    continue
                i = int(idx[pick(tid, k, len(idx))])
                new = copy.deepcopy(row)
                new["init_camera"]["extrinsics"] = to_c2w(P[i:i + 1], R[i:i + 1])[0].tolist()
                new.setdefault("difficulty", {})
                new["difficulty"] = dict(new["difficulty"], d_star=d_star + k, certificate=None,
                                         certificate_source="perturbed start (no certificate)")
                new["perturbation"] = {"version": VERSION, "k": k, "d_star_original": d_star,
                                       "candidates": int(len(idx)), "key": int(keys[i]),
                                       "init_extrinsics_original": row["init_camera"]["extrinsics"]}
                out.append(dict(rec, status="ok", row=new))
        except Exception as exc:  # one task failing does not stop the rest
            out.append({"task_id": tid, "status": "error", "error": f"{type(exc).__name__}: {exc}"})
    # the check: the env's own start pose for each new row
    made = [r for r in out if r.get("status") == "ok"]
    if made:
        path = Path(scratch) / f"check_{rows[0]['scene_id']}_{hashlib.sha1(str([r['task_id'] for r in rows]).encode()).hexdigest()[:8]}.jsonl"
        path.write_text("".join(json.dumps(r["row"]) + "\n" for r in made))
        env = _env(path, gs_root)
        for j, r in enumerate(made):
            try:
                env.reset(seed=j)
                assert env.current_item["task_id"] == r["task_id"], (env.current_item["task_id"], r["task_id"])
                c2w = np.asarray(env.view_engine.get_pose(), float)
                same = bool(np.allclose(c2w, np.asarray(r["row"]["init_camera"]["extrinsics"]), atol=1e-6))
                k_env = int(contract_keys(*_split(c2w))[0])
                table = DT.load(DT.table_path(Path(tables), r["task_id"]))
                at = int(np.searchsorted(table["keys"], k_env))
                d_env = int(table["dist"][at]) if at < len(table["keys"]) and int(table["keys"][at]) == k_env else None
                metric = env._calculate_canonical_metric()
                met = bool(metric and metric.get("success"))
                r["check"] = {"pose_same": same, "d_env": d_env, "met_at_start": met,
                              "ok": same and d_env == r["d_star"] + r["k"] and not met}
            except Exception as exc:
                r["check"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        path.unlink()
    return out


def build(sets: Sequence[Tuple[str, Path]], tables: str, gs_root: str, ks: Sequence[int], out: Path,
          workers: int = 4) -> Dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    scratch = out / "scratch"
    scratch.mkdir(exist_ok=True)
    summary: Dict[str, Any] = {"version": VERSION, "tables": tables, "ks": list(ks), "sets": {}}
    for name, tasks_path in sets:
        rows = [json.loads(l) for l in Path(tasks_path).read_text().splitlines() if l.strip()]
        by_scene: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in rows:
            by_scene[str(r["scene_id"])].append(r)
        jobs = [(v, tables, gs_root, list(ks), str(scratch)) for _, v in sorted(by_scene.items())]
        with Pool(max(1, workers)) as pool:
            results = [x for part in pool.imap_unordered(_work, jobs) for x in part]
        order = {r["task_id"]: i for i, r in enumerate(rows)}
        set_summary: Dict[str, Any] = {"tasks": len(rows), "file": str(tasks_path)}
        for k in ks:
            got = sorted((r for r in results if r.get("k") == k), key=lambda r: order[r["task_id"]])
            errors = [r for r in results if r.get("status") == "error"]
            good = [r for r in got if r["status"] == "ok" and r.get("check", {}).get("ok")]
            bad_check = [r for r in got if r["status"] == "ok" and not r.get("check", {}).get("ok")]
            folder = out / f"{name}_k{k}"
            folder.mkdir(exist_ok=True)
            (folder / "tasks.jsonl").write_text("".join(json.dumps(r["row"], ensure_ascii=False) + "\n" for r in good))
            fam = defaultdict(lambda: [0, 0])
            dstar = defaultdict(lambda: [0, 0])
            for r in got:
                fam[r["task_type"]][0] += 1
                dstar[r["d_star"]][0] += 1
                if r in good:
                    fam[r["task_type"]][1] += 1
                    dstar[r["d_star"]][1] += 1
            set_summary[f"k{k}"] = {"made": len(good), "no_candidate": sum(r["status"] == "no_candidate" for r in got),
                                    "failed_check": [{"task_id": r["task_id"], **r.get("check", {})} for r in bad_check],
                                    "errors": [{"task_id": r["task_id"], "error": r["error"]} for r in errors],
                                    "candidates_median": float(np.median([r["candidates"] for r in got if r["candidates"]]))
                                    if any(r.get("candidates") for r in got) else None,
                                    "by_family": {f: {"tasks": v[0], "made": v[1]} for f, v in sorted(fam.items())},
                                    "by_d_star": {str(d): {"tasks": v[0], "made": v[1]} for d, v in sorted(dstar.items())},
                                    "file": str(folder / "tasks.jsonl")}
        summary["sets"][name] = set_summary
    scratch.rmdir()
    (out / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False))
    lines = ["| 题集 | k | 题数 | 做成 | 没有候选 | 核对没过 | 出错 |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for name, s in summary["sets"].items():
        for k in ks:
            t = s[f"k{k}"]
            lines.append(f"| {name} | {k} | {s['tasks']} | {t['made']} | {t['no_candidate']} | {len(t['failed_check'])} "
                         f"| {len(t['errors'])} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main(argv: Sequence[str] = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--set", action="append", required=True, help="NAME=TASKS.jsonl")
    b.add_argument("--tables", required=True)
    b.add_argument("--gs-root", required=True)
    b.add_argument("--k", type=int, action="append", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    sets = [(s.split("=", 1)[0], Path(s.split("=", 1)[1])) for s in a.set]
    summary = build(sets, a.tables, a.gs_root, sorted(set(a.k)), Path(a.out), a.workers)
    print(json.dumps({n: {k: {kk: vv for kk, vv in v.items() if kk in ("made", "no_candidate")}
                          for k, v in s.items() if k.startswith("k")} for n, s in summary["sets"].items()}, indent=1))


if __name__ == "__main__":
    main()
