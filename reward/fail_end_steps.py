"""Reward line (phase 2, the user's brief of 2026-11-01, item 5.1): the true remaining number of actions where failed
evaluation episodes end, for any evaluation result, given by path; for the main line as a standard metric.

The evaluation records each turn's answer but not the pose after the last one, so every failed episode of a result
is replayed without rendering in the evaluation's env (the template, default configs/rl_env_v10_S0.yaml, with the
result's prompt format and env switches from its model.json): env.reset on the task's line in the tasks file (found
by its task_id, so a result assembled from several evaluations works too), one env.step per
recorded answer, checked turn by turn against the record (task met after the turn, blocked moves).  The last pose is
looked up in the task's distance table:

* "d": the pose's cell is in the table with a reachable success: d is the remaining minimum (exact when the pose's
  depth from the start + d <= the table's depth, else an upper bound);
* "near": the cell is missing (the 1 cm / 1 deg merge), a neighbour cell has a reachable success: d is the smallest
  among the 26 neighbours (data_gen/interiorgs_v10/distance_table_near.py; --no-near counts these as missing);
* "unreached": in the table, no success reachable inside the table's graph from there;
* "missing": neither the cell nor a neighbour in the table.

Tables: one ``<task_id>.npz`` a task (data_gen.interiorgs_v10.distance_table's format).  The 705 evaluation tasks of
eval_val_v2 + eval_test_v2 have depth-12 tables in reward_runs/tables/eval705 (read only; AOSS copy
fj:s3://baiqiao/vagen_lite_archive/reward_runs/tables/eval705).  An evaluation set without tables needs them built
first (the tool stops and lists the tasks without a table), from the set's own tasks.jsonl:

    python -m reward.distance12 build --rows SET/tasks.jsonl --task-ids IDS.txt --out TABLES --gs-root GS \
        --depth 12 --prefilter strict --workers N

about 0.38 CPU hours a task (the 705: 269 CPU hours; one 8-card sco job's 110 workers take about 2.5 hours, B's
one-card-equivalent run 4.9 hours); data making goes to sco, not the login node.

    python -m reward.fail_end_steps --result DIR [--result DIR ...] --tables DIR --out FILE.json
        [--tasks FILE] [--env-template YAML] [--gs-root DIR] [--no-near] [--allow-missing] [--workers 4]

A result folder holds results.jsonl, model.json and done.json; its tasks file is --tasks or the set's tasks.jsonl
next to it (DIR/../tasks.jsonl).  FILE.json (per result the summary and every failed episode) and FILE.md (a table).
"""

import argparse
import json
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

TEMPLATE = Path(__file__).resolve().parents[1] / "data_gen" / "interiorgs_v10" / "configs" / "rl_env_v10_S0.yaml"
GS_ROOT = "/mnt/umm/users/yinbaiqiao/InteriorGS_ply"  # the evaluation's default


def worker(args):
    tasks_path, template, gs_root, model_json, tables, near, items = args
    import yaml

    from data_gen.interiorgs_v10 import distance_table as DT
    from data_gen.interiorgs_v10.difficulty import _split, contract_keys
    from data_gen.interiorgs_v10.distance_table_near import neighbour_distance
    from vagen.envs.active_spatial.env import ActiveSpatialEnv
    from vagen.envs.active_spatial.env_config import ActiveSpatialEnvConfig

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
    cache = DT.TableCache(tables)
    # the task's line in the tasks file by its id (a result folder assembled from several evaluations can carry
    # another file's indices)
    position = {}
    for i, line in enumerate(l for l in Path(tasks_path).read_text().splitlines() if l.strip()):
        position.setdefault(json.loads(line)["task_id"], i)
    out = []
    for r in items:
        try:
            env.reset(seed=position[r["task_id"]])
            assert env.current_item["task_id"] == r["task_id"], (env.current_item["task_id"], r["task_id"])
            met, blocked = [], []
            for t in r["trajectory"]:
                _, _, done, info = env.step(t["answer"])
                tm = (info.get("metrics") or {}).get("turn_metrics") or {}
                blocked.append(int(tm.get("collision_count", 0) or 0))
                m = env._calculate_canonical_metric()
                met.append(bool(m and m.get("success")))
                if done:
                    break
            same = (met == [bool(t["met"]) for t in r["trajectory"]] and
                    blocked == [int(t["collisions"]) for t in r["trajectory"]])
            table = cache.get(r["task_id"])
            depth = int(table.get("depth") or 12)
            k = int(contract_keys(*_split(env.view_engine.get_pose()))[0])
            keys = table["keys"]
            i = int(np.searchsorted(keys, k))
            d = dep = None
            if i < len(keys) and int(keys[i]) == k:
                d, dep = int(table["dist"][i]), int(table["depth_of"][i])
                kind = "d" if d >= 0 else "unreached"
                d = d if d >= 0 else None
            else:
                nd = neighbour_distance(table, k) if near else None
                kind, d = ("near", nd) if nd is not None else ("missing", None)
            out.append({"index": r["index"], "task_id": r["task_id"], "task_type": r["task_type"],
                        "d_star": int(r["d_star"]), "replay_same": bool(same), "kind": kind, "d_end": d,
                        "depth_end": dep, "exact": bool(kind == "d" and dep + d <= depth)})
        except Exception as exc:
            out.append({"index": r["index"], "task_id": r["task_id"], "error": f"{type(exc).__name__}: {exc}"})
    return out


def summarize(rows):
    ok = [x for x in rows if "error" not in x]
    known = [x for x in ok if x["kind"] in ("d", "near")]
    ds = [x["d_end"] for x in known]
    kinds = defaultdict(int)
    for x in ok:
        kinds[x["kind"]] += 1
    mean = lambda v: round(float(np.mean(v)), 3) if v else None  # noqa: E731
    by_family = defaultdict(list)
    for x in known:
        by_family[x["task_type"]].append(x["d_end"])
    return {"failed": len(rows), "errors": len(rows) - len(ok), "replay_same": sum(x["replay_same"] for x in ok),
            "kinds": dict(kinds), "d_end_mean": mean(ds), "d_end_median": float(np.median(ds)) if ds else None,
            "d_star_mean": mean([x["d_star"] for x in known]),
            "d_end_le1": sum(d <= 1 for d in ds), "d_end_le2": sum(d <= 2 for d in ds),
            "exact": sum(x["exact"] for x in ok),
            "d_end_mean_by_family": {k: mean(v) for k, v in sorted(by_family.items())}}


def failed_episodes(result: Path):
    rows = [json.loads(l) for l in (result / "results.jsonl").read_text().splitlines() if l.strip()]
    return [r for r in rows if "error" not in r and not r["success"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--result", action="append", required=True, help="an evaluation result folder (repeatable)")
    ap.add_argument("--tables", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", default=None, help="the set's tasks.jsonl (default: next to each result folder)")
    ap.add_argument("--env-template", default=str(TEMPLATE))
    ap.add_argument("--gs-root", default=GS_ROOT)
    ap.add_argument("--no-near", action="store_true", help="count a missing cell as missing, no neighbour fallback")
    ap.add_argument("--allow-missing", action="store_true", help="replay the tasks that have a table, skip the rest")
    ap.add_argument("--workers", type=int, default=4)  # the login node: at most 4 cores a task
    args = ap.parse_args()
    tables = Path(args.tables)
    jobs, keys, no_table = [], [], {}
    for res in [Path(p) for p in args.result]:
        if not (res / "done.json").is_file():
            raise SystemExit(f"{res}: no done.json (the evaluation is not finished)")
        tasks = Path(args.tasks) if args.tasks else res.parent / "tasks.jsonl"
        if not tasks.is_file():
            raise SystemExit(f"{res}: no tasks file {tasks} (give --tasks)")
        failed = failed_episodes(res)
        missing = sorted({r["task_id"] for r in failed if not (tables / f"{r['task_id']}.npz").is_file()})
        if missing:
            no_table[str(res)] = missing
            failed = [r for r in failed if r["task_id"] not in set(missing)]
        for k in range(0, len(failed), 40):
            jobs.append((str(tasks), args.env_template, args.gs_root, str(res / "model.json"), str(tables),
                         not args.no_near, failed[k:k + 40]))
            keys.append(str(res))
    if no_table and not args.allow_missing:
        n = len({t for v in no_table.values() for t in v})
        raise SystemExit(f"{n} failed tasks have no table in {tables} (e.g. {next(iter(no_table.values()))[:3]}): "
                         "build them first (see the module's docstring), or pass --allow-missing")
    per = defaultdict(list)
    with Pool(args.workers) as pool:
        for key, part in zip(keys, pool.imap(worker, jobs)):
            per[key] += part
    out = {k: summarize(v) for k, v in sorted(per.items())}
    for k, v in no_table.items():
        out.setdefault(k, {})["skipped_without_table"] = len(v)
    Path(args.out).write_text(json.dumps({"tables": str(tables), "near": not args.no_near, "results": out,
                                          "no_table": no_table, "episodes": dict(sorted(per.items()))}, indent=1))
    L = ["| 结果 | 失败局 | 重放一致 | 终点 d 平均（中位） | 这些题的 d\\* 平均 | d ≤ 1 | d ≤ 2 | 相邻格给出 | 不可达 | 查不到 |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for k, s in out.items():
        if "failed" not in s:
            continue
        kinds = s["kinds"]
        L.append(f"| {Path(k).parent.name}/{Path(k).name} | {s['failed']} | {s['replay_same']} | {s['d_end_mean']}"
                 f"（{s['d_end_median']}） | {s['d_star_mean']} | {s['d_end_le1']} | {s['d_end_le2']} | "
                 f"{kinds.get('near', 0)} | {kinds.get('unreached', 0)} | {kinds.get('missing', 0)} |")
    Path(args.out).with_suffix(".md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
