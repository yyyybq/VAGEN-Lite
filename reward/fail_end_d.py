"""Reward line, step 4 (PREREG_B section 5): where the failed evaluation episodes end, as the true remaining number of
actions d of their last pose in the evaluation task's depth-12 table (reward_runs/tables/eval705).

The evaluation records each turn's answer but not the pose after the last one, so every failed episode is replayed
in the evaluation's own env (the evaluation's template, default configs/rl_env_v10_S0.yaml, with the result's
prompt format and env switches from its model.json, the evaluation's gs_root) without rendering: env.reset on the
task's index, one env.step per recorded answer.  The replay is checked turn by turn against the record (task met
after the turn, blocked moves); the last pose is looked up in the table:

* d >= 0: the remaining minimum (exact when the pose's depth from the start + d <= 12, else an upper bound);
* unreached: in the table, no success reachable inside the depth-12 graph from there;
* missing: the pose's 1 cm / 1 deg cell not in the table (the merge, see STATUS's known limitations).

    python -m reward.fail_end_d --set val=DIR --set test=DIR --tables DIR --out FILE.json
        [--results rw_o_050,...] [--env-template YAML] [--gs-root DIR] [--workers 4]

Without --results every rw_* result folder with done.json in the sets.  FILE.json and FILE.md.
"""

import argparse
import json
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

DEPTH = 12
TEMPLATE = Path(__file__).resolve().parents[1] / "data_gen" / "interiorgs_v10" / "configs" / "rl_env_v10_S0.yaml"


def worker(args):
    tasks_path, template, gs_root, model_json, tables, items = args
    import yaml

    from data_gen.interiorgs_v10 import distance_table as DT
    from data_gen.interiorgs_v10.difficulty import _split, contract_keys
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
    out = []
    for r in items:
        try:
            env.reset(seed=int(r["index"]))
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
            k = int(contract_keys(*_split(env.view_engine.get_pose()))[0])
            keys = table["keys"]
            i = int(np.searchsorted(keys, k))
            if i < len(keys) and int(keys[i]) == k:
                d, dep = int(table["dist"][i]), int(table["depth_of"][i])
                kind = "d" if d >= 0 else "unreached"
            else:
                d, dep, kind = None, None, "missing"
            out.append({"index": r["index"], "task_id": r["task_id"], "task_type": r["task_type"],
                        "d_star": int(r["d_star"]), "replay_same": bool(same), "kind": kind,
                        "d_end": d if kind == "d" else None, "depth_end": dep,
                        "exact": bool(kind == "d" and dep + d <= DEPTH)})
        except Exception as exc:
            out.append({"index": r["index"], "task_id": r["task_id"], "error": f"{type(exc).__name__}: {exc}"})
    return out


def summarize(rows):
    ok = [x for x in rows if "error" not in x]
    ds = [x["d_end"] for x in ok if x["kind"] == "d"]
    kinds = defaultdict(int)
    for x in ok:
        kinds[x["kind"]] += 1
    mean = lambda v: round(float(np.mean(v)), 3) if v else None  # noqa: E731
    by_family = defaultdict(list)
    for x in ok:
        if x["kind"] == "d":
            by_family[x["task_type"]].append(x["d_end"])
    return {"failed": len(rows), "errors": len(rows) - len(ok), "replay_same": sum(x["replay_same"] for x in ok),
            "kinds": dict(kinds), "d_end_mean": mean(ds), "d_end_median": float(np.median(ds)) if ds else None,
            "d_star_mean": mean([x["d_star"] for x in ok if x["kind"] == "d"]),
            "d_end_le1": sum(d <= 1 for d in ds), "d_end_le2": sum(d <= 2 for d in ds),
            "exact": sum(x["exact"] for x in ok),
            "d_end_mean_by_family": {k: mean(v) for k, v in sorted(by_family.items())}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", action="append", required=True, help="NAME=DIR")
    ap.add_argument("--tables", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--results", default=None, help="comma-separated result names (default: every rw_* folder)")
    ap.add_argument("--env-template", default=str(TEMPLATE))
    ap.add_argument("--gs-root", default="/mnt/umm/users/yinbaiqiao/InteriorGS_ply")  # the evaluation's default
    ap.add_argument("--workers", type=int, default=4)  # the login node: at most 4 cores a task
    args = ap.parse_args()
    sets = dict(s.split("=", 1) for s in args.set)
    want = set(args.results.split(",")) if args.results else None
    jobs, keys = [], []
    for set_name, d in sets.items():
        for p in sorted(Path(d).glob("rw_*")):
            if not (p / "done.json").is_file() or (want is not None and p.name not in want):
                continue
            failed = [json.loads(l) for l in (p / "results.jsonl").read_text().splitlines() if l.strip()]
            failed = [r for r in failed if "error" not in r and not r["success"]]
            for k in range(0, len(failed), 40):
                jobs.append((str(Path(d) / "tasks.jsonl"), args.env_template, args.gs_root, str(p / "model.json"),
                             args.tables, failed[k:k + 40]))
                keys.append((p.name, set_name))
    per = defaultdict(list)
    with Pool(args.workers) as pool:
        for key, part in zip(keys, pool.imap(worker, jobs)):
            per[key] += part
    out = {f"{name}|{s}": summarize(rows) for (name, s), rows in sorted(per.items())}
    Path(args.out).write_text(json.dumps({"sets": sets, "tables": args.tables, "results": out,
                                          "episodes": {f"{n}|{s}": v for (n, s), v in sorted(per.items())}}, indent=1))
    L = ["| 结果 | 集合 | 失败局 | 重放一致 | 终点 d 平均（中位） | 这些题的 d\\* 平均 | d ≤ 1 | d ≤ 2 | 不可达 | 查不到 |",
         "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for k, s in out.items():
        name, set_name = k.split("|")
        L.append(f"| {name} | {set_name} | {s['failed']} | {s['replay_same']} | {s['d_end_mean']}（{s['d_end_median']}） | "
                 f"{s['d_star_mean']} | {s['d_end_le1']} | {s['d_end_le2']} | {s['kinds'].get('unreached', 0)} | "
                 f"{s['kinds'].get('missing', 0)} |")
    Path(args.out).with_suffix(".md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
