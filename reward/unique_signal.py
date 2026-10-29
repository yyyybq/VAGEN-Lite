"""Reward line, step 4 (PREREG_B section 5): how much of a reward's training signal S0 does not already have, the way
DENSE_SIGNAL_R13 section A.1 measured it (the main line, 2026-10-26).

A run's recorded candidate groups (every EVERY-th update, the 24 tasks x 8 episodes of each) are replayed without
rendering under each given reward config (a run's train_env.yaml: S0, O, G ...), replay_q3's way; the config the
run trained with is checked against the logged returns.  Each setting = a reward + a group filter as
vagen/custom_filter/filter.py has them (target_groups 12, candidates in the file's order):

* mixed: the tasks whose episodes both succeed and fail, the nominal batch's first, refilled from the extra ones;
* keep:T:F: then the all-fail tasks with return spread (std, n - 1) >= T, by decreasing spread, into the places
  still free, their advantages scaled by (spread + eps) / (max(spread, F) + eps).

Per task the GRPO advantage vector a = (R - mean) / (std + 1e-6).  For a setting X against the base B (S0): a task
X trains and B also trains has unique signal |a_X|^2 (1 - cos^2(a_X, a_B)); a task only X trains, all of |a_X|^2.
Unique share = the sum of the unique parts / the sum of |a_X|^2 over the tasks X trains.  Also by family, the part
from tasks only X trains, the tasks per update, and for the mixed tasks the mean cos^2.

    python -m reward.unique_signal --run RUN_DIR --rows ROWS --reward S0=YAML --reward O=YAML [--reward G=YAML]
        --own O --out OUT [--setting OK=O:keep:0.6403:1.0 ...] [--every 4] [--max-update N] [--workers 4]

Default settings: S0=S0:mixed, O=O:mixed, G=G:keep:0.3318:1.0, OK=O:keep:0.6403:1.0 (each kept if its reward is
given).  OUT/summary.json, OUT/groups.jsonl.
"""

import argparse
import json
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from reward.replay_q3 import answers_of

EPS = 1e-6
TARGET = 12
DEFAULT_SETTINGS = {"S0": "S0:mixed", "O": "O:mixed", "G": "G:keep:0.3318:1.0", "OK": "O:keep:0.6403:1.0"}


def worker(args):
    chunk_path, rewards = args
    import yaml

    from vagen.envs.active_spatial.env import ActiveSpatialEnv
    from vagen.envs.active_spatial.env_config import ActiveSpatialEnvConfig

    groups = [json.loads(l) for l in Path(chunk_path).read_text().splitlines() if l.strip()]
    rows_path = Path(chunk_path).with_suffix(".rows.jsonl")
    rows_path.write_text("".join(json.dumps(g["row"]) + "\n" for g in groups))
    fields = set(ActiveSpatialEnvConfig.__dataclass_fields__)
    envs = {}
    for name, env_yaml in rewards.items():
        cfg = dict(yaml.safe_load(Path(env_yaml).read_text())["envs"][0]["config"])
        cfg.update(jsonl_path=str(rows_path), render_backend=None)
        env = ActiveSpatialEnv(ActiveSpatialEnvConfig(**{k: v for k, v in cfg.items() if k in fields}))

        def fake_render(env=env, **_kwargs):
            env.last_image_std = 30.0
            env.last_low_info_frame = False
            env.consecutive_low_info_frame_count = 0
            return {"obs_str": "", "multi_modal_data": {}}

        env._render = fake_render
        envs[name] = env
    out = []
    for gi, g in enumerate(groups):
        tid = g["row"]["task_id"]
        try:
            rets = {name: [] for name in envs}
            for ep in g["episodes"]:
                for name, env in envs.items():
                    env.reset(seed=gi)
                    assert env.current_item["task_id"] == tid
                    total = 0.0
                    for a in ep["answers"]:
                        _, r, done, _ = env.step(a)
                        total += float(r)
                        if done:
                            break
                    rets[name].append(total)
            out.append({"update": g["update"], "pos": g["pos"], "task_id": tid, "task_type": g["row"]["task_type"],
                        "success": [e["success"] for e in g["episodes"]], "kept": [e["kept"] for e in g["episodes"]],
                        "logged": [e["score"] for e in g["episodes"]], "returns": rets})
        except Exception as exc:
            out.append({"update": g["update"], "pos": g["pos"], "task_id": tid, "error": f"{type(exc).__name__}: {exc}"})
    return out


def kind(success):
    s = np.asarray(success, bool)
    return "mixed" if s.any() and not s.all() else ("all_success" if s.all() else "all_fail")


def spread(r):
    return float(np.std(r, ddof=1)) if len(r) > 1 else 0.0


def advantage(r):
    r = np.asarray(r, float)
    return (r - r.mean()) / (spread(r) + EPS)


def select(update_groups, reward, filt, target=TARGET):
    """{task_id: advantage scale} the filter keeps (1.0 for mixed tasks), candidates in update_groups' order."""
    order = sorted(update_groups, key=lambda g: g["pos"])
    nominal, extra = order[:target], order[target:]
    kept = [g for g in nominal if kind(g["success"]) == "mixed"]
    refill = [g for g in extra if kind(g["success"]) == "mixed"][: max(0, target - len(kept))]
    out = {g["task_id"]: 1.0 for g in kept + refill}
    if filt[0] == "keep":
        thr, floor = float(filt[1]), float(filt[2])
        fails = [(spread(g["returns"][reward]), k, g) for k, g in enumerate(order) if kind(g["success"]) == "all_fail"]
        eligible = sorted([x for x in fails if x[0] >= thr], key=lambda x: (-x[0], x[1]))
        for s, _, g in eligible[: max(0, target - len(out))]:
            out[g["task_id"]] = (s + EPS) / (max(s, floor) + EPS)
    return out


def summarize(groups, settings, base="S0"):
    by_update = defaultdict(list)
    for g in groups:
        by_update[g["update"]].append(g)
    sel = {x: {} for x in settings}
    for u, gs in by_update.items():
        for x, (reward, filt) in settings.items():
            sel[x][u] = select(gs, reward, filt)
    out = {}
    for x, (reward, filt) in settings.items():
        if x == base or base not in settings:
            continue
        b_reward = settings[base][0]
        tot = uniq = only = 0.0
        fam = defaultdict(lambda: [0.0, 0.0])
        cos2 = []
        n_tasks = n_only = 0
        for u, gs in by_update.items():
            for g in gs:
                if g["task_id"] not in sel[x][u]:
                    continue
                a = advantage(g["returns"][reward]) * sel[x][u][g["task_id"]]
                size = float(a @ a)
                n_tasks += 1
                if g["task_id"] in sel[base][u]:
                    b = advantage(g["returns"][b_reward]) * sel[base][u][g["task_id"]]
                    nb = float(b @ b)
                    c2 = min(1.0, float((a @ b) ** 2 / (size * nb))) if size > 0 and nb > 0 else 0.0
                    if kind(g["success"]) == "mixed" and size > 0 and nb > 0:
                        cos2.append(c2)
                    part = size * (1.0 - c2)
                else:
                    part = size
                    only += size
                    n_only += 1
                tot += size
                uniq += part
                fam[g["task_type"]][0] += part
                fam[g["task_type"]][1] += size
        updates = max(1, len(by_update))
        out[x] = {"unique_share": round(uniq / tot, 4) if tot else None,
                  "from_tasks_only_this_trains": round(only / tot, 4) if tot else None,
                  "tasks_per_update": round(n_tasks / updates, 2), "only_this_per_update": round(n_only / updates, 2),
                  "mixed_cos2_mean": round(float(np.mean(cos2)), 4) if cos2 else None,
                  "mixed_cos2_median": round(float(np.median(cos2)), 4) if cos2 else None,
                  "by_family": {k: round(v[0] / v[1], 4) if v[1] else None for k, v in sorted(fam.items())}}
    counts = defaultdict(float)
    for gs in by_update.values():
        for g in gs:
            counts[kind(g["success"])] += 1
    return {"against": base, "settings": {k: list(v[:1]) + [":".join(v[1])] for k, v in settings.items()},
            "updates": len(by_update), "per_update": {k: round(v / max(1, len(by_update)), 2) for k, v in counts.items()},
            "selection_counts": {x: round(sum(len(s) for s in sel[x].values()) / max(1, len(by_update)), 2)
                                 for x in settings},
            "unique": out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--reward", action="append", required=True, help="NAME=TRAIN_ENV.yaml")
    ap.add_argument("--setting", action="append", default=[], help="NAME=REWARD:mixed or NAME=REWARD:keep:T:F")
    ap.add_argument("--own", required=True, help="the reward the run trained with (checked against the logged returns)")
    ap.add_argument("--own-setting", default=None,
                    help="the setting the run trained with (its selection checked against kept_for_update; "
                         "default: the own reward with the mixed filter)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--max-update", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)  # the login node: at most 4 cores a task
    args = ap.parse_args()
    rewards = dict(s.split("=", 1) for s in args.reward)
    if args.own not in rewards:
        raise SystemExit(f"--own {args.own} is not among the rewards {sorted(rewards)}")
    raw = {**DEFAULT_SETTINGS, **dict(s.split("=", 1) for s in args.setting)}
    settings = {}
    for name, spec in raw.items():
        reward, *filt = spec.split(":")
        if reward in rewards:
            settings[name] = (reward, tuple(filt))
    rows = {}
    for l in Path(args.rows).read_text().splitlines():
        if l.strip():
            r = json.loads(l)
            rows[r["task_id"]] = r
    files = sorted(Path(args.run, "rollout_data").glob("*.jsonl"), key=lambda p: int(p.stem))
    if args.max_update:
        files = [f for f in files if int(f.stem) <= args.max_update]
    files = files[::args.every]
    groups = []
    for f in files:
        by, order = defaultdict(list), []
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if r.get("task_id") in rows:
                if r["task_id"] not in by:
                    order.append(r["task_id"])
                by[r["task_id"]].append(r)
        for pos, tid in enumerate(order):
            groups.append({"update": int(f.stem), "pos": pos, "row": rows[tid],
                           "episodes": [{"answers": answers_of(e["output"]), "score": float(e["score"]),
                                         "kept": bool(e.get("kept_for_update")),
                                         "success": bool(float(e.get("traj_success") or 0))} for e in by[tid]]})
    out = Path(args.out)
    work = out / "work"
    work.mkdir(parents=True, exist_ok=True)
    chunks = []
    for k in range(0, len(groups), 20):
        p = work / f"chunk_{k:06d}.jsonl"
        p.write_text("".join(json.dumps(g) + "\n" for g in groups[k:k + 20]))
        chunks.append((str(p), rewards))
    res = []
    with Pool(args.workers) as pool:
        for part in pool.imap_unordered(worker, chunks):
            res += part
    ok = [x for x in res if "error" not in x]
    with open(out / "groups.jsonl", "w") as f:
        for x in sorted(res, key=lambda x: (x["update"], x["pos"])):
            f.write(json.dumps(x) + "\n")
    checks = [abs(l - o) < 1e-6 for x in ok for l, o in zip(x["logged"], x["returns"][args.own])]
    own_setting = args.own_setting or next((k for k, v in settings.items()
                                            if v[0] == args.own and v[1] == ("mixed",)), None)
    match = None
    if own_setting:  # the simulated selection against the records' kept_for_update
        by_update = defaultdict(list)
        for g in ok:
            by_update[g["update"]].append(g)
        match = [sum(set(select(gs, *settings[own_setting])) == {g["task_id"] for g in gs if any(g["kept"])}
                     for gs in by_update.values()), len(by_update)]
    summary = {"run": args.run, "rewards": rewards, "own": args.own, "every": args.every, "groups": len(res),
               "errors": len(res) - len(ok), "error_examples": [x["error"] for x in res if "error" in x][:5],
               "own_replay_matches_logged": [sum(checks), len(checks)],
               "own_selection_matches_kept": match, **summarize(ok, settings)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
