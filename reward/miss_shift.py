"""Reward line, step 4 (the user's rule of 2026-10-29): in an optimal-progress run's own training records, how often
the env's table lookup missed (the pose's contract key absent from the task's table, so phi = 0 there) and how much
progress reward that took away or added, against a neighbour-cell fallback (a missing key takes the smallest d of the
cells next to it: x, y +-1 cm, yaw +-1 deg, the cells the 1 cm / 1 deg merge puts an exact pose next to).

Every recorded episode is replayed twice without rendering (replay_q3's way, the run's own train_env.yaml): as
trained ("own") and with the fallback ("fb"); the two differ only where a lookup missed. Per turn the two rewards,
so a miss's effect is read off the env's own reward (with the run's gamma and scale), not re-derived.

    python -m reward.miss_shift --run RUN_DIR --env-yaml TRAIN_ENV.yaml --rows ROWS --out OUT
        [--max-update N] [--workers 4]

Summary (OUT/summary.json):
* lookups and misses (all poses the env scored: the start and each primitive action), by kind (a neighbour with a
  distance, neighbours with no reachable success only, no neighbour within 1 cm / 1 deg), how many of them the
  fallback would give a positive phi (the others are phi 0 either way: d >= d* + 4), by 25-update block, and in the
  episodes the update trained on (the records' kept_for_update);
* the reward shift: per turn own - fb, summed as under-paid (own < fb, "多扣") and over-paid (own > fb, "多给"),
  against the progress reward's size (scale * |gamma phi_t - phi_t-1| summed over turns) and the episodes' returns;
* per episode the return shift, and in the kept groups the GRPO advantage ((R - mean) / (std + 1e-6), std with
  n - 1, as verl's grpo) under both, with sign flips;
* the own replay against the logged score (the env's low-information-frame penalty needs the frames, which the
  replay does not render, so a few episodes can differ).
"""

import argparse
import json
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from reward.replay_q3 import answers_of

OFF = 1 << 21
NEIGHBOURS = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1) if (dx, dy, dz) != (0, 0, 0)]


def unpack(k):
    return (k >> 31) - OFF, ((k >> 9) & ((1 << 22) - 1)) - OFF, k & 511


def pack(kx, ky, kyaw):
    return ((kx + OFF) << 31) | ((ky + OFF) << 9) | (kyaw % 360)


def neighbour_distance(table, key):
    """(kind, d) for a key absent from the table: the smallest reachable d among the 26 cells next to it."""
    kx, ky, kyaw = unpack(int(key))
    cand = np.array([pack(kx + dx, ky + dy, kyaw + dz) for dx, dy, dz in NEIGHBOURS], dtype=np.int64)
    keys = table["keys"]
    j = np.minimum(np.searchsorted(keys, cand), len(keys) - 1)
    hit = keys[j] == cand
    if not hit.any():
        return "none", None, None
    ds = table["dist"][j][hit]
    ds = ds[ds >= 0]
    if not len(ds):
        return "unreached", None, None
    return "neighbour", int(ds.min()), int(ds.max() - ds.min())


def make_cache(root, fallback):
    """The env's TableCache with each lookup recorded; with fallback, a missing key takes neighbour_distance."""
    from data_gen.interiorgs_v10 import distance_table as DT
    from data_gen.interiorgs_v10.difficulty import _split, contract_keys

    class Cache(DT.TableCache):
        def __init__(self):
            super().__init__(root)
            self.events = []

        def progress(self, item, c2w, success, slack=DT.DEFAULT_SLACK):
            table = self.get(str(item.get("task_id")))
            d_star = int((item.get("difficulty") or {}).get("d_star") or 0)
            dist = DT.lookup(table, c2w) if table is not None else None
            ev = {"miss": False, "success": bool(success)}
            if table is not None:
                k = int(contract_keys(*_split(c2w))[0])
                keys = table["keys"]
                i = int(np.searchsorted(keys, k))
                if i >= len(keys) or int(keys[i]) != k:
                    kind, d_nb, spread = neighbour_distance(table, k)
                    ev.update(miss=True, kind=kind, d_nb=d_nb, spread=spread,
                              phi_nb=DT.phi(d_nb, d_star, slack, success))  # what the fallback would give
                    if fallback and d_nb is not None:
                        dist = d_nb
            self.events.append(ev)
            return {"optimal_distance": dist, "optimal_table": table is not None,
                    "optimal_progress": DT.phi(dist, d_star, slack, success)}

    return Cache()


def worker(args):
    chunk_path, env_yaml = args
    import yaml

    from vagen.envs.active_spatial.env import ActiveSpatialEnv
    from vagen.envs.active_spatial.env_config import ActiveSpatialEnvConfig

    groups = [json.loads(l) for l in Path(chunk_path).read_text().splitlines() if l.strip()]
    rows_path = Path(chunk_path).with_suffix(".rows.jsonl")
    rows_path.write_text("".join(json.dumps(g["row"]) + "\n" for g in groups))
    cfg = dict(yaml.safe_load(Path(env_yaml).read_text())["envs"][0]["config"])
    if cfg.get("progress_version") != "optimal" or not cfg.get("distance_tables_dir"):
        raise ValueError("miss_shift needs an optimal-progress run (progress_version optimal, distance_tables_dir)")
    cfg.update(jsonl_path=str(rows_path), render_backend=None)
    fields = set(ActiveSpatialEnvConfig.__dataclass_fields__)
    envs = {}
    for name in ("own", "fb"):
        env = ActiveSpatialEnv(ActiveSpatialEnvConfig(**{k: v for k, v in cfg.items() if k in fields}))

        def fake_render(env=env, **_kwargs):
            env.last_image_std = 30.0
            env.last_low_info_frame = False
            env.consecutive_low_info_frame_count = 0
            return {"obs_str": "", "multi_modal_data": {}}

        env._render = fake_render
        env._distance_tables = make_cache(cfg["distance_tables_dir"], fallback=(name == "fb"))
        envs[name] = env
    scale = float(cfg.get("potential_field_reward_scale", 1.0))
    gamma = float(cfg.get("potential_field_gamma", 1.0))
    out = []
    for gi, g in enumerate(groups):
        tid = g["row"]["task_id"]
        try:
            eps = []
            for ep in g["episodes"]:
                rec = {}
                for name, env in envs.items():
                    env._distance_tables.events.clear()
                    env.reset(seed=gi)
                    assert env.current_item["task_id"] == tid
                    phis, rewards = [float(env.prev_potential_score)], []
                    for a in ep["answers"]:
                        _, r, done, _ = env.step(a)
                        rewards.append(float(r))
                        phis.append(float(env.prev_potential_score))
                        if done:
                            break
                    rec[name] = {"rewards": rewards, "phis": phis, "events": list(env._distance_tables.events)}
                own, fb = rec["own"], rec["fb"]
                ev = own["events"]
                diffs = [o - f for o, f in zip(own["rewards"], fb["rewards"])]
                prog = [scale * abs(gamma * b - a) for a, b in zip(own["phis"], own["phis"][1:])]
                kinds = defaultdict(int)
                spreads = []
                phi_pos = 0
                for e in ev:
                    if e["miss"]:
                        kinds[e["kind"]] += 1
                        phi_pos += int(e["phi_nb"] > 0 and not e["success"])
                        if e.get("spread") is not None:
                            spreads.append(e["spread"])
                eps.append({"logged": ep["score"], "kept": ep["kept"], "success": ep["success"],
                            "return_own": sum(own["rewards"]), "return_fb": sum(fb["rewards"]),
                            "turns": len(own["rewards"]), "same_turns": len(own["rewards"]) == len(fb["rewards"]),
                            "lookups": len(ev), "misses": sum(e["miss"] for e in ev),
                            "misses_not_success": sum(e["miss"] and not e["success"] for e in ev),
                            "misses_phi_nb_pos": phi_pos,
                            "kinds": dict(kinds), "spread_pos": sum(s > 0 for s in spreads),
                            "under_paid": -sum(d for d in diffs if d < 0), "over_paid": sum(d for d in diffs if d > 0),
                            "turns_changed": sum(abs(d) > 1e-9 for d in diffs), "progress_abs": sum(prog),
                            "reward_abs": sum(abs(r) for r in own["rewards"])})
            out.append({"update": g["update"], "task_id": tid, "task_type": g["row"]["task_type"], "episodes": eps})
        except Exception as exc:
            out.append({"update": g["update"], "task_id": tid, "error": f"{type(exc).__name__}: {exc}"})
    return out


def advantages(returns):
    r = np.asarray(returns, float)
    if len(r) < 2:
        return np.zeros_like(r)
    return (r - r.mean()) / (r.std(ddof=1) + 1e-6)


def block_of(update):
    lo = (update - 1) // 25 * 25 + 1
    return f"{lo}-{lo + 24}"


def summarize(res):
    ok = [x for x in res if "error" not in x]
    tot = defaultdict(float)
    by_block = defaultdict(lambda: defaultdict(float))
    by_family = defaultdict(lambda: defaultdict(float))
    kinds = defaultdict(int)
    ret_shift, flips, dadv, kept_groups_hit = [], 0, [], 0
    for g in ok:
        eps = g["episodes"]
        kept = [e for e in eps if e["kept"]]
        for e in eps:
            for k in ("lookups", "misses", "misses_not_success", "misses_phi_nb_pos", "under_paid", "over_paid",
                      "turns_changed", "progress_abs", "reward_abs", "turns", "spread_pos"):
                tot[k] += e[k]
                if e["kept"]:
                    tot["kept_" + k] += e[k]
            tot["episodes"] += 1
            tot["kept_episodes"] += int(e["kept"])
            tot["own_matches_logged"] += int(abs(e["return_own"] - e["logged"]) < 1e-6)
            tot["same_turns"] += int(e["same_turns"])
            for k, v in e["kinds"].items():
                kinds[k] += v
            d = e["return_own"] - e["return_fb"]
            if abs(d) > 1e-9:
                ret_shift.append(d)
                tot["episodes_return_changed"] += 1
                tot["kept_episodes_return_changed"] += int(e["kept"])
            for b in (by_block[block_of(g["update"])], by_family[g["task_type"]]):
                b["lookups"] += e["lookups"]
                b["misses"] += e["misses"]
                b["under_paid"] += e["under_paid"]
                b["over_paid"] += e["over_paid"]
                b["progress_abs"] += e["progress_abs"]
        tot["groups"] += 1
        tot["groups_with_miss"] += int(any(e["misses"] for e in eps))
        if len(kept) > 1:
            a_own = advantages([e["return_own"] for e in kept])
            a_fb = advantages([e["return_fb"] for e in kept])
            if any(abs(e["return_own"] - e["return_fb"]) > 1e-9 for e in kept):
                kept_groups_hit += 1
                dadv += [float(x) for x in np.abs(a_own - a_fb)]
                flips += int(np.sum(np.sign(np.round(a_own, 9)) != np.sign(np.round(a_fb, 9))))
            tot["kept_groups"] += 1

    def pct(a, b):
        return round(100.0 * a / b, 4) if b else None

    def rate(b):
        return {"lookups": int(b["lookups"]), "misses": int(b["misses"]), "miss_pct": pct(b["misses"], b["lookups"]),
                "under_paid": round(b["under_paid"], 4), "over_paid": round(b["over_paid"], 4),
                "shift_pct_of_progress": pct(b["under_paid"] + b["over_paid"], b["progress_abs"])}

    rs = np.asarray(ret_shift, float)
    return {
        "groups": int(tot["groups"]), "episodes": int(tot["episodes"]), "errors": len(res) - len(ok),
        "error_examples": [x["error"] for x in res if "error" in x][:5],
        "own_replay_matches_logged": [int(tot["own_matches_logged"]), int(tot["episodes"])],
        "same_turn_count_own_fb": [int(tot["same_turns"]), int(tot["episodes"])],
        "lookups": int(tot["lookups"]), "misses": int(tot["misses"]), "miss_pct": pct(tot["misses"], tot["lookups"]),
        "misses_not_on_success": int(tot["misses_not_success"]), "miss_kinds": dict(kinds),
        "misses_fallback_phi_pos": int(tot["misses_phi_nb_pos"]),  # the rest: phi 0 either way (d >= d* + 4)
        "misses_neighbour_spread_gt0": int(tot["spread_pos"]),
        "groups_with_miss": int(tot["groups_with_miss"]),
        "reward": {"under_paid": round(tot["under_paid"], 4), "over_paid": round(tot["over_paid"], 4),
                   "net_own_minus_fb": round(tot["over_paid"] - tot["under_paid"], 4),
                   "progress_abs": round(tot["progress_abs"], 2), "reward_abs": round(tot["reward_abs"], 2),
                   "shift_pct_of_progress": pct(tot["under_paid"] + tot["over_paid"], tot["progress_abs"]),
                   "shift_pct_of_reward": pct(tot["under_paid"] + tot["over_paid"], tot["reward_abs"]),
                   "turns": int(tot["turns"]), "turns_changed": int(tot["turns_changed"])},
        "returns": {"episodes_changed": int(tot["episodes_return_changed"]),
                    "mean_shift": round(float(rs.mean()), 4) if len(rs) else 0.0,
                    "max_abs_shift": round(float(np.abs(rs).max()), 4) if len(rs) else 0.0,
                    "lower_than_fb": int((rs < 0).sum()), "higher_than_fb": int((rs > 0).sum())},
        "kept": {"episodes": int(tot["kept_episodes"]), "lookups": int(tot["kept_lookups"]),
                 "misses": int(tot["kept_misses"]), "miss_pct": pct(tot["kept_misses"], tot["kept_lookups"]),
                 "under_paid": round(tot["kept_under_paid"], 4), "over_paid": round(tot["kept_over_paid"], 4),
                 "shift_pct_of_progress": pct(tot["kept_under_paid"] + tot["kept_over_paid"], tot["kept_progress_abs"]),
                 "episodes_return_changed": int(tot["kept_episodes_return_changed"]),
                 "groups": int(tot["kept_groups"]), "groups_changed": kept_groups_hit,
                 "advantage_sign_flips": flips,
                 "mean_abs_advantage_change_in_changed_groups": round(float(np.mean(dadv)), 4) if dadv else 0.0,
                 "max_abs_advantage_change": round(float(np.max(dadv)), 4) if dadv else 0.0},
        "by_block": {k: rate(v) for k, v in sorted(by_block.items(), key=lambda kv: int(kv[0].split("-")[0]))},
        "by_family": {k: rate(v) for k, v in sorted(by_family.items())},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--env-yaml", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-update", type=int, default=None)
    ap.add_argument("--workers", type=int, default=4)  # the login node: at most 4 cores a task
    args = ap.parse_args()

    rows = {}
    for l in Path(args.rows).read_text().splitlines():
        if l.strip():
            r = json.loads(l)
            rows[r["task_id"]] = r
    files = sorted(Path(args.run, "rollout_data").glob("*.jsonl"), key=lambda p: int(p.stem))
    if args.max_update:
        files = [f for f in files if int(f.stem) <= args.max_update]
    groups, unknown = [], 0
    for f in files:
        by = defaultdict(list)
        for line in f.read_text().splitlines():
            r = json.loads(line)
            if r.get("task_id") in rows:
                by[r["task_id"]].append(r)
            else:
                unknown += 1
        for tid, eps in sorted(by.items()):
            groups.append({"update": int(f.stem), "row": rows[tid],
                           "episodes": [{"answers": answers_of(e["output"]), "score": float(e["score"]),
                                         "kept": bool(e.get("kept_for_update")),
                                         "success": bool(float(e.get("traj_success") or 0))} for e in eps]})
    out = Path(args.out)
    work = out / "work"
    work.mkdir(parents=True, exist_ok=True)
    chunks = []
    for k in range(0, len(groups), 20):
        p = work / f"chunk_{k:06d}.jsonl"
        p.write_text("".join(json.dumps(g) + "\n" for g in groups[k:k + 20]))
        chunks.append((str(p), args.env_yaml))
    res = []
    with Pool(args.workers) as pool:
        for part in pool.imap_unordered(worker, chunks):
            res += part
    with open(out / "episodes.jsonl", "w") as f:
        for x in sorted(res, key=lambda x: (x["update"], x["task_id"])):
            f.write(json.dumps(x) + "\n")
    summary = {"run": args.run, "env_yaml": args.env_yaml, "updates": len(files),
               "records_outside_rows": unknown, **summarize(res)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("by_block", "by_family")}, indent=1))


if __name__ == "__main__":
    main()
