"""Reward line (the user's item 1 of 2026-11-01, the plan's numbers; CPU only): what a per-turn credit would look like
on recorded episodes, before any training code.

Credit of turn t (the user's formula): c_t = r_t + SCALE * (GAMMA * phi_{t+1} - phi_t), r_t the turn's S0 reward (the
success +5, the unreadable-answer and collision penalties; replayed in the S0 env), phi the optimal progress from the
depth-12 table (1 on success), phi_t before the turn and phi_{t+1} after it.

Every recorded episode of a run (every EVERY-th update) is replayed without rendering in the S0 env with
progress_version optimal (shaping off, so the reward is S0's and the score is the table's phi), turn by turn:

* the turn's table distance before and after (the last pose the env scored in the turn), so the turn is "closer",
  "same" or "farther" by the table, "success", "unreadable" (no action executed) or "unknown" (a pose off the table or
  with no success reachable in it); and the credit;
* per group (task, update): the outcome kind (mixed / all success / all fail, what GRPO keeps or drops) and the
  episode-level GRPO advantage of each episode under S0 (std n-1), against which each turn's credit sign is compared.

    python -m reward.turn_credit_study --run RUN_DIR --s0-yaml TRAIN_ENV.yaml --tables DIR --rows ROWS --out OUT
        [--every 4] [--workers 4]

OUT/summary.json and OUT/turns.jsonl.gz (one line per turn).
"""

import argparse
import gzip
import json
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from reward.replay_q3 import answers_of

SCALE, GAMMA = 10.0 / 3.0, 0.99


def worker(args):
    chunk_path, s0_yaml, tables = args
    import yaml

    from data_gen.interiorgs_v10 import distance_table as DT
    from vagen.envs.active_spatial.env import ActiveSpatialEnv
    from vagen.envs.active_spatial.env_config import ActiveSpatialEnvConfig

    groups = [json.loads(l) for l in Path(chunk_path).read_text().splitlines() if l.strip()]
    rows_path = Path(chunk_path).with_suffix(".rows.jsonl")
    rows_path.write_text("".join(json.dumps(g["row"]) + "\n" for g in groups))
    cfg = dict(yaml.safe_load(Path(s0_yaml).read_text())["envs"][0]["config"])
    if cfg.get("enable_potential_shaping_reward"):
        raise ValueError("the S0 env config must have the shaping reward off")
    cfg.update(jsonl_path=str(rows_path), render_backend=None, progress_version="optimal", distance_tables_dir=tables)
    fields = set(ActiveSpatialEnvConfig.__dataclass_fields__)
    env = ActiveSpatialEnv(ActiveSpatialEnvConfig(**{k: v for k, v in cfg.items() if k in fields}))

    def fake_render(**_kwargs):
        env.last_image_std = 30.0
        env.last_low_info_frame = False
        env.consecutive_low_info_frame_count = 0
        return {"obs_str": "", "multi_modal_data": {}}

    env._render = fake_render

    class Cache(DT.TableCache):
        last = None

        def progress(self, item, c2w, success, slack=DT.DEFAULT_SLACK):
            out = super().progress(item, c2w, success, slack)
            Cache.last = out["optimal_distance"]
            return out

    env._distance_tables = Cache(tables)
    out = []
    for gi, g in enumerate(groups):
        tid = g["row"]["task_id"]
        d_star = int(g["row"]["difficulty"]["d_star"])
        eps = []
        try:
            for ep in g["episodes"]:
                env.reset(seed=gi)
                assert env.current_item["task_id"] == tid
                m0 = env._calculate_canonical_metric()  # the start pose's table lookup
                phi, d = float((m0 or {}).get("optimal_progress") or 0.0), Cache.last
                turns, total = [], 0.0
                for a in ep["answers"]:
                    _, r, done, info = env.step(a)
                    tm = (info.get("metrics") or {}).get("turn_metrics") or {}
                    m = env._calculate_canonical_metric()  # the turn-end pose (the evaluation's way of reading success)
                    success = bool(m and m.get("success"))
                    phi2 = 1.0 if success else float((m or {}).get("optimal_progress") or 0.0)
                    d2 = 0 if success else Cache.last
                    valid = bool(tm.get("action_is_valid", True))
                    if success:
                        kind = "success"
                    elif not valid:
                        kind = "unreadable"
                    elif d is None or d2 is None:
                        kind = "unknown"
                    else:
                        kind = "closer" if d2 < d else ("same" if d2 == d else "farther")
                    credit = float(r) + SCALE * (GAMMA * phi2 - phi)
                    turns.append({"kind": kind, "r": round(float(r), 4), "credit": round(credit, 4),
                                  "phi": round(phi, 4), "phi_next": round(phi2, 4), "d": d, "d_next": d2,
                                  "collisions": int(tm.get("collision_count", 0) or 0)})
                    total += float(r)
                    phi, d = phi2, d2
                    if done:
                        break
                eps.append({"success": ep["success"], "kept": ep["kept"], "return_s0": round(total, 4), "turns": turns})
            out.append({"update": g["update"], "task_id": tid, "task_type": g["row"]["task_type"], "d_star": d_star,
                        "episodes": eps})
        except Exception as exc:
            out.append({"update": g["update"], "task_id": tid, "error": f"{type(exc).__name__}: {exc}"})
    return out


def summarize(groups):
    ok = [g for g in groups if "error" not in g]
    by_kind = defaultdict(lambda: Counter())
    group_kind = Counter()
    signal = defaultdict(lambda: [0, 0])  # group kind -> [turns with credit != 0, turns]
    agree = Counter()
    pos_mass = neg_mass = 0.0
    pos_mass_c = neg_mass_c = 0.0
    zero_closer = closer = 0
    for g in ok:
        succ = [e["success"] for e in g["episodes"]]
        gk = "mixed" if any(succ) and not all(succ) else ("all_success" if all(succ) else "all_fail")
        group_kind[gk] += 1
        rets = np.array([e["return_s0"] for e in g["episodes"]], float)
        adv = (rets - rets.mean()) / (rets.std(ddof=1) + 1e-6) if len(rets) > 1 else np.zeros_like(rets)
        # a per-(group, turn index) baseline: the mean credit of the group's turns at that position
        by_pos = defaultdict(list)
        for e in g["episodes"]:
            for i, t in enumerate(e["turns"]):
                by_pos[i].append(t["credit"])
        for e, a in zip(g["episodes"], adv):
            for i, t in enumerate(e["turns"]):
                c = t["credit"]
                sign = "pos" if c > 1e-9 else ("neg" if c < -1e-9 else "zero")
                by_kind[t["kind"]][sign] += 1
                signal[gk][0] += int(sign != "zero")
                signal[gk][1] += 1
                if c > 0:
                    pos_mass += c
                else:
                    neg_mass -= c
                cc = c - float(np.mean(by_pos[i]))
                if cc > 0:
                    pos_mass_c += cc
                else:
                    neg_mass_c -= cc
                if t["kind"] == "closer":
                    closer += 1
                    zero_closer += int(abs(c) <= 1e-9)
                if gk == "mixed" and abs(a) > 1e-9 and sign != "zero":
                    agree["same" if (a > 0) == (c > 0) else "opposite"] += 1
    kinds = {k: dict(v) for k, v in sorted(by_kind.items())}
    turns = sum(sum(v.values()) for v in by_kind.values())
    return {"groups": len(ok), "errors": len(groups) - len(ok), "group_kinds": dict(group_kind), "turns": turns,
            "turn_kinds": {k: {"turns": sum(v.values()), "share": round(sum(v.values()) / turns, 4), **v} for k, v in kinds.items()},
            "closer_with_zero_credit": [zero_closer, closer],
            "nonzero_credit_share_by_group_kind": {k: round(v[0] / v[1], 4) for k, v in signal.items() if v[1]},
            "credit_mass": {"positive": round(pos_mass, 1), "negative": round(neg_mass, 1),
                            "positive_share": round(pos_mass / (pos_mass + neg_mass), 4) if pos_mass + neg_mass else None},
            "credit_mass_after_turn_baseline": {"positive": round(pos_mass_c, 1), "negative": round(neg_mass_c, 1)},
            "mixed_groups_sign_vs_episode_advantage": dict(agree)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--s0-yaml", required=True)
    ap.add_argument("--tables", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)  # the login node: at most 4 cores a task
    args = ap.parse_args()
    rows = {}
    for l in Path(args.rows).read_text().splitlines():
        if l.strip():
            r = json.loads(l)
            rows[r["task_id"]] = r
    files = sorted(Path(args.run, "rollout_data").glob("*.jsonl"), key=lambda p: int(p.stem))[::args.every]
    groups = []
    for f in files:
        by = defaultdict(list)
        for line in f.read_text(errors="ignore").splitlines():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if r.get("task_id") in rows:
                by[r["task_id"]].append(r)
        for tid, eps in sorted(by.items()):
            groups.append({"update": int(f.stem), "row": rows[tid],
                           "episodes": [{"answers": answers_of(e["output"]), "kept": bool(e.get("kept_for_update")),
                                         "success": bool(float(e.get("traj_success") or 0))} for e in eps]})
    out = Path(args.out)
    work = out / "work"
    work.mkdir(parents=True, exist_ok=True)
    chunks = []
    for k in range(0, len(groups), 20):
        p = work / f"chunk_{k:06d}.jsonl"
        p.write_text("".join(json.dumps(g) + "\n" for g in groups[k:k + 20]))
        chunks.append((str(p), args.s0_yaml, args.tables))
    res = []
    with Pool(args.workers) as pool:
        for part in pool.imap_unordered(worker, chunks):
            res += part
    with gzip.open(out / "turns.jsonl.gz", "wt") as f:
        for g in sorted(res, key=lambda g: (g["update"], g["task_id"])):
            f.write(json.dumps(g) + "\n")
    summary = {"run": args.run, "every": args.every, "scale": SCALE, "gamma": GAMMA, **summarize(res)}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
