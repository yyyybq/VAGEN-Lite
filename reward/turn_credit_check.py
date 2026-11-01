"""Reward line (phase 2, direction 1, the user's brief of 2026-11-01; CPU only): before the smoke, check on recorded
episodes that what the agent loop will record equals the table, and what the estimator does to a batch.

1. Every recorded episode of a run (every EVERY-th update) is replayed without rendering in the training env (the
   run's env config with progress_version optimal, the depth-12 tables and the neighbour fallback, shaping off) the
   way the agent loop steps it: phi_0 and d_0 from the reset info, per turn vagen.turn_credit.turn_record(reward, info,
   extract_success(info), not turn_invalid(info)) with the agent loop's own two functions.  Each turn is compared
   with the env's canonical metric at the turn end (turn_credit_study's way): phi_after against optimal_progress,
   d_after against optimal_distance, success against the metric's success; the replayed success against the
   recorded traj_success; the 11-01 credit (reward + shaping) against turn_credit_study's turns.jsonl.gz (built
   without the fallback, so episodes where a neighbour answered are counted apart).
2. The groups of each update, in batches of BATCH groups (the nominal filter with target_groups = candidates; uid = the
   group), go through vagen.turn_credit.token_advantages, the answers' tokens counted with the run's tokenizer and
   laid end to end: "P2" the brief's first round (the defaults), "P2-stall" the same with the stall penalty 0.05 (or
   --arms, e.g. a sweep of turn_weight).  Per
   batch the estimator's metrics (the two terms, the turn term's magnitude share, the sign flips, groups with equal
   returns, table misses, stalls) and the answer tokens' positive share, overall and at turns 4+.

    python -m reward.turn_credit_check --run RUN_DIR --env-yaml TRAIN_ENV.yaml --tables DIR --rows ROWS
        --study STUDY_DIR --tokenizer PATH --out OUT [--every 4] [--batch 24] [--workers 4] [--arms JSON]

OUT/summary.json, OUT/batches.jsonl (one line per batch and arm).
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
TOL = 1e-6
ARMS = {"P2": {}, "P2-stall": {"stall_penalty": 0.05}}


def worker(args):
    chunk_path, env_yaml, tables, tokenizer = args
    import yaml
    from transformers import AutoTokenizer

    from vagen.agent_loop.gym_agent_loop import extract_success, turn_invalid
    from vagen.envs.active_spatial.env import ActiveSpatialEnv
    from vagen.envs.active_spatial.env_config import ActiveSpatialEnvConfig
    from vagen.turn_credit import episode_record, reset_distance, turn_credits, turn_record

    tok = AutoTokenizer.from_pretrained(tokenizer)
    groups = [json.loads(l) for l in Path(chunk_path).read_text().splitlines() if l.strip()]
    rows_path = Path(chunk_path).with_suffix(".rows.jsonl")
    rows_path.write_text("".join(json.dumps(g["row"]) + "\n" for g in groups))
    cfg = dict(yaml.safe_load(Path(env_yaml).read_text())["envs"][0]["config"])
    if cfg.get("enable_potential_shaping_reward"):
        raise ValueError("the env config must have the shaping reward off")
    cfg.update(jsonl_path=str(rows_path), render_backend=None, progress_version="optimal", distance_tables_dir=tables,
               optimal_progress_near=True)
    fields = set(ActiveSpatialEnvConfig.__dataclass_fields__)
    env = ActiveSpatialEnv(ActiveSpatialEnvConfig(**{k: v for k, v in cfg.items() if k in fields}))

    def fake_render(**_kwargs):
        env.last_image_std = 30.0
        env.last_low_info_frame = False
        env.consecutive_low_info_frame_count = 0
        return {"obs_str": "", "multi_modal_data": {}}

    env._render = fake_render
    out = []
    for gi, g in enumerate(groups):
        tid = g["row"]["task_id"]
        eps = []
        n = Counter()
        try:
            for ep in g["episodes"]:
                _, info0 = env.reset(seed=gi)
                assert env.current_item["task_id"] == tid
                m0 = env._calculate_canonical_metric() or {}
                phi0 = float(info0.get("initial_potential_score", 0.0))  # the agent loop's _ep_init_score
                d0 = reset_distance(info0)
                n["phi0"] += 1
                n["phi0_off"] += int(abs(phi0 - float(m0.get("optimal_progress") or 0.0)) > TOL)
                n["d0_off"] += int(d0 != m0.get("optimal_distance"))
                near = bool(m0.get("optimal_near"))
                n["lookups"] += 1
                n["near"] += int(near)
                n["miss"] += int(m0.get("optimal_distance") is None)
                turns, ntok = [], []
                for a in ep["answers"]:
                    _, r, done, info = env.step(a)
                    rec = turn_record(None, r, info, extract_success(info), not turn_invalid(info))
                    m = env._calculate_canonical_metric() or {}
                    ms = bool(m.get("success"))
                    n["turns"] += 1
                    n["lookups"] += 1
                    n["near"] += int(bool(m.get("optimal_near")))
                    n["miss"] += int(m.get("optimal_distance") is None and not ms)
                    near = near or bool(m.get("optimal_near"))
                    n["success_off"] += int(rec["success"] != ms)
                    if ms:
                        n["success_turns"] += 1
                    else:
                        n["phi_after_off"] += int(abs(rec["phi_after"] - float(m.get("optimal_progress") or 0.0)) > TOL)
                        n["d_after_off"] += int(rec["d_after"] != m.get("optimal_distance"))
                    turns.append(rec)
                    ntok.append(len(tok.encode(a, add_special_tokens=False)))
                    if done:
                        break
                record = episode_record(phi0, turns, 10 ** 9, d0)
                replay_success = bool(turns and turns[-1]["success"])
                n["episodes"] += 1
                n["recorded_success_off"] += int(replay_success != ep["success"])
                eps.append({"record": record, "credits_1101": [round(c, 6) for c in turn_credits(record, SCALE, GAMMA)],
                            "near": near, "tokens": ntok, "return": round(sum(t["reward"] for t in turns), 6),
                            "success": replay_success})
            out.append({"update": g["update"], "task_id": tid, "episodes": eps, "counts": dict(n)})
        except Exception as exc:
            out.append({"update": g["update"], "task_id": tid, "error": f"{type(exc).__name__}: {exc}"})
    return out


def batch_stats(groups, overrides):
    """One batch of groups through token_advantages: the answers' tokens laid end to end (no observation tokens)."""
    import torch

    from vagen.turn_credit import LATE_TURN, settings, token_advantages

    cfg = settings({"turn_credit": overrides})
    records, uids, returns, spans_all = [], [], [], []
    for gi, g in enumerate(groups):
        for e in g["episodes"]:
            pos, spans = 0, []
            for n in e["tokens"]:
                spans.append([pos, pos + n])
                pos += n
            records.append(dict(e["record"], turns=[dict(t, span=s) for t, s in zip(e["record"]["turns"], spans)]))
            uids.append(gi)
            returns.append(e["return"])
            spans_all.append(spans)
    length = max([s[-1][1] for s in spans_all if s] + [1])
    mask = torch.zeros((len(records), length), dtype=torch.int64)
    for i, sp in enumerate(spans_all):
        if sp:
            mask[i, :sp[-1][1]] = 1
    adv, met = token_advantages(records, uids, mask, returns, cfg)
    tok_pos = tok_all = late_pos = late_all = 0
    for i, sp in enumerate(spans_all):
        for t, (s, e) in enumerate(sp):
            if e <= s:
                continue
            pos = int((adv[i, s:e] > 1e-9).sum())
            tok_all += e - s
            tok_pos += pos
            if t >= LATE_TURN:
                late_all += e - s
                late_pos += pos
    met.update({"tokens/pos_share": tok_pos / max(1, tok_all), "tokens/late_pos_share": late_pos / max(1, late_all)})
    return met


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--env-yaml", required=True)
    ap.add_argument("--tables", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--study", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--every", type=int, default=4)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--workers", type=int, default=4)  # the login node: at most 4 cores a task
    ap.add_argument("--arms", default=None, help="JSON {name: turn_credit settings} instead of P2 and P2-stall")
    args = ap.parse_args()
    arms = json.loads(args.arms) if args.arms else ARMS
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
                           "episodes": [{"answers": answers_of(e["output"]),
                                         "success": bool(float(e.get("traj_success") or 0))} for e in eps]})
    out = Path(args.out)
    work = out / "work"
    work.mkdir(parents=True, exist_ok=True)
    chunks = []
    for k in range(0, len(groups), 20):
        p = work / f"chunk_{k:06d}.jsonl"
        p.write_text("".join(json.dumps(g) + "\n" for g in groups[k:k + 20]))
        chunks.append((str(p), args.env_yaml, args.tables, args.tokenizer))
    res = []
    with Pool(args.workers) as pool:
        for part in pool.imap_unordered(worker, chunks):
            res += part
    res.sort(key=lambda g: (g["update"], g["task_id"]))
    ok = [g for g in res if "error" not in g]
    counts = Counter()
    for g in ok:
        counts.update(g["counts"])
    # the study's 11-01 credits (rounded to 4 places there, no fallback), episodes in the same order
    study = {}
    with gzip.open(Path(args.study, "turns.jsonl.gz"), "rt") as f:
        for line in f:
            g = json.loads(line)
            if "error" not in g:
                study[(g["update"], g["task_id"])] = [[t["credit"] for t in e["turns"]] for e in g["episodes"]]
    vs_study = Counter()
    for g in ok:
        theirs = study.get((g["update"], g["task_id"]))
        if theirs is None:
            vs_study["groups_missing"] += 1
            continue
        for e, c2 in zip(g["episodes"], theirs):
            kind = "near" if e["near"] else "plain"
            vs_study[f"{kind}_episodes"] += 1
            vs_study[f"{kind}_turns"] += len(e["credits_1101"])
            vs_study[f"{kind}_length_off"] += int(len(e["credits_1101"]) != len(c2))
            vs_study[f"{kind}_credit_off"] += sum(abs(a - b) > 6e-5 for a, b in zip(e["credits_1101"], c2))
    by_update = defaultdict(list)
    for g in ok:
        by_update[g["update"]].append(g)
    lines = []
    for upd, gs in sorted(by_update.items()):
        for k in range(0, len(gs), args.batch):
            part = gs[k:k + args.batch]
            if len(part) < args.batch // 2:
                continue
            for arm, over in arms.items():
                lines.append({"update": upd, "batch": k // args.batch, "arm": arm, **batch_stats(part, over)})
    with open(out / "batches.jsonl", "w") as f:
        for l in lines:
            f.write(json.dumps(l) + "\n")

    def dist(sel, key):
        v = np.asarray([l[key] for l in sel if key in l], float)
        if not v.size:
            return None
        return {"mean": round(float(v.mean()), 4), "p10": round(float(np.quantile(v, 0.1)), 4),
                "median": round(float(np.median(v)), 4), "p90": round(float(np.quantile(v, 0.9)), 4)}

    keys = sorted({k for l in lines for k in l if k.startswith(("turn_credit/", "tokens/"))})
    estimator = {}
    for arm in arms:
        sel = [l for l in lines if l["arm"] == arm]
        estimator[arm] = {"batches": len(sel), **{k: dist(sel, k) for k in keys}}
    summary = {"run": args.run, "every": args.every, "batch_groups": args.batch, "scale": SCALE, "gamma": GAMMA,
               "arms": arms, "groups": len(ok), "errors": len(res) - len(ok),
               "error_examples": [g["error"] for g in res if "error" in g][:5],
               "agent_loop_vs_table": dict(counts), "vs_study_1101_credit": dict(vs_study), "estimator": estimator}
    (out / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
